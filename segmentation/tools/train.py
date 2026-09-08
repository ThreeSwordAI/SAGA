#!/usr/bin/env python3
"""
segmentation/tools/train.py
=============================
TASK-09 dense-prediction trainer — ADE20K semantic segmentation, ViT-B
backbone + Simple Feature Pyramid + multi-scale segmentation head.

WHY IT WAS REWRITTEN:
  - B4 mIoU ignore-index leak. The committed metric computed, per class,
        inter = (pred == c) & (gt == c)
        union = (pred == c) | (gt == c)
    over EVERY pixel, so the 255 (unlabelled) pixels — ~8% of ADE20K — kept
    contributing to the UNION whenever the model predicted class c there.
    Unions were inflated, so every mIoU was biased DOWNWARD by an amount
    that depends on how the model happens to label unlabelled regions, i.e.
    differently per variant. Any pre-fix mIoU is unusable.
    The fix: accumulate a confusion matrix over `gt != 255` pixels only.
    Intersections and unions both come out of that matrix, so ignored pixels
    are structurally absent from both. The matrix is also written out
    (conf_matrix.npz), which the per-class table and Phase C read.
  - B6 registers backbone — fixed in detection/models/backbone.py, which
    this pipeline now imports instead of keeping its own stale copy.
  - validation used a padded DistributedSampler (duplicate images on the
    last shard, bug B1's shape); it now shards exactly rank::world_size and
    asserts full coverage.

TRAINING MATH IS THE COMMITTED ONE: read from segmentation/configs/base.yaml
through configs/dense_matrix.yaml — AdamW two groups (backbone 1e-5 / head
1e-4), weight decay 0.05, grad clip 1.0 after unscale_, fp16 AMP,
CosineAnnealingLR(T_max=epochs, eta_min=1e-7) per epoch, backbone frozen for
2 epochs, 80 epochs, global batch 16, 512 crops, flip + scale [0.5, 2.0],
CrossEntropy(ignore_index=255), eval every 10 epochs. `warmup_epochs` is
declared in that config but was never implemented; it stays unimplemented
(implementing it now would change the schedule).

OUTPUTS  results/segmentation/<run_id>/
    meta.json / config.resolved.yaml / log.csv
    miou_ss.json          single-scale mIoU at the best epoch (+ pixel/mean
                          accuracy, n_images, provenance)
    per_class_iou.csv     150 rows: iou_frac (0-1), inter/union/gt/pred
                          pixel counts. NOTE the units: the JSONs report
                          mIoU/pixel_acc/mean_acc as PERCENTAGES and
                          iou_per_class as FRACTIONS; both files record a
                          "units" block / a unit-bearing column name.
    conf_matrix.npz       the 150x150 confusion matrix behind those numbers
    preds_fixed20/        for the 20 committed probe images
                          (results/probe/ade20k_fixed20.json):
                          <stem>_pred.png (mode-L class indices),
                          <stem>_gt.png, <stem>_img.jpg
    miou_ms.json          multi-scale + flip TTA mIoU, computed ONCE at the
                          end from the val-selected weights (skipped if
                          eval.ms_scales is empty)
    ckpt/{last,best_model}.pth   git-ignored

Launch (identical fresh and on every resubmission):
    torchrun --nproc_per_node=4 segmentation/tools/train.py \
        --matrix configs/dense_matrix.yaml --run seg_vitb_saga_s1 \
        --data_root $DATA_ROOT --resume auto
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from saga.run_registry import create_run, file_sha256, finalize_run, git_sha
from segmentation.data.ade20k_dataset import ADE20KDataset
from segmentation.data.transforms import (get_train_transforms,
                                          get_val_transforms)
from segmentation.models.segmentor import build_segmentor
from tools.dense_runtime import (append_log_row, atomic_bytes_write,
                                 atomic_json_dump, atomic_npz_save,
                                 atomic_torch_save, atomic_write_text,
                                 check_schedule_geometry,
                                 clear_stale_artifacts, gather_rng_states,
                                 log_epochs, read_meta, repo_path,
                                 resolve_backbone, resolve_dense_config,
                                 restore_rng_state, run_is_complete,
                                 sanitize_log, save_dense_checkpoint,
                                 set_seed, shard_indices,
                                 sync_buffers_from_rank0)

LOG_FIELDS = ["epoch", "lr_backbone", "lr_head", "train_loss", "mIoU",
              "img_per_sec", "wall_time"]
IGNORE_INDEX = 255
SS_JSON = "miou_ss.json"
MS_JSON = "miou_ms.json"
PER_CLASS_CSV = "per_class_iou.csv"
CONF_NPZ = "conf_matrix.npz"
PROBE_DIR = "preds_fixed20"
FIXED20 = "results/probe/ade20k_fixed20.json"
PER_CLASS_FIELDS = ["class_index", "name", "iou_frac",
                    "intersection", "union", "gt_pixels",
                    "pred_pixels"]


# ── the B4 fix: confusion matrix over non-ignored pixels only ────────────────

def update_confusion(conf, preds, target, num_classes,
                     ignore_index=IGNORE_INDEX):
    """Accumulate `conf[gt, pred]` over pixels whose label is not ignored.

    THIS is the B4 fix: the `valid` mask removes ignored pixels before
    anything is counted, so they can reach neither the intersection nor the
    union of any class. Both quantities are later derived from this matrix.
    """
    valid = target != ignore_index
    if valid.any():
        gt = target[valid].to(torch.int64)
        pr = preds[valid].to(torch.int64)
        if int(gt.max()) >= num_classes or int(gt.min()) < 0:
            raise ValueError(
                f"ground-truth labels outside [0,{num_classes}) after the "
                f"ignore mask: min={int(gt.min())}, max={int(gt.max())}")
        idx = gt * num_classes + pr
        conf += torch.bincount(
            idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf


def iou_from_confusion(conf):
    """(iou[C], inter[C], union[C], gt[C], pred[C]) as float64/int64 numpy."""
    conf = conf.detach().cpu().numpy().astype(np.int64)
    inter = np.diag(conf)
    gt = conf.sum(axis=1)
    pred = conf.sum(axis=0)
    union = gt + pred - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return iou, inter, gt, pred, union


def summarize_confusion(conf):
    """mIoU (%) over classes with union>0, plus pixel and mean accuracy."""
    iou, inter, gt, pred, union = iou_from_confusion(conf)
    scored = union > 0
    n_scored = int(scored.sum())
    miou = float(np.mean(iou[scored]) * 100.0) if n_scored else float("nan")
    total = int(gt.sum())
    pixel_acc = float(inter.sum() / total * 100.0) if total else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(gt > 0, inter / np.maximum(gt, 1), np.nan)
    present = gt > 0
    mean_acc = (float(np.mean(per_class_acc[present]) * 100.0)
                if present.any() else float("nan"))
    return {
        "mIoU": round(miou, 4),
        "pixel_acc": round(pixel_acc, 4),
        "mean_acc": round(mean_acc, 4),
        "n_classes_scored": n_scored,
        "n_labelled_pixels": total,
    }, iou, inter, gt, pred, union


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_confusion(model, loader, device, num_classes, world_size,
                       ms_scales=None, flip=False):
    """Confusion matrix over this rank's shard, all-reduced across ranks.
    Returns (conf, n_images_total). `ms_scales` (+ optional flip) turns this
    into multi-scale TTA: logits are averaged as softmax probabilities at the
    label resolution."""
    # the metric must not depend on which val images landed on which rank:
    # every rank evaluates with rank 0's BatchNorm running stats, i.e. with
    # exactly the weights that get saved as best_model.pth
    sync_buffers_from_rank0(model, world_size > 1)
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64,
                       device=device)
    n_local = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        n_local += int(images.shape[0])

        if not ms_scales:
            # argmax of the logits == argmax of their softmax; skipping the
            # softmax avoids a second [B,150,H,W] allocation
            preds = model(images).float().argmax(dim=1)
            update_confusion(conf, preds, masks, num_classes)
            continue
        else:
            h, w = int(images.shape[-2]), int(images.shape[-1])
            probs = torch.zeros(images.shape[0], num_classes, h, w,
                                dtype=torch.float32, device=device)
            for s in ms_scales:
                sh, sw = max(1, int(round(h * s))), max(1, int(round(w * s)))
                scaled = (images if (sh, sw) == (h, w) else
                          F.interpolate(images, size=(sh, sw),
                                        mode="bilinear", align_corners=False))
                views = [scaled] + ([torch.flip(scaled, dims=[-1])]
                                    if flip else [])
                for vi, view in enumerate(views):
                    out = model(view).float()
                    if vi == 1:
                        out = torch.flip(out, dims=[-1])
                    if out.shape[-2:] != (h, w):
                        out = F.interpolate(out, size=(h, w), mode="bilinear",
                                            align_corners=False)
                    probs += out.softmax(dim=1)
        preds = probs.argmax(dim=1)
        update_confusion(conf, preds, masks, num_classes)

    n_images = torch.tensor([float(n_local)], dtype=torch.float64,
                            device=device)
    if world_size > 1:
        dist.all_reduce(conf, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_images, op=dist.ReduceOp.SUM)
    return conf, int(round(n_images.item()))


def write_per_class_csv(path, iou, inter, gt, pred, union, class_names):
    """csv.writer, NOT ",".join: ADE20K class names legitimately contain
    commas ("person, individual, someone, somebody, mortal, soul"), which a
    naive join would splice into extra columns and silently shift every
    number in the row one field to the right."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(PER_CLASS_FIELDS)
    for c in range(len(iou)):
        val = iou[c]
        w.writerow([
            c,
            (class_names[c] if class_names else "MISSING"),
            ("MISSING" if not np.isfinite(val) else f"{float(val):.6f}"),
            int(inter[c]), int(union[c]), int(gt[c]), int(pred[c]),
        ])
    atomic_write_text(buf.getvalue(), path)


def load_class_names(data_root, num_classes, explicit=None):
    """ADE20K class names from the dataset's own objectInfo150.csv (row order
    == class index + 1, verified against its Idx column). Returns
    (names, source_description) or (None, reason) — names are reported
    MISSING rather than guessed.

    The official file is TAB-separated despite its .csv name (its `Name`
    column contains commas), so the delimiter is sniffed instead of assumed.

    Several candidate locations are tried because the staged tree does not
    always carry it: the 2026-09-08 smoke found the committed
    ADEChallengeData2016.zip extracts WITHOUT objectInfo150.csv, which left
    150 MISSING names in per_class_iou.csv and would have blocked Phase C's
    mandated sky/wall/floor rows. `explicit` (the matrix's
    `class_names_file`) wins when set, so pointing at a real file on the HPC
    is a config change rather than a code change. Numbers are unaffected
    either way — the class INDICES are always correct.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit) if Path(explicit).is_absolute()
                          else repo_path(explicit))
    root = Path(data_root)
    candidates += [root / "objectInfo150.csv", root / "objectInfo150.txt",
                   root.parent / "objectInfo150.csv",
                   root / "sceneparsing" / "objectInfo150.csv"]

    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        return None, ("objectInfo150.csv not found; tried "
                      + ", ".join(str(c) for c in candidates))
    raw = path.read_text(encoding="utf-8", errors="replace")
    for delim, label in (("\t", "tab"), (",", "comma"), (";", "semicolon")):
        rows = list(csv.DictReader(io.StringIO(raw), delimiter=delim))
        if not rows:
            continue
        key = next((k for k in rows[0]
                    if k and k.strip().lower() == "name"), None)
        if key is None:
            continue
        names = [(r.get(key) or "").strip() for r in rows]
        if len(names) != num_classes or not all(names):
            continue
        # The class index comes from ROW POSITION, so row order must be the
        # Idx order. Check the Idx column that is already parsed rather than
        # trusting it: a re-sorted file would attach every name in the
        # committed per_class_iou.csv to the wrong IoU.
        idx_key = next((k for k in rows[0]
                        if k and k.strip().lower() == "idx"), None)
        if idx_key is None:
            return None, (f"{path} ({label}-separated) has no Idx "
                          f"column, so row order cannot be verified")
        try:
            idx = [int((r.get(idx_key) or "").strip()) for r in rows]
        except ValueError:
            return None, (f"{path} ({label}-separated) has a "
                          f"non-integer Idx value")
        if idx != list(range(1, num_classes + 1)):
            return None, (f"{path} ({label}-separated) is not in Idx "
                          f"order 1..{num_classes}; names would be attached "
                          f"to the wrong classes")
        return names, (f"{path} ({label}-separated, Idx-verified)")
    return None, (f"{path} did not yield {num_classes} non-empty names "
                  f"under any of tab/comma/semicolon")


# ── the committed 20-image probe set ─────────────────────────────────────────

def load_fixed20(data_root, dataset):
    """Read results/probe/ade20k_fixed20.json and resolve it against the
    staged val split. Every listed stem must exist — a probe list that does
    not match the data is a hard error, never a silent subset."""
    path = repo_path(FIXED20)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build and commit the fixed-20 probe list "
            f"first (segmentation/tools/build_fixed20.py)")
    with open(path) as f:
        spec = json.load(f)
    stems = list(spec["stems"])
    if len(set(stems)) != len(stems):
        raise ValueError(f"{path} contains duplicate stems")
    available = {s: i for i, s in enumerate(dataset.stems)}
    missing = [s for s in stems if s not in available]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the {len(stems)} committed probe images are "
            f"not in the staged validation split (e.g. {missing[:3]}) — the "
            f"probe list and the data disagree")
    return stems, [available[s] for s in stems], file_sha256(path)


@torch.no_grad()
def dump_fixed20(model, dataset, indices, stems, device, out_dir,
                 num_classes):
    """Rank-0-only: per probe image write the predicted label map, the ground
    truth and the (val-transformed) RGB input, so Phase C can build the
    qualitative panel without the dataset."""
    from PIL import Image
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for stem, idx in zip(stems, indices):
        img, mask = dataset[idx]
        logits = model(img.unsqueeze(0).to(device))
        pred = logits.float().argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        if pred.max() >= num_classes:
            raise ValueError(f"prediction for {stem} has label "
                             f"{int(pred.max())} >= {num_classes}")

        def _png(arr):
            buf = io.BytesIO()
            Image.fromarray(arr, mode="L").save(buf, format="PNG",
                                                optimize=True)
            return buf.getvalue()

        atomic_bytes_write(_png(pred), out_dir / f"{stem}_pred.png")
        gt = mask.to(torch.int32).cpu().numpy()
        gt = np.where(gt == IGNORE_INDEX, 255, gt).astype(np.uint8)
        atomic_bytes_write(_png(gt), out_dir / f"{stem}_gt.png")

        rgb = (img.cpu() * std + mean).clamp(0, 1)
        rgb = (rgb * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=90)
        atomic_bytes_write(buf.getvalue(), out_dir / f"{stem}_img.jpg")


# ── training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, epoch, cfg, device,
                    rank, max_steps=None):
    model.train()
    t = cfg["train"]
    log_freq = cfg.get("logging", {}).get("log_freq", 50)
    amp_on = t["amp"] and device.type == "cuda"
    clip = t.get("grad_clip", 1.0)
    total_loss, n_batches, n_images = 0.0, 0, 0

    for step, (images, masks) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=amp_on):
            loss = model(images, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1
        n_images += int(images.shape[0])

        if rank == 0 and step % log_freq == 0:
            print(f"  [{epoch}][{step}/{len(loader)}]  "
                  f"loss={float(loss.item()):.4f}", flush=True)

    return total_loss / max(n_batches, 1), n_images


# ── main ──────────────────────────────────────────────────────────────────────

def run_training(matrix, run_id, data_root, out_root=None, resume="auto",
                 max_epochs=None, max_steps=None, max_eval_images=None,
                 device_str=None, ckpt_root=None):
    cfg = resolve_dense_config(matrix, run_id)
    if cfg["task"] != "segmentation":
        raise ValueError(f"run {run_id!r} is a {cfg['task']} run — use "
                         f"detection/tools/train.py")
    epochs_configured = int(cfg["train"]["epochs"])
    if max_epochs:
        cfg["train"]["epochs"] = int(max_epochs)
    total_epochs = int(cfg["train"]["epochs"])
    smoke = (total_epochs != epochs_configured or max_steps is not None
             or max_eval_images is not None)
    cfg["smoke"] = smoke

    out_root = repo_path(out_root or cfg["out_root"])
    run_dir = out_root / run_id
    ckpt_dir = (repo_path(ckpt_root) / run_id / "ckpt" if ckpt_root
                else run_dir / "ckpt")
    last_path = ckpt_dir / "last.pth"
    best_path = ckpt_dir / "best_model.pth"
    log_path = run_dir / "log.csv"

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ── fast path: a surplus chain job must not build a model or a dataset.
    # RANK 0 decides BOTH "already complete" and "resuming" and broadcasts
    # them (see the detection trainer): ranks that disagreed would run
    # different epoch counts and hang the job on the next collective.
    decision = [None]
    if rank == 0:
        done = resume == "auto" and run_is_complete(run_dir, total_epochs)
        decision = [{"complete": done,
                     "resuming": (not done) and resume == "auto"
                     and last_path.exists()}]
    if distributed:
        dist.broadcast_object_list(decision, src=0, device=device)
    resuming = bool(decision[0]["resuming"])
    if decision[0]["complete"]:
        if rank == 0:
            print(f"{run_id}: already complete (meta.end_time set, "
                  f"{log_epochs(log_path)} epochs logged) — nothing to do",
                  flush=True)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return "skipped"

    seed = int(cfg["seed"])
    set_seed(seed)

    backbone = resolve_backbone(cfg["backbone_spec"])
    cfg["backbone"] = backbone
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # ── data ──────────────────────────────────────────────────────────────
    t = cfg["train"]
    num_classes = int(cfg["model"]["num_classes"])
    crop = int(t["input_size"])
    min_sc, max_sc = t["random_scale"]
    val_size = int(cfg.get("eval", {}).get("input_size", crop))

    train_ds = ADE20KDataset(
        data_root, split="training",
        transforms=get_train_transforms(crop, min_sc, max_sc))
    val_ds = ADE20KDataset(
        data_root, split="validation",
        transforms=get_val_transforms(val_size))

    per_gpu = max(1, t["batch_size"] // world_size)
    train_sampler = (DistributedSampler(train_ds, shuffle=True, seed=seed)
                     if distributed else None)
    train_loader = DataLoader(
        train_ds, batch_size=per_gpu, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=t["num_workers"],
        pin_memory=device.type == "cuda", drop_last=True)

    shards = [shard_indices(len(val_ds), r, world_size)
              for r in range(world_size)]
    if max_eval_images is not None:                    # smoke only
        keep = max(1, int(max_eval_images) // world_size)
        shards = [s[:keep] for s in shards]
    n_val_expected = sum(len(s) for s in shards)
    val_loader = DataLoader(Subset(val_ds, shards[rank]), batch_size=1,
                            shuffle=False, num_workers=t["num_workers"],
                            pin_memory=device.type == "cuda")

    probe_stems, probe_indices, probe_sha = load_fixed20(data_root, val_ds)
    class_names, names_source = load_class_names(
        data_root, num_classes, explicit=cfg.get("class_names_file"))
    if class_names is None and rank == 0:
        print(f"  WARNING: class names unavailable ({names_source}) — "
              f"per_class_iou.csv will carry MISSING names, which blocks the "
              f"sky/wall/floor rows Phase C asks for", flush=True)

    if rank == 0:
        print(f"\n{'=' * 64}\n  {run_id}  ({cfg['variant']})\n"
              f"  backbone {backbone['ckpt']}  [{backbone['source']}]\n"
              f"  GPUs {world_size}  batch {t['batch_size']}  "
              f"epochs {total_epochs}  smoke={smoke}\n{'=' * 64}\n",
              flush=True)

    # ── model (this is where the backbone hash is verified) ───────────────
    segmentor = build_segmentor(cfg).to(device)

    # Environment is now validated (matrix resolved, data opened, probe list
    # resolved against the staged split, backbone strict-loaded and
    # hash-checked). ONLY NOW may a fresh start clear the previous attempt:
    # a resubmission into a broken environment must not destroy a crashed
    # attempt's evidence (TASK-08's ordering). Clearing miou_ms.json matters
    # in particular — a stale one makes the multi-scale eval skip itself and
    # the run would finalize carrying an mIoU from weights that no longer
    # exist.
    if rank == 0:
        if not resuming:
            clear_stale_artifacts([run_dir / SS_JSON, run_dir / MS_JSON,
                                   run_dir / PER_CLASS_CSV,
                                   run_dir / CONF_NPZ,
                                   run_dir / PROBE_DIR,
                                   best_path])
            create_run(out_root, run_id, cfg, seed)
            sanitize_log(log_path, LOG_FIELDS, 0)
        else:
            meta = read_meta(run_dir, run_id)
            if not meta:
                # meta.json was missing or torn (quarantined by read_meta):
                # rebuild the full provenance record rather than continue
                # with a meta that carries nothing but a resume list
                print("  rebuilding lost provenance record for this resume",
                      flush=True)
                create_run(out_root, run_id, cfg, seed)
                meta = read_meta(run_dir, run_id)
            meta.setdefault("resumes", []).append(
                {"cmd": " ".join(sys.argv), "world_size": world_size})
            atomic_json_dump(meta, run_dir / "meta.json")
    if distributed:
        dist.barrier()

    model = segmentor
    if distributed:
        model = DDP(segmentor, device_ids=[local_rank]
                    if device.type == "cuda" else None,
                    find_unused_parameters=True)

    backbone_params = list(segmentor.backbone.parameters())
    head_params = (list(segmentor.neck.parameters())
                   + list(segmentor.head.parameters()))
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": float(t["backbone_lr"])},
         {"params": head_params, "lr": float(t["head_lr"])}],
        weight_decay=float(t["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-7)
    amp_on = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu",
                                  enabled=amp_on)
    steps_per_epoch = len(train_loader) if max_steps is None else int(max_steps)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch, best_miou, best_epoch = 0, -1.0, -1
    if resuming:
        try:
            ckpt = torch.load(last_path, map_location="cpu",
                              weights_only=False)
        except Exception as exc:
            # An unreadable last.pth would otherwise kill THIS job and, via
            # --dependency=afterany, every remaining job of the chain the
            # same way. Quarantine it so the NEXT chain job finds no
            # checkpoint and fresh-starts, and fail this one loudly.
            if rank == 0:
                dead = last_path.with_name(
                    last_path.name + f".corrupt.{time.time_ns()}")
                try:
                    os.replace(last_path, dead)
                    print(f"  QUARANTINED unreadable {last_path} -> "
                          f"{dead.name}; the next job in the chain will "
                          f"fresh-start", flush=True)
                except OSError:
                    pass
            raise RuntimeError(
                f"could not read {last_path} ({exc}). It has been moved "
                f"aside; resubmit and the run restarts from scratch.") from exc
        segmentor.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if check_schedule_geometry(ckpt, steps_per_epoch, total_epochs):
            scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

        # PROVENANCE: the checkpoint we are resuming was produced by a
        # specific backbone, and the load_state_dict above has ALREADY
        # replaced the freshly-loaded backbone weights with it. If the
        # matrix now resolves a DIFFERENT backbone, the new one contributes
        # nothing to the run, yet meta.json and every results JSON written
        # after this point would name it as the source of the number.
        # configs/dense_matrix.yaml explicitly invites the trigger ("once
        # that run finishes, copy its ckpt_sha256 into sha256 below"), so
        # refuse rather than silently mis-attribute.
        ckpt_bb = ckpt.get("backbone_sha256")
        if ckpt_bb and ckpt_bb != segmentor.backbone.ckpt_sha256:
            raise RuntimeError(
                f"backbone changed under a resume: ckpt/last.pth was trained "
                f"from sha256 {ckpt_bb}, the matrix now resolves "
                f"{segmentor.backbone.ckpt_sha256} ({backbone['ckpt']}, "
                f"source={backbone['source']}). Resuming would keep the OLD "
                f"weights while recording the NEW backbone as their origin. "
                f"Start a new run id for the new backbone instead.")
        start_epoch = int(ckpt["epoch"]) + 1
        best_miou = float(ckpt.get("best_metric", -1.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        rng_states = ckpt.get("rng_states", [])
        if rank < len(rng_states):
            restore_rng_state(rng_states[rank])
        else:
            print(f"  WARNING: no saved RNG state for rank {rank} "
                  f"(world size changed?) — reseeding", flush=True)
            set_seed(seed + 100_000 + start_epoch + rank)
        if rank == 0:
            sanitize_log(log_path, LOG_FIELDS, start_epoch)
            print(f"Resumed at epoch {start_epoch}  "
                  f"best_mIoU={best_miou:.4f} (epoch {best_epoch})",
                  flush=True)
            problem = check_best_artifacts(run_dir, best_epoch)
            if problem:
                print(f"  best artifacts are stale/inconsistent ({problem}) "
                      f"— discarding them so the next eval rewrites the set",
                      flush=True)
                clear_stale_artifacts([run_dir / SS_JSON, run_dir / MS_JSON,
                                       run_dir / PER_CLASS_CSV,
                                       run_dir / CONF_NPZ,
                                       run_dir / PROBE_DIR])
                best_miou, best_epoch = -1.0, -1
    if distributed:
        dist.barrier()

    if rank == 0:
        meta_path = run_dir / "meta.json"
        meta = read_meta(run_dir, run_id)
        meta.update({
            "task": "segmentation",
            "variant": cfg["variant"],
            "arch": cfg["model"]["arch"],
            "backbone_run": backbone["run"],
            "backbone_ckpt": backbone["ckpt"],
            "backbone_source": backbone["source"],
            "backbone_note": backbone["note"],
            "backbone_sha256_expected": backbone["sha256"],
            "backbone_sha256_observed": segmentor.backbone.ckpt_sha256,
            "backbone_ckpt_meta": segmentor.backbone.ckpt_meta,
            "task_config_file": cfg["task_config_file"],
            "epochs": total_epochs,
            "epochs_configured": epochs_configured,
            "smoke": smoke,
            "steps_per_epoch": steps_per_epoch,
            "n_train_images": len(train_ds),
            "n_val_images": len(val_ds),
            "probe_list": FIXED20,
            "probe_list_sha256": probe_sha,
            "class_names_source": names_source,
            "ckpt_dir": str(ckpt_dir),
            "miou_definition": "confusion matrix over gt != 255 pixels; mean "
                               "over classes with union > 0 (bug B4 fixed)",
            "determinism": "seeded (matrix seed); GPU kernels not forced "
                           "deterministic — same policy as the e2r trainer",
        })
        atomic_json_dump(meta, meta_path)

    eval_freq = int(cfg.get("eval", {}).get("eval_freq", 10))
    freeze_epochs = int(t.get("freeze_backbone_epochs", 2))
    ms_scales = list(cfg.get("eval", {}).get("ms_scales", []) or [])
    ms_flip = bool(cfg.get("eval", {}).get("ms_flip", True))

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch < freeze_epochs:
            segmentor.backbone.freeze()
        else:
            segmentor.backbone.unfreeze()

        lr_bb = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        t0 = time.time()
        train_loss, n_images = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, cfg, device, rank,
            max_steps=max_steps)
        scheduler.step()
        if distributed:
            dist.barrier()
        epoch_seconds = time.time() - t0

        miou_value = ""
        is_eval_epoch = ((epoch + 1) % eval_freq == 0
                         or epoch == total_epochs - 1)
        if is_eval_epoch:
            conf, n_seen = evaluate_confusion(
                segmentor, val_loader, device, num_classes, world_size)
            if n_seen != n_val_expected:
                raise RuntimeError(
                    f"validated {n_seen} images but {n_val_expected} were "
                    f"expected — sharding is broken, refusing to report mIoU")
            summary, iou, inter, gt, pred, union = summarize_confusion(conf)
            miou_value = summary["mIoU"]
            segmentor.train()

        rng_states = gather_rng_states(distributed)

        if rank == 0:
            append_log_row(log_path, LOG_FIELDS, {
                "epoch": epoch,
                "lr_backbone": round(lr_bb, 10),
                "lr_head": round(lr_head, 10),
                "train_loss": round(train_loss, 4),
                "mIoU": miou_value,
                "img_per_sec": round(n_images * world_size
                                     / max(epoch_seconds, 1e-6), 2),
                "wall_time": round(epoch_seconds, 1),
            })
            print(f"[{epoch + 1}/{total_epochs}] loss={train_loss:.4f}  "
                  f"mIoU={miou_value if miou_value != '' else '-'}  "
                  f"t={epoch_seconds:.0f}s", flush=True)

            if is_eval_epoch and float(summary["mIoU"]) > best_miou:
                best_miou, best_epoch = float(summary["mIoU"]), epoch
                # artifacts first, miou_ss.json LAST (the pair's marker)
                atomic_npz_save(run_dir / CONF_NPZ,
                                conf=conf.detach().cpu().numpy(),
                                epoch=np.array(epoch))
                write_per_class_csv(run_dir / PER_CLASS_CSV, iou, inter, gt,
                                    pred, union, class_names)
                dump_fixed20(segmentor, val_ds, probe_indices, probe_stems,
                             device, run_dir / PROBE_DIR, num_classes)
                segmentor.train()
                atomic_torch_save({"model": segmentor.state_dict(),
                                   "epoch": epoch, "mIoU": best_miou,
                                   "run_id": run_id}, best_path)
                write_miou_json(run_dir / SS_JSON, cfg, backbone, segmentor,
                                epoch, summary, iou, n_seen, seed, smoke,
                                scales=None, flip=False,
                                probe_sha=probe_sha,
                                class_names_ok=class_names is not None)
                print(f"  *** new best mIoU {best_miou:.4f} -> {SS_JSON}",
                      flush=True)

            save_dense_checkpoint(
                last_path, model_sd=segmentor.state_dict(),
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_metric=best_miou, best_epoch=best_epoch,
                last_metric=(float(summary["mIoU"]) if is_eval_epoch
                             else None),
                rng_states=rng_states, steps_per_epoch=steps_per_epoch,
                total_epochs=total_epochs,
                extra={"run_id": run_id, "task": "segmentation",
                       "backbone_sha256": segmentor.backbone.ckpt_sha256})
        if distributed:
            dist.barrier()

    # ── multi-scale TTA, once, from the val-selected weights ──────────────
    # rank 0 decides whether it still has to run (a resubmitted job may find
    # miou_ms.json already there) and broadcasts: evaluate_confusion is
    # collective, so ranks disagreeing here would hang the job.
    do_ms = [None]
    if rank == 0:
        do_ms = [bool(ms_scales) and not (run_dir / MS_JSON).exists()]
    if distributed:
        dist.broadcast_object_list(do_ms, src=0, device=device)
    if do_ms[0]:
        if not best_path.exists():
            raise RuntimeError(
                f"{best_path} missing — cannot run the multi-scale eval on "
                f"the val-selected weights")
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        segmentor.load_state_dict(state["model"])
        segmentor.to(device)
        if rank == 0:
            print(f"\nMulti-scale eval (scales {ms_scales}, flip={ms_flip}) "
                  f"on the epoch-{state['epoch']} weights...", flush=True)
        conf, n_seen = evaluate_confusion(
            segmentor, val_loader, device, num_classes, world_size,
            ms_scales=ms_scales, flip=ms_flip)
        if n_seen != n_val_expected:
            raise RuntimeError(f"MS eval saw {n_seen} != {n_val_expected}")
        summary, iou, inter, gt, pred, union = summarize_confusion(conf)
        if rank == 0:
            write_miou_json(run_dir / MS_JSON, cfg, backbone, segmentor,
                            int(state["epoch"]), summary, iou, n_seen, seed,
                            smoke, scales=ms_scales, flip=ms_flip,
                            probe_sha=probe_sha,
                            class_names_ok=class_names is not None)
            print(f"  MS mIoU {summary['mIoU']:.4f} -> {MS_JSON}", flush=True)
    if distributed:
        dist.barrier()

    if rank == 0:
        if not (run_dir / SS_JSON).exists():
            raise RuntimeError(
                f"{run_dir / SS_JSON} was never written — no epoch produced "
                f"an evaluation; refusing to mark this run complete")
        problem = check_best_artifacts(run_dir, best_epoch)
        if problem:
            raise RuntimeError(
                f"refusing to mark {run_id} complete: {problem}. Every "
                f"results file must describe the same epoch as the "
                f"checkpoint's best; resubmit and the next job repairs them.")
        finalize_run(run_dir)          # end_time LAST = completion marker
        print(f"\nDone — {run_id}  best mIoU {best_miou:.4f} at epoch "
              f"{best_epoch}\n  -> {run_dir / SS_JSON}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return "completed"


def check_best_artifacts(run_dir, best_epoch):
    """Is the best-artifact SET consistent, and the one the checkpoint says is
    best? Returns None when fine, else a short reason.

    miou_ss.json is written last of {conf_matrix.npz, per_class_iou.csv,
    preds_fixed20/, best_model.pth, miou_ss.json}, but a marker only helps if
    someone READS it: a job killed inside that block leaves a half-updated
    set, and since best_miou is restored from the checkpoint, a re-run of the
    same epoch is not a "new best" and would never rewrite it.
    """
    run_dir = Path(run_dir)
    ss = run_dir / SS_JSON
    if best_epoch is None or best_epoch < 0:
        return f"checkpoint has no best epoch but {SS_JSON} exists"             if ss.exists() else None
    if not ss.exists():
        return f"{SS_JSON} missing for best epoch {best_epoch}"
    try:
        with open(ss) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return f"{SS_JSON} unreadable ({exc})"
    if int(payload.get("epoch", -1)) != int(best_epoch):
        return (f"{SS_JSON} is for epoch {payload.get('epoch')} but the "
                f"checkpoint's best is {best_epoch}")
    for name in (CONF_NPZ, PER_CLASS_CSV):
        if not (run_dir / name).exists():
            return f"{name} missing"
    try:
        conf_epoch = int(np.load(run_dir / CONF_NPZ)["epoch"])
    except Exception as exc:
        return f"{CONF_NPZ} unreadable ({exc})"
    if conf_epoch != int(best_epoch):
        return (f"{CONF_NPZ} is for epoch {conf_epoch} but the checkpoint's "
                f"best is {best_epoch}")
    return None


def write_miou_json(path, cfg, backbone, segmentor, epoch, summary, iou,
                    n_images, seed, smoke, scales, flip, probe_sha,
                    class_names_ok):
    atomic_json_dump({
        "run_id": cfg["run_id"],
        "task": "segmentation",
        "variant": cfg["variant"],
        "arch": cfg["model"]["arch"],
        "seed": seed,
        "epoch": epoch,
        "smoke": smoke,
        "eval_mode": "single_scale" if not scales else "multi_scale",
        "ms_scales": scales,
        "ms_flip": bool(flip) if scales else False,
        **summary,
        "iou_per_class": [None if not np.isfinite(v) else round(float(v), 6)
                          for v in iou],
        "n_images": n_images,
        "num_classes": int(cfg["model"]["num_classes"]),
        "ignore_index": IGNORE_INDEX,
        "miou_definition": "confusion matrix over gt != 255 pixels; mean over "
                           "classes with union > 0 (bug B4 fixed)",
        # mIoU/pixel_acc/mean_acc are PERCENTAGES while iou_per_class are
        # FRACTIONS; without this, Phase C has to guess and a 100x error in
        # T5_ade20k.csv would look plausible
        "units": {"mIoU": "percent", "pixel_acc": "percent",
                  "mean_acc": "percent", "iou_per_class": "fraction",
                  "per_class_iou_csv.iou_frac": "fraction"},
        "class_names_available": bool(class_names_ok),
        "probe_list": FIXED20,
        "probe_list_sha256": probe_sha,
        "backbone_run": backbone["run"],
        "backbone_ckpt": backbone["ckpt"],
        "backbone_source": backbone["source"],
        "backbone_sha256": segmentor.backbone.ckpt_sha256,
        "git_sha": git_sha(),
    }, path)


def main():
    p = argparse.ArgumentParser("SAGA dense segmentation trainer (TASK-09)")
    p.add_argument("--matrix", default="configs/dense_matrix.yaml")
    p.add_argument("--run", required=True, help="run_id in the dense matrix")
    p.add_argument("--data_root", required=True,
                   help="staged ADE20K root (ADEChallengeData2016/)")
    p.add_argument("--out_root", default=None,
                   help="default: the matrix's out_roots[segmentation]")
    p.add_argument("--ckpt_root", default=None,
                   help="put ckpt/ under <ckpt_root>/<run_id>/ instead of the "
                        "run dir (hpc quota escape hatch)")
    p.add_argument("--resume", choices=["auto", "none"], default="auto")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="smoke only — flags the run as smoke")
    p.add_argument("--max_steps", type=int, default=None,
                   help="smoke only — train iterations per epoch")
    p.add_argument("--max_eval_images", type=int, default=None,
                   help="smoke only — cap the eval pass")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    run_training(args.matrix, args.run, args.data_root,
                 out_root=args.out_root, resume=args.resume,
                 max_epochs=args.max_epochs, max_steps=args.max_steps,
                 max_eval_images=args.max_eval_images,
                 device_str=args.device, ckpt_root=args.ckpt_root)


if __name__ == "__main__":
    main()
