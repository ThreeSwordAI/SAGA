#!/usr/bin/env python3
"""
detection/tools/train.py
=========================
TASK-09 dense-prediction trainer — COCO 2017 detection, ViT-B backbone +
Simple Feature Pyramid + torchvision Faster R-CNN.

WHY IT WAS REWRITTEN (all prior detection numbers are void):
  - B5 double normalization — fixed in detection/models/detector.py
    (identity normalization inside FasterRCNN); asserted here on the first
    real batch of every job, ON DEVICE, and recorded in meta.json.
  - B6 registers backbone — fixed in detection/models/backbone.py (the timm
    register-token model is used directly, via its own
    forward_intermediates(); it is no longer wrapped in SAGAViT, which
    dropped the reg_token parameter).
  - category-id corruption — the old eval scored predictions in the
    contiguous 1-80 label space against ground-truth annotations that had
    kept their original 1-90 COCO ids, so classes were compared against the
    wrong classes and every annotation with an original id > 80 was silently
    dropped. Predictions are now mapped back through `idx_to_coco_id` and
    scored against the OFFICIAL annotation file.
  - the "new best AP" branch wrote its checkpoint to last.pth (never to a
    best file), so best-AP state was indistinguishable from last state.

TRAINING MATH IS THE COMMITTED ONE: every hyperparameter is read from
detection/configs/base.yaml through configs/dense_matrix.yaml — AdamW with
two param groups (backbone 1e-5 / head 1e-4), weight decay 0.05, grad clip
1.0 after unscale_, fp16 AMP, CosineAnnealingLR(T_max=epochs, eta_min=1e-7)
stepped per epoch, backbone frozen for the first epoch, 25 epochs, global
batch 8, 800/1333. `warmup_epochs` exists in that config but the committed
trainer never implemented warmup, so it stays unimplemented (implementing it
now would change the schedule, i.e. the numbers).

HYGIENE (mirrors the e2r trainer, TASK-05): run_registry provenance,
append-safe log.csv, atomic ckpt/last.pth EVERY epoch with per-rank RNG,
`--resume auto` (one command fresh-starts and resumes) with schedule
geometry guards, exact rank::world_size val sharding (no padded sampler),
and a fast-path exit for a surplus job in the dependency chain.

OUTPUTS  results/detection/<run_id>/
    meta.json                 provenance incl. backbone ckpt + sha256 +
                              which registers checkpoint was used, and the
                              on-device normalization check
    config.resolved.yaml
    log.csv                   one row per epoch
    coco_eval_best.json       all six APs (+ ARs) + per-category AP/AP50/AP_S
                              at the best-AP epoch; written LAST of the pair
    detections_val.json       raw COCO-format predictions at that epoch, so
                              PR curves / qualitative crops never need
                              re-inference
    ckpt/last.pth             git-ignored

Launch (identical fresh and on every resubmission):
    torchrun --nproc_per_node=4 detection/tools/train.py \
        --matrix configs/dense_matrix.yaml --run det_vitb_saga_s1 \
        --data_root $STAGE_DIR --resume auto
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from detection.data.coco_dataset import COCODetectionDataset, collate_fn
from detection.data.transforms import get_train_transforms, get_val_transforms
from detection.models.detector import build_detector
from saga.run_registry import create_run, file_sha256, finalize_run, git_sha
from tools.dense_runtime import (append_log_row, atomic_bytes_write,
                                 atomic_json_dump, check_schedule_geometry,
                                 clear_stale_artifacts, gather_rng_states,
                                 log_epochs, read_meta, repo_path,
                                 resolve_backbone, resolve_dense_config,
                                 restore_rng_state, run_is_complete,
                                 sanitize_log, save_dense_checkpoint,
                                 set_seed, shard_indices)

LOG_FIELDS = ["epoch", "lr_backbone", "lr_head", "train_loss", "AP", "AP50",
              "AP75", "AP_S", "AP_M", "AP_L", "img_per_sec", "wall_time"]
AP_KEYS = ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")
AR_KEYS = ("AR_1", "AR_10", "AR_100", "AR_S", "AR_M", "AR_L")
BEST_JSON = "coco_eval_best.json"
DETECTIONS_JSON = "detections_val.json"


# ── COCO evaluation ───────────────────────────────────────────────────────────

def boxes_to_original_frame(boxes, resize_scale):
    """Map [x1,y1,x2,y2] boxes from the frame the MODEL saw back to the true
    original image frame — the EXACT inverse of the dataloader's
    ResizeDetection, which multiplied the boxes by the single scalar
    `resize_scale`.

    Why the mapping is needed at all: this pipeline resizes in the
    DATALOADER, so by the time torchvision's postprocess runs, its
    "original_image_sizes" are already the resized ones and its rescaling is
    a no-op. COCO ground truth is in the true original frame, so without
    this a 480x640 image's predictions come out ~1.67x too large and score
    essentially zero AP.

    Why the SCALAR and not a per-axis size ratio: the forward transform
    scaled boxes by `scale` but rounded the image size to whole pixels, so
    `orig / round(orig * scale)` is only approximately `1 / scale`. Using
    the size ratio leaves a systematic sub-pixel stretch (measured up to
    0.39 px / 4.9e-4 relative) which costs a perfect detector several points
    of AP_S — the number Gate 2 turns on. Dividing by the recorded scalar is
    exact.
    """
    scale = float(resize_scale)
    if not scale > 0:
        raise ValueError(f"bad resize_scale {resize_scale}")
    return boxes.astype(np.float64) / scale


@torch.no_grad()
def infer_val_shard(model, loader, device, idx_to_coco_id):
    """Run inference on this rank's shard. Returns (results, image_ids) with
    results in OFFICIAL COCO format: original category ids AND original image
    coordinates. `image_ids` is every image actually seen — including ones
    that produced no detection, which must still be scored (they can only add
    false negatives)."""
    model.eval()
    results, image_ids = [], []
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)
        for img, pred, tgt in zip(images, preds, targets):
            img_id = int(tgt["image_id"].reshape(-1)[0].item())
            image_ids.append(img_id)
            if "resize_scale" not in tgt:
                raise KeyError(
                    "val targets carry no 'resize_scale' — predictions "
                    "cannot be mapped back to COCO's coordinate frame, so AP "
                    "would be meaningless (see boxes_to_original_frame). The "
                    "val transform must be get_val_transforms(), whose "
                    "ResizeDetection records it.")
            boxes = pred["boxes"].detach().float().cpu().numpy()
            boxes = boxes_to_original_frame(
                boxes, tgt["resize_scale"].reshape(-1)[0].item())
            scores = pred["scores"].detach().float().cpu().numpy()
            labels = pred["labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                label = int(label)
                if label not in idx_to_coco_id:
                    # background (0) or an out-of-range head output: a
                    # prediction we cannot name must never be scored
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                results.append({
                    "image_id": img_id,
                    "category_id": int(idx_to_coco_id[label]),
                    "bbox": [round(x1, 2), round(y1, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(float(score), 5),
                })
    return results, image_ids


def _cat_ap(precisions, k, iou_slice, area_idx, maxdet_idx=2):
    """Mean precision over the requested slice, ignoring -1 (class absent)."""
    p = precisions[iou_slice, :, k, area_idx, maxdet_idx]
    p = p[p > -1]
    return round(float(np.mean(p)) * 100, 3) if p.size else None


def score_coco(all_results, coco_gt, img_ids, cat_names):
    """COCOeval over `img_ids`; returns (metrics dict, per-category list)."""
    from pycocotools.cocoeval import COCOeval
    # loadRes MUTATES the dicts it is given (it adds a box-outline
    # `segmentation` polygon, an `area`, a synthetic `id` and `iscrowd`).
    # Those same dicts are what detections_val.json is written from, and a
    # fabricated segmentation polygon the model never predicted has no
    # business in a results file (it also inflates the artifact ~2.2x).
    # A shallow copy per entry is enough: loadRes only assigns new top-level
    # keys and never mutates `bbox` in place.
    coco_dt = coco_gt.loadRes([dict(r) for r in all_results])
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = sorted(int(i) for i in img_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # COCOeval reports -1 for a slice with no ground truth at all (e.g.
    # AP_M when no medium object exists). That is "no data", not a score, so
    # it becomes None — writing -100.0 into a results file would hand Phase C
    # a number that looks like a measurement.
    def _stat(v):
        v = float(v)
        return None if v <= -1.0 else round(v * 100, 3)

    stats = list(ev.stats)
    metrics = {k: _stat(stats[i]) for i, k in enumerate(AP_KEYS)}
    metrics.update({k: _stat(stats[6 + i]) for i, k in enumerate(AR_KEYS)})

    precisions = ev.eval["precision"]          # [T, R, K, A, M]
    per_category = []
    for k, cat_id in enumerate(ev.params.catIds):
        per_category.append({
            "category_id": int(cat_id),
            "name": cat_names.get(int(cat_id), "MISSING"),
            "AP": _cat_ap(precisions, k, slice(None), 0),
            "AP50": _cat_ap(precisions, k, slice(0, 1), 0),
            "AP_S": _cat_ap(precisions, k, slice(None), 1),
        })
    return metrics, per_category


def evaluate_coco(model, loader, device, rank, world_size, dataset,
                  n_val_expected, shard_dir):
    """All ranks infer their shard and write it to `shard_dir`; rank 0 merges
    and scores. Returns (metrics, per_category, all_results, n_images) —
    per_category/all_results are None off rank 0; `metrics` is broadcast so
    every rank agrees on what "best" means."""
    results, image_ids = infer_val_shard(model, loader, device,
                                         dataset.idx_to_coco_id)

    shard_dir.mkdir(parents=True, exist_ok=True)
    atomic_bytes_write(
        json.dumps({"results": results, "image_ids": image_ids},
                   separators=(",", ":")).encode(),
        shard_dir / f"rank{rank}.json")

    n_images = torch.tensor([float(len(image_ids))], dtype=torch.float64,
                            device=device)
    if world_size > 1:
        dist.all_reduce(n_images, op=dist.ReduceOp.SUM)
        dist.barrier()      # every shard file is on disk before the merge
    n_total = int(round(n_images.item()))
    if n_total != n_val_expected:
        raise RuntimeError(
            f"inference covered {n_total} val images but {n_val_expected} "
            f"were expected — sharding is broken, refusing to report AP")

    payload = [None]
    all_results = None
    per_category = None
    if rank == 0:
        all_results, all_ids = [], []
        for r in range(world_size):
            with open(shard_dir / f"rank{r}.json") as f:
                shard = json.load(f)
            all_results.extend(shard["results"])
            all_ids.extend(shard["image_ids"])
        if len(set(all_ids)) != len(all_ids):
            raise RuntimeError(
                f"{len(all_ids) - len(set(all_ids))} val images were "
                f"evaluated more than once — refusing to report AP")
        if not all_results:
            print("  WARNING: no predictions at all — every AP is 0",
                  flush=True)
            metrics = {k: 0.0 for k in AP_KEYS}
            metrics.update({k: 0.0 for k in AR_KEYS})
            per_category = []
        else:
            coco_gt = dataset.official_coco_gt()
            cat_names = {int(c["id"]): c["name"]
                         for c in coco_gt.loadCats(coco_gt.getCatIds())}
            metrics, per_category = score_coco(
                all_results, coco_gt, all_ids, cat_names)
        payload = [metrics]
        for r in range(world_size):          # transient, not an artifact
            (shard_dir / f"rank{r}.json").unlink(missing_ok=True)
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=device)
    return payload[0], per_category, all_results, n_total


# ── training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, epoch, cfg, device,
                    rank, detector, max_steps=None, norm_check=None):
    model.train()
    log_freq = cfg.get("logging", {}).get("log_freq", 50)
    amp_on = cfg["train"]["amp"] and device.type == "cuda"
    clip = cfg["train"].get("grad_clip", 1.0)
    total_loss, n_batches, n_images = 0.0, 0, 0

    for step, (images, targets) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                   for t in targets]

        if norm_check is not None and step == 0:
            # B5, ON DEVICE, on a real batch, at the start of every job
            norm_check.append(detector.check_normalization(images))

        with torch.autocast("cuda", enabled=amp_on):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1
        n_images += len(images)

        if rank == 0 and step % log_freq == 0:
            parts = "  ".join(f"{k}={float(v.item()):.4f}"
                              for k, v in loss_dict.items())
            print(f"  [{epoch}][{step}/{len(loader)}] "
                  f"total={float(loss.item()):.4f}  {parts}", flush=True)

    return total_loss / max(n_batches, 1), n_images


# ── main ──────────────────────────────────────────────────────────────────────

def run_training(matrix, run_id, data_root, out_root=None, resume="auto",
                 max_epochs=None, max_steps=None, max_eval_images=None,
                 device_str=None, ckpt_root=None):
    cfg = resolve_dense_config(matrix, run_id)
    if cfg["task"] != "detection":
        raise ValueError(f"run {run_id!r} is a {cfg['task']} run — use "
                         f"segmentation/tools/train.py")
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
    # them: if ranks disagreed (a lazily cached network FS) some would exit
    # or run a different number of epochs while the others blocked on a
    # collective, hanging the job until the wall clock.
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

    # fail fast, before 10-15 min of COCO staging and hours of training: the
    # eval needs pycocotools, and discovering that at the first eval epoch
    # would throw away everything trained so far
    try:
        import pycocotools.cocoeval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for COCO scoring but is not importable "
            "in this environment (pip install -r requirements.txt). Refusing "
            "to start a detection run whose evaluation cannot run.") from exc

    seed = int(cfg["seed"])
    set_seed(seed)                      # identical construction on every rank

    backbone = resolve_backbone(cfg["backbone_spec"])
    cfg["backbone"] = backbone
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # ── data ──────────────────────────────────────────────────────────────
    data_root = Path(data_root)
    t = cfg["train"]
    train_ds = COCODetectionDataset(
        img_dir=data_root / "train2017",
        ann_file=data_root / "annotations" / "instances_train2017.json",
        transforms=get_train_transforms(t["min_size"], t["max_size"]))
    # include_empty=True: AP over the STANDARD 5000-image val2017 split
    val_ds = COCODetectionDataset(
        img_dir=data_root / "val2017",
        ann_file=data_root / "annotations" / "instances_val2017.json",
        transforms=get_val_transforms(t["min_size"], t["max_size"]),
        include_empty=True)

    per_gpu = max(1, t["batch_size"] // world_size)
    train_sampler = (DistributedSampler(train_ds, shuffle=True, seed=seed)
                     if distributed else None)
    train_loader = DataLoader(
        train_ds, batch_size=per_gpu, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=t["num_workers"],
        collate_fn=collate_fn, pin_memory=device.type == "cuda")

    # exact sharding (B1 pattern): every val image on exactly one rank
    shards = [shard_indices(len(val_ds), r, world_size)
              for r in range(world_size)]
    if max_eval_images is not None:                    # smoke only
        keep = max(1, int(max_eval_images) // world_size)
        shards = [s[:keep] for s in shards]
    val_shard = shards[rank]
    n_val_expected = sum(len(s) for s in shards)
    val_loader = DataLoader(Subset(val_ds, val_shard), batch_size=1,
                            shuffle=False, num_workers=t["num_workers"],
                            collate_fn=collate_fn,
                            pin_memory=device.type == "cuda")

    if rank == 0:
        print(f"\n{'=' * 64}\n  {run_id}  ({cfg['variant']})\n"
              f"  backbone {backbone['ckpt']}  [{backbone['source']}]\n"
              f"  GPUs {world_size}  batch {t['batch_size']}  "
              f"epochs {total_epochs}  smoke={smoke}\n{'=' * 64}\n", flush=True)

    # ── model (this is where the backbone hash is verified) ───────────────
    detector = build_detector(cfg).to(device)

    # Environment is now validated (matrix resolved, data opened, backbone
    # strict-loaded and hash-checked). ONLY NOW may a fresh start touch the
    # previous attempt's files: a resubmission into a broken environment must
    # not destroy a crashed attempt's evidence (TASK-08's ordering).
    if rank == 0:
        if not resuming:
            clear_stale_artifacts([run_dir / BEST_JSON,
                                   run_dir / DETECTIONS_JSON,
                                   run_dir / (DETECTIONS_JSON + ".gz"),
                                   run_dir / "eval_shards"])
            create_run(out_root, run_id, cfg, seed)
            # no checkpoint => a leftover log.csv is from a crashed attempt
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

    model = detector
    if distributed:
        model = DDP(detector, device_ids=[local_rank]
                    if device.type == "cuda" else None,
                    find_unused_parameters=True)

    backbone_params = list(detector.backbone.parameters())
    head_params = (list(detector.neck.parameters())
                   + list(detector.frcnn.parameters()))
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
    start_epoch, best_ap, best_epoch = 0, -1.0, -1
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
        detector.load_state_dict(ckpt["model"])
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
        if ckpt_bb and ckpt_bb != detector.backbone.ckpt_sha256:
            raise RuntimeError(
                f"backbone changed under a resume: ckpt/last.pth was trained "
                f"from sha256 {ckpt_bb}, the matrix now resolves "
                f"{detector.backbone.ckpt_sha256} ({backbone['ckpt']}, "
                f"source={backbone['source']}). Resuming would keep the OLD "
                f"weights while recording the NEW backbone as their origin. "
                f"Start a new run id for the new backbone instead.")
        start_epoch = int(ckpt["epoch"]) + 1
        best_ap = float(ckpt.get("best_metric", -1.0))
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
            print(f"Resumed at epoch {start_epoch}  best_AP={best_ap:.3f} "
                  f"(epoch {best_epoch})", flush=True)
            # The best-artifact PAIR is written detections-then-JSON, but the
            # marker is only useful if someone READS it: a job killed inside
            # write_best_artifacts leaves a half-updated pair, and since
            # best_ap is restored from the checkpoint, a re-run of that epoch
            # is not a "new best" and would never rewrite it. Reconcile here
            # — if the pair disagrees with the checkpoint, forget the best so
            # the next eval epoch rewrites the whole set.
            problem = check_best_artifacts(run_dir, best_epoch)
            if problem:
                print(f"  best artifacts are stale/inconsistent ({problem}) "
                      f"— discarding them so the next eval rewrites the set",
                      flush=True)
                clear_stale_artifacts([run_dir / BEST_JSON,
                                       run_dir / DETECTIONS_JSON,
                                       run_dir / (DETECTIONS_JSON + ".gz")])
                best_ap, best_epoch = -1.0, -1
    if distributed:
        dist.barrier()

    if rank == 0:
        meta_path = run_dir / "meta.json"
        meta = read_meta(run_dir, run_id)
        meta.update({
            "task": "detection",
            "variant": cfg["variant"],
            "arch": cfg["model"]["arch"],
            "backbone_run": backbone["run"],
            "backbone_ckpt": backbone["ckpt"],
            "backbone_source": backbone["source"],
            "backbone_note": backbone["note"],
            "backbone_sha256_expected": backbone["sha256"],
            "backbone_sha256_observed": detector.backbone.ckpt_sha256,
            "backbone_ckpt_meta": detector.backbone.ckpt_meta,
            "task_config_file": cfg["task_config_file"],
            "epochs": total_epochs,
            "epochs_configured": epochs_configured,
            "smoke": smoke,
            "steps_per_epoch": steps_per_epoch,
            "n_train_images": len(train_ds),
            "n_val_images": len(val_ds),
            "ckpt_dir": str(ckpt_dir),
            "determinism": "seeded (matrix seed); GPU kernels not forced "
                           "deterministic — same policy as the e2r trainer",
        })
        atomic_json_dump(meta, meta_path)

    eval_freq = int(cfg.get("eval", {}).get("eval_freq", 5))
    freeze_epochs = int(t.get("freeze_backbone_epochs", 1))
    shard_dir = run_dir / "eval_shards"
    norm_stats = None

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch < freeze_epochs:
            detector.backbone.freeze()
        else:
            detector.backbone.unfreeze()

        norm_check = [] if norm_stats is None else None
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        t0 = time.time()
        train_loss, n_images = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, cfg, device, rank,
            detector, max_steps=max_steps, norm_check=norm_check)
        scheduler.step()
        if norm_check:
            norm_stats = norm_check[0]
            if rank == 0:
                meta_path = run_dir / "meta.json"
                meta = read_meta(run_dir, run_id)
                meta["normalization_check"] = norm_stats
                atomic_json_dump(meta, meta_path)
                print(f"  B5 normalization check PASSED: max_abs_diff="
                      f"{norm_stats['max_abs_elementwise_diff']:.2e}, "
                      f"backbone-input channel means "
                      f"{norm_stats['backbone_input_channel_mean']}",
                      flush=True)
        if distributed:
            dist.barrier()
        epoch_seconds = time.time() - t0

        metrics = {}
        is_eval_epoch = ((epoch + 1) % eval_freq == 0
                         or epoch == total_epochs - 1)
        if is_eval_epoch:
            metrics, per_category, all_results, n_seen = evaluate_coco(
                model=detector, loader=val_loader, device=device, rank=rank,
                world_size=world_size, dataset=val_ds,
                n_val_expected=n_val_expected, shard_dir=shard_dir)
            detector.train()

        rng_states = gather_rng_states(distributed)

        if rank == 0:
            append_log_row(log_path, LOG_FIELDS, {
                "epoch": epoch,
                "lr_backbone": round(lr_bb, 10),
                "lr_head": round(lr_head, 10),
                "train_loss": round(train_loss, 4),
                **{k: metrics.get(k, "") for k in AP_KEYS},
                "img_per_sec": round(n_images * world_size
                                     / max(epoch_seconds, 1e-6), 2),
                "wall_time": round(epoch_seconds, 1),
            })
            if is_eval_epoch:
                shown = "  ".join(f"{k}={metrics[k]}"
                                  for k in ("AP", "AP50", "AP_S"))
                print(f"[{epoch + 1}/{total_epochs}] loss={train_loss:.4f}  "
                      f"{shown}  t={epoch_seconds:.0f}s", flush=True)
            else:
                print(f"[{epoch + 1}/{total_epochs}] loss={train_loss:.4f}  "
                      f"(no eval)  t={epoch_seconds:.0f}s", flush=True)

            ap_now = metrics.get("AP") if is_eval_epoch else None
            if ap_now is not None and float(ap_now) > best_ap:
                best_ap, best_epoch = float(ap_now), epoch
                write_best_artifacts(run_dir, cfg, backbone, detector, epoch,
                                     metrics, per_category, all_results,
                                     n_seen, seed, smoke, norm_stats)
                print(f"  *** new best AP {best_ap:.3f} -> {BEST_JSON}",
                      flush=True)

            save_dense_checkpoint(
                last_path, model_sd=detector.state_dict(),
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_metric=best_ap, best_epoch=best_epoch,
                last_metric=metrics.get("AP"), rng_states=rng_states,
                steps_per_epoch=steps_per_epoch, total_epochs=total_epochs,
                extra={"run_id": run_id, "task": "detection",
                       "backbone_sha256": detector.backbone.ckpt_sha256})
        if distributed:
            dist.barrier()

    if rank == 0:
        if not (run_dir / BEST_JSON).exists():
            raise RuntimeError(
                f"{run_dir / BEST_JSON} was never written — no epoch produced "
                f"an evaluation; refusing to mark this run complete")
        problem = check_best_artifacts(run_dir, best_epoch)
        if problem:
            raise RuntimeError(
                f"refusing to mark {run_id} complete: {problem}. The two "
                f"results files must describe the same epoch as the "
                f"checkpoint's best; resubmit and the next job repairs them.")
        finalize_run(run_dir)          # end_time LAST = completion marker
        print(f"\nDone — {run_id}  best AP {best_ap:.3f} at epoch "
              f"{best_epoch}\n  -> {run_dir / BEST_JSON}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return "completed"


def check_best_artifacts(run_dir, best_epoch):
    """Is the best-artifact pair internally consistent AND the one the
    checkpoint says is best? Returns None when fine, else a short reason.

    Used on resume (to discard a half-written pair) and before finalize_run
    (so a run can never be marked complete while its two results files
    describe different epochs, or a different epoch than the checkpoint).
    """
    best_path = Path(run_dir) / BEST_JSON
    det_path = Path(run_dir) / DETECTIONS_JSON
    if best_epoch is None or best_epoch < 0:
        return ("checkpoint has no best epoch but "
                f"{BEST_JSON} exists") if best_path.exists() else None
    if not best_path.exists():
        return f"{BEST_JSON} missing for best epoch {best_epoch}"
    try:
        with open(best_path) as f:
            best = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return f"{BEST_JSON} unreadable ({exc})"
    if int(best.get("epoch", -1)) != int(best_epoch):
        return (f"{BEST_JSON} is for epoch {best.get('epoch')} but the "
                f"checkpoint's best is {best_epoch}")
    if not det_path.exists():
        return f"{DETECTIONS_JSON} missing"
    if best.get("detections_sha256") != file_sha256(det_path):
        return f"{DETECTIONS_JSON} does not match its recorded sha256"
    return None


def write_best_artifacts(run_dir, cfg, backbone, detector, epoch, metrics,
                         per_category, all_results, n_val_seen, seed, smoke,
                         norm_stats):
    """detections first, coco_eval_best.json LAST (it references the
    detections file's size and sha256, so the pair can never disagree)."""
    det_path = run_dir / DETECTIONS_JSON
    payload = json.dumps(all_results or [], separators=(",", ":")).encode()
    atomic_bytes_write(payload, det_path)

    result = {
        "run_id": cfg["run_id"],
        "task": "detection",
        "variant": cfg["variant"],
        "arch": cfg["model"]["arch"],
        "seed": seed,
        "epoch": epoch,
        "smoke": smoke,
        **{k: metrics[k] for k in AP_KEYS},
        **{k: metrics[k] for k in AR_KEYS},
        "per_category": per_category,
        "n_val_images": n_val_seen,
        "n_detections": len(all_results or []),
        "val_protocol": "official instances_val2017.json as ground truth; "
                        "predictions mapped back to official category ids; "
                        "all val images including annotation-free ones",
        "backbone_run": backbone["run"],
        "backbone_ckpt": backbone["ckpt"],
        "backbone_source": backbone["source"],
        "backbone_sha256": detector.backbone.ckpt_sha256,
        "detections_file": DETECTIONS_JSON,
        "detections_bytes": len(payload),
        "detections_sha256": file_sha256(det_path),
        "normalization_check": norm_stats,
        "git_sha": git_sha(),
    }
    atomic_json_dump(result, run_dir / BEST_JSON)


def main():
    p = argparse.ArgumentParser("SAGA dense detection trainer (TASK-09)")
    p.add_argument("--matrix", default="configs/dense_matrix.yaml")
    p.add_argument("--run", required=True, help="run_id in the dense matrix")
    p.add_argument("--data_root", required=True,
                   help="staged COCO root (train2017/, val2017/, annotations/)")
    p.add_argument("--out_root", default=None,
                   help="default: the matrix's out_roots[detection]")
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
