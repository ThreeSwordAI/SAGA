#!/usr/bin/env python3
"""
evaluation/e6_finegrained/tools/train_ft.py
===========================================
TASK-08: clean fine-grained transfer protocol (CUB-200-2011, FGVC-Aircraft).
Replaces the legacy evaluation/e6_finegrained/tools/train.py, whose numbers
are VOID (bug B7: it used the official TEST split as its val set, selected
best.pth on it, and evaluated it every 5 epochs — a test-tuned peak).

Protocol (the B7 fix):
  - train on the official train split MINUS a frozen, committed, stratified
    10% val split (results/ftsplit/<dataset>_val_split.json, built once by
    tools/build_ft_split.py); disjointness re-asserted here at every start.
  - validate on that val split EVERY epoch; best.pth is selected on val.
  - the official TEST split is touched EXACTLY ONCE per run: after training,
    best.pth is loaded and evaluated into eval/test_final.json. The test
    loader is constructed nowhere else (final_test_eval is the only place),
    and a second call refuses.

Hyperparameters MIRROR the legacy e6 trainer (verified against its code and
scripts): head replacement (trunc_normal std=0.02, zero bias), AdamW with
two param groups (backbone lr 1e-5, head lr 1e-3), weight decay 0.05,
CosineAnnealingLR per epoch (eta_min 1e-7), 100 epochs, batch 64,
CrossEntropyLoss(label_smoothing=0.1), grad clip 1.0 after unscale_, fp16
AMP, legacy augmentation/eval transforms, input resolution 224.
Resolution note: the legacy runs used 224 everywhere, and 224 is the grid
the backbones (and SAGA's phi = [H, 196]) were trained on, so no pos-embed
or gate interpolation is needed; any other img_size is REFUSED because the
interpolation path is unverified (and strict loading would reject the
pos_embed/phi shapes anyway).

Backbones: the seeded e2r mixup s1 checkpoints
(results/runs/e2r_<arch>_mixup_<variant>_s1/ckpt/last.pth), loaded via
tools/model_factory with strict=True — zero missing/unexpected keys after
prefix handling, or it raises (a strict=False bug burned this project
before). The backbone file's sha256 is asserted against the matrix value
(which tests pin to the committed e2r eval JSONs) and recorded in meta.json
and eval/test_final.json.

Hygiene (light): full seeding (ft_seed, separate from the backbone's
training seed which is fixed at s1), saga.run_registry integration,
append-mode log.csv (epoch, lr, train_loss, val_top1), atomic last.pth
every epoch. NO resume machinery — every run is well under the 24 h wall
(1-3 h), so an interrupted run simply restarts from scratch; the requeue
guard below makes resubmission after COMPLETION a no-op (idempotent), and
a fresh start truncates any stale log.csv/ckpts from a crashed attempt.

Launch (one GPU):
    python evaluation/e6_finegrained/tools/train_ft.py \
        --matrix configs/ft_matrix.yaml --run ft_cub_vits_saga_bs1_f0 \
        --stage_base $STAGE_DIR
"""

import argparse
import csv
import json
import os
import random
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.e6_finegrained.data.ft_meta import (
    FTItemsDataset, build_eval_transform, build_train_transform,
    official_splits)
from saga.run_registry import (create_run, file_sha256, finalize_run,
                               git_sha)
from tools.model_factory import build_model, load_checkpoint

# task-mandated log schema — exactly these four fields
LOG_FIELDS = ["epoch", "lr", "train_loss", "val_top1"]
MARKER_NAME = "test_final.json"  # completion marker, written LAST
LOCK_NAME = "run.lock"           # guards against concurrent double-submission
LOCK_STALE_SECONDS = 1800        # heartbeat-touched every epoch; older = dead


# ── Config resolution ─────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_run_config(matrix, run_id: str) -> dict:
    """Resolve one run of configs/ft_matrix.yaml into a flat config dict.
    `matrix` is a path or an already-loaded dict (tests)."""
    if not isinstance(matrix, dict):
        with open(matrix) as f:
            matrix = yaml.safe_load(f)
    if run_id not in matrix["runs"]:
        raise KeyError(f"run {run_id!r} not in the ft matrix")
    run = matrix["runs"][run_id]

    cfg = dict(matrix.get("defaults", {}))
    for key in ("dataset", "arch", "variant", "backbone", "ft_seed"):
        if key not in run:
            raise KeyError(f"run {run_id!r} is missing required key {key!r}")
    dataset = run["dataset"]
    cfg.update({
        "run_id": run_id,
        "dataset": dataset,
        "arch": run["arch"],
        "variant": run["variant"],
        "ft_seed": int(run["ft_seed"]),
        "val_split": matrix["splits"][dataset],
        "data_tar": matrix.get("data", {}).get(f"{dataset}_tar"),
        "expected_counts": matrix.get("expected_counts", {}).get(dataset),
        "backbone": dict(matrix["backbones"][run["backbone"]]),
    })
    # per-run overrides (tests / smoke; empty in production)
    cfg = deep_merge(cfg, run.get("overrides", {}))

    if cfg["img_size"] != 224:
        raise ValueError(
            f"img_size {cfg['img_size']} != 224: the legacy e6 protocol and "
            f"the e2r backbones (incl. SAGA's phi=[H,196] gate grid) are "
            f"224-only; the pos-embed/gate interpolation path is unverified "
            f"for this protocol, so any other resolution is refused")
    return cfg


# ── Seeding (ft_seed; the backbone's own training seed is fixed at s1) ────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


# ── Atomic writes + append-safe log (same pattern as the e2r trainer) ─────────

def atomic_torch_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # data durable BEFORE the rename publishes it
    os.replace(tmp, path)


def append_log_row(log_path: Path, row: dict):
    new = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ── Data ──────────────────────────────────────────────────────────────────────

def stage_dataset(cfg, stage_base: str) -> str:
    """Extract the dataset tarball to stage_base (legacy stage functions).
    The legacy functions treat directory EXISTENCE as completeness, but a
    requeued SLURM task can reuse its job id and find a half-extracted tree
    from a killed attempt — so completion is marked by a sentinel written
    LAST, and a tree without it is wiped and re-extracted."""
    if not cfg.get("data_tar"):
        raise ValueError(f"no {cfg['dataset']}_tar in the matrix data block "
                         f"— cannot stage; give --data_root instead")
    target = Path(stage_base) / f"{cfg['dataset']}_ft"
    sentinel = target / ".staged_ok"
    if target.exists() and not sentinel.exists():
        print(f"  {target} exists without {sentinel.name} — partial "
              f"extraction from a killed attempt, wiping", flush=True)
        shutil.rmtree(target)
    if cfg["dataset"] == "cub":
        from evaluation.e6_finegrained.data.cub_dataset import stage_cub
        root = stage_cub(cfg["data_tar"], str(target))
    else:
        from evaluation.e6_finegrained.data.aircraft_dataset import (
            stage_aircraft)
        root = stage_aircraft(cfg["data_tar"], str(target))
    sentinel.touch()
    return root


def load_items(cfg, data_root):
    """Load the frozen split file and the official enumeration, re-assert
    every disjointness/consistency guarantee, and return
    (train_items, val_items, test_items, n_classes, split_sha256)."""
    split_path = REPO_ROOT / cfg["val_split"] \
        if not Path(cfg["val_split"]).is_absolute() else Path(cfg["val_split"])
    if not split_path.exists():
        raise FileNotFoundError(
            f"frozen val split {split_path} not found — build and commit it "
            f"first (evaluation/e6_finegrained/tools/build_ft_split.py)")
    with open(split_path) as f:
        split = json.load(f)
    if split["dataset"] != cfg["dataset"]:
        raise ValueError(f"split file is for {split['dataset']!r}, "
                         f"run is {cfg['dataset']!r}")

    # pin WHICH carve this is — a rebuilt split (other seed / val_frac)
    # must not silently replace the frozen one
    for field, key in (("seed", "split_seed"), ("val_frac", "val_frac")):
        if cfg.get(key) is not None and split[field] != cfg[key]:
            raise ValueError(
                f"split file has {field}={split[field]}, the matrix pins "
                f"{key}={cfg[key]} — this is not the frozen carve")

    official_train, test_items, n_classes = official_splits(
        cfg["dataset"], root=data_root)
    train_items = [tuple(it) for it in split["items_train"]]
    val_items = [tuple(it) for it in split["items_val"]]

    train_set = set(it[0] for it in train_items)
    val_set = set(it[0] for it in val_items)
    test_set = set(it[0] for it in test_items)
    official_set = set(it[0] for it in official_train)

    # B7 guards: the val split must be a carve-out of the official train
    # split, nothing may intersect the official test split, and the split
    # file must be internally consistent. Explicit raises (not asserts) so
    # PYTHONOPTIMIZE can never strip them.
    def _refuse(cond, msg):
        if cond:
            raise ValueError(f"split guard failed: {msg}")

    _refuse(len(train_items) != len(train_set)
            or len(val_items) != len(val_set),
            "duplicate items inside the split file")
    _refuse(split["n_train"] != len(train_items)
            or split["n_val"] != len(val_items),
            "split file n_train/n_val disagree with its own item lists")
    _refuse(not val_set <= official_set,
            "val split is not a subset of the official train split — "
            "split file does not match the staged data")
    _refuse(bool(official_set & test_set),
            "official train and TEST splits overlap in the staged data")
    _refuse(bool(val_set & test_set), "val split intersects the TEST split")
    _refuse(bool(train_set & val_set), "train and val splits intersect")
    _refuse(train_set | val_set != official_set,
            "train + val do not reassemble the official train split")
    # not just the paths — the LABELS must agree with the official metadata
    _refuse({**dict(train_items), **dict(val_items)} != dict(official_train),
            "split-file labels disagree with the official metadata")
    _refuse(split["n_official_train"] != len(official_train),
            f"split file was built from {split['n_official_train']} official "
            f"train images, staged data has {len(official_train)}")
    _refuse(split["n_classes"] != n_classes,
            f"split file has {split['n_classes']} classes, staged data "
            f"has {n_classes}")

    exp = cfg.get("expected_counts")
    if exp:
        _refuse(len(official_train) != exp["official_train"],
                f"official train count {len(official_train)} != "
                f"expected {exp['official_train']}")
        _refuse(len(test_items) != exp["test"],
                f"official test count {len(test_items)} != "
                f"expected {exp['test']}")
        _refuse(n_classes != exp["n_classes"],
                f"{n_classes} classes != expected {exp['n_classes']}")

    return (train_items, val_items, test_items, n_classes,
            file_sha256(split_path))


# ── Model (strict load + legacy head replacement) ─────────────────────────────

def load_backbone(cfg, device, n_classes: int):
    """Build the backbone exactly as trained (tools/model_factory mirrors
    the e2r trainer), strict-load the checkpoint (zero missing/unexpected
    keys or it raises), verify its sha256, then replace the head the legacy
    e6 way. Returns (model, backbone_sha256, ckpt_meta)."""
    bb = cfg["backbone"]
    ckpt_path = REPO_ROOT / bb["ckpt"] \
        if not Path(bb["ckpt"]).is_absolute() else Path(bb["ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"backbone checkpoint {ckpt_path} not found")

    sha = file_sha256(ckpt_path)
    expected = bb.get("sha256")
    if expected and sha != expected:
        raise RuntimeError(
            f"backbone sha256 mismatch for {ckpt_path}:\n  expected "
            f"{expected}\n  found    {sha}\nThis is not the checkpoint the "
            f"matrix (and the committed e2r eval numbers) refer to — "
            f"refusing to fine-tune from it.")

    model = build_model(cfg["arch"], cfg["variant"],
                        img_size=cfg["img_size"], num_classes=1000)
    # strict=True: zero missing, zero unexpected keys after 'module.'
    # prefix handling, or load_checkpoint raises (B7's strict=False burned
    # this project before)
    ckpt_meta = load_checkpoint(model, ckpt_path)

    # legacy e6 head replacement, verbatim
    model.head = nn.Linear(model.embed_dim, n_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model.to(device), sha, ckpt_meta


# ── Train / eval loops (legacy e6 math) ───────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    amp_on, grad_clip):
    model.train()
    total_loss, total_n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=amp_on):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    """Full-precision eval; returns (top1, top5, n_seen)."""
    model.eval()
    correct1, correct5, n = 0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct1 += (logits.argmax(1) == labels).sum().item()
        _, top5 = logits.topk(min(5, logits.size(1)), dim=1)
        correct5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
        n += images.size(0)
    return 100.0 * correct1 / n, 100.0 * correct5 / n, n


def final_test_eval(model, cfg, run_dir: Path, test_items, data_root,
                    device, extra: dict):
    """THE ONLY PLACE THE OFFICIAL TEST SPLIT IS TOUCHED — exactly once per
    run, after training, with the val-selected best.pth already loaded.
    Writes eval/test_final.json (atomic, and LAST: it is the run's
    completion marker). Refuses to run twice."""
    marker = run_dir / "eval" / MARKER_NAME
    if marker.exists():
        raise RuntimeError(
            f"{marker} already exists — the test split may only be touched "
            f"once per run; refusing a second evaluation")

    test_ds = FTItemsDataset(data_root, test_items,
                             transform=build_eval_transform(cfg["img_size"]))
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["workers"],
                        pin_memory=device.type == "cuda")
    top1, top5, n = evaluate(model, loader, device)
    if n != len(test_ds):
        raise RuntimeError(f"evaluated {n} test images but the dataset has "
                           f"{len(test_ds)} — refusing to report")

    result = {
        "run_id": cfg["run_id"],
        "dataset": cfg["dataset"],
        "arch": cfg["arch"],
        "variant": cfg["variant"],
        "ft_seed": cfg["ft_seed"],
        "seed": cfg["ft_seed"],  # task-mandated field name; == ft_seed
        "top1": round(top1, 3),
        "top5": round(top5, 3),
        "n_images": n,
        "n_classes": extra["n_classes"],
        **extra,
        "git_sha": git_sha(),
    }
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, marker)
    return result


# ── Run lock (double-submission guard) ────────────────────────────────────────

def acquire_lock(run_dir: Path):
    """Refuse to start while another process is training the same run_dir —
    the legacy e2 repeats were lost to exactly this double-submission
    overwrite. The lock is heartbeat-touched every epoch; one older than
    LOCK_STALE_SECONDS belongs to a dead job. A stale lock is claimed by an
    ATOMIC rename to a unique name (two racers cannot both win: the loser's
    rename finds no source), then the real lock is created with O_EXCL.
    Returns (lock_path, ownership_token)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / LOCK_NAME
    msg = (f"{lock} exists — another process appears to be training this "
           f"run RIGHT NOW; refusing to start a second one (it would mix "
           f"checkpoints). If that job is truly dead, wait "
           f"{LOCK_STALE_SECONDS}s for the lock to go stale (or delete it) "
           f"and resubmit.")
    try:
        if lock.exists():
            if time.time() - lock.stat().st_mtime < LOCK_STALE_SECONDS:
                raise RuntimeError(msg)
            claimed = lock.with_name(
                f"{LOCK_NAME}.stale.{os.getpid()}.{time.time_ns()}")
            lock.rename(claimed)  # atomic: exactly one contender succeeds
            claimed.unlink()
    except FileNotFoundError:
        # the lock vanished mid-check: someone else just released or claimed
        # it — treat as contention, one clean resubmission costs nothing
        raise RuntimeError(msg)
    token = f"{socket.gethostname()} pid={os.getpid()} t={time.time_ns()}"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(msg)
    os.write(fd, token.encode())
    os.close(fd)
    return lock, token


def release_lock(lock: Path, token: str):
    """Unlink only a lock we still own — after a stale takeover by a
    successor, the zombie's release must not remove the successor's lock."""
    try:
        if lock.read_text() == token:
            lock.unlink()
    except (FileNotFoundError, OSError):
        pass


def marker_is_valid(marker: Path, run_id: str) -> bool:
    """A completion marker counts only if it parses and carries the keystone
    fields — a torn write on a crashed node must not pin the run as done
    forever (same contract as tools/check_done.py)."""
    try:
        with open(marker) as f:
            d = json.load(f)
        return d.get("run_id") == run_id and bool(d.get("finetuned_sha256"))
    except (json.JSONDecodeError, OSError):
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training(matrix, run_id, data_root=None, stage_base=None,
                 out_root="results/runs", max_epochs=None, device_str=None):
    cfg = resolve_run_config(matrix, run_id)
    cfg["epochs_configured"] = cfg["epochs"]
    if max_epochs is not None:
        cfg["epochs"] = max_epochs
    if cfg["epochs"] < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg['epochs']}")
    seed = cfg["ft_seed"]

    out_root = Path(out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root  # requeue guard must not depend on cwd
    run_dir = out_root / run_id
    ckpt_dir = run_dir / "ckpt"
    log_path = run_dir / "log.csv"
    marker = run_dir / "eval" / MARKER_NAME

    # requeue guard — a completed run is never redone or overwritten
    def _skip():
        print(f"{run_id}: {marker} exists — run already complete, skipping",
              flush=True)
        # dust: a SIGKILL between marker write and lock release can leave a
        # lock beside a completed run; clear it once it is stale
        lock_dust = run_dir / LOCK_NAME
        try:
            if (lock_dust.exists() and time.time() -
                    lock_dust.stat().st_mtime >= LOCK_STALE_SECONDS):
                lock_dust.unlink()
        except OSError:
            pass
        return "skipped"

    if marker.exists() and marker_is_valid(marker, run_id):
        return _skip()

    lock, token = acquire_lock(run_dir)
    try:
        # re-check under the lock: the marker may have landed between the
        # pre-lock check and the acquisition (a just-finishing duplicate) —
        # without this, a resubmit would wipe a completed run's files
        if marker.exists():
            if marker_is_valid(marker, run_id):
                return _skip()
            # torn marker from a crashed node: keep the bytes for forensics,
            # clear the completion signal, redo the run
            corrupt = marker.with_name(MARKER_NAME + ".corrupt")
            os.replace(marker, corrupt)
            print(f"{run_id}: {marker} exists but is INVALID (torn write?) "
                  f"— moved to {corrupt}, re-running", flush=True)
        return _run_training_locked(cfg, run_id, run_dir, ckpt_dir, log_path,
                                    out_root, data_root, stage_base,
                                    device_str, seed, lock)
    finally:
        # success, crash or Ctrl-C: release IF still ours; a SIGKILL leaves
        # the lock to go stale
        release_lock(lock, token)


def _run_training_locked(cfg, run_id, run_dir, ckpt_dir, log_path, out_root,
                         data_root, stage_base, device_str, seed, lock):

    device = (torch.device(device_str) if device_str
              else torch.device("cuda" if torch.cuda.is_available()
                                else "cpu"))
    amp_on = device.type == "cuda"

    # no resume machinery (runs are 1-3 h, far below any wall limit): every
    # start without the completion marker is a FRESH start; the previous
    # attempt's outputs are cleared LATER, only once the environment has
    # validated (split, backbone, sha) — a resubmit into a broken
    # environment must not destroy the crashed attempt's evidence
    create_run(out_root, run_id, cfg, seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # seed BEFORE model build (head init) and loader construction
    set_seed(seed)

    if data_root is None:
        if stage_base is None:
            raise ValueError("give --data_root (extracted dataset root) or "
                             "--stage_base (tar staging directory)")
        data_root = stage_dataset(cfg, stage_base)

    (train_items, val_items, test_items, n_classes,
     split_sha) = load_items(cfg, data_root)

    model, backbone_sha, ckpt_meta = load_backbone(cfg, device, n_classes)

    if len(train_items) < cfg["batch_size"]:
        raise ValueError(
            f"train split ({len(train_items)}) smaller than batch_size "
            f"({cfg['batch_size']}) — with drop_last=True nothing would "
            f"train; override batch_size for smoke/tests")

    # environment validated — NOW clear the previous (crashed) attempt so
    # nothing can mix attempts: stale log, checkpoints, atomic-write dust
    if log_path.exists():
        log_path.unlink()
    for stale in [ckpt_dir / "best.pth", ckpt_dir / "last.pth",
                  *ckpt_dir.glob("*.tmp"), *run_dir.glob(f"{LOCK_NAME}.stale.*")]:
        if stale.exists():
            stale.unlink()

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        FTItemsDataset(data_root, train_items,
                       transform=build_train_transform(cfg["img_size"])),
        batch_size=cfg["batch_size"], shuffle=True, generator=gen,
        num_workers=cfg["workers"], worker_init_fn=_worker_init,
        pin_memory=amp_on, drop_last=True)
    val_loader = DataLoader(
        FTItemsDataset(data_root, val_items,
                       transform=build_eval_transform(cfg["img_size"])),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["workers"], pin_memory=amp_on)

    # legacy e6 optimizer/schedule, verbatim: AdamW two param groups,
    # cosine per-epoch
    head_params = list(model.head.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg["backbone_lr"]},
         {"params": head_params, "lr": cfg["head_lr"]}],
        weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["eta_min"])
    scaler = torch.amp.GradScaler("cuda" if amp_on else "cpu",
                                  enabled=amp_on)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # record provenance the moment everything is verified
    meta_path = run_dir / "meta.json"
    meta = json.load(open(meta_path))
    meta.update({
        "ft_seed": seed,
        "backbone_run": cfg["backbone"].get("run"),
        "backbone_ckpt": cfg["backbone"]["ckpt"],
        "backbone_sha256": backbone_sha,
        "backbone_ckpt_meta": ckpt_meta,
        "val_split": cfg["val_split"],
        "val_split_sha256": split_sha,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(test_items),
        "n_classes": n_classes,
        "resume": "none (runs are 1-3 h; interrupted runs restart fresh)",
        "determinism": "seeded (ft_seed); GPU kernels not forced "
                       "deterministic — same policy as the e2r trainer",
    })
    atomic_json_dump(meta, meta_path)

    print(f"\n{'=' * 60}\n  {run_id}\n  backbone {cfg['backbone']['ckpt']} "
          f"(sha256 {backbone_sha[:12]}...)\n  train {len(train_items)}  "
          f"val {len(val_items)}  test {len(test_items)}  "
          f"classes {n_classes}\n{'=' * 60}\n", flush=True)

    best_top1, best_epoch = 0.0, -1
    for epoch in range(cfg["epochs"]):
        lock.touch()  # heartbeat: keeps the double-submission guard fresh
        t0 = time.time()
        lr_epoch = optimizer.param_groups[0]["lr"]  # the lr this epoch
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, scaler, device, amp_on,
                                     cfg["grad_clip"])
        scheduler.step()

        # B7 fix: validate EVERY epoch, on the held-out val split only
        val_top1, val_top5, n_val = evaluate(model, val_loader, device)
        if n_val != len(val_items):
            raise RuntimeError(f"validated {n_val} != {len(val_items)} "
                               f"val images")

        append_log_row(log_path, {
            "epoch": epoch,
            "lr": lr_epoch,
            "train_loss": round(train_loss, 4),
            "val_top1": round(val_top1, 3),
        })
        print(f"  [{epoch + 1:3d}/{cfg['epochs']}]  loss={train_loss:.4f}  "
              f"val={val_top1:.2f}%  t={time.time() - t0:.0f}s", flush=True)

        if val_top1 > best_top1:
            best_top1, best_epoch = val_top1, epoch
            atomic_torch_save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_top1": val_top1,
                "ft_seed": seed,
                "backbone_sha256": backbone_sha,
            }, ckpt_dir / "best.pth")
            print(f"  *** new best (val): {val_top1:.2f}%", flush=True)

        # atomic last.pth every epoch (post-mortem state; not for resume)
        atomic_torch_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
            "best_epoch": best_epoch,
            "val_top1": val_top1,
            "ft_seed": seed,
        }, ckpt_dir / "last.pth")

    # select on val: reload the val-best weights, then touch the test split
    # exactly once
    if best_epoch < 0:
        raise RuntimeError(
            "no epoch ever improved on val_top1=0.0 — best.pth was never "
            "written; refusing to evaluate the test split")
    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location="cpu",
                           weights_only=True)
    model.load_state_dict(best_ckpt["model"])
    model.to(device)

    meta = json.load(open(meta_path))
    meta["best_epoch"] = best_epoch
    meta["val_top1_at_best"] = round(best_top1, 3)
    atomic_json_dump(meta, meta_path)
    finalize_run(run_dir)

    result = final_test_eval(
        model, cfg, run_dir, test_items, data_root, device,
        extra={
            "n_classes": n_classes,
            "epochs_trained": cfg["epochs"],
            "epochs_configured": cfg["epochs_configured"],
            # a run whose epoch count was overridden (smoke) is flagged so
            # its test number can never be mistaken for a result
            "smoke": cfg["epochs"] != cfg["epochs_configured"],
            "best_epoch": best_epoch,
            "val_top1_at_best": round(best_top1, 3),
            "backbone_run": cfg["backbone"].get("run"),
            "backbone_ckpt": cfg["backbone"]["ckpt"],
            "backbone_sha256": backbone_sha,
            "finetuned_ckpt": "ckpt/best.pth",
            "finetuned_sha256": file_sha256(ckpt_dir / "best.pth"),
            "val_split": cfg["val_split"],
            "val_split_sha256": split_sha,
        })

    print(f"\nDone — {run_id}\n  best val top-1 {best_top1:.2f}% at epoch "
          f"{best_epoch}\n  TEST top-1 {result['top1']:.3f}%  "
          f"(n={result['n_images']})\n  -> {run_dir / 'eval' / MARKER_NAME}",
          flush=True)
    return "completed"


def main():
    parser = argparse.ArgumentParser("SAGA fine-grained clean fine-tune "
                                     "(TASK-08)")
    parser.add_argument("--matrix", required=True,
                        help="configs/ft_matrix.yaml")
    parser.add_argument("--run", required=True, help="run_id in the matrix")
    parser.add_argument("--data_root", default=None,
                        help="already-extracted dataset root (overrides "
                             "staging)")
    parser.add_argument("--stage_base", default=None,
                        help="directory to extract the dataset tar into "
                             "(e.g. a /scratch job dir)")
    parser.add_argument("--out_root", default="results/runs")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="smoke runs only — overrides the matrix epochs")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_training(args.matrix, args.run, data_root=args.data_root,
                 stage_base=args.stage_base, out_root=args.out_root,
                 max_epochs=args.max_epochs, device_str=args.device)


if __name__ == "__main__":
    main()
