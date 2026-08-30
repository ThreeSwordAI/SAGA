#!/usr/bin/env python3
"""
classification/tools/train.py
==============================
E2R trainer (TASK-05). Successor of the legacy E2 trainer — every
MATHEMATICAL setting is kept legacy-identical so the new runs pool with the
9 valid legacy runs as "same training distribution, different seed":

- model construction: build_model() is the legacy function verbatim
  (baseline/SAGA via saga.build_saga_vit, registers via timm reg_tokens);
  new knobs (gate_init_logit, wd_phi_zero, drop_path_rate) DEFAULT to the
  legacy values (0.0 / false / 0.0) and are no-ops at those defaults.
- data: timm create_dataset/create_loader with the legacy arguments
  (same augmentation flags, prefetcher on GPU, bilinear eval pipeline).
- loss/optimizer/LR: identical criteria, create_optimizer_v2 call, cosine
  schedule with per-iteration step_update (t_in_epochs=False). The legacy
  per-epoch `scheduler.step(epoch+1)` call was verified to be a VALUE NO-OP
  under t_in_epochs=False and is removed (M9) — one stepping path remains.

What changed (TASK-05 A1):
- seeding (M4): set_seed() for python/numpy/torch/cuda; loader workers
  seeded via the torch global seed + timm worker_seeding='all' (timm's
  create_loader has no generator parameter; the DataLoader base seed is
  drawn from the torch RNG seeded here); numpy is reseeded per rank
  (seed + rank) so Mixup draws are deterministic per rank.
- validation (B1): exact full-val — manual rank::world_size sharding (no
  padded sampler), float64 count accumulation, all_reduce; best.pth is
  selected on the reduced val_top1_full.
- results contract: results/runs/<run_id>/ with meta.json +
  config.resolved.yaml (saga/run_registry), append-safe log.csv,
  ckpt/{last,best}.pth, gates/phi_e###.npz, diag/diag_e###.json,
  grads/grad_phi.csv.
- checkpointing: ckpt/last.pth EVERY epoch, atomic (tmp + os.replace),
  carrying model/optimizer/scheduler/scaler/epoch/best and the RNG states
  of every rank plus the sampler epoch.
- resume: `--resume auto` restores everything and continues appending to
  log.csv (rows >= the resume epoch are dropped once, atomically); the
  same launch command fresh-starts and resumes.

Launch (one command, fresh start and every resubmission):
    torchrun --nproc_per_node=4 classification/tools/train.py \
        --matrix configs/e2r_matrix.yaml --run e2r_vits_nomix_saga_s1 \
        --data_root $STAGE_DIR --resume auto
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from timm.data import create_dataset, create_loader, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter

import yaml

# ── SAGA ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from saga import build_saga_vit
from saga.run_registry import create_run, finalize_run

LOG_FIELDS = ["epoch", "lr", "train_loss", "val_top1_full", "val_top5_full",
              "val_loss", "img_per_sec", "wall_time"]


# ── Config helpers (legacy deep-merge semantics) ──────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_run_config(matrix_path: str, run_id: str) -> dict:
    """Resolve one run of the e2r matrix onto the LEGACY config files.

    recipe 'mixup' = classification/configs/base.yaml as-is;
    recipe 'nomix' = base.yaml + base_nomix.yaml overrides — exactly the
    two configs the legacy runs trained with."""
    matrix = load_yaml(matrix_path)
    if run_id not in matrix["runs"]:
        raise KeyError(f"run {run_id!r} not in {matrix_path}")
    run = matrix["runs"][run_id]

    cfg_dir = Path(matrix_path).parent / matrix.get(
        "config_dir", "../classification/configs")
    cfg = load_yaml(cfg_dir / "base.yaml")
    if run["recipe"] == "nomix":
        nomix = load_yaml(cfg_dir / "base_nomix.yaml")
        nomix.pop("_base_", None)
        cfg = deep_merge(cfg, nomix)
    elif run["recipe"] != "mixup":
        raise ValueError(f"unknown recipe {run['recipe']!r}")

    variant = run["variant"]
    cfg["model"] = deep_merge(cfg["model"], {
        "arch": run["arch"],
        "gate": variant == "saga",
        "registers": 4 if variant == "registers" else 0,
    })

    if "seed" not in run:
        raise KeyError(f"run {run_id!r} has no seed — seed is required")

    cfg["run_id"] = run_id
    cfg["variant"] = variant
    cfg["recipe"] = run["recipe"]
    cfg["seed"] = int(run["seed"])

    # new knobs — DEFAULT to the legacy values (no-ops); resolved values
    # land in config.resolved.yaml and meta.json
    defaults = matrix.get("defaults", {})
    cfg["knobs"] = {
        "gate_init_logit": float(run.get(
            "gate_init_logit", defaults.get("gate_init_logit", 0.0))),
        "wd_phi_zero": bool(run.get(
            "wd_phi_zero", defaults.get("wd_phi_zero", False))),
        "drop_path_rate": float(run.get(
            "drop_path_rate", defaults.get("drop_path_rate", 0.0))),
    }
    cfg["instrumentation"] = {
        "log_grad_phi": bool(run.get("log_grad_phi", False)),
        "diag_freq": int(defaults.get("diag_freq", 10)),
        "diag_split": defaults.get(
            "diag_split", "results/diagsplit/val_diag_split.json"),
        "diag_n_effrank": int(defaults.get("diag_n_effrank", 10000)),
        "grad_log_every": int(defaults.get("grad_log_every", 100)),
        "grad_log_epochs": int(defaults.get("grad_log_epochs", 31)),
    }
    # per-run overrides (used by tests / smoke runs; empty in production)
    cfg = deep_merge(cfg, run.get("overrides", {}))
    return cfg


# ── Seeding (M4) ──────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_rng_state() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


# ── Model builder (LEGACY verbatim) + default-off knobs ──────────────────────

def build_model(cfg: dict) -> nn.Module:
    m = cfg['model']
    arch = m['arch']
    use_gate = m.get('gate', False)
    n_reg = m.get('registers', 0)
    img_size = cfg['model'].get('img_size', 224)
    patch_size = cfg['model'].get('patch_size', 16)
    n_classes = cfg['model'].get('num_classes', 1000)

    if n_reg > 0 and use_gate:
        raise ValueError("Cannot use both registers and SAGA gate in the same variant.")

    if n_reg > 0:
        import timm
        model = timm.create_model(
            arch,
            pretrained=False,
            num_classes=n_classes,
            img_size=img_size,
            reg_tokens=n_reg,
        )
    else:
        model = build_saga_vit(
            arch=arch,
            gate=use_gate,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            pretrained=False,
        )

    return model


def apply_knobs(model: nn.Module, knobs: dict):
    """All knobs default to legacy values, where this function is a NO-OP."""
    logit = knobs.get("gate_init_logit", 0.0)
    if logit != 0.0:
        with torch.no_grad():
            for blk in model.blocks:
                gate = getattr(blk.attn, "gate", None)
                if gate is not None:
                    gate.phi.fill_(logit)

    dpr = knobs.get("drop_path_rate", 0.0)
    if dpr > 0.0:
        # timm's own linear ramp, applied post-hoc (saga/vit.py untouched)
        from timm.layers import DropPath
        depth = len(model.blocks)
        for i, blk in enumerate(model.blocks):
            p = dpr * i / max(depth - 1, 1)
            if p > 0:
                blk.drop_path1 = DropPath(p)
                blk.drop_path2 = DropPath(p)


def optimizer_parameters(model: nn.Module, knobs: dict):
    """Legacy default: one group, model.parameters(), WD on phi like
    everything else. wd_phi_zero=true excludes phi from weight decay."""
    if not knobs.get("wd_phi_zero", False):
        return model.parameters()
    phi, rest = [], []
    for name, p in model.named_parameters():
        (phi if name.endswith(".gate.phi") else rest).append(p)
    return [{"params": rest},
            {"params": phi, "weight_decay": 0.0}]


# ── Atomic writes ─────────────────────────────────────────────────────────────

def atomic_torch_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def atomic_npz_save(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ── log.csv (append-safe) ─────────────────────────────────────────────────────

def append_log_row(log_path: Path, row: dict):
    new = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def sanitize_log(log_path: Path, start_epoch: int):
    """Resume-time cleanup: drop rows with epoch >= start_epoch (a crash
    between log-append and checkpoint-save leaves one such row) — atomic."""
    if not log_path.exists():
        return
    with open(log_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start_epoch]
    tmp = log_path.with_name(log_path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, log_path)


# ── Exact full-val (B1) ───────────────────────────────────────────────────────

def build_val_loader(data_root, cfg, rank, world_size):
    """Exact-val pipeline: same image processing values as the legacy timm
    eval path (verified in TASK-01: Resize(bilinear)+CenterCrop+Normalize),
    manual rank::world_size sharding so every image is counted exactly once
    (timm's padded distributed sampler would duplicate samples)."""
    from tools.eval import build_val_transform
    dataset = create_dataset('imagefolder', root=data_root, split='val',
                             is_training=False)
    dataset.transform = build_val_transform(cfg['data']['input_size'])
    shard = list(range(len(dataset)))[rank::world_size]
    per_gpu = cfg['train']['batch_size'] // world_size
    loader = DataLoader(Subset(dataset, shard), batch_size=per_gpu,
                        shuffle=False, num_workers=cfg['data']['num_workers'],
                        pin_memory=cfg['data'].get('pin_memory', True)
                        and torch.cuda.is_available())
    return loader, len(dataset)


@torch.no_grad()
def validate_full(model, loader, device, n_total, amp_on, distributed):
    model.eval()
    counts = torch.zeros(4, dtype=torch.float64, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            output = model(images)
        logits = output.float()
        loss = nn.functional.cross_entropy(logits, targets,
                                           reduction="none").double().sum()
        _, pred = logits.topk(min(5, logits.shape[1]), dim=1)
        correct = pred.eq(targets.view(-1, 1))
        counts[0] += correct[:, :1].sum().double()
        counts[1] += correct.sum().double()
        counts[2] += loss
        counts[3] += targets.shape[0]

    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    c1, c5, loss_sum, n = counts.tolist()
    assert int(n) == n_total, (
        f"validated {int(n)} images but dataset has {n_total} — sharding "
        f"broken, refusing to report")
    return {"val_top1_full": round(100.0 * c1 / n, 3),
            "val_top5_full": round(100.0 * c5 / n, 3),
            "val_loss": round(loss_sum / n, 4)}


# ── Checkpointing (every epoch, atomic, RNG-complete) ────────────────────────

def gather_rng_states(distributed):
    mine = collect_rng_state()
    if not distributed:
        return [mine]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, mine)
    return states


def save_checkpoint(path, model_sd, optimizer, scheduler, scaler, epoch,
                    best_top1, top1, rng_states, rank,
                    steps_per_epoch, total_epochs):
    if rank != 0:
        return
    atomic_torch_save({
        "epoch": epoch,
        "sampler_epoch": epoch,
        "model": model_sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_top1": best_top1,
        "top1": top1,
        "rng_states": rng_states,
        # schedule geometry — checked at resume so a changed staged dataset
        # or a stray --max_epochs can never silently shift the LR curve
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": total_epochs,
    }, path)


# ── Instrumentation ───────────────────────────────────────────────────────────

def dump_phi(model, epoch, gates_dir: Path):
    blocks = getattr(model, "blocks", None)
    phis = []
    for blk in blocks:
        gate = getattr(blk.attn, "gate", None)
        if gate is None:
            return
        phis.append(gate.phi.detach().float().cpu().numpy())
    gates_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz_save(gates_dir / f"phi_e{epoch:03d}.npz", phi=np.stack(phis))


def run_diag(model, cfg, data_root, epoch, diag_dir: Path, device):
    """Norms-only diagnostic set on the frozen diag split (rank 0 only)."""
    from saga.metrics import compute_diagnostics
    from tools.build_diag_split import DiagSplitDataset
    from tools.eval import build_val_transform

    inst = cfg["instrumentation"]
    split = Path(inst["diag_split"])
    if not split.exists():
        print(f"  diag skipped: {split} not found", flush=True)
        return
    ds = DiagSplitDataset(data_root, split,
                          transform=build_val_transform(
                              cfg['data']['input_size']))
    per_gpu = cfg['train']['batch_size'] // max(
        int(os.environ.get("WORLD_SIZE", 1)), 1)
    loader = DataLoader(ds, batch_size=per_gpu, shuffle=False,
                        num_workers=cfg['data']['num_workers'],
                        pin_memory=torch.cuda.is_available())
    out = compute_diagnostics(model, loader, device, with_attn=False,
                              fixed_thr=None,
                              n_effrank=inst["diag_n_effrank"])
    out["epoch"] = epoch
    diag_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(out, diag_dir / f"diag_e{epoch:03d}.json")
    model.train()


class GradPhiLogger:
    """Per-layer ||dL/dphi|| every N iters for the first K epochs.
    Values are read AFTER scaler.unscale_ (the legacy grad-clip path always
    unscales), so they are true gradient norms. Append-safe CSV."""

    def __init__(self, path: Path, every: int, max_epoch: int):
        self.path, self.every, self.max_epoch = path, every, max_epoch
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(["epoch", "iter", "layer",
                                        "grad_phi_norm"])

    def maybe_log(self, model, epoch, step):
        if epoch >= self.max_epoch or step % self.every != 0:
            return
        rows = []
        for i, blk in enumerate(model.blocks):
            gate = getattr(blk.attn, "gate", None)
            if gate is None or gate.phi.grad is None:
                continue
            rows.append([epoch, step, i,
                         float(gate.phi.grad.detach().norm().item())])
        if rows:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerows(rows)
                f.flush()


# ── Training (legacy loop math; M9 double-step removed) ──────────────────────

def train_one_epoch(model, loader, optimizer, criterion, mixup_fn,
                    scaler, scheduler, epoch, cfg, device, rank,
                    grad_logger=None, model_unwrapped=None):
    model.train()
    loss_meter = AverageMeter()
    log_freq = cfg.get('logging', {}).get('log_freq', 100)
    amp_on = cfg['train']['amp'] and device.type == "cuda"
    n_images = 0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp_on):
            output = model(images)
            loss = criterion(output, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        clip = cfg['train'].get('grad_clip', 1.0)
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        if grad_logger is not None:
            grad_logger.maybe_log(model_unwrapped, epoch, step)
        scaler.step(optimizer)
        scaler.update()

        # the ONLY scheduler stepping path (per-iteration, t_in_epochs=False;
        # the legacy per-epoch scheduler.step(epoch+1) was a value no-op
        # and is removed — M9)
        steps_per_epoch = len(loader)
        scheduler.step_update(epoch * steps_per_epoch + step)
        loss_meter.update(loss.item(), images.size(0))
        n_images += images.size(0)

        if rank == 0 and step % log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [{epoch}][{step}/{len(loader)}] "
                  f"loss={loss_meter.avg:.4f}  lr={lr:.2e}", flush=True)

    return loss_meter.avg, n_images


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training(matrix_path, run_id, data_root, out_root="results/runs",
                 resume="auto", max_epochs=None, device_str=None):
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

    cfg = resolve_run_config(matrix_path, run_id)
    if max_epochs:
        cfg['train']['epochs'] = max_epochs
    seed = cfg["seed"]

    # M4: deterministic construction — identical on every rank
    set_seed(seed)

    run_dir = Path(out_root) / run_id
    ckpt_dir = run_dir / "ckpt"
    last_path = ckpt_dir / "last.pth"
    log_path = run_dir / "log.csv"
    resuming = resume == "auto" and last_path.exists()

    if rank == 0:
        if not resuming:
            create_run(out_root, run_id, cfg, seed)
            # no checkpoint => nothing here is trustworthy: a leftover
            # log.csv (e.g. a crash before the very first ckpt landed)
            # would otherwise collect duplicate epoch rows
            sanitize_log(log_path, 0)
        else:
            # keep the original meta; record the resubmission
            meta_path = run_dir / "meta.json"
            meta = json.load(open(meta_path))
            meta.setdefault("resumes", []).append(
                {"cmd": " ".join(sys.argv),
                 "world_size": world_size})
            atomic_json_dump(meta, meta_path)
        meta_path = run_dir / "meta.json"
        meta = json.load(open(meta_path))
        meta["knobs"] = cfg["knobs"]        # resolved knob values (TASK-05)
        atomic_json_dump(meta, meta_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # ── model (legacy construction; knobs are no-ops at defaults) ──────────
    model = build_model(cfg)
    apply_knobs(model, cfg["knobs"])
    model = model.to(device)
    is_saga = cfg['model'].get('gate', False)

    if rank == 0:
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n{'=' * 60}\n  {run_id}\n  Total params: {n_total:.1f}M  |  "
              f"GPUs: {world_size}  |  Batch: {cfg['train']['batch_size']}"
              f"\n{'=' * 60}\n", flush=True)

    # Legacy runs were UNSEEDED: every rank's augmentation/dropout streams
    # were independent. Reproduce that independence deterministically —
    # per-rank reseed AFTER the (identical) model construction, BEFORE any
    # loader is built (the DataLoader base seed derives from the torch RNG).
    torch.manual_seed(seed * 1000 + rank)
    torch.cuda.manual_seed_all(seed * 1000 + rank)
    random.seed(seed * 1000 + rank)
    np.random.seed(seed + rank)     # per-rank Mixup draws (numpy, main proc)

    # ── data (legacy timm pipeline for training) ──────────────────────────
    per_gpu = cfg['train']['batch_size'] // world_size
    aug = cfg.get('augmentation', {})

    train_ds = create_dataset('imagefolder', root=data_root,
                              split='train', is_training=True)

    mixup_active = (aug.get('mixup_alpha', 0) > 0 or
                    aug.get('cutmix_alpha', 0) > 0)
    mixup_fn = Mixup(
        mixup_alpha=aug.get('mixup_alpha', 0.8),
        cutmix_alpha=aug.get('cutmix_alpha', 1.0),
        num_classes=cfg['model']['num_classes'],
        label_smoothing=cfg['train'].get('label_smoothing', 0.1),
    ) if mixup_active else None

    auto_aug = None
    if aug.get('rand_aug', True):
        m = aug.get('rand_aug_magnitude', 9)
        n = aug.get('rand_aug_layers', 2)
        auto_aug = f'rand-m{m}-n{n}-mstd0.5'

    train_loader = create_loader(
        train_ds,
        input_size=(3, cfg['data']['input_size'], cfg['data']['input_size']),
        batch_size=per_gpu,
        is_training=True,
        re_prob=aug.get('random_erase_prob', 0.25),
        auto_augment=auto_aug,
        num_workers=cfg['data']['num_workers'],
        distributed=distributed,
        pin_memory=cfg['data'].get('pin_memory', True),
        use_prefetcher=torch.cuda.is_available(),   # legacy default on GPU
        # timm default (True) whenever workers exist — matches legacy runs;
        # torch forbids it at num_workers=0 (CPU tests)
        persistent_workers=cfg['data']['num_workers'] > 0,
    )
    val_loader, n_val = build_val_loader(data_root, cfg, rank, world_size)

    # ── loss / optimizer / scheduler (legacy-identical) ───────────────────
    train_criterion = (SoftTargetCrossEntropy() if mixup_active
                       else LabelSmoothingCrossEntropy(
                           smoothing=cfg['train'].get('label_smoothing', 0.1)))

    tr = cfg['train']
    lr = tr['lr'] * tr['batch_size'] / 1024 if tr.get('lr_scaling', True) else tr['lr']

    optimizer = create_optimizer_v2(
        optimizer_parameters(model, cfg["knobs"]),
        opt=tr['optimizer'],
        lr=lr,
        weight_decay=tr['weight_decay'],
        betas=tuple(tr.get('betas', [0.9, 0.999])),
    )

    total_epochs = tr['epochs']
    steps_per_epoch = len(train_loader)
    warmup_epochs = tr.get('warmup_epochs', 20)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_epochs * steps_per_epoch,
        lr_min=tr.get('min_lr', 1e-6),
        warmup_t=warmup_epochs * steps_per_epoch,
        warmup_lr_init=1e-6,
        cycle_limit=1,
        t_in_epochs=False,
    )

    amp_on = tr['amp'] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu",
                                  enabled=amp_on)

    # ── resume (auto) ──────────────────────────────────────────────────────
    start_epoch, best_top1 = 0, 0.0
    if resuming:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        saved_spe = ckpt.get("steps_per_epoch")
        if saved_spe is not None and saved_spe != steps_per_epoch:
            raise RuntimeError(
                f"steps_per_epoch changed across resubmissions: checkpoint "
                f"has {saved_spe}, current staged dataset gives "
                f"{steps_per_epoch}. The staged ImageNet copy differs — "
                f"resuming would silently shift the LR schedule. Fix the "
                f"staging (re-run the job) instead of resuming this state.")
        saved_te = ckpt.get("total_epochs")
        if saved_te is not None and saved_te != total_epochs:
            # --max_epochs at resume redefines the schedule (dev/test only;
            # the production chain never passes it): keep the freshly built
            # scheduler — its geometry is authoritative, position comes from
            # the absolute step_update counter
            print(f"  WARNING: total_epochs changed {saved_te} -> "
                  f"{total_epochs}; scheduler rebuilt for the new schedule "
                  f"(dev/test path — production resubmits never do this)",
                  flush=True)
        else:
            scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        rng_states = ckpt.get("rng_states", [])
        if rank < len(rng_states):
            restore_rng_state(rng_states[rank])
        else:
            print(f"  WARNING: no saved RNG state for rank {rank} "
                  f"(world size changed?) — reseeding", flush=True)
            set_seed(seed + 100_000 + start_epoch + rank)
        if rank == 0:
            sanitize_log(log_path, start_epoch)
            print(f"Resumed at epoch {start_epoch}  "
                  f"best_top1={best_top1:.2f}%", flush=True)
    if distributed:
        dist.barrier()

    model_unwrapped = model
    if distributed:
        model = DDP(model, device_ids=[local_rank]
                    if device.type == "cuda" else None)

    grad_logger = None
    if cfg["instrumentation"]["log_grad_phi"] and is_saga and rank == 0:
        grad_logger = GradPhiLogger(
            run_dir / "grads" / "grad_phi.csv",
            every=cfg["instrumentation"]["grad_log_every"],
            max_epoch=cfg["instrumentation"]["grad_log_epochs"])

    diag_freq = cfg["instrumentation"]["diag_freq"]

    # ── training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, n_images = train_one_epoch(
            model, train_loader, optimizer, train_criterion, mixup_fn,
            scaler, scheduler, epoch, cfg, device, rank,
            grad_logger=grad_logger, model_unwrapped=model_unwrapped)
        if distributed:
            dist.barrier()
        epoch_seconds = time.time() - t0

        val_metrics = validate_full(model, val_loader, device, n_val,
                                    amp_on, distributed)
        model.train()

        top1 = val_metrics["val_top1_full"]
        rng_states = gather_rng_states(distributed)

        if rank == 0:
            # log row FIRST, checkpoint LAST (a crash in between duplicates
            # the last row, which sanitize_log drops on the next resume)
            append_log_row(log_path, {
                "epoch": epoch,
                "lr": round(optimizer.param_groups[0]['lr'], 8),
                "train_loss": round(train_loss, 4),
                **{k: val_metrics[k] for k in
                   ("val_top1_full", "val_top5_full", "val_loss")},
                "img_per_sec": round(n_images * world_size / epoch_seconds, 1),
                "wall_time": round(epoch_seconds, 1),
            })
            print(f"[{epoch + 1}/{total_epochs}]  top1={top1:.2f}%  "
                  f"top5={val_metrics['val_top5_full']:.2f}%  "
                  f"loss={train_loss:.4f}", flush=True)

            if is_saga:
                dump_phi(model_unwrapped, epoch, run_dir / "gates")

            model_sd = model_unwrapped.state_dict()
            if top1 > best_top1:
                best_top1 = top1
                save_checkpoint(ckpt_dir / "best.pth", model_sd, optimizer,
                                scheduler, scaler, epoch, best_top1, top1,
                                rng_states, rank, steps_per_epoch,
                                total_epochs)
                print(f"  *** New best: {top1:.2f}%", flush=True)
            # last.pth EVERY epoch — at most one epoch of work is ever lost
            save_checkpoint(last_path, model_sd, optimizer, scheduler,
                            scaler, epoch, best_top1, top1, rng_states,
                            rank, steps_per_epoch, total_epochs)

            if (epoch + 1) % diag_freq == 0:
                print(f"  running diag at epoch {epoch}", flush=True)
                run_diag(model_unwrapped, cfg, data_root, epoch,
                         run_dir / "diag", device)
        if distributed:
            dist.barrier()

    if rank == 0:
        finalize_run(run_dir)
        print(f"\nDone — {run_id}  best_top1={best_top1:.2f}%", flush=True)
    if distributed:
        dist.destroy_process_group()
    return run_dir


def main():
    parser = argparse.ArgumentParser("SAGA e2r trainer")
    parser.add_argument("--matrix", required=True,
                        help="configs/e2r_matrix.yaml")
    parser.add_argument("--run", required=True, help="run_id in the matrix")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", default="results/runs")
    parser.add_argument("--resume", choices=["auto", "none"], default="auto",
                        help="auto: load ckpt/last.pth if present (the same "
                             "command fresh-starts and resumes)")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_training(args.matrix, args.run, args.data_root,
                 out_root=args.out_root, resume=args.resume,
                 max_epochs=args.max_epochs, device_str=args.device)


if __name__ == '__main__':
    main()
