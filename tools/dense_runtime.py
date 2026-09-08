#!/usr/bin/env python3
"""
tools/dense_runtime.py
======================
TASK-09: shared plumbing for the two dense-prediction trainers
(detection/tools/train.py, segmentation/tools/train.py).

Everything here is the pattern the e2r trainer (classification/tools/train.py,
TASK-05) established and production has already validated:

  - config resolution: one run of configs/dense_matrix.yaml resolved ONTO the
    committed task config (detection/configs/base.yaml or
    segmentation/configs/base.yaml), so no hyperparameter is ever re-typed;
  - seeding + RNG capture/restore across resubmissions;
  - append-safe log.csv with resume-time row sanitisation;
  - atomic writes everywhere (tmp + fsync + os.replace), completion marker
    LAST (meta.json's end_time, written by run_registry.finalize_run);
  - `--resume auto`: the same launch command fresh-starts and resumes, with
    schedule-geometry guards so a changed staged dataset can never silently
    shift the LR curve;
  - a fast-path "already complete" check so a surplus job in a SLURM
    dependency chain exits without touching a GPU.

Nothing in here is task-specific: detection and segmentation differ only in
their eval/artifact code.
"""

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

META_NAME = "meta.json"
TASKS = ("detection", "segmentation")
VARIANTS = ("baseline", "saga", "registers")


# ── yaml / merge ──────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def repo_path(p) -> Path:
    """Interpret a config path as repo-relative unless it is absolute.

    A POSIX absolute path ("/home/vault/...") is NOT `is_absolute()` on
    Windows, so a naive check would silently rebase an HPC path under the
    repo root — and then a same-named file that happened to exist there
    would be loaded instead of failing. Treat a leading separator as
    absolute on every platform.
    """
    s = str(p)
    if s.startswith("/") or s.startswith("\\"):
        return Path(s)
    q = Path(s)
    return q if q.is_absolute() else REPO_ROOT / q


# ── config resolution ─────────────────────────────────────────────────────────

def resolve_dense_config(matrix, run_id: str) -> dict:
    """Resolve one run of configs/dense_matrix.yaml.

    `matrix` is a path or an already-loaded dict (tests). The returned config
    is the committed task base config with ONLY the variant switches merged
    in, plus bookkeeping keys (run_id/task/variant/seed/backbone_spec) and
    any explicit per-run `overrides` (empty in production).
    """
    if not isinstance(matrix, dict):
        matrix = load_yaml(matrix)
    if run_id not in matrix.get("runs", {}):
        raise KeyError(f"run {run_id!r} not in the dense matrix")
    run = dict(matrix["runs"][run_id])

    for key in ("task", "arch", "variant", "backbone", "seed"):
        if key not in run:
            raise KeyError(f"run {run_id!r} is missing required key {key!r}")
    task, variant = run["task"], run["variant"]
    if task not in TASKS:
        raise ValueError(f"run {run_id!r}: task must be one of {TASKS}")
    if variant not in VARIANTS:
        raise ValueError(f"run {run_id!r}: variant must be one of {VARIANTS}")

    base_rel = matrix["task_configs"][task]
    cfg = load_yaml(repo_path(base_rel))
    cfg = deep_merge(cfg, matrix.get("common_overrides", {}))

    cfg["model"] = deep_merge(cfg["model"], {
        "arch": run["arch"],
        "gate": variant == "saga",
        "registers": int(matrix.get("n_register_tokens", 4))
        if variant == "registers" else 0,
    })

    cfg.update({
        "run_id": run_id,
        "task": task,
        "variant": variant,
        "seed": int(run["seed"]),
        "backbone_key": run["backbone"],
        "backbone_spec": dict(matrix["backbones"][run["backbone"]]),
        "task_config_file": str(base_rel),
        "out_root": matrix["out_roots"][task],
    })
    # per-run overrides (tests / smokes; empty in production)
    cfg = deep_merge(cfg, run.get("overrides", {}))
    return cfg


def resolve_backbone(spec: dict) -> dict:
    """Pick the backbone checkpoint to use, honouring an optional `fallback`.

    TASK-09 mandates: the seeded e2r ViT-B registers run IF **finished** by
    launch time, else the legacy ViT-B registers checkpoint — and meta.json
    must record which. "Finished" is the operative word: the e2r trainer
    rewrites `ckpt/last.pth` at the END OF EVERY EPOCH, so mere file
    existence would let an in-flight run (epoch 0 weights!) silently become
    the dense backbone. A candidate is therefore usable only when ALL of:
      * its checkpoint file exists;
      * if it declares `require_complete`, THAT run carries a completion
        marker (meta.json end_time + a full log.csv) — i.e. it finished;
      * it has a pinned `sha256`, so the hash guard that protects the
        baseline/saga backbones is never inert. Pinning the hash of a
        just-finished run is one deliberate human edit (copy it out of that
        run's own eval JSON), which is exactly the point: three GPU-days
        must not start against an unverified checkpoint.
    Every rejection prints its reason, so the log says why the fallback won.

    Returns {'run', 'ckpt', 'sha256', 'source', 'note'}.
    """
    candidates = [("primary", spec)]
    if spec.get("fallback"):
        candidates.append(("fallback", spec["fallback"]))

    tried = []
    for source, cand in candidates:
        if not cand.get("ckpt"):
            raise KeyError(f"backbone {source} entry has no 'ckpt' key")
        path = repo_path(cand["ckpt"])

        if not path.exists():
            tried.append(f"{source}: {path} — file does not exist")
            continue
        if not cand.get("sha256"):
            tried.append(
                f"{source}: {path} — exists but the matrix pins no sha256, "
                f"so the checkpoint cannot be verified (pin it from that "
                f"run's own eval JSON to use this backbone)")
            continue
        gate = cand.get("require_complete")
        if gate:
            gate_dir = repo_path(gate)
            need = int(cand.get("require_epochs", 0))
            if not run_is_complete(gate_dir, need):
                tried.append(
                    f"{source}: {path} — exists, but {gate_dir} is not a "
                    f"COMPLETED run ({need} epochs + meta.end_time); its "
                    f"last.pth is rewritten every epoch, so it would be a "
                    f"mid-training checkpoint")
                continue

        out = {
            "run": cand.get("run"),
            "ckpt": str(path),
            "ckpt_config_value": cand["ckpt"],
            "sha256": cand["sha256"],
            "source": source,
            "note": cand.get("note"),
        }
        if tried:
            print("  backbone selection skipped:\n    "
                  + "\n    ".join(tried), flush=True)
        print(f"  backbone [{source}]: {path}", flush=True)
        return out

    raise FileNotFoundError(
        "no usable backbone checkpoint; every candidate was rejected:\n  "
        + "\n  ".join(tried))


# ── seeding ───────────────────────────────────────────────────────────────────

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


def gather_rng_states(distributed: bool):
    import torch.distributed as dist
    mine = collect_rng_state()
    if not distributed:
        return [mine]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, mine)
    return states


# ── atomic writes ─────────────────────────────────────────────────────────────

def atomic_torch_save(obj, path):
    """fsync before the rename, like every other writer here: ckpt/last.pth
    is the ONE file the whole SLURM chain's recovery depends on, and a
    renamed-but-unflushed file is exactly what a node death produces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(text: str, path, newline="\n", encoding="utf-8"):
    """UTF-8 by default: without an explicit encoding, open() would use the
    compute node's locale, so a non-ASCII class name would land in a
    committed results CSV in cp1252/latin-1 and Phase C's utf-8 read of it
    would raise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline=newline, encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_bytes_write(data: bytes, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── log.csv (append-safe, resume-sanitised) ───────────────────────────────────

def append_log_row(log_path, fields, row: dict):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists() or log_path.stat().st_size == 0
    unknown = set(row) - set(fields)
    if unknown:
        raise KeyError(f"log row carries unknown fields {sorted(unknown)}")
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())


def sanitize_log(log_path, fields, start_epoch: int):
    """Drop rows with epoch >= start_epoch (a crash between the log append and
    the checkpoint save leaves exactly one such row) — atomically."""
    log_path = Path(log_path)
    if not log_path.exists():
        return
    with open(log_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("epoch") not in (None, "")
                and int(r["epoch"]) < start_epoch]
    tmp = log_path.with_name(log_path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, log_path)


def log_epochs(log_path) -> int:
    """Number of data rows in log.csv (0 if absent)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return 0
    with open(log_path, newline="") as f:
        return sum(1 for r in csv.DictReader(f)
                   if r.get("epoch") not in (None, ""))


# ── completion / requeue guard ────────────────────────────────────────────────

def read_meta(run_dir, run_id: str = None) -> dict:
    """Read meta.json, quarantining a TORN one instead of dying on it.

    `run_registry.finalize_run` rewrites meta.json non-atomically (a
    pre-existing, e2r-wide property that TASK-08 recorded as accepted), so a
    node death inside that write can leave unparseable JSON. Letting the
    exception escape would kill the resumed job — and, through
    `--dependency=afterany`, every remaining job of the chain — for a
    completed run. The torn bytes are kept for forensics under
    `meta.json.corrupt.<ns>` and an empty dict is returned, so the caller
    rebuilds the record and the run continues.
    """
    meta_path = Path(run_dir) / META_NAME
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        quarantine = meta_path.with_name(
            f"{META_NAME}.corrupt.{time.time_ns()}")
        try:
            os.replace(meta_path, quarantine)
        except OSError:
            pass
        print(f"  WARNING: {meta_path} was unreadable (torn write?) — moved "
              f"to {quarantine.name} and rebuilding the record"
              + (f" for {run_id}" if run_id else ""), flush=True)
        return {}


def clear_stale_artifacts(paths) -> list:
    """Delete a previous ATTEMPT's outputs (files or whole directories).

    Call this on a fresh start and ONLY AFTER the environment has validated
    (backbone resolved and hash-checked, probe list resolved, model built) —
    a resubmission into a broken environment must not destroy the crashed
    attempt's evidence. Without it, a stale artifact from a discarded attempt
    can survive: e.g. a leftover miou_ms.json makes the multi-scale eval skip
    itself, and the run finalizes carrying an mIoU produced by weights that
    no longer exist.
    """
    removed = []
    for p in paths:
        p = Path(p)
        try:
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
                removed.append(str(p))
            elif p.exists():
                p.unlink()
                removed.append(str(p))
        except OSError as exc:
            print(f"  WARNING: could not remove stale {p}: {exc}", flush=True)
    if removed:
        print(f"  fresh start: cleared {len(removed)} stale artifact(s) from "
              f"a previous attempt", flush=True)
    return removed


def run_is_complete(run_dir, expected_epochs: int) -> bool:
    """True once meta.json carries an end_time (written LAST, by
    finalize_run) and log.csv holds the full epoch count. A surplus job in a
    dependency chain then exits without allocating a model or a loader."""
    run_dir = Path(run_dir)
    meta_path = run_dir / META_NAME
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # a torn meta is NOT proof of completion; read_meta() quarantines it
        # when the run actually starts
        return False
    if not meta.get("end_time"):
        return False
    return log_epochs(run_dir / "log.csv") >= int(expected_epochs)


# ── checkpointing ─────────────────────────────────────────────────────────────

def save_dense_checkpoint(path, *, model_sd, optimizer, scheduler, scaler,
                          epoch, best_metric, best_epoch, last_metric,
                          rng_states, steps_per_epoch, total_epochs,
                          extra=None):
    payload = {
        "epoch": epoch,
        "sampler_epoch": epoch,
        "model": model_sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "last_metric": last_metric,
        "rng_states": rng_states,
        # schedule geometry — checked at resume
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": total_epochs,
    }
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)


def check_schedule_geometry(ckpt: dict, steps_per_epoch: int,
                            total_epochs: int) -> bool:
    """Refuse a resume whose per-epoch step count changed (a differently
    staged dataset would silently shift the LR schedule). Returns True when
    the saved scheduler state may be restored, False when the scheduler was
    deliberately rebuilt (--max_epochs; dev/smoke only)."""
    saved_spe = ckpt.get("steps_per_epoch")
    if saved_spe is not None and int(saved_spe) != int(steps_per_epoch):
        raise RuntimeError(
            f"steps_per_epoch changed across resubmissions: checkpoint has "
            f"{saved_spe}, the current staged dataset gives {steps_per_epoch}. "
            f"Resuming would shift the LR schedule — fix the staging and "
            f"resubmit instead of resuming this state.")
    saved_te = ckpt.get("total_epochs")
    if saved_te is not None and int(saved_te) != int(total_epochs):
        print(f"  WARNING: total_epochs changed {saved_te} -> {total_epochs}; "
              f"scheduler rebuilt for the new schedule (dev/smoke path — "
              f"production resubmits never do this)", flush=True)
        return False
    return True


# ── exact sharding (no padded DistributedSampler; cf. bug B1) ────────────────

def sync_buffers_from_rank0(module, distributed: bool) -> int:
    """Copy rank 0's buffers (BatchNorm running stats) to every rank.

    Why this matters for a NUMBER: DDP synchronises buffers at the start of
    each forward, but every rank then updates its own BatchNorm running
    stats on its own batch, so at the end of an epoch the ranks differ by
    one batch's momentum update. Evaluating a sharded validation set with
    per-rank buffers makes the reported metric depend on which images landed
    on which rank, and makes it differ from a metric computed later from the
    saved (rank-0) weights. Calling this immediately before an evaluation
    removes both effects. Training math is untouched: the next training
    forward re-broadcasts rank 0's buffers anyway.

    Returns the number of buffers synchronised.
    """
    if not distributed:
        return 0
    import torch.distributed as dist
    n = 0
    for buf in module.buffers():
        if buf is not None and buf.numel():
            dist.broadcast(buf, src=0)
            n += 1
    return n


def shard_indices(n: int, rank: int, world_size: int):
    """indices[rank::world_size] — every item is seen exactly once across
    ranks, unlike DistributedSampler which pads the last shard by repeating
    samples (that padding is how B1 double-counted validation images)."""
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside [0,{world_size})")
    return list(range(n))[rank::world_size]
