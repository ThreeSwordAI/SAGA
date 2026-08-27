"""
saga/run_registry.py
====================
Provenance registry for every run that produces a paper number.

Every entrypoint (train / eval / diagnose) calls create_run() once at start
and finalize_run() at the end, so each results/runs/<run_id>/ directory
carries a meta.json answering: which code, which command, which seed,
which machine, which library versions.

    from saga.run_registry import create_run, finalize_run, file_sha256

    run_dir = create_run("results/runs", run_id, cfg_dict, seed=0)
    ...
    finalize_run(run_dir)
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

META_NAME = "meta.json"
CONFIG_NAME = "config.resolved.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_info(cwd: Path):
    """Return (sha, dirty) of the repo containing `cwd`; ('unknown', None) if no git."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True,
            stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True,
            stderr=subprocess.DEVNULL)
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", None


def git_sha(cwd=None) -> str:
    """Convenience: sha of the repo containing this file (or `cwd`)."""
    return _git_info(Path(cwd) if cwd else Path(__file__).resolve().parent)[0]


def file_sha256(path, max_bytes=None) -> str:
    """sha256 hex digest of a file. `max_bytes` limits hashing to the first
    N bytes (faster on multi-GB checkpoints; note it in the consumer if used)."""
    h = hashlib.sha256()
    remaining = max_bytes
    with open(path, "rb") as f:
        while True:
            chunk_size = 1 << 20
            if remaining is not None:
                if remaining <= 0:
                    break
                chunk_size = min(chunk_size, remaining)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def _gpu_names():
    try:
        import torch
        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def _versions():
    out = {"python": platform.python_version(), "torch": None,
           "timm": None, "cuda": None}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        import timm
        out["timm"] = timm.__version__
    except ImportError:
        pass
    return out


def _world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except ImportError:
        pass
    return int(os.environ.get("WORLD_SIZE", 1))


def create_run(out_root, run_id: str, config_dict: dict, seed: int) -> Path:
    """Create results-run directory `out_root/run_id`, write
    config.resolved.yaml and meta.json. Returns the run dir Path."""
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / CONFIG_NAME, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    sha, dirty = _git_info(Path(__file__).resolve().parent)
    v = _versions()
    meta = {
        "run_id": run_id,
        "git_sha": sha,
        "git_dirty": dirty,
        "cmd": " ".join(sys.argv),
        "seed": seed,
        "world_size": _world_size(),
        "hostname": socket.gethostname(),
        "gpu_names": _gpu_names(),
        "torch": v["torch"],
        "timm": v["timm"],
        "python": v["python"],
        "cuda": v["cuda"],
        "start_time": _utc_now(),
        "end_time": None,
    }
    with open(run_dir / META_NAME, "w") as f:
        json.dump(meta, f, indent=2)
    return run_dir


def finalize_run(run_dir) -> None:
    """Fill meta.json's end_time."""
    meta_path = Path(run_dir) / META_NAME
    with open(meta_path) as f:
        meta = json.load(f)
    meta["end_time"] = _utc_now()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
