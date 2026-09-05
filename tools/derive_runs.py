#!/usr/bin/env python3
"""
tools/derive_runs.py
====================
TASK-06 Phase 1.1 — idempotent derivation driver for the e2r run dirs.
Runs ON THE HPC (single GPU, ~3-5 h for 6 runs), where the checkpoints live.

    python tools/derive_runs.py [--runs-root results/runs] \
        [--pattern 'e2r_*'] [--data $STAGE_DIR] \
        [--split-file results/diagsplit/val_diag_split.json]

For every run dir matching the pattern, for each of {best, last}:
- tools/eval.py            -> <run>/eval/imagenet_val_<tag>.json
- tools/diagnose.py --attn -> <run>/diag/diag_final_<tag>.json (+ npz)
then tools/summarize_norms.py over the run's diag dir.

Idempotency (requeue safety): each step is skipped iff its output JSON
exists and its ckpt_sha256 matches the CURRENT sha256 of the checkpoint
(hashed here at runtime — the e2r checkpoints exist only on the HPC, so no
local manifest can supply the hashes). Failures append to
<runs-root>/derive_failures.log and the driver continues; strict model
loading is never relaxed.

Design note (recorded reconciliation): TASK 06 asked for a script GENERATOR
like tools/gen_rederive_jobs.py; a runtime driver is used instead because
the skip-guard hashes must come from checkpoints that are not visible to
the local generator. Same idempotency contract, same tools underneath.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.run_registry import file_sha256
from tools.check_done import is_done

ARCH_SHORT = {"vit_small_patch16_224": "vit_small",
              "vit_base_patch16_224": "vit_base"}


def run_complete(run_dir: Path):
    """(is_complete, reason). A run is derivable only when it FINISHED:
    meta.json end_time set (finalize_run reached) and log.csv's final epoch
    equals train.epochs - 1. Mid-training checkpoints (last.pth is written
    every epoch) must never masquerade as 'final' eval/diag results."""
    import csv
    import json
    meta_path = run_dir / "meta.json"
    log_path = run_dir / "log.csv"
    if not meta_path.exists() or not log_path.exists():
        return False, "missing meta.json/log.csv"
    if json.load(open(meta_path)).get("end_time") is None:
        return False, "end_time not set (training in progress or killed)"
    cfg = yaml.safe_load(open(run_dir / "config.resolved.yaml"))
    expected = int(cfg["train"]["epochs"]) - 1
    rows = list(csv.DictReader(open(log_path, newline="")))
    final = max(int(r["epoch"]) for r in rows) if rows else -1
    if final != expected:
        return False, f"final log epoch {final} != expected {expected}"
    return True, ""


def run_meta(run_dir: Path):
    """(arch_short, variant) from the run's committed config.resolved.yaml."""
    cfg = yaml.safe_load(open(run_dir / "config.resolved.yaml"))
    arch = ARCH_SHORT.get(cfg["model"]["arch"])
    if arch is None:
        raise ValueError(f"{run_dir.name}: unsupported arch "
                         f"{cfg['model']['arch']!r}")
    return arch, cfg["variant"]


def plan_steps(run_dir: Path, data_root: str, split_file: str,
               python=sys.executable):
    """[(step_name, output_json, ckpt_path, argv)] for one run dir.
    Steps whose checkpoint file is missing are omitted (reported upstream)."""
    arch, variant = run_meta(run_dir)
    steps = []
    for tag in ("best", "last"):
        ckpt = run_dir / "ckpt" / f"{tag}.pth"
        if not ckpt.exists():
            steps.append((f"{run_dir.name}:{tag}:MISSING-CKPT", None, ckpt,
                          None))
            continue
        eval_out = run_dir / "eval" / f"imagenet_val_{tag}.json"
        steps.append((
            f"{run_dir.name}:eval:{tag}", eval_out, ckpt,
            [python, "tools/eval.py", "--ckpt", str(ckpt), "--arch", arch,
             "--variant", variant, "--data", data_root,
             "--out", str(eval_out)]))
        diag_out = run_dir / "diag" / f"diag_final_{tag}.json"
        steps.append((
            f"{run_dir.name}:diag:{tag}", diag_out, ckpt,
            [python, "tools/diagnose.py", "--ckpt", str(ckpt), "--arch", arch,
             "--variant", variant, "--data", data_root,
             "--split-file", split_file, "--out", str(diag_out), "--attn"]))
    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Idempotent eval+diagnose derivation over e2r run dirs.")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--pattern", default="e2r_*")
    parser.add_argument("--data", required=True,
                        help="staged ImageNet root ($STAGE_DIR)")
    parser.add_argument("--split-file",
                        default="results/diagsplit/val_diag_split.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dirs = sorted(d for d in runs_root.glob(args.pattern) if d.is_dir())
    if not run_dirs:
        sys.exit(f"no run dirs matching {args.pattern} under {runs_root}")
    fail_log = runs_root / "derive_failures.log"
    if not args.dry_run:
        fail_log.unlink(missing_ok=True)

    sha_cache = {}
    n_run = n_skip = n_fail = n_incomplete = 0
    for run_dir in run_dirs:
        complete, reason = run_complete(run_dir)
        if not complete:
            # not a failure: e.g. the nomix chains still training — skip
            # loudly, derive them on a later (idempotent) invocation
            print(f"  INCOMPLETE {run_dir.name}: {reason} — skipped")
            n_incomplete += 1
            continue
        for name, out_json, ckpt, argv in plan_steps(
                run_dir, args.data, args.split_file):
            if argv is None:
                print(f"  {name} — checkpoint absent, skipping")
                with open(fail_log, "a") as f:
                    f.write(f"{name}\n")
                n_fail += 1
                continue
            if ckpt not in sha_cache:
                sha_cache[ckpt] = file_sha256(ckpt)
            sha = sha_cache[ckpt]
            if is_done(out_json, sha):
                print(f"  SKIP {name} (up to date)")
                n_skip += 1
                continue
            print(f"  RUN  {name}")
            if args.dry_run:
                continue
            result = subprocess.run(argv)
            if result.returncode != 0:
                with open(fail_log, "a") as f:
                    f.write(f"{name} FAILED (exit {result.returncode})\n")
                n_fail += 1
            else:
                n_run += 1
        # norm summaries for whatever npz now exists in this run's diag dir
        if not args.dry_run and (run_dir / "diag").exists():
            result = subprocess.run(
                [sys.executable, "tools/summarize_norms.py",
                 "--diag-dir", str(run_dir / "diag")])
            if result.returncode != 0:
                with open(fail_log, "a") as f:
                    f.write(f"{run_dir.name}:summarize_norms FAILED "
                            f"(exit {result.returncode})\n")
                n_fail += 1

    print(f"\nderive_runs: {n_run} executed, {n_skip} skipped, "
          f"{n_fail} failed, {n_incomplete} incomplete run(s) deferred")
    if fail_log.exists() and fail_log.stat().st_size > 0:
        print(f"FAILURES logged in {fail_log}:")
        print(fail_log.read_text())
        sys.exit(1)


if __name__ == "__main__":
    main()
