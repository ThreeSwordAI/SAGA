#!/usr/bin/env python3
"""
analysis/check_eval_consistency.py
==================================
TASK-06 Phase 1.3: for every run dir, compare eval/imagenet_val_last.json
top-1 against the FINAL epoch's val_top1_full row of log.csv. Both come from
the same exact-val code path, so any |delta| > tolerance is a WARNING that
something is wrong (different checkpoint, different data, a broken shard).

    python analysis/check_eval_consistency.py [--runs-root results/runs]
        [--tolerance 0.02]

Exit code 1 if any WARNING fired.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def check_run(run_dir: Path, tolerance: float, expected_epoch: int = 299):
    """(status, message). status in {'ok', 'warn', 'skip'}.

    Compares against the EXPECTED final epoch's row (spec: epoch 299), not
    merely the last row present — a run truncated before the final epoch is
    itself a WARNING (a mid-training last.pth would otherwise pass this
    check self-consistently)."""
    eval_path = run_dir / "eval" / "imagenet_val_last.json"
    log_path = run_dir / "log.csv"
    if not eval_path.exists() or not log_path.exists():
        return "skip", f"{run_dir.name}: missing " + (
            "eval JSON" if not eval_path.exists() else "log.csv")

    eval_top1 = json.load(open(eval_path))["top1"]
    rows = list(csv.DictReader(open(log_path, newline="")))
    final = max(rows, key=lambda r: int(r["epoch"]))
    if int(final["epoch"]) != expected_epoch:
        return "warn", (f"WARNING {run_dir.name}: log.csv ends at epoch "
                        f"{final['epoch']} != expected {expected_epoch} — "
                        f"run truncated; eval(last) is NOT a final-epoch "
                        f"result")
    log_top1 = float(final["val_top1_full"])
    delta = eval_top1 - log_top1
    msg = (f"{run_dir.name}: eval(last)={eval_top1}  "
           f"log.csv(e{final['epoch']})={log_top1}  delta={delta:+.4f}")
    if abs(delta) > tolerance:
        return "warn", "WARNING " + msg
    return "ok", msg


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check eval JSONs against log.csv final epochs.")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--pattern", default="e2r_*")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--expected-epoch", type=int, default=299)
    args = parser.parse_args()

    warnings = 0
    for run_dir in sorted(Path(args.runs_root).glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        status, msg = check_run(run_dir, args.tolerance,
                                expected_epoch=args.expected_epoch)
        print(("  " if status == "ok" else "") + msg)
        warnings += status == "warn"

    print(f"\nconsistency check: {warnings} warning(s)")
    sys.exit(1 if warnings else 0)


if __name__ == "__main__":
    main()
