#!/usr/bin/env python3
"""
tools/apply_fixed_thr.py
========================
Backfill the fixed-threshold sink count into existing diagnose JSONs
(TASK-02 1.3). No GPU, no forward pass: counts come from the sibling
_norms.npz, so tools/diagnose.py never needs --fixed-thr in this task.

    python tools/apply_fixed_thr.py --diag-dir results/legacy/diag \
        --thresholds results/diagsplit/fixed_thresholds.json

For every <name>.json with a sibling <name>_norms.npz: count per image
v > tau(arch) on last_block_patch_norms, take the mean, and update ONLY the
fields `sink_fixed_thr` and `fixed_thr_value` (idempotent overwrite of
exactly those two keys; everything else byte-preserved by JSON rewrite).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def fixed_count_mean(norms: np.ndarray, tau: float) -> float:
    """norms [n, N] -> mean over images of #(tokens strictly above tau)."""
    v = norms.astype(np.float64)
    return float((v > tau).sum(axis=1).mean())


def apply_to_file(json_path: Path, thresholds: dict) -> str:
    npz_path = json_path.with_name(json_path.stem + "_norms.npz")
    if not npz_path.exists():
        return "no _norms.npz sibling"

    with open(json_path) as f:
        diag = json.load(f)

    arch = diag.get("arch")
    if arch not in thresholds:
        return f"arch {arch!r} not in thresholds"

    tau = float(thresholds[arch])
    with np.load(npz_path) as z:
        norms = z["last_block_patch_norms"]

    diag["sink_fixed_thr"] = fixed_count_mean(norms, tau)
    diag["fixed_thr_value"] = tau
    with open(json_path, "w") as f:
        json.dump(diag, f, indent=2)
    return f"sink_fixed_thr={diag['sink_fixed_thr']:.4f} (tau={tau:.4f})"


def main():
    parser = argparse.ArgumentParser(
        description="Backfill sink_fixed_thr/fixed_thr_value into diagnose "
                    "JSONs from their _norms.npz siblings.")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--thresholds",
                        default="results/diagsplit/fixed_thresholds.json")
    args = parser.parse_args()

    with open(args.thresholds) as f:
        thresholds = json.load(f)

    updated = skipped = 0
    for json_path in sorted(Path(args.diag_dir).glob("*.json")):
        status = apply_to_file(json_path, thresholds)
        if status.startswith("sink_fixed_thr"):
            updated += 1
            print(f"  {json_path.name}: {status}")
        else:
            skipped += 1
            print(f"  {json_path.name}: SKIP ({status})")
    print(f"updated {updated}, skipped {skipped}")


if __name__ == "__main__":
    main()
