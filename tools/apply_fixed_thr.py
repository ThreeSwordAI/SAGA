#!/usr/bin/env python3
"""
tools/apply_fixed_thr.py
========================
Backfill fixed-threshold sink counts into existing diagnose JSONs. No GPU,
no forward pass: counts come from the sibling _norms.npz, so
tools/diagnose.py never needs --fixed-thr.

v1 (TASK-02 1.3) — per-arch tau; updates ONLY `sink_fixed_thr` and
`fixed_thr_value`:

    python tools/apply_fixed_thr.py --diag-dir results/legacy/diag \
        --thr-file results/diagsplit/fixed_thresholds.json

v2 (TASK-02C A1) — per-(arch, recipe) tau keyed "<arch>|<recipe>" (recipe
parsed from the diag file stem); adds ONLY `sink_fixed_v2` and
`fixed_thr_v2_value`, never touching the v1 fields:

    python tools/apply_fixed_thr.py --version v2 \
        --diag-dir results/legacy/diag \
        --thr-file results/diagsplit/fixed_thresholds_v2.json

Both modes are idempotent overwrites of exactly their two keys; everything
else is preserved by the JSON rewrite.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

FIELDS = {"v1": ("sink_fixed_thr", "fixed_thr_value"),
          "v2": ("sink_fixed_v2", "fixed_thr_v2_value")}


def fixed_count_mean(norms: np.ndarray, tau: float) -> float:
    """norms [n, N] -> mean over images of #(tokens strictly above tau)."""
    v = norms.astype(np.float64)
    return float((v > tau).sum(axis=1).mean())


def recipe_from_stem(stem: str):
    """e2_vit_small_nomix_saga_rlast_last -> 'nomix' (None if unparsable)."""
    parts = stem.split("_")
    return parts[3] if len(parts) == 7 else None


def apply_to_file(json_path: Path, thresholds: dict,
                  version: str = "v1") -> str:
    npz_path = json_path.with_name(json_path.stem + "_norms.npz")
    if not npz_path.exists():
        return "no _norms.npz sibling"

    with open(json_path) as f:
        diag = json.load(f)

    arch = diag.get("arch")
    if version == "v1":
        key = arch
    else:
        recipe = recipe_from_stem(json_path.stem)
        if recipe is None:
            return "cannot parse recipe from filename"
        key = f"{arch}|{recipe}"
    if key not in thresholds:
        return f"key {key!r} not in thresholds"

    tau = float(thresholds[key])
    with np.load(npz_path) as z:
        norms = z["last_block_patch_norms"]

    count_field, value_field = FIELDS[version]
    diag[count_field] = fixed_count_mean(norms, tau)
    diag[value_field] = tau
    # atomic in-place rewrite: a kill mid-write must never truncate a
    # results JSON (requeue-safety standing rule)
    tmp = json_path.with_name(json_path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(diag, f, indent=2)
    os.replace(tmp, json_path)
    return f"{count_field}={diag[count_field]:.4f} (tau={tau:.4f})"


def main():
    parser = argparse.ArgumentParser(
        description="Backfill fixed-threshold sink counts into diagnose "
                    "JSONs from their _norms.npz siblings.")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--thresholds", "--thr-file", dest="thresholds",
                        default="results/diagsplit/fixed_thresholds.json")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    with open(args.thresholds) as f:
        thresholds = json.load(f)

    updated = skipped = 0
    for json_path in sorted(Path(args.diag_dir).glob("*.json")):
        if json_path.name.endswith("_normstats.json"):
            continue
        status = apply_to_file(json_path, thresholds, version=args.version)
        if status.startswith("sink_fixed"):
            updated += 1
            print(f"  {json_path.name}: {status}")
        else:
            skipped += 1
            print(f"  {json_path.name}: SKIP ({status})")
    print(f"updated {updated}, skipped {skipped}")


if __name__ == "__main__":
    main()
