#!/usr/bin/env python3
"""
tools/compute_fixed_thr.py
==========================
Calibrate absolute sink thresholds from BASELINE (last.pth) diagnose norms.
No GPU.

v1 (TASK-02 1.2) — one tau per ARCH from the nomix baseline:

    python tools/compute_fixed_thr.py --diag-dir results/legacy/diag \
        --out results/diagsplit/fixed_thresholds.json

v2 (TASK-02C A1) — one tau per (ARCH, RECIPE) cell from that cell's own
baseline (the per-arch v1 tau saturated on ViT-B/mixup, whose norm scale
dwarfs the nomix calibration). Keys are "<arch>|<recipe>". The v1 file is
left untouched (provenance):

    python tools/compute_fixed_thr.py --per-cell \
        --diag-dir results/legacy/diag \
        --out results/diagsplit/fixed_thresholds_v2.json

Either way: tau = median over images of (median(v_i) + k*MAD(v_i)) on
last_block_patch_norms of the matching baseline-last norms file (exactly one
must match per arch / per cell).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFINITION = ("per arch, on the nomix baseline last.pth diagnose norms: "
              "tau = median over images of (median(v_i) + k*MAD(v_i)) on "
              "last_block_patch_norms, k={k}; all medians are LOWER medians, "
              "matching torch.median as used by saga.metrics.sink_counts_mad")

DEFINITION_V2 = ("per (arch, recipe) cell, on THAT CELL's baseline last.pth "
                 "diagnose norms: tau = median over images of (median(v_i) + "
                 "k*MAD(v_i)) on last_block_patch_norms, k={k}; all medians "
                 "are LOWER medians, matching torch.median as used by "
                 "saga.metrics.sink_counts_mad")


def _lower_median(a: np.ndarray, axis: int) -> np.ndarray:
    """Lower median (element at index (n-1)//2 of the sorted axis) — matches
    torch.median, unlike np.median which interpolates on even counts."""
    idx = (a.shape[axis] - 1) // 2
    return np.take(np.sort(a, axis=axis), idx, axis=axis)


def per_image_mad_thresholds(norms: np.ndarray, k: float) -> np.ndarray:
    """norms [n, N] -> per-image median + k*MAD thresholds [n] (float64),
    same median/MAD convention as saga.metrics.sink_counts_mad."""
    v = norms.astype(np.float64)
    med = _lower_median(v, axis=1)
    mad = _lower_median(np.abs(v - med[:, None]), axis=1)
    return med + k * mad


def compute_tau(npz_path: Path, k: float) -> float:
    with np.load(npz_path) as z:
        norms = z["last_block_patch_norms"]
    return float(_lower_median(per_image_mad_thresholds(norms, k), axis=0))


def find_source(diag_dir: Path, arch: str, recipe: str = "nomix") -> Path:
    pattern = f"*_{arch}_{recipe}_baseline_*_last_norms.npz"
    matches = sorted(diag_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly 1 file matching {pattern} in {diag_dir}, "
            f"found {len(matches)}: {[m.name for m in matches]}")
    return matches[0]


def source_sha(npz_path: Path) -> str:
    sibling = npz_path.with_name(npz_path.name.replace("_norms.npz", ".json"))
    with open(sibling) as f:
        return json.load(f)["ckpt_sha256"]


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate fixed sink thresholds from baseline diagnose "
                    "norms (per arch by default; per (arch, recipe) cell "
                    "with --per-cell).")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--out", default="results/diagsplit/fixed_thresholds.json")
    parser.add_argument("--archs", nargs="+",
                        default=["vit_small", "vit_base"])
    parser.add_argument("--per-cell", action="store_true",
                        help="v2: one tau per (arch, recipe), from that "
                             "cell's own baseline; keys '<arch>|<recipe>'")
    parser.add_argument("--recipes", nargs="+", default=["mixup", "nomix"],
                        help="recipes for --per-cell mode")
    parser.add_argument("--k", type=float, default=5.0)
    args = parser.parse_args()

    if args.per_cell and args.out == parser.get_default("out"):
        raise SystemExit(
            "--per-cell must not write to the v1 default path "
            "(spec: leave the v1 file untouched) — pass --out explicitly, "
            "e.g. --out results/diagsplit/fixed_thresholds_v2.json")

    diag_dir = Path(args.diag_dir)
    result = {}
    sources = {}
    if args.per_cell:
        for arch in args.archs:
            for recipe in args.recipes:
                key = f"{arch}|{recipe}"
                npz_path = find_source(diag_dir, arch, recipe)
                result[key] = compute_tau(npz_path, args.k)
                sources[key] = source_sha(npz_path)
                print(f"{key}: tau = {result[key]:.6f}  "
                      f"(from {npz_path.name})")
        result["definition"] = DEFINITION_V2.format(k=args.k)
    else:
        for arch in args.archs:
            npz_path = find_source(diag_dir, arch)
            result[arch] = compute_tau(npz_path, args.k)
            sources[arch] = source_sha(npz_path)
            print(f"{arch}: tau = {result[arch]:.6f}  (from {npz_path.name})")
        result["definition"] = DEFINITION.format(k=args.k)

    result["k"] = args.k
    result["source_ckpt_sha256"] = sources
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
