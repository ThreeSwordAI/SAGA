#!/usr/bin/env python3
"""
tools/summarize_norms.py
========================
TASK-02B Phase B1 — runs on the HPC (CPU, ~5 min), where the git-ignored
`_norms.npz` files live.

    python tools/summarize_norms.py --diag-dir results/legacy/diag

For every `<stem>_norms.npz` with a sibling `<stem>.json` (which supplies
arch and ckpt_sha256), reads `last_block_patch_norms [n, N]` and writes a
small committable `<stem>_normstats.json`:

    {"median_of_medians", "mean_mad", "mean_threshold_mad_k5",
     "p50", "p90", "p99", "p999", "max",
     "hist_bin_edges": [65 log-spaced edges spanning all models of the arch],
     "hist_counts": [64], "n_images", "ckpt_sha256"}

Bin edges are identical WITHIN an arch (arch-wide min/max over all flattened
norms is computed first) so histograms overlay cleanly. Median/MAD use the
LOWER-median convention (torch.median / saga.metrics.sink_counts_mad).

Idempotent / requeue-safe: an existing output is skipped iff its
ckpt_sha256 matches the sibling diag JSON AND its bin edges match the
currently computed arch-wide edges (so adding npz files that widen the
arch range rewrites the stats for consistency).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

K_MAD = 5.0
N_EDGES = 65


def _lower_median(a: np.ndarray, axis: int) -> np.ndarray:
    """Lower median — matches torch.median (see tools/compute_fixed_thr.py)."""
    idx = (a.shape[axis] - 1) // 2
    return np.take(np.sort(a, axis=axis), idx, axis=axis)


def norm_stats(norms: np.ndarray) -> dict:
    """Scalar stats of last-block patch norms [n, N] (float64)."""
    v = norms.astype(np.float64)
    med = _lower_median(v, axis=1)                     # per-image median
    mad = _lower_median(np.abs(v - med[:, None]), axis=1)
    flat = v.ravel()
    p50, p90, p99, p999 = np.percentile(flat, [50, 90, 99, 99.9])
    return {
        "median_of_medians": float(_lower_median(med, axis=0)),
        "mean_mad": float(mad.mean()),
        "mean_threshold_mad_k5": float((med + K_MAD * mad).mean()),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "p999": float(p999),
        "max": float(flat.max()),
        "n_images": int(v.shape[0]),
    }


def collect_files(diag_dir: Path):
    """[(npz_path, sibling_json_dict)] for every npz with a usable sibling."""
    out = []
    for npz_path in sorted(diag_dir.glob("*_norms.npz")):
        sibling = npz_path.with_name(
            npz_path.name.replace("_norms.npz", ".json"))
        if not sibling.exists():
            print(f"WARNING: no sibling JSON for {npz_path.name}, skipping",
                  file=sys.stderr)
            continue
        out.append((npz_path, json.load(open(sibling))))
    return out


def arch_edges(files) -> dict:
    """arch -> 65 log-spaced edges spanning all that arch's flattened norms."""
    lo, hi = {}, {}
    for npz_path, meta in files:
        arch = meta["arch"]
        with np.load(npz_path) as z:
            flat = z["last_block_patch_norms"].astype(np.float64).ravel()
        lo[arch] = min(lo.get(arch, np.inf), float(flat.min()))
        hi[arch] = max(hi.get(arch, -np.inf), float(flat.max()))
    return {arch: np.geomspace(max(lo[arch], 1e-6), hi[arch], N_EDGES)
            for arch in lo}


def main():
    parser = argparse.ArgumentParser(
        description="Summarize last-block patch-norm distributions "
                    "(runs on the HPC beside the _norms.npz files).")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)
    files = collect_files(diag_dir)
    if not files:
        sys.exit(f"no *_norms.npz with sibling JSONs under {diag_dir}")

    edges = arch_edges(files)

    written = skipped = 0
    for npz_path, meta in files:
        arch, sha = meta["arch"], meta["ckpt_sha256"]
        e = edges[arch]
        out_path = npz_path.with_name(
            npz_path.name.replace("_norms.npz", "_normstats.json"))

        if out_path.exists():
            try:
                old = json.load(open(out_path))
            except ValueError:
                old = {}
            if (old.get("ckpt_sha256") == sha
                    and np.allclose(old.get("hist_bin_edges", []), e)):
                skipped += 1
                print(f"  SKIP {out_path.name} (up to date)")
                continue

        with np.load(npz_path) as z:
            norms = z["last_block_patch_norms"]
        counts, _ = np.histogram(norms.astype(np.float64).ravel(), bins=e)
        stats = norm_stats(norms)
        stats["hist_bin_edges"] = e.tolist()
        stats["hist_counts"] = counts.tolist()
        stats["ckpt_sha256"] = sha
        with open(out_path, "w") as f:
            json.dump(stats, f)
        written += 1
        print(f"  wrote {out_path.name}")

    print(f"done: {written} written, {skipped} skipped "
          f"(arches: {sorted(edges)})")


if __name__ == "__main__":
    main()
