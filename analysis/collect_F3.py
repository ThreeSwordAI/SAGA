#!/usr/bin/env python3
"""
analysis/collect_F3.py
======================
TASK-02 Phase 2 (2.4): collect the F3 draft data — one row per e2 run,
values verbatim from the diag(last.pth) JSONs.

    python analysis/collect_F3.py \
        [--manifest results/legacy/checkpoint_manifest.csv]
        [--diag-dir results/legacy/diag]
        [--out results/figures_data/F3_legacy.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = ["arch", "recipe", "variant", "sink_mad_k5",
           "oversmooth_pairwise", "oversmooth_pairwise_nosink",
           "ckpt_sha256"]


def main():
    parser = argparse.ArgumentParser(description="Collect F3 draft data.")
    parser.add_argument("--manifest",
                        default="results/legacy/checkpoint_manifest.csv")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--out", default="results/figures_data/F3_legacy.csv")
    args = parser.parse_args()

    runs = sorted({(r["arch"], r["recipe"], r["variant"], r["seed"]): r["sha256"]
                   for r in csv.DictReader(open(args.manifest, newline=""))
                   if r["exp"] == "e2" and r["filename"] == "last.pth"}.items())

    rows, gaps = [], []
    for (arch, recipe, variant, seed), manifest_sha in runs:
        path = Path(args.diag_dir) / f"e2_{arch}_{recipe}_{variant}_{seed}_last.json"
        if not path.exists():
            gaps.append(f"missing: {path}")
            continue
        d = json.load(open(path))
        # same rule as build_legacy_tables: a JSON computed from a different
        # checkpoint than the manifest lists is a gap, never a plotted point
        if d.get("ckpt_sha256") != manifest_sha:
            gaps.append(f"ckpt_sha256 mismatch vs manifest: {path}")
            continue
        rows.append({
            "arch": arch, "recipe": recipe, "variant": variant,
            "sink_mad_k5": d["sink_mad_k5"],
            "oversmooth_pairwise": d["oversmooth_pairwise"],
            "oversmooth_pairwise_nosink": d["oversmooth_pairwise_nosink"],
            "ckpt_sha256": d["ckpt_sha256"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} runs")
    if gaps:
        print("GAPS (rows omitted from the figure data):")
        for g in gaps:
            print("  -", g)
        sys.exit(1)


if __name__ == "__main__":
    main()
