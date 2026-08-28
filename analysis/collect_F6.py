#!/usr/bin/env python3
"""
analysis/collect_F6.py
======================
TASK-02B Phase A2: collect the F6 (relocation) data — per-block CLS
norm-ratio and CLS attention-share curves, verbatim from the diag(last.pth)
JSONs.

    python analysis/collect_F6.py \
        [--manifest results/legacy/checkpoint_manifest.csv]
        [--diag-dir results/legacy/diag]
        [--out results/figures_data/F6_legacy.csv]

Long format: one row per (arch, recipe, variant, block).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = ["arch", "recipe", "variant", "block",
           "cls_norm_ratio", "cls_attn_share", "ckpt_sha256"]


def main():
    parser = argparse.ArgumentParser(description="Collect F6 relocation data.")
    parser.add_argument("--manifest",
                        default="results/legacy/checkpoint_manifest.csv")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--out", default="results/figures_data/F6_legacy.csv")
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
        if d.get("ckpt_sha256") != manifest_sha:
            gaps.append(f"ckpt_sha256 mismatch vs manifest: {path}")
            continue
        ratio, share = d["cls_norm_ratio"], d["cls_attn_share"]
        if share is None:
            share = [None] * len(ratio)
        for b, (rt, sh) in enumerate(zip(ratio, share)):
            rows.append({"arch": arch, "recipe": recipe, "variant": variant,
                         "block": b, "cls_norm_ratio": rt,
                         "cls_attn_share": "" if sh is None else sh,
                         "ckpt_sha256": d["ckpt_sha256"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")
    if gaps:
        print("GAPS (runs omitted):")
        for g in gaps:
            print("  -", g)
        sys.exit(1)


if __name__ == "__main__":
    main()
