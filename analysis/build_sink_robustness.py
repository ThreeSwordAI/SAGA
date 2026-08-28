#!/usr/bin/env python3
"""
analysis/build_sink_robustness.py
=================================
TASK-02B Phase A1: how does the SAGA-vs-baseline (and registers-vs-baseline)
sink ordering depend on the threshold definition?

    python analysis/build_sink_robustness.py \
        [--manifest results/legacy/checkpoint_manifest.csv]
        [--diag-dir results/legacy/diag]
        [--out results/tables/sink_robustness.csv]
        [--verdict-out results/tables/sink_robustness_verdict.csv]

Outputs (values verbatim from the diag(last.pth) JSONs):
- sink_robustness.csv: one row per (arch, recipe, variant) with all seven
  sink counts.
- sink_robustness_verdict.csv: the at-a-glance matrix — rows = threshold
  definitions, columns = the four (arch, recipe) cells, entries S<B / S>B
  (SAGA vs baseline) and R<B / R>B (registers vs baseline).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

THRESHOLDS = ["sink_mad_k5", "sink_mu2s", "sink_mu3s", "sink_mu4s",
              "sink_mu5s", "sink_mu6s", "sink_fixed_thr", "sink_fixed_v2"]

CAVEAT = ("# CAVEAT: sink_fixed_thr (v1) tau is PER-ARCH (calibrated on the "
          "arch's nomix baseline) and saturated on ViT-B/mixup; "
          "sink_fixed_v2 tau is PER-(ARCH, RECIPE), calibrated on each "
          "cell's own baseline. Fixed-threshold comparisons are valid "
          "within one calibration cell, never across.")


def load_diag_last(manifest, diag_dir):
    """(arch, recipe, variant) -> diag(last) JSON dict, sha-checked."""
    runs = {(r["arch"], r["recipe"], r["variant"], r["seed"]): r["sha256"]
            for r in csv.DictReader(open(manifest, newline=""))
            if r["exp"] == "e2" and r["filename"] == "last.pth"}
    out, gaps = {}, []
    for (arch, recipe, variant, seed), sha in sorted(runs.items()):
        path = Path(diag_dir) / f"e2_{arch}_{recipe}_{variant}_{seed}_last.json"
        if not path.exists():
            gaps.append(f"missing: {path}")
            continue
        d = json.load(open(path))
        if d.get("ckpt_sha256") != sha:
            gaps.append(f"ckpt_sha256 mismatch vs manifest: {path}")
            continue
        out[(arch, recipe, variant)] = d
    return out, gaps


def compare(a, b, less: str, greater: str) -> str:
    """Order marker for a vs b; MISSING propagates."""
    if a == "MISSING" or b == "MISSING":
        return "MISSING"
    if a < b:
        return less
    if a > b:
        return greater
    return less.replace("<", "=")


def build_verdict(diags):
    cells = sorted({(a, r) for a, r, _ in diags})
    rows = []
    for block, variant, lt, gt in (("saga_vs_baseline", "saga", "S<B", "S>B"),
                                   ("registers_vs_baseline", "registers",
                                    "R<B", "R>B")):
        for thr in THRESHOLDS:
            row = {"comparison": block, "threshold": thr}
            for arch, recipe in cells:
                v = diags.get((arch, recipe, variant), {}).get(thr, "MISSING")
                b = diags.get((arch, recipe, "baseline"), {}).get(thr, "MISSING")
                if v is None or b is None:      # fixed thr never null here,
                    row[f"{arch}/{recipe}"] = "MISSING"   # but be safe
                else:
                    row[f"{arch}/{recipe}"] = compare(v, b, lt, gt)
            rows.append(row)
    return cells, rows


def main():
    parser = argparse.ArgumentParser(
        description="Sink-threshold robustness table + verdict matrix.")
    parser.add_argument("--manifest",
                        default="results/legacy/checkpoint_manifest.csv")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--out", default="results/tables/sink_robustness.csv")
    parser.add_argument("--verdict-out",
                        default="results/tables/sink_robustness_verdict.csv")
    args = parser.parse_args()

    diags, gaps = load_diag_last(args.manifest, args.diag_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "recipe", "variant"]
                           + THRESHOLDS + ["ckpt_sha256"])
        w.writeheader()
        for (arch, recipe, variant), d in sorted(diags.items()):
            w.writerow({"arch": arch, "recipe": recipe, "variant": variant,
                        **{t: d.get(t, "MISSING") for t in THRESHOLDS},
                        "ckpt_sha256": d["ckpt_sha256"]})
    print(f"wrote {out}: {len(diags)} runs")

    cells, verdict_rows = build_verdict(diags)
    with open(args.verdict_out, "w", newline="") as f:
        f.write(CAVEAT + "\n")
        w = csv.DictWriter(f, fieldnames=["comparison", "threshold"]
                           + [f"{a}/{r}" for a, r in cells])
        w.writeheader()
        w.writerows(verdict_rows)
    print(f"wrote {args.verdict_out}: {len(verdict_rows)} rows "
          f"({len(THRESHOLDS)} thresholds x 2 comparisons, {len(cells)} cells)")

    if gaps:
        print("GAPS:")
        for g in gaps:
            print("  -", g)
        sys.exit(1)


if __name__ == "__main__":
    main()
