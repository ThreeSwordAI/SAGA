#!/usr/bin/env python3
"""
analysis/collect_gradphi.py
===========================
TASK-06B Part 3.4 (amended): merge the grad-phi logs of the two
instrumented SAGA runs (mixup s1 AND nomix s1) into one figure-data CSV.

    python analysis/collect_gradphi.py \
        [--out results/figures_data/gradphi.csv]
"""

import argparse
import csv
from pathlib import Path

RUNS = {"mixup": "results/runs/e2r_vits_mixup_saga_s1/grads/grad_phi.csv",
        "nomix": "results/runs/e2r_vits_nomix_saga_s1/grads/grad_phi.csv"}


def main():
    parser = argparse.ArgumentParser(description="Collect grad-phi logs.")
    parser.add_argument("--out", default="results/figures_data/gradphi.csv")
    args = parser.parse_args()

    rows = []
    for recipe, path in RUNS.items():
        p = Path(path)
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        for r in csv.DictReader(open(p, newline="")):
            rows.append({"recipe": recipe, **r})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recipe", "epoch", "iter", "layer",
                                          "grad_phi_norm"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows "
          f"({', '.join(sorted({r['recipe'] for r in rows}))})")


if __name__ == "__main__":
    main()
