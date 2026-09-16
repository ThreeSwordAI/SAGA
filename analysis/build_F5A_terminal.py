#!/usr/bin/env python3
"""
analysis/build_F5A_terminal.py
==============================
TASK I2 D6 — the data behind Figure 5A.

    python analysis/build_F5A_terminal.py
        -> figures_data/frozen/F5A_terminal.npz

The panel the plan asks for (§6.5): **diagnostic movement against zero
classifier response.** So the file carries, per SAGA/baseline PAIR and at
BOTH stages the terminal conditions are captured at:

  value|<cell>|<tag>|<stage>|<diag>     the SAGA checkpoint's mean at each
                                        condition, in CONDITIONS order
  baseline|<cell>|<tag>|<stage>|<diag>  its paired baseline under `native`,
                                        the level the movement is against
  gap|<cell>|<tag>|<stage>|<diag>       value - baseline, the quantity
                                        T_I2c bootstraps

and, for the horizontal line that makes the panel a claim rather than a
plot:

  logit_diff|<run_id>                   max abs logit difference vs native
                                        at each condition — the ZERO LINE
  top1_agree|<run_id>                   top-1 agreement at each condition

Everything is read from the committed tables (`T_I2a_invariance.csv` and
`T_I2b_sweep.csv`); nothing is recomputed from a records file, so the figure
and the tables cannot disagree. Pairing is `analysis/build_I2_tables.pairs_for`,
which is `analysis/build_pooled_tables`' own.

No training, no optimizer, no probe fitting.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.build_I2_tables import (CELLS, DIAGNOSTICS, MISSING,  # noqa: E402
                                      pairs_for)
from saga.frozen.runner import eligible_cohort  # noqa: E402

#: The x axis: `native` first, then the four declared constants in order.
CONDITIONS = ("native", "term_0.25", "term_0.50", "term_0.75", "term_1.00")
#: The constant each condition applies; `native` has none (NaN).
CONSTANTS = (np.nan, 0.25, 0.50, 0.75, 1.00)
#: Both stages the term_* conditions are captured at (task §3).
STAGES = ("hist", "s12_post_norm")


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v):
    if v in (None, "", MISSING):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def build(tables, manifest):
    tables = Path(tables)
    sweep = _read(tables / "T_I2b_sweep.csv")
    inv = _read(tables / "T_I2a_invariance.csv")
    cohort = eligible_cohort(manifest)

    by = {}
    for r in sweep:
        by[(r["run_id"], r["condition_id"], r["stage"])] = r
    arrays, covered = {}, []

    for arch, rec in CELLS:
        cell = f"{arch}|{rec}"
        for tag, base_id, saga_id in pairs_for(cohort, arch, rec, "saga"):
            for stage in STAGES:
                for diag in DIAGNOSTICS:
                    vals = np.array(
                        [_num((by.get((saga_id, c, stage)) or {}).get(diag))
                         for c in CONDITIONS], dtype=np.float64)
                    base = _num((by.get((base_id, "native", stage)) or {})
                                .get(diag))
                    if np.all(np.isnan(vals)) and np.isnan(base):
                        continue
                    key = f"{cell}|{tag}|{stage}|{diag}"
                    arrays[f"value|{key}"] = vals
                    arrays[f"baseline|{key}"] = np.float64(base)
                    arrays[f"gap|{key}"] = vals - base
            covered.append((cell, tag, base_id, saga_id))

    # the zero line: one row per (SAGA checkpoint, constant) in T_I2a, with
    # `native` prepended as an exact 0 (it is the reference, not a measurement)
    inv_by = {(r["run_id"], r["condition_id"]): r for r in inv}
    for run_id in sorted({r["run_id"] for r in inv}):
        arrays[f"logit_diff|{run_id}"] = np.array(
            [0.0] + [_num((inv_by.get((run_id, c)) or {})
                          .get("max_abs_logit_diff_vs_native"))
                     for c in CONDITIONS[1:]], dtype=np.float64)
        arrays[f"top1_agree|{run_id}"] = np.array(
            [1.0] + [_num((inv_by.get((run_id, c)) or {})
                          .get("top1_agreement"))
                     for c in CONDITIONS[1:]], dtype=np.float64)

    prov = sweep[0]
    meta = {
        "generated_by": "analysis/build_F5A_terminal.py",
        "tables": str(tables),
        "work_package": prov["work_package"],
        "split_name": prov["split_name"],
        "split_sha256": prov["split_sha256"],
        "git_sha": prov["git_sha"],
        "n_images": prov["n_images"],
        "conditions": list(CONDITIONS),
        "stages": list(STAGES),
        "diagnostics": list(DIAGNOSTICS),
        "pairs": [{"cell": c, "tag": t, "baseline_run_id": b,
                   "saga_run_id": s} for c, t, b, s in covered],
        "note": ("value/baseline/gap are means over images, read from "
                 "T_I2b_sweep.csv; logit_diff and top1_agree are read from "
                 "T_I2a_invariance.csv. The native entry of logit_diff is 0 "
                 "by definition — it is the reference the others are measured "
                 "against, not a measurement."),
    }
    return arrays, meta


def main():
    p = argparse.ArgumentParser(description="Build Figure 5A's data.")
    p.add_argument("--tables",
                   default="results/frozen/I2_terminal/evaluation/tables")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out", default="figures_data/frozen/F5A_terminal.npz")
    args = p.parse_args()

    arrays, meta = build(args.tables, args.manifest)
    payload = dict(sorted(arrays.items()))
    payload["__conditions"] = np.asarray(list(CONDITIONS))
    payload["__constants"] = np.asarray(CONSTANTS, dtype=np.float64)
    payload["__meta_json"] = np.asarray(
        json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    print(f"wrote {out}: {len(arrays)} series over {len(meta['pairs'])} pairs "
          f"({out.stat().st_size / 1e3:.1f} kB, split {meta['split_name']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
