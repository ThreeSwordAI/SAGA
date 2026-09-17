#!/usr/bin/env python3
"""
analysis/build_F4_interventions.py
==================================
TASK B B9 — `figures_data/frozen/F4_interventions.npz`.

    python analysis/build_F4_interventions.py

The numbers a Figure 4 would plot, read from the COMMITTED tables rather
than recomputed: `T_I3b`/`T_I3c` and `T_I4b`/`T_I4c`. A figure that
recomputed its own values could disagree with the table beside it, and then
neither would be the number.

Four panels' worth of arrays:

    i3_contrast   C1/C2/S1/S2 per (contrast, layer, run): mean, CI, verdict,
                  scope, whether it decides
    i3_strata     Delta-NLL by delta_update_norm decile per (family, layer,
                  run) — the panel that shows the families at equal energy
    i4_theta      C3 per (layer, method, run): theta(primary), the control
                  band, the contrast and its CI, whether the primary sits
                  inside the band
    i4_cross      baseline - SAGA paired on the provenance tag

Every array is parallel: index k of `i3_contrast_mean` belongs to index k of
`i3_contrast_run`, and so on. Strings are stored as unicode arrays, floats as
float64 with NaN for MISSING — an npz cannot hold the literal MISSING, and a
NaN propagates through a mean instead of being silently skipped, which is the
behaviour a figure wants.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MISSING = "MISSING"
I3_TABLES = "results/frozen/I3_gate_edits/evaluation/tables"
I4_TABLES = "results/frozen/I4_perturbation/evaluation/tables"


def read(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v):
    """float, or NaN for MISSING — an npz cannot carry the literal."""
    if v in (None, "", MISSING):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def col(rows, key, cast=str):
    return np.asarray([cast(r.get(key, "")) if cast is str
                       else fnum(r.get(key)) for r in rows])


def build(i3_tables=I3_TABLES, i4_tables=I4_TABLES) -> dict:
    out, meta = {}, {}

    b3 = read(Path(i3_tables) / "T_I3b_contrasts.csv")
    for name, key in (("contrast", "contrast"), ("role", "role"),
                      ("run", "run_id"), ("scope", "scope"),
                      ("verdict", "verdict"), ("branch", "branch")):
        out[f"i3_contrast_{name}"] = col(b3, key)
    for name in ("layer", "mean_nats", "ci_lo", "ci_hi", "delta_top1_points",
                 "frac_positive"):
        out[f"i3_contrast_{name}"] = col(b3, name, float)
    out["i3_contrast_decides"] = col(b3, "decides", float)
    meta["i3_contrast_rows"] = len(b3)

    c3 = read(Path(i3_tables) / "T_I3c_energy_strata.csv")
    for name, key in (("run", "run_id"), ("family", "family"),
                      ("scope", "scope")):
        out[f"i3_strata_{name}"] = col(c3, key)
    for name in ("layer", "decile", "n_pairs", "mean_delta_nll", "ci_lo",
                 "ci_hi", "mean_delta_update_norm"):
        out[f"i3_strata_{name}"] = col(c3, name, float)
    meta["i3_strata_rows"] = len(c3)

    b4 = read(Path(i4_tables) / "T_I4b_contrast.csv")
    for name, key in (("contrast", "contrast"), ("run", "run_id"),
                      ("method", "method"), ("scope", "scope"),
                      ("verdict", "verdict"), ("branch", "branch"),
                      ("control_kind", "control_kind")):
        out[f"i4_theta_{name}"] = col(b4, key)
    for name in ("layer", "theta_primary", "theta_control_mean",
                 "control_band_lo", "control_band_hi", "mean_nats", "ci_lo",
                 "ci_hi", "delta_top1_points"):
        out[f"i4_theta_{name}"] = col(b4, name, float)
    out["i4_theta_inside_band"] = col(b4, "primary_inside_band", float)
    out["i4_theta_decides"] = col(b4, "decides", float)
    meta["i4_theta_rows"] = len(b4)

    c4 = [r for r in read(Path(i4_tables) / "T_I4c_cross_method.csv")
          if r.get("comparison") == "baseline_minus_saga"]
    for name, key in (("run", "run_id"), ("paired_with", "paired_with"),
                      ("tag", "provenance_tag")):
        out[f"i4_cross_{name}"] = col(c4, key)
    for name in ("layer", "theta_primary", "paired_difference"):
        out[f"i4_cross_{name}"] = col(c4, name, float)
    meta["i4_cross_rows"] = len(c4)

    for src, dst in ((Path(i3_tables) / "I3_tables_meta.json", "i3"),
                     (Path(i4_tables) / "I4_tables_meta.json", "i4")):
        if src.exists():
            meta[dst] = json.loads(src.read_text(encoding="utf-8"))
    meta["generated_by"] = "analysis/build_F4_interventions.py"
    meta["generated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    meta["note"] = ("read from the committed tables, never recomputed: a "
                    "figure that derived its own values could disagree with "
                    "the table beside it")
    out["meta_json"] = np.array(json.dumps(meta, indent=2, sort_keys=True,
                                           default=str))
    return out


def main():
    p = argparse.ArgumentParser(description="Build F4's figure data.")
    p.add_argument("--out", default="figures_data/frozen/F4_interventions.npz")
    p.add_argument("--i3-tables", default=I3_TABLES)
    p.add_argument("--i4-tables", default=I4_TABLES)
    args = p.parse_args()

    payload = build(args.i3_tables, args.i4_tables)
    empty = [k for k in ("i3_contrast_run", "i4_theta_run")
             if payload[k].size == 0]
    if empty:
        print(f"REFUSING: {empty} are empty — the tables have not been built",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:            # np.savez appends .npz to a PATH
        np.savez_compressed(fh, **payload)
    tmp.replace(out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
    for k in ("i3_contrast", "i3_strata", "i4_theta", "i4_cross"):
        print(f"  {k}: {payload[k + '_run'].size} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
