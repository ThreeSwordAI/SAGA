#!/usr/bin/env python3
"""
analysis/frozen_I7_ttr_curve.py
===============================
TASK C / I7 §6 — the test-time-registers OPERATING CURVE, built from files
that already exist.

    python analysis/frozen_I7_ttr_curve.py

WHY A CURVE AND NOT A POINT
---------------------------
TASK-10 reports ONE test-time-register configuration per cell and its
verdict. A single point cannot be read: it does not say whether the chosen
neuron count sits on a plateau or on a cliff, nor what was given up to reach
it. This module draws the trade-off the point sits on — outlier reduction
against top-1 change — with the validation gate's own thresholds drawn as
lines and the chosen operating point marked.

NO NEW RUNS. Every number comes from a committed file. A grid point that is
not in one is `MISSING` and stays `MISSING`: it is never interpolated from
its neighbours, never carried over from another layer range, and never
averaged away (I0 handoff §8.2).

WHERE THE POINTS LIVE, AND THE ONE THING A READER MUST KNOW
------------------------------------------------------------
There are TWO layer ranges in this repository, swept at DIFFERENT neuron
counts, and only one of them covers every cell:

    results/ttr/<base_run_id>/          layer_range [0, 12]  ("all")
                                        n_neurons 0, 9..15
                                        1 of the 4 cells
    results/ttr_midlayer/<base_run_id>/ layer_range [3, 12]  ("midlayer")
                                        n_neurons 0, 8, 10, 12, 16, 24
                                        4 of the 4 cells

`results/tables/T_ttr_sweep.csv` is the committed aggregate of the MIDLAYER
sweep only — the all-layer sweep is not in it — so this module reads the
per-run `sweep.csv` / `validate.json` files directly and uses the table as a
CROSS-CHECK rather than as the source. `check_against_table()` reports any
row where the two disagree; a disagreement is a finding, not something to
resolve silently.

The two ranges are NOT a crossed grid, and the coverage table says so
explicitly rather than presenting 56 `MISSING` cells as if someone had
intended to run them.

THE GATE
--------
The PASS/FAIL thresholds are read from each run's own `validate.json`
(`min_sink_reduction`, `max_top1_drop`) — the values TASK-10's gate actually
applied — and are asserted to agree across every file rather than restated
here. `docs/Task10_handsoff.md` §3-§4 describes that gate.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MISSING = "MISSING"

#: The two swept layer ranges, as {label: directory}. A label is this
#: module's name for a range; the RANGE itself is read from each
#: `validate.json` and never assumed from the directory name.
SWEEP_ROOTS = {
    "all": REPO / "results" / "ttr",
    "midlayer": REPO / "results" / "ttr_midlayer",
}

#: The committed aggregate, used only as a cross-check (see the docstring).
SWEEP_TABLE = REPO / "results" / "tables" / "T_ttr_sweep.csv"

#: The committed per-cell result, which carries the CHOSEN operating point.
TTR_TABLE = REPO / "results" / "tables" / "T_ttr.csv"

#: Descriptive throughout: §6 draws a trade-off that already exists in the
#: repository; it tests nothing and decides nothing.
DESCRIPTIVE = "descriptive"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rel(path) -> str:
    """The path relative to the repository, or its absolute form.

    Provenance must never be the thing that raises. A sweep root outside the
    repository — a test fixture, or a tree staged elsewhere on the cluster —
    has no relative form, and `Path.relative_to` raises rather than returning
    one. The source file is recorded either way.
    """
    path = Path(path)
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def num(v):
    """A float, or MISSING — never a silent NaN and never a zero."""
    if v is None or v == "" or v == MISSING:
        return MISSING
    try:
        f = float(v)
    except (TypeError, ValueError):
        return MISSING
    return f if np.isfinite(f) else MISSING


def load_sweeps():
    """Every committed sweep point, as a flat list of dicts.

    One dict per (layer_range label, base_run_id, n_neurons), carrying the
    quantities the curve needs and the provenance that identifies them.
    """
    points, gates, sources = [], {}, []
    for label, root in SWEEP_ROOTS.items():
        if not root.exists():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            sweep_csv = run_dir / "sweep.csv"
            validate = run_dir / "validate.json"
            if not sweep_csv.exists():
                continue
            meta = {}
            if validate.exists():
                meta = json.loads(validate.read_text(encoding="utf-8"))
                verdict = meta.get("verdict", {})
                gates[f"{label}|{run_dir.name}"] = {
                    "min_sink_reduction": verdict.get("min_sink_reduction"),
                    "max_top1_drop": verdict.get("max_top1_drop"),
                }
            scan = meta.get("scan_stats", {}) or {}
            layer_range = meta.get("layer_range") or scan.get("layer_range")
            sources.append(rel(sweep_csv))
            for r in read_csv(sweep_csv):
                points.append({
                    "layer_range_label": label,
                    "layer_range": (",".join(str(x) for x in layer_range)
                                    if layer_range else MISSING),
                    "base_run_id": run_dir.name,
                    "arch": meta.get("arch", MISSING),
                    "recipe_actual": meta.get("recipe", MISSING),
                    "n_neurons": int(r["n_neurons"]),
                    "n_images": num(r.get("n_images")),
                    "top1": num(r.get("top1")),
                    "top1_drop": num(r.get("top1_drop")),
                    "outlier_count_canon": num(r.get("sink_fixed_canon")),
                    "outlier_count_mad": num(r.get("sink_mad_k5")),
                    "outlier_reduction_frac": num(r.get("sink_reduction_frac")),
                    "oversmooth_pairwise": num(r.get("oversmooth_pairwise")),
                    "layers_touched": r.get("layers_touched") or MISSING,
                    "passes": r.get("passes") or MISSING,
                    "ckpt_sha256": meta.get("ckpt_sha256", MISSING),
                    "canon_tau": num(meta.get("canon_tau")),
                    "source_file": rel(sweep_csv),
                    "git_sha_at_run": meta.get("git_sha", MISSING),
                    "endpoint_class": DESCRIPTIVE,
                })
    return points, gates, sources


def gate_thresholds(gates):
    """The ONE gate, asserted to be the same in every file that declares it.

    Read, not restated: these are the numbers TASK-10's validation gate
    actually applied. A file that disagreed would mean two different gates
    had been run and the curve's threshold lines would be a fiction.
    """
    seen = {(g["min_sink_reduction"], g["max_top1_drop"])
            for g in gates.values()
            if g["min_sink_reduction"] is not None}
    if not seen:
        return {"min_outlier_reduction": MISSING, "max_top1_drop": MISSING,
                "agreed": False, "n_files": 0}
    if len(seen) > 1:
        raise ValueError(
            f"the committed validate.json files declare {len(seen)} different "
            f"gates: {sorted(seen)}. The curve cannot draw one threshold line "
            f"for two gates — reconcile them before plotting.")
    lo, hi = seen.pop()
    return {"min_outlier_reduction": float(lo), "max_top1_drop": float(hi),
            "agreed": True, "n_files": len(gates)}


def chosen_points():
    """{base_run_id: the operating point T_ttr.csv reports} from the table."""
    out = {}
    for r in read_csv(TTR_TABLE):
        out[r["base_run_id"]] = {
            "cell": r.get("cell", MISSING),
            "n_neurons": int(r["n_neurons"]) if str(
                r.get("n_neurons", "")).strip().isdigit() else MISSING,
            "layer_range": (r.get("layer_range") or MISSING).strip('"'),
            "gate_verdict": r.get("gate_verdict", MISSING),
            "gate_best_n": r.get("gate_best_n", MISSING),
            "top1_drop_vs_baseline": num(r.get("top1_drop_vs_baseline")),
            "outlier_reduction_frac": num(r.get("sink_canon_reduction_frac")),
        }
    return out


def coverage(points):
    """Which (layer_range, cell) combinations were swept, and on what grid.

    Deliberately NOT a crossed grid of every label against every cell: the
    two ranges were swept at different neuron counts on different cell sets,
    and presenting the cross product would report dozens of `MISSING` points
    nobody ever intended to run. The table states, per range, which cells are
    present and which are absent — which is the fact §6 asks for.
    """
    all_cells = sorted({p["base_run_id"] for p in points})
    rows = []
    for label in SWEEP_ROOTS:
        here = [p for p in points if p["layer_range_label"] == label]
        present = sorted({p["base_run_id"] for p in here})
        grid = sorted({p["n_neurons"] for p in here})
        for cell in all_cells:
            rows.append({
                "layer_range_label": label,
                "layer_range": next((p["layer_range"] for p in here), MISSING),
                "base_run_id": cell,
                "status": "present" if cell in present else MISSING,
                "n_neurons_grid": ("|".join(str(n) for n in grid)
                                   if cell in present else MISSING),
                "n_points": (len([p for p in here
                                  if p["base_run_id"] == cell])
                             if cell in present else 0),
                "endpoint_class": DESCRIPTIVE,
            })
    return rows


def check_against_table(points):
    """Rows where a per-run sweep and `T_ttr_sweep.csv` disagree.

    The committed table covers the MIDLAYER sweep only, so every row of it
    should match a midlayer point exactly. A mismatch is reported, never
    reconciled here.
    """
    table = read_csv(SWEEP_TABLE)
    if not table:
        return [{"issue": "T_ttr_sweep.csv is absent", "detail": MISSING}]
    index = {(p["base_run_id"], p["n_neurons"]): p for p in points
             if p["layer_range_label"] == "midlayer"}
    issues = []
    for r in table:
        key = (r["base_run_id"], int(r["n_neurons"]))
        p = index.get(key)
        if p is None:
            issues.append({"issue": "table row has no per-run point",
                           "detail": f"{key}"})
            continue
        for tcol, pcol in (("top1", "top1"), ("top1_drop", "top1_drop"),
                           ("sink_canon", "outlier_count_canon"),
                           ("sink_reduction_frac",
                            "outlier_reduction_frac")):
            a, b = num(r.get(tcol)), p.get(pcol)
            if a is MISSING or b is MISSING:
                continue
            if abs(a - b) > 1e-6:
                issues.append({
                    "issue": "value disagrees with T_ttr_sweep.csv",
                    "detail": f"{key} {tcol}: table={a} per-run={b}"})
    return issues


CURVE_FIELDS = ("layer_range_label", "layer_range", "base_run_id", "arch",
                "recipe_actual", "n_neurons", "n_images", "top1", "top1_drop",
                "outlier_count_canon", "outlier_count_mad",
                "outlier_reduction_frac", "oversmooth_pairwise",
                "layers_touched", "passes", "is_chosen_operating_point",
                "canon_tau", "ckpt_sha256", "source_file", "git_sha_at_run",
                "endpoint_class")

COVERAGE_FIELDS = ("layer_range_label", "layer_range", "base_run_id",
                   "status", "n_neurons_grid", "n_points", "endpoint_class")


def fmt(v) -> str:
    if v is MISSING or v is None:
        return MISSING
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return MISSING if not np.isfinite(f) else f"{f:.10g}"


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, MISSING)) for k in fields})
    return path


def build(out_dir=None, npz_path=None):
    """The curve, the coverage table and the figure data. Returns a report."""
    points, gates, sources = load_sweeps()
    if not points:
        return {"status": MISSING,
                "reason": f"no sweep.csv under {list(SWEEP_ROOTS.values())}"}

    gate = gate_thresholds(gates)
    chosen = chosen_points()
    for p in points:
        c = chosen.get(p["base_run_id"], {})
        # The chosen point is the one T_ttr.csv reports: same cell, same
        # neuron count AND same layer range. Matching on n_neurons alone
        # would mark a point in the other range as chosen.
        p["is_chosen_operating_point"] = int(
            c.get("n_neurons") == p["n_neurons"]
            and str(c.get("layer_range", "")).replace(" ", "")
            == str(p["layer_range"]).replace(" ", ""))

    out_dir = Path(out_dir or REPO / "results" / "frozen" / "I7_attention"
                   / "tables")
    curve_path = write_csv(out_dir / "T_I7d_ttr_curve.csv", CURVE_FIELDS,
                           points)
    cov = coverage(points)
    cov_path = write_csv(out_dir / "T_I7d_ttr_coverage.csv", COVERAGE_FIELDS,
                         cov)
    issues = check_against_table(points)

    report = {
        "status": "OK",
        "n_points": len(points),
        "n_points_by_range": {
            label: len([p for p in points if p["layer_range_label"] == label])
            for label in SWEEP_ROOTS},
        "cells_by_range": {
            label: sorted({p["base_run_id"] for p in points
                           if p["layer_range_label"] == label})
            for label in SWEEP_ROOTS},
        "n_neurons_by_range": {
            label: sorted({p["n_neurons"] for p in points
                           if p["layer_range_label"] == label})
            for label in SWEEP_ROOTS},
        "missing_cells_by_range": {
            label: [r["base_run_id"] for r in cov
                    if r["layer_range_label"] == label
                    and r["status"] == MISSING]
            for label in SWEEP_ROOTS},
        "gate": gate,
        "chosen_operating_points": chosen,
        "n_marked_chosen": sum(p["is_chosen_operating_point"] for p in points),
        "cross_check_issues": issues,
        "source_files": sources,
        "tables": {"T_I7d_ttr_curve": str(curve_path),
                   "T_I7d_ttr_coverage": str(cov_path)},
    }

    if npz_path is not None:
        report["figure_data"] = str(write_npz(npz_path, points, gate, report))
    return report


def write_npz(path, points, gate, report):
    """`F_ttr_curve.npz` — the trade-off, the gate lines, the chosen point.

    Arrays are parallel and index-aligned, so a plotting script needs no
    join. `MISSING` becomes NaN HERE and only here, at the boundary with a
    numeric array that cannot hold a string; `*_is_missing` boolean arrays
    travel beside every such column so a reader can tell an absent value from
    a measured one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ("n_neurons", "top1", "top1_drop", "outlier_count_canon",
            "outlier_count_mad", "outlier_reduction_frac",
            "oversmooth_pairwise", "is_chosen_operating_point")
    arrays = {}
    for c in cols:
        vals = [p.get(c, MISSING) for p in points]
        arrays[c] = np.asarray(
            [np.nan if v is MISSING else float(v) for v in vals],
            dtype=np.float64)
        arrays[f"{c}_is_missing"] = np.asarray(
            [v is MISSING for v in vals], dtype=bool)
    for c in ("layer_range_label", "layer_range", "base_run_id", "arch",
              "recipe_actual", "passes", "source_file"):
        arrays[c] = np.asarray([str(p.get(c, MISSING)) for p in points])
    arrays["gate_min_outlier_reduction"] = np.asarray(
        [np.nan if gate["min_outlier_reduction"] is MISSING
         else float(gate["min_outlier_reduction"])])
    arrays["gate_max_top1_drop"] = np.asarray(
        [np.nan if gate["max_top1_drop"] is MISSING
         else float(gate["max_top1_drop"])])
    arrays["meta_json"] = np.asarray(json.dumps(
        {k: v for k, v in report.items() if k != "chosen_operating_points"},
        sort_keys=True, default=str))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


def main():
    p = argparse.ArgumentParser(
        description="TASK C / I7 §6 — the TTR operating curve, from "
                    "committed files only. No new TTR runs.")
    p.add_argument("--out", default=None, help="tables directory")
    p.add_argument("--npz", default=None,
                   help="also write figures_data/frozen/F_ttr_curve.npz here")
    args = p.parse_args()
    report = build(args.out, args.npz)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
