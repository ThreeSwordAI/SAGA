#!/usr/bin/env python3
"""
analysis/frozen_I3_analysis.py
==============================
TASK B B3 — I3's four tables, built from the sweep's own record rows.

    T_I3a_conditions     per checkpoint x layer x condition: mean Delta-NLL vs
                         `original`, bootstrap CI, Delta-top-1 in points, the
                         fraction of images with Delta-NLL > 0, mean/median
                         `delta_update_norm`, `max_abs_logit_diff_vs_native`
    T_I3b_contrasts      C1 and C2 per checkpoint with CI, and the
                         decision-rule outcome for the fresh pair
    T_I3c_energy_strata  Delta-NLL by `delta_update_norm` decile, per condition
                         family, per checkpoint
    T_I3d_diag_s11       the `s11_out` diagnostics under `original` vs
                         `mean_L7` / `mean_L8`: paired Delta with CI

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not decide anything. C1 and C2, the decision rule and the
interpretation branch all come from `analysis/i34_contrasts.py`, which was
committed before any evaluation number existed; this module arranges its
output into CSV. A verdict that appears in `T_I3b` and a verdict that appears
in the notes are therefore the same object, not two readings of one table.

Every row carries its SCOPE. TASK B §2: the fresh ViT-S/mixup pair decides,
the two ViT-S/mixup legacy repeats are reported beside them, and the ViT-B and
ViT-S/nomix SAGA checkpoints are exploratory — their gates have different
spatial structure (nomix has ~1.5x the spatial std), so they answer a
different question and are labelled so in every table.

NO FUNCTIONAL LANGUAGE. These tables report Delta-NLL, top-1, injected energy
and the contrasts. A test greps every generated file for the vocabulary this
project has banned.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import i34_contrasts as ic  # noqa: E402
from analysis.build_I2_tables import (BOOTSTRAP_RESAMPLES,  # noqa: E402
                                      BOOTSTRAP_SEED, fmt, paired_bootstrap,
                                      read_rows)
from saga.frozen.records import records_path  # noqa: E402

MISSING = "MISSING"
WORK_PACKAGE = "I3_gate_edits"

#: The reference condition of every I3 Delta.
REFERENCE = "original"

#: The `s11_out` diagnostics T_I3d reports, in table order. The five TASK B §3
#: names, spelled as `saga/frozen/diag.py` writes them.
DIAG_KEYS = ("count_fixed_cal", "count_mad", "cos_all", "cos_nosink_mad",
             "eff_rank")

#: Deciles (TASK B §3). Ten quantile bins of `delta_update_norm`, computed per
#: checkpoint over the POOLED edited conditions.
N_STRATA = 10

#: Condition families, in table order.
FAMILIES = ("mean", "mean_half", "perm", "ringperm", "dihedral")


class AnalysisError(ValueError):
    """A table this module refuses to build."""


# ─────────────────────────────────────────────────────────────────────────────
# Scope — which checkpoints decide, which are reported, which are exploratory
# ─────────────────────────────────────────────────────────────────────────────

def scope_of(run_id: str, arch: str, recipe: str) -> str:
    """`fresh` | `legacy` | `exploratory` for one checkpoint (TASK B §2).

    `fresh` is the two seeded ViT-S/mixup SAGA runs and nothing else: they are
    the pair LOCKED_ANALYSIS §10.1 gives the vote to. `legacy` is the rest of
    the ViT-S/mixup cell — same question, no recorded seed. `exploratory` is
    every other cell, whose gates have different spatial structure.
    """
    if run_id in ic.FRESH_SAGA:
        return "fresh"
    if (arch, recipe) == ("vit_small", "mixup"):
        return "legacy"
    return "exploratory"


def provenance(rows) -> dict:
    """The provenance every table row repeats, from the records themselves."""
    if not rows:
        raise AnalysisError("no record rows to take provenance from")
    r = rows[0]
    return {k: r.get(k, MISSING) for k in (
        "run_id", "arch", "recipe_actual", "variant", "ckpt_kind",
        "ckpt_sha256", "split_name", "split_sha256", "stage", "precision",
        "git_sha", "permutations_sha256")}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_run(results_root, run_id: str):
    """(records rows, diag rows) for one checkpoint, or (None, None).

    A run whose sweep has not landed is an ABSENCE and is reported as one;
    it never becomes a zero row.
    """
    run_dir = Path(results_root) / run_id
    rec_path = records_path(run_dir, "records")
    if not rec_path.exists():
        return None, None
    diag_path = records_path(run_dir, "diag")
    diag = read_rows(diag_path) if diag_path.exists() else []
    return read_rows(rec_path), diag


def _num(v):
    """A cell as a float, or the literal MISSING. Never a guess."""
    if v is None or v == "" or v == MISSING:
        return MISSING
    try:
        return float(v)
    except (TypeError, ValueError):
        return MISSING


def _col(grouped, cond, ids, field):
    """One field of one condition over `ids`, as a float array."""
    return np.asarray([float(grouped[cond][i][field]) for i in ids],
                      dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# T_I3a — every condition against `original`
# ─────────────────────────────────────────────────────────────────────────────

T_A_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "scope",
              "split_name", "split_sha256", "permutations_sha256",
              "condition_id", "family", "layer", "index", "n_images",
              "mean_delta_nll", "ci_lo", "ci_hi", "excludes_zero",
              "delta_top1_points", "frac_delta_nll_positive",
              "mean_delta_update_norm", "median_delta_update_norm",
              "max_abs_logit_diff_vs_native", "bit_identical_to_original",
              "bootstrap_resamples", "bootstrap_seed", "ckpt_sha256",
              "git_sha")


def build_a(run_id, records, *, resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED) -> list:
    grouped = ic.by_condition(records)
    prov = provenance(records)
    scope = scope_of(run_id, prov["arch"], prov["recipe_actual"])
    if REFERENCE not in grouped:
        raise AnalysisError(
            f"{run_id}: no {REFERENCE!r} condition — every I3 Delta is "
            f"measured against it and there is nothing to subtract")
    rows = []
    for cond in sorted(grouped):
        if cond == REFERENCE:
            continue
        parsed = ic.parse_condition_id(cond)
        if parsed is None:
            raise AnalysisError(
                f"{run_id}: condition {cond!r} is not one the I3 conditions "
                f"file declares; refusing to put an undeclared condition in a "
                f"table")
        ids, d = ic.paired_delta(grouped, cond, REFERENCE, "nll")
        _ids, dtop = ic.paired_delta(grouped, cond, REFERENCE, "correct")
        mean, lo, hi = paired_bootstrap([d], resamples=resamples, seed=seed)
        dun = [_num(grouped[cond][i].get("delta_update_norm")) for i in ids]
        dun = np.asarray([v for v in dun if v is not MISSING],
                         dtype=np.float64)
        mal = [_num(grouped[cond][i].get("max_abs_logit_diff_vs_native"))
               for i in ids]
        mal = [v for v in mal if v is not MISSING]
        rows.append(dict(
            prov, run_id=run_id, scope=scope, condition_id=cond,
            family=parsed["family"], layer=parsed["layer"],
            index=parsed["index"], n_images=len(ids),
            mean_delta_nll=mean, ci_lo=lo, ci_hi=hi,
            excludes_zero=(MISSING if mean is MISSING
                           else bool(lo > 0.0 or hi < 0.0)),
            delta_top1_points=float(dtop.mean() * 100),
            frac_delta_nll_positive=float(np.mean(d > 0)),
            mean_delta_update_norm=(MISSING if dun.size == 0
                                    else float(dun.mean())),
            median_delta_update_norm=(MISSING if dun.size == 0
                                      else float(np.median(dun))),
            max_abs_logit_diff_vs_native=(MISSING if not mal else max(mal)),
            # TASK B §7: `dihedral0` must be bit-identical to `original`. The
            # equality is MEASURED and recorded for every condition, so the
            # one place it must hold is visible beside the places it must not.
            bit_identical_to_original=bool(
                mal and max(mal) == 0.0 and float(np.abs(d).max()) == 0.0),
            bootstrap_resamples=resamples, bootstrap_seed=seed))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I3b — C1 and C2, and the decision
# ─────────────────────────────────────────────────────────────────────────────

T_B_FIELDS = ("contrast", "layer", "run_id", "arch", "recipe_actual", "scope",
              "decides", "n_images", "mean_nats", "ci_lo", "ci_hi",
              "excludes_zero", "direction", "delta_top1_points",
              "frac_positive", "verdict", "verdict_reason", "branch",
              "branch_text", "definition", "bootstrap_resamples",
              "bootstrap_seed", "split_sha256", "permutations_sha256",
              "ckpt_sha256", "git_sha")


def build_b(loaded, *, layers=(7, 8), resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED) -> tuple:
    """(rows, decisions). `loaded` is `{run_id: (records, diag)}`.

    The contrasts and the verdict come from `analysis/i34_contrasts.py`; this
    function reshapes them and adds the scope labels. It computes nothing.
    """
    records_by_run = {r: recs for r, (recs, _d) in loaded.items()
                      if recs is not None}
    if not records_by_run:
        return [], {}
    prov = {r: provenance(recs) for r, recs in records_by_run.items()}
    decisions = ic.i3_contrasts(records_by_run, layers=layers,
                                resamples=resamples, seed=seed)
    rows = []
    for layer in layers:
        d = decisions[int(layer)]
        interp = d["interpretation"]
        for name in ("C1", "C2"):
            dec = d[name]
            for run_id in sorted(dec["per_checkpoint"]):
                s = dec["per_checkpoint"][run_id]
                p = prov[run_id]
                rows.append({
                    "contrast": name, "layer": int(layer), "run_id": run_id,
                    "arch": p["arch"], "recipe_actual": p["recipe_actual"],
                    "scope": scope_of(run_id, p["arch"], p["recipe_actual"]),
                    "decides": run_id in dec["deciding"],
                    "n_images": s["n_images"], "mean_nats": s["mean_nats"],
                    "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                    "excludes_zero": s["excludes_zero"],
                    "direction": s["direction"],
                    "delta_top1_points": s["delta_top1_points"],
                    "frac_positive": s["frac_positive"],
                    "verdict": dec["verdict"],
                    "verdict_reason": dec["reason"],
                    "branch": interp["branch"],
                    "branch_text": interp["text"],
                    "definition": s["definition"],
                    "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                    "split_sha256": p["split_sha256"],
                    "permutations_sha256": p["permutations_sha256"],
                    "ckpt_sha256": p["ckpt_sha256"], "git_sha": p["git_sha"],
                })
    return rows, decisions


# ─────────────────────────────────────────────────────────────────────────────
# T_I3c — the energy strata
# ─────────────────────────────────────────────────────────────────────────────

T_C_FIELDS = ("run_id", "arch", "recipe_actual", "scope", "layer", "family",
              "decile", "edge_lo", "edge_hi", "n_pairs", "mean_delta_nll",
              "ci_lo", "ci_hi", "mean_delta_update_norm",
              "bootstrap_resamples", "bootstrap_seed", "split_sha256",
              "ckpt_sha256", "git_sha")


def decile_edges(values, n_strata: int = N_STRATA) -> np.ndarray:
    """The `n_strata`-quantile edges of `values`, deterministically.

    `np.quantile(..., method="linear")` on the SORTED pooled values, with the
    two outer edges pushed to -inf / +inf so that every observation falls in
    exactly one bin and none is dropped at the boundary. The bins are
    half-open [lo, hi) with the last one closed, which `np.searchsorted` with
    `side="right"` implements exactly.

    Deterministic by construction: no sampling, no tie-breaking rule that
    depends on input order (the quantiles are computed on a sorted copy), so
    a re-run on the same records gives the same edges.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        raise AnalysisError("no delta_update_norm values to stratify")
    qs = np.linspace(0.0, 1.0, int(n_strata) + 1)
    edges = np.quantile(arr, qs, method="linear")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_decile(values, edges) -> np.ndarray:
    """The 0-based decile of every value, given `decile_edges` output."""
    idx = np.searchsorted(np.asarray(edges)[1:-1],
                          np.asarray(values, dtype=np.float64), side="right")
    return np.clip(idx, 0, len(edges) - 2)


def build_c(run_id, records, *, n_strata=N_STRATA,
            resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> list:
    """Delta-NLL by `delta_update_norm` decile, per family, per layer.

    The deciles are computed ONCE per checkpoint over the POOLED edited
    conditions (TASK B §3), so `mean`, `mean_half`, `perm`, `ringperm` and
    `dihedral` are read on ONE energy axis. Deciles computed per family would
    put each family's own median in bin 5 and make the comparison vacuous.

    The unit of a stratum row is an (image, condition) PAIR, not an image: one
    image contributes to several deciles because its conditions inject
    different energies. The bootstrap therefore resamples pairs and the column
    is named `n_pairs`, so nobody reads it as an image count.
    """
    grouped = ic.by_condition(records)
    prov = provenance(records)
    scope = scope_of(run_id, prov["arch"], prov["recipe_actual"])

    pooled_vals, pooled = [], []
    for cond in sorted(grouped):
        if cond == REFERENCE:
            continue
        parsed = ic.parse_condition_id(cond)
        if parsed is None:
            continue
        ids, d = ic.paired_delta(grouped, cond, REFERENCE, "nll")
        dun = np.asarray(
            [_num(grouped[cond][i].get("delta_update_norm")) for i in ids],
            dtype=object)
        keep = np.asarray([v is not MISSING for v in dun])
        if not keep.any():
            continue
        vals = dun[keep].astype(np.float64)
        pooled_vals.append(vals)
        pooled.append((parsed, d[keep], vals))
    if not pooled_vals:
        raise AnalysisError(
            f"{run_id}: no `delta_update_norm` on any edited condition — the "
            f"energy strata cannot be built without it. Was the sweep run "
            f"with `measure_update_norm: true`?")
    edges = decile_edges(np.concatenate(pooled_vals), n_strata)

    rows = []
    for layer in sorted({p["layer"] for p, _d, _v in pooled}):
        for family in FAMILIES:
            sel = [(d, v) for p, d, v in pooled
                   if p["layer"] == layer and p["family"] == family]
            if not sel:
                continue
            d_all = np.concatenate([d for d, _v in sel])
            v_all = np.concatenate([v for _d, v in sel])
            bins = assign_decile(v_all, edges)
            for k in range(int(n_strata)):
                m = bins == k
                if not m.any():
                    continue
                mean, lo, hi = paired_bootstrap([d_all[m]],
                                                resamples=resamples, seed=seed)
                rows.append(dict(
                    prov, run_id=run_id, scope=scope, layer=int(layer),
                    family=family, decile=k,
                    edge_lo=(MISSING if k == 0 else float(edges[k])),
                    edge_hi=(MISSING if k == n_strata - 1
                             else float(edges[k + 1])),
                    n_pairs=int(m.sum()), mean_delta_nll=mean,
                    ci_lo=lo, ci_hi=hi,
                    mean_delta_update_norm=float(v_all[m].mean()),
                    bootstrap_resamples=resamples, bootstrap_seed=seed))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I3d — the s11_out diagnostics under the mean collapse
# ─────────────────────────────────────────────────────────────────────────────

T_D_FIELDS = ("run_id", "arch", "recipe_actual", "scope", "stage",
              "condition_id", "layer", "diagnostic", "n_images",
              "mean_original", "mean_condition", "mean_delta", "ci_lo",
              "ci_hi", "excludes_zero", "tau_cal_value",
              "bootstrap_resamples", "bootstrap_seed", "split_sha256",
              "ckpt_sha256", "git_sha")


def build_d(run_id, records, diag, *, resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED) -> list:
    """Paired Delta of each `s11_out` diagnostic, `mean_L{l}` vs `original`.

    Reads the `diag` rows the sweep wrote for the three conditions the
    conditions file marks `diag: true`. A diagnostic that is the literal
    MISSING on either side is skipped for that image, never averaged
    (I0 handoff §8.2).
    """
    if not diag:
        return []
    prov = provenance(records) if records else {}
    by = {}
    for r in diag:
        by.setdefault((str(r["condition_id"]), str(r["stage"])), {})[
            str(r["image_id"])] = r
    stages = sorted({s for _c, s in by})
    rows = []
    for stage in stages:
        ref = by.get((REFERENCE, stage))
        if not ref:
            continue
        for cond, st in sorted(by):
            if st != stage or cond == REFERENCE:
                continue
            got = by[(cond, st)]
            ids = sorted(set(got) & set(ref))
            parsed = ic.parse_condition_id(cond)
            for key in DIAG_KEYS:
                a = [_num(got[i].get(key)) for i in ids]
                b = [_num(ref[i].get(key)) for i in ids]
                pairs = [(x, y) for x, y in zip(a, b)
                         if x is not MISSING and y is not MISSING]
                if not pairs:
                    rows.append(dict(
                        prov, run_id=run_id, stage=stage, condition_id=cond,
                        layer=(MISSING if parsed is None else parsed["layer"]),
                        diagnostic=key, n_images=0, mean_original=MISSING,
                        mean_condition=MISSING, mean_delta=MISSING,
                        ci_lo=MISSING, ci_hi=MISSING, excludes_zero=MISSING,
                        scope=scope_of(run_id, prov.get("arch", MISSING),
                                       prov.get("recipe_actual", MISSING)),
                        tau_cal_value=_num(got[ids[0]].get("tau_cal_value"))
                        if ids else MISSING,
                        bootstrap_resamples=resamples, bootstrap_seed=seed))
                    continue
                x = np.asarray([p[0] for p in pairs], dtype=np.float64)
                y = np.asarray([p[1] for p in pairs], dtype=np.float64)
                mean, lo, hi = paired_bootstrap([x - y], resamples=resamples,
                                                seed=seed)
                rows.append(dict(
                    prov, run_id=run_id, stage=stage, condition_id=cond,
                    layer=(MISSING if parsed is None else parsed["layer"]),
                    diagnostic=key, n_images=len(pairs),
                    mean_original=float(y.mean()),
                    mean_condition=float(x.mean()), mean_delta=mean,
                    ci_lo=lo, ci_hi=hi,
                    excludes_zero=(MISSING if mean is MISSING
                                   else bool(lo > 0.0 or hi < 0.0)),
                    scope=scope_of(run_id, prov.get("arch", MISSING),
                                   prov.get("recipe_actual", MISSING)),
                    tau_cal_value=_num(got[ids[0]].get("tau_cal_value")),
                    bootstrap_resamples=resamples, bootstrap_seed=seed))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, MISSING)) for k in fields})
    return path


def main():
    from saga.run_registry import git_sha

    p = argparse.ArgumentParser(description="Build TASK B / I3's four tables.")
    p.add_argument("--results",
                   default=f"results/frozen/{WORK_PACKAGE}/evaluation",
                   help="the sweep directory for ONE split")
    p.add_argument("--out", default=None, help="default: <results>/tables")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED,
                   help="bootstrap seed (LOCKED_ANALYSIS D6; default 0)")
    p.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    p.add_argument("--strata", type=int, default=N_STRATA)
    args = p.parse_args()

    from saga.frozen.runner import eligible_cohort
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "tables"

    cohort = [r for r in eligible_cohort(args.manifest)
              if r["variant"] == "saga"]
    loaded, absent = {}, []
    for r in cohort:
        recs, diag = load_run(results, r["run_id"])
        if recs is None:
            absent.append(r["run_id"])
            continue
        loaded[r["run_id"]] = (recs, diag)
    if not loaded:
        print(f"no I3 records under {results} — nothing to build "
              f"({len(absent)} run(s) absent)", file=sys.stderr)
        return 1

    rows_a, rows_c, rows_d = [], [], []
    for run_id, (recs, diag) in sorted(loaded.items()):
        rows_a += build_a(run_id, recs, resamples=args.resamples,
                          seed=args.seed)
        rows_c += build_c(run_id, recs, n_strata=args.strata,
                          resamples=args.resamples, seed=args.seed)
        rows_d += build_d(run_id, recs, diag, resamples=args.resamples,
                          seed=args.seed)
    rows_b, decisions = build_b(loaded, resamples=args.resamples,
                                seed=args.seed)

    write_csv(out / "T_I3a_conditions.csv", T_A_FIELDS, rows_a)
    write_csv(out / "T_I3b_contrasts.csv", T_B_FIELDS, rows_b)
    write_csv(out / "T_I3c_energy_strata.csv", T_C_FIELDS, rows_c)
    write_csv(out / "T_I3d_diag_s11.csv", T_D_FIELDS, rows_d)

    meta = {
        "work_package": WORK_PACKAGE, "results": str(results),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "generated_by": "analysis/frozen_I3_analysis.py",
        "git_sha": git_sha(), "bootstrap_seed": args.seed,
        "bootstrap_resamples": args.resamples, "n_strata": args.strata,
        "runs_loaded": sorted(loaded), "runs_absent": sorted(absent),
        "decisions": {str(k): {"C1": v["C1"]["verdict"],
                               "C2": v["C2"]["verdict"],
                               "branch": v["interpretation"]["branch"],
                               "branch_text": v["interpretation"]["text"]}
                      for k, v in decisions.items()},
    }
    (out / "I3_tables_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n")
    print(f"wrote 4 tables to {out}")
    for k, v in sorted(meta["decisions"].items()):
        print(f"  layer {k}: C1={v['C1']}  C2={v['C2']}  -> {v['branch']}")
    if absent:
        print(f"  ABSENT (reported, never zeroed): {sorted(absent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
