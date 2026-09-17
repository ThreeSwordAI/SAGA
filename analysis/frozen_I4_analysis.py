#!/usr/bin/env python3
"""
analysis/frozen_I4_analysis.py
==============================
TASK B B7 — I4's four tables, built from the sweep's own record rows.

    T_I4a_theta         per checkpoint x layer x eps x mask: theta = mean
                        paired Delta-NLL vs `native`, CI, Delta-top-1, mean
                        injected norm, mean `energy_rel_error`
    T_I4b_contrast      C3 per checkpoint: theta(primary) - mean theta(matched
                        controls) with CI; the CONTROL BAND (min, max of the
                        10 theta_control); the decision-rule outcome for the
                        fresh baselines and the fresh SAGA checkpoints
                        SEPARATELY; the unmatched controls beside them
    T_I4c_cross_method  on the common mask at matched energy: theta(primary)
                        for baseline (4), SAGA (4), registers (2); the
                        paired-by-seed baseline - SAGA difference where the
                        seeds match; registers n = 2, labelled
    T_I4d_diag_s11      the `s11_out` diagnostics under `prim_e10_L7/L8` vs
                        `native`: paired Delta with CI

As in I3, this module DECIDES NOTHING. C3, the decision rule and the
interpretation branch come from `analysis/i34_contrasts.py`, which was
committed before any evaluation number existed.

THE TWO THINGS THAT MUST BE READ BEFORE ANY THETA IS BELIEVED, and which
T_I4a puts in its own columns rather than a footnote:

  * `energy_rel_error` — the matched controls are only controls if they
    actually injected the energy they were asked for. TASK B §5 fixes the
    tolerance at 1%; the maximum observed is reported per checkpoint and a
    row above tolerance is FLAGGED, never quietly averaged.
  * the UNMATCHED controls (`*_e10f_*`) — reported beside the matched ones,
    because the difference between the two is the size of the energy
    confound, which this task reports rather than removes silently.

NO FUNCTIONAL LANGUAGE. These tables report Delta-NLL, top-1, injected energy
and the contrasts.
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
from analysis.frozen_I3_analysis import (DIAG_KEYS, _num, build_d,  # noqa: E402
                                         load_run, provenance, write_csv)
from saga.frozen.records import records_path  # noqa: E402

MISSING = "MISSING"
WORK_PACKAGE = "I4_perturbation"

#: The reference condition of every I4 theta.
REFERENCE = ic.I4_REFERENCE

#: The epsilons the sweep declares, primary first (LOCKED §6: 0.10 is the
#: operating point, 0.25 is a stability test and NOT one to optimise).
EPSILONS = ("10", "25")

#: The control regimes, and how each is labelled in the tables.
KINDS = {ic.I4_MATCHED: "energy_matched", ic.I4_FIXED: "fixed_epsilon"}

#: TASK B §5 fixes the per-image energy tolerance.
ENERGY_TOL = 0.01


class AnalysisError(ValueError):
    """A table this module refuses to build."""


def scope_of(run_id: str, variant: str) -> str:
    """`fresh` | `legacy` | `registers` for one I4 checkpoint (TASK B §2).

    I4 runs one cell, so the I3 `exploratory` label has no members here; what
    replaces it is `registers`, which is n = 2 and never decides.
    """
    if variant == "registers":
        return "registers"
    if run_id in ic.FRESH_BASELINE or run_id in ic.FRESH_SAGA:
        return "fresh"
    return "legacy"


def _condition_meta(cond: str) -> dict:
    """`{role, layer, eps, kind, j}` for an I4 condition id, or None."""
    if cond == REFERENCE:
        return {"role": "native", "layer": MISSING, "eps": MISSING,
                "kind": MISSING, "j": MISSING}
    if "_L" not in cond:
        return None
    head, _, tail = cond.rpartition("_L")
    if not tail.isdigit():
        return None
    layer = int(tail)
    if head.startswith("prim_e"):
        return {"role": "primary", "layer": layer, "eps": head[6:],
                "kind": "fixed_epsilon", "j": MISSING}
    if head.startswith("ctrl"):
        body = head[4:]
        j, _, rest = body.partition("_e")
        if j.isdigit() and rest[:-1].isdigit() and rest[-1:] in KINDS:
            return {"role": "control", "layer": layer, "eps": rest[:-1],
                    "kind": KINDS[rest[-1]], "j": int(j)}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# T_I4a — every condition against `native`
# ─────────────────────────────────────────────────────────────────────────────

T_A_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "scope",
              "split_name", "split_sha256", "masks_sha256", "condition_id",
              "role", "layer", "epsilon", "control_kind", "control_index",
              "mask_id", "n_masked_coords", "n_images", "theta_nats",
              "ci_lo", "ci_hi", "excludes_zero", "delta_top1_points",
              "frac_theta_positive", "mean_injected_norm",
              "mean_energy_rel_error", "max_energy_rel_error",
              "energy_within_tolerance", "n_zero_norm_images",
              "max_abs_logit_diff_vs_native", "bootstrap_resamples",
              "bootstrap_seed", "ckpt_sha256", "git_sha")


def build_a(run_id, records, *, resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED, tol=ENERGY_TOL) -> list:
    grouped = ic.by_condition(records)
    prov = provenance(records)
    prov.setdefault("masks_sha256", MISSING)
    scope = scope_of(run_id, prov.get("variant", MISSING))
    if REFERENCE not in grouped:
        raise AnalysisError(
            f"{run_id}: no {REFERENCE!r} condition — every I4 theta is "
            f"measured against it and there is nothing to subtract")
    rows = []
    for cond in sorted(grouped):
        if cond == REFERENCE:
            continue
        meta = _condition_meta(cond)
        if meta is None:
            raise AnalysisError(
                f"{run_id}: condition {cond!r} is not one the I4 conditions "
                f"file declares; refusing to put an undeclared condition in "
                f"a table")
        ids, d = ic.paired_delta(grouped, cond, REFERENCE, "nll")
        _ids, dtop = ic.paired_delta(grouped, cond, REFERENCE, "correct")
        mean, lo, hi = paired_bootstrap([d], resamples=resamples, seed=seed)

        def _col(key):
            vals = [_num(grouped[cond][i].get(key)) for i in ids]
            return [v for v in vals if v is not MISSING]

        injected = _col("measured_perturbation_norm")
        rel = _col("energy_rel_error")
        mal = _col("max_abs_logit_diff_vs_native")
        first = grouped[cond][ids[0]]
        rows.append(dict(
            prov, run_id=run_id, scope=scope, condition_id=cond,
            role=meta["role"], layer=meta["layer"], epsilon=meta["eps"],
            control_kind=meta["kind"], control_index=meta["j"],
            mask_id=first.get("mask_id", MISSING),
            n_masked_coords=first.get("n_masked_coords", MISSING),
            n_images=len(ids), theta_nats=mean, ci_lo=lo, ci_hi=hi,
            excludes_zero=(MISSING if mean is MISSING
                           else bool(lo > 0.0 or hi < 0.0)),
            delta_top1_points=float(dtop.mean() * 100),
            frac_theta_positive=float(np.mean(d > 0)),
            mean_injected_norm=(MISSING if not injected
                                else float(np.mean(injected))),
            mean_energy_rel_error=(MISSING if not rel
                                   else float(np.mean(rel))),
            max_energy_rel_error=(MISSING if not rel else float(max(rel))),
            # A control that did not inject what it was asked for is not a
            # control. The flag is a VALUE in the table, not a filter.
            energy_within_tolerance=(MISSING if not rel
                                     else bool(max(rel) < tol)),
            n_zero_norm_images=(MISSING if not injected
                                else int(sum(1 for v in injected if v == 0.0))),
            max_abs_logit_diff_vs_native=(MISSING if not mal else max(mal)),
            bootstrap_resamples=resamples, bootstrap_seed=seed))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I4b — C3, the control band, and the decision
# ─────────────────────────────────────────────────────────────────────────────

T_B_FIELDS = ("contrast", "layer", "epsilon", "control_kind", "run_id",
              "arch", "variant", "scope", "method", "decides", "n_images",
              "theta_primary", "theta_control_mean", "control_band_lo",
              "control_band_hi", "primary_inside_band", "mean_nats", "ci_lo",
              "ci_hi", "excludes_zero", "direction", "delta_top1_points",
              "verdict", "verdict_reason", "branch", "branch_text",
              "definition", "bootstrap_resamples", "bootstrap_seed",
              "split_sha256", "masks_sha256", "ckpt_sha256", "git_sha")


def build_b(loaded, *, layers=(7, 8), resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED) -> tuple:
    """(rows, decisions). Matched controls decide; unmatched are reported."""
    records_by_run = {r: recs for r, (recs, _d) in loaded.items()
                      if recs is not None}
    if not records_by_run:
        return [], {}
    prov = {r: provenance(recs) for r, recs in records_by_run.items()}
    decisions = ic.i4_contrasts(records_by_run, layers=layers, eps="10",
                                kind=ic.I4_MATCHED, resamples=resamples,
                                seed=seed)
    rows = []
    for layer in layers:
        d = decisions[int(layer)]
        interp = d["interpretation"]
        for method, dec in sorted(d["C3"].items()):
            for run_id in sorted(dec["per_checkpoint"]):
                s = dec["per_checkpoint"][run_id]
                p = prov[run_id]
                rows.append({
                    "contrast": "C3", "layer": int(layer), "epsilon": "10",
                    "control_kind": KINDS[ic.I4_MATCHED], "run_id": run_id,
                    "arch": p["arch"], "variant": p["variant"],
                    "scope": scope_of(run_id, p["variant"]), "method": method,
                    "decides": run_id in dec["deciding"],
                    "n_images": s["n_images"],
                    "theta_primary": s["theta_primary"],
                    "theta_control_mean": s["theta_control_mean"],
                    "control_band_lo": s["control_band_lo"],
                    "control_band_hi": s["control_band_hi"],
                    "primary_inside_band": s["primary_inside_band"],
                    "mean_nats": s["mean_nats"], "ci_lo": s["ci_lo"],
                    "ci_hi": s["ci_hi"],
                    "excludes_zero": s["excludes_zero"],
                    "direction": s["direction"],
                    "delta_top1_points": s["delta_top1_points"],
                    "verdict": dec["verdict"],
                    "verdict_reason": dec["reason"],
                    "branch": interp["branch"],
                    "branch_text": interp["text"],
                    "definition": s["definition"],
                    "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                    "split_sha256": p["split_sha256"],
                    "masks_sha256": p.get("masks_sha256", MISSING),
                    "ckpt_sha256": p["ckpt_sha256"], "git_sha": p["git_sha"],
                })

        # The UNMATCHED controls, beside the matched ones (TASK B §6). They
        # do not decide; the gap between the two rows is the energy confound.
        grouped = {r: ic.by_condition(recs)
                   for r, recs in records_by_run.items()}
        for run_id in sorted(grouped):
            try:
                s = ic.c3(grouped[run_id], layer, eps="10",
                          kind=ic.I4_FIXED, resamples=resamples, seed=seed)
            except ic.ContrastError:
                continue
            p = prov[run_id]
            rows.append({
                "contrast": "C3_unmatched", "layer": int(layer),
                "epsilon": "10", "control_kind": KINDS[ic.I4_FIXED],
                "run_id": run_id, "arch": p["arch"], "variant": p["variant"],
                "scope": scope_of(run_id, p["variant"]),
                "method": p["variant"], "decides": False,
                "n_images": s["n_images"],
                "theta_primary": s["theta_primary"],
                "theta_control_mean": s["theta_control_mean"],
                "control_band_lo": s["control_band_lo"],
                "control_band_hi": s["control_band_hi"],
                "primary_inside_band": s["primary_inside_band"],
                "mean_nats": s["mean_nats"], "ci_lo": s["ci_lo"],
                "ci_hi": s["ci_hi"], "excludes_zero": s["excludes_zero"],
                "direction": s["direction"],
                "delta_top1_points": s["delta_top1_points"],
                "verdict": "reported", "verdict_reason":
                    "unmatched controls never decide; the gap to the matched "
                    "row is the size of the energy confound",
                "branch": MISSING, "branch_text": MISSING,
                "definition": s["definition"],
                "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                "split_sha256": p["split_sha256"],
                "masks_sha256": p.get("masks_sha256", MISSING),
                "ckpt_sha256": p["ckpt_sha256"], "git_sha": p["git_sha"],
            })
    return rows, decisions


# ─────────────────────────────────────────────────────────────────────────────
# T_I4c — the cross-method row
# ─────────────────────────────────────────────────────────────────────────────

T_C_FIELDS = ("layer", "epsilon", "comparison", "method", "run_id",
              "provenance_tag", "scope", "n", "theta_primary", "ci_lo",
              "ci_hi", "paired_with", "paired_difference", "note",
              "bootstrap_resamples", "bootstrap_seed", "git_sha")


def _tag_of(run_id: str) -> str:
    """The seed/provenance tag a run pairs on: `s1`, `s2`, or the legacy dir."""
    for tag in ("s1", "s2"):
        if run_id.endswith("_" + tag):
            return tag
    if "mixupdir" in run_id:
        return "legacy-mixupdir"
    if "nomixdir" in run_id:
        return "legacy-nomixdir"
    return MISSING


def build_c(loaded, *, layers=(7, 8), resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED) -> list:
    """theta(primary) per method on the common mask, paired by seed.

    The PAIRING is on `provenance_tag` — `s1` with `s1`, `legacy-mixupdir`
    with `legacy-mixupdir` — so a baseline and a SAGA checkpoint are compared
    only when they are the same training run modulo the method. Registers are
    n = 2 and are listed with that stated on every row; nothing pools them
    with anything.
    """
    records_by_run = {r: recs for r, (recs, _d) in loaded.items()
                      if recs is not None}
    grouped = {r: ic.by_condition(recs) for r, recs in records_by_run.items()}
    prov = {r: provenance(recs) for r, recs in records_by_run.items()}
    rows = []
    for layer in layers:
        thetas = {}
        for run_id in sorted(grouped):
            cond = ic.I4_PRIMARY.format(eps="10", layer=int(layer))
            if cond not in grouped[run_id]:
                continue
            thetas[run_id] = ic.theta(grouped[run_id], cond,
                                      resamples=resamples, seed=seed)
        by_method = {}
        for run_id, t in thetas.items():
            by_method.setdefault(prov[run_id]["variant"], []).append(run_id)
        for method in sorted(by_method):
            members = sorted(by_method[method])
            for run_id in members:
                t = thetas[run_id]
                rows.append({
                    "layer": int(layer), "epsilon": "10",
                    "comparison": "per_checkpoint", "method": method,
                    "run_id": run_id, "provenance_tag": _tag_of(run_id),
                    "scope": scope_of(run_id, method), "n": len(members),
                    "theta_primary": t["mean_nats"], "ci_lo": t["ci_lo"],
                    "ci_hi": t["ci_hi"], "paired_with": MISSING,
                    "paired_difference": MISSING,
                    "note": ("registers: n = 2, labelled, never pooled"
                             if method == "registers" else ""),
                    "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                    "git_sha": prov[run_id]["git_sha"]})
        # baseline - SAGA, paired where the provenance tag matches
        base = {_tag_of(r): r for r in by_method.get("baseline", [])}
        saga = {_tag_of(r): r for r in by_method.get("saga", [])}
        for tag in sorted(set(base) & set(saga) - {MISSING}):
            b, s = thetas[base[tag]], thetas[saga[tag]]
            if b["mean_nats"] is MISSING or s["mean_nats"] is MISSING:
                continue
            rows.append({
                "layer": int(layer), "epsilon": "10",
                "comparison": "baseline_minus_saga", "method": "paired",
                "run_id": base[tag], "provenance_tag": tag,
                "scope": scope_of(base[tag], "baseline"), "n": 1,
                "theta_primary": b["mean_nats"], "ci_lo": MISSING,
                "ci_hi": MISSING, "paired_with": saga[tag],
                "paired_difference": float(b["mean_nats"] - s["mean_nats"]),
                "note": "same provenance tag, same images, same coordinates",
                "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                "git_sha": prov[base[tag]]["git_sha"]})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from saga.run_registry import git_sha

    p = argparse.ArgumentParser(description="Build TASK B / I4's four tables.")
    p.add_argument("--results",
                   default=f"results/frozen/{WORK_PACKAGE}/evaluation")
    p.add_argument("--out", default=None, help="default: <results>/tables")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = p.parse_args()

    from saga.frozen.runner import eligible_cohort
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "tables"

    cohort = [r for r in eligible_cohort(args.manifest)
              if (r["arch"], r["recipe_actual"]) == ("vit_small", "mixup")]
    loaded, absent = {}, []
    for r in cohort:
        recs, diag = load_run(results, r["run_id"])
        if recs is None:
            absent.append(r["run_id"])
            continue
        loaded[r["run_id"]] = (recs, diag)
    if not loaded:
        print(f"no I4 records under {results} — nothing to build "
              f"({len(absent)} run(s) absent)", file=sys.stderr)
        return 1

    rows_a, rows_d = [], []
    for run_id, (recs, diag) in sorted(loaded.items()):
        rows_a += build_a(run_id, recs, resamples=args.resamples,
                          seed=args.seed)
        rows_d += build_d(run_id, recs, diag, resamples=args.resamples,
                          seed=args.seed)
    rows_b, decisions = build_b(loaded, resamples=args.resamples,
                                seed=args.seed)
    rows_c = build_c(loaded, resamples=args.resamples, seed=args.seed)

    from analysis.frozen_I3_analysis import T_D_FIELDS
    write_csv(out / "T_I4a_theta.csv", T_A_FIELDS, rows_a)
    write_csv(out / "T_I4b_contrast.csv", T_B_FIELDS, rows_b)
    write_csv(out / "T_I4c_cross_method.csv", T_C_FIELDS, rows_c)
    write_csv(out / "T_I4d_diag_s11.csv", T_D_FIELDS, rows_d)

    worst = [r["max_energy_rel_error"] for r in rows_a
             if r["max_energy_rel_error"] is not MISSING]
    meta = {
        "work_package": WORK_PACKAGE, "results": str(results),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "generated_by": "analysis/frozen_I4_analysis.py",
        "git_sha": git_sha(), "bootstrap_seed": args.seed,
        "bootstrap_resamples": args.resamples,
        "energy_rel_error_tol": ENERGY_TOL,
        "energy_rel_error_max": (MISSING if not worst else max(worst)),
        "runs_loaded": sorted(loaded), "runs_absent": sorted(absent),
        "decisions": {str(k): {m: d["verdict"] for m, d in v["C3"].items()}
                      | {"branch": v["interpretation"]["branch"],
                         "branch_text": v["interpretation"]["text"]}
                      for k, v in decisions.items()},
    }
    (out / "I4_tables_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n")
    print(f"wrote 4 tables to {out}")
    for k, v in sorted(meta["decisions"].items()):
        print(f"  layer {k}: baseline={v['baseline']}  saga={v['saga']}  "
              f"-> {v['branch']}")
    print(f"  max energy_rel_error: {meta['energy_rel_error_max']} "
          f"(tolerance {ENERGY_TOL})")
    if absent:
        print(f"  ABSENT (reported, never zeroed): {sorted(absent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
