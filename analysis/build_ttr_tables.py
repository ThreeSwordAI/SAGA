#!/usr/bin/env python3
"""
analysis/build_ttr_tables.py
============================
TASK-10 PHASE C.1 — `results/tables/T_ttr.csv` and `T_ttr_sweep.csv`.

Per base cell: TTR's top-1 and its delta against BOTH the unpatched baseline
and SAGA, the sink count under the base cell's canon tau, both oversmoothing
variants, effective rank, CLS-norm ratio, and the address-map correlation
against that cell's own baseline.

Three conventions, all from the human's Phase-C decisions:

1. **Every cell appears with its own verdict**, passing or failing. Dropping
   the failures would be selection on outcome.
2. **Absolute sink counts sit beside every percentage.** 42% of 3.70 sinks
   is ~1.5 tokens; 55% of 19.71 is ~10.8. The percentages are not comparable
   across cells and must never be read as if they were.
3. **The A3 threshold is frozen at 1.00 and is never recomputed here.** The
   verdict is whatever `tools/ttr_validate.py` recorded. Where
   `paired_ci.json` exists its CI is carried through, as a statement about
   measurement precision only.

Pairing is by CHECKPOINT SHA256, never by filename: the TTR run records the
sha of the base checkpoint it patched, and the unpatched eval/diag/addr for
that cell are the ones carrying the same sha. Each cell resolves to exactly
one of each, or the row is MISSING.

    python analysis/build_ttr_tables.py
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

#: (ttr_run_id, base root for the unpatched artifacts, arch, recipe, label)
CELLS = [
    ("e2r_vits_mixup_baseline_s1", "results/runs/e2r_vits_mixup_baseline_s1",
     "vit_small", "mixup", "ViT-S/mixup (e2r s1)"),
    ("e2r_vitb_mixup_baseline_s1", "results/runs/e2r_vitb_mixup_baseline_s1",
     "vit_base", "mixup", "ViT-B/mixup (e2r s1)"),
    ("e2r_vits_nomix_baseline_s1", "results/runs/e2r_vits_nomix_baseline_s1",
     "vit_small", "nomix", "ViT-S/nomix (e2r s1)"),
    ("legacy_vits_baseline", "results/legacy",
     "vit_small", "mixup", "ViT-S/mixup (legacy)"),
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def find_by_sha(root: Path, subdir: str, sha: str, *, suffix=".json",
                exclude=("_addr.json", "_normstats.json")):
    """The unique artifact under root/subdir whose ckpt_sha256 == sha."""
    hits = []
    for f in sorted((root / subdir).glob(f"*{suffix}")):
        if any(f.name.endswith(x) for x in exclude):
            continue
        try:
            if _load(f).get("ckpt_sha256") == sha:
                hits.append(f)
        except (json.JSONDecodeError, OSError):
            continue
    if len(hits) != 1:
        return None, (f"{len(hits)} artifacts in {root/subdir} match sha "
                      f"{sha[:12]} (need exactly 1)")
    return hits[0], None


def find_addr_by_sha(root: Path, sha: str):
    hits = [f for f in sorted((root / "diag").glob("*_addr.json"))
            if _load(f).get("ckpt_sha256") == sha]
    if len(hits) != 1:
        return None, f"{len(hits)} addr maps match sha {sha[:12]}"
    return hits[0], None


def addr_rho(a, b):
    """(rho, p_spatial, null_sd) using TASK-07's exact permutation null.

    NOT an iid Spearman reference. These maps are spatially smooth, so the
    iid null (sd 0.0716 at n=196) is far too generous; TASK-07 established
    the exact null over the 8 dihedral x 196 torus-roll transforms
    (1568 of them, deterministic, no seed), whose sd is 0.157-0.342, i.e.
    n_eff ~ 10-42. Reusing that code rather than re-implementing a weaker
    one is the point: a raw |rho| of ~0.1-0.3 is NOT distinguishable from
    zero under the correct null.
    """
    from analysis.address_analysis import rho_spatial
    res = rho_spatial(np.asarray(a, float), np.asarray(b, float))
    if res is None:
        return MISSING, MISSING, MISSING
    r, p, sd, _ = res
    return (r,
            p if p is not None else MISSING,
            sd if sd is not None else MISSING)


def saga_reference(pooled_rows, arch, recipe):
    """The SAGA cell mean from e2_pooled.csv (kind == 'mean')."""
    for r in pooled_rows:
        if (r["arch"] == arch and r["recipe_actual"] == recipe
                and r["variant"] == "saga" and r["kind"] == "mean"):
            return r
    return None


def build_rows(repo: Path):
    pooled_path = repo / "results" / "tables" / "e2_pooled.csv"
    pooled = list(csv.DictReader(open(pooled_path))) if pooled_path.exists() else []

    rows, notes = [], []
    for run_id, base_root, arch, recipe, label in CELLS:
        ttr_dir = repo / "results" / "runs" / f"ttr_{run_id}"
        gate_dir = repo / "results" / "ttr_midlayer" / run_id
        row = {"cell": label, "base_run_id": run_id, "arch": arch,
               "recipe_actual": recipe}

        if not (ttr_dir / "eval" / "eval_last.json").exists():
            notes.append(f"{label}: no TTR eval — cell omitted")
            continue
        t_ev = _load(ttr_dir / "eval" / "eval_last.json")
        t_dg = _load(ttr_dir / "diag" / "diag_last.json")
        sha = t_ev["ckpt_sha256"]

        # ── the gate's own verdict, never recomputed here ────────────────
        v = _load(gate_dir / "validate.json")
        ver = v["verdict"]
        row.update(
            n_neurons=t_ev["ttr"]["n_neurons"],
            layer_range=",".join(str(x) for x in v["scan_stats"]["layer_range"]),
            layers_touched=",".join(str(l) for l in t_dg["ttr"]["layers_touched"]),
            canon_tau=v["canon_tau"],
            tau_recalibrated=v["tau_recalibrated_on_patched_model"],
            gate_verdict="PASS" if ver["passed"] else "FAIL",
            gate_best_n=ver["best_n_neurons"] if ver["best_n_neurons"] else MISSING,
            gate_n_passing=ver["n_passing"],
            gate_n_swept=len(v["sweep"]),
            gate_min_sink_reduction=ver["min_sink_reduction"],
            gate_max_top1_drop=ver["max_top1_drop"],
            ckpt_sha256=sha,
        )

        # ── unpatched baseline, resolved by sha ──────────────────────────
        base = repo / base_root
        b_ev_p, e1 = find_by_sha(base, "eval", sha)
        b_dg_p, e2 = find_by_sha(base, "diag", sha)
        b_ad_p, e3 = find_addr_by_sha(base, sha)
        for e in (e1, e2, e3):
            if e:
                notes.append(f"{label}: {e}")
        b_ev = _load(b_ev_p) if b_ev_p else None
        b_dg = _load(b_dg_p) if b_dg_p else None

        # ── top-1: TTR, baseline, delta ──────────────────────────────────
        row["top1_ttr"] = t_ev["top1"]
        row["top5_ttr"] = t_ev["top5"]
        row["n_images_eval"] = t_ev["n_images"]
        row["top1_baseline"] = b_ev["top1"] if b_ev else MISSING
        row["top1_delta_vs_baseline"] = (
            t_ev["top1"] - b_ev["top1"] if b_ev else MISSING)
        row["top1_drop_vs_baseline"] = (
            b_ev["top1"] - t_ev["top1"] if b_ev else MISSING)

        # ── SAGA reference (cell mean over its repeats) ──────────────────
        s = saga_reference(pooled, arch, recipe)
        if s:
            row["top1_saga_mean"] = float(s["top1_last"])
            row["top1_saga_n_repeats"] = int(s["n"])
            row["top1_delta_vs_saga"] = t_ev["top1"] - float(s["top1_last"])
            row["sink_canon_saga_mean"] = float(s["sink_fixed_canon"])
            row["oversmooth_saga_mean"] = float(s["oversmooth_pairwise"])
            row["eff_rank_saga_mean"] = float(s["eff_rank"])
        else:
            for k in ("top1_saga_mean", "top1_saga_n_repeats",
                      "top1_delta_vs_saga", "sink_canon_saga_mean",
                      "oversmooth_saga_mean", "eff_rank_saga_mean"):
                row[k] = MISSING
            notes.append(f"{label}: no SAGA mean row in e2_pooled.csv")

        # ── sinks: ABSOLUTE counts first, percentage second ──────────────
        # Decision 2: "42% of 3.70 is ~1.5 sinks vs 55% of 19.71 being ~10.8
        # — the percentages are not comparable."
        row["sink_canon_ttr"] = t_dg["sink_fixed_canon"]
        row["sink_mad_ttr"] = t_dg["sink_mad_k5"]
        if b_dg:
            row["sink_canon_baseline"] = b_dg["sink_fixed_canon"]
            row["sink_mad_baseline"] = b_dg["sink_mad_k5"]
            removed = b_dg["sink_fixed_canon"] - t_dg["sink_fixed_canon"]
            row["sink_canon_removed_abs"] = removed
            row["sink_canon_reduction_frac"] = (
                removed / b_dg["sink_fixed_canon"]
                if b_dg["sink_fixed_canon"] > 0 else MISSING)
            row["n_patch_tokens"] = 196
        else:
            for k in ("sink_canon_baseline", "sink_mad_baseline",
                      "sink_canon_removed_abs", "sink_canon_reduction_frac"):
                row[k] = MISSING
            row["n_patch_tokens"] = 196

        # ── the other diagnostic axes ────────────────────────────────────
        for field, key in [("oversmooth_pairwise", "oversmooth_pairwise"),
                           ("oversmooth_nosink", "oversmooth_pairwise_nosink"),
                           ("eff_rank", "eff_rank")]:
            row[f"{field}_ttr"] = t_dg.get(key)
            row[f"{field}_baseline"] = b_dg.get(key) if b_dg else MISSING
        # CLS-norm ratio is a per-block list; the last block is the paper's
        row["cls_norm_ratio_ttr"] = (t_dg["cls_norm_ratio"][-1]
                                     if t_dg.get("cls_norm_ratio") else MISSING)
        row["cls_norm_ratio_baseline"] = (
            b_dg["cls_norm_ratio"][-1]
            if b_dg and b_dg.get("cls_norm_ratio") else MISSING)
        row["num_prefix_tokens_ttr"] = t_dg["num_prefix_tokens"]
        row["num_prefix_tokens_baseline"] = (b_dg["num_prefix_tokens"]
                                             if b_dg else MISSING)

        # ── address map: does TTR relocate the address, or preserve it? ──
        t_ad_p = ttr_dir / "diag" / "diag_last_addr.json"
        if t_ad_p.exists() and b_ad_p:
            t_ad, b_ad = _load(t_ad_p), _load(b_ad_p)
            (row["addr_rho_canon"], row["addr_rho_canon_p"],
             row["addr_rho_canon_null_sd"]) = addr_rho(
                t_ad["freq_canon"], b_ad["freq_canon"])
            (row["addr_rho_mad"], row["addr_rho_mad_p"],
             row["addr_rho_mad_null_sd"]) = addr_rho(
                t_ad["freq_mad"], b_ad["freq_mad"])
            row["addr_mass_canon_ttr"] = t_ad["mass_canon"]
            row["addr_mass_canon_baseline"] = b_ad["mass_canon"]
            row["addr_null_reference"] = ("TASK-07 exact permutation null: "
                                          "8 dihedral x 196 torus rolls")
        else:
            for k in ("addr_rho_canon", "addr_rho_canon_p",
                      "addr_rho_canon_null_sd", "addr_rho_mad",
                      "addr_rho_mad_p", "addr_rho_mad_null_sd",
                      "addr_null_reference",
                      "addr_mass_canon_ttr", "addr_mass_canon_baseline"):
                row[k] = MISSING
            notes.append(f"{label}: address map unavailable")

        # ── paired CI (decision 1); MISSING until the HPC run returns ────
        ci_p = gate_dir / "paired_ci.json"
        if ci_p.exists():
            ci = _load(ci_p)
            if ci["ckpt_sha256"] != sha:
                notes.append(f"{label}: paired_ci.json sha mismatch — ignored")
                ci = None
        else:
            ci = None
        if ci:
            row.update(
                paired_drop=ci["top1_drop"],
                paired_ci_low=ci["bootstrap"]["ci_low"],
                paired_ci_high=ci["bootstrap"]["ci_high"],
                paired_ci_se=ci["bootstrap"]["se"],
                paired_b_broke=ci["b_unpatched_correct_patched_wrong"],
                paired_c_fixed=ci["c_unpatched_wrong_patched_correct"],
                paired_discordant=ci["n_discordant"],
                mcnemar_p=ci["mcnemar"]["p_value"],
                threshold_inside_ci=ci["threshold_inside_ci"],
                paired_n_boot=ci["bootstrap"]["n_boot"],
            )
        else:
            for k in ("paired_drop", "paired_ci_low", "paired_ci_high",
                      "paired_ci_se", "paired_b_broke", "paired_c_fixed",
                      "paired_discordant", "mcnemar_p",
                      "threshold_inside_ci", "paired_n_boot"):
                row[k] = MISSING

        rows.append(row)
    return rows, notes


def build_sweep_rows(repo: Path):
    out = []
    for run_id, _, arch, recipe, label in CELLS:
        p = repo / "results" / "ttr_midlayer" / run_id / "sweep.csv"
        if not p.exists():
            continue
        v = _load(repo / "results" / "ttr_midlayer" / run_id / "validate.json")
        base_sink = None
        for r in csv.DictReader(open(p)):
            n = int(r["n_neurons"])
            if n == 0:
                base_sink = float(r["sink_fixed_canon"])
            sink = float(r["sink_fixed_canon"])
            out.append({
                "cell": label, "base_run_id": run_id, "arch": arch,
                "recipe_actual": recipe, "n_neurons": n,
                "n_images": int(r["n_images"]),
                "top1": float(r["top1"]),
                "top1_drop": float(r["top1_drop"]),
                "sink_canon": sink,
                "sink_canon_removed_abs": (MISSING if base_sink is None
                                           else base_sink - sink),
                "sink_reduction_frac": (float(r["sink_reduction_frac"])
                                        if r["sink_reduction_frac"] else MISSING),
                "oversmooth_pairwise": float(r["oversmooth_pairwise"]),
                "meets_sink": r["meets_sink"] or MISSING,
                "meets_top1": r["meets_top1"] or MISSING,
                "passes": r["passes"] or MISSING,
                "layers_touched": r["layers_touched"],
                "canon_tau": v["canon_tau"],
            })
    return out


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k, ""))
                        for k in fields})


def main():
    p = argparse.ArgumentParser(description="TASK-10 Phase C tables.")
    p.add_argument("--repo", default=".")
    p.add_argument("--out", default="results/tables/T_ttr.csv")
    p.add_argument("--sweep-out", default="results/tables/T_ttr_sweep.csv")
    args = p.parse_args()
    repo = Path(args.repo)

    rows, notes = build_rows(repo)
    if not rows:
        raise SystemExit("no TTR cells found under results/runs/ttr_*")
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    write_csv(repo / args.out, rows, fields)

    sweep = build_sweep_rows(repo)
    if sweep:
        write_csv(repo / args.sweep_out, sweep, list(sweep[0].keys()))

    print(f"wrote {args.out}: {len(rows)} cells, {len(fields)} columns")
    print(f"wrote {args.sweep_out}: {len(sweep)} sweep rows")
    for n in notes:
        print(f"  NOTE: {n}")
    n_missing_ci = sum(1 for r in rows if r["paired_ci_low"] == MISSING)
    if n_missing_ci:
        print(f"  {n_missing_ci}/{len(rows)} cells have no paired_ci.json yet "
              f"(decision 1's HPC run); those columns are MISSING and the "
              f"note says so. Re-run this script after it lands.")
    print(f"  generated {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
