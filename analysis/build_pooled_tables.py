#!/usr/bin/env python3
"""
analysis/build_pooled_tables.py
===============================
TASK-06B Part 3.2 (amended): pooled repeat statistics keyed by
(arch, recipe_actual, variant) -> results/tables/e2_pooled.csv.

Cell membership (recipe_actual per results/notes/recipe_erratum.md and the
Part-2 provenance harvest; directory names NEVER key a cell):
- vit_small|mixup: legacy-mixupdir + legacy-nomixdir (both rlast; both
  mixup-trained) + e2r s1 + s2 (baseline/saga; registers has the two legacy
  repeats only).
- vit_small|nomix: e2r s1 + s2 (the first true-nomix cell; n=2 — SE stated
  but flagged unreliable per the Part-3 amendment).
- vit_base|mixup: legacy-nomixdir (rlast; completed, mixup-trained per the
  harvest) + e2r s1 (baseline/saga; registers legacy-nomixdir only).
  The VOID legacy ViT-B mixup-dir trio (never finished training) appears
  NOWHERE.

Every repeat is listed individually with its provenance tag before any
mean. top-1 values come from the EXACT evaluator JSONs (tools/eval.py);
diagnostics from the diag(last) JSONs. Paired deltas pair repeats by
provenance tag; an unpaired Welch test is included as a robustness line.
Missing values are the literal MISSING, never interpolated.

    python analysis/build_pooled_tables.py \
        [--out results/tables/e2_pooled.csv]
"""

import argparse
import csv
import json
import math
from pathlib import Path

MISSING = "MISSING"

DIAG_METRICS = ["sink_fixed_canon", "sink_mad_k5", "oversmooth_pairwise",
                "oversmooth_pairwise_nosink", "eff_rank",
                "cls_norm_ratio_lastblock", "cls_attn_share_lastblock"]
METRICS = ["top1_last", "top1_best"] + DIAG_METRICS

# (arch, recipe_actual) -> {provenance_tag: (eval_stem_fn, diag_stem_fn)}
LEGACY_MEMBERS = {
    ("vit_small", "mixup"): [("legacy-mixupdir", "mixup"),
                             ("legacy-nomixdir", "nomix")],
    ("vit_base", "mixup"): [("legacy-nomixdir", "nomix")],
}
E2R_MEMBERS = {
    ("vit_small", "mixup"): [("s1", "e2r_vits_mixup_{v}_s1"),
                             ("s2", "e2r_vits_mixup_{v}_s2")],
    ("vit_small", "nomix"): [("s1", "e2r_vits_nomix_{v}_s1"),
                             ("s2", "e2r_vits_nomix_{v}_s2")],
    ("vit_base", "mixup"): [("s1", "e2r_vitb_mixup_{v}_s1")],
}
CELLS = [("vit_small", "mixup"), ("vit_small", "nomix"),
         ("vit_base", "mixup")]
VARIANTS = ["baseline", "registers", "saga"]


def _get(d, key):
    v = d.get(key)
    return MISSING if v is None else v


def load_repeat_values(arch, recipe_actual, variant):
    """[(provenance, {metric: value|MISSING})] for one cell+variant."""
    out = []
    for tag, dirname in LEGACY_MEMBERS.get((arch, recipe_actual), []):
        stem = f"e2_{arch}_{dirname}_{variant}_rlast"
        ev_l = Path(f"results/legacy/eval/{stem}_last.json")
        ev_b = Path(f"results/legacy/eval/{stem}_best.json")
        dg = Path(f"results/legacy/diag/{stem}_last.json")
        if not ev_l.exists():
            continue
        vals = {"top1_last": json.load(open(ev_l))["top1"],
                "top1_best": (json.load(open(ev_b))["top1"]
                              if ev_b.exists() else MISSING)}
        d = json.load(open(dg)) if dg.exists() else {}
        vals.update(diag_values(d))
        out.append((tag, vals))
    for tag, pattern in E2R_MEMBERS.get((arch, recipe_actual), []):
        run = Path("results/runs") / pattern.format(v=variant)
        ev_l = run / "eval" / "imagenet_val_last.json"
        ev_b = run / "eval" / "imagenet_val_best.json"
        dg = run / "diag" / "diag_final_last.json"
        if not run.exists():
            continue
        vals = {"top1_last": (json.load(open(ev_l))["top1"]
                              if ev_l.exists() else MISSING),
                "top1_best": (json.load(open(ev_b))["top1"]
                              if ev_b.exists() else MISSING)}
        d = json.load(open(dg)) if dg.exists() else {}
        vals.update(diag_values(d))
        out.append((tag, vals))
    return out


def diag_values(d):
    vals = {}
    for m in DIAG_METRICS:
        if m == "cls_norm_ratio_lastblock":
            src = d.get("cls_norm_ratio")
            vals[m] = src[-1] if src else MISSING
        elif m == "cls_attn_share_lastblock":
            src = d.get("cls_attn_share")
            vals[m] = src[-1] if src else MISSING
        else:
            vals[m] = _get(d, m)
    return vals


def mean_std(xs):
    xs = [x for x in xs if x != MISSING]
    if not xs:
        return MISSING, MISSING
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, MISSING
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def welch(xs, ys):
    """Welch t-test p-value (robustness line); MISSING when undersized."""
    xs = [x for x in xs if x != MISSING]
    ys = [y for y in ys if y != MISSING]
    if len(xs) < 2 or len(ys) < 2:
        return MISSING, MISSING
    from scipy.stats import ttest_ind
    t, p = ttest_ind(xs, ys, equal_var=False)
    return round(float(t), 4), round(float(p), 4)


def build_rows():
    rows = []

    def emit(arch, rec, variant, kind, provenance="", n="", **metrics):
        row = {"arch": arch, "recipe_actual": rec, "variant": variant,
               "kind": kind, "provenance": provenance, "n": n}
        for m in METRICS:
            v = metrics.get(m, "")
            row[m] = round(v, 6) if isinstance(v, float) else v
        rows.append(row)

    for arch, rec in CELLS:
        per_variant = {v: load_repeat_values(arch, rec, v) for v in VARIANTS}
        for variant in VARIANTS:
            reps = per_variant[variant]
            if not reps:
                continue
            for tag, vals in reps:
                emit(arch, rec, variant, "repeat", provenance=tag, **vals)
            stats_m, stats_s = {}, {}
            for m in METRICS:
                mu, sd = mean_std([v[m] for _, v in reps])
                stats_m[m], stats_s[m] = mu, sd
            emit(arch, rec, variant, "mean", n=len(reps), **stats_m)
            emit(arch, rec, variant, "std", n=len(reps), **stats_s)

        base = dict(per_variant["baseline"])
        for variant in ("registers", "saga"):
            reps = per_variant[variant]
            if not reps:
                continue
            deltas = {}
            for tag, vals in reps:
                if tag not in base:
                    continue
                deltas[tag] = {
                    m: (vals[m] - base[tag][m]
                        if vals[m] != MISSING and base[tag][m] != MISSING
                        else MISSING)
                    for m in METRICS}
            for tag, dv in deltas.items():
                emit(arch, rec, variant, "paired_delta", provenance=tag, **dv)
            d_mean, d_se, d_sig = {}, {}, {}
            n_pairs = len(deltas)
            for m in METRICS:
                mu, sd = mean_std([dv[m] for dv in deltas.values()])
                d_mean[m] = mu
                if sd == MISSING or mu == MISSING:
                    d_se[m] = MISSING
                    d_sig[m] = MISSING
                else:
                    se = sd / math.sqrt(n_pairs)
                    d_se[m] = se
                    d_sig[m] = int(abs(mu) > 2 * se)
            emit(arch, rec, variant, "paired_delta_mean", n=n_pairs, **d_mean)
            emit(arch, rec, variant, "paired_delta_se", n=n_pairs, **d_se)
            emit(arch, rec, variant, "significant_2xSE", n=n_pairs, **d_sig)
            wt, wp = {}, {}
            for m in METRICS:
                t, p = welch([v[m] for _, v in reps],
                             [v[m] for v in base.values()])
                wt[m], wp[m] = t, p
            emit(arch, rec, variant, "welch_t", **wt)
            emit(arch, rec, variant, "welch_p", **wp)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Pooled e2 repeat stats.")
    parser.add_argument("--out", default="results/tables/e2_pooled.csv")
    args = parser.parse_args()

    rows = build_rows()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arch", "recipe_actual", "variant", "kind", "provenance",
              "n"] + METRICS
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")
    for r in rows:
        if r["kind"] == "paired_delta_mean":
            print(f"  {r['arch']}|{r['recipe_actual']} {r['variant']} "
                  f"(n={r['n']}): d_top1_last = {r['top1_last']}")


if __name__ == "__main__":
    main()
