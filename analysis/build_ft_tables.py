#!/usr/bin/env python3
"""
analysis/build_ft_tables.py
===========================
TASK-08 Phase C.1: fine-grained transfer table T3 ->
results/tables/T3_finegrained.csv.

One cell per (dataset, arch); within a cell, each variant's ft-seed
repeats are listed INDIVIDUALLY before any aggregate, then mean/std, then
the SAGA-minus-baseline paired delta (paired BY ft-seed, the only valid
pairing: both sides share the frozen val split and the ft-seed's data
order), its mean/SE/significant_2xSE, and an unpaired Welch line as a
robustness check.

Statistics conventions are the project's existing ones
(analysis/build_pooled_tables.py): sample std (n-1), MISSING when n<2,
SE = sd/sqrt(n_pairs), significant_2xSE = |mean| > 2*SE, Welch requires
n>=2 per side. MISSING is never interpolated or averaged. The ViT-B cells
have a single ft-seed by design (task matrix) and are flagged n=1: no std,
no SE, no significance claim is possible for them.

Every value is read from results/runs/ft_*/eval/test_final.json — the
file each run wrote after touching the official test split exactly once.
A run whose recorded backbone sha256 does not match configs/ft_matrix.yaml
yields MISSING rather than a number from an unverified checkpoint.

    python analysis/build_ft_tables.py \
        [--runs-root results/runs] [--out results/tables/T3_finegrained.csv]
"""

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

MISSING = "MISSING"
REPO = Path(__file__).resolve().parents[1]

# aggregated metrics (test_top1 is the headline)
METRICS = ["test_top1", "test_top5", "val_top1_at_best"]
# per-repeat context columns (never aggregated)
INFO = ["best_epoch", "epochs_trained", "val_minus_test", "n_test_images"]

CELLS = [("cub", "vit_small_patch16_224"), ("cub", "vit_base_patch16_224"),
         ("aircraft", "vit_small_patch16_224"),
         ("aircraft", "vit_base_patch16_224")]
VARIANTS = ["baseline", "saga"]
FIELDS = (["dataset", "arch", "variant", "kind", "ft_seed", "n"]
          + METRICS + INFO + ["flag"])


def load_matrix(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_repeats(runs_root: Path, matrix: dict, dataset: str, arch: str,
                 variant: str):
    """[(ft_seed, {metric/info: value|MISSING})] for one cell+variant,
    ordered by ft_seed. Only runs declared in the matrix are considered, so
    a stray directory can never enter a paper table."""
    out = []
    for run_id, run in matrix["runs"].items():
        if (run["dataset"], run["arch"], run["variant"]) != (dataset, arch,
                                                             variant):
            continue
        path = runs_root / run_id / "eval" / "test_final.json"
        seed = int(run["ft_seed"])
        if not path.exists():
            out.append((seed, {k: MISSING for k in METRICS + INFO}))
            continue
        with open(path) as f:
            d = json.load(f)

        expected_sha = matrix["backbones"][run["backbone"]]["sha256"]
        ok = (d.get("backbone_sha256") == expected_sha
              and not d.get("smoke", False))
        if not ok:
            # unverified provenance (or a smoke artifact): never a number
            out.append((seed, {k: MISSING for k in METRICS + INFO}))
            continue

        val_best = d.get("val_top1_at_best")
        top1 = d.get("top1")
        vals = {
            "test_top1": top1,
            "test_top5": d.get("top5"),
            "val_top1_at_best": val_best,
            "best_epoch": d.get("best_epoch"),
            "epochs_trained": d.get("epochs_trained"),
            "val_minus_test": (round(val_best - top1, 3)
                               if val_best is not None and top1 is not None
                               else MISSING),
            "n_test_images": d.get("n_images"),
        }
        out.append((seed, {k: (MISSING if v is None else v)
                           for k, v in vals.items()}))
    return sorted(out, key=lambda t: t[0])


def mean_std(xs):
    """(mean, sample std); std MISSING at n<2. MISSING values are dropped,
    never imputed."""
    xs = [x for x in xs if x != MISSING]
    if not xs:
        return MISSING, MISSING
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, MISSING
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def welch(xs, ys):
    """Welch t-test (robustness line); MISSING when either side has n<2."""
    xs = [x for x in xs if x != MISSING]
    ys = [y for y in ys if y != MISSING]
    if len(xs) < 2 or len(ys) < 2:
        return MISSING, MISSING
    from scipy.stats import ttest_ind
    t, p = ttest_ind(xs, ys, equal_var=False)
    return round(float(t), 4), round(float(p), 4)


def build_rows(runs_root: Path, matrix: dict):
    rows = []

    def emit(dataset, arch, variant, kind, ft_seed="", n="", flag="",
             **values):
        row = {"dataset": dataset, "arch": arch, "variant": variant,
               "kind": kind, "ft_seed": ft_seed, "n": n, "flag": flag}
        for key in METRICS + INFO:
            v = values.get(key, "")
            row[key] = round(v, 6) if isinstance(v, float) else v
        rows.append(row)

    for dataset, arch in CELLS:
        per_variant = {v: load_repeats(runs_root, matrix, dataset, arch, v)
                       for v in VARIANTS}
        n_seeds = max((len(r) for r in per_variant.values()), default=0)
        single = n_seeds < 2
        cell_flag = "n=1 (single ft-seed by design: no std/SE/significance)" \
            if single else ""

        for variant in VARIANTS:
            reps = per_variant[variant]
            if not reps:
                continue
            for seed, vals in reps:
                emit(dataset, arch, variant, "repeat", ft_seed=f"f{seed}",
                     flag=cell_flag, **vals)
            mus, sds = {}, {}
            for m in METRICS:
                mu, sd = mean_std([v[m] for _, v in reps])
                mus[m], sds[m] = mu, sd
            emit(dataset, arch, variant, "mean", n=len(reps),
                 flag=cell_flag, **mus)
            emit(dataset, arch, variant, "std", n=len(reps),
                 flag=cell_flag, **sds)

        base = dict(per_variant["baseline"])
        reps = per_variant["saga"]
        if not reps or not base:
            continue
        deltas = {}
        for seed, vals in reps:
            if seed not in base:
                continue  # unpaired ft-seed: never averaged in
            deltas[seed] = {
                m: (vals[m] - base[seed][m]
                    if vals[m] != MISSING and base[seed][m] != MISSING
                    else MISSING)
                for m in METRICS}
        for seed in sorted(deltas):
            emit(dataset, arch, "saga", "paired_delta", ft_seed=f"f{seed}",
                 flag=cell_flag, **deltas[seed])

        n_pairs = len(deltas)
        d_mean, d_se, d_sig = {}, {}, {}
        for m in METRICS:
            mu, sd = mean_std([dv[m] for dv in deltas.values()])
            d_mean[m] = mu
            if sd == MISSING or mu == MISSING:
                d_se[m] = d_sig[m] = MISSING
            else:
                se = sd / math.sqrt(n_pairs)
                d_se[m] = se
                d_sig[m] = int(abs(mu) > 2 * se)
        emit(dataset, arch, "saga", "paired_delta_mean", n=n_pairs,
             flag=cell_flag, **d_mean)
        emit(dataset, arch, "saga", "paired_delta_se", n=n_pairs,
             flag=cell_flag, **d_se)
        emit(dataset, arch, "saga", "significant_2xSE", n=n_pairs,
             flag=cell_flag, **d_sig)

        wt, wp = {}, {}
        for m in METRICS:
            t, p = welch([v[m] for _, v in reps],
                         [v[m] for v in base.values()])
            wt[m], wp[m] = t, p
        emit(dataset, arch, "saga", "welch_t", n=n_pairs, flag=cell_flag,
             **wt)
        emit(dataset, arch, "saga", "welch_p", n=n_pairs, flag=cell_flag,
             **wp)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build T3_finegrained.csv from the ft run test JSONs.")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--matrix", default="configs/ft_matrix.yaml")
    parser.add_argument("--out", default="results/tables/T3_finegrained.csv")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    rows = build_rows(Path(args.runs_root), matrix)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_rep = sum(1 for r in rows if r["kind"] == "repeat")
    n_miss = sum(1 for r in rows for k in METRICS if r[k] == MISSING)
    print(f"wrote {out}: {len(rows)} rows ({n_rep} repeats, "
          f"{n_miss} MISSING metric cells)")


if __name__ == "__main__":
    main()
