#!/usr/bin/env python3
"""
analysis/build_localization_tables.py
=====================================
TASK-11 C2 — turn the per-image localization rows into the committed table:

    results/figures_data/F1_localization.csv   ->   results/tables/T_localization.csv

One row per (arch, group, metric, run): n, mean, sample std, SE, the metric's
null reference, and — for every non-baseline model — the delta against its OWN
architecture's baseline, PAIRED BY IMAGE, with SE and the 2xSE flag.

CONVENTIONS, inherited from analysis/build_ft_tables.py and
analysis/build_pooled_tables.py so the paper's tables agree with each other:
  * sample std (n-1); MISSING when n < 2, never a zero
  * SE = sd / sqrt(n_pairs); significant_2xSE = |mean| > 2*SE
  * Welch is an unpaired robustness line only, never the headline
  * MISSING is never averaged, interpolated or imputed

PAIRING IS WITHIN ARCHITECTURE. A ViT-S model is only ever compared with the
ViT-S baseline. There is no ViT-B registers run, so that cell is reported
ABSENT — not zero, and not silently pooled with ViT-S.

TWO METRICS ARE NOT MODEL-DEPENDENT. `box_area_frac` (boxes20) and
`uniform_ring1_mass` (random200) are properties of the image and the grid, so
they are identical for every model by construction — that is precisely why
they exist: they are the uniform-attention NULL that `inbox_mass` and
`ring1_mass` are meaningless without (TASK-07 established that these
statistics can invert sign once their null is subtracted). They are emitted
once per (arch, group) as `null_reference` rows with no delta, and the builder
ASSERTS their model-independence rather than assuming it.
"""

import argparse
import csv
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.run_registry import git_sha                     # noqa: E402
from tools.dense_runtime import atomic_write_text         # noqa: E402

MISSING = "MISSING"

# metric -> its model-independent null reference (None = no null)
NULL_OF = {
    "inbox_mass": "box_area_frac",
    "ring1_mass": "uniform_ring1_mass",
    "pointing_hit": None,
    "attn_entropy_bits": None,
}
NULL_METRICS = {v for v in NULL_OF.values() if v}

FIELDS = ("arch", "group", "metric", "run_id", "variant", "role", "n",
          "mean", "std", "se", "null_mean", "delta_vs_baseline_mean",
          "delta_sd", "delta_se", "delta_significant_2xSE", "delta_direction",
          "welch_p", "note")


def fmt(x, nd=6):
    if x == MISSING or x is None:
        return MISSING
    if isinstance(x, bool):
        return "YES" if x else "no"
    return f"{x:.{nd}g}"


def mean_std(xs):
    """(mean, sample std); std MISSING at n < 2 — never 0."""
    if not xs:
        return MISSING, MISSING
    m = statistics.fmean(xs)
    return (m, statistics.stdev(xs)) if len(xs) > 1 else (m, MISSING)


def welch_p(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return MISSING
    try:
        from scipy import stats
    except ImportError:
        return MISSING
    return float(stats.ttest_ind(xs, ys, equal_var=False).pvalue)


def load_rows(csv_path):
    """(run_id, group, metric) -> {image_id: value}."""
    table = defaultdict(dict)
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            table[(r["run_id"], r["group"], r["metric"])][r["image_id"]] = \
                float(r["value"])
    return table


def check_null_is_model_independent(table, runs, group, metric):
    """A null that differs between models is not a null. Raise rather than
    quietly publish a per-model 'reference'."""
    seen = {}
    for run_id in runs:
        per_image = table.get((run_id, group, metric))
        if per_image:
            seen[run_id] = per_image
    if len(seen) < 2:
        return next(iter(seen.values()), {})
    ref_run, ref = next(iter(seen.items()))
    for run_id, other in seen.items():
        if other.keys() != ref.keys():
            raise ValueError(
                f"{metric}: {run_id} and {ref_run} cover different images")
        worst = max((abs(other[k] - ref[k]) for k in ref), default=0.0)
        if worst > 1e-9:
            raise ValueError(
                f"{metric} is supposed to be model-independent but {run_id} "
                f"differs from {ref_run} by up to {worst:.3g} — it cannot be "
                f"used as a null reference")
    return ref


def build(table, meta, archs_of):
    rows = []

    def emit(**kw):
        row = {k: MISSING for k in FIELDS}
        row.update({k: v for k, v in kw.items() if v is not None})
        rows.append(row)

    groups = sorted({g for (_, g, _) in table})
    for arch in sorted(archs_of):
        runs = archs_of[arch]
        baselines = [r for r in runs if meta[r]["variant"] == "baseline"]
        if len(baselines) != 1:
            raise ValueError(
                f"{arch}: expected exactly one baseline, got {baselines}")
        baseline = baselines[0]

        for group in groups:
            metrics = sorted({m for (r, g, m) in table
                              if g == group and r in runs})
            for metric in metrics:
                if metric in NULL_METRICS:
                    ref = check_null_is_model_independent(
                        table, runs, group, metric)
                    mu, sd = mean_std(sorted(ref.values()))
                    emit(arch=arch, group=group, metric=metric,
                         run_id="(all models)", variant="(image)",
                         role="null_reference", n=len(ref),
                         mean=fmt(mu), std=fmt(sd),
                         se=fmt(sd / math.sqrt(len(ref))
                                if sd != MISSING and ref else MISSING),
                         note="model-independent by construction; verified "
                              "equal across models to 1e-9")
                    continue

                base_vals = table.get((baseline, group, metric), {})
                null_name = NULL_OF.get(metric)
                null_vals = (table.get((baseline, group, null_name), {})
                             if null_name else {})

                for run_id in runs:
                    per_image = table.get((run_id, group, metric))
                    variant = meta[run_id]["variant"]
                    if not per_image:
                        emit(arch=arch, group=group, metric=metric,
                             run_id=run_id, variant=variant,
                             role="baseline" if run_id == baseline else variant,
                             n=0, note="ABSENT — no rows for this run")
                        continue

                    vals = [per_image[k] for k in sorted(per_image)]
                    mu, sd = mean_std(vals)
                    se = (sd / math.sqrt(len(vals))) if sd != MISSING else MISSING
                    null_mu = (statistics.fmean(null_vals.values())
                               if null_vals else MISSING)

                    if run_id == baseline:
                        emit(arch=arch, group=group, metric=metric,
                             run_id=run_id, variant=variant, role="baseline",
                             n=len(vals), mean=fmt(mu), std=fmt(sd),
                             se=fmt(se), null_mean=fmt(null_mu),
                             note="reference for this architecture")
                        continue

                    shared = sorted(set(per_image) & set(base_vals))
                    deltas = [per_image[k] - base_vals[k] for k in shared]
                    d_mu, d_sd = mean_std(deltas)
                    if d_sd == MISSING or not deltas:
                        d_se = d_sig = d_dir = MISSING
                    else:
                        d_se = d_sd / math.sqrt(len(deltas))
                        d_sig = abs(d_mu) > 2 * d_se
                        d_dir = ("above baseline" if d_mu > 0
                                 else "below baseline" if d_mu < 0 else "equal")
                    emit(arch=arch, group=group, metric=metric, run_id=run_id,
                         variant=variant, role=variant, n=len(vals),
                         mean=fmt(mu), std=fmt(sd), se=fmt(se),
                         null_mean=fmt(null_mu),
                         delta_vs_baseline_mean=fmt(d_mu), delta_sd=fmt(d_sd),
                         delta_se=fmt(d_se),
                         delta_significant_2xSE=fmt(d_sig),
                         delta_direction=d_dir,
                         welch_p=fmt(welch_p(vals, [base_vals[k] for k in
                                                    sorted(base_vals)])),
                         note=f"paired by image vs {baseline} "
                              f"(n_pairs={len(deltas)})")
    return rows


def main():
    p = argparse.ArgumentParser(description="Build T_localization.csv.")
    p.add_argument("--localization",
                   default="results/figures_data/F1_localization.csv")
    p.add_argument("--meta", default=None,
                   help="default: the CSV's sibling _meta.json")
    p.add_argument("--out", default="results/tables/T_localization.csv")
    args = p.parse_args()

    csv_path = Path(args.localization)
    meta_path = Path(args.meta) if args.meta else csv_path.with_name(
        csv_path.stem + "_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)["runs"]

    table = load_rows(csv_path)
    archs_of = defaultdict(list)
    for run_id, prov in sorted(meta.items()):
        arch = prov.get("arch")
        if not arch:
            raise ValueError(f"{run_id}: no arch recorded — cannot assign a cell")
        archs_of[arch].append(run_id)

    rows = build(table, meta, archs_of)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    atomic_write_text(buf.getvalue(), Path(args.out))

    side = Path(args.out).with_name(Path(args.out).stem + "_meta.json")
    atomic_write_text(json.dumps({
        "git_sha": git_sha(), "source": str(csv_path),
        "n_rows": len(rows), "archs": {a: v for a, v in sorted(archs_of.items())},
    }, indent=2, sort_keys=True) + "\n", side)

    print(f"wrote {args.out}: {len(rows)} rows over "
          f"{len(archs_of)} architectures")
    for a, v in sorted(archs_of.items()):
        print(f"  {a}: {', '.join(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
