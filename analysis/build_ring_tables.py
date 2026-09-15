#!/usr/bin/env python3
"""
analysis/build_ring_tables.py — TASK-13 C2
==========================================
Builds results/tables/T_ring_ablation.csv from the committed
results/finegrained/ring_ablation/<run_id>.json files.

THE QUANTITY THAT MATTERS is the CONTRAST

    contrast = drop_ring1 - drop_random44
             = (full - ring1) - (full - random44)
             = top1_random44 - top1_ring1

Both terms come from the SAME run, the SAME checkpoint and the SAME test
images, and the two masks are area-matched at 44 patches. So the contrast is
a WITHIN-RUN paired difference: everything that makes one run or one dataset
easier than another — class count, train size, backbone quality, how much
accuracy any masking costs at all — cancels out of it. That is why the
hypothesis is tested on the contrast and never on a raw drop.

`mask_center44` is the second control: area-matched and also positional, so
it separates "ring 1 specifically" from "any coherent interior region".

Reading rule (stated in the task, restated here so the table cannot be read
without it): the hypothesis — positional suppression helps when the border is
uninformative for the class and hurts when it carries class evidence —
predicts a LARGER contrast for CUB (birds in habitat) than for Aircraft
(planes against sky). This script reports the numbers whichever way they
fall and takes no position.

Layout mirrors T3_finegrained.csv: long format, repeats BEFORE any mean,
MISSING never averaged, SE = sd/sqrt(n), significant_2xSE = |mean| > 2*SE.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

MISSING = "MISSING"
REPO = Path(__file__).resolve().parents[1]
CONDITIONS = ("full", "mask_ring1", "mask_center44", "mask_random44")
DROPS = ("drop_ring1", "drop_center44", "drop_random44")
VALUES = (["top1_" + c.replace("mask_", "") for c in CONDITIONS]
          + list(DROPS) + ["contrast_ring1_minus_random44"])
FIELDS = (["dataset", "arch", "variant", "kind", "ft_seed", "n"]
          + VALUES + ["flag"])
ARCH_ORDER = ["vit_small_patch16_224", "vit_base_patch16_224"]


def mean_std(xs):
    """(mean, sample std, se). std/se are MISSING for n < 2 — never 0.0,
    which would read as 'measured and found identical'."""
    n = len(xs)
    if n == 0:
        return MISSING, MISSING, MISSING
    m = sum(xs) / n
    if n < 2:
        return m, MISSING, MISSING
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd, sd / math.sqrt(n)


def load_runs(matrix, ring_dir: Path, runs_root: Path):
    """One record per matrix run that has a ring-ablation JSON.

    Every record is cross-checked against the run's OWN test_final.json
    through an independent read: same fine-tuned sha256, same committed
    top-1. A run whose `full` condition did not reproduce that number
    contributes MISSING values and a flag — never a drop computed against an
    unverified baseline.
    """
    out = []
    for run_id, run in matrix["runs"].items():
        p = ring_dir / f"{run_id}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        flags = []

        final_p = runs_root / run_id / "eval" / "test_final.json"
        if not final_p.exists():
            flags.append("no test_final.json")
        else:
            final = json.loads(final_p.read_text(encoding="utf-8"))
            if final.get("finetuned_sha256") != d.get("finetuned_sha256"):
                flags.append("finetuned sha mismatch vs test_final.json")
            if float(final["top1"]) != float(d["committed_test_top1"]):
                flags.append("committed top1 disagrees with test_final.json")
            if final.get("smoke"):
                flags.append("smoke artifact")
        if not d.get("full_reproduces_test_final"):
            flags.append(f"full did not reproduce "
                         f"(delta {d.get('full_minus_committed')})")
        for c in ("mask_ring1", "mask_center44", "mask_random44"):
            if d["n_masked_patches"].get(c) != 44:
                flags.append(f"{c} masks {d['n_masked_patches'].get(c)} != 44")

        rec = {"dataset": d["dataset"], "arch": d["arch"],
               "variant": d["variant"], "ft_seed": d["ft_seed"],
               "run_id": run_id, "flag": "; ".join(flags)}
        if flags:
            rec.update({k: MISSING for k in VALUES})
        else:
            t = d["top1"]
            rec.update({
                "top1_full": t["full"],
                "top1_ring1": t["mask_ring1"],
                "top1_center44": t["mask_center44"],
                "top1_random44": t["mask_random44"],
                "drop_ring1": t["full"] - t["mask_ring1"],
                "drop_center44": t["full"] - t["mask_center44"],
                "drop_random44": t["full"] - t["mask_random44"],
                # within-run paired contrast; algebraically random44 - ring1
                "contrast_ring1_minus_random44":
                    t["mask_random44"] - t["mask_ring1"],
            })
        out.append(rec)
    return out


def _agg_rows(rows, records, key_fields, label, group):
    """Emit mean / std / se / significant_2xSE for one group of records."""
    base = {f: "" for f in FIELDS}
    base.update(key_fields)
    stats = {"mean": {}, "std": {}, "se": {}, "significant_2xSE": {}}
    n_used = 0
    for col in VALUES:
        xs = [r[col] for r in group if r[col] != MISSING]
        n_used = max(n_used, len(xs))
        m, sd, se = mean_std(xs)
        stats["mean"][col] = m
        stats["std"][col] = sd
        stats["se"][col] = se
        if m == MISSING or se == MISSING:
            stats["significant_2xSE"][col] = MISSING
        else:
            stats["significant_2xSE"][col] = int(abs(m) > 2 * se)
    n_missing = sum(1 for r in group if r[VALUES[0]] == MISSING)
    for kind in ("mean", "std", "se", "significant_2xSE"):
        row = dict(base)
        row["kind"] = f"{label}{kind}" if label else kind
        row["n"] = n_used
        row.update({c: stats[kind][c] for c in VALUES})
        if n_missing:
            row["flag"] = f"{n_missing} run(s) excluded as MISSING"
        rows.append(row)


def build_rows(matrix, ring_dir: Path, runs_root: Path):
    records = load_runs(matrix, ring_dir, runs_root)
    rows = []

    # ── 1. repeats, individually, BEFORE any mean ────────────────────────
    def sort_key(r):
        return (r["dataset"], ARCH_ORDER.index(r["arch"]), r["variant"],
                r["ft_seed"])
    for r in sorted(records, key=sort_key):
        row = {f: "" for f in FIELDS}
        row.update({k: r[k] for k in ("dataset", "arch", "variant", "flag")})
        row["kind"] = "repeat"
        # same "f<seed>" spelling T3_finegrained.csv uses, so the two sibling
        # tables join on (dataset, arch, variant, ft_seed) without munging
        row["ft_seed"] = f"f{r['ft_seed']}"
        row["n"] = 1
        row.update({c: r[c] for c in VALUES})
        rows.append(row)

    # ── 2. per (dataset, arch, variant) ──────────────────────────────────
    for ds in sorted({r["dataset"] for r in records}):
        for arch in ARCH_ORDER:
            for variant in ("baseline", "saga"):
                g = [r for r in records if r["dataset"] == ds
                     and r["arch"] == arch and r["variant"] == variant]
                if g:
                    _agg_rows(rows, records,
                              {"dataset": ds, "arch": arch, "variant": variant},
                              "", g)

    # ── 3. per dataset — the level the hypothesis is stated at ───────────
    # pooled over both architectures and both variants: the claim is about
    # the DATASET's border being informative or not, not about one model
    for ds in sorted({r["dataset"] for r in records}):
        g = [r for r in records if r["dataset"] == ds]
        _agg_rows(rows, records,
                  {"dataset": ds, "arch": "ALL", "variant": "ALL"}, "", g)

    # ── 4. the prediction test: CUB contrast vs Aircraft contrast ────────
    # UNPAIRED by nature — different images, different models, so the two
    # sides share nothing. SE of the difference = sqrt(se_a^2 + se_b^2).
    col = "contrast_ring1_minus_random44"
    per_ds = {}
    for ds in sorted({r["dataset"] for r in records}):
        xs = [r[col] for r in records if r["dataset"] == ds
              and r[col] != MISSING]
        per_ds[ds] = mean_std(xs) + (len(xs),)
    if "cub" in per_ds and "aircraft" in per_ds:
        (mc, _, sec, nc) = per_ds["cub"]
        (ma, _, sea, na) = per_ds["aircraft"]
        row = {f: "" for f in FIELDS}
        row.update({"dataset": "cub_minus_aircraft", "arch": "ALL",
                    "variant": "ALL", "kind": "between_dataset_difference",
                    "n": f"{nc}v{na}"})
        if MISSING in (mc, ma, sec, sea):
            row[col] = MISSING
            row["flag"] = "a side had n < 2"
        else:
            diff = mc - ma
            se = math.sqrt(sec ** 2 + sea ** 2)
            row[col] = diff
            row["flag"] = (f"se={se:.4f}; |diff|-2se={abs(diff) - 2 * se:+.4f}; "
                           f"significant_2xSE={int(abs(diff) > 2 * se)}; "
                           f"UNPAIRED (different images and models); "
                           f"prediction: CUB > Aircraft")
        rows.append(row)
    return rows, records


def main():
    ap = argparse.ArgumentParser("TASK-13 ring-ablation table")
    ap.add_argument("--matrix", default="configs/ft_matrix.yaml")
    ap.add_argument("--ring-dir",
                    default="results/finegrained/ring_ablation")
    ap.add_argument("--runs-root", default="results/runs")
    ap.add_argument("--out", default="results/tables/T_ring_ablation.csv")
    a = ap.parse_args()

    def _p(x):
        x = Path(x)
        return x if x.is_absolute() else REPO / x

    matrix = yaml.safe_load(_p(a.matrix).read_text(encoding="utf-8"))
    rows, records = build_rows(matrix, _p(a.ring_dir), _p(a.runs_root))

    out = _p(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_flag = sum(1 for r in records if r["flag"])
    print(f"wrote {out.relative_to(REPO)}: {len(rows)} rows "
          f"({len(records)} runs, {n_flag} flagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
