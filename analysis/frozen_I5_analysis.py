#!/usr/bin/env python3
"""
analysis/frozen_I5_analysis.py
==============================
TASK C / I5 §3, §4 — the I5 tables, built from committed records only.

    python analysis/frozen_I5_analysis.py --split sub2k

    T_I5a_readout        per checkpoint x stage x transform x descriptor:
                         mean acc_exact / acc_1, image-level bootstrap CI,
                         and CHANCE on every row
    T_I5b_methods        SAGA - baseline paired by seed; registers - baseline
                         (n = 2, labelled); per stage, per transform
    T_I5c_terminal       the 8 SAGA checkpoints, native vs term_1.00 —
                         can a feature consumer see the terminal constant?
    T_I5d_positions      accuracy at MAD-exceedance positions vs the rest
    T_I5e_diag_vs_readout  within checkpoint: Spearman over images between
                         acc_exact and the diagnostics; across checkpoints,
                         descriptive scatter data only, no test
    T_I5f_seg            I5b: mIoU_ss, paired per-image delta with CI,
                         per-class IoU deltas (all 150)
    T_I5g_seg_terminal   I5b under the terminal-gate bypass (exploratory)

EVERY TABLE IS STAMPED
----------------------
`endpoint_class` is a column on every row: `secondary` for everything I5
measures (D7 fixed the three primary contrasts in Track B) and `descriptive`
for the across-checkpoint scatter and the register rows. No row of any I5
table is a primary claim, and none is worded as one: the tables carry
measurements and intervals, and the interpretation lives in the handoff.

CHANCE ON EVERY READOUT TABLE
-----------------------------
A correspondence accuracy without its chance level is unreadable — 0.31 is
either 60x chance or nothing, depending on the grid. Both chance values are
columns, computed by `saga/frozen/transforms.py` from the geometry, on every
row of every readout table (§3).

MISSING IS NEVER AVERAGED
-------------------------
`mean_or_missing` drops nothing and guesses nothing: a column with any
MISSING in it yields MISSING for the statistic unless the MISSING values are
a declared, counted subset (an image with no exceedance position genuinely
has no accuracy at exceedance positions, and `n_images_scored` records how
many contributed). I0 handoff §8.2.

NOTHING HERE READS THE HPC
--------------------------
Every number comes from a file under `results/frozen/I5_readout/`. A run
whose records are absent is reported as MISSING with its expected path, and
never interpolated from a sibling.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from saga.frozen.records import CORR_MISSING_COLUMNS, records_path  # noqa: E402
from saga.frozen.runner import eligible_cohort  # noqa: E402
from saga.frozen.transforms import MATCHED_TRANSFORMS  # noqa: E402

MISSING = "MISSING"

#: LOCKED_ANALYSIS §8 / D6.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CHUNK = 500
CI_PERCENTILES = (2.5, 97.5)

#: D7: the three primary contrasts are Track B's. Everything here is one of
#: these two, and the column saying which is not optional.
SECONDARY = "secondary"
DESCRIPTIVE = "descriptive"

WORK_PACKAGE = "I5_readout"


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────

def read_records(path):
    """The rows of a records file, parquet or CSV, as dicts."""
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row, key):
    """A numeric column, or MISSING — never a silent NaN.

    The columns in `CORR_MISSING_COLUMNS` are strings on disk (a parquet
    column cannot hold both a float and the literal MISSING), so this is
    where a reader turns one back into a number, having first checked it is
    not MISSING.
    """
    v = row.get(key, MISSING)
    if v is None or v == MISSING or v == "":
        return MISSING
    if key in CORR_MISSING_COLUMNS or isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            return MISSING
    return float(v)


def load_run(root, run_id):
    """One run's correspondence records, or None."""
    return read_records(records_path(Path(root) / run_id, "corr_records"))


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def mean_or_missing(values):
    """Mean over the values that exist, with MISSING dropped and COUNTED.

    Returns (mean, n_scored, n_missing). A caller writes all three, so a
    reader can see that a mean over 1,943 of 2,000 images is not a mean over
    2,000 — which is the whole reason MISSING is a value here.
    """
    kept = [v for v in values if v is not MISSING]
    n_missing = len(values) - len(kept)
    if not kept:
        return MISSING, 0, n_missing
    return float(np.mean(kept)), len(kept), n_missing


def bootstrap_ci(per_image, *, resamples=BOOTSTRAP_RESAMPLES,
                 seed=BOOTSTRAP_SEED, chunk=BOOTSTRAP_CHUNK):
    """(mean, lo, hi) by image-level bootstrap (LOCKED_ANALYSIS §8, D6 seed 0).

    The resampling unit is the IMAGE. Deterministic: the same input gives the
    same interval on every machine, and `tests/test_I5_readout.py` asserts
    that twice-run output is byte-identical.
    """
    arr = np.asarray([v for v in per_image if v is not MISSING],
                     dtype=np.float64)
    if arr.size == 0:
        return MISSING, MISSING, MISSING
    rng = np.random.RandomState(seed)
    stats = np.empty(resamples, dtype=np.float64)
    done = 0
    while done < resamples:
        take = min(chunk, resamples - done)
        idx = rng.randint(0, arr.size, size=(take, arr.size))
        stats[done:done + take] = arr[idx].mean(axis=1)
        done += take
    lo, hi = np.percentile(stats, CI_PERCENTILES)
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap(deltas, **kw):
    """The same image-level bootstrap over a PAIRED per-image difference.

    The pairing is done before this function sees anything: `deltas[i]` is
    one image's (a - b), so resampling images resamples pairs and the
    pairing cannot be broken by the resampler.
    """
    return bootstrap_ci(deltas, **kw)


def spearman(x, y):
    """Spearman rho over the images where BOTH values exist.

    scipy is already a dependency of this repository (analysis/ imports it
    elsewhere); using it keeps the tie handling standard rather than
    hand-rolled.
    """
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not MISSING and b is not MISSING]
    if len(pairs) < 3:
        return MISSING, len(pairs)
    from scipy.stats import spearmanr
    a, b = zip(*pairs)
    if len(set(a)) < 2 or len(set(b)) < 2:
        return MISSING, len(pairs)
    rho = spearmanr(np.asarray(a), np.asarray(b)).statistic
    return (float(rho) if np.isfinite(rho) else MISSING), len(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Grouping
# ─────────────────────────────────────────────────────────────────────────────

def group(rows, *keys):
    """{key tuple: [row, ...]} in first-seen order."""
    out = {}
    for r in rows:
        out.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return out


def by_image(rows, key):
    """{image_id: value} for one metric, MISSING preserved."""
    return {r["image_id"]: value(r, key) for r in rows}


def paired_deltas(a_rows, b_rows, key):
    """[a - b] over the images BOTH sides scored, and the counts.

    Images only one side scored are dropped and COUNTED, never zero-filled:
    a pair that does not exist is not a difference of zero.
    """
    a, b = by_image(a_rows, key), by_image(b_rows, key)
    shared = [i for i in a if i in b]
    deltas = [a[i] - b[i] for i in shared
              if a[i] is not MISSING and b[i] is not MISSING]
    return deltas, len(shared), len(a) + len(b) - 2 * len(shared)


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v) -> str:
    """One float format for every table cell; MISSING and labels pass through.

    Deliberately the same shape as `analysis/build_I2_tables.py::fmt`, which
    every committed frozen table is already formatted by, so an I5 table and
    an I2 table print a number the same way.
    """
    if v is MISSING or v == MISSING or v is None:
        return MISSING
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)                     # a run_id, a stage, a label
    return MISSING if not np.isfinite(f) else f"{f:.10g}"


def write_csv(path, fields, rows):
    """A table, deterministically. LF line endings, fixed field order, every
    field formatted by `fmt`, so re-running the builder on the same records
    reproduces the file BYTE FOR BYTE (a test asserts it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, MISSING)) for k in fields})
    return path


# ─────────────────────────────────────────────────────────────────────────────
# T_I5a — the readout table
# ─────────────────────────────────────────────────────────────────────────────

A_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "provenance_tag",
            "condition_id", "stage", "transform", "descriptor",
            "n_images", "n_shared", "grid", "chance_exact", "chance_1",
            "acc_exact_mean", "acc_exact_ci_lo", "acc_exact_ci_hi",
            "acc_1_mean", "acc_1_ci_lo", "acc_1_ci_hi",
            "mean_nn_sim", "acc_exact_over_chance", "endpoint_class")


def build_a(loaded, cohort):
    rows = []
    for r in cohort:
        recs = loaded.get(r["run_id"])
        if not recs:
            continue
        for key, g in group(recs, "condition_id", "stage", "transform",
                            "descriptor").items():
            cond, stage, transform, desc = key
            acc_e = [value(x, "acc_exact") for x in g]
            acc_1 = [value(x, "acc_1") for x in g]
            m_e, lo_e, hi_e = bootstrap_ci(acc_e)
            m_1, lo_1, hi_1 = bootstrap_ci(acc_1)
            chance_e = value(g[0], "chance_exact")
            rows.append({
                "run_id": r["run_id"], "arch": r["arch"],
                "recipe_actual": r["recipe_actual"], "variant": r["variant"],
                "provenance_tag": r.get("provenance_tag", MISSING),
                "condition_id": cond, "stage": stage, "transform": transform,
                "descriptor": desc, "n_images": len(g),
                "n_shared": g[0].get("n_shared"), "grid": g[0].get("grid"),
                "chance_exact": chance_e, "chance_1": value(g[0], "chance_1"),
                "acc_exact_mean": m_e, "acc_exact_ci_lo": lo_e,
                "acc_exact_ci_hi": hi_e, "acc_1_mean": m_1,
                "acc_1_ci_lo": lo_1, "acc_1_ci_hi": hi_1,
                "mean_nn_sim": mean_or_missing(
                    [value(x, "mean_nn_sim") for x in g])[0],
                "acc_exact_over_chance": (
                    m_e / chance_e if m_e is not MISSING
                    and chance_e not in (MISSING, 0) else MISSING),
                "endpoint_class": SECONDARY,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I5b — the method contrast, paired by seed
# ─────────────────────────────────────────────────────────────────────────────

B_FIELDS = ("cell", "arch", "recipe_actual", "contrast", "pair_label",
            "pair_kind", "stage", "transform", "descriptor",
            "n_pairs", "n_images_paired", "n_images_dropped",
            "delta_acc_exact", "ci_lo", "ci_hi", "delta_acc_1",
            "a_acc_exact", "b_acc_exact", "chance_exact", "endpoint_class")


def pair_key(row):
    """What a checkpoint is paired ON: its seed tag where one exists.

    `provenance_tag` is `s1`/`s2` for the seeded e2r runs. The legacy repeats
    recorded no seed, so they pair on their provenance tag as a LABEL and the
    table says `pair_kind=legacy` — a legacy pair is not a seed pair and the
    column is what stops them being read as one.
    """
    tag = str(row.get("provenance_tag") or "")
    seeded = str(row.get("seed_controlled") or "") in ("1", "True", "true")
    return tag, ("seed" if seeded else "legacy")


def build_b(loaded, cohort):
    rows = []
    cells = group(cohort, "arch", "recipe_actual")
    for (arch, recipe), members in cells.items():
        baselines = [m for m in members if m["variant"] == "baseline"]
        for variant, label in (("saga", "saga_minus_baseline"),
                               ("registers", "registers_minus_baseline")):
            others = [m for m in members if m["variant"] == variant]
            if not others:
                continue
            pairs = []
            for o in others:
                tag, kind = pair_key(o)
                mate = next((b for b in baselines
                             if pair_key(b) == (tag, kind)), None)
                if mate is None:
                    continue
                pairs.append((o, mate, tag, kind))
            for stage, transform, descriptor in _axes(loaded, cohort):
                for o, b, tag, kind in pairs:
                    a_recs = _select(loaded, o["run_id"], "native", stage,
                                     transform, descriptor)
                    b_recs = _select(loaded, b["run_id"], "native", stage,
                                     transform, descriptor)
                    if a_recs is None or b_recs is None:
                        continue
                    deltas, n_paired, n_dropped = paired_deltas(
                        a_recs, b_recs, "acc_exact")
                    d1, _, _ = paired_deltas(a_recs, b_recs, "acc_1")
                    mean, lo, hi = paired_bootstrap(deltas)
                    rows.append({
                        "cell": f"{arch}|{recipe}", "arch": arch,
                        "recipe_actual": recipe, "contrast": label,
                        "pair_label": f"{o['run_id']} - {b['run_id']}",
                        "pair_kind": kind, "stage": stage,
                        "transform": transform, "descriptor": descriptor,
                        "n_pairs": 1, "n_images_paired": n_paired,
                        "n_images_dropped": n_dropped,
                        "delta_acc_exact": mean, "ci_lo": lo, "ci_hi": hi,
                        "delta_acc_1": mean_or_missing(d1)[0],
                        "a_acc_exact": mean_or_missing(
                            [value(x, "acc_exact") for x in a_recs])[0],
                        "b_acc_exact": mean_or_missing(
                            [value(x, "acc_exact") for x in b_recs])[0],
                        "chance_exact": value(a_recs[0], "chance_exact"),
                        # n = 2 for registers, and the table says so on every
                        # row rather than in a caption a reader may not reach.
                        "endpoint_class": (
                            DESCRIPTIVE if variant == "registers"
                            else SECONDARY),
                    })
    return rows


def _axes(loaded, cohort):
    """Every (stage, transform, descriptor) the records actually contain."""
    for r in cohort:
        recs = loaded.get(r["run_id"])
        if recs:
            seen = sorted({(x["stage"], x["transform"], x["descriptor"])
                           for x in recs if x["transform"]
                           in MATCHED_TRANSFORMS})
            return seen
    return []


def _select(loaded, run_id, condition, stage, transform, descriptor):
    recs = loaded.get(run_id)
    if not recs:
        return None
    out = [r for r in recs
           if r["condition_id"] == condition and r["stage"] == stage
           and r["transform"] == transform and r["descriptor"] == descriptor]
    return out or None


# ─────────────────────────────────────────────────────────────────────────────
# T_I5c — the terminal-gate bypass, on the feature consumer
# ─────────────────────────────────────────────────────────────────────────────

C_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "stage",
            "transform", "descriptor", "n_images_paired",
            "acc_native", "acc_term_1.00", "delta", "ci_lo", "ci_hi",
            "max_abs_s11_diff_vs_native", "chance_exact", "endpoint_class")


def build_c(loaded, cohort):
    """native vs term_1.00 on the 8 SAGA checkpoints.

    I2 measured that a CLS-only classifier cannot see this gate at all (max
    abs logit difference exactly 0, 8 of 8). This table asks the next
    question with a consumer that reads the patches: can a FEATURE CONSUMER
    see it? `s11_out` rows are the control — the bypass swaps the LAST
    block's gate, so that stage cannot move, and
    `max_abs_s11_diff_vs_native` carries the measured evidence.
    """
    rows = []
    for r in cohort:
        if r["variant"] != "saga":
            continue
        recs = loaded.get(r["run_id"])
        if not recs:
            continue
        for stage, transform, descriptor in sorted(
                {(x["stage"], x["transform"], x["descriptor"]) for x in recs}):
            nat = _select(loaded, r["run_id"], "native", stage, transform,
                          descriptor)
            byp = _select(loaded, r["run_id"], "term_1.00", stage, transform,
                          descriptor)
            if nat is None or byp is None:
                continue
            deltas, n_paired, _ = paired_deltas(byp, nat, "acc_exact")
            mean, lo, hi = paired_bootstrap(deltas)
            s11 = [value(x, "max_abs_s11_diff_vs_native") for x in byp]
            rows.append({
                "run_id": r["run_id"], "arch": r["arch"],
                "recipe_actual": r["recipe_actual"], "variant": r["variant"],
                "stage": stage, "transform": transform,
                "descriptor": descriptor, "n_images_paired": n_paired,
                "acc_native": mean_or_missing(
                    [value(x, "acc_exact") for x in nat])[0],
                "acc_term_1.00": mean_or_missing(
                    [value(x, "acc_exact") for x in byp])[0],
                "delta": mean, "ci_lo": lo, "ci_hi": hi,
                "max_abs_s11_diff_vs_native": max(
                    [v for v in s11 if v is not MISSING], default=MISSING),
                "chance_exact": value(nat[0], "chance_exact"),
                "endpoint_class": SECONDARY,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I5d — exceedance positions vs the rest
# ─────────────────────────────────────────────────────────────────────────────

D_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "condition_id",
            "stage", "transform", "descriptor",
            "n_images", "n_images_with_exceedance",
            "mean_n_shared_exc", "mean_n_shared_nonexc",
            "acc_exc", "acc_exc_ci_lo", "acc_exc_ci_hi",
            "acc_nonexc", "acc_nonexc_ci_lo", "acc_nonexc_ci_hi",
            "delta_exc_minus_nonexc", "delta_ci_lo", "delta_ci_hi",
            "chance_exact", "endpoint_class")


def build_d(loaded, cohort):
    rows = []
    for r in cohort:
        recs = loaded.get(r["run_id"])
        if not recs:
            continue
        for key, g in group(recs, "condition_id", "stage", "transform",
                            "descriptor").items():
            cond, stage, transform, descriptor = key
            exc = [value(x, "acc_exact_exc") for x in g]
            non = [value(x, "acc_exact_nonexc") for x in g]
            # The paired delta uses only images that scored BOTH halves; an
            # image with no exceedance position contributes to neither, and
            # `n_images_with_exceedance` says how many that was.
            both = [(a, b) for a, b in zip(exc, non)
                    if a is not MISSING and b is not MISSING]
            deltas = [a - b for a, b in both]
            m_e, lo_e, hi_e = bootstrap_ci(exc)
            m_n, lo_n, hi_n = bootstrap_ci(non)
            d_m, d_lo, d_hi = paired_bootstrap(deltas)
            rows.append({
                "run_id": r["run_id"], "arch": r["arch"],
                "recipe_actual": r["recipe_actual"], "variant": r["variant"],
                "condition_id": cond, "stage": stage, "transform": transform,
                "descriptor": descriptor, "n_images": len(g),
                "n_images_with_exceedance": sum(
                    1 for v in exc if v is not MISSING),
                "mean_n_shared_exc": mean_or_missing(
                    [float(x.get("n_shared_exc") or 0) for x in g])[0],
                "mean_n_shared_nonexc": mean_or_missing(
                    [float(x.get("n_shared_nonexc") or 0) for x in g])[0],
                "acc_exc": m_e, "acc_exc_ci_lo": lo_e, "acc_exc_ci_hi": hi_e,
                "acc_nonexc": m_n, "acc_nonexc_ci_lo": lo_n,
                "acc_nonexc_ci_hi": hi_n,
                "delta_exc_minus_nonexc": d_m, "delta_ci_lo": d_lo,
                "delta_ci_hi": d_hi,
                "chance_exact": value(g[0], "chance_exact"),
                "endpoint_class": SECONDARY,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I5e — does a diagnostic predict the readout?
# ─────────────────────────────────────────────────────────────────────────────

E_FIELDS = ("scope", "run_id", "arch", "recipe_actual", "variant", "stage",
            "transform", "descriptor", "diagnostic", "n",
            "spearman_rho", "mean_acc_exact", "mean_diagnostic",
            "endpoint_class")


def build_e(loaded, cohort):
    """Within checkpoint: Spearman over images. Across checkpoints: the
    scatter data ONLY — no test, because 19 checkpoints of mixed provenance
    are not 19 independent draws and a p-value over them would say so
    without saying it.
    """
    rows = []
    across = []
    for r in cohort:
        recs = loaded.get(r["run_id"])
        if not recs:
            continue
        for key, g in group(recs, "stage", "transform", "descriptor").items():
            stage, transform, descriptor = key
            g = [x for x in g if x["condition_id"] == "native"]
            if not g:
                continue
            acc = [value(x, "acc_exact") for x in g]
            for diag_key in ("count_mad_s11", "count_mad_stage"):
                diag = [value(x, diag_key) for x in g]
                rho, n = spearman(acc, diag)
                rows.append({
                    "scope": "within_checkpoint", "run_id": r["run_id"],
                    "arch": r["arch"], "recipe_actual": r["recipe_actual"],
                    "variant": r["variant"], "stage": stage,
                    "transform": transform, "descriptor": descriptor,
                    "diagnostic": diag_key, "n": n, "spearman_rho": rho,
                    "mean_acc_exact": mean_or_missing(acc)[0],
                    "mean_diagnostic": mean_or_missing(diag)[0],
                    "endpoint_class": SECONDARY,
                })
                across.append({
                    "scope": "across_checkpoints", "run_id": r["run_id"],
                    "arch": r["arch"], "recipe_actual": r["recipe_actual"],
                    "variant": r["variant"], "stage": stage,
                    "transform": transform, "descriptor": descriptor,
                    "diagnostic": diag_key, "n": len(g),
                    "spearman_rho": MISSING,          # no test across 19
                    "mean_acc_exact": mean_or_missing(acc)[0],
                    "mean_diagnostic": mean_or_missing(diag)[0],
                    "endpoint_class": DESCRIPTIVE,
                })
    return rows + across


# ─────────────────────────────────────────────────────────────────────────────
# T_I5f / T_I5g — I5b, the frozen ADE20K heads (conditional on the manifest)
# ─────────────────────────────────────────────────────────────────────────────

F_FIELDS = ("contrast", "a_run_id", "b_run_id", "backbone_matched",
            "condition_id", "n_images_paired", "n_images_dropped",
            "miou_ss_a", "miou_ss_b", "miou_ss_delta",
            "pixel_acc_a", "pixel_acc_b",
            "delta_miou_present_paired", "ci_lo", "ci_hi",
            "delta_pixel_acc_paired", "pixel_ci_lo", "pixel_ci_hi",
            "endpoint_class", "note")

F_CLASS_FIELDS = ("contrast", "a_run_id", "b_run_id", "condition_id",
                  "class_index", "iou_a", "iou_b", "iou_delta",
                  "gt_pixels_a", "gt_pixels_b", "endpoint_class")

G_FIELDS = ("run_id", "variant", "backbone_matched", "n_images_paired",
            "miou_ss_native", "miou_ss_term_1.00", "miou_ss_delta",
            "delta_miou_present_paired", "ci_lo", "ci_hi",
            "endpoint_class", "note")

#: The dataset-level mIoU NEVER comes from averaging `miou_present`: it is
#: read from the pooled confusion matrix the evaluator wrote, which is the
#: TASK-09 B4-correct metric. This constant is the file that carries it.
CONF_STEM = "conf_matrix_{condition}.npz"


def load_seg(seg_root):
    """{run_id: {"rows": [...], "pooled": {condition: npz dict}}} or {}."""
    seg_root = Path(seg_root)
    if not seg_root.exists():
        return {}
    out = {}
    for run_dir in sorted(p for p in seg_root.iterdir() if p.is_dir()):
        rows = read_records(records_path(run_dir, "seg_records"))
        if not rows:
            continue
        pooled = {}
        for npz in sorted(run_dir.glob("conf_matrix_*.npz")):
            cond = npz.stem[len("conf_matrix_"):]
            with np.load(npz) as z:
                pooled[cond] = {k: z[k] for k in z.files}
        out[run_dir.name] = {"rows": rows, "pooled": pooled}
    return out


def _seg_miou(pooled, condition):
    """mIoU (%) and pixel accuracy from the POOLED confusion matrix."""
    data = pooled.get(condition)
    if data is None or "conf" not in data:
        return MISSING, MISSING
    import torch
    from segmentation.tools.train import summarize_confusion
    summary, *_ = summarize_confusion(torch.as_tensor(data["conf"]))
    return summary["mIoU"], summary["pixel_acc"]


def _seg_by_image(rows, condition, key):
    return {r["image_id"]: value(r, key) for r in rows
            if r["condition_id"] == condition}


def build_f(seg):
    """T_I5f — the frozen ADE20K comparison, paired per image.

    `miou_ss_*` is the DATASET metric from the pooled confusion matrix.
    `delta_miou_present_paired` is the paired per-image difference, which is
    the quantity that has an interval; the two answer different questions and
    the table carries both rather than letting one stand in for the other.
    """
    rows, class_rows = [], []
    a_id = next((k for k, v in seg.items()
                 if v["rows"] and v["rows"][0]["variant"] == "saga"), None)
    b_id = next((k for k, v in seg.items()
                 if v["rows"] and v["rows"][0]["variant"] == "baseline"), None)
    if a_id is None or b_id is None:
        return rows, class_rows
    a, b = seg[a_id], seg[b_id]
    matched = int(all(str(r.get("backbone_matched")) in ("1", "True", "true")
                      for r in (a["rows"][0], b["rows"][0])))

    for cond in ("native",):
        ai = _seg_by_image(a["rows"], cond, "miou_present")
        bi = _seg_by_image(b["rows"], cond, "miou_present")
        shared = [i for i in ai if i in bi]
        deltas = [ai[i] - bi[i] for i in shared
                  if ai[i] is not MISSING and bi[i] is not MISSING]
        mean, lo, hi = paired_bootstrap(deltas)
        ap = _seg_by_image(a["rows"], cond, "pixel_acc")
        bp = _seg_by_image(b["rows"], cond, "pixel_acc")
        pdel = [ap[i] - bp[i] for i in shared
                if ap.get(i) is not MISSING and bp.get(i) is not MISSING]
        pm, plo, phi = paired_bootstrap(pdel)
        miou_a, pix_a = _seg_miou(a["pooled"], cond)
        miou_b, pix_b = _seg_miou(b["pooled"], cond)
        rows.append({
            "contrast": "saga_minus_baseline", "a_run_id": a_id,
            "b_run_id": b_id, "backbone_matched": matched,
            "condition_id": cond, "n_images_paired": len(shared),
            "n_images_dropped": len(ai) + len(bi) - 2 * len(shared),
            "miou_ss_a": miou_a, "miou_ss_b": miou_b,
            "miou_ss_delta": (miou_a - miou_b
                              if MISSING not in (miou_a, miou_b) else MISSING),
            "pixel_acc_a": pix_a, "pixel_acc_b": pix_b,
            "delta_miou_present_paired": mean, "ci_lo": lo, "ci_hi": hi,
            "delta_pixel_acc_paired": pm, "pixel_ci_lo": plo,
            "pixel_ci_hi": phi,
            "endpoint_class": SECONDARY,
            "note": "miou_ss_* is the dataset metric from the pooled "
                    "confusion matrix; delta_miou_present_paired is the "
                    "paired per-image difference and is the quantity the CI "
                    "belongs to",
        })
        ia = a["pooled"].get(cond, {}).get("iou")
        ib = b["pooled"].get(cond, {}).get("iou")
        if ia is not None and ib is not None:
            ga = a["pooled"][cond].get("gt_pixels")
            gb = b["pooled"][cond].get("gt_pixels")
            for c in range(len(ia)):
                va, vb = float(ia[c]), float(ib[c])
                class_rows.append({
                    "contrast": "saga_minus_baseline", "a_run_id": a_id,
                    "b_run_id": b_id, "condition_id": cond,
                    "class_index": c,
                    "iou_a": MISSING if not np.isfinite(va) else va,
                    "iou_b": MISSING if not np.isfinite(vb) else vb,
                    "iou_delta": (va - vb if np.isfinite(va)
                                  and np.isfinite(vb) else MISSING),
                    "gt_pixels_a": (int(ga[c]) if ga is not None else MISSING),
                    "gt_pixels_b": (int(gb[c]) if gb is not None else MISSING),
                    "endpoint_class": SECONDARY,
                })
    return rows, class_rows


def build_g(seg):
    """T_I5g — the terminal-gate bypass on the SAGA segmentation head.

    EXPLORATORY, and labelled so on every row: the head was TRAINED on gated
    features, so a drop under the bypass measures sensitivity to an input
    distribution the head never saw, not the utility of the features
    themselves (§4).
    """
    rows = []
    for run_id, data in sorted(seg.items()):
        conds = {r["condition_id"] for r in data["rows"]}
        if "term_1.00" not in conds:
            continue
        nat = _seg_by_image(data["rows"], "native", "miou_present")
        byp = _seg_by_image(data["rows"], "term_1.00", "miou_present")
        shared = [i for i in nat if i in byp]
        deltas = [byp[i] - nat[i] for i in shared
                  if nat[i] is not MISSING and byp[i] is not MISSING]
        mean, lo, hi = paired_bootstrap(deltas)
        m_nat, _ = _seg_miou(data["pooled"], "native")
        m_byp, _ = _seg_miou(data["pooled"], "term_1.00")
        rows.append({
            "run_id": run_id, "variant": data["rows"][0]["variant"],
            "backbone_matched": data["rows"][0].get("backbone_matched"),
            "n_images_paired": len(shared),
            "miou_ss_native": m_nat, "miou_ss_term_1.00": m_byp,
            "miou_ss_delta": (m_byp - m_nat
                              if MISSING not in (m_nat, m_byp) else MISSING),
            "delta_miou_present_paired": mean, "ci_lo": lo, "ci_hi": hi,
            "endpoint_class": DESCRIPTIVE,
            "note": "EXPLORATORY: the head was trained on gated features, so "
                    "this measures sensitivity to an unseen input "
                    "distribution, not feature utility",
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

TABLES = (
    ("T_I5a_readout", A_FIELDS, build_a),
    ("T_I5b_methods", B_FIELDS, build_b),
    ("T_I5c_terminal", C_FIELDS, build_c),
    ("T_I5d_positions", D_FIELDS, build_d),
    ("T_I5e_diag_vs_readout", E_FIELDS, build_e),
)


def build_all(root, manifest, out_dir):
    """Every I5a table, from the records under `root`. Returns a report."""
    cohort = eligible_cohort(manifest)
    loaded, missing = {}, []
    for r in cohort:
        recs = load_run(root, r["run_id"])
        if recs is None:
            missing.append(str(records_path(Path(root) / r["run_id"],
                                            "corr_records")))
        else:
            loaded[r["run_id"]] = recs

    written = {}
    for name, fields, builder in TABLES:
        rows = builder(loaded, cohort)
        path = write_csv(Path(out_dir) / f"{name}.csv", fields, rows)
        written[name] = {"path": str(path), "n_rows": len(rows)}
    return {"n_runs_loaded": len(loaded), "n_runs_missing": len(missing),
            "missing_records": missing, "tables": written,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES,
                          "seed": BOOTSTRAP_SEED},
            "endpoint_note": "Every I5 comparison is a SECONDARY endpoint "
                             "under D7; the three primary contrasts are "
                             "C1-C3 in Track B."}


def build_seg_tables(seg_root, out_dir):
    """T_I5f / T_I5g, or a recorded DROP. Never a partial table.

    I5b is conditional (§2): if its records are not there, this writes
    nothing and says so, and the handoff carries `I5b: DROPPED (reason)`. It
    does not fall back to the historical aggregate mIoU, which has no
    pairing and is the number I5b exists to improve on.
    """
    seg = load_seg(seg_root)
    if not seg:
        return {"status": "MISSING",
                "reason": f"no seg records under {seg_root}",
                "tables": {}}
    f_rows, class_rows = build_f(seg)
    g_rows = build_g(seg)
    written = {
        "T_I5f_seg": {"path": str(write_csv(
            Path(out_dir) / "T_I5f_seg.csv", F_FIELDS, f_rows)),
            "n_rows": len(f_rows)},
        "T_I5f_seg_per_class": {"path": str(write_csv(
            Path(out_dir) / "T_I5f_seg_per_class.csv", F_CLASS_FIELDS,
            class_rows)), "n_rows": len(class_rows)},
        "T_I5g_seg_terminal": {"path": str(write_csv(
            Path(out_dir) / "T_I5g_seg_terminal.csv", G_FIELDS, g_rows)),
            "n_rows": len(g_rows)},
    }
    return {"status": "OK", "runs": sorted(seg), "tables": written}


def main():
    p = argparse.ArgumentParser(description="TASK C / I5 — build the tables.")
    p.add_argument("--split", default="sub2k")
    p.add_argument("--root", default=None,
                   help="results/frozen/I5_readout/<split> by default")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out", default=None,
                   help="<root>/tables by default")
    p.add_argument("--seg-root", default=None,
                   help="results/frozen/I5_readout/seg by default (I5b)")
    args = p.parse_args()

    root = Path(args.root or REPO / "results" / "frozen" / WORK_PACKAGE
                / args.split)
    out_dir = Path(args.out or root / "tables")
    if not root.exists():
        raise SystemExit(
            f"{root} does not exist — the I5a sweep has not landed. Nothing "
            f"is estimated from a sibling split; run Phase B first.")
    report = build_all(root, args.manifest, out_dir)
    report["I5b"] = build_seg_tables(
        Path(args.seg_root or REPO / "results" / "frozen" / WORK_PACKAGE
             / "seg"), out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
