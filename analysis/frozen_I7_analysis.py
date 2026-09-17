#!/usr/bin/env python3
"""
analysis/frozen_I7_analysis.py
==============================
TASK C / I7 §5 — the incoming-attention tables, built from the committed
`incoming_mass.npz` files and nothing else.

    python analysis/frozen_I7_analysis.py --split sub1k

    T_I7a_incoming     per checkpoint x block x query group: mean incoming
                       mass per exceedance token vs per non-exceedance token;
                       the RATIO with an image-level bootstrap CI; the share
                       of total mass exceedance tokens receive against their
                       share of positions; the AUC of incoming mass for the
                       exceedance flag
    T_I7b_registers    register checkpoints: mass landing on the 4 register
                       tokens vs CLS vs the patch exceedances (DESCRIPTIVE)
    T_I7c_value_norm   value norm at exceedance vs non-exceedance positions
                       (EXPLORATORY)
    T_I7d_ttr_curve    built by analysis/frozen_I7_ttr_curve.py from the
                       existing sweep files

THE RATIO IS A RATIO OF MEANS, NOT A MEAN OF RATIOS
---------------------------------------------------
Per image, mass-per-exceedance-token and mass-per-non-exceedance-token are
both means over positions. The statistic D10 names is the ratio of those two
quantities, and it is computed as

    mean_over_images(exceedance_mean_i) / mean_over_images(other_mean_i)

not as `mean_over_images(exceedance_mean_i / other_mean_i)`. The second is a
different estimator with a much heavier tail: an image whose non-exceedance
mean is near zero contributes an enormous ratio and can carry the average on
its own. The bootstrap resamples IMAGES and recomputes the ratio of means on
each resample, so the interval belongs to the statistic being reported.

Images with no exceedance position contribute no exceedance mean; they are
dropped from that side and COUNTED (`n_images_with_exceedance`), never
counted as zero.

EVERY ROW IS STAMPED
--------------------
`endpoint_class` on every row: `secondary` for the incoming-mass comparison
(D7 fixed the primary contrasts in Track B) and `exploratory` for the
value-norm and register-token rows, which §5 labels as such. None of it
decides a primary claim, and the wording verdict is NOT computed here — it
lives in `analysis/i7_wording.py`, is run once, and its sentence is copied
rather than rewritten.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from saga.frozen.runner import eligible_cohort  # noqa: E402

MISSING = "MISSING"

#: LOCKED_ANALYSIS §8 / D6.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CHUNK = 500
CI_PERCENTILES = (2.5, 97.5)

SECONDARY = "secondary"
EXPLORATORY = "exploratory"
DESCRIPTIVE = "descriptive"

WORK_PACKAGE = "I7_attention"

#: The two query groups, and the npz array each reads.
QUERY_GROUPS = {"patch": "in_mass_patchq", "cls": "in_mass_clsq"}

#: The exceedance bases, primary first (D10: MAD is primary).
BASES = ("mad", "tau_cal")


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────

def load_run(root, run_id):
    """One run's `incoming_mass.npz` as a dict, or None."""
    path = Path(root) / run_id / "incoming_mass.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out.pop("meta_json")))
    return out


def blocks_of(data):
    return [int(b) for b in data["blocks"]]


def patch_slice(data):
    """The key positions that are PATCH tokens, from the model's own count."""
    n_prefix = int(data["meta"]["n_prefix"])
    return slice(n_prefix, None), n_prefix


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _resample_index(n, resamples, seed):
    rng = np.random.RandomState(seed)
    done = 0
    while done < resamples:
        take = min(BOOTSTRAP_CHUNK, resamples - done)
        yield rng.randint(0, n, size=(take, n))
        done += take


def ratio_of_means_ci(exc_mean, other_mean, *, resamples=BOOTSTRAP_RESAMPLES,
                      seed=BOOTSTRAP_SEED):
    """(ratio, lo, hi) for mean(exc) / mean(other), bootstrapped over IMAGES.

    `exc_mean[i]` and `other_mean[i]` are one image's two per-position means;
    an image missing either side is dropped by the caller and counted. The
    SAME resampled image indices are used for both numerator and denominator,
    which is what keeps the pairing intact.
    """
    a = np.asarray(exc_mean, dtype=np.float64)
    b = np.asarray(other_mean, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return MISSING, MISSING, MISSING
    denom = b.mean()
    if not np.isfinite(denom) or denom == 0:
        return MISSING, MISSING, MISSING
    point = float(a.mean() / denom)

    stats = np.empty(resamples, dtype=np.float64)
    pos = 0
    for idx in _resample_index(a.size, resamples, seed):
        num = a[idx].mean(axis=1)
        den = b[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            stats[pos:pos + idx.shape[0]] = np.where(den != 0, num / den,
                                                     np.nan)
        pos += idx.shape[0]
    good = stats[np.isfinite(stats)]
    if good.size == 0:
        return point, MISSING, MISSING
    lo, hi = np.percentile(good, CI_PERCENTILES)
    return point, float(lo), float(hi)


def mean_ci(values, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """(mean, lo, hi) by image-level bootstrap; MISSING dropped."""
    arr = np.asarray([v for v in values if v is not MISSING and
                      np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return MISSING, MISSING, MISSING
    stats = np.empty(resamples, dtype=np.float64)
    pos = 0
    for idx in _resample_index(arr.size, resamples, seed):
        stats[pos:pos + idx.shape[0]] = arr[idx].mean(axis=1)
        pos += idx.shape[0]
    lo, hi = np.percentile(stats, CI_PERCENTILES)
    return float(arr.mean()), float(lo), float(hi)


def auc_per_image(scores, labels):
    """Mann-Whitney AUC of `scores` for the binary `labels`, per image.

    Rank-based with ties averaged, which is the definition that matches the
    probability statement "a randomly chosen exceedance token receives more
    incoming mass than a randomly chosen other token". An image with only one
    class present has no AUC and yields MISSING rather than 0.5.
    """
    out = []
    for s, y in zip(scores, labels):
        pos = int(y.sum())
        neg = int(y.size - pos)
        if pos == 0 or neg == 0:
            out.append(MISSING)
            continue
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(s.size, dtype=np.float64)
        ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
        # average ranks within ties
        sorted_s = s[order]
        i = 0
        while i < sorted_s.size:
            j = i
            while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
            i = j + 1
        out.append(float((ranks[y].sum() - pos * (pos + 1) / 2.0)
                         / (pos * neg)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-run reduction
# ─────────────────────────────────────────────────────────────────────────────

def per_image_split(data, block_pos, query_group, basis):
    """Per-image (exceedance mean, other mean, share, positions share, AUC).

    Returns a dict of lists, one entry per image, with MISSING wherever a
    quantity is undefined for that image.
    """
    key = f"exceedance_{basis}"
    if key not in data:
        return None
    mass = data[QUERY_GROUPS[query_group]][:, block_pos, :]   # [N, T]
    flags = data[key][:, block_pos, :].astype(bool)           # [N, P]
    sl, _ = patch_slice(data)
    patch_mass = mass[:, sl]                                  # [N, P]
    if patch_mass.shape[1] != flags.shape[1]:
        raise ValueError(
            f"{data['meta']['run_id']}: {patch_mass.shape[1]} patch key "
            f"positions but {flags.shape[1]} exceedance flags — the prefix "
            f"count and the flag array disagree")

    exc_mean, other_mean, share, pos_share = [], [], [], []
    for i in range(patch_mass.shape[0]):
        f = flags[i]
        m = patch_mass[i].astype(np.float64)
        n_exc = int(f.sum())
        total = float(m.sum())
        exc_mean.append(float(m[f].mean()) if n_exc else MISSING)
        other_mean.append(float(m[~f].mean()) if n_exc < f.size else MISSING)
        share.append(float(m[f].sum() / total)
                     if n_exc and total > 0 else MISSING)
        pos_share.append(float(n_exc / f.size))
    return {
        "exc_mean": exc_mean, "other_mean": other_mean,
        "mass_share": share, "position_share": pos_share,
        "auc": auc_per_image(patch_mass.astype(np.float64), flags),
        "n_images": patch_mass.shape[0],
        "n_exceedance_positions": [int(f.sum()) for f in flags],
    }


def paired_lists(split):
    """The images where BOTH means exist, as two aligned arrays."""
    a, b = [], []
    for x, y in zip(split["exc_mean"], split["other_mean"]):
        if x is not MISSING and y is not MISSING:
            a.append(x)
            b.append(y)
    return a, b


def clean(values):
    return [v for v in values if v is not MISSING]


# ─────────────────────────────────────────────────────────────────────────────
# T_I7a — the incoming-mass table
# ─────────────────────────────────────────────────────────────────────────────

A_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "provenance_tag",
            "block", "aligned_stage", "query_group", "basis",
            "n_images", "n_images_with_exceedance",
            "mean_n_exceedance_positions",
            "mass_per_exceedance_token", "mass_per_other_token",
            "ratio", "ratio_ci_lo", "ratio_ci_hi",
            "mass_share_exceedance", "position_share_exceedance",
            "share_over_position_share", "auc", "auc_ci_lo", "auc_ci_hi",
            "tau_cal_s11_out", "endpoint_class")


def build_a(loaded, cohort):
    rows = []
    for r in cohort:
        data = loaded.get(r["run_id"])
        if data is None:
            continue
        blocks = blocks_of(data)
        stages = [str(s) for s in data["aligned_stage"]]
        for bi, block in enumerate(blocks):
            for qg in QUERY_GROUPS:
                for basis in BASES:
                    split = per_image_split(data, bi, qg, basis)
                    if split is None:
                        continue
                    a, b = paired_lists(split)
                    ratio, lo, hi = ratio_of_means_ci(a, b)
                    auc, auc_lo, auc_hi = mean_ci(clean(split["auc"]))
                    share = np.mean(clean(split["mass_share"])) if clean(
                        split["mass_share"]) else MISSING
                    pshare = float(np.mean(split["position_share"]))
                    rows.append({
                        "run_id": r["run_id"], "arch": r["arch"],
                        "recipe_actual": r["recipe_actual"],
                        "variant": r["variant"],
                        "provenance_tag": r.get("provenance_tag", MISSING),
                        "block": block, "aligned_stage": stages[bi],
                        "query_group": qg, "basis": basis,
                        "n_images": split["n_images"],
                        "n_images_with_exceedance": len(a),
                        "mean_n_exceedance_positions": float(
                            np.mean(split["n_exceedance_positions"])),
                        "mass_per_exceedance_token": (
                            float(np.mean(a)) if a else MISSING),
                        "mass_per_other_token": (
                            float(np.mean(b)) if b else MISSING),
                        "ratio": ratio, "ratio_ci_lo": lo, "ratio_ci_hi": hi,
                        "mass_share_exceedance": share,
                        "position_share_exceedance": pshare,
                        "share_over_position_share": (
                            share / pshare if share is not MISSING
                            and pshare > 0 else MISSING),
                        "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
                        "tau_cal_s11_out": data["meta"].get(
                            "tau_cal_s11_out", MISSING),
                        "endpoint_class": SECONDARY,
                    })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I7b — where a register model's mass actually lands
# ─────────────────────────────────────────────────────────────────────────────

B_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "block",
            "query_group", "n_images", "n_prefix",
            "mass_cls_token", "mass_register_tokens_total",
            "mass_per_register_token", "mass_patch_exceedance_total",
            "mass_patch_total", "n_register_tokens",
            "register_index", "mass_this_register_token", "endpoint_class")


def build_b(loaded, cohort):
    """Register checkpoints only. DESCRIPTIVE: n = 2, and the two register
    checkpoints in this cell are legacy repeats with no seed, so nothing here
    is a paired contrast."""
    rows = []
    for r in cohort:
        if r["variant"] != "registers":
            continue
        data = loaded.get(r["run_id"])
        if data is None:
            continue
        n_prefix = int(data["meta"]["n_prefix"])
        n_reg = max(0, n_prefix - 1)
        sl, _ = patch_slice(data)
        for bi, block in enumerate(blocks_of(data)):
            for qg in QUERY_GROUPS:
                mass = data[QUERY_GROUPS[qg]][:, bi, :].astype(np.float64)
                flags = (data["exceedance_mad"][:, bi, :].astype(bool)
                         if "exceedance_mad" in data else None)
                patch = mass[:, sl]
                cls_mass = float(mass[:, 0].mean())
                reg = mass[:, 1:n_prefix] if n_reg else None
                exc_total = (float(np.mean([patch[i][flags[i]].sum()
                                            for i in range(patch.shape[0])]))
                             if flags is not None else MISSING)
                base = {
                    "run_id": r["run_id"], "arch": r["arch"],
                    "recipe_actual": r["recipe_actual"],
                    "variant": r["variant"], "block": block,
                    "query_group": qg, "n_images": int(mass.shape[0]),
                    "n_prefix": n_prefix,
                    "mass_cls_token": cls_mass,
                    "mass_register_tokens_total": (
                        float(reg.sum(axis=1).mean()) if reg is not None
                        else MISSING),
                    "mass_per_register_token": (
                        float(reg.mean()) if reg is not None else MISSING),
                    "mass_patch_exceedance_total": exc_total,
                    "mass_patch_total": float(patch.sum(axis=1).mean()),
                    "n_register_tokens": n_reg,
                    "endpoint_class": DESCRIPTIVE,
                }
                rows.append(dict(base, register_index="all",
                                 mass_this_register_token=MISSING))
                for j in range(n_reg):
                    rows.append(dict(
                        base, register_index=j,
                        mass_this_register_token=float(
                            mass[:, 1 + j].mean())))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I7c — the value-norm diagnostic (EXPLORATORY)
# ─────────────────────────────────────────────────────────────────────────────

C_FIELDS = ("run_id", "arch", "recipe_actual", "variant", "block",
            "aligned_stage", "basis", "n_images", "n_images_with_exceedance",
            "value_norm_exceedance", "value_norm_other", "ratio",
            "ratio_ci_lo", "ratio_ci_hi", "endpoint_class", "note")


def build_c(loaded, cohort):
    rows = []
    for r in cohort:
        data = loaded.get(r["run_id"])
        if data is None:
            continue
        sl, _ = patch_slice(data)
        stages = [str(s) for s in data["aligned_stage"]]
        for bi, block in enumerate(blocks_of(data)):
            for basis in BASES:
                key = f"exceedance_{basis}"
                if key not in data:
                    continue
                vn = data["value_norm"][:, bi, sl].astype(np.float64)
                flags = data[key][:, bi, :].astype(bool)
                a, b = [], []
                for i in range(vn.shape[0]):
                    f = flags[i]
                    if not f.any() or f.all():
                        continue
                    a.append(float(vn[i][f].mean()))
                    b.append(float(vn[i][~f].mean()))
                ratio, lo, hi = ratio_of_means_ci(a, b)
                rows.append({
                    "run_id": r["run_id"], "arch": r["arch"],
                    "recipe_actual": r["recipe_actual"],
                    "variant": r["variant"], "block": block,
                    "aligned_stage": stages[bi], "basis": basis,
                    "n_images": int(vn.shape[0]),
                    "n_images_with_exceedance": len(a),
                    "value_norm_exceedance": (float(np.mean(a)) if a
                                              else MISSING),
                    "value_norm_other": (float(np.mean(b)) if b else MISSING),
                    "ratio": ratio, "ratio_ci_lo": lo, "ratio_ci_hi": hi,
                    "endpoint_class": EXPLORATORY,
                    "note": "value norm is the Fesser et al. no-op/broadcast "
                            "diagnostic; EXPLORATORY, and it decides nothing "
                            "about the wording rule",
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v) -> str:
    if v is MISSING or v == MISSING or v is None:
        return MISSING
    if isinstance(v, (bool, np.bool_)):
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


TABLES = (
    ("T_I7a_incoming", A_FIELDS, build_a),
    ("T_I7b_registers", B_FIELDS, build_b),
    ("T_I7c_value_norm", C_FIELDS, build_c),
)


def build_all(root, manifest, out_dir, *, cell=None):
    """Every I7 table from the npz files under `root`. Returns a report."""
    cohort = eligible_cohort(manifest)
    if cell:
        arch, recipe = cell.split("|", 1)
        cohort = [r for r in cohort
                  if r["arch"] == arch and r["recipe_actual"] == recipe]
    loaded, missing = {}, []
    for r in cohort:
        data = load_run(root, r["run_id"])
        if data is None:
            missing.append(str(Path(root) / r["run_id"] / "incoming_mass.npz"))
        else:
            loaded[r["run_id"]] = data

    written = {}
    for name, fields, builder in TABLES:
        rows = builder(loaded, cohort)
        written[name] = {"path": str(write_csv(Path(out_dir) / f"{name}.csv",
                                               fields, rows)),
                         "n_rows": len(rows)}
    return {"n_runs_loaded": len(loaded), "n_runs_missing": len(missing),
            "missing_npz": missing, "tables": written,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES,
                          "seed": BOOTSTRAP_SEED},
            "endpoint_note": "I7 is a SECONDARY endpoint under D7; the "
                             "value-norm and register-token rows are "
                             "EXPLORATORY. The wording verdict is produced "
                             "by analysis/i7_wording.py, once, and copied."}


def main():
    p = argparse.ArgumentParser(description="TASK C / I7 — build the tables.")
    p.add_argument("--split", default="sub1k")
    p.add_argument("--root", default=None)
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out", default=None)
    p.add_argument("--cell", default="vit_small|mixup",
                   help="the declared I7 cell (§2); pass '' for all")
    args = p.parse_args()

    root = Path(args.root or REPO / "results" / "frozen" / WORK_PACKAGE
                / args.split)
    out_dir = Path(args.out or root / "tables")
    if not root.exists():
        raise SystemExit(
            f"{root} does not exist — the I7 sweep has not landed. Nothing "
            f"is estimated from a sibling split; run Phase B first.")
    report = build_all(root, args.manifest, out_dir, cell=args.cell or None)

    from analysis.frozen_I7_ttr_curve import build as build_curve
    report["ttr_curve"] = build_curve(out_dir)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
