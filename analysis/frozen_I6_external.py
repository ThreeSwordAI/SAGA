#!/usr/bin/env python3
"""
analysis/frozen_I6_external.py
==============================
TASK A / I6 A9 — `T_I6a_external`, and the prediction outcome per model.

    python analysis/frozen_I6_external.py

THE OUTCOME IS COMPUTED, NEVER TYPED. `prediction_outcome` in
`saga/frozen/external.py` applies the rule declared beside the prediction in
`configs/frozen/I6_models.yaml`:

    supervised_mixing      met <=> ring-1 peak AND Gini excess > 0
    no_mixing              met <=> NO ring-1 peak
    registers_exploratory  always "not applicable"

This module is committed in the same phase as the registry — before
`tools/frozen_I6_download.py` has fetched a single weight. That ordering is
what the pre-registration claim rests on: the code that decides "met" existed
before the data it would decide on.

NOTHING IS POOLED. One row per (model, stage, basis). No mean over a group,
no mean over models. The nine checkpoints differ in training data,
augmentation recipe, patch size and — for the three DINOv2 models — native
resolution, so a group average would be an average over incomparable things.
The groups exist to say which prediction applies to which model, and for
nothing else.

Unlike I1's tables before Phase B, I6's maps carry per-image INDICATORS from
the start (`saga/frozen/norms.py` writes them), so the bootstrap CIs and the
split-half reliability here are real numbers rather than `PENDING B`.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.address_analysis import (NULL_SEED, NULL_SIMS,  # noqa: E402
                                       border_distance_map, position_rel_se,
                                       ring_indices)
from saga.frozen import external as ext  # noqa: E402
from saga.frozen import norms as fnorms  # noqa: E402
from saga.frozen import prevalence as P  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"
I6_ROOT = Path("results/frozen/I6_external")
TABLE_DIR = I6_ROOT / "tables"
BASES = ("fixed_cal", "mad")
#: LOCKED_ANALYSIS §8; D6 = seed 0, CLOSED.
BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 10000
#: The widest grid in the registry is 16x16, which has 8 rings.
MAX_RINGS = 8

FIELDS = [
    "work_package", "generated_by", "git_sha",
    "model_id", "timm_name", "group", "mixing", "arch_note",
    "weight_sha256", "timm_version", "input_size", "native_input_size",
    "resolution_deviation", "patch_size", "grid_side", "n_patches",
    "n_prefix", "depth", "norm_mean", "norm_std", "crop_pct",
    "split_name", "split_sha256", "threshold_split_name",
    "threshold_split_sha256", "stage", "stage_resolved", "basis", "tau",
    "n_images", "count_per_image", "p_any",
    "entropy_normalized", "entropy_normalized_excess",
    "gini", "gini_null", "gini_excess",
    "top5_share", "top5_share_uniform", "top5_share_excess",
    "position_rel_se", "eta2_pos",
] + [f"ring{k}_freq" for k in range(MAX_RINGS)] + [
    "ring0_freq_ci_lo", "ring0_freq_ci_hi",
    "ring1_freq_ci_lo", "ring1_freq_ci_hi",
    "centre_freq", "centre_freq_ci_lo", "centre_freq_ci_hi",
    "ring1_share", "peak_ring", "ring_1_peak",
    "ring1_minus_ring0", "ring1_minus_ring0_ci_lo",
    "ring1_minus_ring0_ci_hi",
    "prediction_group", "prediction_outcome",
    "split_half_rho", "split_half_p", "split_half_n_transforms",
    "bootstrap_seed", "bootstrap_resamples", "source", "note",
]


def _num(v, nd=6):
    if v is None:
        return MISSING
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return MISSING if not np.isfinite(v) else round(v, nd)


def per_image_ring_means(indicators, side: int) -> np.ndarray:
    """`[n_images, n_rings]` — each image's mean indicator per border ring.

    The unit the image-level bootstrap resamples. Collapsing positions into
    ring means FIRST makes 10,000 resamples a matter of averaging a small
    matrix rather than re-reducing the whole indicator array each time.
    """
    ind = np.asarray(indicators, dtype=np.float64)
    n_rings = side // 2
    return np.column_stack([ind[:, ring_indices(side, k)].mean(axis=1)
                            for k in range(n_rings)])


def bootstrap_ci(per_image, *, seed=BOOTSTRAP_SEED,
                 resamples=BOOTSTRAP_RESAMPLES, alpha=0.05):
    """Percentile CI of the MEAN of a per-image quantity, resampling IMAGES.

    The unit of analysis is the image (LOCKED_ANALYSIS §8), so the resample
    is over images and not over positions: the 196 or 256 positions of one
    image are not independent observations of anything.
    """
    x = np.asarray(per_image, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 2:
        return None, None
    rng = np.random.RandomState(int(seed))
    means = np.empty(int(resamples), dtype=np.float64)
    for i in range(int(resamples)):
        means[i] = x[rng.randint(0, n, size=n)].mean()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def load_model_maps(model_id: str, *, root=I6_ROOT):
    """`{(stage, basis): (PrevalenceMap, indicators, meta)}` for one model."""
    out = {}
    for stage in ext.EXT_STAGES:
        path = Path(root) / model_id / f"maps_{stage}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            for basis in BASES:
                key = f"native|{basis}"
                if f"counts__{key}" not in z.files:
                    continue
                counts = z[f"counts__{key}"]
                n_images = int(z[f"n_images__{key}"])
                tau = (meta.get("tau_cal", {}) or {}).get(stage, MISSING)
                pm = P.from_counts(
                    counts, n_images=n_images, basis=basis,
                    threshold=(float(tau) if basis == "fixed_cal"
                               and isinstance(tau, (int, float)) else MISSING),
                    stage=stage, split_sha=meta["split_sha256"],
                    split_name=meta.get("split_name", MISSING),
                    condition_id="native", run_id=model_id,
                    ckpt_sha256=meta["ckpt_sha256"],
                    n_prefix=int(meta.get("model_n_prefix", 1)),
                    git_sha=meta["git_sha"])
                out[(stage, basis)] = (
                    pm, fnorms.read_indicators(path, "native", basis), meta,
                    str(path))
    return out


def build(registry_path="configs/frozen/I6_models.yaml", *, root=I6_ROOT,
          out_dir=None, git_sha_value=None, resamples=BOOTSTRAP_RESAMPLES):
    """`T_I6a_external.csv`. `git_sha_value` and `resamples` are arguments so
    two builds from the same inputs produce the same bytes."""
    registry = ext.load_registry(registry_path)
    head = {"work_package": registry["work_package"],
            "generated_by": "analysis/frozen_I6_external.py",
            "git_sha": git_sha_value if git_sha_value else git_sha()}
    rows, missing_models = [], []

    for entry in registry["models"]:
        model_id = entry["model_id"]
        found = load_model_maps(model_id, root=root)
        if not found:
            missing_models.append(model_id)
            continue
        for (stage, basis), (pm, ind, meta, source) in sorted(found.items()):
            side = pm.grid_side
            prof = P.ring_profile(pm)
            conc = P.concentration_with_reference(pm, sims=NULL_SIMS,
                                                  seed=NULL_SEED)
            peak = ext.ring_1_peak(prof)
            outcome = ext.prediction_outcome(
                entry["group"], has_ring_1_peak=peak,
                gini_excess=conc["gini_excess"])

            ring_means = per_image_ring_means(ind, side)
            ci = {k: bootstrap_ci(ring_means[:, k], resamples=resamples)
                  for k in (0, 1, ring_means.shape[1] - 1)}
            diff_ci = bootstrap_ci(ring_means[:, 1] - ring_means[:, 0],
                                   resamples=resamples)
            fa, fb = fnorms.split_half_frequency(ind, seed=BOOTSTRAP_SEED)
            half = P.spatial_rho(fa, fb)

            row = dict(head)
            row.update(
                model_id=model_id, timm_name=entry["timm_name"],
                group=entry["group"], mixing=str(entry["mixing"]),
                arch_note=str(entry.get("arch_note", "")).strip(),
                weight_sha256=meta.get("weight_sha256", MISSING),
                timm_version=meta.get("model_timm_version", MISSING),
                input_size=meta.get("model_input_size", MISSING),
                native_input_size=meta.get("model_native_input_size", MISSING),
                resolution_deviation=int(
                    bool(meta.get("model_resolution_deviation", False))),
                patch_size=meta.get("model_patch_size", MISSING),
                grid_side=side, n_patches=pm.n_positions,
                n_prefix=pm.n_prefix, depth=meta.get("model_depth", MISSING),
                norm_mean=json.dumps(meta.get("model_norm_mean", MISSING)),
                norm_std=json.dumps(meta.get("model_norm_std", MISSING)),
                crop_pct=meta.get("model_crop_pct", MISSING),
                split_name=pm.split_name, split_sha256=pm.split_sha,
                threshold_split_name=meta.get("threshold_split_name", MISSING),
                threshold_split_sha256=meta.get("threshold_split_sha256",
                                                MISSING),
                stage=stage, stage_resolved=ext.ext_stage_alias(stage),
                basis=basis,
                tau=_num(pm.threshold) if isinstance(pm.threshold, float)
                else MISSING,
                n_images=pm.n_images, count_per_image=_num(pm.mass),
                p_any=_num(float((ind.sum(axis=1) > 0).mean())),
                entropy_normalized=_num(conc["entropy_normalized"]),
                entropy_normalized_excess=_num(
                    conc["entropy_normalized_excess"]),
                gini=_num(conc["gini"]), gini_null=_num(conc["gini_null"]),
                gini_excess=_num(conc["gini_excess"]),
                top5_share=_num(conc["top5_share"]),
                top5_share_uniform=_num(5.0 / pm.n_positions),
                top5_share_excess=_num(conc["top5_share_excess"]),
                position_rel_se=_num(position_rel_se(pm.mass, pm.n_images,
                                                     pm.n_positions)),
                eta2_pos=_num(P.eta2_positional(pm)),
                ring0_freq_ci_lo=_num(ci[0][0]), ring0_freq_ci_hi=_num(ci[0][1]),
                ring1_freq_ci_lo=_num(ci[1][0]), ring1_freq_ci_hi=_num(ci[1][1]),
                centre_freq=_num(prof[-1]),
                centre_freq_ci_lo=_num(ci[ring_means.shape[1] - 1][0]),
                centre_freq_ci_hi=_num(ci[ring_means.shape[1] - 1][1]),
                ring1_share=_num(P.ring_share(pm, 1)),
                peak_ring=int(np.argmax(prof)), ring_1_peak=int(peak),
                ring1_minus_ring0=_num(prof[1] - prof[0]),
                ring1_minus_ring0_ci_lo=_num(diff_ci[0]),
                ring1_minus_ring0_ci_hi=_num(diff_ci[1]),
                prediction_group=entry["group"], prediction_outcome=outcome,
                split_half_rho=_num(half[0]) if half else MISSING,
                split_half_p=_num(half[1]) if half else MISSING,
                split_half_n_transforms=int(half[3]) if half else 0,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=resamples,
                source=source,
                note="outcome computed by "
                     "saga.frozen.external.prediction_outcome from the rule "
                     "registered in configs/frozen/I6_models.yaml; nothing "
                     "pooled across models or groups")
            for k in range(MAX_RINGS):
                row[f"ring{k}_freq"] = (_num(prof[k]) if k < len(prof)
                                        else MISSING)
            rows.append(row)

    out_dir = Path(out_dir) if out_dir else TABLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "T_I6a_external.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    return {"table": path, "rows": len(rows), "missing": missing_models,
            "registry": registry}


def main():
    p = argparse.ArgumentParser(description="TASK A / I6 external tables.")
    p.add_argument("--registry", default="configs/frozen/I6_models.yaml")
    p.add_argument("--root", default=str(I6_ROOT))
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    result = build(args.registry, root=Path(args.root), out_dir=args.out_dir)
    print(f"wrote {result['table']}  ({result['rows']} row(s))")
    if result["missing"]:
        print(f"no maps yet for {len(result['missing'])} model(s): "
              f"{result['missing']}")
        print("  Run scripts/jobs/frozen_I6.sbatch. A model with no maps is "
              "ABSENT from the table, never a row of zeros.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
