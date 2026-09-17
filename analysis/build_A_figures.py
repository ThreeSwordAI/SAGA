#!/usr/bin/env python3
"""
analysis/build_A_figures.py
===========================
Figure data for Track A — TASK A / I1 A7 and I6 A9, Phase C.

    python analysis/build_A_figures.py

Writes three packs under `figures_data/frozen/`:

    F3_spatial.npz            the spatial-structure figure: every map this
                              task measured, with its ring profile, at every
                              stage and basis, plus the pairwise seed
                              correlations and the block-input-vs-s11_out
                              correlations that F3 draws.
    F1_prevalence_draft.npz   DRAFT panels A/B: the mixup-vs-true-nomix
                              contrast on the reporting split, as cell mean
                              maps and ring profiles.
    F3_external.npz           the nine public checkpoints: maps, ring
                              profiles and the pre-registered outcome.

A PACK CARRIES ARRAYS AND PROVENANCE, NOT A PICTURE. Nothing here decides a
colour, an axis or a panel layout: `plotting/` does that, and keeping the
two apart is what lets a figure be redrawn without re-running anything.
Every array is keyed `"<what>__<run_id>|<condition>|<stage>|<basis>"` so a
plotting script selects by pattern and can never silently pick up a map from
the wrong stage.

EVERY MAP HERE IS FROM THE REPORTING SPLIT. The discovery maps exist and are
the source of D5's masks, but they are selection material: a figure drawn
from them would be showing the data the coordinates were chosen on. The one
exception is explicitly named `d5_source_*` and is the cell mean map D5
selected from, carried so the note's claim can be redrawn.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.frozen_I1_spatial import MapStore, _cell_of  # noqa: E402
from saga.frozen import external as ext  # noqa: E402
from saga.frozen import prevalence as P  # noqa: E402
from saga.frozen.runner import eligible_cohort  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

OUT = Path("figures_data/frozen")
REPORTING_SPLIT = "evaluation"
DISCOVERY_SPLIT = "val_diag_split"
I6_ROOT = Path("results/frozen/I6_external")


def _key(run_id, cond, stage, basis) -> str:
    return f"{run_id}|{cond}|{stage}|{basis}"


def build_F3(store, cohort, *, git_sha_value) -> dict:
    """Every reporting map, its ring profile, and the correlations F3 draws."""
    arrays, keys = {}, []
    for k in store.keys():
        run_id, cond, stage, basis = k
        pm = store.maps[k]
        name = _key(run_id, cond, stage, basis)
        keys.append(name)
        arrays[f"freq__{name}"] = pm.freq.astype(np.float64)
        arrays[f"ring__{name}"] = np.asarray(P.ring_profile(pm),
                                             dtype=np.float64)
        arrays[f"resid__{name}"] = P.ring_adjusted(pm).astype(np.float64)

    # seed stability: every baseline pair, per (cell, stage, basis)
    groups = {}
    for k in store.keys(variant="baseline", condition="native"):
        groups.setdefault((_cell_of(store.rows[k[0]]), k[2], k[3]), []).append(k)
    pair_names, pair_rho, pair_resid = [], [], []
    for (cell, stage, basis), members in sorted(groups.items()):
        for a, b in combinations(sorted(members), 2):
            r = P.spatial_rho(store.maps[a], store.maps[b])
            rr = P.spatial_rho(P.ring_adjusted(store.maps[a]),
                               P.ring_adjusted(store.maps[b]))
            pair_names.append(f"{cell}|{stage}|{basis}|{a[0]}|{b[0]}")
            pair_rho.append(np.nan if r is None else float(r[0]))
            pair_resid.append(np.nan if rr is None else float(rr[0]))
    arrays["seed_pair_names"] = np.array(pair_names)
    arrays["seed_pair_rho"] = np.asarray(pair_rho, dtype=np.float64)
    arrays["seed_pair_resid_rho"] = np.asarray(pair_resid, dtype=np.float64)

    # block input vs the reporting stage, same run
    bi_names, bi_rho, bi_resid = [], [], []
    for stage in ("in_b07", "in_b08"):
        for k in store.keys(stage=stage, condition="native"):
            own = (k[0], "native", "s11_out", k[3])
            if own not in store.maps:
                continue
            r = P.spatial_rho(store.maps[k], store.maps[own])
            rr = P.spatial_rho(P.ring_adjusted(store.maps[k]),
                               P.ring_adjusted(store.maps[own]))
            bi_names.append(f"{k[0]}|{stage}|{k[3]}")
            bi_rho.append(np.nan if r is None else float(r[0]))
            bi_resid.append(np.nan if rr is None else float(rr[0]))
    arrays["blockinput_names"] = np.array(bi_names)
    arrays["blockinput_rho_vs_s11_out"] = np.asarray(bi_rho, dtype=np.float64)
    arrays["blockinput_resid_rho_vs_s11_out"] = np.asarray(bi_resid,
                                                           dtype=np.float64)

    arrays["map_keys"] = np.array(keys)
    arrays["meta_json"] = np.array(json.dumps({
        "work_package": "I1_spatial", "split_name": REPORTING_SPLIT,
        "split_sha256": sorted({m.split_sha for m in store.maps.values()})[0],
        "git_sha": git_sha_value, "generated_by": "analysis/build_A_figures.py",
        "key_format": "<run_id>|<condition>|<stage>|<basis>",
        "n_maps": len(keys),
        "note": "every map is from the REPORTING split; the discovery maps "
                "are selection material and are not drawn",
    }, sort_keys=True))
    return arrays


def build_F1(store, *, git_sha_value) -> dict:
    """DRAFT panels A/B — the mixup vs true-nomix contrast.

    Cell mean maps, equal weight per checkpoint, at the reporting stage and
    the historical control. This is the contrast that says whether the
    border structure follows the training recipe.
    """
    arrays, rows = {}, []
    by_cell = {}
    for k in store.keys(variant="baseline", condition="native"):
        by_cell.setdefault((_cell_of(store.rows[k[0]]), k[2], k[3]), []).append(k)
    for (cell, stage, basis), members in sorted(by_cell.items()):
        maps = [store.maps[k] for k in sorted(members)]
        mean = P.cell_mean_map(maps)
        name = f"{cell}|{stage}|{basis}"
        arrays[f"cellmean__{name}"] = mean.astype(np.float64)
        arrays[f"cellmean_ring__{name}"] = np.asarray(
            P.ring_profile(mean), dtype=np.float64)
        rows.append({"cell": cell, "stage": stage, "basis": basis,
                     "n_members": len(maps),
                     "run_ids": [k[0] for k in sorted(members)],
                     "mass": float(mean.sum()),
                     "peak_ring": int(np.argmax(P.ring_profile(mean)))})
    arrays["cellmean_keys"] = np.array([f"{r['cell']}|{r['stage']}|{r['basis']}"
                                        for r in rows])
    arrays["meta_json"] = np.array(json.dumps({
        "work_package": "I1_spatial", "split_name": REPORTING_SPLIT,
        "status": "DRAFT — panels A/B", "git_sha": git_sha_value,
        "generated_by": "analysis/build_A_figures.py",
        "cells": rows,
        "note": "cell means weight every CHECKPOINT equally, not every "
                "image (LOCKED_ANALYSIS §8)",
    }, sort_keys=True))
    return arrays


def build_F3_external(*, git_sha_value, registry_path, root=I6_ROOT) -> dict:
    """The nine public checkpoints, with the registered outcome per map."""
    from analysis.frozen_I6_external import load_model_maps

    registry = ext.load_registry(registry_path)
    arrays, keys, rows = {}, [], []
    for entry in registry["models"]:
        model_id = entry["model_id"]
        for (stage, basis), (pm, ind, meta, src) in sorted(
                load_model_maps(model_id, root=Path(root)).items()):
            name = f"{model_id}|{stage}|{basis}"
            keys.append(name)
            prof = P.ring_profile(pm)
            conc = P.concentration_with_reference(pm)
            peak = ext.ring_1_peak(prof)
            arrays[f"freq__{name}"] = pm.freq.astype(np.float64)
            arrays[f"ring__{name}"] = np.asarray(prof, dtype=np.float64)
            rows.append({
                "model_id": model_id, "stage": stage, "basis": basis,
                "group": entry["group"], "mixing": str(entry["mixing"]),
                "grid_side": pm.grid_side, "mass": float(pm.mass),
                "ring_1_peak": bool(peak),
                "gini_excess": float(conc["gini_excess"]),
                "prediction_outcome": ext.prediction_outcome(
                    entry["group"], has_ring_1_peak=peak,
                    gini_excess=conc["gini_excess"]),
            })
    arrays["map_keys"] = np.array(keys)
    arrays["meta_json"] = np.array(json.dumps({
        "work_package": "I6_external", "split_name": REPORTING_SPLIT,
        "git_sha": git_sha_value, "generated_by": "analysis/build_A_figures.py",
        "key_format": "<model_id>|<stage>|<basis>",
        "prediction": registry["prediction"],
        "outcome_rule": registry["outcome_rule"],
        "models": rows,
        "note": "grids differ: 14x14 for patch-16 models, 16x16 for patch-14 "
                "at 224. Nothing is pooled across models or groups.",
    }, sort_keys=True))
    return arrays


def build_d5_source(*, git_sha_value, manifest) -> dict:
    """The cell mean maps D5 selected from — DISCOVERY, and labelled so.

    Carried so the D5 note's claim can be redrawn, and named `d5_source_` so
    no plotting script can mistake it for a reporting map.
    """
    from analysis.build_D5_masks import D5_BASIS, D5_STAGES, cell_members

    cohort = eligible_cohort(manifest)
    disc = MapStore(DISCOVERY_SPLIT, cohort)
    arrays = {}
    for stage in D5_STAGES:
        keys = cell_members(disc, stage, D5_BASIS)
        if not keys:
            continue
        maps = [disc.maps[k] for k in keys]
        mean = P.cell_mean_map(maps)
        arrays[f"d5_source_freq__{stage}"] = mean.astype(np.float64)
        arrays[f"d5_source_ring__{stage}"] = np.asarray(P.ring_profile(mean),
                                                        dtype=np.float64)
        arrays[f"d5_source_runs__{stage}"] = np.array([k[0] for k in keys])
    return arrays


def write_pack(path: Path, arrays: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
    tmp.replace(path)
    return path


def build(*, manifest="results/frozen/I0_manifest/manifest.json",
          out=OUT, git_sha_value=None,
          registry_path="configs/frozen/I6_models.yaml") -> dict:
    sha = git_sha_value if git_sha_value else git_sha()
    cohort = eligible_cohort(manifest)
    store = MapStore(REPORTING_SPLIT, cohort)
    out = Path(out)

    f3 = build_F3(store, cohort, git_sha_value=sha)
    f3.update(build_d5_source(git_sha_value=sha, manifest=manifest))
    written = {"F3_spatial": write_pack(out / "F3_spatial.npz", f3),
               "F1_prevalence_draft": write_pack(
                   out / "F1_prevalence_draft.npz",
                   build_F1(store, git_sha_value=sha))}
    ext_arrays = build_F3_external(git_sha_value=sha,
                                   registry_path=registry_path)
    if ext_arrays.get("map_keys") is not None and len(ext_arrays["map_keys"]):
        written["F3_external"] = write_pack(out / "F3_external.npz",
                                            ext_arrays)
    return written


def main():
    p = argparse.ArgumentParser(description="Track A figure data.")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    for name, path in build(manifest=args.manifest, out=args.out).items():
        kb = path.stat().st_size / 1024
        with np.load(path, allow_pickle=False) as z:
            n = len(z.files)
        print(f"wrote {path}  ({n} arrays, {kb:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
