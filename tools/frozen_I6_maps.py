#!/usr/bin/env python3
"""
tools/frozen_I6_maps.py
=======================
TASK A / I6 A8 — thresholds on CALIBRATION, maps and indicators on
EVALUATION, for one external public checkpoint.

    python tools/frozen_I6_maps.py \
        --model-id deit_small_patch16_224 \
        --registry configs/frozen/I6_models.yaml \
        --conditions configs/frozen/I6_external.yaml \
        --calibration-split results/frozen/splits/calibration.json \
        --evaluation-split results/frozen/splits/evaluation.json \
        --data $PROBE_IN_DIR --device cuda --seed 0 --skip-if-done

TWO SPLITS, ON PURPOSE. The threshold is the canon recipe — the lower median
over images of each image's median + 5*MAD — computed on the CALIBRATION
split and APPLIED to the evaluation split. I1 and I2 calibrate and report on
the SAME split, because their tau comes from a cell's designated BASELINE and
is applied to that cell's other members; a public checkpoint has no baseline
to borrow from, so it would otherwise be calibrated and reported on the same
images. The thresholds file records both shas so the difference is visible
rather than inferred.

EVERYTHING ELSE IS SHARED CODE. `capture_stages` for the tokens (the same
hooks that read our own cohort), `saga/frozen/norms.py` for the norms,
indicators and the packbits writer, `tools/compute_fixed_thr` — through
`saga/frozen/diag.py` — for the threshold arithmetic. Nothing here
re-implements a statistic, and the transform is timm's OWN for each tag, not
our cohort's.

Writes `results/frozen/I6_external/<model_id>/maps_<stage>.npz` (committed)
and the shared `results/frozen/I6_external/thresholds_cal.json`.

No gate, no SAGAViT, no fine-tuning, no training, no optimizer.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import diag as fdiag  # noqa: E402
from saga.frozen import external as ext  # noqa: E402
from saga.frozen import norms as fnorms  # noqa: E402
from saga.frozen import records as rec  # noqa: E402
from saga.frozen.runner import (image_id_for, load_conditions,  # noqa: E402
                                load_split)
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"
DONE_KIND = "maps"


@torch.no_grad()
def calibrate(model, dataset, stages, *, device, batch_size, k,
              max_images=None):
    """`({stage: tau}, n_images)` by the canon recipe on this split."""
    from torch.utils.data import DataLoader

    from saga.frozen.stages import capture_stages
    from saga.metrics import token_norms

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    chunks = {s: [] for s in stages}
    seen = 0
    model.eval()
    with capture_stages(model, tuple(stages)) as store:
        for images, _targets in loader:
            if max_images is not None and seen >= max_images:
                break
            images = images.to(device)
            if max_images is not None and seen + images.shape[0] > max_images:
                images = images[:max_images - seen]
            store.clear()
            model(images)
            for s in stages:
                norms = token_norms(store[s]).cpu().numpy()
                chunks[s].append(fdiag.canon_per_image_mad_thresholds(norms, k))
            seen += images.shape[0]
    if not seen:
        raise SystemExit("the calibration split yielded no images")
    return ({s: fdiag.calibrate_tau(np.concatenate(chunks[s]))
             for s in stages}, seen)


def main():
    p = argparse.ArgumentParser(
        description="TASK A / I6: external-checkpoint thresholds and maps.")
    p.add_argument("--model-id", required=True)
    p.add_argument("--registry", default="configs/frozen/I6_models.yaml")
    p.add_argument("--conditions", default="configs/frozen/I6_external.yaml")
    p.add_argument("--calibration-split",
                   default="results/frozen/splits/calibration.json")
    p.add_argument("--evaluation-split",
                   default="results/frozen/splits/evaluation.json")
    p.add_argument("--data", required=True, metavar="ROOT")
    p.add_argument("--out-root", default="results/frozen/I6_external")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-images", type=int, default=None,
                   help="smoke only: the first N images of each split")
    p.add_argument("--skip-if-done", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (torch.cuda.is_available()
                        or not args.device.startswith("cuda")) else "cpu")

    registry = ext.load_registry(args.registry)
    conditions = load_conditions(args.conditions)
    entry = ext.model_entry(registry, args.model_id)
    cache_root = ext.set_cache_env(registry)         # before timm downloads
    stages = list(ext.EXT_STAGES)

    out_dir = Path(args.out_root) / args.model_id
    print(f"frozen_I6_maps: {args.model_id}  ({entry['timm_name']})")
    print(f"                group={entry['group']}  mixing={entry['mixing']}  "
          f"timm={ext.timm_version()}")
    print(f"                HF_HOME={cache_root}")
    print(f"                -> {out_dir}")

    # The sha the registry recorded on the login node, checked BEFORE load.
    sha = ext.verify_weight_sha(entry)
    print(f"                weight_sha256={sha[:16]}… verified")

    if args.skip_if_done and rec.is_done(out_dir, DONE_KIND, sha):
        print("                already complete for these weights — "
              "nothing rewritten")
        return 0

    model, info = ext.build_external(args.model_id, registry, device=device)
    print(f"                grid={info['grid_side']}x{info['grid_side']} "
          f"({info['n_patches']} patches)  n_prefix={info['n_prefix']}  "
          f"depth={info['depth']}")
    if info["resolution_deviation"]:
        print(f"                NOTE: native input {info['native_input_size']}, "
              f"run at {info['input_size']} — position embedding interpolated")

    from tools.frozen_eval import FrozenSplitDataset, output_digests
    # The transform is built at the size the MODEL was built at, not the size
    # the weights were published at — see external_transform.
    transform, transform_info = ext.external_transform(
        model, input_size=int(registry["input_size"]))
    print(f"                transform: {transform_info['transform_input_size']} "
          f"(published {transform_info['published_input_size']}), "
          f"crop_pct={transform_info['crop_pct']}")

    # ── thresholds, on CALIBRATION ───────────────────────────────────────────
    cal_doc, cal_items, cal_sha = load_split(args.calibration_split)
    cal_name = cal_doc.get("name", Path(args.calibration_split).stem)
    eval_doc, eval_items, eval_sha = load_split(args.evaluation_split)
    eval_name = eval_doc.get("name", Path(args.evaluation_split).stem)
    if args.max_images is not None:
        cal_items, eval_items = (cal_items[:args.max_images],
                                 eval_items[:args.max_images])

    thr_path = Path(registry["thresholds_cal"])
    k = float(registry.get("mad_k", fdiag.MAD_K))
    existing = {}
    if thr_path.exists():
        existing = ext.load_thresholds(thr_path, calibration_sha=cal_sha)
    have = (existing.get("tau_cal", {}) or {}).get(args.model_id)
    if have and set(have) >= set(stages):
        tau_cal = {s: float(have[s]) for s in stages}
        print(f"                tau_cal already calibrated on {cal_name}: "
              + "  ".join(f"{s}={tau_cal[s]:.6f}" for s in stages))
    else:
        tau_cal, n_cal = calibrate(
            model, FrozenSplitDataset(args.data, cal_items,
                                      transform=transform),
            stages, device=device, batch_size=args.batch_size, k=k,
            max_images=args.max_images)
        print(f"                tau_cal on {cal_name} (n={n_cal}): "
              + "  ".join(f"{s}={tau_cal[s]:.6f}" for s in stages))
        fresh = ext.thresholds_doc(
            registry, tau_cal={args.model_id: tau_cal},
            sources={args.model_id: dict(info, n_images=int(n_cal),
                                         weight_sha256=sha)},
            calibration_split=cal_name, calibration_sha=cal_sha,
            reporting_split=eval_name, reporting_sha=eval_sha,
            git_sha_value=git_sha())
        # Read-merge-write under one lock. Nine array tasks share this one
        # file and each adds its own model; without the lock the later write
        # drops the earlier one's key, and with a shared temp name one task
        # fails outright (the I1 discovery array hit exactly that on
        # 2026-09-17). Append-safe: a model already calibrated is never
        # recomputed away.
        fdiag.update_thresholds_cal(thr_path, fresh)
        print(f"                wrote {thr_path}")

    # ── maps + indicators, on EVALUATION ─────────────────────────────────────
    image_ids = [image_id_for(rel) for rel, _ in eval_items]
    acc = fnorms.extract_stage_norms(
        model, FrozenSplitDataset(args.data, eval_items, transform=transform),
        stages, device=device, batch_size=args.batch_size,
        image_ids=image_ids, tau_cal=tau_cal, k=k,
        max_images=args.max_images, condition_id="native")

    meta = {
        "work_package": conditions["work_package"],
        "run_id": args.model_id, "model_id": args.model_id,
        "ckpt_sha256": sha, "weight_sha256": sha,
        "split_name": eval_name, "split_sha256": eval_sha,
        "threshold_split_name": cal_name, "threshold_split_sha256": cal_sha,
        "tau_cal": tau_cal, "mad_k": k,
        "thresholds_cal_file": str(thr_path),
        "git_sha": git_sha(), "seed": args.seed,
        "generated_by": "tools/frozen_I6_maps.py",
        # count_fixed_canon has no definition for an external model
        "tau_canon": MISSING,
        **{f"model_{key}": value for key, value in info.items()},
        **{f"transform_{key}": value for key, value in transform_info.items()},
    }

    written = []
    for stage in stages:
        written.append(fnorms.write_stage_maps_npz(out_dir, [acc[stage]],
                                                   meta))
    for path in written:
        print(f"wrote {path}")

    marker = dict(meta, stages=stages, n_images=acc[stages[0]].n_images,
                  files=[q.name for q in written])
    (out_dir / "run_meta.json").write_text(
        json.dumps(dict(marker, outputs=output_digests(out_dir)),
                   indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n")
    rec.mark_done(out_dir, DONE_KIND, marker)     # LAST, always
    print(f"\n{len(written)} file(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
