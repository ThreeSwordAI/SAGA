#!/usr/bin/env python3
"""
tools/frozen_I2_thresholds.py
=============================
TASK I2 D2 — the per-stage calibrated thresholds, written as NEW KEYS in a
NEW file.

    python tools/frozen_I2_thresholds.py \
        --conditions configs/frozen/I2_terminal.yaml \
        --split results/frozen/splits/calibration.json \
        --data $PROBE_IN_DIR --if-missing

The historical tau in `results/diagsplit/fixed_thresholds_canon.json` was
calibrated at the LAST BLOCK on the discovery split. It is meaningful at
stage `hist` and at no other stage, so a stage comparison needs a threshold
of its own. This tool applies THE SAME RECIPE per stage, on the split being
evaluated, from the cell's DESIGNATED BASELINE checkpoint — the same
checkpoint the canon file names in its own `source_run` block, so `tau_cal`
at `hist` is the canon recipe on a different split and the two can be read
side by side.

  tau_cal[cell][stage] = lower median over images of
                         ( median(v_i) + 5 * MAD(v_i) )

`v_i` is the per-token L2 norm vector of image i's PATCH rows at `stage`,
prefix rows removed by the model's own count. Both inner medians and the
outer median are LOWER medians: the arithmetic is not re-implemented here,
`tools/compute_fixed_thr.per_image_mad_thresholds` — the function that
produced every canon value — is imported through `saga/frozen/diag.py`.

`fixed_thresholds_canon.json` is opened READ-ONLY. `write_thresholds_cal`
refuses to write to it by resolved path, and a test pins its sha256.

Append-safe: `--if-missing` is a no-op once every cell is calibrated for this
split, a resubmission adds the cells that are missing and rewrites nothing,
and a cell whose recalibration disagrees with the file is a hard error rather
than a silent overwrite.

No training, no optimizer, no probe fitting.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import diag as fdiag  # noqa: E402
from saga.frozen.runner import (RunnerError, build_from_row,  # noqa: E402
                                eligible_cohort, load_conditions, load_split,
                                verify_checkpoint)
from saga.frozen.stages import capture_stages, resolve_stage  # noqa: E402
from saga.metrics import token_norms  # noqa: E402
from saga.run_registry import file_sha256, git_sha  # noqa: E402

MISSING = "MISSING"

DEFINITION = (
    "per (arch, recipe_actual) cell and per STAGE, on the split named below: "
    "tau_cal = lower median over images of (median(v_i) + k*MAD(v_i)) on the "
    "PATCH-row L2 norms at that stage of the cell's DESIGNATED BASELINE last "
    "checkpoint — the same checkpoint results/diagsplit/"
    "fixed_thresholds_canon.json names in its source_run block. Medians are "
    "LOWER medians via tools/compute_fixed_thr.per_image_mad_thresholds, the "
    "function that produced every canon value. At stage hist this is the "
    "canon recipe on a different split; it is a NEW KEY in a NEW file and "
    "the canon file is never rewritten.")


def designated_baselines(cohort, canon: dict) -> dict:
    """{cell: manifest row} for each cell's canon-designated baseline.

    The designation is READ from the canon file rather than re-decided here:
    "the seeded s1 baseline when one exists, else the legacy designated
    baseline" is a historical choice, and a second implementation of it could
    pick a different checkpoint and produce a tau_cal[hist] that is not
    comparable with tau_canon at all.
    """
    by_run = {r["run_id"]: r for r in cohort}
    cells = sorted({fdiag.cell_key(r["arch"], r["recipe_actual"])
                    for r in cohort})
    source_run = canon.get("source_run", {})
    source_sha = canon.get("source_ckpt_sha256", {})
    out = {}
    for cell in cells:
        run_id = source_run.get(cell)
        if not run_id:
            raise RunnerError(
                f"cell {cell!r} has no designated baseline in the canon "
                f"thresholds file (source_run). I2 calibrates from the canon "
                f"file's own designation and refuses to invent one.")
        if run_id not in by_run:
            raise RunnerError(
                f"cell {cell!r} designates baseline {run_id!r}, which is not "
                f"an eligible cohort row — the manifest and the canon file "
                f"disagree about the cohort")
        row = by_run[run_id]
        recorded = source_sha.get(cell)
        if recorded and row.get("ckpt_sha256") not in (MISSING, "", None) \
                and recorded != row["ckpt_sha256"]:
            raise RunnerError(
                f"cell {cell!r}: the canon file calibrated on checkpoint "
                f"{recorded} but the manifest records {row['ckpt_sha256']} "
                f"for {run_id!r} — refusing to calibrate against a different "
                f"checkpoint than the historical tau used")
        out[cell] = row
    return out


@torch.no_grad()
def per_image_thresholds(model, dataset, stages, *, device, batch_size, k,
                        max_images=None):
    """{stage: float64 [n_images]} of per-image median + k*MAD thresholds."""
    from torch.utils.data import DataLoader

    wanted = tuple(stages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    chunks = {s: [] for s in wanted}
    seen = 0
    model.eval()
    for images, _targets in loader:
        if max_images is not None and seen >= max_images:
            break
        images = images.to(device)
        if max_images is not None and seen + images.shape[0] > max_images:
            images = images[:max_images - seen]
        with capture_stages(model, wanted) as store:
            model(images)
            for s in wanted:
                norms = token_norms(store[s]).cpu().numpy()
                chunks[s].append(
                    fdiag.canon_per_image_mad_thresholds(norms, k))
        seen += images.shape[0]
    if not seen:
        raise RunnerError("the split yielded no images")
    return {s: np.concatenate(chunks[s]) for s in wanted}, seen


def main():
    p = argparse.ArgumentParser(
        description="Calibrate TASK I2's per-stage thresholds (new keys).")
    p.add_argument("--conditions", required=True,
                   help="the work package's conditions YAML; the stages, the "
                        "output path and the canon path are read from it")
    p.add_argument("--split", required=True)
    p.add_argument("--data", required=True, metavar="ROOT")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-images", type=int, default=None,
                   help="smoke only: take the first N images of the split")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--if-missing", action="store_true",
                   help="exit 0 without touching anything when every cell is "
                        "already calibrated for this split")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (torch.cuda.is_available()
                        or not args.device.startswith("cuda")) else "cpu")

    conditions = load_conditions(args.conditions)
    stages = list(conditions.get("calibration_stages") or [])
    if not stages:
        raise SystemExit(
            f"{args.conditions}: no `calibration_stages` — the stages to "
            f"calibrate are declared in the conditions file, never here")
    for s in stages:
        resolve_stage(s)
    canon_path = Path(conditions["thresholds_canon"])

    split_doc, items, split_sha = load_split(args.split)
    split_name = split_doc.get("name", Path(args.split).stem)
    # one file per split: the YAML declares a `{split_name}` template
    out_path = fdiag.thresholds_cal_path(conditions["thresholds_cal"],
                                         split_name)

    canon = fdiag.load_canon_thresholds(canon_path)
    k = float(canon.get("k", fdiag.MAD_K))
    cohort = eligible_cohort(args.manifest)
    targets = designated_baselines(cohort, canon)

    existing = {}
    if out_path.exists():
        existing = fdiag.load_thresholds_cal(out_path, split_sha256=split_sha)
        have = set(existing.get("tau_cal", {}))
        todo = {c: r for c, r in targets.items()
                if c not in have
                or set(existing["tau_cal"][c]) < set(stages)}
        if not todo:
            print(f"{out_path}: every cell already calibrated for "
                  f"{split_name} ({split_sha[:16]}…) — nothing to do")
            return 0
        if args.if_missing:
            print(f"{out_path}: adding {sorted(todo)}")
        targets = todo

    from tools.eval import build_val_transform
    from tools.frozen_eval import FrozenSplitDataset

    tau_cal, sources = {}, {}
    for cell, row in sorted(targets.items()):
        sha = verify_checkpoint(row)
        model = build_from_row(row, device=device)
        dataset = FrozenSplitDataset(args.data, items,
                                     transform=build_val_transform(224))
        thresholds, n_images = per_image_thresholds(
            model, dataset, stages, device=device,
            batch_size=args.batch_size, k=k, max_images=args.max_images)
        tau_cal[cell] = {s: fdiag.calibrate_tau(thresholds[s]) for s in stages}
        sources[cell] = {
            "run_id": row["run_id"], "variant": row["variant"],
            "arch": row["arch"], "recipe_actual": row["recipe_actual"],
            "ckpt_kind": row["ckpt_kind"], "ckpt_sha256": sha,
            "n_images": int(n_images),
            "canon_source_run": canon.get("source_run", {}).get(cell, MISSING),
            "tau_canon": fdiag.canon_tau_for(cell, canon),
        }
        print(f"{cell}: " + "  ".join(
            f"{s}={tau_cal[cell][s]:.6f}" for s in stages)
            + f"   (baseline {row['run_id']}, n={n_images}, "
              f"tau_canon={sources[cell]['tau_canon']})")
        del model

    fresh = {
        "work_package": conditions["work_package"],
        "definition": DEFINITION, "k": k,
        "split_name": split_name, "split_sha256": split_sha,
        "stages": list(stages),
        "precision": conditions.get("precision", "fp32"),
        "canon_file": str(canon_path),
        # raw bytes as this filesystem holds them, AND the LF-normalised
        # digest, which is the portable one: the repo is checked out on
        # Windows locally and Linux on the cluster, so only the normalised
        # value can be compared across the two.
        "canon_file_sha256": file_sha256(canon_path),
        "canon_file_sha256_lf": fdiag.canon_sha256(canon_path),
        "canon_stage": fdiag.HIST_STAGE,
        "baseline_run_ids": sorted(s["run_id"] for s in sources.values()),
        "git_sha": git_sha(), "seed": args.seed,
        "generated_by": "tools/frozen_I2_thresholds.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tau_cal": tau_cal, "sources": sources,
    }
    merged = fdiag.merge_thresholds_cal(existing, fresh)
    merged["baseline_run_ids"] = sorted(
        s["run_id"] for s in merged["sources"].values() if s)
    fdiag.write_thresholds_cal(out_path, merged)
    print(f"\nwrote {out_path}: {len(merged['tau_cal'])} cell(s) x "
          f"{len(stages)} stage(s) on {split_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
