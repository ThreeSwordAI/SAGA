#!/usr/bin/env python3
"""
tools/frozen_I5_corr.py
=======================
TASK C / I5a — the correspondence-readout driver.

    python tools/frozen_I5_corr.py \
        --run-id e2r_vits_mixup_saga_s1 \
        --conditions configs/frozen/I5_readout.yaml \
        --split results/frozen/splits/sub2k.json \
        --data $STAGE_DIR

It loads a manifest row, verifies the checkpoint's sha256, builds the model
through `tools/model_factory.py` at the run's RECORDED `gate_mode`, loads the
split and checks its sha, walks the split once per declared transform, and
writes `corr_records` under `results/frozen/I5_readout/<split>/<run_id>/`.

WHY A DRIVER OF ITS OWN, AND NOT `tools/frozen_eval.py`
-------------------------------------------------------
`frozen_eval.py` runs ONE image per forward and records a functional
response (nll, top-1) against a label. A correspondence readout runs a PAIR
of images per forward step, has no label-based response at all, and writes a
different row shape. Every other work package in this repository that needed
a different loop got its own WP-prefixed driver beside the shared one —
`tools/frozen_I1_norms.py`, `tools/frozen_I2_thresholds.py`,
`tools/frozen_I6_maps.py` — and this follows that structure rather than
adding a second mode to the shared driver.

The CONDITIONS CONTRACT is unaffected: the conditions, transforms, stages and
descriptors are read by `saga.frozen.runner.load_conditions`, the project's
one parser, whose TASK C hook refuses anything not in the YAML and not in D9.
The only command-line knobs here are WHICH run, WHICH split, WHERE the data
is, and how many images.

Deterministic inference, fp32, the split's own image order, `num_workers=0`,
no shuffling. No training, no optimizer, no probe fitting.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import features as feat  # noqa: E402
from saga.frozen import records as rec  # noqa: E402
from saga.frozen.records import records_path  # noqa: E402
from saga.frozen.runner import (build_from_row, image_id_for,  # noqa: E402
                                load_conditions, load_manifest_row,
                                load_split, verify_checkpoint)
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"


def main():
    p = argparse.ArgumentParser(
        description="TASK C / I5a — the correspondence readout for one run.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ckpt-kind", default="last")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--conditions", default="configs/frozen/I5_readout.yaml",
                   help="the work package's conditions YAML (configs/frozen/)")
    p.add_argument("--split", required=True,
                   help="a frozen split JSON (results/frozen/splits/)")
    p.add_argument("--data", required=True, metavar="ROOT",
                   help="ImageNet root containing val/")
    p.add_argument("--ckpt", default=None,
                   help="override the manifest's ckpt_path (same sha still "
                        "required)")
    p.add_argument("--out-root", default="results/frozen")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-images", type=int, default=None,
                   help="smoke only: take the first N images of the split")
    p.add_argument("--skip-if-done", action="store_true",
                   help="exit 0 when this run's completion marker already "
                        "names this exact checkpoint (resubmit-safe)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (torch.cuda.is_available()
                        or not args.device.startswith("cuda")) else "cpu")

    row = load_manifest_row(args.manifest, args.run_id, args.ckpt_kind)
    sha = verify_checkpoint(row, args.ckpt)
    conditions = load_conditions(args.conditions)
    if conditions.get("readout") != "correspondence":
        raise SystemExit(
            f"{args.conditions} declares readout="
            f"{conditions.get('readout')!r}; this driver runs the "
            f"correspondence readout only")
    split_doc, items, split_sha = load_split(args.split)
    split_name = split_doc.get("name", Path(args.split).stem)
    image_ids = [image_id_for(rel) for rel, _ in items]

    # THE SPLIT IS PART OF THE PATH, for the reason tools/frozen_eval.py
    # states: without it a second split's sweep appends its images into the
    # first one's file and `--skip-if-done` then sees a marker for the same
    # checkpoint and exits 0 having done nothing. Neither failure is loud.
    out_dir = (Path(args.out_root) / conditions["work_package"] / split_name
               / args.run_id)

    print(f"frozen_I5_corr: {args.run_id}/{args.ckpt_kind} "
          f"{row['arch']}/{row['variant']} gate_mode={row.get('gate_mode')}")
    print(f"                ckpt_sha256={sha[:16]}…  split={split_name} "
          f"sha={split_sha[:16]}… n={len(items)}")
    print(f"                transforms={conditions['transforms']} "
          f"descriptors={conditions['descriptors']}")
    print(f"                -> {out_dir}")

    if args.skip_if_done and rec.is_done(out_dir, "corr_records", sha):
        print("                already complete for this checkpoint — "
              "nothing rewritten")
        return 0

    model = build_from_row(row, ckpt_path=args.ckpt, device=device)

    def dataset_for(transform_id):
        return feat.CorrespondencePairDataset(args.data, items, transform_id)

    summary = feat.run_correspondence_package(
        row=row, conditions=conditions, dataset_for=dataset_for,
        out_dir=out_dir, image_ids=image_ids, device=device,
        batch_size=args.batch_size, split_name=split_name,
        split_sha=split_sha, git_sha=git_sha(),
        git_dirty=row.get("git_dirty", MISSING),
        patch_file=row.get("patch_file", MISSING), ckpt_path=args.ckpt,
        max_images=args.max_images, model=model)

    print(f"\nwrote {records_path(out_dir, 'corr_records')} "
          f"(+{summary['n_records']} rows, {summary['n_skipped']} already "
          f"present)")
    print(f"state restored: {summary['state_restored']}")
    print(f"T0 vs plain forward: {summary['t0_matches_plain_forward']}")
    (out_dir / "run_meta.json").write_text(
        json.dumps({"run_id": args.run_id, "ckpt_sha256": sha,
                    "conditions_file": args.conditions,
                    "split": args.split, "split_sha256": split_sha,
                    "git_sha": git_sha(), "seed": args.seed,
                    "device": str(device),
                    "outputs": output_digests(out_dir), **summary},
                   indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n")
    return 0


def output_digests(out_dir) -> dict:
    """`{filename: {sha256, bytes}}` for everything the run wrote, so a file
    is identifiable even when it is too large to commit (I2's convention)."""
    from saga.run_registry import file_sha256

    out = {}
    for path in sorted(Path(out_dir).iterdir()):
        if not path.is_file() or path.name == "run_meta.json":
            continue
        out[path.name] = {"sha256": file_sha256(path),
                          "bytes": path.stat().st_size}
    return out


if __name__ == "__main__":
    sys.exit(main())
