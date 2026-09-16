#!/usr/bin/env python3
"""
tools/frozen_eval.py
====================
TASK I0 D4 — the one driver every frozen work package (I1-I7) runs.

    python tools/frozen_eval.py \
        --run-id e2r_vits_mixup_saga_s1 \
        --conditions configs/frozen/smoke.yaml \
        --split results/frozen/splits/evaluation.json \
        --data $STAGE_DIR

It loads a manifest row, verifies the checkpoint's sha256, builds the model
through `tools/model_factory.py` at the run's RECORDED `gate_mode`, loads the
split and checks its sha, runs the conditions declared in the YAML, and
writes `records`/`diag` under `results/frozen/<WP>/<run_id>/`.

NO condition, layer, epsilon, mask or permutation is selectable from the
command line. They live in the per-work-package YAML, which is committed
before the job runs — that is what makes "the analysis configuration was
fixed before the outcomes were inspected" a property of this repository
rather than a promise (plan §6.2). The only command-line knobs are WHICH run,
WHICH split, WHERE the data is, and how many images (for a smoke).

Deterministic inference, fp32, the split's own image order, `num_workers=0`,
no shuffling. No training, no optimizer, no probe fitting.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import records as rec  # noqa: E402
from saga.frozen.records import records_path  # noqa: E402
from saga.frozen.runner import (load_conditions, load_manifest_row,  # noqa: E402
                                load_split, run_work_package,
                                run_work_package_stages, verify_checkpoint)
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"


def is_multistage(conditions: dict) -> bool:
    """True when the conditions file declares per-condition `stages`.

    Which loop runs is a property of the COMMITTED conditions file, not a
    command-line flag: a work package that measures three stages in one
    forward says so in its YAML, and `load_conditions` has already refused a
    document that declares `stages` on only some of its conditions.
    """
    return any("stages" in c for c in conditions["conditions"])


class FrozenSplitDataset(torch.utils.data.Dataset):
    """A frozen split (tools/build_frozen_splits.py) as a dataset.

    Deliberately parallel to tools/build_diag_split.DiagSplitDataset and
    using the SAME val transform as tools/eval.py, so a frozen record and a
    historical eval number were produced from the same pixels.
    """

    def __init__(self, root, items, transform=None):
        self.root = Path(root)
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        rel, label = self.items[idx]
        img = Image.open(self.root / rel).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def main():
    p = argparse.ArgumentParser(
        description="Run a frozen work package's declared conditions.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ckpt-kind", default="last")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--conditions", required=True,
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
    p.add_argument("--fixed-thr", type=float, default=None,
                   help="absolute norm threshold for the fixed sink count "
                        "(results/diagsplit/fixed_thresholds_canon.json)")
    p.add_argument("--effrank", action="store_true",
                   help="also compute effective rank (SVD; sub1k only). "
                        "Ignored by a multi-stage conditions file, which "
                        "records the TASK I2 §4 key set unconditionally.")
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
    split_doc, items, split_sha = load_split(args.split)
    split_name = split_doc.get("name", Path(args.split).stem)

    from tools.eval import build_val_transform
    dataset = FrozenSplitDataset(args.data, items,
                                 transform=build_val_transform(224))

    # THE SPLIT IS PART OF THE PATH. Without it a second split's sweep would
    # write into the first one's directory: `append_rows` would add the new
    # split's images beside the old ones in one file, and `--skip-if-done`
    # would see the previous marker for the SAME checkpoint and exit 0 having
    # done nothing. Neither failure is loud. Measured on the I2 calibration
    # results before B2 was submitted.
    out_dir = (Path(args.out_root) / conditions["work_package"] / split_name
               / args.run_id)
    multistage = is_multistage(conditions)
    print(f"frozen_eval: {args.run_id}/{args.ckpt_kind} "
          f"{row['arch']}/{row['variant']} gate_mode={row.get('gate_mode')}")
    print(f"             ckpt_sha256={sha[:16]}…  split={split_name} "
          f"sha={split_sha[:16]}… n={len(items)}")
    print(f"             stage={conditions['stage']}  "
          f"{len(conditions['conditions'])} condition(s) from "
          f"{args.conditions}")
    print(f"             -> {out_dir}")

    if args.skip_if_done and rec.is_done(out_dir, "records", sha):
        print(f"             already complete for this checkpoint — "
              f"nothing rewritten")
        return 0

    if multistage:
        from saga.frozen import diag as fdiag
        tau_path = fdiag.thresholds_cal_path(conditions["thresholds_cal"],
                                             split_name)
        tau_cal_doc = fdiag.load_thresholds_cal(tau_path,
                                                split_sha256=split_sha)
        canon_doc = fdiag.load_canon_thresholds(conditions["thresholds_canon"])
        print(f"             multi-stage: tau_cal from {tau_path} "
              f"(split {tau_cal_doc['split_sha256'][:16]}…)")
        summary = run_work_package_stages(
            row=row, conditions=conditions, dataset=dataset, out_dir=out_dir,
            device=device, batch_size=args.batch_size, split_name=split_name,
            split_sha=split_sha, git_sha=git_sha(),
            git_dirty=row.get("git_dirty", MISSING),
            patch_file=row.get("patch_file", MISSING), ckpt_path=args.ckpt,
            max_images=args.max_images, tau_cal_doc=tau_cal_doc,
            canon_doc=canon_doc)
    else:
        summary = run_work_package(
            row=row, conditions=conditions, dataset=dataset, out_dir=out_dir,
            device=device, batch_size=args.batch_size,
            fixed_thr=args.fixed_thr, with_effrank=args.effrank,
            split_name=split_name, split_sha=split_sha, git_sha=git_sha(),
            git_dirty=row.get("git_dirty", MISSING),
            patch_file=row.get("patch_file", MISSING), ckpt_path=args.ckpt,
            max_images=args.max_images)

    print(f"\nwrote {records_path(out_dir, 'records')} "
          f"(+{summary['n_records']} rows)")
    if summary["n_diag"]:
        print(f"wrote {records_path(out_dir, 'diag')} "
              f"(+{summary['n_diag']} rows)")
    print(f"state restored: {summary['state_restored']}")
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
    """`{filename: {sha256, bytes}}` for everything the run wrote.

    A records file that is too large to commit still has to be identifiable:
    the I2 evaluation sweep leaves `records.parquet` / `diag.parquet` on the
    cluster beside the norm dumps, and `run_meta.json` — which IS committed —
    is then the only record that says which bytes produced the tables. Every
    number this project reports must be traceable to a file; this is how a
    file that lives outside git stays traceable.
    """
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
