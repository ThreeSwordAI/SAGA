#!/usr/bin/env python3
"""
tools/frozen_I7_attn.py
=======================
TASK C / I7 — the incoming-attention driver.

    python tools/frozen_I7_attn.py \
        --run-id e2r_vits_mixup_baseline_s1 \
        --conditions configs/frozen/I7_attention.yaml \
        --split results/frozen/splits/sub1k.json \
        --data $STAGE_DIR

It loads a manifest row, verifies the checkpoint's sha256, builds the model
through `tools/model_factory.py` at the run's RECORDED `gate_mode`, loads the
split and checks its sha, walks it once, and writes `incoming_mass.npz` under
`results/frozen/I7_attention/<split>/<run_id>/`.

NO ATTENTION MAPS ARE WRITTEN. The maps exist inside one forward and are
reduced to per-image, per-key summaries before the next batch (§7). What
lands on disk is ~5 MB per run and is committed.

The threshold this run records for the SECONDARY exceedance basis is I2's
`tau_cal` at `s11_out` for this checkpoint's cell, read from the committed
per-split file. It is the literal MISSING for a cell that has none, and the
run proceeds — MAD is the PRIMARY basis precisely because it needs no
calibration and therefore applies to every checkpoint.

Deterministic inference, fp32, the split's own image order, `num_workers=0`,
no shuffling. No training, no optimizer, no probe fitting.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import attention as att  # noqa: E402
from saga.frozen.runner import (build_from_row, image_id_for,  # noqa: E402
                                load_conditions, load_manifest_row,
                                load_split, verify_checkpoint)
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"


def tau_cal_for_row(conditions, row, split_name):
    """I2's calibrated tau at `s11_out` for this row's cell, or MISSING.

    The tau file is PER SPLIT. `sub1k` is drawn from `evaluation`, so the
    conditions file names `thresholds_cal_split: evaluation` and that is the
    file read — applying a calibration-split tau to sub1k would be an
    undeclared threshold with the same key and a different meaning.
    """
    from saga.frozen import diag as fdiag

    template = conditions.get("thresholds_cal")
    if not template:
        return MISSING, MISSING
    which = conditions.get("thresholds_cal_split") or split_name
    path = fdiag.thresholds_cal_path(template, which)
    if not Path(path).exists():
        return MISSING, str(path)
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    cell = fdiag.cell_key(row["arch"], row["recipe_actual"])
    return fdiag.tau_cal_for(doc, cell, "s11_out"), str(path)


def main():
    p = argparse.ArgumentParser(
        description="TASK C / I7 — incoming attention for one run.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ckpt-kind", default="last")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--conditions", default="configs/frozen/I7_attention.yaml")
    p.add_argument("--split", required=True)
    p.add_argument("--data", required=True, metavar="ROOT",
                   help="ImageNet root containing val/")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out-root", default="results/frozen")
    p.add_argument("--batch-size", type=int, default=16,
                   help="the explicit [B, H, T, T] map is materialised for "
                        "two blocks at a time; 16 is ~50 MB for ViT-S at 224")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--skip-if-done", action="store_true")
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
    if conditions.get("capture") != "attention":
        raise SystemExit(
            f"{args.conditions} declares capture="
            f"{conditions.get('capture')!r}; this driver runs the attention "
            f"capture only")
    split_doc, items, split_sha = load_split(args.split)
    split_name = split_doc.get("name", Path(args.split).stem)
    image_ids = [image_id_for(rel) for rel, _ in items]

    out_dir = (Path(args.out_root) / conditions["work_package"] / split_name
               / args.run_id)
    tau, tau_path = tau_cal_for_row(conditions, row, split_name)

    print(f"frozen_I7_attn: {args.run_id}/{args.ckpt_kind} "
          f"{row['arch']}/{row['variant']} gate_mode={row.get('gate_mode')}")
    print(f"                ckpt_sha256={sha[:16]}…  split={split_name} "
          f"sha={split_sha[:16]}… n={len(items)}")
    print(f"                blocks={conditions['blocks']} "
          f"alignment={att.BLOCK_ALIGNMENT}")
    print(f"                tau_cal[s11_out]={tau} (from {tau_path})")
    print(f"                -> {out_dir}")

    if args.skip_if_done and att.is_done(out_dir, sha):
        print("                already complete for this checkpoint — "
              "nothing rewritten")
        return 0

    from tools.eval import build_val_transform
    from tools.frozen_eval import FrozenSplitDataset
    dataset = FrozenSplitDataset(args.data, items,
                                 transform=build_val_transform(224))
    model = build_from_row(row, ckpt_path=args.ckpt, device=device)

    summary = att.run_attention_package(
        row=row, conditions=conditions, dataset=dataset, out_dir=out_dir,
        image_ids=image_ids, device=device, batch_size=args.batch_size,
        split_name=split_name, split_sha=split_sha, git_sha=git_sha(),
        tau_cal=(tau if isinstance(tau, float) else None), model=model,
        max_images=args.max_images)

    print(f"\nwrote {summary['path']} ({summary['bytes'] / 1e6:.2f} MB, "
          f"{summary['n_images']} images)")
    print(f"state restored: {summary['state_restored']}  "
          f"fused_attn flags unchanged: True")
    (out_dir / "run_meta.json").write_text(
        json.dumps({"run_id": args.run_id, "ckpt_sha256": sha,
                    "conditions_file": args.conditions,
                    "split": args.split, "split_sha256": split_sha,
                    "git_sha": git_sha(), "seed": args.seed,
                    "device": str(device),
                    "tau_cal_source": tau_path, **summary},
                   indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
