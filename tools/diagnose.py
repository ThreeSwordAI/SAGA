#!/usr/bin/env python3
"""
tools/diagnose.py
=================
Run the canonical diagnostics (saga/metrics.py) for one checkpoint on the
frozen 10k diagnostic split.

    python tools/diagnose.py --ckpt CKPT --arch vit_base --variant saga \
        --data /path/to/imagenet \
        --split-file results/diagsplit/val_diag_split.json \
        --out results/legacy/diag/<name>.json

Writes:
- <name>.json — the diagnostics schema (sink counts, oversmoothing,
  effective rank, per-block cls profiles; see saga/metrics.py).
- <name>_norms.npz (git-ignored) — per-image norm arrays fp16 for the
  histogram figures: last_block_patch_norms [n, N], cls_norms [n, L],
  median_patch_norms [n, L].
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.metrics import compute_diagnostics
from saga.run_registry import file_sha256, git_sha
from tools.build_diag_split import DiagSplitDataset
from tools.eval import build_val_transform
from tools.model_factory import build_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Canonical diagnostics on the frozen 10k split.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--arch", required=True, choices=["vit_small", "vit_base"])
    parser.add_argument("--variant", required=True,
                        choices=["baseline", "registers", "saga"])
    parser.add_argument("--data", required=True, metavar="ROOT",
                        help="ImageNet root containing val/ (ImageFolder layout)")
    parser.add_argument("--split-file",
                        default="results/diagsplit/val_diag_split.json")
    parser.add_argument("--out", required=True,
                        help="output JSON; <name>_norms.npz lands next to it")
    parser.add_argument("--attn", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="capture attention for cls_attn_share (default on)")
    parser.add_argument("--fixed-thr", type=float, default=None,
                        help="absolute norm threshold for sink_counts_fixed")
    parser.add_argument("--n-effrank", type=int, default=10000,
                        help="max images for effective rank (SVD is the slow part)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = build_model(args.arch, args.variant)
    load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    dataset = DiagSplitDataset(args.data, args.split_file,
                               transform=build_val_transform(224))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    print(f"diagnose: {args.arch}/{args.variant}  ckpt={args.ckpt}")
    print(f"          {len(dataset)} images, attn={args.attn}, "
          f"fixed_thr={args.fixed_thr}")

    diag = compute_diagnostics(
        model, loader, device,
        with_attn=args.attn,
        fixed_thr=args.fixed_thr,
        n_effrank=args.n_effrank,
        collect_norms=True,
    )
    arrays = diag.pop("_norms_arrays")

    result = {
        "ckpt": Path(args.ckpt).resolve().as_posix(),
        "ckpt_sha256": file_sha256(args.ckpt),
        "git_sha": git_sha(),
        "arch": args.arch,
        "variant": args.variant,
        **diag,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    npz_path = out.with_name(out.stem + "_norms.npz")
    np.savez(
        npz_path,
        last_block_patch_norms=arrays["last_block_patch_norms"],
        cls_norms=arrays["cls_norms"],
        median_patch_norms=arrays["median_patch_norms"],
    )

    print(f"\nsink_mad_k5={result['sink_mad_k5']:.4f}  "
          f"oversmooth_pairwise={result['oversmooth_pairwise']:.4f}  "
          f"eff_rank={result['eff_rank']:.2f}")
    print(f"wrote {out}")
    print(f"wrote {npz_path} (git-ignored)")


if __name__ == "__main__":
    main()
