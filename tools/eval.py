#!/usr/bin/env python3
"""
tools/eval.py
=============
EXACT full-val ImageNet evaluation (fixes audit bug B1: the trainers'
validate() never all-reduced across DDP ranks, so recorded top-1 was
rank-0's ~12.5k-image shard, not the 50k val set).

Single process:
    python tools/eval.py --ckpt CKPT --arch vit_base --variant saga \
        --data /path/to/imagenet --out results/legacy/eval/name.json

Distributed (per-GPU sharding, exact same counts as single-process):
    torchrun --nproc_per_node=4 tools/eval.py --ckpt ... (same args)

Guarantees:
- Model built exactly as trained (tools/model_factory.py, strict=True load).
- Val transform reuses the trainers' code path (timm create_loader defaults).
- Manual index sharding `indices[rank::world_size]` — NO DistributedSampler
  (its padding duplicates samples). Counts are all-reduced as tensors and
  n == len(dataset) == 50000 is asserted before writing.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.run_registry import file_sha256, git_sha
from tools.model_factory import build_model, load_checkpoint

IMAGENET_VAL_SIZE = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Val transform — reuse the trainers' code path, do not re-implement.
# ─────────────────────────────────────────────────────────────────────────────

def build_val_transform(input_size: int = 224):
    """
    The exact eval transform both trainers get from
    timm.data.create_loader(val_ds, input_size=(3, S, S), is_training=False)
    — same defaults, same resize/crop/interpolation:
    Resize(int(S/0.875), bilinear) + CenterCrop(S) + ToTensor + Normalize.
    (use_prefetcher=False folds the prefetcher's GPU normalization into the
    transform; arithmetic is equivalent.)
    """
    from timm.data import create_transform
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    return create_transform(
        (3, input_size, input_size),
        is_training=False,
        interpolation="bilinear",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        crop_pct=None,   # -> timm DEFAULT_CROP_PCT, as in create_loader
        crop_mode=None,
        use_prefetcher=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator logic — pure functions (unit-tested in tests/test_eval_reduction)
# ─────────────────────────────────────────────────────────────────────────────

def shard_counts(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """[correct_top1, correct_top5, loss_sum, n] as a float64 tensor.
    Exact integer counts in float64; summing shard counts == pooled counts."""
    logits = logits.float()
    # per-sample losses summed in float64: identical per-row values regardless
    # of shard boundaries, so shard sums match the pooled computation exactly
    loss_sum = F.cross_entropy(logits, targets, reduction="none").double().sum()
    _, pred = logits.topk(5, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    c1 = correct[:, :1].sum().double()
    c5 = correct.sum().double()
    n = torch.tensor(float(targets.shape[0]), dtype=torch.float64,
                     device=logits.device)
    return torch.stack([c1, c5, loss_sum, n])


def counts_to_metrics(counts: torch.Tensor) -> dict:
    """Final metrics from (possibly all-reduced) counts."""
    c1, c5, loss_sum, n = counts.tolist()
    return {
        "top1": 100.0 * c1 / n,
        "top5": 100.0 * c5 / n,
        "loss": loss_sum / n,
        "n_images": int(n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exact full-val ImageNet evaluation (all ranks, all images).")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--arch", required=True, choices=["vit_small", "vit_base"])
    parser.add_argument("--variant", required=True,
                        choices=["baseline", "registers", "saga"])
    parser.add_argument("--data", required=True, metavar="ROOT",
                        help="ImageNet root containing val/ (ImageFolder layout)")
    parser.add_argument("--out", required=True,
                        help="output JSON, e.g. results/legacy/eval/<name>.json")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="per-process batch size")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", default="off", choices=["bf16", "off"],
                        help="bf16 autocast for speed; off = exact fp32 (default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl" if args.device.startswith("cuda") else "gloo")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}"
                              if args.device.startswith("cuda") else args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)

    torch.manual_seed(args.seed)

    # Model — exactly as trained, strict load
    model = build_model(args.arch, args.variant)
    ckpt_meta = load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    # Data — trainers' dataset + val transform code path
    from timm.data import create_dataset
    dataset = create_dataset("imagefolder", root=args.data, split="val",
                             is_training=False)
    dataset.transform = build_val_transform(224)

    # Manual sharding: every index exactly once across ranks, no padding
    shard = list(range(len(dataset)))[rank::world_size]
    loader = DataLoader(
        Subset(dataset, shard),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if rank == 0:
        print(f"eval: {args.arch}/{args.variant}  ckpt={args.ckpt}")
        print(f"      {len(dataset)} images, world_size={world_size}, "
              f"amp={args.amp}, ckpt_epoch={ckpt_meta.get('epoch', '?')}")

    counts = torch.zeros(4, dtype=torch.float64, device=device)
    autocast_on = (args.amp == "bf16" and device.type == "cuda")
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_on):
                logits = model(images)
            counts += shard_counts(logits, targets)
            if rank == 0 and step % 20 == 0:
                print(f"  [{step + 1}/{len(loader)}] "
                      f"top1 so far: {100.0 * counts[0] / counts[3]:.3f}%",
                      flush=True)

    if distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    n = int(counts[3].item())
    assert n == len(dataset) == IMAGENET_VAL_SIZE, (
        f"evaluated {n} images but dataset has {len(dataset)} "
        f"(expected {IMAGENET_VAL_SIZE}) -- sharding or dataset is broken; "
        f"refusing to write results")

    if rank == 0:
        metrics = counts_to_metrics(counts.cpu())
        result = {
            **metrics,
            "ckpt": Path(args.ckpt).resolve().as_posix(),
            "ckpt_sha256": file_sha256(args.ckpt),
            "git_sha": git_sha(),
            "arch": args.arch,
            "variant": args.variant,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "world_size": world_size,
            "amp": args.amp,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\ntop1={metrics['top1']:.3f}%  top5={metrics['top5']:.3f}%  "
              f"loss={metrics['loss']:.4f}  n={metrics['n_images']}")
        print(f"wrote {out}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
