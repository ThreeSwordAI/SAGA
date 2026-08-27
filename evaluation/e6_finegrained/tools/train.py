#!/usr/bin/env python3
"""
evaluation/e6_finegrained/tools/train.py
==========================================
E6 fine-grained recognition — fine-tune E2 ViT-B checkpoints.

Fine-tunes baseline and SAGA ViT-B nomix checkpoints on:
  - CUB-200-2011  (200 bird species, train=5994, test=5794)
  - FGVC-Aircraft (100 aircraft variants, trainval=6667, test=3333)

Strategy:
  - Replace classification head (1000 → num_classes)
  - Two param groups: backbone LR 1e-5, head LR 1e-3
  - 100 epochs, cosine LR schedule
  - Batch 64 on single GPU (TinyGPU RTX 2080 Ti, 11GB)
  - No MixUp/CutMix — consistent with nomix backbone training

Usage:
    # CUB baseline
    python3 evaluation/e6_finegrained/tools/train.py \
        --dataset cub \
        --backbone baseline \
        --paths   evaluation/e6_finegrained/configs/paths.yaml

    # Aircraft SAGA
    python3 evaluation/e6_finegrained/tools/train.py \
        --dataset aircraft \
        --backbone saga \
        --paths   evaluation/e6_finegrained/configs/paths.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from saga import build_saga_vit


# ── Model ──────────────────────────────────────────────────────────────────────

def load_backbone(backbone_key: str, paths: dict, num_classes: int,
                  device, arch: str = 'vit_base_patch16_224') -> nn.Module:
    """
    Load E2 checkpoint, replace head with num_classes output.
    """
    is_saga = (backbone_key == 'saga')

    # ViT-S uses its own checkpoint key if present in paths.yaml
    if 'small' in arch.lower():
        ckpt_path = paths['backbones'].get(
            f'{backbone_key}_small', paths['backbones'][backbone_key])
    else:
        ckpt_path = paths['backbones'][backbone_key]

    print(f"  Loading {backbone_key} ({arch}) from {ckpt_path}...")
    model = build_saga_vit(
        arch,
        gate       = is_saga,
        img_size   = 224,
        num_classes= 1000,   # original head size
        pretrained = False,
    )

    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = {k.replace('module.', ''): v
             for k, v in ckpt.get('model', ckpt).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  E2 top-1: {ckpt.get('top1', '?')}%")
    if missing:
        non_head = [k for k in missing if 'head' not in k]
        if non_head:
            print(f"  Missing non-head keys: {non_head[:3]}")

    # Replace classification head
    embed_dim = model.embed_dim
    model.head = nn.Linear(embed_dim, num_classes)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)

    return model.to(device)


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast('cuda', enabled=True):
            logits = model(images)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs              = images.size(0)
        total_loss     += loss.item() * bs
        total_correct  += (logits.argmax(1) == labels).sum().item()
        total_n        += bs

    return total_loss / total_n, 100.0 * total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct, total_n = 0, 0
    total_correct5 = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        # Top-1
        total_correct  += (logits.argmax(1) == labels).sum().item()
        # Top-5
        _, top5 = logits.topk(min(5, logits.size(1)), dim=1)
        total_correct5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
        total_n += images.size(0)

    return (100.0 * total_correct  / total_n,
            100.0 * total_correct5 / total_n)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  required=True,
                        choices=['cub', 'aircraft'])
    parser.add_argument('--backbone', required=True,
                        choices=['baseline', 'saga'])
    parser.add_argument('--paths',    required=True)
    parser.add_argument('--epochs',   type=int, default=100)
    parser.add_argument('--batch',    type=int, default=64)
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr',     type=float, default=1e-3)
    parser.add_argument('--img_size',    type=int,   default=224)
    parser.add_argument('--arch',        type=str,
                        default='vit_base_patch16_224',
                        help='timm arch name — overrides default per dataset')
    parser.add_argument('--run_name',    type=str,   default=None,
                        help='Override output name (default: dataset_backbone)')
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name or f'{args.dataset}_{args.backbone}'

    ckpt_dir = Path(paths['outputs']['checkpoints']) / run_name
    res_dir  = Path(paths['outputs']['results'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True,  exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  E6 Fine-grained: {run_name}")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")

    # ── Stage and load dataset ─────────────────────────────────────────────────
    stage_base = paths['data']['stage_base']

    if args.dataset == 'cub':
        from evaluation.e6_finegrained.data.cub_dataset import (
            CUBDataset, stage_cub)
        cub_root = stage_cub(paths['data']['cub_tar'],
                             str(Path(stage_base) / 'cub_e6'))
        train_ds = CUBDataset(cub_root, split='train',
                              img_size=args.img_size, augment=True)
        val_ds   = CUBDataset(cub_root, split='test',
                              img_size=args.img_size, augment=False)
        num_classes = CUBDataset.NUM_CLASSES

    else:  # aircraft
        from evaluation.e6_finegrained.data.aircraft_dataset import (
            AircraftDataset, stage_aircraft)
        ac_root  = stage_aircraft(paths['data']['aircraft_tar'],
                                  str(Path(stage_base) / 'aircraft_e6'))
        train_ds = AircraftDataset(ac_root, split='trainval',
                                   img_size=args.img_size, augment=True)
        val_ds   = AircraftDataset(ac_root, split='test',
                                   img_size=args.img_size, augment=False)
        num_classes = AircraftDataset.NUM_CLASSES

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_backbone(args.backbone, paths, num_classes, device,
                          arch=args.arch)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_gate   = sum(p.numel() for n, p in model.named_parameters()
                   if 'phi' in n)
    print(f"  Params: {n_params:.1f}M  |  Gate: {n_gate}  |  Classes: {num_classes}")

    # ── Optimiser — two param groups ──────────────────────────────────────────
    head_params     = list(model.head.parameters())
    head_ids        = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters()
                       if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params,     'lr': args.head_lr},
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda')

    # ── Training loop ─────────────────────────────────────────────────────────
    best_top1 = 0.0
    history   = []

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_top1 = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch)
        scheduler.step()

        # Evaluate every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_top1, val_top5 = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            print(f"  [{epoch+1:3d}/{args.epochs}]  "
                  f"loss={train_loss:.4f}  "
                  f"train={train_top1:.2f}%  "
                  f"val={val_top1:.2f}%  "
                  f"top5={val_top5:.2f}%  "
                  f"t={elapsed:.0f}s", flush=True)

            history.append({
                'epoch':      epoch + 1,
                'train_loss': round(train_loss, 4),
                'train_top1': round(train_top1, 2),
                'val_top1':   round(val_top1,   2),
                'val_top5':   round(val_top5,   2),
            })

            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save({
                    'epoch':     epoch,
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'top1':      best_top1,
                }, ckpt_dir / 'best.pth')
                print(f"  *** New best: {best_top1:.2f}%", flush=True)
        else:
            # Light logging on non-eval epochs
            if (epoch + 1) % 10 == 0:
                print(f"  [{epoch+1:3d}/{args.epochs}]  "
                      f"loss={train_loss:.4f}  "
                      f"train={train_top1:.2f}%", flush=True)

    # ── Final results ─────────────────────────────────────────────────────────
    final_top1, final_top5 = evaluate(model, val_loader, device)

    results = {
        'run':        run_name,
        'dataset':    args.dataset,
        'backbone':   args.backbone,
        'epochs':     args.epochs,
        'best_top1':  round(best_top1,   2),
        'final_top1': round(final_top1,  2),
        'final_top5': round(final_top5,  2),
        'num_classes':num_classes,
        'history':    history,
    }

    out_file = res_dir / f'{run_name}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Done: {run_name}")
    print(f"  Best top-1:  {best_top1:.2f}%")
    print(f"  Final top-1: {final_top1:.2f}%")
    print(f"  Final top-5: {final_top5:.2f}%")
    print(f"  Saved: {out_file}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()