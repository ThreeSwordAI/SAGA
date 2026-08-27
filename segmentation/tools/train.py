#!/usr/bin/env python3
"""
segmentation/tools/train.py
=============================
E4 training script — ADE20K semantic segmentation.
Backbone: E2 ViT-B checkpoints (baseline / registers / SAGA).
Head: Simple Feature Pyramid + multi-scale segmentation head.

Usage:
    torchrun --nproc_per_node=4 segmentation/tools/train.py \
        --config   segmentation/configs/variants.yaml \
        --id       2 \
        --paths    segmentation/configs/paths.yaml \
        --data_root $STAGE_DIR

    # Resume after timeout:
    torchrun ... --resume /path/to/e4/checkpoints/ViT-B_SAGA_seg/last.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from segmentation.data.ade20k_dataset import ADE20KDataset
from segmentation.data.transforms     import get_train_transforms, get_val_transforms
from segmentation.models.segmentor    import build_segmentor


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def deep_merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(variants_path, variant_id, paths_path):
    raw       = load_yaml(variants_path)
    base_file = Path(variants_path).parent / raw.get('_base_', 'base.yaml')
    cfg       = load_yaml(base_file)
    variant   = next(v for v in raw['variants'] if v['id'] == variant_id)
    top       = {k: v for k, v in raw.items() if k not in ('_base_', 'variants')}
    cfg = deep_merge(cfg, top)
    cfg = deep_merge(cfg, {k: v for k, v in variant.items() if k != 'id'})
    cfg['variant_id']   = variant_id
    cfg['variant_name'] = variant['name']
    cfg['paths']        = load_yaml(paths_path)
    return cfg


# ── mIoU evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_miou(model, loader, device, num_classes=150, ignore_index=255):
    """Compute mIoU over the validation set."""
    model.eval()

    # Accumulate intersection and union per class
    intersection = torch.zeros(num_classes, device=device)
    union        = torch.zeros(num_classes, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)   # [B, H, W]

        logits = model(images)          # [B, C, H, W]
        preds  = logits.argmax(dim=1)   # [B, H, W]

        for cls in range(num_classes):
            pred_mask = (preds  == cls)
            gt_mask   = (masks  == cls)
            inter     = (pred_mask & gt_mask).sum().float()
            uni       = (pred_mask | gt_mask).sum().float()
            intersection[cls] += inter
            union[cls]        += uni

    # Aggregate across DDP ranks
    dist.all_reduce(intersection, op=dist.ReduceOp.SUM)
    dist.all_reduce(union,        op=dist.ReduceOp.SUM)

    iou_per_class = intersection / (union + 1e-10)
    # Only average over classes that appear in validation
    valid = union > 0
    miou  = iou_per_class[valid].mean().item() * 100.0

    return {
        'mIoU':      round(miou, 2),
        'iou_per_class': iou_per_class.cpu().tolist(),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler,
                    epoch, cfg, device, rank):
    model.train()
    t        = cfg['train']
    log_freq = cfg.get('logging', {}).get('log_freq', 50)

    total_loss = 0.0
    n_batches  = 0

    for step, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        with torch.autocast('cuda', enabled=t['amp']):
            loss = model(images, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        clip = t.get('grad_clip', 1.0)
        if clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        if rank == 0 and step % log_freq == 0:
            print(f"  [{epoch}][{step}/{len(loader)}]  "
                  f"loss={loss.item():.4f}", flush=True)

    return total_loss / max(n_batches, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--id',         type=int, required=True)
    parser.add_argument('--paths',      required=True)
    parser.add_argument('--data_root',  required=True,
                        help='Staged ADE20K root (ADEChallengeData2016/)')
    parser.add_argument('--resume',     default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    args = parser.parse_args()

    dist.init_process_group('nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    cfg  = load_config(args.config, args.id, args.paths)
    if args.max_epochs:
        cfg['train']['epochs'] = args.max_epochs

    name      = cfg['variant_name']
    paths     = cfg['paths']
    t         = cfg['train']
    ckpt_dir  = Path(paths['outputs']['checkpoints']) / name
    res_dir   = Path(paths['outputs']['results'])

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True,  exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    crop_size  = t['input_size']
    min_sc, max_sc = t['random_scale']

    train_ds = ADE20KDataset(
        args.data_root, split='training',
        transforms=get_train_transforms(crop_size, min_sc, max_sc))
    val_ds = ADE20KDataset(
        args.data_root, split='validation',
        transforms=get_val_transforms(crop_size))

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)

    per_gpu = max(1, t['batch_size'] // world_size)

    train_loader = DataLoader(
        train_ds, batch_size=per_gpu, sampler=train_sampler,
        num_workers=t['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, sampler=val_sampler,
        num_workers=t['num_workers'], pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  E4 Segmentation — {name}")
        print(f"  GPUs: {world_size}  |  Batch: {t['batch_size']}")
        print(f"{'='*60}\n", flush=True)

    segmentor = build_segmentor(cfg, paths).to(device)
    segmentor = DDP(segmentor, device_ids=[local_rank],
                    find_unused_parameters=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    backbone_params = list(segmentor.module.backbone.parameters())
    head_params     = (list(segmentor.module.neck.parameters()) +
                       list(segmentor.module.head.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': float(t['backbone_lr'])},
        {'params': head_params,     'lr': float(t['head_lr'])},
    ], weight_decay=float(t['weight_decay']))

    total_epochs  = t['epochs']
    freeze_epochs = t.get('freeze_backbone_epochs', 2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda', enabled=t['amp'])

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    history     = []

    if args.resume and Path(args.resume).exists():
        if rank == 0:
            print(f"Resuming from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location='cpu')
        segmentor.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_miou   = ckpt.get('best_miou', 0.0)
        hist_file   = res_dir / f'{name}.json'
        if hist_file.exists():
            history = json.load(open(hist_file)).get('history', [])
        for _ in range(start_epoch):
            scheduler.step()
        if rank == 0:
            print(f"Resumed at epoch {start_epoch}  "
                  f"best_mIoU={best_miou:.2f}", flush=True)

    save_freq = cfg.get('logging', {}).get('save_freq', 10)
    eval_freq = cfg.get('eval',    {}).get('eval_freq', 10)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)

        if epoch < freeze_epochs:
            segmentor.module.backbone.freeze()
        else:
            segmentor.module.backbone.unfreeze()

        t0 = time.time()
        avg_loss = train_one_epoch(
            segmentor, train_loader, optimizer, scaler,
            epoch, cfg, device, rank)
        scheduler.step()
        dist.barrier()

        # Evaluate
        if (epoch + 1) % eval_freq == 0 or epoch == total_epochs - 1:
            metrics = evaluate_miou(
                segmentor.module, val_loader, device,
                num_classes=cfg['model']['num_classes'])

            if rank == 0:
                miou = metrics['mIoU']
                print(f"[{epoch+1}/{total_epochs}]  "
                      f"loss={avg_loss:.4f}  "
                      f"mIoU={miou:.2f}%  "
                      f"time={time.time()-t0:.0f}s", flush=True)

                history.append({
                    'epoch':      epoch,
                    'train_loss': round(avg_loss, 4),
                    'mIoU':       miou,
                })

                if miou > best_miou:
                    best_miou = miou
                    torch.save({
                        'epoch':     epoch,
                        'model':     segmentor.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler':    scaler.state_dict(),
                        'best_miou': best_miou,
                        'mIoU':      miou,
                    }, ckpt_dir / 'best.pth')
                    print(f"  *** New best mIoU: {best_miou:.2f}%", flush=True)

        # Save checkpoint
        if rank == 0:
            if (epoch + 1) % save_freq == 0 or epoch == total_epochs - 1:
                torch.save({
                    'epoch':     epoch,
                    'model':     segmentor.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler':    scaler.state_dict(),
                    'best_miou': best_miou,
                }, ckpt_dir / 'last.pth')

    # ── Final results ──────────────────────────────────────────────────────────
    # ALL ranks must call evaluate_miou — it uses dist.all_reduce internally.
    # Calling only from rank 0 causes NCCL timeout on the other ranks.
    if rank == 0:
        print("\nRunning final evaluation...", flush=True)

    final = evaluate_miou(
        segmentor.module, val_loader, device,
        num_classes=cfg['model']['num_classes'])

    # Only rank 0 saves and prints results
    if rank == 0:
        results = {
            'variant':   name,
            'arch':      cfg['model']['arch'],
            'gate':      cfg['model'].get('gate', False),
            'registers': cfg['model'].get('registers', 0),
            'epochs':    total_epochs,
            'best_mIoU': best_miou,
            'final_mIoU':final['mIoU'],
            'history':   history,
        }
        with open(res_dir / f'{name}.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Done — {name}")
        print(f"  best_mIoU:  {best_miou:.2f}%")
        print(f"  final_mIoU: {final['mIoU']:.2f}%")
        print(f"{'='*60}", flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()