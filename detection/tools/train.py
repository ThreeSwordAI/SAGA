#!/usr/bin/env python3
"""
detection/tools/train.py
=========================
E3 training script — COCO 2017 object detection.
Backbone: E2 ViT-B checkpoints (baseline / registers / SAGA).
Head: Simple Feature Pyramid + Faster R-CNN (torchvision).

Usage:
    torchrun --nproc_per_node=4 detection/tools/train.py \
        --config   detection/configs/variants.yaml \
        --id       2 \
        --paths    detection/configs/paths.yaml \
        --data_root $STAGE_DIR

    # Resume:
    torchrun --nproc_per_node=4 detection/tools/train.py \
        --config   detection/configs/variants.yaml \
        --id       2 \
        --paths    detection/configs/paths.yaml \
        --data_root $STAGE_DIR \
        --resume   /path/to/checkpoints/ViT-B_SAGA_det/last.pth
"""

import argparse
import json
import math
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

from detection.data.coco_dataset import COCODetectionDataset, collate_fn
from detection.data.transforms   import get_train_transforms, get_val_transforms
from detection.models.detector   import build_detector


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(variants_path: str, variant_id: int, paths_path: str) -> dict:
    raw      = load_yaml(variants_path)
    base_file= Path(variants_path).parent / raw.get('_base_', 'base.yaml')
    cfg      = load_yaml(base_file)
    variant  = next(v for v in raw['variants'] if v['id'] == variant_id)
    # Merge top-level overrides from variants file
    top_overrides = {k: v for k, v in raw.items()
                     if k not in ('_base_', 'variants')}
    cfg = deep_merge(cfg, top_overrides)
    cfg = deep_merge(cfg, {k: v for k, v in variant.items() if k != 'id'})
    cfg['variant_id']   = variant_id
    cfg['variant_name'] = variant['name']
    cfg['paths']        = load_yaml(paths_path)
    return cfg


# ── COCO evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_coco(model, loader, device, rank, world_size=1):
    """
    Run COCO evaluation using pycocotools.

    With DDP, each rank runs inference on its shard of the val set.
    All predictions are gathered to rank 0 before COCO scoring.
    This ensures all 4,952 val images are covered, not just rank 0's shard.
    """
    model.eval()

    local_results = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, tgt in zip(preds, targets):
            img_id = tgt['image_id'].item()
            boxes  = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                local_results.append({
                    'image_id':    img_id,
                    'category_id': int(label),
                    'bbox':        [x1, y1, x2 - x1, y2 - y1],
                    'score':       float(score),
                })

    # ── Gather predictions from all ranks to rank 0 ────────────────────────────
    if world_size > 1:
        import pickle
        import torch.distributed as dist_mod

        # Serialise local results to bytes
        local_bytes = pickle.dumps(local_results)
        local_size  = torch.tensor(
            len(local_bytes), dtype=torch.long, device=device)

        # All ranks broadcast their result sizes
        all_sizes = [torch.zeros(1, dtype=torch.long, device=device)
                     for _ in range(world_size)]
        dist_mod.all_gather(all_sizes, local_size.unsqueeze(0))
        all_sizes = [s.item() for s in all_sizes]

        max_size = max(all_sizes)

        # Pad local bytes to max_size and all-gather
        padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
        padded[:len(local_bytes)] = torch.frombuffer(
            local_bytes, dtype=torch.uint8)

        all_padded = [torch.zeros(max_size, dtype=torch.uint8, device=device)
                      for _ in range(world_size)]
        dist_mod.all_gather(all_padded, padded)

        if rank == 0:
            all_results = []
            for size, buf in zip(all_sizes, all_padded):
                chunk = pickle.loads(buf[:size].cpu().numpy().tobytes())
                all_results.extend(chunk)
        else:
            all_results = []
    else:
        all_results = local_results

    # ── Only rank 0 runs COCO eval ─────────────────────────────────────────────
    if rank != 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'AP_S': 0.0, 'AP_M': 0.0, 'AP_L': 0.0}

    if not all_results:
        print("  WARNING: no predictions produced by any rank.")
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
                'AP_S': 0.0, 'AP_M': 0.0, 'AP_L': 0.0}

    from pycocotools.cocoeval import COCOeval
    coco_gt   = loader.dataset.get_coco_api()
    coco_dt   = coco_gt.loadRes(all_results)
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    return {
        'AP':    round(float(stats[0]) * 100, 2),
        'AP50':  round(float(stats[1]) * 100, 2),
        'AP75':  round(float(stats[2]) * 100, 2),
        'AP_S':  round(float(stats[3]) * 100, 2),
        'AP_M':  round(float(stats[4]) * 100, 2),
        'AP_L':  round(float(stats[5]) * 100, 2),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler,
                    epoch, cfg, device, rank):
    model.train()
    log_freq = cfg.get('logging', {}).get('log_freq', 50)
    total_loss = 0.0
    n_batches  = 0

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        with torch.autocast('cuda', enabled=cfg['train']['amp']):
            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        clip = cfg['train'].get('grad_clip', 1.0)
        if clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        if rank == 0 and step % log_freq == 0:
            loss_strs = '  '.join(
                f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            print(f"  [{epoch}][{step}/{len(loader)}] "
                  f"total={loss.item():.4f}  {loss_strs}", flush=True)

    return total_loss / max(n_batches, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--id',         type=int, required=True)
    parser.add_argument('--paths',      required=True)
    parser.add_argument('--data_root',  required=True)
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
    ckpt_dir  = Path(paths['outputs']['checkpoints']) / name
    res_dir   = Path(paths['outputs']['results'])

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True,  exist_ok=True)
        with open(ckpt_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # ── Dataset ────────────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    t         = cfg['train']

    train_ds = COCODetectionDataset(
        img_dir  = data_root / 'train2017',
        ann_file = data_root / 'annotations' / 'instances_train2017.json',
        transforms = get_train_transforms(t['min_size'], t['max_size']),
    )
    val_ds = COCODetectionDataset(
        img_dir  = data_root / 'val2017',
        ann_file = data_root / 'annotations' / 'instances_val2017.json',
        transforms = get_val_transforms(t['min_size'], t['max_size']),
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)

    per_gpu = max(1, t['batch_size'] // world_size)

    train_loader = DataLoader(
        train_ds, batch_size=per_gpu, sampler=train_sampler,
        num_workers=t['num_workers'], collate_fn=collate_fn,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=1, sampler=val_sampler,
        num_workers=t['num_workers'], collate_fn=collate_fn,
        pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  E3 Detection — {name}")
        print(f"  GPUs: {world_size}  |  Batch: {t['batch_size']}")
        print(f"{'='*60}\n", flush=True)

    detector = build_detector(cfg, paths).to(device)
    detector = DDP(detector, device_ids=[local_rank],
                   find_unused_parameters=True)

    # ── Optimiser — two param groups (backbone + head) ─────────────────────────
    backbone_params = list(detector.module.backbone.parameters())
    head_params     = (list(detector.module.neck.parameters()) +
                       list(detector.module.frcnn.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': t['backbone_lr']},
        {'params': head_params,     'lr': t['head_lr']},
    ], weight_decay=t['weight_decay'])

    total_epochs     = t['epochs']
    warmup_epochs    = t.get('warmup_epochs', 1)
    freeze_epochs    = t.get('freeze_backbone_epochs', 1)
    steps_per_epoch  = len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda', enabled=t['amp'])

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_ap     = 0.0
    history     = []

    if args.resume and Path(args.resume).exists():
        if rank == 0:
            print(f"Resuming from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location='cpu')
        detector.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_ap     = ckpt.get('best_ap', 0.0)
        hist_file   = res_dir / f'{name}.json'
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f).get('history', [])
        for _ in range(start_epoch):
            scheduler.step()
        if rank == 0:
            print(f"Resumed at epoch {start_epoch}  best_AP={best_ap:.2f}",
                  flush=True)

    save_freq = cfg.get('logging', {}).get('save_freq', 5)
    eval_freq = cfg.get('eval', {}).get('eval_freq', 5)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)

        # Freeze / unfreeze backbone
        if epoch < freeze_epochs:
            detector.module.backbone.freeze()
        else:
            detector.module.backbone.unfreeze()

        t0 = time.time()
        avg_loss = train_one_epoch(
            detector, train_loader, optimizer, scaler,
            epoch, cfg, device, rank)
        scheduler.step()
        dist.barrier()

        # Evaluate
        metrics = {}
        if (epoch + 1) % eval_freq == 0 or epoch == total_epochs - 1:
            metrics = evaluate_coco(detector.module, val_loader,
                                    device, rank, world_size)
            if rank == 0:
                ap = metrics.get('AP', 0.0)
                print(f"[{epoch+1}/{total_epochs}]  "
                      f"loss={avg_loss:.4f}  "
                      f"AP={ap:.2f}  AP50={metrics.get('AP50',0):.2f}  "
                      f"AP_S={metrics.get('AP_S',0):.2f}  "
                      f"time={time.time()-t0:.0f}s", flush=True)

                history.append({'epoch': epoch,
                                'train_loss': round(avg_loss, 4),
                                **metrics})

                if ap > best_ap:
                    best_ap = ap
                    torch.save(
                        {'epoch': epoch,
                         'model': detector.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict(),
                         'best_ap': best_ap,
                         **metrics},
                        ckpt_dir / 'last.pth')
                    print(f"  *** New best AP: {best_ap:.2f}%", flush=True)

        # Checkpoint
        if rank == 0:
            if (epoch + 1) % save_freq == 0 or epoch == total_epochs - 1:
                torch.save(
                    {'epoch': epoch,
                     'model': detector.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(),
                     'best_ap': best_ap},
                    ckpt_dir / 'last.pth')

    if rank == 0:
        print("\nRunning final evaluation...", flush=True)
    final_metrics = evaluate_coco(detector.module, val_loader,
                                  device, rank, world_size)
    if rank == 0:
        results = {
            'variant':  name,
            'arch':     cfg['model']['arch'],
            'gate':     cfg['model'].get('gate', False),
            'registers':cfg['model'].get('registers', 0),
            'epochs':   total_epochs,
            'best_AP':  best_ap,
            **{f'final_{k}': v for k, v in final_metrics.items()},
            'history':  history,
        }
        with open(res_dir / f'{name}.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Done — {name}")
        print(f"  best_AP:  {best_ap:.2f}%")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.2f}%")
        print(f"{'='*60}", flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()