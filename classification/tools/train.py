#!/usr/bin/env python3
"""
classification/tools/train.py
==============================
E2 training script — 300 epochs, ImageNet-1K.
Handles ViT baseline, ViT + registers, ViT + SAGA gate.

Usage:
    torchrun --nproc_per_node=4 tools/train.py \
        --config   configs/variants.yaml \
        --id       5 \
        --paths    configs/paths.yaml \
        --data_root $STAGE_DIR

    # Resume after hitting 24h time limit:
    torchrun --nproc_per_node=4 tools/train.py \
        --config   configs/variants.yaml \
        --id       5 \
        --paths    configs/paths.yaml \
        --data_root $STAGE_DIR \
        --resume   /path/to/checkpoints/ViT-B_SAGA/last.pth
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.data import create_dataset, create_loader, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter, accuracy

import yaml

# ── SAGA ──────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from saga import build_saga_vit


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
    raw = load_yaml(variants_path)

    # Load base config
    base_file = Path(variants_path).parent / raw.get('_base_', 'base.yaml')
    cfg = load_yaml(base_file)

    # Find and merge variant
    variant = next(v for v in raw['variants'] if v['id'] == variant_id)
    cfg = deep_merge(cfg, {k: v for k, v in variant.items() if k != 'id'})
    cfg['variant_id']   = variant_id
    cfg['variant_name'] = variant['name']

    # Merge paths
    cfg['paths'] = load_yaml(paths_path)

    return cfg


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    m = cfg['model']
    arch      = m['arch']
    use_gate  = m.get('gate', False)
    n_reg     = m.get('registers', 0)
    img_size  = cfg['model'].get('img_size', 224)
    patch_size= cfg['model'].get('patch_size', 16)
    n_classes = cfg['model'].get('num_classes', 1000)

    if n_reg > 0 and use_gate:
        raise ValueError("Cannot use both registers and SAGA gate in the same variant.")

    if n_reg > 0:
        # ViT + register tokens
        # timm 1.0.12+ uses reg_tokens parameter
        import timm
        model = timm.create_model(
            arch,
            pretrained  = False,
            num_classes = n_classes,
            img_size    = img_size,
            reg_tokens  = n_reg,   # timm 1.0.12+ parameter name
        )
    else:
        # ViT baseline (gate=False) or SAGA (gate=True)
        model = build_saga_vit(
            arch       = arch,
            gate       = use_gate,
            img_size   = img_size,
            patch_size = patch_size,
            num_classes= n_classes,
            pretrained = False,
        )

    return model


# ── Metrics helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_saga_metrics(model_unwrapped, val_loader, device, num_batches=50):
    """
    Compute sink_score and oversmoothing score from validation set.
    Also reads learned gate maps (φ_h) if the model has SAGA gates.
    """
    last_feat = {}
    hook = model_unwrapped.blocks[-1].register_forward_hook(
        lambda m, i, o: last_feat.update({'x': o.detach().cpu()})
    )

    all_norms = []
    model_unwrapped.eval()
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        _ = model_unwrapped(images.to(device))
        if 'x' in last_feat:
            all_norms.append(last_feat['x'][:, 1:, :].norm(dim=-1))
    hook.remove()

    metrics = {}
    if all_norms:
        norms = torch.cat(all_norms).float()
        mu, sigma = norms.mean(), norms.std()
        metrics['sink_score'] = round(
            (norms > mu + 3 * sigma).float().mean().item() * 100, 3)
        p = nn.functional.normalize(last_feat['x'][:, 1:, :], dim=-1)
        cos = (p[:, :-1, :] * p[:, 1:, :]).sum(-1).mean().item()
        metrics['oversmoothing_score'] = round(cos, 4)
    else:
        metrics['sink_score'] = None
        metrics['oversmoothing_score'] = None

    # Gate map statistics (only if SAGA model)
    gate_maps = []
    for block in model_unwrapped.blocks:
        attn = block.attn
        if hasattr(attn, 'gate') and attn.gate is not None:
            gate_maps.append(attn.gate.get_gate_maps())  # [H, 14, 14]

    if gate_maps:
        all_gates = torch.stack(gate_maps)               # [L, H, 14, 14]
        metrics['gate_mean']     = round(all_gates.mean().item(), 4)
        metrics['gate_std']      = round(all_gates.std().item(), 4)
        metrics['gate_min']      = round(all_gates.min().item(), 4)
        metrics['gate_max']      = round(all_gates.max().item(), 4)
    else:
        metrics['gate_mean'] = None
        metrics['gate_std']  = None
        metrics['gate_min']  = None
        metrics['gate_max']  = None

    return metrics


def save_gate_snapshot(model_unwrapped, epoch: int, out_dir: Path):
    """
    Save learned gate maps φ_h at a given epoch for later visualisation.
    Saved as a dict: {layer_idx: tensor [H, H_g, W_g]}
    """
    snapshots = {}
    for i, block in enumerate(model_unwrapped.blocks):
        attn = block.attn
        if hasattr(attn, 'gate') and attn.gate is not None:
            snapshots[i] = attn.gate.get_gate_maps()

    if snapshots:
        save_path = out_dir / f'gate_maps_epoch{epoch:04d}.pt'
        torch.save(snapshots, save_path)


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, mixup_fn,
                    scaler, scheduler, epoch, cfg, device, rank):
    model.train()
    loss_meter = AverageMeter()
    log_freq   = cfg.get('logging', {}).get('log_freq', 100)

    for step, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['train']['amp']):
            output = model(images)
            loss   = criterion(output, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        clip = cfg['train'].get('grad_clip', 1.0)
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        steps_per_epoch = len(loader)
        scheduler.step_update(epoch * steps_per_epoch + step)
        loss_meter.update(loss.item(), images.size(0))

        if rank == 0 and step % log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [{epoch}][{step}/{len(loader)}] "
                  f"loss={loss_meter.avg:.4f}  lr={lr:.2e}", flush=True)

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    loss_m = AverageMeter()

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(images)
            loss   = criterion(output, targets)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))
        loss_m.update(loss.item(), images.size(0))

    return {
        'val_loss': round(loss_m.avg, 4),
        'top1':     round(top1_m.avg, 3),
        'top5':     round(top5_m.avg, 3),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True, help='configs/variants.yaml')
    parser.add_argument('--id',         type=int, required=True, help='variant id (0-8)')
    parser.add_argument('--paths',      required=True, help='configs/paths.yaml')
    parser.add_argument('--data_root',  required=True, help='staged ImageNet root')
    parser.add_argument('--resume',     default=None,  help='path to last.pth for resume')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='override epochs from config (e.g. 3 for sanity check)')
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
    name = cfg['variant_name']

    # Output directories
    ckpt_dir = Path(cfg['paths']['outputs']['checkpoints']) / name
    res_dir  = Path(cfg['paths']['outputs']['results'])
    fig_dir  = Path(cfg['paths']['outputs']['figures'])
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # Model
    model = build_model(cfg).to(device)

    if rank == 0:
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        n_gate  = sum(p.numel() for n, p in model.named_parameters()
                      if 'gate' in n.lower() or 'phi' in n) / 1e3
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"  Total params: {n_total:.1f}M  |  Gate params: {n_gate:.1f}K")
        print(f"  GPUs: {world_size}  |  Batch: {cfg['train']['batch_size']}")
        if args.resume:
            print(f"  Resume: {args.resume}")
        print(f"{'='*60}\n", flush=True)

    # Data
    per_gpu = cfg['train']['batch_size'] // world_size
    aug     = cfg.get('augmentation', {})

    train_ds = create_dataset('imagefolder', root=args.data_root,
                               split='train', is_training=True)
    val_ds   = create_dataset('imagefolder', root=args.data_root,
                               split='val',   is_training=False)

    mixup_active = (aug.get('mixup_alpha', 0) > 0 or aug.get('cutmix_alpha', 0) > 0)
    mixup_fn = Mixup(
        mixup_alpha     = aug.get('mixup_alpha',  0.8),
        cutmix_alpha    = aug.get('cutmix_alpha', 1.0),
        num_classes     = cfg['model']['num_classes'],
        label_smoothing = cfg['train'].get('label_smoothing', 0.1),
    ) if mixup_active else None

    auto_aug = None
    if aug.get('rand_aug', True):
        m = aug.get('rand_aug_magnitude', 9)
        n = aug.get('rand_aug_layers', 2)
        auto_aug = f'rand-m{m}-n{n}-mstd0.5'

    train_loader = create_loader(
        train_ds,
        input_size   = (3, cfg['data']['input_size'], cfg['data']['input_size']),
        batch_size   = per_gpu,
        is_training  = True,
        re_prob      = aug.get('random_erase_prob', 0.25),
        auto_augment = auto_aug,
        num_workers  = cfg['data']['num_workers'],
        distributed  = True,
        pin_memory   = cfg['data'].get('pin_memory', True),
    )
    val_loader = create_loader(
        val_ds,
        input_size  = (3, cfg['data']['input_size'], cfg['data']['input_size']),
        batch_size  = per_gpu,
        is_training = False,
        num_workers = cfg['data']['num_workers'],
        distributed = True,
        pin_memory  = cfg['data'].get('pin_memory', True),
    )

    # Loss
    train_criterion = (SoftTargetCrossEntropy() if mixup_active
                       else LabelSmoothingCrossEntropy(
                           smoothing=cfg['train'].get('label_smoothing', 0.1)))
    val_criterion = nn.CrossEntropyLoss()

    # Optimiser + scheduler
    tr = cfg['train']
    lr = tr['lr'] * tr['batch_size'] / 1024 if tr.get('lr_scaling', True) else tr['lr']

    optimizer = create_optimizer_v2(
        model.parameters(),
        opt          = tr['optimizer'],
        lr           = lr,
        weight_decay = tr['weight_decay'],
        betas        = tuple(tr.get('betas', [0.9, 0.999])),
    )

    total_epochs    = tr['epochs']
    steps_per_epoch = len(train_loader)
    warmup_epochs   = tr.get('warmup_epochs', 20)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial      = total_epochs * steps_per_epoch,
        lr_min         = tr.get('min_lr', 1e-6),
        warmup_t       = warmup_epochs * steps_per_epoch,
        warmup_lr_init = 1e-6,
        cycle_limit    = 1,
        t_in_epochs    = False,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=tr['amp'])

    # Resume
    start_epoch = 0
    best_top1   = 0.0
    history     = []

    if args.resume and Path(args.resume).exists():
        if rank == 0:
            print(f"Resuming from: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_top1   = ckpt.get('best_top1', 0.0)

        # Fast-forward scheduler
        for ep in range(start_epoch):
            for step in range(steps_per_epoch):
                scheduler.step_update(ep * steps_per_epoch + step)
            scheduler.step(ep + 1)

        # Load existing history
        hist_file = res_dir / f'{name}.json'
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f).get('history', [])

        if rank == 0:
            print(f"Resumed at epoch {start_epoch}  best_top1={best_top1:.2f}%", flush=True)

    # DDP wrap after loading checkpoint
    model = DDP(model, device_ids=[local_rank])

    save_freq = cfg.get('logging', {}).get('save_freq', 25)

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, train_criterion, mixup_fn,
            scaler, scheduler, epoch, cfg, device, rank)
        dist.barrier()

        val_metrics = validate(model, val_loader, val_criterion, device)
        scheduler.step(epoch + 1)

        if rank == 0:
            top1 = val_metrics['top1']
            history.append({'epoch': epoch,
                            'train_loss': round(train_loss, 4),
                            'lr': round(optimizer.param_groups[0]['lr'], 8),
                            **val_metrics})
            print(f"[{epoch+1}/{total_epochs}]  "
                  f"top1={top1:.2f}%  top5={val_metrics['top5']:.2f}%  "
                  f"loss={train_loss:.4f}", flush=True)

            if top1 > best_top1:
                best_top1 = top1
                torch.save(
                    {'epoch': epoch,
                     'model': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(),
                     'best_top1': best_top1,
                     'top1': top1},
                    ckpt_dir / 'best.pth',
                )
                print(f"  *** New best: {top1:.2f}%", flush=True)

            # Save checkpoint every save_freq epochs
            if (epoch + 1) % save_freq == 0 or epoch == total_epochs - 1:
                torch.save(
                    {'epoch': epoch,
                     'model': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(),
                     'best_top1': best_top1,
                     'top1': top1},
                    ckpt_dir / 'last.pth',
                )
                # Save gate maps snapshot for visualisation
                save_gate_snapshot(model.module, epoch + 1, ckpt_dir)
                print(f"  Checkpoint + gate snapshot saved at epoch {epoch+1}", flush=True)

    # Final metrics
    if rank == 0:
        print("\nComputing final metrics...", flush=True)
        saga_metrics = compute_saga_metrics(model.module, val_loader, device)

        results = {
            'variant':     name,
            'arch':        cfg['model']['arch'],
            'gate':        cfg['model'].get('gate', False),
            'registers':   cfg['model'].get('registers', 0),
            'epochs':      total_epochs,
            'best_top1':   best_top1,
            'final_top1':  history[-1]['top1'],
            'final_top5':  history[-1]['top5'],
            **saga_metrics,
            'history':     history,
        }

        with open(res_dir / f'{name}.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Done — {name}")
        print(f"  best_top1:    {best_top1:.2f}%")
        print(f"  sink_score:   {saga_metrics.get('sink_score')}%")
        print(f"  oversmoothing:{saga_metrics.get('oversmoothing_score')}")
        print(f"  gate_mean:    {saga_metrics.get('gate_mean')}")
        print(f"{'='*60}", flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()