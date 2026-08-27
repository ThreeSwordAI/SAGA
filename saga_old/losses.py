#!/usr/bin/env python3
"""
SAGA universal training script.

Usage (E1 array job):
    torchrun --nproc_per_node=4 tools/train.py \\
        --config  configs/e1_variants.yaml \\
        --variant_id 0 \\
        --data_root   $IMAGENET_ROOT \\
        --out_dir     $OUT_ROOT/checkpoints/e1_variants

Usage (E1 sanity check — V00, 5 epochs):
    torchrun --nproc_per_node=4 tools/train.py \\
        --config  configs/e1_variants.yaml \\
        --variant_id 0 \\
        --max_epochs 5 \\
        --data_root   $IMAGENET_ROOT \\
        --out_dir     $OUT_ROOT/checkpoints/e1_sanity
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

import timm
from timm.data import create_dataset, create_loader, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.optim import create_optimizer_v2
from timm.utils import AverageMeter, accuracy

import yaml

from saga_old import build_saga_vit
from saga_old.gate import SpatialGate


# ── Config ────────────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str, variant_id: int = -1) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve _base_ inheritance
    if '_base_' in cfg:
        base_path = Path(config_path).parent / cfg.pop('_base_')
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = deep_merge(base, cfg)

    # Apply variant overrides (E1 array job)
    if variant_id >= 0 and 'variants' in cfg:
        variants = cfg.pop('variants')
        variant  = next(v for v in variants if v['id'] == variant_id)
        cfg['variant'] = variant
        if 'gate' in variant:
            cfg['gate'] = deep_merge(cfg.get('gate', {}), variant['gate'])
    else:
        cfg.pop('variants', None)

    return cfg


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_e1_metrics(model_unwrapped: nn.Module, val_loader, device,
                       num_batches: int = 50) -> dict:
    """
    Compute the 4 E1-specific metrics beyond top-1/top-5:
        gate_sparsity     mean gate value  (lower = more suppression)
        gate_variance     std  gate value  (higher = gate is doing something)
        sink_score        % patch tokens with L2 norm > mean + 3*std  (lower = fewer sinks)
        oversmoothing     mean cosine sim between adjacent patches at last layer (lower = more diverse)
    """

    gates = [m for m in model_unwrapped.modules() if isinstance(m, SpatialGate)]
    has_gate = len(gates) > 0

    all_gate_vals   = []
    all_patch_norms = []
    last_block_feats = {}

    # Hook: capture last ViT block output for oversmoothing
    def feat_hook(module, input, output):
        last_block_feats['x'] = output.detach().cpu()

    hook = None
    if hasattr(model_unwrapped, 'blocks'):
        hook = model_unwrapped.blocks[-1].register_forward_hook(feat_hook)

    # Hook: capture gate values
    gate_store = []
    gate_hooks = []
    if has_gate:
        def make_gate_hook(gate_module):
            def gh(module, input, output):
                # output is gated sdpa_out [B, H, N, D]
                # gate ≈ |output_patch| / (|input_patch| + ε)
                inp = input[0][:, :, 1:, :]   # patch tokens of sdpa_out
                out = output[:, :, 1:, :]
                g_approx = (out.norm(dim=-1) / (inp.norm(dim=-1) + 1e-8)).clamp(0, 1)
                gate_store.append(g_approx.detach().cpu())
            return gh
        for gate in gates:
            gate_hooks.append(gate.register_forward_hook(make_gate_hook(gate)))

    model_unwrapped.eval()
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        images = images.to(device)
        _ = model_unwrapped(images)

        if 'x' in last_block_feats:
            # Patch token norms at last layer (exclude CLS)
            feats = last_block_feats['x']  # [B, N, C]
            patch_feats = feats[:, 1:, :]  # [B, n_patches, C]
            all_patch_norms.append(patch_feats.norm(dim=-1))  # [B, n_patches]

    # Clean up hooks
    if hook:
        hook.remove()
    for gh in gate_hooks:
        gh.remove()

    metrics = {}

    # Gate sparsity and variance
    if gate_store:
        all_g = torch.cat(gate_store, dim=0).float()  # [total_B*layers, H, n]
        metrics['gate_sparsity'] = round(all_g.mean().item(), 4)
        metrics['gate_variance'] = round(all_g.std().item(),  4)
    else:
        metrics['gate_sparsity'] = None
        metrics['gate_variance'] = None

    # Sink score: % patches with norm > mean + 3*std
    if all_patch_norms:
        norms = torch.cat(all_patch_norms, dim=0).float()  # [total_B, n_patches]
        mu, sigma = norms.mean(), norms.std()
        metrics['sink_score'] = round(
            ((norms > mu + 3 * sigma).float().mean().item()) * 100, 3
        )
    else:
        metrics['sink_score'] = None

    # Oversmoothing: mean cosine similarity between adjacent patch tokens
    # Re-run a small pass after removing gate hooks to get clean features
    if 'x' in last_block_feats and all_patch_norms:
        # Already collected features in the loop above
        # Approximate oversmoothing from last batch
        feats = last_block_feats['x']  # [B, N, C]
        p = nn.functional.normalize(feats[:, 1:, :], dim=-1)  # [B, n, C]
        # Horizontal neighbor cosine similarity (right neighbor)
        cos = (p[:, :-1, :] * p[:, 1:, :]).sum(-1).mean().item()
        metrics['oversmoothing_score'] = round(cos, 4)
    else:
        metrics['oversmoothing_score'] = None

    return metrics


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, scheduler, epoch, total_steps, cfg, device, rank):
    model.train()
    loss_meter = AverageMeter()
    log_freq   = cfg.get('logging', {}).get('log_freq', 100)

    for step, (images, targets) in enumerate(loader):
        images  = images.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16,
                             enabled=cfg['train']['amp']):
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

        global_step = epoch * len(loader) + step
        scheduler.step_update(global_step)

        loss_meter.update(loss.item(), images.size(0))

        if rank == 0 and step % log_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [{epoch}][{step}/{len(loader)}] "
                  f"loss={loss_meter.avg:.4f}  lr={lr:.2e}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    loss_m = AverageMeter()

    for images, targets in loader:
        images  = images.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(images)
            loss   = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))
        loss_m.update(loss.item(),  images.size(0))

    return {
        'val_loss': round(loss_m.avg,  4),
        'top1':     round(top1_m.avg, 3),
        'top5':     round(top5_m.avg, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser('SAGA trainer')
    parser.add_argument('--config',      required=True)
    parser.add_argument('--variant_id',  type=int, default=-1)
    parser.add_argument('--data_root',   default=None)
    parser.add_argument('--out_dir',     default=None)
    parser.add_argument('--max_epochs',  type=int, default=None)
    args = parser.parse_args()

    # ── Distributed init ──────────────────────────────────────────────────────
    dist.init_process_group('nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config, args.variant_id)

    if args.data_root:
        cfg['data']['root'] = args.data_root
    if args.max_epochs:
        cfg['train']['epochs'] = args.max_epochs

    variant_name = cfg.get('variant', {}).get('name', 'run')
    out_dir      = Path(args.out_dir or cfg.get('out_dir', 'outputs')) / variant_name

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"\n{'='*56}")
        print(f"  SAGA  |  {variant_name}")
        print(f"  Terms: {cfg['gate'].get('terms', [])}  |  GPUs: {world_size}")
        print(f"  Out:   {out_dir}")
        print(f"{'='*56}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = cfg.get('gate', {})
    model = build_saga_vit(
        arch          = cfg['model']['arch'],
        gate_terms    = g.get('terms', []),
        img_size      = cfg['model']['img_size'],
        patch_size    = cfg['model']['patch_size'],
        granularity   = g.get('granularity',   'head_specific'),
        gate_position = g.get('position',       'G1'),
        lambda_0      = g.get('lambda_0',       0.10),
        beta          = g.get('beta',           0.10),
        mu            = g.get('mu',             0.05),
        init_bias     = g.get('init_bias',      4.0),
        num_classes   = cfg['model']['num_classes'],
        pretrained    = cfg['model'].get('pretrained', False),
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        n_gate  = sum(p.numel() for n, p in model.named_parameters()
                      if 'gate' in n.lower()) / 1e3
        print(f"Params: {n_total:.1f}M total  |  {n_gate:.1f}K gate")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root = cfg['data']['root']
    assert data_root, "Set --data_root or data.root in config"

    tr_cfg   = cfg['train']
    aug_cfg  = cfg.get('augmentation', {})
    per_gpu  = tr_cfg['batch_size'] // world_size

    train_ds = create_dataset('torch/imagenet', root=data_root, split='train',
                               is_training=True, download=False, batch_size=per_gpu)
    val_ds   = create_dataset('torch/imagenet', root=data_root, split='validation',
                               is_training=False, download=False, batch_size=per_gpu)

    mixup_fn    = None
    mixup_active = aug_cfg.get('mixup_alpha', 0) > 0 or aug_cfg.get('cutmix_alpha', 0) > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha  = aug_cfg.get('mixup_alpha',  0.8),
            cutmix_alpha = aug_cfg.get('cutmix_alpha', 1.0),
            num_classes  = cfg['model']['num_classes'],
            label_smoothing = tr_cfg.get('label_smoothing', 0.1),
        )

    ra_str = None
    if aug_cfg.get('rand_aug', True):
        m = aug_cfg.get('rand_aug_magnitude', 9)
        n = aug_cfg.get('rand_aug_layers',    2)
        ra_str = f'rand-m{m}-n{n}'

    train_loader = create_loader(
        train_ds, input_size=cfg['data']['input_size'],
        batch_size=per_gpu, is_training=True, use_prefetcher=True,
        rand_erase_prob=aug_cfg.get('random_erase_prob', 0.25),
        rand_aug=ra_str, num_workers=cfg['data']['num_workers'],
        distributed=True, pin_memory=cfg['data'].get('pin_memory', True),
    )
    val_loader = create_loader(
        val_ds, input_size=cfg['data']['input_size'],
        batch_size=per_gpu, is_training=False, use_prefetcher=True,
        num_workers=cfg['data']['num_workers'],
        distributed=True, pin_memory=cfg['data'].get('pin_memory', True),
    )

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    train_criterion = SoftTargetCrossEntropy() if mixup_active else \
                      LabelSmoothingCrossEntropy(smoothing=tr_cfg.get('label_smoothing', 0.1))
    val_criterion   = nn.CrossEntropyLoss()

    lr = tr_cfg['lr']
    if tr_cfg.get('lr_scaling', True):
        lr = lr * tr_cfg['batch_size'] / 1024

    optimizer = create_optimizer_v2(
        model.parameters(), opt=tr_cfg['optimizer'], lr=lr,
        weight_decay=tr_cfg['weight_decay'],
        betas=tuple(tr_cfg.get('betas', [0.9, 0.999])),
    )

    total_epochs      = tr_cfg['epochs']
    steps_per_epoch   = len(train_loader)
    warmup_epochs     = tr_cfg.get('warmup_epochs', 10)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial     = total_epochs  * steps_per_epoch,
        lr_min        = tr_cfg.get('min_lr', 1e-6),
        warmup_t      = warmup_epochs * steps_per_epoch,
        warmup_lr_init= 1e-6,
        cycle_limit   = 1,
        t_in_epochs   = False,
    )

    scaler = GradScaler(enabled=tr_cfg['amp'])

    # ── Training loop ─────────────────────────────────────────────────────────
    log_cfg     = cfg.get('logging', {})
    save_freq   = log_cfg.get('save_freq', 20)
    best_top1   = 0.0
    history     = []

    for epoch in range(total_epochs):
        train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, train_criterion,
            scaler, scheduler, epoch, total_epochs * steps_per_epoch,
            cfg, device, rank,
        )
        dist.barrier()

        val_metrics = validate(model, val_loader, val_criterion, device)
        scheduler.step(epoch + 1)

        if rank == 0:
            top1 = val_metrics['top1']
            row  = {'epoch': epoch, 'train_loss': round(train_loss, 4),
                    'lr': round(optimizer.param_groups[0]['lr'], 8),
                    **val_metrics}
            history.append(row)

            print(f"[{epoch+1}/{total_epochs}]  "
                  f"top1={top1:.2f}%  top5={val_metrics['top5']:.2f}%  "
                  f"loss={train_loss:.4f}")

            # Best checkpoint
            if top1 > best_top1:
                best_top1 = top1
                torch.save({'epoch': epoch, 'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(), 'top1': top1},
                           out_dir / 'best.pth')
                print(f"  *** New best: {top1:.2f}%")

            # Periodic checkpoint
            if (epoch + 1) % save_freq == 0:
                torch.save({'epoch': epoch, 'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(), 'top1': top1},
                           out_dir / 'last.pth')

    # ── Final: E1-specific gate metrics (rank 0 only) ─────────────────────────
    if rank == 0:
        print("\nComputing E1 gate metrics on validation set...")
        e1_metrics = compute_e1_metrics(model.module, val_loader, device, num_batches=50)

        results = {
            'variant':           variant_name,
            'gate_terms':        list(g.get('terms', [])),
            'gate_position':     g.get('position', 'G1'),
            'granularity':       g.get('granularity', 'head_specific'),
            'lambda_0':          g.get('lambda_0', 0.10),
            'mu':                g.get('mu', 0.05),
            'best_top1':         best_top1,
            'final_top1':        history[-1]['top1'],
            'final_top5':        history[-1]['top5'],
            **e1_metrics,
            'history':           history,
        }

        # Save per-variant JSON to results/ next to the checkpoints root
        results_dir = out_dir.parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f'{variant_name}.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nDone — {variant_name}")
        print(f"  best_top1:        {best_top1:.2f}%")
        print(f"  gate_sparsity:    {e1_metrics.get('gate_sparsity')}")
        print(f"  sink_score:       {e1_metrics.get('sink_score')}%")
        print(f"  oversmoothing:    {e1_metrics.get('oversmoothing_score')}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()