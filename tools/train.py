#!/usr/bin/env python3
"""
SAGA universal training script.
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
    if '_base_' in cfg:
        base_path = Path(config_path).parent / cfg.pop('_base_')
        with open(base_path) as f:
            base = yaml.safe_load(f)
        cfg = deep_merge(base, cfg)
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
def compute_e1_metrics(model_unwrapped, val_loader, device, num_batches=50):
    gates     = [m for m in model_unwrapped.modules() if isinstance(m, SpatialGate)]
    last_feat = {}

    def feat_hook(mod, inp, out):
        last_feat['x'] = out.detach().cpu()

    hook = None
    if hasattr(model_unwrapped, 'blocks'):
        hook = model_unwrapped.blocks[-1].register_forward_hook(feat_hook)

    gate_vals  = []
    gate_hooks = []
    if gates:
        def make_gh(g):
            def gh(mod, inp, out):
                ip = inp[0][:, :, 1:, :]
                op = out[:, :, 1:, :]
                gate_vals.append(
                    (op.norm(dim=-1) / (ip.norm(dim=-1) + 1e-8)).clamp(0,1).detach().cpu()
                )
            return gh
        for gate in gates:
            gate_hooks.append(gate.register_forward_hook(make_gh(gate)))

    all_norms = []
    model_unwrapped.eval()
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        _ = model_unwrapped(images.to(device))
        if 'x' in last_feat:
            all_norms.append(last_feat['x'][:, 1:, :].norm(dim=-1))

    if hook:       hook.remove()
    for gh in gate_hooks: gh.remove()

    metrics = {}
    if gate_vals:
        g = torch.cat(gate_vals).float()
        metrics['gate_sparsity'] = round(g.mean().item(), 4)
        metrics['gate_variance'] = round(g.std().item(),  4)
    else:
        metrics['gate_sparsity'] = None
        metrics['gate_variance'] = None

    if all_norms:
        norms     = torch.cat(all_norms).float()
        mu, sigma = norms.mean(), norms.std()
        metrics['sink_score'] = round(
            (norms > mu + 3*sigma).float().mean().item() * 100, 3)
        p   = nn.functional.normalize(last_feat['x'][:, 1:, :], dim=-1)
        cos = (p[:, :-1, :] * p[:, 1:, :]).sum(-1).mean().item()
        metrics['oversmoothing_score'] = round(cos, 4)
    else:
        metrics['sink_score']          = None
        metrics['oversmoothing_score'] = None

    return metrics


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, mixup_fn,
                    scaler, scheduler, epoch, cfg, device, rank):
    model.train()
    loss_meter = AverageMeter()
    log_freq   = cfg.get('logging', {}).get('log_freq', 100)

    for step, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup/cutmix in the training loop if enabled
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

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

        scheduler.step_update(epoch * len(loader) + step)
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
        loss_m.update(loss.item(),  images.size(0))

    return {
        'val_loss': round(loss_m.avg, 4),
        'top1':     round(top1_m.avg, 3),
        'top5':     round(top5_m.avg, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser('SAGA trainer')
    parser.add_argument('--config',     required=True)
    parser.add_argument('--variant_id', type=int, default=-1)
    parser.add_argument('--data_root',  default=None)
    parser.add_argument('--out_dir',    default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    dist.init_process_group('nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    cfg = load_config(args.config, args.variant_id)
    if args.data_root:  cfg['data']['root']      = args.data_root
    if args.max_epochs: cfg['train']['epochs']   = args.max_epochs
    if args.batch_size: cfg['train']['batch_size'] = args.batch_size

    variant_name = cfg.get('variant', {}).get('name', 'run')
    out_dir      = Path(args.out_dir or 'outputs') / variant_name

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"\n{'='*56}")
        print(f"  SAGA  |  {variant_name}")
        print(f"  Terms: {cfg['gate'].get('terms', [])}  |  GPUs: {world_size}")
        print(f"  Batch: {cfg['train']['batch_size']} global  |  "
              f"{cfg['train']['batch_size']//world_size} per GPU")
        print(f"  Out:   {out_dir}")
        print(f"{'='*56}\n", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    g = cfg.get('gate', {})
    model = build_saga_vit(
        arch          = cfg['model']['arch'],
        gate_terms    = g.get('terms', []),
        img_size      = cfg['model']['img_size'],
        patch_size    = cfg['model']['patch_size'],
        granularity   = g.get('granularity',  'head_specific'),
        gate_position = g.get('position',      'G1'),
        lambda_0      = g.get('lambda_0',      0.10),
        beta          = g.get('beta',          0.10),
        mu            = g.get('mu',            0.05),
        init_bias     = g.get('init_bias',     4.0),
        num_classes   = cfg['model']['num_classes'],
        pretrained    = cfg['model'].get('pretrained', False),
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        n_gate  = sum(p.numel() for n, p in model.named_parameters()
                      if 'gate' in n.lower()) / 1e3
        print(f"Params: {n_total:.1f}M total  |  {n_gate:.1f}K gate", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root = cfg['data']['root']
    assert data_root, "Set --data_root"

    tr_cfg  = cfg['train']
    aug_cfg = cfg.get('augmentation', {})
    per_gpu = tr_cfg['batch_size'] // world_size

    train_ds = create_dataset('imagefolder', root=data_root,
                               split='train', is_training=True)
    val_ds   = create_dataset('imagefolder', root=data_root,
                               split='val',   is_training=False)

    mixup_active = (aug_cfg.get('mixup_alpha', 0) > 0 or
                    aug_cfg.get('cutmix_alpha', 0) > 0)

    # Mixup is applied manually in the training loop (not inside create_loader)
    # This works with all timm versions
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha     = aug_cfg.get('mixup_alpha',  0.8),
            cutmix_alpha    = aug_cfg.get('cutmix_alpha', 1.0),
            num_classes     = cfg['model']['num_classes'],
            label_smoothing = tr_cfg.get('label_smoothing', 0.1),
        )

    auto_augment = None
    if aug_cfg.get('rand_aug', True):
        m = aug_cfg.get('rand_aug_magnitude', 9)
        n = aug_cfg.get('rand_aug_layers',    2)
        auto_augment = f'rand-m{m}-n{n}-mstd0.5'

    train_loader = create_loader(
        train_ds,
        input_size   = (3, cfg['data']['input_size'], cfg['data']['input_size']),
        batch_size   = per_gpu,
        is_training  = True,
        re_prob      = aug_cfg.get('random_erase_prob', 0.25),
        auto_augment = auto_augment,
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

    # SoftTargetCrossEntropy when mixup is on (receives soft labels from mixup_fn)
    # LabelSmoothingCrossEntropy when mixup is off (receives hard integer labels)
    train_criterion = (SoftTargetCrossEntropy() if mixup_active else
                       LabelSmoothingCrossEntropy(
                           smoothing=tr_cfg.get('label_smoothing', 0.1)))
    val_criterion = nn.CrossEntropyLoss()

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    lr = tr_cfg['lr']
    if tr_cfg.get('lr_scaling', True):
        lr = lr * tr_cfg['batch_size'] / 1024

    optimizer = create_optimizer_v2(
        model.parameters(),
        opt          = tr_cfg['optimizer'],
        lr           = lr,
        weight_decay = tr_cfg['weight_decay'],
        betas        = tuple(tr_cfg.get('betas', [0.9, 0.999])),
    )

    total_epochs    = tr_cfg['epochs']
    steps_per_epoch = len(train_loader)
    warmup_epochs   = tr_cfg.get('warmup_epochs', 10)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial      = total_epochs  * steps_per_epoch,
        lr_min         = tr_cfg.get('min_lr', 1e-6),
        warmup_t       = warmup_epochs * steps_per_epoch,
        warmup_lr_init = 1e-6,
        cycle_limit    = 1,
        t_in_epochs    = False,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=tr_cfg['amp'])

    # ── Training loop ─────────────────────────────────────────────────────────
    log_cfg   = cfg.get('logging', {})
    save_freq = log_cfg.get('save_freq', 20)
    best_top1 = 0.0
    history   = []

    for epoch in range(total_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, train_criterion, mixup_fn,
            scaler, scheduler, epoch, cfg, device, rank,
        )
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
                torch.save({'epoch': epoch,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'top1': top1},
                           out_dir / 'best.pth')
                print(f"  *** New best: {top1:.2f}%", flush=True)

            if (epoch + 1) % save_freq == 0:
                torch.save({'epoch': epoch,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'top1': top1},
                           out_dir / 'last.pth')

    # ── Final metrics ─────────────────────────────────────────────────────────
    if rank == 0:
        print("\nComputing E1 gate metrics...", flush=True)
        e1 = compute_e1_metrics(model.module, val_loader, device, num_batches=50)

        results = {
            'variant':       variant_name,
            'gate_terms':    list(g.get('terms', [])),
            'gate_position': g.get('position', 'G1'),
            'granularity':   g.get('granularity', 'head_specific'),
            'lambda_0':      g.get('lambda_0', 0.10),
            'mu':            g.get('mu', 0.05),
            'best_top1':     best_top1,
            'final_top1':    history[-1]['top1'],
            'final_top5':    history[-1]['top5'],
            **e1,
            'history':       history,
        }

        results_dir = out_dir.parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f'{variant_name}.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nDone — {variant_name}")
        print(f"  best_top1:     {best_top1:.2f}%")
        print(f"  gate_sparsity: {e1.get('gate_sparsity')}")
        print(f"  sink_score:    {e1.get('sink_score')}%")
        print(f"  oversmoothing: {e1.get('oversmoothing_score')}", flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()