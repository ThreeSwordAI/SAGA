#!/usr/bin/env python3
"""
segmentation/tools/evaluate.py
================================
SUPERSEDED (TASK-09) — REFUSES TO RUN. Kept for provenance only.

Its `evaluate()` takes an `ignore_index=255` argument and then NEVER USES IT
(bug B4: the 255 unlabelled pixels land in every class's union), and it
averages per-IMAGE IoU instead of computing dataset-level IoU — two
different reasons its mIoU is not the metric anyone means. It also depends
on segmentation/configs/paths.yaml, which is a stale copy of the e6
fine-grained paths file and carries no ADE20K entries at all.

The mIoU that Gate 2 uses is produced by segmentation/tools/train.py, which
accumulates a confusion matrix over `gt != 255` pixels only and writes
miou_ss.json / miou_ms.json / per_class_iou.csv / conf_matrix.npz. To
re-evaluate a checkpoint, resume that run (its outputs are idempotent)
rather than reviving this script — reviving it would need the B4 fix and the
dataset-level metric ported over first.

Original usage (no longer functional):
    python3 segmentation/tools/evaluate.py \
        --config    segmentation/configs/variants.yaml \
        --id        2 \
        --paths     segmentation/configs/paths.yaml \
        --ckpt      /path/to/ViT-B_SAGA_seg/best.pth \
        --data_root /tmp/ade20k_staged/ADEChallengeData2016
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from segmentation.data.ade20k_dataset import ADE20KDataset
from segmentation.data.transforms     import get_val_transforms
from segmentation.models.segmentor    import build_segmentor


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


@torch.no_grad()
def evaluate(model, loader, device, num_classes=150, ignore_index=255):
    model.eval()
    import numpy as np
    iou_sum   = np.zeros(num_classes, dtype=np.float64)
    iou_count = np.zeros(num_classes, dtype=np.float64)

    for i, (images, masks) in enumerate(loader):
        if i % 200 == 0:
            print(f"  {i}/{len(loader)}", flush=True)
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        preds  = logits.argmax(dim=1)   # [B, H, W]

        for cls in range(num_classes):
            pred_c = (preds == cls).cpu().numpy().astype(bool)
            gt_c   = (masks == cls).cpu().numpy().astype(bool)
            inter  = (pred_c & gt_c).sum()
            uni    = (pred_c | gt_c).sum()
            if uni > 0:
                iou_sum[cls]   += inter / uni
                iou_count[cls] += 1

    valid = iou_count > 0
    miou  = (iou_sum[valid] / iou_count[valid]).mean() * 100.0
    return {'mIoU': round(float(miou), 2),
            'iou_per_class': (iou_sum / np.maximum(iou_count, 1)).tolist()}


def main():
    # TASK-09: this entry point is disabled on purpose. Its metric is bug
    # B4 (ignore_index accepted and never applied) PLUS a per-image IoU
    # average, so any number it printed would be void — and "numbers are
    # sacred" means a runnable path to a void number is a defect, not a
    # convenience. See the module docstring.
    raise SystemExit(
        "segmentation/tools/evaluate.py is SUPERSEDED (TASK-09) and refuses "
        "to run: its mIoU ignores ignore_index=255 (bug B4) and averages "
        "per-image IoU. Use segmentation/tools/train.py (--matrix "
        "configs/dense_matrix.yaml --run seg_vitb_<variant>_s1), which "
        "writes miou_ss.json / miou_ms.json / per_class_iou.csv / "
        "conf_matrix.npz from a confusion matrix over labelled pixels only.")


def _disabled_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    required=True)
    parser.add_argument('--id',        type=int, required=True)
    parser.add_argument('--paths',     required=True)
    parser.add_argument('--ckpt',      required=True)
    parser.add_argument('--data_root', required=True,
                        help='Path to ADEChallengeData2016/')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = load_config(args.config, args.id, args.paths)
    name   = cfg['variant_name']
    paths  = cfg['paths']
    t      = cfg['train']

    print(f"\n{'='*55}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"{'='*55}\n")

    ds = ADE20KDataset(
        args.data_root, split='validation',
        transforms=get_val_transforms(t['input_size']))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=args.num_workers,
        shuffle=False, pin_memory=True)

    model = build_segmentor(cfg, paths).to(device)
    ckpt  = torch.load(args.ckpt, map_location='cpu')
    state = {k.replace('module.', ''): v
             for k, v in ckpt.get('model', ckpt).items()}
    model.load_state_dict(state, strict=True)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"  Saved mIoU:       {ckpt.get('best_miou', ckpt.get('mIoU', '?'))}")

    metrics = evaluate(model, loader, device,
                       num_classes=cfg['model']['num_classes'])

    print(f"\n  mIoU = {metrics['mIoU']:.2f}%")

    res_dir  = Path(paths['outputs']['results'])
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / f'{name}_eval.json'
    with open(out_file, 'w') as f:
        json.dump({'variant': name, 'checkpoint': args.ckpt, **metrics},
                  f, indent=2)
    print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()