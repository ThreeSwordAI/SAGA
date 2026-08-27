#!/usr/bin/env python3
"""
detection/tools/evaluate.py
============================
Standalone COCO evaluation on a saved checkpoint.
Use this after training to re-run evaluation or evaluate best.pth
on a different split.

Usage:
    python3 detection/tools/evaluate.py \
        --config  detection/configs/variants.yaml \
        --id      2 \
        --paths   detection/configs/paths.yaml \
        --ckpt    /home/vault/iwi5/iwi5359h/SAGA/e3/checkpoints/ViT-B_SAGA_det/best.pth \
        --data_root /tmp/coco_staged \
        --split   val

Outputs:
    Prints AP / AP50 / AP75 / AP_S / AP_M / AP_L to terminal.
    Saves results JSON to the results directory.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detection.data.coco_dataset import COCODetectionDataset, collate_fn
from detection.data.transforms   import get_val_transforms
from detection.models.detector   import build_detector


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
    raw      = load_yaml(variants_path)
    base_file= Path(variants_path).parent / raw.get('_base_', 'base.yaml')
    cfg      = load_yaml(base_file)
    variant  = next(v for v in raw['variants'] if v['id'] == variant_id)
    top_overrides = {k: v for k, v in raw.items()
                     if k not in ('_base_', 'variants')}
    cfg = deep_merge(cfg, top_overrides)
    cfg = deep_merge(cfg, {k: v for k, v in variant.items() if k != 'id'})
    cfg['variant_id']   = variant_id
    cfg['variant_name'] = variant['name']
    cfg['paths']        = load_yaml(paths_path)
    return cfg


@torch.no_grad()
def run_evaluation(detector, loader, device):
    """Run COCO evaluation and return metrics dict."""
    detector.eval()
    from pycocotools.cocoeval import COCOeval

    coco_gt = loader.dataset.get_coco_api()
    results = []

    print(f"  Running inference on {len(loader.dataset):,} images...")
    for i, (images, targets) in enumerate(loader):
        if i % 200 == 0:
            print(f"    {i}/{len(loader)}", flush=True)

        images = [img.to(device) for img in images]
        preds  = detector(images)

        for pred, tgt in zip(preds, targets):
            img_id = tgt['image_id'].item()
            boxes  = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    'image_id':    img_id,
                    'category_id': int(label),
                    'bbox':        [x1, y1, x2 - x1, y2 - y1],
                    'score':       float(score),
                })

    if not results:
        print("  WARNING: no predictions produced.")
        return {k: 0.0 for k in
                ['AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L']}

    coco_dt   = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()

    print("\n  === COCO Evaluation Results ===")
    evaluator.summarize()

    stats = evaluator.stats
    return {
        'AP':   round(float(stats[0]) * 100, 2),
        'AP50': round(float(stats[1]) * 100, 2),
        'AP75': round(float(stats[2]) * 100, 2),
        'AP_S': round(float(stats[3]) * 100, 2),
        'AP_M': round(float(stats[4]) * 100, 2),
        'AP_L': round(float(stats[5]) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Standalone COCO evaluation on a detection checkpoint.')
    parser.add_argument('--config',    required=True,
                        help='detection/configs/variants.yaml')
    parser.add_argument('--id',        type=int, required=True,
                        help='Variant id (0-2)')
    parser.add_argument('--paths',     required=True,
                        help='detection/configs/paths.yaml')
    parser.add_argument('--ckpt',      required=True,
                        help='Path to checkpoint (best.pth or last.pth)')
    parser.add_argument('--data_root', required=True,
                        help='Staged COCO root directory')
    parser.add_argument('--split',     default='val',
                        choices=['val', 'train'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--num_workers',type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg   = load_config(args.config, args.id, args.paths)
    name  = cfg['variant_name']
    paths = cfg['paths']
    t     = cfg['train']

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Split:      {args.split}")
    print(f"{'='*60}\n")

    # Dataset
    data_root = Path(args.data_root)
    ds = COCODetectionDataset(
        img_dir   = data_root / f'{args.split}2017',
        ann_file  = data_root / 'annotations' / f'instances_{args.split}2017.json',
        transforms= get_val_transforms(t['min_size'], t['max_size']),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn, shuffle=False)

    # Model
    detector = build_detector(cfg, paths).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    detector.load_state_dict(state, strict=True)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"  Saved AP:         {ckpt.get('AP', ckpt.get('best_ap', '?'))}")

    # Evaluate
    metrics = run_evaluation(detector, loader, device)

    print(f"\n{'='*60}")
    print(f"  {name}  [{args.split}]")
    for k, v in metrics.items():
        print(f"  {k:8s} = {v:.2f}%")
    print(f"{'='*60}\n")

    # Save results
    res_dir  = Path(paths['outputs']['results'])
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / f'{name}_eval_{args.split}.json'

    with open(out_file, 'w') as f:
        json.dump({
            'variant':   name,
            'checkpoint': args.ckpt,
            'split':      args.split,
            **metrics,
        }, f, indent=2)
    print(f"  Results saved to: {out_file}")


if __name__ == '__main__':
    main()