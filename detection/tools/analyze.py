#!/usr/bin/env python3
"""
detection/tools/analyze.py
============================
Deep analysis of detection results for the paper.

Produces:
  1. Per-category AP breakdown (which classes benefit most from SAGA)
  2. AP_S / AP_M / AP_L comparison across all three variants
  3. Sink score vs AP_S correlation (if E2 results available)
  4. Summary table printed to terminal (copy-paste ready for paper)

Usage:
    python3 detection/tools/analyze.py \
        --config    detection/configs/variants.yaml \
        --paths     detection/configs/paths.yaml \
        --data_root /tmp/coco_staged \
        --ckpts     /path/to/ViT-B_baseline_det/best.pth \
                    /path/to/ViT-B_registers_det/best.pth \
                    /path/to/ViT-B_SAGA_det/best.pth \
        --out_dir   /home/woody/iwi5/iwi5359h/saga_e3_figures

Requires all 3 checkpoints to be available for comparison.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
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
    top      = {k: v for k, v in raw.items() if k not in ('_base_', 'variants')}
    cfg = deep_merge(cfg, top)
    cfg = deep_merge(cfg, {k: v for k, v in variant.items() if k != 'id'})
    cfg['variant_id']   = variant_id
    cfg['variant_name'] = variant['name']
    cfg['paths']        = load_yaml(paths_path)
    return cfg


@torch.no_grad()
def get_predictions(detector, loader, device, variant_name):
    """Run inference and return raw predictions for pycocotools."""
    detector.eval()
    results = []
    print(f"  Running inference: {variant_name}")

    for i, (images, targets) in enumerate(loader):
        if i % 200 == 0:
            print(f"    {i}/{len(loader)}", flush=True)
        images = [img.to(device) for img in images]
        preds  = detector(images)

        for pred, tgt in zip(preds, targets):
            img_id = tgt['image_id'].item()
            for box, score, label in zip(pred['boxes'].cpu(),
                                         pred['scores'].cpu(),
                                         pred['labels'].cpu()):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    'image_id':    img_id,
                    'category_id': int(label),
                    'bbox':        [x1, y1, x2 - x1, y2 - y1],
                    'score':       float(score),
                })
    return results


def per_category_ap(coco_gt, results, cat_ids):
    """Compute AP per category using pycocotools."""
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {}
    coco_dt   = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()

    cat_ap = {}
    for i, cat_id in enumerate(evaluator.params.catIds):
        # AP for this category across all IoU thresholds, all areas
        ap_vals = evaluator.eval['precision'][:, :, i, 0, 2]
        ap_vals = ap_vals[ap_vals > -1]
        cat_ap[cat_id] = float(np.mean(ap_vals)) * 100 if len(ap_vals) > 0 else 0.0
    return cat_ap


def size_breakdown(coco_gt, results):
    """AP for Small / Medium / Large objects."""
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {'AP_S': 0.0, 'AP_M': 0.0, 'AP_L': 0.0}
    coco_dt   = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    stats = evaluator.stats
    return {
        'AP':   round(float(stats[0]) * 100, 2),
        'AP50': round(float(stats[1]) * 100, 2),
        'AP75': round(float(stats[2]) * 100, 2),
        'AP_S': round(float(stats[3]) * 100, 2),
        'AP_M': round(float(stats[4]) * 100, 2),
        'AP_L': round(float(stats[5]) * 100, 2),
    }


def print_summary_table(all_metrics, variant_names):
    """Print a comparison table ready for the paper."""
    keys = ['AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L']
    col_w = 10

    print(f"\n{'='*65}")
    print("  E3 Detection Summary — COCO 2017 val")
    print(f"{'='*65}")
    header = f"  {'Model':<28}" + "".join(f"{k:>{col_w}}" for k in keys)
    print(header)
    print(f"  {'-'*63}")

    for name, metrics in zip(variant_names, all_metrics):
        short = name.replace('ViT-B_', '').replace('_det', '')
        row   = f"  {short:<28}" + "".join(
            f"{metrics.get(k, 0.0):>{col_w}.2f}" for k in keys)
        print(row)

    # Delta rows (SAGA vs baseline)
    if len(all_metrics) == 3:
        base = all_metrics[0]
        saga = all_metrics[2]
        print(f"  {'-'*63}")
        delta_row = f"  {'SAGA Δ vs baseline':<28}" + "".join(
            f"{(saga.get(k,0)-base.get(k,0)):>+{col_w}.2f}"
            for k in keys)
        print(delta_row)

    print(f"{'='*65}\n")


def plot_category_comparison(cat_ap_dict, cat_names, out_dir, variant_names):
    """Bar chart of per-category AP for baseline vs SAGA."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    # Top 20 categories where SAGA improves most
    if len(variant_names) < 3:
        return

    baseline_ap = cat_ap_dict[variant_names[0]]
    saga_ap     = cat_ap_dict[variant_names[2]]

    # Delta for each category
    deltas = {cat_id: saga_ap.get(cat_id, 0) - baseline_ap.get(cat_id, 0)
              for cat_id in baseline_ap}
    sorted_cats = sorted(deltas.keys(), key=lambda c: deltas[c], reverse=True)
    top20 = sorted_cats[:20]

    fig, ax = plt.subplots(figsize=(14, 5))
    x     = range(len(top20))
    width = 0.35

    b_vals = [baseline_ap.get(c, 0) for c in top20]
    s_vals = [saga_ap.get(c, 0)     for c in top20]
    labels = [cat_names.get(c, str(c)) for c in top20]

    ax.bar([xi - width/2 for xi in x], b_vals, width,
           label='Baseline', color='#6B7E9B', alpha=0.85)
    ax.bar([xi + width/2 for xi in x], s_vals, width,
           label='SAGA',     color='#00A99D', alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AP (%)')
    ax.set_title('Per-category AP — Top 20 SAGA improvements (COCO val2017)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = Path(out_dir) / 'e3_per_category_ap.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    required=True)
    parser.add_argument('--paths',     required=True)
    parser.add_argument('--data_root', required=True,
                        help='Staged COCO root')
    parser.add_argument('--ckpts',     required=True, nargs='+',
                        help='Checkpoints in order: baseline registers saga')
    parser.add_argument('--out_dir',   required=True,
                        help='Output directory for figures and JSON')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_metrics      = []
    all_cat_aps      = {}
    variant_names    = []
    cat_names        = {}

    for variant_id, ckpt_path in enumerate(args.ckpts):
        cfg  = load_config(args.config, variant_id, args.paths)
        name = cfg['variant_name']
        t    = cfg['train']
        variant_names.append(name)

        print(f"\n{'='*60}")
        print(f"  Analyzing: {name}")
        print(f"{'='*60}")

        # Dataset
        ds = COCODetectionDataset(
            img_dir   = Path(args.data_root) / 'val2017',
            ann_file  = Path(args.data_root) / 'annotations' /
                        'instances_val2017.json',
            transforms= get_val_transforms(t['min_size'], t['max_size']),
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1,
            num_workers=args.num_workers,
            collate_fn=collate_fn, shuffle=False)

        # Collect category names once
        if not cat_names:
            cat_names = {v: k for k, v in
                         ds.coco_id_to_idx.items()}
            cat_names = {cid: ds.idx_to_name[idx]
                         for cid, idx in ds.coco_id_to_idx.items()}

        # Build and load model
        detector = build_detector(cfg, cfg['paths']).to(device)
        ckpt     = torch.load(ckpt_path, map_location='cpu')
        state    = {k.replace('module.', ''): v
                    for k, v in ckpt.get('model', ckpt).items()}
        detector.load_state_dict(state, strict=True)

        # Run predictions
        preds = get_predictions(detector, loader, device, name)

        # Overall metrics
        coco_gt = ds.get_coco_api()
        metrics = size_breakdown(coco_gt, preds)
        all_metrics.append(metrics)

        # Per-category AP
        cat_ap = per_category_ap(coco_gt, preds,
                                  list(ds.coco_id_to_idx.keys()))
        all_cat_aps[name] = cat_ap

        print(f"  AP={metrics['AP']:.2f}  AP_S={metrics['AP_S']:.2f}  "
              f"AP_M={metrics['AP_M']:.2f}  AP_L={metrics['AP_L']:.2f}")

        del detector

    # Summary table
    print_summary_table(all_metrics, variant_names)

    # Per-category bar chart
    plot_category_comparison(all_cat_aps, cat_names, args.out_dir, variant_names)

    # Save full analysis JSON
    out = {
        'variants': variant_names,
        'metrics':  dict(zip(variant_names, all_metrics)),
        'per_category_ap': {
            name: {str(k): v for k, v in aps.items()}
            for name, aps in all_cat_aps.items()
        },
    }
    out_file = Path(args.out_dir) / 'e3_analysis.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full analysis saved to: {out_file}")


if __name__ == '__main__':
    main()