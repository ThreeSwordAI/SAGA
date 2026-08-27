#!/usr/bin/env python3
"""
evaluation/e5_lost/tools/run_lost.py
=====================================
LOST algorithm for unsupervised object discovery.

Reference: Simeoni et al. "Localizing Objects with Self-Supervised
Transformers and no Labels" BMVC 2021. arXiv:2109.14279.

Input:  Patch features saved by extract_features.py
Output: Predicted bounding boxes + CorLoc% metric

The LOST algorithm:
  1. For each image, take patch features [N_patches, dim]
  2. Compute pairwise cosine similarity [N_patches, N_patches]
  3. Find the seed: patch with lowest correlation to all others
     (background/sink patches are low-correlation with foreground)
  4. Expand from seed using inverse correlation:
     foreground patches are those correlated with the seed's anti-neighbours
  5. Predict bounding box from the foreground patch set

CorLoc%: fraction of images where predicted box has IoU ≥ 0.5 with GT.

Usage:
    python3 evaluation/e5_lost/tools/run_lost.py \
        --paths    evaluation/e5_lost/configs/paths.yaml \
        --models   baseline registers saga pretrained \
        --split    test
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml


# ── LOST algorithm ─────────────────────────────────────────────────────────────

def cosine_similarity_matrix(feats: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity.
    feats: [N_patches, dim]
    Returns: [N_patches, N_patches]
    """
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    feats_norm = feats / norms
    return feats_norm @ feats_norm.T


def lost_predict_box(
    patch_feats:  np.ndarray,   # [N_patches, dim]
    grid_h:       int = 14,
    grid_w:       int = 14,
    threshold:    float = 0.0,  # correlation threshold for expansion
) -> np.ndarray:
    """
    Run LOST on a single image.

    Returns predicted box as [x1, y1, x2, y2] in patch grid coordinates.
    These are later converted to pixel coordinates.
    """
    N = patch_feats.shape[0]   # 196 for 14×14 grid

    # Step 1: pairwise cosine similarity
    sim = cosine_similarity_matrix(patch_feats)   # [N, N]

    # Step 2: find seed — patch with fewest positively correlated neighbours
    # (sink patches correlate with everything; foreground patches are selective)
    pos_corr_count = (sim > threshold).sum(axis=1)   # [N]
    seed_idx       = int(np.argmin(pos_corr_count))

    # Step 3: expand — find patches positively correlated with
    # the INVERSE neighbourhood of the seed
    # Anti-neighbours of seed: patches NOT correlated with seed
    anti_neighbours = (sim[seed_idx] <= threshold)    # [N] boolean

    # Foreground: patches correlated with the anti-neighbours of the seed
    # i.e. patches that "belong together" in opposition to the seed
    if anti_neighbours.sum() == 0:
        # Fallback: use the patch with highest average similarity
        scores = sim.mean(axis=1)
        seed_idx = int(np.argmax(scores))
        anti_neighbours = (sim[seed_idx] > threshold)

    # Score each patch: how correlated is it with anti-neighbours?
    anti_sim = sim[:, anti_neighbours].mean(axis=1)   # [N]

    # Foreground mask: patches more correlated with anti-neighbours than seed
    foreground_mask = anti_sim > anti_sim[seed_idx]   # [N] boolean

    if foreground_mask.sum() == 0:
        # Fallback: use top 25% most similar to anti-neighbours
        cutoff = np.percentile(anti_sim, 75)
        foreground_mask = anti_sim >= cutoff

    # Step 4: predict bounding box from foreground patches
    fg_indices = np.where(foreground_mask)[0]
    rows = fg_indices // grid_w
    cols = fg_indices %  grid_w

    r1, r2 = int(rows.min()), int(rows.max())
    c1, c2 = int(cols.min()), int(cols.max())

    # Return as [x1, y1, x2, y2] in grid coordinates
    return np.array([c1, r1, c2 + 1, r2 + 1], dtype=float)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    Boxes may be in any consistent coordinate system.
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    feat_file:   str,
    ann_file:    str,
    grid_h:      int = 14,
    grid_w:      int = 14,
    iou_thresh:  float = 0.5,
    threshold:   float = 0.0,
) -> dict:
    """
    Run LOST on all images for one model and compute CorLoc%.

    Returns dict with CorLoc%, per-image predictions, and details.
    """
    print(f"  Loading features from {Path(feat_file).name}...")
    data         = np.load(feat_file)
    patch_feats  = data['patch_feats']   # [N_images, 196, 768]
    img_ids      = data['img_ids']       # [N_images]

    with open(ann_file, 'rb') as f:
        annotations = pickle.load(f)

    N = len(img_ids)
    print(f"  Running LOST on {N} images...")

    correct    = 0
    total      = 0
    results    = []

    for i in range(N):
        if i % 500 == 0:
            print(f"    {i}/{N}", flush=True)

        feats   = patch_feats[i]   # [196, 768]
        ann     = annotations[i]
        img_id  = str(img_ids[i])

        # Get largest GT box
        if not ann['boxes']:
            continue

        boxes = ann['boxes']
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        gt_box = np.array(boxes[areas.index(max(areas))])
        # Normalise GT box to [0,1]
        w, h   = ann.get('orig_w', ann['img_w']), ann.get('orig_h', ann['img_h'])
        gt_norm = gt_box / np.array([w, h, w, h])

        # LOST prediction in grid coords [0, grid_w] × [0, grid_h]
        pred_grid = lost_predict_box(feats, grid_h, grid_w, threshold)

        # Convert predicted box to normalised [0,1] coords
        pred_norm = pred_grid / np.array([grid_w, grid_h, grid_w, grid_h])

        iou = box_iou(pred_norm, gt_norm)

        if iou >= iou_thresh:
            correct += 1
        total += 1

        results.append({
            'img_id':    img_id,
            'iou':       round(float(iou), 4),
            'correct':   iou >= iou_thresh,
            'pred_norm': pred_norm.tolist(),
            'gt_norm':   gt_norm.tolist(),
        })

    corloc = 100.0 * correct / max(total, 1)
    print(f"  CorLoc: {corloc:.2f}%  ({correct}/{total})")

    return {
        'corloc':  round(corloc, 2),
        'correct': correct,
        'total':   total,
        'results': results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths',   required=True)
    parser.add_argument('--models',  nargs='+',
                        default=['baseline', 'registers', 'saga', 'pretrained'])
    parser.add_argument('--split',   default='test')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Cosine similarity threshold for LOST expansion')
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='IoU threshold for CorLoc (standard: 0.5)')
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    import json
    feat_dir   = Path(paths['outputs']['features'])
    res_dir    = Path(paths['outputs']['results'])
    res_dir.mkdir(parents=True, exist_ok=True)

    all_corloc = {}

    for model_key in args.models:
        feat_file = feat_dir / f'{model_key}_{args.split}_features.npz'
        ann_file  = feat_dir / f'{model_key}_{args.split}_annotations.pkl'

        if not feat_file.exists():
            print(f"  WARNING: Features not found for {model_key} — run extract_features.py first")
            continue

        print(f"\n{'='*50}")
        print(f"  LOST evaluation: {model_key}")
        print(f"{'='*50}")

        result = evaluate_model(
            str(feat_file), str(ann_file),
            iou_thresh=args.iou_thresh,
            threshold=args.threshold,
        )

        all_corloc[model_key] = result['corloc']

        # Save per-model detailed results
        out = res_dir / f'{model_key}_{args.split}_lost.json'
        with open(out, 'w') as f:
            json.dump({
                'model':       model_key,
                'split':       args.split,
                'corloc':      result['corloc'],
                'correct':     result['correct'],
                'total':       result['total'],
                'iou_thresh':  args.iou_thresh,
                'threshold':   args.threshold,
                'per_image':   result['results'],
            }, f, indent=2)
        print(f"  Saved: {out}")

    # Print summary table
    print(f"\n{'='*55}")
    print(f"  E5 LOST Results — VOC 2007 {args.split}")
    print(f"  IoU threshold: {args.iou_thresh}")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'CorLoc%':>10}")
    print(f"  {'-'*35}")
    for model_key, corloc in all_corloc.items():
        print(f"  {model_key:<25} {corloc:>10.2f}%")
    print(f"{'='*55}\n")

    # Save summary
    summary_file = res_dir / f'e5_summary_{args.split}.json'
    with open(summary_file, 'w') as f:
        json.dump(all_corloc, f, indent=2)
    print(f"Summary saved: {summary_file}")


if __name__ == '__main__':
    main()