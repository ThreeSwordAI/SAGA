"""
detection/data/inspect_coco.py
===============================
Inspects COCO 2017 dataset structure after extracting from zip files.
Run this on TinyGPU via salloc before writing E3 training code.

Usage:
    salloc --partition=tgpu --gres=gpu:rtx2080ti:1 --time=01:00:00
    python3 detection/data/inspect_coco.py \
        --zip_dir  /home/woody/iwi5/iwi5359h/Data/COCO \
        --out_dir  /tmp/coco_inspect

Prints everything needed to design E3:
    - Folder structure after extraction
    - Annotation file format (keys, types)
    - Image count (train/val)
    - Category count and names
    - Annotation count
    - Sample annotation (bounding box format)
    - Image size distribution
    - Small/medium/large object breakdown (AP_S/AP_M/AP_L areas)
"""

import argparse
import json
import os
import zipfile
from pathlib import Path
from collections import Counter
import math


def extract_zip(zip_path, out_dir, desc):
    out = Path(out_dir)
    print(f"\n{'='*60}")
    print(f"  Extracting {desc}")
    print(f"  From: {zip_path}")
    print(f"  To:   {out_dir}")
    print(f"{'='*60}")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        print(f"  {len(members):,} files in archive")
        for i, m in enumerate(members):
            z.extract(m, out)
            if i % 5000 == 0 and i > 0:
                print(f"    {i:,}/{len(members):,} extracted")
    print(f"  Done.")


def print_dir_tree(root, max_depth=3, prefix=""):
    root = Path(root)
    if not root.exists():
        print(f"  {root} does not exist")
        return
    items = sorted(root.iterdir())
    for i, item in enumerate(items):
        connector = "└── " if i == len(items)-1 else "├── "
        if item.is_file():
            size_mb = item.stat().st_size / 1e6
            print(f"{prefix}{connector}{item.name}  ({size_mb:.1f} MB)")
        else:
            n_files = len(list(item.iterdir())) if item.is_dir() else 0
            print(f"{prefix}{connector}{item.name}/  [{n_files} items]")
            if max_depth > 1:
                ext = "    " if i == len(items)-1 else "│   "
                print_dir_tree(item, max_depth-1, prefix+ext)


def inspect_annotations(ann_file, split):
    print(f"\n{'='*60}")
    print(f"  Annotation file: {Path(ann_file).name}  [{split}]")
    print(f"{'='*60}")

    with open(ann_file) as f:
        data = json.load(f)

    # Top-level keys
    print(f"\n  Top-level keys: {list(data.keys())}")

    # Info
    if 'info' in data:
        print(f"\n  Info: {data['info']}")

    # Images
    images = data.get('images', [])
    print(f"\n  Images: {len(images):,}")
    if images:
        img = images[0]
        print(f"  Sample image entry keys: {list(img.keys())}")
        print(f"  Sample image entry: {img}")

    # Categories
    cats = data.get('categories', [])
    print(f"\n  Categories: {len(cats)}")
    cat_names = [c['name'] for c in cats]
    print(f"  Category names: {cat_names}")
    print(f"  Category id range: {min(c['id'] for c in cats)} – {max(c['id'] for c in cats)}")

    # Annotations
    anns = data.get('annotations', [])
    print(f"\n  Annotations: {len(anns):,}")
    if anns:
        ann = anns[0]
        print(f"  Sample annotation keys: {list(ann.keys())}")
        print(f"  Sample annotation: {ann}")

    # Bounding box format
    if anns and 'bbox' in anns[0]:
        print(f"\n  BBox format: [x_min, y_min, width, height]  (COCO standard)")
        bboxes = [a['bbox'] for a in anns if a.get('bbox')]
        widths  = [b[2] for b in bboxes]
        heights = [b[3] for b in bboxes]
        areas   = [b[2]*b[3] for b in bboxes]
        print(f"  BBox width  range: {min(widths):.1f} – {max(widths):.1f}  "
              f"mean={sum(widths)/len(widths):.1f}")
        print(f"  BBox height range: {min(heights):.1f} – {max(heights):.1f}  "
              f"mean={sum(heights)/len(heights):.1f}")

        # COCO area thresholds for AP_S / AP_M / AP_L
        n_small  = sum(1 for a in areas if a < 32**2)
        n_medium = sum(1 for a in areas if 32**2 <= a < 96**2)
        n_large  = sum(1 for a in areas if a >= 96**2)
        total    = len(areas)
        print(f"\n  Object size distribution (COCO AP thresholds):")
        print(f"    Small  (area < 32²  = 1024):   {n_small:,}  ({100*n_small/total:.1f}%)")
        print(f"    Medium (1024 ≤ area < 9216):    {n_medium:,}  ({100*n_medium/total:.1f}%)")
        print(f"    Large  (area ≥ 9216):           {n_large:,}  ({100*n_large/total:.1f}%)")

    # Image size distribution
    if images and 'width' in images[0]:
        widths  = [img['width']  for img in images]
        heights = [img['height'] for img in images]
        print(f"\n  Image size distribution:")
        print(f"    Width:  min={min(widths)}  max={max(widths)}  "
              f"mean={sum(widths)/len(widths):.0f}")
        print(f"    Height: min={min(heights)}  max={max(heights)}  "
              f"mean={sum(heights)/len(heights):.0f}")
        # Most common sizes
        size_counter = Counter(f"{w}×{h}" for w,h in zip(widths,heights))
        print(f"    Top 5 sizes: {size_counter.most_common(5)}")

    # Annotations per image
    if anns:
        img_ann_counts = Counter(a['image_id'] for a in anns)
        counts = list(img_ann_counts.values())
        print(f"\n  Annotations per image:")
        print(f"    Min: {min(counts)}  Max: {max(counts)}  "
              f"Mean: {sum(counts)/len(counts):.1f}")
        print(f"    Images with 0 annotations: "
              f"{len(images) - len(img_ann_counts):,}")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', required=True,
                        help='Directory containing COCO zip files')
    parser.add_argument('--out_dir', required=True,
                        help='Directory to extract into (temp, can delete after)')
    parser.add_argument('--skip_extract', action='store_true',
                        help='Skip extraction if already done')
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir)
    out_dir = Path(args.out_dir)

    # ── Step 1: Extract ───────────────────────────────────────────────────────
    if not args.skip_extract:
        # Annotations first (small)
        ann_zip = zip_dir / 'annotations_trainval2017.zip'
        if ann_zip.exists():
            extract_zip(ann_zip, out_dir, 'annotations')
        else:
            print(f"WARNING: {ann_zip} not found")

        # Val images (small — 1GB)
        val_zip = zip_dir / 'val2017.zip'
        if val_zip.exists():
            extract_zip(val_zip, out_dir, 'val2017 images')
        else:
            print(f"WARNING: {val_zip} not found")

        # Train images (large — 18GB, skip for inspection)
        print(f"\n  NOTE: Skipping train2017.zip extraction for inspection.")
        print(f"  (18GB — only needed for actual training)")
    else:
        print("Skipping extraction (--skip_extract)")

    # ── Step 2: Print directory structure ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Directory structure after extraction")
    print(f"{'='*60}")
    print_dir_tree(out_dir)

    # ── Step 3: Inspect annotation files ─────────────────────────────────────
    ann_dir = out_dir / 'annotations'
    if ann_dir.exists():
        ann_files = sorted(ann_dir.glob('*.json'))
        print(f"\n  Annotation files found: {[f.name for f in ann_files]}")

        # Inspect instances (detection) for train and val
        for split in ['val2017', 'train2017']:
            inst_file = ann_dir / f'instances_{split}.json'
            if inst_file.exists():
                inspect_annotations(inst_file, split)
            else:
                print(f"\n  {inst_file.name} not found")
    else:
        print(f"\n  No annotations directory found at {ann_dir}")

    # ── Step 4: Check image files ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Image file check")
    print(f"{'='*60}")
    for split in ['val2017', 'train2017']:
        img_dir = out_dir / split
        if img_dir.exists():
            imgs = list(img_dir.glob('*.jpg'))
            print(f"  {split}: {len(imgs):,} images")
            if imgs:
                print(f"  Sample filenames: {[p.name for p in imgs[:3]]}")
        else:
            print(f"  {split}: not extracted")

    print(f"\n{'='*60}")
    print(f"  Inspection complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()