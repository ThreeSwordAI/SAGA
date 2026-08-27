#!/usr/bin/env python3
"""
segmentation/data/inspect_ade20k.py
=====================================
Inspects ADE20K dataset structure from the zip file.
Run on TinyGPU interactive before writing E4 code.

Usage:
    python3 segmentation/data/inspect_ade20k.py \
        --zip_path /home/woody/iwi5/iwi5359h/Data/ADE20K/ADEChallengeData2016.zip \
        --out_dir  /tmp/ade20k_inspect
"""

import argparse
import zipfile
from pathlib import Path
from collections import Counter
import os


def peek_zip(zip_path, n_lines=50):
    """List first N entries of the zip file without extracting."""
    print(f"\n  First {n_lines} entries in {Path(zip_path).name}:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        print(f"  Total entries: {len(names):,}")
        for name in names[:n_lines]:
            info = zf.getinfo(name)
            size = info.file_size // 1024
            tag  = 'DIR ' if name.endswith('/') else f'{size:6d}KB'
            print(f"    [{tag}]  {name}")
    return names


def extract_zip(zip_path, out_dir):
    """Extract zip to out_dir."""
    out = Path(out_dir)
    if (out / 'ADEChallengeData2016').exists():
        print(f"  Already extracted at {out}")
        return str(out / 'ADEChallengeData2016')
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n  Extracting ADE20K (~900MB)...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out)
    print(f"  Done.")
    return str(out / 'ADEChallengeData2016')


def print_tree(root, max_depth=3, prefix="", _depth=0):
    root = Path(root)
    if not root.is_dir() or _depth > max_depth:
        return
    items = sorted(root.iterdir())[:25]
    for i, item in enumerate(items):
        conn = "└── " if i == len(items)-1 else "├── "
        if item.is_file():
            print(f"{prefix}{conn}{item.name}  ({item.stat().st_size//1024}KB)")
        else:
            n = len(list(item.iterdir())) if item.is_dir() else 0
            print(f"{prefix}{conn}{item.name}/  [{n} items]")
            ext = "    " if i == len(items)-1 else "│   "
            print_tree(item, max_depth, prefix+ext, _depth+1)


def inspect_ade20k(ade_root):
    ade = Path(ade_root)

    print(f"\n{'='*60}")
    print(f"  ADE20K Structure")
    print(f"{'='*60}")
    print_tree(ade, max_depth=2)

    # Count images and masks per split
    print(f"\n{'='*60}")
    print(f"  Image / Mask counts")
    print(f"{'='*60}")
    for split in ['training', 'validation']:
        img_dir = ade / 'images' / split
        ann_dir = ade / 'annotations' / split

        if img_dir.exists():
            imgs = list(img_dir.glob('*.jpg'))
            print(f"  images/{split}:      {len(imgs):,} images")

        if ann_dir.exists():
            masks = list(ann_dir.glob('*.png'))
            print(f"  annotations/{split}: {len(masks):,} masks")

    # Check filename correspondence
    print(f"\n{'='*60}")
    print(f"  Filename format check")
    print(f"{'='*60}")
    img_dir  = ade / 'images' / 'training'
    ann_dir  = ade / 'annotations' / 'training'

    if img_dir.exists():
        sample_imgs = sorted(img_dir.glob('*.jpg'))[:5]
        print(f"\n  Sample image filenames:")
        for p in sample_imgs:
            print(f"    {p.name}")

        print(f"\n  Corresponding mask filenames:")
        for p in sample_imgs:
            mask_name = p.stem + '.png'
            mask_path = ann_dir / mask_name
            exists = '✓' if mask_path.exists() else '✗ MISSING'
            print(f"    {mask_name}  {exists}")

    # Inspect mask format
    print(f"\n{'='*60}")
    print(f"  Mask format inspection")
    print(f"{'='*60}")
    if ann_dir.exists():
        sample_masks = sorted(ann_dir.glob('*.png'))[:3]
        try:
            from PIL import Image
            import numpy as np
            for mask_path in sample_masks:
                mask = np.array(Image.open(mask_path))
                unique_vals = sorted(set(mask.flatten().tolist()))
                print(f"\n  {mask_path.name}:")
                print(f"    Shape:        {mask.shape}")
                print(f"    dtype:        {mask.dtype}")
                print(f"    Unique vals:  {unique_vals[:20]}{'...' if len(unique_vals)>20 else ''}")
                print(f"    Min/Max:      {mask.min()} / {mask.max()}")
                print(f"    0 = background/unlabeled?  1-150 = classes")
        except ImportError:
            print("  (PIL not available for mask inspection)")

    # Image sizes
    print(f"\n{'='*60}")
    print(f"  Image size distribution (first 100)")
    print(f"{'='*60}")
    if img_dir.exists():
        try:
            from PIL import Image
            sizes = []
            for p in sorted(img_dir.glob('*.jpg'))[:100]:
                with Image.open(p) as im:
                    sizes.append(im.size)
            widths  = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            print(f"  Width:  min={min(widths)}  max={max(widths)}  "
                  f"mean={sum(widths)//len(widths)}")
            print(f"  Height: min={min(heights)}  max={max(heights)}  "
                  f"mean={sum(heights)//len(heights)}")
            size_counter = Counter(f"{w}×{h}" for w,h in sizes)
            print(f"  Top 5 sizes: {size_counter.most_common(5)}")
        except ImportError:
            print("  (PIL not available for image size inspection)")

    # objectInfo150.txt — class names
    print(f"\n{'='*60}")
    print(f"  Class information")
    print(f"{'='*60}")
    obj_file = ade / 'objectInfo150.txt'
    if obj_file.exists():
        lines = open(obj_file).readlines()
        print(f"  objectInfo150.txt: {len(lines)} lines")
        print(f"  Header: {lines[0].rstrip()}")
        print(f"  First 10 classes:")
        for l in lines[1:11]:
            print(f"    {l.rstrip()}")
        print(f"  Last 5 classes:")
        for l in lines[-5:]:
            print(f"    {l.rstrip()}")
    else:
        print(f"  objectInfo150.txt not found — checking for other class files:")
        for f in ade.rglob('*.txt'):
            print(f"    {f.relative_to(ade)}")

    # sceneCategories.txt
    scene_file = ade / 'sceneCategories.txt'
    if scene_file.exists():
        lines = open(scene_file).readlines()
        print(f"\n  sceneCategories.txt: {len(lines)} lines")
        print(f"  Sample: {lines[:3]}")

    print(f"\n{'='*60}")
    print(f"  Inspection complete")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_path', required=True)
    parser.add_argument('--out_dir',  default='/tmp/ade20k_inspect')
    args = parser.parse_args()

    # Peek at zip contents first
    all_names = peek_zip(args.zip_path, n_lines=50)

    # Summarise zip structure
    print(f"\n  Top-level directories in zip:")
    top_dirs = set()
    for name in all_names:
        parts = Path(name).parts
        if parts:
            top_dirs.add(parts[0])
    for d in sorted(top_dirs):
        count = sum(1 for n in all_names if n.startswith(d))
        print(f"    {d}/  ({count} entries)")

    # Extract and inspect
    ade_root = extract_zip(args.zip_path, args.out_dir)
    inspect_ade20k(ade_root)


if __name__ == '__main__':
    main()