#!/usr/bin/env python3
"""
evaluation/e6_finegrained/data/inspect_datasets.py
====================================================
Inspects CUB-200-2011 and FGVC-Aircraft dataset structure
by peeking inside the tar files without full extraction.

Run this on TinyGPU interactive before writing E6 code.

Usage:
    python3 evaluation/e6_finegrained/data/inspect_datasets.py \
        --cub_tar      /home/woody/iwi5/iwi5359h/Data/CUB-200/CUB_200_2011.tgz \
        --aircraft_tar /home/woody/iwi5/iwi5359h/Data/FGVC-Aircraft/fgvc-aircraft-2013b.tar.gz \
        --out_dir      /tmp/e6_inspect
"""

import argparse
import tarfile
from pathlib import Path
from collections import Counter, defaultdict


# ── helpers ────────────────────────────────────────────────────────────────────

def peek_tar(tar_path, n_lines=60):
    """List first N entries of a tar file without extracting."""
    print(f"\n  First {n_lines} entries in {Path(tar_path).name}:")
    with tarfile.open(tar_path, 'r:gz') as tf:
        members = tf.getmembers()
        print(f"  Total entries: {len(members):,}")
        for m in members[:n_lines]:
            tag = 'DIR ' if m.isdir() else f'{m.size//1024:6d}KB'
            print(f"    [{tag}]  {m.name}")
    return members


def extract_and_inspect(tar_path, out_dir, dataset_name):
    """Extract to temp dir and inspect structure."""
    out = Path(out_dir) / dataset_name
    if not out.exists():
        print(f"\n  Extracting {dataset_name} to {out}...")
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, 'r:gz') as tf:
            tf.extractall(out, filter='data')
        print(f"  Done.")
    else:
        print(f"\n  Already extracted at {out}")
    return out


def print_tree(root, max_depth=3, prefix="", _depth=0):
    """Print directory tree."""
    if _depth > max_depth:
        return
    root = Path(root)
    items = sorted(root.iterdir()) if root.is_dir() else []
    for i, item in enumerate(items[:20]):  # limit breadth
        conn = "└── " if i == len(items)-1 else "├── "
        if item.is_file():
            print(f"{prefix}{conn}{item.name}  ({item.stat().st_size//1024}KB)")
        else:
            n = len(list(item.iterdir())) if item.is_dir() else 0
            print(f"{prefix}{conn}{item.name}/  [{n} items]")
            ext = "    " if i == len(items)-1 else "│   "
            print_tree(item, max_depth, prefix+ext, _depth+1)
    if len(items) > 20:
        print(f"{prefix}    ... ({len(items)-20} more)")


# ── CUB-200-2011 inspection ────────────────────────────────────────────────────

def inspect_cub(cub_root):
    print(f"\n{'='*60}")
    print(f"  CUB-200-2011 Structure")
    print(f"{'='*60}")

    # Find the actual CUB root (may be nested)
    candidates = list(Path(cub_root).rglob('images.txt'))
    if not candidates:
        print("  WARNING: images.txt not found — showing raw tree")
        print_tree(cub_root, max_depth=2)
        return

    cub = candidates[0].parent
    print(f"  CUB root: {cub}")
    print_tree(cub, max_depth=2)

    # images.txt
    img_file = cub / 'images.txt'
    if img_file.exists():
        lines = open(img_file).readlines()
        print(f"\n  images.txt: {len(lines):,} images")
        print(f"  Sample lines:")
        for l in lines[:5]:
            print(f"    {l.rstrip()}")

    # train_test_split.txt
    split_file = cub / 'train_test_split.txt'
    if split_file.exists():
        splits = [l.split()[1] for l in open(split_file)]
        n_train = splits.count('1')
        n_test  = splits.count('0')
        print(f"\n  train_test_split.txt:")
        print(f"    train: {n_train}  test: {n_test}  total: {len(splits)}")

    # classes.txt
    cls_file = cub / 'classes.txt'
    if cls_file.exists():
        classes = open(cls_file).readlines()
        print(f"\n  classes.txt: {len(classes)} classes")
        print(f"  First 5: {[c.split()[1] for c in classes[:5]]}")
        print(f"  Last 5:  {[c.split()[1] for c in classes[-5:]]}")

    # image_class_labels.txt
    lbl_file = cub / 'image_class_labels.txt'
    if lbl_file.exists():
        labels = [int(l.split()[1]) for l in open(lbl_file)]
        print(f"\n  image_class_labels.txt:")
        print(f"    Label range: {min(labels)} – {max(labels)}")
        print(f"    Unique classes: {len(set(labels))}")

    # images folder
    img_dir = cub / 'images'
    if img_dir.exists():
        class_dirs = sorted(img_dir.iterdir())
        print(f"\n  images/ folder:")
        print(f"    Class dirs: {len(class_dirs)}")
        if class_dirs:
            sample = class_dirs[0]
            imgs   = list(sample.glob('*.jpg'))
            print(f"    Sample class: {sample.name}  ({len(imgs)} images)")
            print(f"    Sample filenames: {[p.name for p in imgs[:3]]}")

    # bounding_boxes.txt
    bb_file = cub / 'bounding_boxes.txt'
    if bb_file.exists():
        bbs = open(bb_file).readlines()
        print(f"\n  bounding_boxes.txt: {len(bbs)} entries")
        print(f"  Sample: {bbs[0].rstrip()}")
        print(f"  Format: image_id  x  y  width  height")

    print(f"\n  File formats in images/:")
    all_exts = Counter()
    if img_dir.exists():
        for f in img_dir.rglob('*'):
            if f.is_file():
                all_exts[f.suffix.lower()] += 1
    print(f"    {dict(all_exts)}")


# ── FGVC-Aircraft inspection ───────────────────────────────────────────────────

def inspect_aircraft(aircraft_root):
    print(f"\n{'='*60}")
    print(f"  FGVC-Aircraft Structure")
    print(f"{'='*60}")

    # Find actual root
    candidates = list(Path(aircraft_root).rglob('data'))
    if candidates:
        ac = candidates[0].parent
    else:
        ac = Path(aircraft_root)

    print(f"  Aircraft root: {ac}")
    print_tree(ac, max_depth=2)

    # Data folder
    data_dir = ac / 'data' if (ac / 'data').exists() else ac

    # images folder
    for img_dir_name in ['images', 'data/images']:
        img_dir = ac / img_dir_name
        if img_dir.exists():
            imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            print(f"\n  images dir: {img_dir}")
            print(f"    Total images: {len(imgs)}")
            if imgs:
                print(f"    Sample: {[p.name for p in imgs[:5]]}")
            break

    # Split files
    print(f"\n  Split files found:")
    for f in sorted(ac.rglob('*.txt')):
        lines = open(f).readlines()
        print(f"    {f.name}: {len(lines)} lines  |  sample: {lines[0].rstrip() if lines else ''}")

    # Count per split
    for split in ['train', 'val', 'test', 'trainval']:
        for variant in ['images_variant', 'images']:
            split_file = None
            for candidate in ac.rglob(f'{variant}_{split}.txt'):
                split_file = candidate
                break
            if split_file and split_file.exists():
                lines = [l.strip() for l in open(split_file) if l.strip()]
                print(f"\n  {split_file.name}: {len(lines)} entries")
                print(f"    Sample lines: {lines[:3]}")
                break

    # Classes (variants)
    for cls_file_name in ['variants.txt', 'families.txt', 'manufacturers.txt']:
        cls_file = None
        for candidate in ac.rglob(cls_file_name):
            cls_file = candidate
            break
        if cls_file and cls_file.exists():
            classes = [l.strip() for l in open(cls_file) if l.strip()]
            print(f"\n  {cls_file_name}: {len(classes)} classes")
            print(f"    First 5: {classes[:5]}")
            print(f"    Last 5:  {classes[-5:]}")

    # Check image format
    print(f"\n  Image format check:")
    for img_dir in ac.rglob('*images*'):
        if img_dir.is_dir():
            exts = Counter(f.suffix.lower() for f in img_dir.iterdir()
                           if f.is_file())
            if exts:
                print(f"    {img_dir.name}/: {dict(exts)}")
            break


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cub_tar',      required=True)
    parser.add_argument('--aircraft_tar', required=True)
    parser.add_argument('--out_dir',      default='/tmp/e6_inspect')
    args = parser.parse_args()

    print("====================================================")
    print("  E6 Dataset Inspection")
    print("====================================================")

    # Peek at tar contents first (no extraction)
    print("\n── TAR CONTENTS (no extraction) ──────────────────")
    peek_tar(args.cub_tar,      n_lines=40)
    peek_tar(args.aircraft_tar, n_lines=40)

    # Extract and inspect
    print("\n\n── FULL INSPECTION (extracted) ───────────────────")
    cub_root      = extract_and_inspect(args.cub_tar,      args.out_dir, 'cub')
    aircraft_root = extract_and_inspect(args.aircraft_tar, args.out_dir, 'aircraft')

    inspect_cub(cub_root)
    inspect_aircraft(aircraft_root)

    print(f"\n{'='*60}")
    print(f"  Inspection complete")
    print(f"  Extracted to: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()