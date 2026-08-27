#!/usr/bin/env python3
"""
evaluation/e5_lost/data/inspect_voc.py
=======================================
Inspects VOC 2007 dataset structure after extracting from tar files.
Run this on TinyGPU interactive before submitting E5.

Usage:
    salloc --partition=tgpu --gres=gpu:rtx2080ti:1 --time=00:30:00

    python3 evaluation/e5_lost/data/inspect_voc.py \
        --trainval_tar /home/woody/iwi5/iwi5359h/Data/VOC/VOCtrainval_06-Nov-2007.tar \
        --test_tar     /home/woody/iwi5/iwi5359h/Data/VOC/VOCtest_06-Nov-2007.tar \
        --out_dir      /tmp/voc_inspect
"""

import argparse
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter


def extract_tars(trainval_tar, test_tar, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    voc_root = out / 'VOCdevkit'
    if voc_root.exists():
        print(f"Already extracted at {voc_root} — skipping extraction")
        return str(voc_root)

    print(f"\nExtracting trainval tar (~438MB)...")
    with tarfile.open(trainval_tar, 'r') as tf:
        tf.extractall(out)
    print(f"Extracting test tar (~430MB)...")
    with tarfile.open(test_tar, 'r') as tf:
        tf.extractall(out)
    print(f"Done.")
    return str(voc_root)


def inspect_voc(voc_root):
    voc_root = Path(voc_root)
    voc2007  = voc_root / 'VOC2007'

    print(f"\n{'='*60}")
    print(f"  VOC 2007 Structure")
    print(f"{'='*60}")

    # Top-level directories
    print(f"\nVOCdevkit contents:")
    for item in sorted(voc_root.iterdir()):
        print(f"  {item.name}/")

    print(f"\nVOC2007 contents:")
    for item in sorted(voc2007.iterdir()):
        if item.is_dir():
            n = len(list(item.iterdir()))
            print(f"  {item.name}/  [{n} items]")

    # Image sets (splits)
    print(f"\n{'='*60}")
    print(f"  Image Sets (splits)")
    print(f"{'='*60}")
    main_dir = voc2007 / 'ImageSets' / 'Main'
    splits = ['train', 'val', 'trainval', 'test']
    for split in splits:
        split_file = main_dir / f'{split}.txt'
        if split_file.exists():
            with open(split_file) as f:
                ids = [l.strip() for l in f if l.strip()]
            print(f"  {split:<12}: {len(ids):,} images")
        else:
            print(f"  {split:<12}: not found")

    # Images
    print(f"\n{'='*60}")
    print(f"  Images")
    print(f"{'='*60}")
    img_dir = voc2007 / 'JPEGImages'
    imgs    = list(img_dir.glob('*.jpg'))
    print(f"  Total images: {len(imgs):,}")

    # Sample image sizes
    from PIL import Image
    sizes = []
    for img_path in imgs[:50]:
        with Image.open(img_path) as im:
            sizes.append(im.size)
    widths  = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    print(f"  Sample size range (first 50):")
    print(f"    Width:  {min(widths)} – {max(widths)}  mean={sum(widths)//len(widths)}")
    print(f"    Height: {min(heights)} – {max(heights)}  mean={sum(heights)//len(heights)}")

    # Annotations
    print(f"\n{'='*60}")
    print(f"  Annotations")
    print(f"{'='*60}")
    ann_dir = voc2007 / 'Annotations'
    anns    = list(ann_dir.glob('*.xml'))
    print(f"  Total annotation files: {len(anns):,}")

    # Parse a sample annotation
    sample_ann = anns[0]
    tree = ET.parse(sample_ann)
    root = tree.getroot()
    print(f"\n  Sample annotation: {sample_ann.name}")
    print(f"    Top-level keys: {[child.tag for child in root]}")

    size = root.find('size')
    print(f"    Image size: {size.find('width').text} × {size.find('height').text}")

    objects = root.findall('object')
    print(f"    Objects in this image: {len(objects)}")
    for obj in objects[:3]:
        name = obj.find('name').text
        bb   = obj.find('bndbox')
        x1, y1 = bb.find('xmin').text, bb.find('ymin').text
        x2, y2 = bb.find('xmax').text, bb.find('ymax').text
        diff = obj.find('difficult').text if obj.find('difficult') is not None else '0'
        print(f"      {name}: [{x1},{y1},{x2},{y2}]  difficult={diff}")

    # Statistics over test split
    print(f"\n{'='*60}")
    print(f"  Test split statistics (4952 images)")
    print(f"{'='*60}")
    test_file = main_dir / 'test.txt'
    with open(test_file) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    n_objects_per_image = []
    category_counter    = Counter()
    box_areas           = []
    n_difficult         = 0

    for img_id in test_ids:
        ann_file = ann_dir / f'{img_id}.xml'
        if not ann_file.exists():
            continue
        tree = ET.parse(ann_file)
        root = tree.getroot()
        size = root.find('size')
        w    = float(size.find('width').text)
        h    = float(size.find('height').text)

        objs = root.findall('object')
        n_objects_per_image.append(len(objs))

        for obj in objs:
            name = obj.find('name').text
            category_counter[name] += 1
            diff = obj.find('difficult')
            if diff is not None and int(diff.text) == 1:
                n_difficult += 1
            bb   = obj.find('bndbox')
            x1   = float(bb.find('xmin').text)
            y1   = float(bb.find('ymin').text)
            x2   = float(bb.find('xmax').text)
            y2   = float(bb.find('ymax').text)
            area = (x2 - x1) * (y2 - y1) / (w * h)  # normalised area
            box_areas.append(area)

    print(f"  Objects per image: min={min(n_objects_per_image)}  "
          f"max={max(n_objects_per_image)}  "
          f"mean={sum(n_objects_per_image)/len(n_objects_per_image):.1f}")
    print(f"  Total annotations: {sum(n_objects_per_image):,}")
    print(f"  Difficult objects: {n_difficult:,} "
          f"({100*n_difficult/sum(n_objects_per_image):.1f}%)")
    print(f"\n  Normalised box area distribution:")
    import statistics
    print(f"    Min:    {min(box_areas):.4f}")
    print(f"    Max:    {max(box_areas):.4f}")
    print(f"    Mean:   {statistics.mean(box_areas):.4f}")
    print(f"    Median: {statistics.median(box_areas):.4f}")

    print(f"\n  Top 10 categories (test split):")
    for cat, count in category_counter.most_common(10):
        print(f"    {cat:<20}: {count:,}")

    print(f"\n  All {len(category_counter)} categories: {sorted(category_counter.keys())}")

    # Annotation XML structure check
    print(f"\n{'='*60}")
    print(f"  XML annotation structure sample")
    print(f"{'='*60}")
    print(f"  Root tag: {root.tag}")
    print(f"  folder:   {root.findtext('folder')}")
    print(f"  filename: {root.findtext('filename')}")
    print(f"  segmented:{root.findtext('segmented')}")

    print(f"\n{'='*60}")
    print(f"  Inspection complete")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainval_tar', required=True)
    parser.add_argument('--test_tar',     required=True)
    parser.add_argument('--out_dir',      required=True)
    args = parser.parse_args()

    voc_root = extract_tars(args.trainval_tar, args.test_tar, args.out_dir)
    inspect_voc(voc_root)


if __name__ == '__main__':
    main()