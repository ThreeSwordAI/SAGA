#!/usr/bin/env python3
"""
tools/build_diag_split.py
=========================
Freeze the 10k-image diagnostic split: N val images per class, seeded.

    python tools/build_diag_split.py --data /path/to/imagenet \
        --n-per-class 10 --seed 0 --out results/diagsplit/val_diag_split.json

The JSON stores RELATIVE file paths + class ids (not indices), so the split
is robust to directory-ordering differences between machines:

    {"seed": 0, "n": 10000, "n_per_class": 10,
     "items": [["val/n01440764/ILSVRC2012_val_00003014.JPEG", 0], ...]}

Class ids follow ImageFolder convention: sorted synset directory names.
This script runs on the HPC (an extracted ImageFolder copy of ImageNet with
a val/ subdirectory); the JSON gets committed. The importable
DiagSplitDataset loads the frozen split anywhere.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from torch.utils.data import Dataset

IMG_EXTS = {".jpeg", ".jpg", ".png"}


def build_split(data_root, n_per_class: int = 10, seed: int = 0) -> dict:
    """Deterministically select n_per_class val images per class."""
    val_dir = Path(data_root) / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"{val_dir} not found — --data must be an ImageNet root with a "
            f"val/<synset>/*.JPEG layout (e.g. a staged $STAGE_DIR)")

    classes = sorted(d.name for d in val_dir.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class directories under {val_dir}")

    rng = random.Random(seed)
    items = []
    for class_id, cls in enumerate(classes):
        files = sorted(p.name for p in (val_dir / cls).iterdir()
                       if p.suffix.lower() in IMG_EXTS)
        take = min(n_per_class, len(files))
        if take < n_per_class:
            print(f"WARNING: class {cls} has only {len(files)} images "
                  f"(< {n_per_class})", file=sys.stderr)
        chosen = sorted(rng.sample(files, take))
        items.extend([f"val/{cls}/{name}", class_id] for name in chosen)

    return {"seed": seed, "n": len(items), "n_per_class": n_per_class,
            "items": items}


class DiagSplitDataset(Dataset):
    """Load a frozen diagnostic split produced by build_split().

    Args:
        root:       ImageNet root (the directory containing val/).
        split_json: path to val_diag_split.json.
        transform:  applied to the PIL image (use tools.eval.build_val_transform
                    for paper numbers).
    """

    def __init__(self, root, split_json, transform=None):
        self.root = Path(root)
        with open(split_json) as f:
            split = json.load(f)
        self.items = split["items"]
        self.seed = split.get("seed")
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        rel_path, label = self.items[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def main():
    parser = argparse.ArgumentParser(
        description="Freeze the seeded per-class diagnostic val split.")
    parser.add_argument("--data", required=True, metavar="ROOT",
                        help="ImageNet root containing val/ (ImageFolder layout)")
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/diagsplit/val_diag_split.json")
    args = parser.parse_args()

    split = build_split(args.data, args.n_per_class, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(split, f)
    n_classes = len({label for _, label in split["items"]})
    print(f"wrote {out}: {split['n']} images, {n_classes} classes, "
          f"seed={split['seed']}")


if __name__ == "__main__":
    main()
