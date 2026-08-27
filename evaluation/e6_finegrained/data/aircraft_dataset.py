"""
evaluation/e6_finegrained/data/aircraft_dataset.py
====================================================
FGVC-Aircraft 2013b dataset loader.

Extracts from tar at runtime — no pre-extraction needed.

Structure (from inspection):
  fgvc-aircraft-2013b/
    data/
      images/IMAGE_ID.jpg              — flat image folder
      images_variant_trainval.txt      — "image_id variant_name"
      images_variant_test.txt          — "image_id variant_name"
      variants.txt                     — 100 variant class names

Split used: trainval (6667) for training, test (3333) for evaluation.
Standard practice for FGVC-Aircraft in the literature.
"""

import tarfile
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def stage_aircraft(tar_path: str, stage_dir: str) -> str:
    """Extract Aircraft tar to stage_dir. Returns path to fgvc-aircraft-2013b/."""
    ac_root = Path(stage_dir) / 'fgvc-aircraft-2013b'
    if ac_root.exists():
        return str(ac_root)
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Extracting FGVC-Aircraft...")
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(stage_dir, filter='data')
    print(f"  Aircraft extracted: {ac_root}")
    return str(ac_root)


class AircraftDataset(Dataset):
    """
    FGVC-Aircraft 2013b — variant-level classification (100 classes).

    Args:
        ac_root    Path to fgvc-aircraft-2013b/ directory
        split      'trainval' (6667 images) or 'test' (3333 images)
        img_size   Resize to this square size
        augment    Training augmentation
    """

    NUM_CLASSES = 100

    def __init__(self, ac_root: str, split: str = 'trainval',
                 img_size: int = 224, augment: bool = True):
        self.ac_root = Path(ac_root)
        self.img_dir = self.ac_root / 'data' / 'images'
        self.augment = augment and (split == 'trainval')

        # Build variant → index mapping from variants.txt
        variants_file = self.ac_root / 'data' / 'variants.txt'
        variants = [l.strip() for l in open(variants_file) if l.strip()]
        self.class_to_idx = {v: i for i, v in enumerate(variants)}
        self.idx_to_class = {i: v for v, i in self.class_to_idx.items()}

        # Load split file
        split_file = self.ac_root / 'data' / f'images_variant_{split}.txt'
        self.samples = []
        for line in open(split_file):
            line = line.strip()
            if not line:
                continue
            # Format: "image_id variant_name"  (variant may contain spaces/hyphens)
            parts    = line.split(' ', 1)
            img_id   = parts[0]
            variant  = parts[1]
            label    = self.class_to_idx[variant]
            self.samples.append((img_id, label))

        print(f"  FGVC-Aircraft [{split}]: {len(self.samples)} images, "
              f"{len(variants)} classes")

        # Transforms
        if self.augment:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(img_size * 256 / 224)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_id, label = self.samples[idx]
        img_path = self.img_dir / f'{img_id}.jpg'
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label