"""
segmentation/data/ade20k_dataset.py
=====================================
ADE20K 2016 semantic segmentation dataset.

Extracts from zip at runtime — no pre-extraction needed.
Consistent with E3 COCO staging pattern.

Data format (from inspection):
  images/training/ADE_train_XXXXXXXX.jpg
  annotations/training/ADE_train_XXXXXXXX.png

  Mask values: 0 = unlabeled, 1-150 = semantic classes
  Label convention used here:
    mask value 0   → ignore_index 255 (excluded from loss + mIoU)
    mask value k   → class k-1        (0-indexed, 0-149)

Train: 20,210 images | Val: 2,000 images | Classes: 150
"""

import zipfile
from pathlib import Path
from typing import Tuple, Optional
import io

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def stage_ade20k(zip_path: str, stage_dir: str) -> str:
    """
    Extract ADE20K zip to stage_dir.
    Returns path to ADEChallengeData2016/.
    """
    ade_root = Path(stage_dir) / 'ADEChallengeData2016'
    if ade_root.exists():
        print(f"  ADE20K already staged at {ade_root}")
        return str(ade_root)

    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Extracting ADE20K (~900MB)...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(stage_dir)

    n_train = len(list((ade_root / 'images' / 'training').glob('*.jpg')))
    n_val   = len(list((ade_root / 'images' / 'validation').glob('*.jpg')))
    print(f"  ADE20K staged: train={n_train}, val={n_val}")
    return str(ade_root)


def cleanup_ade20k(stage_dir: str):
    """Remove staged ADE20K data."""
    import shutil
    ade_root = Path(stage_dir) / 'ADEChallengeData2016'
    if ade_root.exists():
        shutil.rmtree(ade_root)
        print(f"  Cleaned up {ade_root}")


class ADE20KDataset(Dataset):
    """
    ADE20K semantic segmentation dataset.

    Args:
        ade_root    Path to ADEChallengeData2016/
        split       'training' or 'validation'
        transforms  Joint image+mask transform (from transforms.py)
    """

    NUM_CLASSES  = 150
    IGNORE_INDEX = 255   # pixels with original value 0 (unlabeled)

    def __init__(self, ade_root: str, split: str = 'training',
                 transforms=None):
        self.ade_root   = Path(ade_root)
        self.split      = split
        self.transforms = transforms

        self.img_dir = self.ade_root / 'images'      / split
        self.ann_dir = self.ade_root / 'annotations' / split

        # Build sorted list of image stems
        imgs = sorted(self.img_dir.glob('*.jpg'))
        self.stems = [p.stem for p in imgs]

        # Verify masks exist
        n_missing = sum(
            1 for s in self.stems
            if not (self.ann_dir / f'{s}.png').exists()
        )
        if n_missing > 0:
            raise RuntimeError(f"Missing {n_missing} masks in {self.ann_dir}")

        print(f"  ADE20K [{split}]: {len(self.stems):,} images, "
              f"{self.NUM_CLASSES} classes")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]

        img  = Image.open(self.img_dir / f'{stem}.jpg').convert('RGB')
        mask = Image.open(self.ann_dir / f'{stem}.png')

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        else:
            # Minimal: convert to tensor without augmentation
            import torchvision.transforms.functional as TF
            img  = TF.to_tensor(img)
            img  = TF.normalize(img, [0.485,0.456,0.406], [0.229,0.224,0.225])
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
            # Convert: 0→255, k→k-1
            mask = mask - 1
            mask[mask == -1] = self.IGNORE_INDEX

        return img, mask