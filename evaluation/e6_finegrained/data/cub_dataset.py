"""
evaluation/e6_finegrained/data/cub_dataset.py
===============================================
CUB-200-2011 dataset loader.

Extracts from tar at runtime — no pre-extraction needed.

Structure (from inspection):
  CUB_200_2011/
    images.txt          — image_id  relative_path
    image_class_labels.txt — image_id  class_id (1-200)
    train_test_split.txt   — image_id  is_train (1=train, 0=test)
    images/CLASS_DIR/IMAGE.jpg

Split: train=5994, test=5794, total=11788
Classes: 200 bird species
"""

import tarfile
from pathlib import Path
from typing import Tuple
import io

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def stage_cub(tar_path: str, stage_dir: str) -> str:
    """Extract CUB tar to stage_dir. Returns path to CUB_200_2011/."""
    cub_root = Path(stage_dir) / 'CUB_200_2011'
    if cub_root.exists():
        return str(cub_root)
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Extracting CUB-200-2011...")
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(stage_dir, filter='data')
    print(f"  CUB extracted: {cub_root}")
    return str(cub_root)


class CUBDataset(Dataset):
    """
    CUB-200-2011 for fine-grained classification.

    Args:
        cub_root   Path to CUB_200_2011/ directory
        split      'train' or 'test'
        img_size   Resize to this square size
        augment    Whether to use training augmentation
    """

    NUM_CLASSES = 200

    def __init__(self, cub_root: str, split: str = 'train',
                 img_size: int = 224, augment: bool = True):
        self.cub_root = Path(cub_root)
        self.split    = split
        self.augment  = augment and (split == 'train')

        # Load metadata
        imgs   = {}  # id → relative path
        for line in open(self.cub_root / 'images.txt'):
            img_id, path = line.strip().split()
            imgs[int(img_id)] = path

        labels = {}  # id → class (1-indexed)
        for line in open(self.cub_root / 'image_class_labels.txt'):
            img_id, cls = line.strip().split()
            labels[int(img_id)] = int(cls) - 1  # 0-indexed

        # Filter by split
        is_train = {}
        for line in open(self.cub_root / 'train_test_split.txt'):
            img_id, flag = line.strip().split()
            is_train[int(img_id)] = int(flag) == 1

        want_train = (split == 'train')
        self.samples = [
            (imgs[i], labels[i])
            for i in sorted(imgs)
            if is_train[i] == want_train
        ]

        print(f"  CUB-200 [{split}]: {len(self.samples)} images, 200 classes")

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
        rel_path, label = self.samples[idx]
        img_path = self.cub_root / 'images' / rel_path
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label