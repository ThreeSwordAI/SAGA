"""
evaluation/e5_lost/data/voc_dataset.py
=======================================
VOC 2007 dataset for LOST evaluation.

Reads directly from tar files — no pre-extraction needed.
Extracts to a staging directory at runtime.

VOC 2007 test split: 4,952 images with bounding box annotations.
Used for Correct Localisation (CorLoc%) evaluation following
Simeoni et al. 2021 (LOST) and Darcet et al. 2024 (registers).
"""

import os
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ── Staging helpers ────────────────────────────────────────────────────────────

def stage_voc(trainval_tar: str, test_tar: str, stage_dir: str) -> str:
    """
    Extract VOC 2007 tars to stage_dir if not already there.

    Returns the VOCdevkit path, e.g. stage_dir/VOCdevkit.
    """
    voc_root = Path(stage_dir) / 'VOCdevkit'

    if voc_root.exists():
        print(f"  VOC already staged at {voc_root}")
        return str(voc_root)

    Path(stage_dir).mkdir(parents=True, exist_ok=True)

    print(f"  Extracting VOC trainval (~438MB)...")
    with tarfile.open(trainval_tar, 'r') as tf:
        tf.extractall(stage_dir, filter='data')
    print(f"  Extracting VOC test (~430MB)...")
    with tarfile.open(test_tar, 'r') as tf:
        tf.extractall(stage_dir, filter='data')

    # Verify
    img_dir = voc_root / 'VOC2007' / 'JPEGImages'
    n_imgs  = len(list(img_dir.glob('*.jpg')))
    print(f"  Staged {n_imgs} images to {voc_root}")

    return str(voc_root)


def cleanup_voc(stage_dir: str):
    """Remove staged VOC data."""
    import shutil
    voc_root = Path(stage_dir) / 'VOCdevkit'
    if voc_root.exists():
        shutil.rmtree(voc_root)
        print(f"  Cleaned up {voc_root}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class VOC2007Dataset(Dataset):
    """
    VOC 2007 dataset for LOST evaluation.

    Loads images from the test split (4,952 images).
    Returns image tensor and ground truth bounding boxes.

    Ground truth boxes: list of [x_min, y_min, x_max, y_max] per image.
    For CorLoc evaluation, only the largest object box per image is used.
    """

    # Standard ImageNet normalisation — same as E2 training
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        voc_root:  str,
        split:     str = 'test',    # 'test' = 4952 images (standard for LOST)
        img_size:  int = 224,
    ):
        self.img_dir  = Path(voc_root) / 'VOC2007' / 'JPEGImages'
        self.ann_dir  = Path(voc_root) / 'VOC2007' / 'Annotations'
        self.img_size = img_size

        # Load image list for the requested split
        split_file = (Path(voc_root) / 'VOC2007' / 'ImageSets' /
                      'Main' / f'{split}.txt')
        with open(split_file) as f:
            self.img_ids = [line.strip() for line in f if line.strip()]

        print(f"  VOC 2007 [{split}]: {len(self.img_ids)} images")

        # Pre-load all annotations (fast — XML files are small)
        self.annotations = {}
        for img_id in self.img_ids:
            self.annotations[img_id] = self._parse_annotation(img_id)

        # Transform: resize to img_size, normalise
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def _parse_annotation(self, img_id: str) -> Dict:
        """Parse VOC XML annotation for one image."""
        ann_file = self.ann_dir / f'{img_id}.xml'
        tree     = ET.parse(ann_file)
        root     = tree.getroot()

        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        boxes  = []
        labels = []
        for obj in root.findall('object'):
            # Skip difficult objects (standard VOC practice)
            if obj.find('difficult') is not None:
                if int(obj.find('difficult').text) == 1:
                    continue
            name = obj.find('name').text
            bb   = obj.find('bndbox')
            x1   = float(bb.find('xmin').text)
            y1   = float(bb.find('ymin').text)
            x2   = float(bb.find('xmax').text)
            y2   = float(bb.find('ymax').text)
            boxes.append([x1, y1, x2, y2])
            labels.append(name)

        return {
            'img_w':  img_w,
            'img_h':  img_h,
            'boxes':  boxes,   # [[x1,y1,x2,y2], ...]
            'labels': labels,
        }

    def get_largest_box(self, img_id: str) -> Optional[List[float]]:
        """
        Return the largest bounding box for an image.
        LOST evaluation convention: use the largest object as ground truth.
        Returns [x1, y1, x2, y2] normalised to [0, 1].
        """
        ann = self.annotations[img_id]
        if not ann['boxes']:
            return None
        boxes = ann['boxes']
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        largest = boxes[areas.index(max(areas))]
        # Normalise to [0,1]
        w, h = ann['img_w'], ann['img_h']
        return [largest[0]/w, largest[1]/h, largest[2]/w, largest[3]/h]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict]:
        img_id = self.img_ids[idx]
        img    = Image.open(self.img_dir / f'{img_id}.jpg').convert('RGB')
        orig_w, orig_h = img.size
        tensor = self.transform(img)
        ann    = self.annotations[img_id]
        # Add original size to annotation for box scaling
        ann    = {**ann, 'orig_w': orig_w, 'orig_h': orig_h}
        return tensor, img_id, ann