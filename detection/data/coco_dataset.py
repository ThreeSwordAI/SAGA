"""
detection/data/coco_dataset.py
===============================
COCO 2017 dataset for object detection.

Reads from a staged directory produced by stage_coco.sh.
Returns images and targets in torchvision Faster R-CNN format.

COCO bbox format: [x_min, y_min, width, height]
torchvision format: [x_min, y_min, x_max, y_max]
Conversion is handled here automatically.

Category IDs: COCO uses 1-90 (non-contiguous, 80 classes).
We map to contiguous 1-80 (0 reserved for background in Faster R-CNN).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ── COCO category ID → contiguous index mapping ────────────────────────────
# COCO has 80 categories with non-contiguous IDs (1-90, some missing).
# Faster R-CNN expects 0=background, 1-80=categories.
def build_coco_label_map(categories: List[dict]) -> Tuple[Dict, Dict]:
    """
    Returns:
        coco_id_to_idx  {coco_id: contiguous_idx}  where idx is 1-80
        idx_to_name     {contiguous_idx: category_name}
    """
    sorted_cats = sorted(categories, key=lambda c: c['id'])
    coco_id_to_idx = {}
    idx_to_name    = {}
    for i, cat in enumerate(sorted_cats, start=1):
        coco_id_to_idx[cat['id']] = i
        idx_to_name[i]            = cat['name']
    return coco_id_to_idx, idx_to_name


class COCODetectionDataset(Dataset):
    """
    COCO 2017 detection dataset.

    Args:
        img_dir       Path to image directory (train2017/ or val2017/)
        ann_file      Path to instances JSON (instances_train2017.json)
        transforms    Optional torchvision transform applied to image
        min_area      Skip annotations with area below this threshold
        include_empty Keep images with zero valid annotations. False (the
                      committed default) is what training uses; the TASK-09
                      dense val loader passes True so COCO AP is computed
                      over the STANDARD 5000-image val2017 split rather than
                      the 4952 annotated ones.
    """

    def __init__(
        self,
        img_dir:    str,
        ann_file:   str,
        transforms  = None,
        min_area:   float = 1.0,
        include_empty: bool = False,
    ):
        self.img_dir    = Path(img_dir)
        self.ann_file   = Path(ann_file)
        self.transforms = transforms
        self.min_area   = min_area
        self.include_empty = include_empty
        self._coco_gt   = None

        print(f"  Loading COCO annotations from {Path(ann_file).name}...")
        with open(ann_file) as f:
            data = json.load(f)

        # Build category mapping
        self.coco_id_to_idx, self.idx_to_name = build_coco_label_map(
            data['categories'])
        # inverse map: model label (1-80) -> official COCO category id (1-90).
        # Predictions MUST be translated back through this before COCO
        # scoring — the official GT carries the original ids.
        self.idx_to_coco_id = {v: k for k, v in self.coco_id_to_idx.items()}
        self.num_classes = len(self.coco_id_to_idx)  # 80

        # Build image id → image info mapping
        self.img_info = {img['id']: img for img in data['images']}

        # Group annotations by image_id, filter crowds and tiny objects
        self.img_to_anns: Dict[int, List] = {img_id: []
                                              for img_id in self.img_info}
        n_skipped = 0
        for ann in data['annotations']:
            if ann.get('iscrowd', 0):
                n_skipped += 1
                continue
            if ann.get('area', 0) < self.min_area:
                n_skipped += 1
                continue
            img_id = ann['image_id']
            if img_id in self.img_to_anns:
                self.img_to_anns[img_id].append(ann)

        # Keep only images that have at least one valid annotation
        # (include_empty=True keeps all of them — standard COCO val protocol)
        self.img_ids = [
            img_id for img_id in self.img_info
            if self.include_empty or len(self.img_to_anns[img_id]) > 0
        ]

        print(f"  Images kept: {len(self.img_ids):,} "
              f"(include_empty={self.include_empty})")
        print(f"  Annotations skipped (crowd/tiny): {n_skipped:,}")
        print(f"  Categories: {self.num_classes}")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id   = self.img_ids[idx]
        img_info = self.img_info[img_id]
        anns     = self.img_to_anns[img_id]

        # Load image
        img_path = self.img_dir / img_info['file_name']
        img      = Image.open(img_path).convert('RGB')

        # Parse annotations
        boxes  = []
        labels = []
        areas  = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # Skip degenerate boxes
            if w < 1 or h < 1:
                continue
            # Convert COCO [x,y,w,h] → [x1,y1,x2,y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.coco_id_to_idx[ann['category_id']])
            areas.append(ann['area'])

        target = {
            # reshape(-1, 4): an image with no valid annotation (possible with
            # include_empty=True) would otherwise get a (0,)-shaped boxes
            # tensor, which breaks any boxes[:, k] indexing downstream
            'boxes':    torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
            'labels':   torch.tensor(labels, dtype=torch.int64).reshape(-1),
            'areas':    torch.tensor(areas,  dtype=torch.float32).reshape(-1),
            'image_id': torch.tensor([img_id]),
            'iscrowd':  torch.zeros(len(boxes), dtype=torch.int64),
            # TRUE original image size (h, w), before this pipeline's resize.
            # Predictions come out in the RESIZED frame; COCO ground truth
            # lives in this one, so the evaluator has to map back through it.
            # torchvision ignores target keys it does not know.
            'orig_size': torch.tensor([int(img_info['height']),
                                       int(img_info['width'])],
                                      dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def official_coco_gt(self):
        """The OFFICIAL annotation file as a pycocotools COCO object (cached).

        This is the ground truth the TASK-09 dense trainer scores against:
        original category ids (1-90), all images, crowd annotations intact
        (COCOeval ignores them itself). Predictions must be mapped back to
        official ids with `idx_to_coco_id` first.
        """
        if self._coco_gt is None:
            from pycocotools.coco import COCO
            self._coco_gt = COCO(str(self.ann_file))
        return self._coco_gt

    def get_coco_api(self):
        """Legacy helper (used by detection/tools/{evaluate,analyze}.py).

        FIXED in TASK-09: the annotation `category_id`s are now remapped to
        the SAME contiguous 1-80 label space the model predicts in. Before
        this fix the declared categories were contiguous while the copied
        annotations kept their original 1-90 ids, so COCOeval silently
        compared predictions of class k against ground truth of a different
        class (and dropped every annotation whose original id was > 80) —
        any AP produced through this path was meaningless.
        Prefer `official_coco_gt()` + `idx_to_coco_id` for new code.
        """
        from pycocotools.coco import COCO
        import copy
        import json
        import os
        import tempfile
        remapped = []
        for anns in self.img_to_anns.values():
            for ann in anns:
                a = copy.deepcopy(ann)
                a['category_id'] = self.coco_id_to_idx[ann['category_id']]
                remapped.append(a)
        ann_data = {
            'images':      list(self.img_info.values()),
            'categories':  [{'id': v, 'name': n}
                            for v, n in self.idx_to_name.items()],
            'annotations': remapped,
        }
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                          delete=False)
        try:
            json.dump(ann_data, tmp)
            tmp.close()
            return COCO(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)   # COCO() has already parsed it
            except OSError:
                pass


def collate_fn(batch):
    """Custom collate for variable-size detection targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)