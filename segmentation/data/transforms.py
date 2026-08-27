"""
segmentation/data/transforms.py
=================================
Joint image + mask transforms for semantic segmentation.

All transforms operate on (PIL Image, PIL Image) pairs.
The same geometric transform is applied to both image and mask.
Mask uses NEAREST interpolation to preserve integer class labels.

No MixUp/CutMix — segmentation requires valid per-pixel labels.
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() < self.prob:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask


class RandomScale:
    """
    Randomly scale image and mask by a factor in [min_scale, max_scale].
    Minimum dimension is kept >= crop_size so RandomCrop always succeeds.
    """
    def __init__(self, min_scale=0.5, max_scale=2.0, crop_size=512):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_size = crop_size

    def __call__(self, img, mask):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h  = img.size
        new_w = max(int(round(w * scale)), self.crop_size)
        new_h = max(int(round(h * scale)), self.crop_size)
        img   = TF.resize(img,  [new_h, new_w], TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask, [new_h, new_w], TF.InterpolationMode.NEAREST)
        return img, mask


class RandomCrop:
    """Random square crop of size crop_size × crop_size."""
    def __init__(self, crop_size=512):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        w, h = img.size
        assert w >= self.crop_size and h >= self.crop_size, \
            f"Image {w}×{h} smaller than crop {self.crop_size}"
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        img  = TF.crop(img,  y, x, self.crop_size, self.crop_size)
        mask = TF.crop(mask, y, x, self.crop_size, self.crop_size)
        return img, mask


class PadIfSmaller:
    """Pad image/mask to at least (size, size) with 0 / ignore_index."""
    def __init__(self, size=512, ignore_index=255):
        self.size         = size
        self.ignore_index = ignore_index

    def __call__(self, img, mask):
        w, h    = img.size
        pad_w   = max(0, self.size - w)
        pad_h   = max(0, self.size - h)
        if pad_w > 0 or pad_h > 0:
            # padding: left, top, right, bottom
            img  = TF.pad(img,  [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h],
                          fill=self.ignore_index)
        return img, mask


class CenterCrop:
    def __init__(self, crop_size=512):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        img  = TF.center_crop(img,  self.crop_size)
        mask = TF.center_crop(mask, self.crop_size)
        return img, mask


class ResizeFixed:
    """Resize to fixed size (used for validation)."""
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img, mask):
        img  = TF.resize(img,  [self.size, self.size],
                         TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size],
                         TF.InterpolationMode.NEAREST)
        return img, mask


class ToTensorSeg:
    """
    Convert image to float tensor and mask to long tensor.
    Applies ADE20K label convention:
        mask value 0   → ignore_index 255
        mask value k   → k-1  (0-indexed class)
    """
    IGNORE_INDEX = 255

    def __call__(self, img, mask):
        img_t  = TF.to_tensor(img)           # [3, H, W] float32 in [0,1]
        mask_np = np.array(mask, dtype=np.int64)
        mask_t  = torch.from_numpy(mask_np) - 1
        mask_t[mask_t == -1] = self.IGNORE_INDEX
        return img_t, mask_t


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, img, mask):
        img = TF.normalize(img, self.mean, self.std)
        return img, mask


# ── Composed transforms ────────────────────────────────────────────────────────

def get_train_transforms(crop_size=512, min_scale=0.5, max_scale=2.0):
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        RandomScale(min_scale, max_scale, crop_size),
        PadIfSmaller(crop_size),
        RandomCrop(crop_size),
        ToTensorSeg(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(size=512):
    return Compose([
        ResizeFixed(size),
        ToTensorSeg(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])