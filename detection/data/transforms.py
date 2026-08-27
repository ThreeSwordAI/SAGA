"""
detection/data/transforms.py
=============================
Detection-specific augmentation.

All transforms operate on (PIL Image, target_dict) pairs.
Bounding boxes are updated consistently with the image transform.

No MixUp or CutMix — detection requires valid bounding box annotations.
"""

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Tuple, Dict


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return TF.to_tensor(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, img, target):
        return TF.normalize(img, self.mean, self.std), target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            img = TF.hflip(img)
            w   = img.size[0] if hasattr(img, 'size') else img.shape[-1]
            if target['boxes'].numel() > 0:
                boxes          = target['boxes'].clone()
                boxes[:, 0]   = w - target['boxes'][:, 2]
                boxes[:, 2]   = w - target['boxes'][:, 0]
                target['boxes'] = boxes
        return img, target


class ResizeDetection:
    """
    Resize image so that the shorter side is min_size,
    but the longer side does not exceed max_size.
    Scales bounding boxes accordingly.
    Standard for COCO detection (800 × 1333).
    """
    def __init__(self, min_size=800, max_size=1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        import PIL
        if isinstance(img, PIL.Image.Image):
            w, h = img.size
        else:
            h, w = img.shape[-2:]

        scale = self.min_size / min(h, w)
        if scale * max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)

        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = TF.resize(img, [new_h, new_w])

        if target['boxes'].numel() > 0:
            target['boxes'] = target['boxes'] * scale

        return img, target


def get_train_transforms(min_size=800, max_size=1333):
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        ResizeDetection(min_size=min_size, max_size=max_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std =[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(min_size=800, max_size=1333):
    return Compose([
        ResizeDetection(min_size=min_size, max_size=max_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std =[0.229, 0.224, 0.225]),
    ])