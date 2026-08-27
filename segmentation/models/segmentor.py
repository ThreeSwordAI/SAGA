"""
segmentation/models/segmentor.py
==================================
Full segmentation model: ViT backbone + SFP neck + segmentation head.

Architecture:
  ViT-B backbone → SimpleFPN (P2-P5) → SegmentationHead → mIoU

SegmentationHead:
  - Upsample P3, P4, P5 to P2 resolution (H/4, W/4)
  - Concatenate 4 scales: [B, 4×256, H/4, W/4]
  - 3×3 Conv → BN → ReLU → [B, 256, H/4, W/4]
  - 1×1 Conv → [B, 150, H/4, W/4]
  - Bilinear upsample ×4 → [B, 150, H, W]

Loss: CrossEntropyLoss with ignore_index=255
Metric: mIoU over 150 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .backbone import DetectionBackbone   # identical to detection/models/backbone.py
from .neck     import SimpleFPN           # identical to detection/models/neck.py


class SegmentationHead(nn.Module):
    """
    Multi-scale segmentation head.

    Takes 4 FPN feature maps [P2, P3, P4, P5] and produces
    a per-pixel class prediction at 1/4 input resolution.

    Args:
        in_channels   FPN output channels (256)
        num_classes   150 for ADE20K
        fuse_channels Internal fusion channels (256)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 150,
                 fuse_channels: int = 256):
        super().__init__()

        # Fuse 4 scales → single feature map
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, fuse_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_channels, fuse_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True),
        )

        # Per-pixel classifier
        self.classifier = nn.Conv2d(fuse_channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, fpn_features: List[torch.Tensor],
                target_size: tuple) -> torch.Tensor:
        """
        Args:
            fpn_features  [P2, P3, P4, P5] — each [B, 256, H_i, W_i]
                          P2 is largest (stride 4), P5 is smallest (stride 32)
            target_size   (H, W) of input image — for final upsample

        Returns:
            [B, num_classes, H, W] — full-resolution logits
        """
        p2, p3, p4, p5 = fpn_features
        h2, w2 = p2.shape[-2], p2.shape[-1]

        # Upsample all to P2 resolution
        p3_up = F.interpolate(p3, size=(h2, w2), mode='bilinear',
                              align_corners=False)
        p4_up = F.interpolate(p4, size=(h2, w2), mode='bilinear',
                              align_corners=False)
        p5_up = F.interpolate(p5, size=(h2, w2), mode='bilinear',
                              align_corners=False)

        # Fuse
        fused  = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)
        fused  = self.fuse(fused)

        # Classify
        logits = self.classifier(fused)   # [B, 150, H/4, W/4]

        # Upsample to input resolution
        logits = F.interpolate(logits, size=target_size,
                               mode='bilinear', align_corners=False)
        return logits


class ViTSegmentor(nn.Module):
    """
    ViT-B + SFP + SegmentationHead.

    Args:
        backbone    DetectionBackbone (same as E3, used here for segmentation)
        neck        SimpleFPN
        num_classes 150 for ADE20K
    """

    def __init__(self, backbone: DetectionBackbone, neck: SimpleFPN,
                 num_classes: int = 150):
        super().__init__()
        self.backbone = backbone
        self.neck     = neck
        self.head     = SegmentationHead(
            in_channels  = neck.out_channels,
            num_classes  = num_classes,
        )
        self.num_classes  = num_classes
        self.ignore_index = 255

    def forward(self, images: torch.Tensor,
                targets: torch.Tensor = None):
        """
        Args:
            images   [B, 3, H, W]
            targets  [B, H, W] long tensor with class indices 0-149
                     and ignore_index=255. None during inference.

        Returns:
            Training:  loss scalar
            Inference: logits [B, num_classes, H, W]
        """
        B, C, H, W = images.shape

        # Backbone → intermediate features
        raw_feats = self.backbone(images)   # list of [B, N, embed_dim]

        # Neck → FPN feature maps
        grid_size = self.backbone.last_patch_grid
        fpn_feats = self.neck(raw_feats, grid_size=grid_size)

        # Segmentation head → full-resolution logits
        logits = self.head(fpn_feats, target_size=(H, W))

        if targets is None:
            return logits

        # Loss
        loss = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
        )
        return loss

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Returns predicted class map [B, H, W]."""
        with torch.no_grad():
            logits = self.forward(images)
        return logits.argmax(dim=1)


def build_segmentor(cfg: dict, paths: dict) -> ViTSegmentor:
    """Build ViTSegmentor from merged config and paths."""
    m            = cfg['model']
    backbone_key = cfg.get('backbone_key', 'baseline')
    ckpt_path    = paths['backbones'][backbone_key]

    backbone = DetectionBackbone(
        arch        = m['arch'],
        gate        = m.get('gate', False),
        registers   = m.get('registers', 0),
        ckpt_path   = ckpt_path,
        fpn_indices = m['fpn_indices'],
        img_size    = m['img_size'],
    )

    neck = SimpleFPN(
        in_channels  = m['embed_dim'],
        out_channels = m['fpn_out_channels'],
        grid_size    = backbone.grid_size,
    )

    segmentor = ViTSegmentor(
        backbone    = backbone,
        neck        = neck,
        num_classes = m['num_classes'],
    )

    return segmentor