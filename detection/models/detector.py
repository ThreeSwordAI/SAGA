"""
detection/models/detector.py
=============================
Full detection model: ViT backbone + SFP neck + Faster R-CNN head.

Assembles DetectionBackbone + SimpleFPN + torchvision FasterRCNN.
torchvision handles the RPN, RoI pooling, and detection head.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from collections import OrderedDict

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .backbone import DetectionBackbone
from .neck     import SimpleFPN


class ViTDetector(nn.Module):
    """
    ViT + SFP + Faster R-CNN detector.

    Wraps the full pipeline in one nn.Module so that
    DDP and checkpoint saving work cleanly.

    Args:
        backbone      DetectionBackbone instance
        neck          SimpleFPN instance
        num_classes   Number of detection classes (80 for COCO) + 1 background
        anchor_sizes  Anchor sizes for RPN per FPN level
        anchor_ratios Anchor aspect ratios
        min_size      Min image size for detection
        max_size      Max image size for detection
    """

    def __init__(
        self,
        backbone,
        neck,
        num_classes:  int  = 81,   # 80 COCO + 1 background
        anchor_sizes        = ((32,), (64,), (128,), (256,)),
        anchor_ratios       = ((0.5, 1.0, 2.0),) * 4,
        min_size:     int  = 800,
        max_size:     int  = 1333,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck     = neck

        # Anchor generator for 4 FPN levels (P2-P5)
        anchor_gen = AnchorGenerator(
            sizes      = anchor_sizes,
            aspect_ratios = anchor_ratios,
        )

        # RoI align across 4 FPN levels
        roi_pooler = MultiScaleRoIAlign(
            featmap_names = ['0', '1', '2', '3'],
            output_size   = 7,
            sampling_ratio= 2,
        )

        # Build a temporary backbone wrapper that torchvision Faster R-CNN
        # accepts — it needs a backbone with .out_channels attribute
        fpn_channels = neck.out_channels

        class _FPNWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.out_channels = fpn_channels
            def forward(self, x):
                raise NotImplementedError  # not called directly

        # Build Faster R-CNN
        self.frcnn = FasterRCNN(
            backbone          = _FPNWrapper(),
            num_classes       = num_classes,
            rpn_anchor_generator = anchor_gen,
            box_roi_pool      = roi_pooler,
            min_size          = min_size,
            max_size          = max_size,
        )

        # Replace the dummy backbone with our real one (not used by FRCNN directly)
        # We override forward() below to handle the full pipeline
        self.num_classes = num_classes

    def forward(
        self,
        images:  List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ):
        """
        Args:
            images   List of [3, H, W] tensors (variable sizes OK)
            targets  List of target dicts (train) or None (inference)

        Returns:
            Training: dict of losses
            Inference: list of prediction dicts
        """
        # Step 1: resize and pad images (torchvision handles this)
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # Use torchvision's ImageList transform
        images_tl, targets = self.frcnn.transform(images, targets)

        # Step 2: backbone → neck → FPN features
        # Stack into a batch tensor
        batched = images_tl.tensors  # [B, 3, H, W]
        raw_feats = self.backbone(batched)
        fpn_feats = self.neck(raw_feats, grid_size=self.backbone.last_patch_grid)

        # Step 3: wrap as OrderedDict for torchvision RPN + RoI
        feat_dict = OrderedDict()
        for i, f in enumerate(fpn_feats):
            feat_dict[str(i)] = f

        # Step 4: RPN → proposals
        proposals, proposal_losses = self.frcnn.rpn(
            images_tl, feat_dict, targets)

        # Step 5: RoI heads → detections / losses
        detections, detector_losses = self.frcnn.roi_heads(
            feat_dict, proposals,
            images_tl.image_sizes, targets)

        # Step 6: post-process (rescale boxes to original size)
        detections = self.frcnn.transform.postprocess(
            detections, images_tl.image_sizes, original_image_sizes)

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        else:
            return detections


def build_detector(cfg: dict, paths: dict) -> ViTDetector:
    """
    Build a ViTDetector from config and paths dicts.

    Args:
        cfg    Merged config (from load_config in train.py)
        paths  Paths config (from paths.yaml)
    """
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

    t = cfg['train']
    anchor_sizes  = tuple(tuple(s) for s in m['anchor_sizes'])
    anchor_ratios = tuple(tuple(m['anchor_ratios'])
                          for _ in range(len(anchor_sizes)))

    detector = ViTDetector(
        backbone      = backbone,
        neck          = neck,
        num_classes   = m['num_det_classes'] + 1,  # +1 for background
        anchor_sizes  = anchor_sizes,
        anchor_ratios = anchor_ratios,
        min_size      = t['min_size'],
        max_size      = t['max_size'],
    )

    return detector