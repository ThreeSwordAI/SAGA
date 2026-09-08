"""
detection/models/detector.py
=============================
Full detection model: ViT backbone + SFP neck + Faster R-CNN head.

Assembles DetectionBackbone + SimpleFPN + torchvision FasterRCNN.
torchvision handles the RPN, RoI pooling, and detection head.

TASK-09 bug B5 — double normalization:
    detection/data/transforms.py already applies the ImageNet Normalize,
    and torchvision's GeneralizedRCNNTransform (inside FasterRCNN) applies
    its OWN normalization with ImageNet statistics by default. Every image
    was therefore shifted and scaled twice, i.e. the backbone never saw the
    distribution it was pretrained on. Every pre-fix detection number is
    void.
    THE ONE FIX (never both): FasterRCNN is constructed with the IDENTITY
    normalization (image_mean=[0,0,0], image_std=[1,1,1]) so the dataloader
    Normalize in transforms.py stays the single normalization step.
    `ViTDetector.check_normalization()` asserts this on real batches, on
    the device the training runs on.
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

# B5: the detector's own normalization is the identity — transforms.py owns
# the (single) ImageNet normalization.
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD  = (1.0, 1.0, 1.0)
# the statistics transforms.py normalizes with (kept here only so the
# normalization check can report what a double shift would have looked like)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


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

        # Build Faster R-CNN.
        # B5: identity normalization — the dataloader (transforms.py) is the
        # ONLY place ImageNet mean/std is applied. Do not "also" remove the
        # dataloader Normalize; exactly one of the two must normalize.
        self.frcnn = FasterRCNN(
            backbone          = _FPNWrapper(),
            num_classes       = num_classes,
            rpn_anchor_generator = anchor_gen,
            box_roi_pool      = roi_pooler,
            min_size          = min_size,
            max_size          = max_size,
            image_mean        = list(IDENTITY_MEAN),
            image_std         = list(IDENTITY_STD),
        )

        # Replace the dummy backbone with our real one (not used by FRCNN directly)
        # We override forward() below to handle the full pipeline
        self.num_classes = num_classes

    @torch.no_grad()
    def check_normalization(self, images: List[torch.Tensor],
                            atol: float = 1e-4) -> Dict:
        """B5 guard, meant to run ON THE DEVICE the training runs on, on a
        real batch: the tensor the backbone receives must be the dataloader's
        already-normalized image, NOT that image normalized a second time.

        Runs the detector's own GeneralizedRCNNTransform on `images` and
        compares the (unpadded) result with the input elementwise. Returns a
        dict of statistics for meta.json; raises AssertionError on failure.
        """
        if not images:
            raise ValueError("check_normalization needs at least one image")

        # fork_rng: in training mode GeneralizedRCNNTransform draws from the
        # CPU RNG to pick min_size, so running this check must not shift the
        # training run's random stream
        with torch.random.fork_rng(devices=[]):
            imgs_tl, _ = self.frcnn.transform([img.detach().clone()
                                               for img in images], None)
        got_batch = imgs_tl.tensors

        mean = torch.as_tensor(IMAGENET_MEAN, dtype=got_batch.dtype,
                               device=got_batch.device).view(3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD, dtype=got_batch.dtype,
                              device=got_batch.device).view(3, 1, 1)

        max_abs_diff = 0.0
        obs_means, obs_stds, exp_means, dbl_means = [], [], [], []
        for i, img in enumerate(images):
            h, w = imgs_tl.image_sizes[i]
            got = got_batch[i][:, :h, :w]
            exp = img.detach().to(got.device, got.dtype)
            if got.shape != exp.shape:
                raise AssertionError(
                    f"image {i}: the detector transform changed the shape "
                    f"{tuple(exp.shape)} -> {tuple(got.shape)}; the "
                    f"normalization check cannot compare them. The "
                    f"dataloader must resize to the detector's "
                    f"min_size/max_size (it did until now).")
            max_abs_diff = max(max_abs_diff,
                               float((got - exp).abs().max().item()))
            obs_means.append(got.mean(dim=(1, 2)))
            obs_stds.append(got.std(dim=(1, 2)))
            exp_means.append(exp.mean(dim=(1, 2)))
            dbl_means.append(((exp - mean) / std).mean(dim=(1, 2)))

        obs_mean = torch.stack(obs_means).mean(0)
        obs_std = torch.stack(obs_stds).mean(0)
        exp_mean = torch.stack(exp_means).mean(0)
        dbl_mean = torch.stack(dbl_means).mean(0)

        d_single = float((obs_mean - exp_mean).abs().max().item())
        d_double = float((obs_mean - dbl_mean).abs().max().item())
        # how far apart the two hypotheses are for THIS batch; if they are
        # indistinguishable the comparison below carries no information
        separation = float((exp_mean - dbl_mean).abs().max().item())

        stats = {
            "n_images": len(images),
            "max_abs_elementwise_diff": round(max_abs_diff, 8),
            "backbone_input_channel_mean": [round(v, 5)
                                            for v in obs_mean.tolist()],
            "backbone_input_channel_std": [round(v, 5)
                                           for v in obs_std.tolist()],
            "dataloader_channel_mean": [round(v, 5)
                                        for v in exp_mean.tolist()],
            "double_normalized_channel_mean": [round(v, 5)
                                               for v in dbl_mean.tolist()],
            "dist_to_single_normalized": round(d_single, 8),
            "dist_to_double_normalized": round(d_double, 8),
            "hypothesis_separation": round(separation, 8),
            "detector_image_mean": list(self.frcnn.transform.image_mean),
            "detector_image_std": list(self.frcnn.transform.image_std),
            "atol": atol,
        }

        if list(self.frcnn.transform.image_mean) != list(IDENTITY_MEAN) or \
                list(self.frcnn.transform.image_std) != list(IDENTITY_STD):
            raise AssertionError(
                f"B5: the detector's GeneralizedRCNNTransform normalizes with "
                f"mean={self.frcnn.transform.image_mean} "
                f"std={self.frcnn.transform.image_std}; it must be the "
                f"identity because transforms.py already normalized. {stats}")
        if max_abs_diff > atol:
            raise AssertionError(
                f"B5: the tensor reaching the backbone differs from the "
                f"dataloader's normalized image by {max_abs_diff:.6g} "
                f"(> atol {atol}). {stats}")
        if separation > 10 * atol and d_double <= d_single:
            raise AssertionError(
                f"B5: the backbone input's channel means are no further from "
                f"the DOUBLE-normalized prediction than from the single-"
                f"normalized one — normalization is applied twice. {stats}")
        return stats

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


def build_detector(cfg: dict, paths: dict = None) -> ViTDetector:
    """
    Build a ViTDetector from a resolved config.

    Args:
        cfg    Merged config. TASK-09 form: cfg['backbone'] = {'ckpt': ...,
               'sha256': ...} and optional cfg['model']['model_kwargs'].
        paths  LEGACY only (detection/tools/{evaluate,analyze}.py): a
               paths.yaml dict whose ['backbones'][cfg['backbone_key']] holds
               the checkpoint path. Omit it for the TASK-09 path.
    """
    m = cfg['model']
    if paths is not None:
        ckpt_path = paths['backbones'][cfg.get('backbone_key', 'baseline')]
        expected_sha = None
    else:
        bb = cfg['backbone']
        ckpt_path = bb['ckpt']
        expected_sha = bb.get('sha256')

    backbone = DetectionBackbone(
        arch        = m['arch'],
        gate        = m.get('gate', False),
        registers   = m.get('registers', 0),
        ckpt_path   = ckpt_path,
        fpn_indices = m['fpn_indices'],
        img_size    = m['img_size'],
        expected_sha256 = expected_sha,
        model_kwargs    = m.get('model_kwargs'),
    )

    # in_channels comes from the BUILT backbone (identical to the configured
    # embed_dim for the real archs; shrunken test models would otherwise
    # desync). A silent mismatch is refused.
    if (m.get('embed_dim') is not None and not m.get('model_kwargs')
            and int(m['embed_dim']) != int(backbone.embed_dim)):
        raise ValueError(
            f"config embed_dim {m['embed_dim']} != backbone embed_dim "
            f"{backbone.embed_dim} for arch {m['arch']!r}")
    neck = SimpleFPN(
        in_channels  = backbone.embed_dim,
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