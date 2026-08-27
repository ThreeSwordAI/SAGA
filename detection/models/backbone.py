"""
detection/models/backbone.py
=============================
Loads an E2 checkpoint and wraps it for use as a detection backbone.

The backbone:
  1. Loads ViT-B weights from an E2 best.pth checkpoint
  2. Supports register tokens (if backbone_key == 'registers')
  3. Returns intermediate block features for the feature pyramid
  4. Supports freezing for the first N epochs of detection training
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from saga import build_saga_vit


class DetectionBackbone(nn.Module):
    """
    ViT backbone for detection.

    Wraps build_saga_vit and adds:
    - Checkpoint loading from E2 best.pth
    - Intermediate feature extraction via forward_intermediates()
    - Selective parameter freezing

    Args:
        arch          timm arch name
        gate          True = SAGA, False = baseline
        registers     Number of register tokens (0 for baseline/SAGA)
        ckpt_path     Path to E2 best.pth
        fpn_indices   Which block outputs to return [3, 6, 9, 11]
        img_size      Backbone training image size (224)
    """

    def __init__(
        self,
        arch:       str,
        gate:       bool,
        registers:  int,
        ckpt_path:  str,
        fpn_indices: List[int],
        img_size:   int = 224,
    ):
        super().__init__()
        self.fpn_indices = fpn_indices

        # Build model
        if registers > 0:
            import timm
            timm_model = timm.create_model(
                arch,
                pretrained       = False,
                num_classes      = 1000,
                img_size         = img_size,
                reg_tokens       = registers,
                dynamic_img_size = True,   # allows detection-size inputs
            )
            # Wrap in a minimal SAGAViT-compatible object
            from saga.vit import SAGAViT
            self.vit = SAGAViT(timm_model)
        else:
            self.vit = build_saga_vit(
                arch       = arch,
                gate       = gate,
                img_size   = img_size,
                patch_size = 16,
                num_classes= 1000,
                pretrained = False,
            )

        self.embed_dim = self.vit.embed_dim
        self.grid_size = self.vit.grid_size  # e.g. (14, 14)

        # Load E2 checkpoint
        self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path: str):
        print(f"  Loading backbone from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # E2 saves as {'model': state_dict, ...}
        if 'model' in ckpt:
            state = ckpt['model']
        else:
            state = ckpt

        # Strip DDP prefix if present
        state = {k.replace('module.', ''): v for k, v in state.items()}

        # Load into the wrapped SAGAViT — map keys appropriately
        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"  Backbone loaded. top-1 from E2: {ckpt.get('top1', '?')}")

    def freeze(self):
        """Freeze all backbone parameters."""
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for p in self.vit.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features for the FPN.
        """
        intermediates, _ = self.vit.forward_intermediates(
            x, indices=self.fpn_indices
        )

        self.last_patch_grid = self.vit.last_patch_grid

        return intermediates

    def get_gate_maps(self):
        return self.vit.get_gate_maps()