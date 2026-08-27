"""
detection/models/neck.py
=========================
Simple Feature Pyramid (SFP) neck for ViT backbones.

Takes 4 intermediate ViT block outputs (each [B, n_patches, C])
and produces 4 FPN-ready feature maps at different spatial scales.

Following ViT-Det (Li et al. 2022):
  - Reshape patch tokens from [B, HW, C] → [B, C, H, W]
  - Project each to fpn_channels (256)
  - Upsample/downsample to produce 4 scales (P2, P3, P4, P5)

The 4 scales correspond to strides 4, 8, 16, 32 relative to the
detection input image — standard for Faster R-CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SimpleFPN(nn.Module):
    """
    Simple Feature Pyramid for plain ViT backbones.

    Takes features from 4 intermediate ViT blocks and produces
    4 feature maps at strides {4, 8, 16, 32}.

    Args:
        in_channels    Embedding dim of backbone (768 for ViT-B)
        out_channels   Output FPN channels (256 standard)
        grid_size      Patch grid size, e.g. (14, 14)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        grid_size:    Tuple[int, int],
    ):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.out_channels = out_channels

        # 1×1 projection layers — one per scale
        # We have 4 intermediate features → 4 projections
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, out_channels),
            )
            for _ in range(4)
        ])

        # Scale-specific upsampling/downsampling to reach target strides
        # Input grid is 14×14 at stride 16 (224/16=14)
        # We want strides 4, 8, 16, 32
        # Stride 4  = 4×  upsample  (14 → 56)
        # Stride 8  = 2×  upsample  (14 → 28)
        # Stride 16 = 1×  identity  (14 → 14)
        # Stride 32 = 2×  downsample (14 → 7)
        self.scale_modules = nn.ModuleList([
            # P2 (stride 4): 4× upsample
            nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels,
                                   kernel_size=4, stride=4, padding=0),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
            ),
            # P3 (stride 8): 2× upsample
            nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels,
                                   kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
            ),
            # P4 (stride 16): identity
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=1),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
            ),
            # P5 (stride 32): 2× downsample
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
            ),
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        features: List[torch.Tensor],
        grid_size: Tuple[int, int],
    ) -> List[torch.Tensor]:
        """
        Args:
            features  List of 4 tensors, each [B, n_patches, embed_dim]
                      from backbone intermediate blocks.

        Returns:
            List of 4 tensors [B, out_channels, H_i, W_i]
            at strides [4, 8, 16, 32] relative to input image.
        """
        out = []
        for i, feat in enumerate(features):
            B, N, C = feat.shape

            # Infer actual grid size from token count (dynamic for detection)
            gh, gw = grid_size

            # Project to out_channels
            x = self.proj[i](feat)  # [B, N, out_channels]

            # Reshape from sequence to spatial map
            if gh * gw != N:
                raise RuntimeError(
                    f"FPN grid mismatch: got N={N} tokens, but grid is {gh}×{gw}={gh * gw}"
                )

            x = x.transpose(1, 2).reshape(
                B, self.out_channels, gh, gw
            )
            
            # Apply scale-specific module
            x = self.scale_modules[i](x)
            out.append(x)

        return out  # [P2, P3, P4, P5]