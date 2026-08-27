"""
saga/vit.py
===========
SAGA ViT builder.

build_saga_vit() loads a timm ViT and replaces every attention block
with GatedAttention — a drop-in wrapper that inserts the SpatialGate
at position G1 (after SDPA, before the output projection W_O).

Usage:
    # SAGA model (with spatial gate)
    model = build_saga_vit('vit_base_patch16_224', gate=True)

    # Standard ViT baseline (no gate — identical to timm model)
    model = build_saga_vit('vit_base_patch16_224', gate=False)

    # For detection — return intermediate block features
    model = build_saga_vit('vit_base_patch16_224', gate=True)
    features, out = model.forward_intermediates(x, indices=[3, 6, 9, 11])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List, Tuple

from saga.gate import SpatialGate


class GatedAttention(nn.Module):
    """
    Drop-in replacement for timm's Attention module.

    Identical computation to standard multi-head self-attention except
    that the SpatialGate is applied at G1 (after SDPA, before W_O).

    Uses explicit QKᵀV computation (not fused SDPA) so we can intercept
    the SDPA output tensor and apply the gate before the output projection.

    The gate is optional — passing gate=None gives a standard attention block,
    which is used when build_saga_vit is called with gate=False.
    """

    def __init__(self, original_attn: nn.Module, gate: Optional[SpatialGate]):
        super().__init__()

        # Copy all components from the timm attention module
        self.qkv       = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj      = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.num_heads = original_attn.num_heads
        self.scale     = original_attn.scale

        # head_dim: compatible with timm 0.6+ and 0.9+
        if hasattr(original_attn, 'head_dim'):
            self.head_dim = original_attn.head_dim
        else:
            self.head_dim = original_attn.qkv.weight.shape[0] // (3 * self.num_heads)

        # q_norm and k_norm introduced in newer timm versions
        self.q_norm = getattr(original_attn, 'q_norm', nn.Identity())
        self.k_norm = getattr(original_attn, 'k_norm', nn.Identity())

        # The spatial gate — None for baseline, SpatialGate for SAGA
        self.gate = gate

    def forward(self, x: torch.Tensor, attn_mask=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        H, D    = self.num_heads, self.head_dim

        # QKV projection and reshape to per-head tensors
        qkv     = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)           # each [B, H, N, D]
        q       = self.q_norm(q)
        k       = self.k_norm(k)

        # Memory-efficient scaled dot-product attention (flash attention).
        # This avoids materialising the full [B, H, N, N] attention matrix
        # which OOMs at detection resolution (N~4200 patches at 800×1333).
        # The gate is applied to the OUTPUT of SDPA — not inside it —
        # so this is fully compatible with SAGA.
        dropout_p = self.attn_drop.p if self.training else 0.0
        sdpa_out  = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p  = dropout_p,
            scale      = self.scale,
        )                                             # [B, H, N, D]

        # ── SAGA gate at G1 ────────────────────────────────────────────────
        if self.gate is not None:
            sdpa_out = self.gate(sdpa_out)
        # ──────────────────────────────────────────────────────────────────

        # Reshape and output projection
        x = sdpa_out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SAGAViT(nn.Module):
    """
    Wrapper around a timm ViT model with SAGA gates.

    Adds forward_intermediates() for detection tasks — returns feature
    maps from specified intermediate blocks for use in the feature pyramid.
    """

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        # Copy all components from the timm model
        self.patch_embed = timm_model.patch_embed
        self.cls_token   = timm_model.cls_token
        self.pos_embed   = timm_model.pos_embed
        self.pos_drop    = timm_model.pos_drop
        self.blocks      = timm_model.blocks
        self.norm        = timm_model.norm
        self.head        = timm_model.head

        # Store patch grid size for detection
        if hasattr(timm_model.patch_embed, 'grid_size'):
            self.grid_size = timm_model.patch_embed.grid_size
        else:
            img_size   = timm_model.patch_embed.img_size
            patch_size = timm_model.patch_embed.patch_size
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
            if isinstance(patch_size, (list, tuple)):
                patch_size = patch_size[0]
            self.grid_size = (img_size // patch_size, img_size // patch_size)

        self.embed_dim = timm_model.embed_dim

    def _interpolate_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Interpolate position embeddings to match actual input size.
        Required when input is not 224×224 (e.g. detection at 800×1333).
        """
        N_curr = x.shape[1] - 1  # subtract CLS token
        N_orig = self.pos_embed.shape[1] - 1

        if N_curr == N_orig:
            return self.pos_embed  # no interpolation needed

        # Interpolate patch position embeddings
        cls_pe    = self.pos_embed[:, :1, :]     # [1, 1, C]
        patch_pe  = self.pos_embed[:, 1:, :]     # [1, N_orig, C]

        # Reshape to 2D grid
        gs_orig = int(N_orig ** 0.5)
        C       = patch_pe.shape[-1]
        patch_pe = patch_pe.reshape(1, gs_orig, gs_orig, C).permute(0, 3, 1, 2)

        # Target grid — use last_patch_grid for non-square detection inputs
        gh, gw = self.last_patch_grid
        patch_pe = torch.nn.functional.interpolate(
            patch_pe, size=(gh, gw),
            mode='bicubic', align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, gh * gw, C)

        return torch.cat([cls_pe, patch_pe], dim=1)


    def _set_gate_grid(self, gh: int, gw: int):
        """Tell every gate the current patch grid size (for non-square inputs)."""
        for blk in self.blocks:
            attn = blk.attn
            if hasattr(attn, 'gate') and attn.gate is not None:
                attn.gate.set_current_grid_size((gh, gw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass — returns class logits."""
        B = x.shape[0]
        x = self.patch_embed(x)
        # dynamic_img_size=True returns [B, H, W, C] — flatten to [B, N, C]
        if x.ndim == 4:
            self.last_patch_grid = (int(x.shape[1]), int(x.shape[2]))
            x = x.flatten(1, 2)
        else:
            gs = int(x.shape[1] ** 0.5)
            self.last_patch_grid = (gs, gs)
        gh, gw = self.last_patch_grid
        self._set_gate_grid(gh, gw)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = self._interpolate_pos_embed(x)
        x = self.pos_drop(x + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: List[int],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass that captures intermediate block outputs.

        Used by the detection neck to build a feature pyramid.

        Args:
            x        Input image tensor [B, 3, H, W]
            indices  Which block outputs to capture (0-indexed).
                     E.g. [3, 6, 9, 11] for ViT-B (12 blocks total).

        Returns:
            intermediates  List of tensors, one per index.
                           Each tensor is [B, n_patches, embed_dim]
                           (CLS token removed, patch tokens only).
            final_out      Class logits from the full forward pass.
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        # dynamic_img_size=True returns [B, H, W, C] — flatten to [B, N, C]
        if x.ndim == 4:
            self.last_patch_grid = (int(x.shape[1]), int(x.shape[2]))
            x = x.flatten(1, 2)
        else:
            gs = int(x.shape[1] ** 0.5)
            self.last_patch_grid = (gs, gs)
        gh, gw = self.last_patch_grid
        self._set_gate_grid(gh, gw)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = self._interpolate_pos_embed(x)
        x = self.pos_drop(x + pos)

        index_set     = set(indices)
        intermediates = {}

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in index_set:
                # Remove CLS token — keep only patch tokens
                intermediates[i] = x[:, 1:, :]  # [B, n_patches, C]

        x = self.norm(x)
        final_out = self.head(x[:, 0])

        ordered = [intermediates[i] for i in indices]
        return ordered, final_out

    def get_gate_maps(self) -> dict:
        """
        Return learned gate maps from all blocks.
        {layer_idx: tensor [num_heads, grid_h, grid_w]}
        """
        maps = {}
        for i, blk in enumerate(self.blocks):
            attn = blk.attn
            if hasattr(attn, 'gate') and attn.gate is not None:
                maps[i] = attn.gate.get_gate_maps()
        return maps


def build_saga_vit(
    arch:        str,
    gate:        bool = True,
    img_size:    int  = 224,
    patch_size:  int  = 16,
    num_classes: int  = 1000,
    pretrained:  bool = False,
) -> SAGAViT:
    """
    Build a ViT with or without the SAGA spatial gate.

    Args:
        arch        timm model name, e.g. 'vit_base_patch16_224'
        gate        True  → insert SpatialGate at G1 in every attention block
                    False → return standard timm ViT (baseline, no modification)
        img_size    Input image size (square assumed).
        patch_size  Patch size. Must match arch name.
        num_classes Number of output classes.
        pretrained  Load timm pretrained weights.

    Returns:
        SAGAViT wrapping the timm model.
    """
    model = timm.create_model(
        arch,
        pretrained       = pretrained,
        num_classes      = num_classes,
        img_size         = img_size,
        dynamic_img_size = True,   # allows non-224 input sizes (needed for detection)
    )

    # Compute patch grid dimensions
    grid_h = img_size // patch_size
    grid_w = img_size // patch_size
    num_heads = model.blocks[0].attn.num_heads

    if gate:
        # Replace every attention block with GatedAttention
        for block in model.blocks:
            spatial_gate = SpatialGate(
                grid_h    = grid_h,
                grid_w    = grid_w,
                num_heads = num_heads,
            )
            block.attn = GatedAttention(block.attn, gate=spatial_gate)

    return SAGAViT(model)