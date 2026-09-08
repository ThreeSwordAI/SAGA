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

    NOTE (TASK-09, bug B6): this wrapper copies exactly ONE prefix token
    (cls_token) and its pos-embed layout assumes `pos_embed = [1, 1+HW, C]`.
    A timm model with register tokens (`reg_tokens>0` -> num_prefix_tokens=5,
    `pos_embed = [1, 5+HW, C]`, plus a `reg_token` parameter) is therefore
    NOT wrappable: the register parameter would be silently dropped and the
    pos-embed reshape would either crash or misalign. Such models are
    REFUSED in __init__ — use the timm model directly and call its own
    `forward_intermediates()` (see detection/models/backbone.py).
    """

    def __init__(self, timm_model: nn.Module):
        super().__init__()
        # ── B6 guard: single-prefix-token models only ─────────────────────
        n_prefix = getattr(timm_model, "num_prefix_tokens", 1)
        has_reg = getattr(timm_model, "reg_token", None) is not None
        if n_prefix != 1 or has_reg:
            raise ValueError(
                f"SAGAViT supports single-prefix-token (CLS-only) ViTs; got "
                f"num_prefix_tokens={n_prefix}, reg_token="
                f"{'present' if has_reg else 'absent'}. Wrapping a register-"
                f"token model here drops its reg_token parameter and breaks "
                f"pos-embed interpolation (bug B6). Use the timm model "
                f"directly with timm's forward_intermediates() instead "
                f"(detection/models/backbone.py does this).")
        self.num_prefix_tokens = 1
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
        # B6 hard-assert: the [:1] / [1:] split below is only correct for a
        # single prefix token. __init__ refuses anything else, so this is
        # unreachable — it stays as a tripwire against future edits.
        if getattr(self, "num_prefix_tokens", 1) != 1:
            raise RuntimeError(
                f"pos-embed interpolation assumes exactly 1 prefix token, "
                f"model reports {self.num_prefix_tokens} (bug B6)")

        N_curr = x.shape[1] - 1  # subtract CLS token
        N_orig = self.pos_embed.shape[1] - 1

        if N_curr == N_orig:
            return self.pos_embed  # no interpolation needed

        # Interpolate patch position embeddings
        cls_pe    = self.pos_embed[:, :1, :]     # [1, 1, C]
        patch_pe  = self.pos_embed[:, 1:, :]     # [1, N_orig, C]

        # Reshape to 2D grid
        gs_orig = int(round(N_orig ** 0.5))
        C       = patch_pe.shape[-1]
        if gs_orig * gs_orig != N_orig:
            # a non-square source grid (or extra prefix tokens hiding in
            # pos_embed) would make the reshape below silently wrong
            raise RuntimeError(
                f"pos_embed carries {N_orig} patch positions, which is not a "
                f"square grid — cannot reshape for interpolation. A register-"
                f"token model reaching this point is bug B6.")
        patch_pe = patch_pe.reshape(1, gs_orig, gs_orig, C).permute(0, 3, 1, 2)

        # Target grid — use last_patch_grid for non-square detection inputs
        if getattr(self, "last_patch_grid", None) is None:
            raise RuntimeError(
                "last_patch_grid is unset — _interpolate_pos_embed must be "
                "called from forward()/forward_intermediates() after "
                "patch_embed")
        gh, gw = self.last_patch_grid
        if gh * gw != N_curr:
            raise RuntimeError(
                f"patch grid {gh}x{gw}={gh * gw} disagrees with the "
                f"{N_curr} patch tokens actually in the sequence")
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
    **timm_kwargs,
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
        timm_kwargs Extra kwargs forwarded verbatim to timm.create_model.
                    Production callers pass NONE of these (the legacy
                    construction must stay bit-identical); tests and CPU
                    smokes use them to shrink the model (depth/embed_dim/
                    num_heads) while exercising this exact code path.
                    `reg_tokens` is refused — see SAGAViT's B6 guard.

    Returns:
        SAGAViT wrapping the timm model.
    """
    if timm_kwargs.get("reg_tokens"):
        raise ValueError(
            "build_saga_vit cannot build register-token models: SAGAViT "
            "supports one prefix token only (bug B6). Build the timm model "
            "directly for the registers variant.")
    model = timm.create_model(
        arch,
        pretrained       = pretrained,
        num_classes      = num_classes,
        img_size         = img_size,
        dynamic_img_size = True,   # allows non-224 input sizes (needed for detection)
        **timm_kwargs,
    )

    # Compute patch grid dimensions — read back from the built model so an
    # overridden patch size (timm_kwargs) cannot desync the gate grid.
    # At the legacy defaults this is identical to img_size // patch_size.
    grid_h, grid_w = model.patch_embed.grid_size
    ps = model.patch_embed.patch_size
    ps = ps[0] if isinstance(ps, (tuple, list)) else ps
    if patch_size is not None and int(ps) != int(patch_size):
        raise ValueError(
            f"patch_size={patch_size} disagrees with the built model's "
            f"patch size {ps} (arch {arch!r})")
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