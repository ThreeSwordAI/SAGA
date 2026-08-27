"""
SAGA ViT builder.

build_saga_vit() loads a timm ViT and replaces every attention block
with a GatedAttention wrapper that inserts SpatialGate at position G1
(after SDPA, before the output projection).

If gate_terms=[], returns a standard unmodified timm ViT (baseline V00).
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional

from saga_old.gate import SpatialGate


class GatedAttention(nn.Module):
    """
    Drop-in replacement for timm's Attention block.
    Always uses explicit attention (no fused SDPA) so we can
    intercept the intermediate tensor for gating.
    """

    def __init__(self, original_attn: nn.Module, gate: Optional[SpatialGate]):
        super().__init__()

        self.qkv       = original_attn.qkv
        self.attn_drop = original_attn.attn_drop
        self.proj      = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.num_heads = original_attn.num_heads
        self.scale     = original_attn.scale
        self.gate      = gate

        # head_dim differs across timm versions
        if hasattr(original_attn, 'head_dim'):
            self.head_dim = original_attn.head_dim
        else:
            self.head_dim = original_attn.qkv.weight.shape[0] // (3 * self.num_heads)

        # q_norm / k_norm introduced in timm 0.9+ (some ViT variants)
        self.q_norm = getattr(original_attn, 'q_norm', nn.Identity())
        self.k_norm = getattr(original_attn, 'k_norm', nn.Identity())

    def forward(self, x: torch.Tensor, attn_mask=None, **kwargs) -> torch.Tensor:
        # attn_mask is passed by newer timm versions — we accept and ignore it.
        # Our explicit SDPA implementation does not use attention masks
        # (ImageNet classification has no need for them).
        B, N, C = x.shape
        H, D    = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)              # each [B, H, N, D]
        q, k    = self.q_norm(q), self.k_norm(k)

        # Explicit scaled dot-product (not fused) — needed to intercept sdpa_out
        attn     = (q * self.scale) @ k.transpose(-2, -1)  # [B, H, N, N]
        attn     = attn.softmax(dim=-1)
        attn     = self.attn_drop(attn)
        sdpa_out = attn @ v                   # [B, H, N, D]

        # ── SAGA gate at G1 ───────────────────────────────────────────────────
        if self.gate is not None:
            sdpa_out = self.gate(sdpa_out, k)

        # Reshape and output projection (standard ViT)
        x = sdpa_out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def build_saga_vit(
    arch:         str,
    gate_terms:   List[str],
    img_size:     int   = 224,
    patch_size:   int   = 16,
    granularity:  str   = 'head_specific',
    gate_position:str   = 'G1',
    lambda_0:     float = 0.10,
    beta:         float = 0.10,
    mu:           float = 0.05,
    init_bias:    float = 4.0,
    num_classes:  int   = 1000,
    pretrained:   bool  = False,
) -> nn.Module:
    """
    Build a ViT with SAGA gates.

    Args:
        arch        timm model name, e.g. 'vit_base_patch16_224'
        gate_terms  terms to enable: subset of ['A','B','C','D']
                    [] → returns plain timm ViT (baseline V00, no modification)
        gate_position  only 'G1' is implemented for now
    """

    model = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
    )

    # If no gate terms → standard ViT baseline, no changes
    if not gate_terms:
        return model

    # Infer dimensions from the model
    grid_h    = img_size // patch_size
    grid_w    = img_size // patch_size
    num_heads = model.blocks[0].attn.num_heads
    embed_dim = model.embed_dim
    d_head    = embed_dim // num_heads

    assert gate_position == 'G1', (
        f"Only position G1 is implemented. Got: {gate_position}. "
        "Positions G2/G3/G5 will be added in wave 2 of E1."
    )

    # Replace every attention block with GatedAttention
    for layer_idx, block in enumerate(model.blocks):
        gate = SpatialGate(
            d_head=d_head,
            num_heads=num_heads,
            grid_h=grid_h,
            grid_w=grid_w,
            gate_terms=gate_terms,
            granularity=granularity,
            lambda_0=lambda_0,
            beta=beta,
            mu=mu,
            init_bias=init_bias,
            layer_idx=layer_idx,
        )
        block.attn = GatedAttention(block.attn, gate)

    return model