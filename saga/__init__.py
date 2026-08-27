"""
saga — Spatially-Aware Gated Attention for Vision Transformers
==============================================================

Public API:

    from saga import build_saga_vit, SpatialGate

    # SAGA model (with spatial gate — the proposed method)
    model = build_saga_vit('vit_base_patch16_224', gate=True)

    # Standard ViT baseline (no gate — for fair comparison)
    model = build_saga_vit('vit_base_patch16_224', gate=False)

    # Inspect the learned gate maps after training
    for i, block in enumerate(model.blocks):
        gate_maps = block.attn.gate.get_gate_maps()  # [num_heads, 14, 14]

    # For detection — extract intermediate features
    features, logits = model.forward_intermediates(x, indices=[3, 6, 9, 11])

That is the entire interface. Two imports, one function call.
"""

from saga.gate import SpatialGate
from saga.vit  import build_saga_vit, GatedAttention, SAGAViT

__all__ = ['SpatialGate', 'build_saga_vit', 'GatedAttention', 'SAGAViT']