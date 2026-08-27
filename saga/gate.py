"""
saga/gate.py
============
SAGA spatial gate — the core contribution.

Gate formula:
    G_{h,(r,c)}        = σ( φ_h(r, c) )
    output_{h,(r,c)}   = G_{h,(r,c)}  ⊙  SDPA_{h,(r,c)}

φ_h  ∈  ℝ^{H_g × W_g}   — one learnable scalar per patch position per head.

For ViT-B/16 on 224×224:
    H_g = W_g = 14   →   14 × 14 × 12 heads = 2,352 parameters  (0.003% of model)

Initialisation:
    φ_h = 0   →   G = σ(0) = 0.5  (neutral — identical to standard ViT at epoch 0)

During training:
    Background positions  →  φ_h(r,c) < 0  →  G < 0.5  →  suppressed
    Foreground positions  →  φ_h(r,c) > 0  →  G > 0.5  →  amplified

The CLS token (index 0 in the token sequence) is NEVER gated.
Only the 196 patch tokens are affected.

Position G1:
    Applied AFTER scaled dot-product attention (SDPA),
    BEFORE the output projection W_O.
    This suppresses sink contributions before they enter the residual stream
    without disturbing the attention weight distribution itself.
"""

import torch
import torch.nn as nn


class SpatialGate(nn.Module):
    """
    SAGA spatial gate for one attention layer.

    One instance is created per transformer block and passed into
    the attention module. Each block learns its own independent
    spatial prior.

    Args:
        grid_h    Height of the patch grid (e.g. 14 for ViT-B/16 on 224×224).
        grid_w    Width  of the patch grid.
        num_heads Number of attention heads in this layer.

    Parameters:
        phi  [num_heads, grid_h, grid_w]
             Initialised to zero → G = 0.5 everywhere at the start of training.

    Forward:
        Input  sdpa_out : [B, num_heads, N, head_dim]
                          where N = n_patches + 1  (CLS at index 0)
        Output           [B, num_heads, N, head_dim]  — same shape
                          CLS token (index 0) is unchanged.
                          Patch tokens (index 1:) are multiplied by G ∈ (0,1).
    """

    def __init__(self, grid_h: int, grid_w: int, num_heads: int):
        super().__init__()
        self.grid_h    = grid_h
        self.grid_w    = grid_w
        self.num_heads = num_heads
        self.n_patches = grid_h * grid_w
        self.current_grid_size = None  # for tracking input size during forward pass
        # φ_h: learnable position prior — shape [num_heads, n_patches]
        # Stored flat (not 2D) for efficient indexing during forward pass.
        # The 2D spatial structure is implicit: patch index p = r * grid_w + c.
        self.phi = nn.Parameter(
            torch.zeros(num_heads, self.n_patches)
        )
        
    def set_current_grid_size(self, grid_size):
        self.current_grid_size = grid_size
    
    def forward(self, sdpa_out: torch.Tensor) -> torch.Tensor:
        """
        Apply the spatial gate to the SDPA output.

        sdpa_out : [B, H, N, D]
            B = batch, H = heads, N = tokens (CLS + patches), D = head_dim

        Returns  : [B, H, N, D]  (same shape, CLS unchanged)
        """
        B, H, N, D = sdpa_out.shape
        n_patches   = N - 1   # exclude CLS token

        # Compute gate values: φ_h → σ(φ_h) ∈ (0,1)
        G = torch.sigmoid(self.phi)   # [H, self.n_patches]

        # Interpolate gate if input size differs from training size (e.g. detection)
        if n_patches != self.n_patches:
            if self.current_grid_size is None:
                raise RuntimeError(
                    f"SpatialGate needs current_grid_size for non-224 input. "
                    f"Got {n_patches} patches."
                )

            gh, gw = self.current_grid_size

            if gh * gw != n_patches:
                raise RuntimeError(
                    f"Gate grid mismatch: gh*gw={gh * gw}, but n_patches={n_patches}"
                )

            G = G.reshape(H, 1, self.grid_h, self.grid_w)

            G = torch.nn.functional.interpolate(
                G,
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            )

            G = G.reshape(H, gh * gw)

        # Broadcast: [H, n_patches] → [1, H, n_patches, 1]
        G = G.unsqueeze(0).unsqueeze(-1)

        # Apply gate to patch tokens only (index 1 onwards), leave CLS unchanged
        out = sdpa_out.clone()
        out[:, :, 1:, :] = sdpa_out[:, :, 1:, :] * G

        return out

    def get_gate_maps(self) -> torch.Tensor:
        """
        Return the learned gate values as a 2D spatial map per head.

        Returns: [num_heads, grid_h, grid_w]  — values in (0,1)

        Useful for visualisation and analysis of the learned spatial prior.
        Call this at any point during or after training to inspect φ_h.
        """
        with torch.no_grad():
            return torch.sigmoid(self.phi).reshape(
                self.num_heads, self.grid_h, self.grid_w
            ).cpu()

    def extra_repr(self) -> str:
        return (
            f"grid=({self.grid_h}×{self.grid_w}), "
            f"num_heads={self.num_heads}, "
            f"params={self.phi.numel()}"
        )