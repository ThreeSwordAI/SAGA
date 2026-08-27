"""
SAGA SpatialGate — the core contribution.

Gate formula:
    G_{p,h} = σ( A + B + C + D )
    output  = G_{p,h} ⊙ SDPA_{p,h}

Terms:
    A  Content gate      W_g · SDPA_out  (adapted from Qwen NeurIPS 2025)
    B  2D spatial bias   φ_h(row, col)   learnable position prior
    C  Semantic CRF      λ_l · Σ w_{pq}·G_q  mean-field neighbor smoothing
    D  Local diversity   μ · (1 - cos_sim(k_p, k̄_neighbors))  anti-sink signal

Applied at position G1: after SDPA (attn@v), before output projection.
CLS token is never gated — only the N-1 patch tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SpatialGate(nn.Module):

    def __init__(
        self,
        d_head: int,
        num_heads: int,
        grid_h: int,                    # img_size / patch_size  (e.g. 14 for 224/16)
        grid_w: int,
        gate_terms: List[str],          # any subset of ['A', 'B', 'C', 'D']
        granularity: str = 'head_specific',
        lambda_0: float = 0.10,
        beta: float = 0.10,
        mu: float = 0.05,
        init_bias: float = 4.0,         # σ(4.0) ≈ 0.98 → identity at init
        layer_idx: int = 0,
    ):
        super().__init__()

        self.gate_terms  = set(gate_terms)
        self.granularity = granularity
        self.mu          = mu
        self.num_heads   = num_heads
        self.d_head      = d_head
        self.grid_h      = grid_h
        self.grid_w      = grid_w
        n_patches        = grid_h * grid_w

        # ── Term A: content gate ──────────────────────────────────────────────
        # Linear: d_head → 1 (head_specific) or d_head (elementwise)
        if 'A' in self.gate_terms:
            out_dim = 1 if granularity == 'head_specific' else d_head
            self.gate_proj = nn.Linear(d_head, out_dim, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, init_bias)

        # ── Term B: 2D spatial bias ───────────────────────────────────────────
        # One learnable scalar per head per patch position
        if 'B' in self.gate_terms:
            self.spatial_bias = nn.Parameter(
                torch.zeros(num_heads, n_patches)
            )

        # ── Term C: semantic CRF — pre-compute λ_l ───────────────────────────
        # λ_l = λ_0 · exp(−β · l), decays with layer depth
        if 'C' in self.gate_terms:
            lambda_l = lambda_0 * (-beta * layer_idx)
            self.register_buffer('lambda_l', torch.tensor(lambda_l).exp())

        # ── Term D: diversity weight stored as plain float ────────────────────
        # mu is already set above

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        sdpa_out: torch.Tensor,   # [B, H, N, d_head]  N = 1 + n_patches
        keys:     torch.Tensor,   # [B, H, N, d_head]
    ) -> torch.Tensor:

        B, H, N, D = sdpa_out.shape
        n_patches = N - 1          # exclude CLS token (position 0)

        # Separate CLS (unchanged) from patch tokens (gated)
        patch_out  = sdpa_out[:, :, 1:, :]   # [B, H, n_patches, D]
        patch_keys = keys[:, :, 1:, :]        # [B, H, n_patches, D]

        logit = sdpa_out.new_zeros(B, H, n_patches, 1)

        # ── A: content ───────────────────────────────────────────────────────
        if 'A' in self.gate_terms:
            logit = logit + self.gate_proj(patch_out)  # [B,H,n_patches, 1 or D]

        # ── B: spatial bias ──────────────────────────────────────────────────
        if 'B' in self.gate_terms:
            # spatial_bias: [H, n_patches] → broadcast to [B, H, n_patches, 1]
            logit = logit + self.spatial_bias.unsqueeze(0).unsqueeze(-1)

        # ── C: semantic CRF (mean-field, 4-connected neighbors) ──────────────
        if 'C' in self.gate_terms:
            g = logit.view(B, H, self.grid_h, self.grid_w)   # reshape to 2D grid
            g_pad = F.pad(g, (1, 1, 1, 1), mode='replicate') # pad borders
            neighbor_mean = (
                g_pad[:, :, :-2, 1:-1] +   # top
                g_pad[:, :,  2:, 1:-1] +   # bottom
                g_pad[:, :, 1:-1, :-2] +   # left
                g_pad[:, :, 1:-1,  2:]     # right
            ) / 4.0
            crf = self.lambda_l * neighbor_mean.view(B, H, n_patches, 1)
            logit = logit + crf

        # ── D: local diversity ────────────────────────────────────────────────
        if 'D' in self.gate_terms:
            # Compute mean of 4-connected key neighbors per patch
            k_grid = patch_keys.view(B, H, self.grid_h, self.grid_w, D)
            k_p    = k_grid.permute(0, 1, 4, 2, 3)              # [B,H,D,gh,gw]
            #k_pad  = F.pad(k_p, (1, 1, 1, 1), mode='replicate')
            k_pad  = F.pad(k_p, (1, 1, 1, 1, 0, 0), mode='replicate')
            k_neigh = (
                k_pad[:, :, :, :-2, 1:-1] +
                k_pad[:, :, :,  2:, 1:-1] +
                k_pad[:, :, :, 1:-1, :-2] +
                k_pad[:, :, :, 1:-1,  2:]
            ) / 4.0                                               # [B,H,D,gh,gw]
            k_neigh = k_neigh.permute(0, 1, 3, 4, 2).reshape(B, H, n_patches, D)

            # Cosine similarity between each key and its neighborhood mean
            k_norm  = F.normalize(patch_keys, dim=-1)
            kn_norm = F.normalize(k_neigh,    dim=-1)
            cos_sim = (k_norm * kn_norm).sum(dim=-1, keepdim=True) # [B,H,n,1]

            # Diversity is high for foreground, low for uniform background
            diversity = (1.0 - cos_sim)
            logit = logit + self.mu * diversity

        # ── Sigmoid → gate ────────────────────────────────────────────────────
        gate = torch.sigmoid(logit)   # [B, H, n_patches, 1]

        if self.granularity == 'elementwise':
            gate = gate.expand(B, H, n_patches, D)

        # ── Apply gate to patch tokens only ───────────────────────────────────
        gated_patches = gate * patch_out

        # Reconstruct: [CLS unchanged] + [gated patches]
        return torch.cat([sdpa_out[:, :, :1, :], gated_patches], dim=2)

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_gate_values(
        self,
        sdpa_out: torch.Tensor,
        keys:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Return gate values [B, H, n_patches] without applying them.
        Used for metric logging during validation.
        """
        B, H, N, D = sdpa_out.shape
        n_patches   = N - 1
        patch_out   = sdpa_out[:, :, 1:, :]
        patch_keys  = keys[:, :, 1:, :]

        logit = sdpa_out.new_zeros(B, H, n_patches, 1)

        if 'A' in self.gate_terms:
            logit = logit + self.gate_proj(patch_out)
        if 'B' in self.gate_terms:
            logit = logit + self.spatial_bias.unsqueeze(0).unsqueeze(-1)
        if 'C' in self.gate_terms:
            g     = logit.view(B, H, self.grid_h, self.grid_w)
            g_pad = F.pad(g, (1, 1, 1, 1), mode='replicate')
            nm    = (g_pad[:,:,:-2,1:-1] + g_pad[:,:,2:,1:-1] +
                     g_pad[:,:,1:-1,:-2] + g_pad[:,:,1:-1,2:]) / 4.0
            logit = logit + self.lambda_l * nm.view(B, H, n_patches, 1)
        if 'D' in self.gate_terms:
            k_grid  = patch_keys.view(B, H, self.grid_h, self.grid_w, D)
            k_p     = k_grid.permute(0, 1, 4, 2, 3)
            k_pad   = F.pad(k_p, (1, 1, 1, 1), mode='replicate')
            k_neigh = (k_pad[:,:,:,:-2,1:-1] + k_pad[:,:,:,2:,1:-1] +
                       k_pad[:,:,:,1:-1,:-2] + k_pad[:,:,:,1:-1,2:]) / 4.0
            k_neigh = k_neigh.permute(0,1,3,4,2).reshape(B, H, n_patches, D)
            cos_sim = (F.normalize(patch_keys, dim=-1) *
                       F.normalize(k_neigh, dim=-1)).sum(-1, keepdim=True)
            logit   = logit + self.mu * (1.0 - cos_sim)

        return torch.sigmoid(logit).squeeze(-1)   # [B, H, n_patches]