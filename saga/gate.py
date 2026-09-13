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

────────────────────────────────────────────────────────────────────────────
TASK-12 — gate_mode (the matched-init ablation)
────────────────────────────────────────────────────────────────────────────
The reviewer attack the project has to answer is: *"SAGA starts at
sigmoid(0) = 0.5, so every patch token is halved at init — isn't the gain
just favourable rescaling, i.e. LayerScale with extra parameters?"*
Answering it needs controls that share SAGA's insertion point (G1) and
differ ONLY in what is learnable there. `gate_mode` selects between them:

    none        no gate at all — stock timm attention (arm A, baseline)
    const       phi[H, N] registered but FROZEN at 0 -> G = 0.5 everywhere
                (arm B: the init-scale floor, 0 trainable gate params)
    headscalar  phi[H, 1] learnable, broadcast over every position
                (arm C: SAGA minus the spatial dimension — same code path,
                 only the shape differs)
    layerscale  standard LayerScale (per-channel gamma) at G1
                (arm D: the rescaling control a reviewer means)
    spatial     phi[H, N] learnable — SAGA as shipped (arms E and F)

`spatial` is byte-for-byte the shipped behaviour: same parameter name
(`blocks.{i}.attn.gate.phi`), same shape, same forward. Every tool that
reads phi (tools/extract_gate.py, figures/fig4_layer_analysis.py, the
trainer's per-epoch dump) keeps working unchanged for const/headscalar too
— only the trailing dimension differs.
"""

import torch
import torch.nn as nn

# Modes understood by build_gate() / build_saga_vit(). "none" means no gate
# module is inserted at all, so it never reaches build_gate().
GATE_MODES = ("none", "const", "headscalar", "layerscale", "spatial")

# Modes carried by SpatialGate (the phi-parameterised ones).
PHI_MODES = ("const", "headscalar", "spatial")

# DeiT-III LayerScale init for ViT-S/16, read from timm's own model config:
# timm.models.deit's `deit3_small_patch16_224` passes init_values=1e-6
# (verified against the installed timm 1.0.28). Recorded per run in
# meta.json / config.resolved.yaml as `layerscale_init`.
LAYERSCALE_INIT_DEIT3 = 1e-6


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
        mode      'spatial' (default, as shipped), 'const' (same shape,
                  requires_grad=False) or 'headscalar' (one value per head,
                  broadcast over positions). See the module docstring.

    Parameters:
        phi  [num_heads, n_patches]   for 'spatial' and 'const'
             [num_heads, 1]           for 'headscalar'
             Initialised to zero → G = 0.5 everywhere at the start of training.

    Forward:
        Input  sdpa_out : [B, num_heads, N, head_dim]
                          where N = n_patches + 1  (CLS at index 0)
        Output           [B, num_heads, N, head_dim]  — same shape
                          CLS token (index 0) is unchanged.
                          Patch tokens (index 1:) are multiplied by G ∈ (0,1).
    """

    def __init__(self, grid_h: int, grid_w: int, num_heads: int,
                 mode: str = "spatial"):
        super().__init__()
        if mode not in PHI_MODES:
            raise ValueError(
                f"SpatialGate mode must be one of {PHI_MODES}, got {mode!r} "
                f"(mode 'layerscale' is LayerScaleGate; mode 'none' inserts "
                f"no gate at all)")
        self.mode      = mode
        self.grid_h    = grid_h
        self.grid_w    = grid_w
        self.num_heads = num_heads
        self.n_patches = grid_h * grid_w
        self.current_grid_size = None  # for tracking input size during forward pass
        # φ_h: learnable position prior — shape [num_heads, n_patches]
        # Stored flat (not 2D) for efficient indexing during forward pass.
        # The 2D spatial structure is implicit: patch index p = r * grid_w + c.
        # 'headscalar' keeps this exact code path and drops only the spatial
        # dimension ([H, 1] broadcasts over positions), so arms C and E of the
        # TASK-12 ablation differ in nothing else.
        n_slots = 1 if mode == "headscalar" else self.n_patches
        self.phi = nn.Parameter(
            torch.zeros(num_heads, n_slots)
        )
        if mode == "const":
            # arm B: the gate exists (and is checkpointed, dumped and plotted
            # like any other) but never learns — G stays exactly 0.5.
            self.phi.requires_grad_(False)

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
        G = torch.sigmoid(self.phi)   # [H, self.n_patches]  ([H, 1] headscalar)

        # Interpolate gate if input size differs from training size (e.g. detection).
        # 'headscalar' has no spatial extent to resample: [H, 1] broadcasts over
        # any number of positions, so it is resolution-independent by construction.
        if self.mode != "headscalar" and n_patches != self.n_patches:
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

        For 'headscalar' the single per-head value is EXPANDED over the grid:
        that is what the gate actually applies at every position, so the map
        is exact (and its within-layer spatial std is 0 by construction).
        """
        with torch.no_grad():
            G = torch.sigmoid(self.phi)
            if self.mode == "headscalar":
                G = G.expand(self.num_heads, self.n_patches)
            return G.reshape(
                self.num_heads, self.grid_h, self.grid_w
            ).cpu()

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, "
            f"grid=({self.grid_h}×{self.grid_w}), "
            f"num_heads={self.num_heads}, "
            f"params={self.phi.numel()}"
            + ("" if self.phi.requires_grad else " (frozen)")
        )


class LayerScaleGate(nn.Module):
    """
    TASK-12 arm D — standard LayerScale at SAGA's insertion point.

    LayerScale (Touvron et al., *Going deeper with Image Transformers*;
    adopted by DeiT-III) multiplies a residual branch by a learnable
    PER-CHANNEL vector `gamma`, initialised to a small constant. It is the
    intervention a reviewer has in mind when asking whether SAGA's gain is
    "just rescaling with extra parameters".

    Two properties are deliberate, and both must be stated in TASK-12's
    `results/notes/ablation.md` (Phase C) rather than left implicit:

    1. **Placement matches the gate, not timm.** timm applies LayerScale
       AFTER the output projection (`x + ls1(attn(norm1(x)))`); here it is
       applied at G1, on the SDPA output before W_O, exactly where
       SpatialGate acts — so placement cannot be the confound that explains
       an arm-D/arm-E difference. The two are equivalent up to which side of
       the (linear, per-token) W_O the scaling sits on; only the
       parameterisation differs (per-channel of the concatenated head
       outputs, [H, D], versus per-output-channel of W_O).
    2. **LayerScale scales EVERY token, CLS included** — that is the standard
       formulation. SpatialGate leaves CLS untouched. Arm D is therefore the
       standard control, not a CLS-matched one.

    Parameters:
        gamma  [num_heads, head_dim]  (= embed_dim values per layer)
               Initialised to `init_values` (default: the DeiT-III value).
    """

    def __init__(self, num_heads: int, head_dim: int,
                 init_values: float = LAYERSCALE_INIT_DEIT3):
        super().__init__()
        self.mode = "layerscale"
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.init_values = float(init_values)
        self.gamma = nn.Parameter(
            torch.full((num_heads, head_dim), float(init_values))
        )

    def set_current_grid_size(self, grid_size):
        """No-op: LayerScale is per-channel, so it is resolution-independent.
        Present so SAGAViT._set_gate_grid can treat every gate alike."""
        return

    def forward(self, sdpa_out: torch.Tensor) -> torch.Tensor:
        """sdpa_out : [B, H, N, D] -> [B, H, N, D], every token scaled."""
        return sdpa_out * self.gamma.unsqueeze(0).unsqueeze(2)

    def extra_repr(self) -> str:
        return (f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"init_values={self.init_values:g}, "
                f"params={self.gamma.numel()}")


def build_gate(mode: str, grid_h: int, grid_w: int, num_heads: int,
               head_dim: int,
               layerscale_init: float = LAYERSCALE_INIT_DEIT3):
    """Gate factory for build_saga_vit. Returns None for mode 'none'.

    The ONLY place a gate_mode string becomes a module, so the arms of the
    TASK-12 ablation cannot drift apart anywhere else."""
    if mode not in GATE_MODES:
        raise ValueError(f"gate_mode must be one of {GATE_MODES}, "
                         f"got {mode!r}")
    if mode == "none":
        return None
    if mode == "layerscale":
        return LayerScaleGate(num_heads=num_heads, head_dim=head_dim,
                              init_values=layerscale_init)
    return SpatialGate(grid_h=grid_h, grid_w=grid_w, num_heads=num_heads,
                       mode=mode)