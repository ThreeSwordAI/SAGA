"""
saga/metrics.py
===============
The single canonical metric module. Every diagnostic number in the paper is
computed by the functions in this file — nothing else re-implements them.

Conventions (fixes audit bugs B2/B3/M1):
- All metrics operate on token tensors captured by forward hooks on EVERY
  ``model.blocks[k]`` (pre-final-norm block outputs), cast to fp32.
- "Patch tokens" always means ``x[:, P:, :]`` with
  ``P = infer_num_prefix_tokens(model)`` — 1 for baseline/SAGA, 1 + 4 for
  reg4 models. Never slice with a hard-coded 1 (audit bug B2).
- Sink threshold: primary metric is the per-image robust threshold
  median + 5·MAD (``sink_mad_k5``); a per-image μ+kσ sweep (k ∈ 2..6) and an
  optional fixed absolute threshold are emitted alongside (audit M1).
- Oversmoothing: full pairwise mean cosine over patch tokens per image,
  closed form (audit bug B3). The old consecutive-neighbor metric is kept —
  correctly prefix-sliced — as ``oversmooth_consecutive_legacy`` for
  comparability only; never for the paper.
"""

from typing import Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Prefix tokens (promoted from figures/make_table4_sink_threshold_from_tar.py)
# ─────────────────────────────────────────────────────────────────────────────

def infer_num_prefix_tokens(model) -> int:
    """
    Number of non-patch tokens at the START of the token sequence.

    Standard ViT / SAGA:      [CLS] [PATCH]        -> 1
    Register ViT (timm reg4): [CLS] [REG×4] [PATCH] -> 1 + 4 = 5

    Derived from the model object: timm models expose ``num_prefix_tokens``;
    the remaining checks are the fallback logic kept from the tar-file script
    for offline use (models that only expose reg-token attributes). SAGAViT
    exposes none of these and correctly falls through to 1.
    """
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)

    if hasattr(model, "num_reg_tokens"):
        return 1 + int(model.num_reg_tokens)

    if hasattr(model, "reg_token") and model.reg_token is not None:
        if model.reg_token.ndim == 3:
            return 1 + int(model.reg_token.shape[1])

    return 1


def _fp32(x: torch.Tensor) -> torch.Tensor:
    """Upcast half-precision inputs; leave fp32/fp64 untouched."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Token norms & sink counts
# ─────────────────────────────────────────────────────────────────────────────

def token_norms(x: torch.Tensor) -> torch.Tensor:
    """L2 norm over the feature dim. x: [B, N, D] -> [B, N]."""
    return _fp32(x).norm(dim=-1)


def sink_counts_mad(norms: torch.Tensor, k: float = 5.0) -> torch.Tensor:
    """
    Per-image robust sink count (the PRIMARY paper metric at k=5).

    For each image with norm vector v: m = median(v), MAD = median(|v − m|),
    threshold m + k·MAD; count of tokens STRICTLY above.
    All-equal norms give MAD = 0 -> threshold = m -> count 0 (no NaN).

    norms: [B, N] -> counts [B] (float).
    """
    v = _fp32(norms)
    m = v.median(dim=1, keepdim=True).values
    mad = (v - m).abs().median(dim=1, keepdim=True).values
    thr = m + k * mad
    return (v > thr).sum(dim=1).float()


def sink_counts_gauss(norms: torch.Tensor, k: float) -> torch.Tensor:
    """
    Per-image Gaussian sink count: threshold μ_i + k·σ_i (population σ,
    unbiased=False — matching the per-image stats of the promoted tar script).
    Count of tokens STRICTLY above. norms: [B, N] -> counts [B] (float).
    """
    v = _fp32(norms)
    mu = v.mean(dim=1, keepdim=True)
    sigma = v.std(dim=1, keepdim=True, unbiased=False)
    return (v > mu + k * sigma).sum(dim=1).float()


def sink_counts_fixed(norms: torch.Tensor, tau: float) -> torch.Tensor:
    """Count of tokens strictly above an absolute threshold τ (used later
    with a baseline-calibrated τ). norms: [B, N] -> counts [B] (float)."""
    return (_fp32(norms) > tau).sum(dim=1).float()


# ─────────────────────────────────────────────────────────────────────────────
# Oversmoothing & effective rank
# ─────────────────────────────────────────────────────────────────────────────

def oversmoothing_pairwise(x_patch: torch.Tensor) -> torch.Tensor:
    """
    Mean pairwise cosine similarity over ALL ordered pairs (i ≠ j) of patch
    tokens, per image, in closed form:

        p̂_j = p_j / ‖p_j‖,   S = ‖Σ_j p̂_j‖²,   mean cos = (S − N) / (N (N − 1))

    x_patch: [B, N, D] (prefix tokens already removed) -> [B].
    """
    x = _fp32(x_patch)
    p = torch.nn.functional.normalize(x, dim=-1)
    n = p.shape[1]
    s = p.sum(dim=1).pow(2).sum(dim=-1)          # [B] = ‖Σ_j p̂_j‖²
    return (s - n) / (n * (n - 1))


def oversmoothing_consecutive_legacy(x_patch: torch.Tensor) -> torch.Tensor:
    """
    The OLD metric: mean cosine of consecutive tokens in raster order
    (includes row-wraparound pairs). Kept — correctly prefix-sliced — for
    comparability with previously recorded numbers ONLY; never for the paper.

    x_patch: [B, N, D] (prefix tokens already removed) -> [B].
    """
    p = torch.nn.functional.normalize(_fp32(x_patch), dim=-1)
    n = p.shape[1]
    left = p.narrow(1, 0, n - 1)      # tokens j   = 0 .. N-2
    right = p.narrow(1, 1, n - 1)     # tokens j+1 = 1 .. N-1 (consecutive pair)
    return (left * right).sum(dim=-1).mean(dim=1)


def _nosink_per_image(x_patch: torch.Tensor, norms: torch.Tensor,
                      k: float = 5.0):
    """Sink-excluded pairwise cosine per image.

    Tokens with norm STRICTLY above median + k*MAD (the same per-image
    threshold as sink_counts_mad) are excluded; the closed-form mean pairwise
    cosine runs on the survivors. Returns (cos [B] with NaN where fewer than
    2 tokens survive, n_excluded [B])."""
    x = _fp32(x_patch)
    v = _fp32(norms)
    m = v.median(dim=1, keepdim=True).values
    mad = (v - m).abs().median(dim=1, keepdim=True).values
    keep = v <= (m + k * mad)                                  # [B, N]

    p = torch.nn.functional.normalize(x, dim=-1)
    w = keep.to(p.dtype).unsqueeze(-1)
    n_kept = keep.sum(dim=1).to(p.dtype)                       # [B]
    s = (p * w).sum(dim=1).pow(2).sum(dim=-1)                  # [B]
    denom = (n_kept * (n_kept - 1)).clamp_min(1.0)
    cos = (s - n_kept) / denom
    cos = torch.where(n_kept >= 2, cos,
                      torch.full_like(cos, float("nan")))
    return cos, (~keep).sum(dim=1).to(p.dtype)


def oversmoothing_pairwise_nosink(x_patch: torch.Tensor, norms: torch.Tensor,
                                  k: float = 5.0):
    """
    Oversmoothing with sink tokens excluded (TASK-02 addendum).

    Removing high-norm outliers mechanically raises the mean pairwise cosine
    among survivors, so a model that merely ABSORBS outliers (registers)
    could look "more oversmoothed" for arithmetic reasons. This variant
    excludes, per image, tokens above the median + k*MAD norm threshold and
    computes the closed-form pairwise mean cosine on the kept tokens. Images
    with fewer than 2 surviving tokens are skipped (still counted in the
    exclusion mean).

    x_patch: [B, N, D] (prefix tokens already removed); norms: [B, N].
    Returns (mean cosine over non-skipped images, mean #excluded per image)
    as scalar tensors; the first is NaN if every image was skipped.
    """
    cos, excluded = _nosink_per_image(x_patch, norms, k)
    valid = torch.isfinite(cos)
    mean_cos = cos[valid].mean() if bool(valid.any()) else \
        torch.tensor(float("nan"))
    return mean_cos, excluded.mean()


def effective_rank(x_patch: torch.Tensor) -> torch.Tensor:
    """
    Effective rank (Roy & Vetterli) of the patch-token matrix per image:
    σ = svdvals(P) in fp32, p = σ / Σσ, effrank = exp(−Σ p log p).

    x_patch: [B, N, D] -> [B].
    """
    x = _fp32(x_patch)
    sv = torch.linalg.svdvals(x)                 # [B, min(N, D)]
    p = sv / sv.sum(dim=1, keepdim=True).clamp_min(1e-12)
    plogp = torch.where(p > 0, p * p.clamp_min(1e-45).log(), p.new_zeros(()))
    return torch.exp(-plogp.sum(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# CLS / register / attention statistics
# ─────────────────────────────────────────────────────────────────────────────

def cls_norm_ratio(x: torch.Tensor, P: int) -> torch.Tensor:
    """
    Mean over images of ‖cls‖ / median_j ‖patch_j‖ for one block.
    x: [B, T, D] full token tensor; cls = token 0, patches = tokens P:.
    Returns a scalar tensor.
    """
    xf = _fp32(x)
    cls_n = xf[:, 0, :].norm(dim=-1)                          # [B]
    med = xf[:, P:, :].norm(dim=-1).median(dim=1).values      # [B]
    return (cls_n / med.clamp_min(1e-12)).mean()


def reg_norm_mean(x: torch.Tensor, P: int) -> Optional[torch.Tensor]:
    """
    Mean norm of the register tokens (tokens 1..P−1) — meaningful at the
    last block. Returns None when the model has no registers (P <= 1).
    x: [B, T, D] -> scalar tensor or None.
    """
    if P <= 1:
        return None
    regs = _fp32(x).narrow(1, 1, P - 1)   # tokens 1 .. P-1 (the registers)
    return regs.norm(dim=-1).mean()


def cls_attn_share(A: torch.Tensor, P: int) -> torch.Tensor:
    """
    Given post-softmax attention A [B, H, T, T], the mean over batch, heads,
    and PATCH queries of A[..., q, 0] — the attention mass patch tokens send
    to CLS. Returns a scalar tensor.
    """
    return _fp32(A)[:, :, P:, 0].mean()


# ─────────────────────────────────────────────────────────────────────────────
# The one-stop diagnostics pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_diagnostics(
    model,
    loader,
    device,
    with_attn: bool = True,
    fixed_thr: Optional[float] = None,
    *,
    n_effrank: Optional[int] = None,
    collect_norms: bool = False,
) -> dict:
    """
    Run `model` over `loader` once and compute every diagnostic metric.

    Hooks capture the output of EVERY model.blocks[k] (pre-final-norm), fp32.
    Per-image metrics (sink counts, oversmoothing, effective rank) use the
    LAST block (block_idx = -1); per-block profiles (cls_norm_ratio,
    cls_attn_share) cover all blocks.

    Args:
        with_attn:  capture attention (saga.attn_extract) for cls_attn_share.
        fixed_thr:  absolute norm threshold τ for sink_counts_fixed (or None).
        n_effrank:  compute effective rank on at most this many images
                    (None = all; SVD is the slow part).
        collect_norms: additionally return per-image norm arrays under key
                    "_norms_arrays" (fp16 numpy) for the histogram figures:
                    last_block_patch_norms [n, N], cls_norms [n, L],
                    median_patch_norms [n, L]. The caller pops this key
                    before writing JSON.

    Returns the metric fields of the diagnostics JSON schema (the caller adds
    provenance fields: ckpt, ckpt_sha256, git_sha, arch, variant, seed,
    timestamp).
    """
    model.eval()
    P = infer_num_prefix_tokens(model)
    blocks = model.blocks
    L = len(blocks)

    feats: dict = {}
    hooks = [
        blk.register_forward_hook(
            lambda mod, inp, out, idx=i: feats.__setitem__(idx, out.detach()))
        for i, blk in enumerate(blocks)
    ]

    gauss_ks = (2, 3, 4, 5, 6)
    sums = {
        "sink_mad_k5": 0.0,
        **{f"sink_mu{k}s": 0.0 for k in gauss_ks},
        "sink_fixed": 0.0,
        "over_pair": 0.0,
        "over_legacy": 0.0,
        "nosink_cos": 0.0,
        "nosink_excl": 0.0,
        "reg_norm": 0.0,
    }
    nosink_valid_n = 0
    effrank_sum, effrank_n = 0.0, 0
    ratio_sums = torch.zeros(L, dtype=torch.float64)
    attn_sums = torch.zeros(L, dtype=torch.float64)
    n_images = 0

    arr_last_norms, arr_cls, arr_median = [], [], []

    from saga.attn_extract import capture_attention
    import contextlib
    attn_ctx = capture_attention(model) if with_attn else contextlib.nullcontext({})

    with attn_ctx as attn_store:
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            B = images.shape[0]

            _ = model(images)

            x_last = _fp32(feats[L - 1])
            patch = x_last[:, P:, :]
            norms = token_norms(patch)

            sums["sink_mad_k5"] += sink_counts_mad(norms, k=5.0).sum().item()
            for k in gauss_ks:
                sums[f"sink_mu{k}s"] += sink_counts_gauss(norms, float(k)).sum().item()
            if fixed_thr is not None:
                sums["sink_fixed"] += sink_counts_fixed(norms, fixed_thr).sum().item()

            sums["over_pair"] += oversmoothing_pairwise(patch).sum().item()
            sums["over_legacy"] += oversmoothing_consecutive_legacy(patch).sum().item()

            ns_cos, ns_excl = _nosink_per_image(patch, norms, k=5.0)
            ns_valid = torch.isfinite(ns_cos)
            sums["nosink_cos"] += ns_cos[ns_valid].sum().item()
            nosink_valid_n += int(ns_valid.sum())
            sums["nosink_excl"] += ns_excl.sum().item()

            if n_effrank is None or effrank_n < n_effrank:
                take = B if n_effrank is None else min(B, n_effrank - effrank_n)
                effrank_sum += effective_rank(patch[:take]).sum().item()
                effrank_n += take

            r = reg_norm_mean(x_last, P)
            if r is not None:
                sums["reg_norm"] += r.item() * B

            block_cls, block_med = [], []
            for i in range(L):
                xf = _fp32(feats[i])
                ratio_sums[i] += cls_norm_ratio(xf, P).item() * B
                if collect_norms:
                    block_cls.append(xf[:, 0, :].norm(dim=-1))
                    block_med.append(xf[:, P:, :].norm(dim=-1).median(dim=1).values)
                if with_attn and i in attn_store:
                    attn_sums[i] += cls_attn_share(attn_store[i], P).item() * B

            if collect_norms:
                arr_last_norms.append(norms.half().cpu())
                arr_cls.append(torch.stack(block_cls, dim=1).half().cpu())
                arr_median.append(torch.stack(block_med, dim=1).half().cpu())

            attn_store.clear()
            feats.clear()
            n_images += B

    for h in hooks:
        h.remove()

    out = {
        "n_images": n_images,
        "block_idx": -1,
        "num_prefix_tokens": P,
        "sink_mad_k5": sums["sink_mad_k5"] / n_images,
        **{f"sink_mu{k}s": sums[f"sink_mu{k}s"] / n_images for k in gauss_ks},
        "sink_fixed_thr": (sums["sink_fixed"] / n_images
                           if fixed_thr is not None else None),
        "fixed_thr_value": fixed_thr,
        "oversmooth_pairwise": sums["over_pair"] / n_images,
        "oversmooth_consecutive_legacy": sums["over_legacy"] / n_images,
        "oversmooth_pairwise_nosink": (sums["nosink_cos"] / nosink_valid_n
                                       if nosink_valid_n else None),
        "nosink_excluded_mean": sums["nosink_excl"] / n_images,
        "eff_rank": effrank_sum / max(effrank_n, 1),
        "cls_norm_ratio": (ratio_sums / n_images).tolist(),
        "cls_attn_share": (attn_sums / n_images).tolist() if with_attn else None,
        "reg_norm_mean": (sums["reg_norm"] / n_images) if P > 1 else None,
    }
    if collect_norms:
        out["_norms_arrays"] = {
            "last_block_patch_norms": torch.cat(arr_last_norms).numpy(),
            "cls_norms": torch.cat(arr_cls).numpy(),
            "median_patch_norms": torch.cat(arr_median).numpy(),
        }
    return out
