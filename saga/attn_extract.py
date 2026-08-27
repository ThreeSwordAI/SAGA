"""
saga/attn_extract.py
====================
Attention-map extraction for fused-SDPA models.

timm (and the repo's GatedAttention) compute attention with
``F.scaled_dot_product_attention``, which never materialises the post-softmax
attention matrix. ``capture_attention(model)`` monkey-patches each attention
module's forward so that, on every call, the explicit
``softmax(q·kᵀ/√d)`` map is computed and stashed — while the module's
ORIGINAL forward still produces the output, so model outputs are
bit-identical with and without capture (enforced by test T4).

Works for the repo's ``GatedAttention`` and stock timm ``Attention``
(baseline / registers models): both expose ``qkv``, ``num_heads``, ``scale``
and optional ``q_norm``/``k_norm``.

Intended for eval-mode use: the captured map carries no attention dropout
(dropout is inactive in eval anyway) and assumes no attention mask.

Usage:
    with capture_attention(model, blocks=[0, 11]) as attn:
        model(images)
        # attn[0], attn[11]: [B, H, T, T], rows sum to 1
    # store is overwritten on each forward; call attn.clear() between batches
    # if you want to be explicit.

Memory guard: only the blocks requested via ``blocks=`` are captured
(default: all blocks). Each captured map is [B, H, T, T] — for ViT-B at 224²
that is ~240 MB fp32 per block at batch 128; request fewer blocks or shrink
the batch if memory is tight.
"""

from contextlib import contextmanager

import torch


def _explicit_attention_map(mod, x: torch.Tensor) -> torch.Tensor:
    """Recompute the post-softmax attention of `mod` for input x [B, N, C].
    Mirrors the q/k path of timm Attention and saga GatedAttention exactly
    (same qkv projection, same q/k norms, same scale, no dropout/mask)."""
    B, N, C = x.shape
    H = mod.num_heads
    head_dim = getattr(mod, "head_dim", C // H)

    qkv = mod.qkv(x).reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)                       # each [B, H, N, head_dim]

    q_norm = getattr(mod, "q_norm", None)
    k_norm = getattr(mod, "k_norm", None)
    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)

    scale = getattr(mod, "scale", head_dim ** -0.5)
    attn = (q * scale) @ k.transpose(-2, -1)      # [B, H, N, N]
    return attn.softmax(dim=-1).detach()


@contextmanager
def capture_attention(model, blocks=None):
    """
    Context manager: capture post-softmax attention maps from
    ``model.blocks[i].attn`` while leaving model outputs bit-identical.

    Args:
        model:  a model exposing ``.blocks`` (timm ViT or SAGAViT).
        blocks: iterable of block indices to capture (memory guard);
                None = all blocks. Negative indices allowed.

    Yields:
        store: dict {block_idx: attention tensor [B, H, T, T]} — entries are
        overwritten on every forward pass; ``store.clear()`` between batches.
    """
    n_blocks = len(model.blocks)
    if blocks is None:
        idxs = list(range(n_blocks))
    else:
        idxs = sorted({i % n_blocks for i in blocks})

    store: dict = {}
    patched = []  # (module, had_instance_forward, saved_value)

    def make_forward(mod, idx, orig_forward):
        def wrapped(x, *args, **kwargs):
            if kwargs.get("attn_mask") is not None or any(
                    a is not None for a in args):
                raise NotImplementedError(
                    "capture_attention assumes no attention mask")
            store[idx] = _explicit_attention_map(mod, x)
            return orig_forward(x, *args, **kwargs)
        return wrapped

    try:
        for i in idxs:
            mod = model.blocks[i].attn
            had = "forward" in mod.__dict__
            saved = mod.__dict__.get("forward")
            patched.append((mod, had, saved))
            mod.forward = make_forward(mod, i, mod.forward)
        yield store
    finally:
        for mod, had, saved in patched:
            if had:
                mod.forward = saved
            else:
                del mod.forward
