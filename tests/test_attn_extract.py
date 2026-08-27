"""T4 — capture_attention: identical logits, rows sum to 1, block selection."""

import timm
import torch

from saga.attn_extract import capture_attention
from saga.vit import build_saga_vit


def _check_model(model, x):
    model.eval()
    with torch.no_grad():
        ref = model(x)
    with capture_attention(model) as store:
        with torch.no_grad():
            out = model(x)
    assert torch.allclose(ref, out, atol=1e-4)

    n_blocks = len(model.blocks)
    assert set(store.keys()) == set(range(n_blocks))
    for attn in store.values():
        b, h, t, t2 = attn.shape
        assert (b, t) == (x.shape[0], t2)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # after the context exits, the patch is gone and outputs unchanged
    with torch.no_grad():
        post = model(x)
    assert torch.allclose(ref, post, atol=1e-4)


def test_stock_timm_vit():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 224, 224, generator=g)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=10)
    _check_model(model, x)


def test_saga_gated_attention():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, 224, 224, generator=g)
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=10)
    # make the gate non-trivial so capture must not disturb it
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.gate.phi.uniform_(-1.0, 1.0, generator=g)
    _check_model(model, x)


def test_registers_timm_vit():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 3, 224, 224, generator=g)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=10, reg_tokens=4)
    _check_model(model, x)


def test_blocks_subset_memory_guard():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 3, 224, 224, generator=g)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=10).eval()
    with capture_attention(model, blocks=[0, -1]) as store:
        with torch.no_grad():
            model(x)
    assert set(store.keys()) == {0, len(model.blocks) - 1}
