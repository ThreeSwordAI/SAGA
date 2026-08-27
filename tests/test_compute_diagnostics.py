"""Integration: compute_diagnostics end-to-end on tiny models, fake data."""

import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from saga.metrics import compute_diagnostics
from saga.vit import build_saga_vit

SCHEMA_KEYS = {
    "n_images", "block_idx", "num_prefix_tokens",
    "sink_mad_k5", "sink_mu2s", "sink_mu3s", "sink_mu4s", "sink_mu5s",
    "sink_mu6s", "sink_fixed_thr", "fixed_thr_value",
    "oversmooth_pairwise", "oversmooth_consecutive_legacy",
    "oversmooth_pairwise_nosink", "nosink_excluded_mean",
    "eff_rank", "cls_norm_ratio", "cls_attn_share", "reg_norm_mean",
}


def fake_loader(n=6, batch_size=3):
    g = torch.Generator().manual_seed(0)
    images = torch.randn(n, 3, 224, 224, generator=g)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_diagnostics_schema_and_reg_handling():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=10, reg_tokens=4).eval()
    out = compute_diagnostics(model, fake_loader(), torch.device("cpu"),
                              with_attn=True, fixed_thr=5.0,
                              collect_norms=True)
    arrays = out.pop("_norms_arrays")

    assert set(out.keys()) == SCHEMA_KEYS
    n_blocks = len(model.blocks)
    assert out["n_images"] == 6
    assert out["num_prefix_tokens"] == 5
    assert out["reg_norm_mean"] is not None
    assert out["fixed_thr_value"] == 5.0
    assert out["sink_fixed_thr"] is not None
    assert len(out["cls_norm_ratio"]) == n_blocks
    assert len(out["cls_attn_share"]) == n_blocks
    # attention shares are probabilities
    assert all(0.0 <= s <= 1.0 for s in out["cls_attn_share"])

    assert arrays["last_block_patch_norms"].shape == (6, 196)
    assert arrays["cls_norms"].shape == (6, n_blocks)
    assert arrays["median_patch_norms"].shape == (6, n_blocks)
    assert arrays["last_block_patch_norms"].dtype.name == "float16"


def test_diagnostics_saga_no_attn():
    model = build_saga_vit("vit_tiny_patch16_224", gate=True,
                           num_classes=10).eval()
    out = compute_diagnostics(model, fake_loader(), torch.device("cpu"),
                              with_attn=False, fixed_thr=None)
    assert out["num_prefix_tokens"] == 1
    assert out["reg_norm_mean"] is None
    assert out["cls_attn_share"] is None
    assert out["sink_fixed_thr"] is None and out["fixed_thr_value"] is None
    assert out["eff_rank"] > 1.0
    assert -1.0 <= out["oversmooth_pairwise"] <= 1.0


def test_n_effrank_subsampling():
    model = build_saga_vit("vit_tiny_patch16_224", gate=False,
                           num_classes=10).eval()
    out = compute_diagnostics(model, fake_loader(), torch.device("cpu"),
                              with_attn=False, n_effrank=2)
    assert out["n_images"] == 6
    assert out["eff_rank"] > 0
