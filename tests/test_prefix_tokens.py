"""T1 — infer_num_prefix_tokens: reg4 -> 5, SAGA/baseline -> 1 (fixes B2)."""

import timm
import torch

from saga.metrics import infer_num_prefix_tokens
from saga.vit import build_saga_vit


def test_reg4_model_has_five_prefix_tokens():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=10, reg_tokens=4)
    p = infer_num_prefix_tokens(model)
    assert p == 5

    x = torch.randn(2, 201, 8)  # CLS + 4 registers + 196 patches
    assert x[:, p:, :].shape[1] == 196


def test_saga_and_baseline_have_one_prefix_token():
    for gate in (True, False):
        model = build_saga_vit("vit_tiny_patch16_224", gate=gate, num_classes=10)
        p = infer_num_prefix_tokens(model)
        assert p == 1

        x = torch.randn(2, 197, 8)  # CLS + 196 patches
        assert x[:, p:, :].shape[1] == 196


def test_plain_timm_vit_has_one_prefix_token():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=10)
    assert infer_num_prefix_tokens(model) == 1
