#!/usr/bin/env python3
"""
tools/model_factory.py
======================
Build evaluation models EXACTLY as the E2 trainer built them, and load
checkpoints with strict=True.

Mirrors classification/tools/train.py::build_model (the code path that
produced the 27 headline checkpoints):
    baseline  -> saga.build_saga_vit(arch, gate=False, ...)
    saga      -> saga.build_saga_vit(arch, gate=True,  ...)
    registers -> timm.create_model(arch, reg_tokens=4, img_size=224, ...)
                 (NO dynamic_img_size — the trainer does not set it)

Checkpoint loading accepts the trainers' wrapped format
({'model': state_dict, 'optimizer': ..., 'epoch': ..., ...}) as well as raw
state_dicts, strips 'module.' prefixes, then loads with strict=True — zero
missing, zero unexpected keys, or it raises. (A silent strict=False bug
already burned this project once; see figures/make_table4_*.py.)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga import build_saga_vit

ARCH_MAP = {
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
}
VARIANTS = ("baseline", "registers", "saga")


def build_model(
    arch: str,
    variant: str,
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000,
    reg_tokens: int = 4,
) -> torch.nn.Module:
    """Build a {baseline, registers, saga} model exactly as trained.
    `arch` is 'vit_small'/'vit_base'/'vit_large' or a full timm model name."""
    timm_arch = ARCH_MAP.get(arch, arch)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    if variant == "registers":
        import timm
        return timm.create_model(
            timm_arch,
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size,
            reg_tokens=reg_tokens,
        )

    return build_saga_vit(
        arch=timm_arch,
        gate=(variant == "saga"),
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        pretrained=False,
    )


def extract_state_dict(ckpt) -> "tuple[dict, bool]":
    """Unwrap a trainer checkpoint ({'model': sd, ...}) or pass through a raw
    state_dict; strip 'module.' (DDP) prefixes.
    Returns (state_dict, was_wrapped)."""
    state, wrapped = ckpt, False
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict"):
            if isinstance(ckpt.get(key), dict):
                state, wrapped = ckpt[key], True
                break
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}
    return state, wrapped


def load_checkpoint(model: torch.nn.Module, ckpt_path) -> dict:
    """Load `ckpt_path` into `model` with strict=True (raises on any missing
    or unexpected key). Returns checkpoint metadata (epoch/top1/... if the
    checkpoint was a wrapped trainer dict, else {})."""
    ckpt_path = Path(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # legacy pickles (older torch saves); trusted project checkpoints only
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state, wrapped = extract_state_dict(ckpt)
    model.load_state_dict(state, strict=True)

    meta = {}
    if wrapped:
        meta = {k: v for k, v in ckpt.items()
                if isinstance(v, (int, float, str, bool))}
    return meta
