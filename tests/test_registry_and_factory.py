"""T6 — run registry meta.json completeness; model factory strict round-trip."""

import json

import pytest
import torch

from saga.run_registry import create_run, finalize_run
from tools.model_factory import build_model, load_checkpoint

REQUIRED_META_KEYS = {
    "run_id", "git_sha", "git_dirty", "cmd", "seed", "world_size", "hostname",
    "gpu_names", "torch", "timm", "python", "cuda", "start_time", "end_time",
}


def test_create_run_writes_complete_meta(tmp_path):
    run_dir = create_run(tmp_path, "e2_vit_small_mixup_saga_s0",
                         {"model": {"arch": "vit_small_patch16_224"}}, seed=3)
    assert run_dir == tmp_path / "e2_vit_small_mixup_saga_s0"
    assert (run_dir / "config.resolved.yaml").exists()

    meta = json.loads((run_dir / "meta.json").read_text())
    assert REQUIRED_META_KEYS <= set(meta.keys())
    assert meta["seed"] == 3
    assert meta["end_time"] is None

    finalize_run(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["end_time"] is not None


def test_factory_state_dict_roundtrip_strict(tmp_path):
    model = build_model("vit_small", "saga")
    ckpt_path = tmp_path / "raw.pth"
    torch.save(model.state_dict(), ckpt_path)

    fresh = build_model("vit_small", "saga")
    load_checkpoint(fresh, ckpt_path)  # raw state_dict, strict=True

    # wrapped trainer format with DDP 'module.' prefixes
    wrapped_path = tmp_path / "wrapped.pth"
    torch.save({
        "epoch": 7,
        "model": {f"module.{k}": v for k, v in model.state_dict().items()},
        "top1": 12.3,
    }, wrapped_path)
    fresh2 = build_model("vit_small", "saga")
    meta = load_checkpoint(fresh2, wrapped_path)
    assert meta.get("epoch") == 7

    for k, v in model.state_dict().items():
        assert torch.equal(fresh2.state_dict()[k], v)


def test_factory_load_is_actually_strict(tmp_path):
    model = build_model("vit_small", "saga")
    state = model.state_dict()
    state.pop(sorted(state.keys())[0])  # drop one key -> must raise
    bad_path = tmp_path / "bad.pth"
    torch.save(state, bad_path)

    fresh = build_model("vit_small", "saga")
    with pytest.raises(RuntimeError):
        load_checkpoint(fresh, bad_path)

    # a baseline checkpoint must not silently load into a SAGA model
    base = build_model("vit_small", "baseline")
    base_path = tmp_path / "base.pth"
    torch.save(base.state_dict(), base_path)
    saga_model = build_model("vit_small", "saga")
    with pytest.raises(RuntimeError):
        load_checkpoint(saga_model, base_path)


def test_registers_factory_matches_trainer_construction():
    model = build_model("vit_small", "registers")
    assert int(model.num_prefix_tokens) == 5
    sd = model.state_dict()
    assert "reg_token" in sd
