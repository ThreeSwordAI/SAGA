"""TASK-02C Phase A: per-cell tau v2, v2 apply, gate extraction, forensics."""

import json
import sys

import numpy as np
import pytest
import torch

import tools.compute_fixed_thr as compute_fixed_thr
from tools.apply_fixed_thr import apply_to_file
from tools.extract_gate import extract_phi, forensics_row, phi_stats
from tools.model_factory import extract_state_dict


def _norms_cell(tmp_path, arch, recipe, variant, tag, sha, norms):
    stem = f"e2_{arch}_{recipe}_{variant}_rlast_{tag}"
    np.savez(tmp_path / f"{stem}_norms.npz",
             last_block_patch_norms=np.asarray(norms, dtype=np.float16))
    (tmp_path / f"{stem}.json").write_text(json.dumps(
        {"arch": arch, "ckpt_sha256": sha}))


# ── compute_fixed_thr --per-cell (v2) ────────────────────────────────────────

def test_per_cell_tau_keys_and_values(tmp_path, monkeypatch):
    # cell taus are hand-computable: all-equal rows -> MAD 0 -> thr = value
    _norms_cell(tmp_path, "vit_small", "mixup", "baseline", "last", "sA",
                [[10.0] * 4, [20.0] * 4, [30.0] * 4])          # tau 20
    _norms_cell(tmp_path, "vit_small", "nomix", "baseline", "last", "sB",
                [[5.0] * 4])                                    # tau 5
    # distractors that must NOT be used: saga run, best tag
    _norms_cell(tmp_path, "vit_small", "mixup", "saga", "last", "sX",
                [[999.0] * 4])
    _norms_cell(tmp_path, "vit_small", "nomix", "baseline", "best", "sY",
                [[999.0] * 4])

    out = tmp_path / "fixed_thresholds_v2.json"
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--per-cell", "--diag-dir", str(tmp_path),
        "--out", str(out), "--archs", "vit_small",
        "--recipes", "mixup", "nomix"])
    compute_fixed_thr.main()

    d = json.loads(out.read_text())
    assert d["vit_small|mixup"] == 20.0
    assert d["vit_small|nomix"] == 5.0
    assert d["source_ckpt_sha256"] == {"vit_small|mixup": "sA",
                                       "vit_small|nomix": "sB"}
    assert "per (arch, recipe) cell" in d["definition"]
    # v1 file untouched (never created here)
    assert not (tmp_path / "fixed_thresholds.json").exists()


def test_per_cell_missing_baseline_raises(tmp_path, monkeypatch):
    out = tmp_path / "v2.json"
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--per-cell", "--diag-dir", str(tmp_path),
        "--out", str(out), "--archs", "vit_small", "--recipes", "mixup"])
    with pytest.raises(FileNotFoundError):
        compute_fixed_thr.main()


# ── apply_fixed_thr --version v2 ─────────────────────────────────────────────

def test_v2_apply_adds_only_v2_fields(tmp_path):
    stem = "e2_vit_small_nomix_saga_rlast_last"
    jp = tmp_path / f"{stem}.json"
    original = {"arch": "vit_small", "variant": "saga",
                "sink_fixed_thr": 3.25, "fixed_thr_value": 19.0,
                "other": "untouched"}
    jp.write_text(json.dumps(original))
    np.savez(tmp_path / f"{stem}_norms.npz",
             last_block_patch_norms=np.array(
                 [[1.0, 2.0, 50.0], [1.0, 1.0, 1.0]], dtype=np.float16))

    thresholds = {"vit_small|nomix": 10.0, "vit_small|mixup": 999.0}
    status = apply_to_file(jp, thresholds, version="v2")
    assert status.startswith("sink_fixed_v2")

    d = json.loads(jp.read_text())
    assert d["sink_fixed_v2"] == 0.5            # 1 token > 10 in image 0
    assert d["fixed_thr_v2_value"] == 10.0      # nomix tau, not mixup's
    # v1 fields and everything else byte-identical
    for k, v in original.items():
        assert d[k] == v

    # idempotent
    apply_to_file(jp, thresholds, version="v2")
    assert json.loads(jp.read_text()) == d

    # unparsable stem -> skip, no write
    weird = tmp_path / "weird_name.json"
    weird.write_text(json.dumps({"arch": "vit_small"}))
    np.savez(tmp_path / "weird_name_norms.npz",
             last_block_patch_norms=np.ones((1, 3), dtype=np.float16))
    assert apply_to_file(weird, thresholds,
                         version="v2") == "cannot parse recipe from filename"


def test_v1_apply_unchanged_by_v2_extension(tmp_path):
    stem = "e2_vit_small_nomix_saga_rlast_last"
    jp = tmp_path / f"{stem}.json"
    jp.write_text(json.dumps({"arch": "vit_small"}))
    np.savez(tmp_path / f"{stem}_norms.npz",
             last_block_patch_norms=np.array([[1.0, 20.0]], dtype=np.float16))
    status = apply_to_file(jp, {"vit_small": 10.0})     # v1 default
    assert status.startswith("sink_fixed_thr")
    d = json.loads(jp.read_text())
    assert d["sink_fixed_thr"] == 1.0 and d["fixed_thr_value"] == 10.0
    assert "sink_fixed_v2" not in d


# ── extract_gate: phi round-trip + stats ─────────────────────────────────────

def test_phi_extraction_roundtrip():
    from saga.vit import build_saga_vit
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=10)
    with torch.no_grad():
        for i, blk in enumerate(model.blocks):
            blk.attn.gate.phi.fill_(float(i) - 5.0)     # layer-identifiable

    ckpt = {"epoch": 3,
            "model": {f"module.{k}": v for k, v in model.state_dict().items()}}
    state, _ = extract_state_dict(ckpt)
    phi = extract_phi(state)

    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    assert phi.shape == (n_layers, n_heads, 196)
    for i in range(n_layers):
        assert np.allclose(phi[i], float(i) - 5.0)      # layer order correct


def test_phi_stats_known_values():
    phi = np.zeros((2, 3, 4), dtype=np.float32)         # gate = 0.5 exactly
    s = phi_stats(phi)
    assert s["shape_LHN"] == [2, 3, 4]
    for layer in s["layers"]:
        assert layer["mean_gate"] == 0.5
        assert layer["frac_below_0.4"] == 0.0
        assert layer["frac_above_0.75"] == 0.0
        assert not layer["has_nan"] and not layer["has_inf"]

    phi[1, 0, 0] = np.nan
    phi[0, :, :] = -10.0                                # gate ~ 0 -> below
    s = phi_stats(phi)
    assert s["layers"][0]["frac_below_0.25"] == 1.0
    assert s["layers"][1]["has_nan"] and not s["layers"][0]["has_nan"]


def test_extract_phi_rejects_non_saga():
    with pytest.raises(KeyError):
        extract_phi({"blocks.0.attn.qkv.weight": torch.zeros(3, 3)})


# ── forensics rows ───────────────────────────────────────────────────────────

def _mrow(**kw):
    base = {"path": "/x/last.pth", "filename": "last.pth", "exp": "e2",
            "arch": "vit_small", "recipe": "nomix", "variant": "saga",
            "seed": "rlast", "sha256": "abc"}
    base.update(kw)
    return base


def test_forensics_trainer_checkpoint():
    ckpt = {"epoch": 299, "top1": 79.13, "best_top1": 79.19,
            "model": {"w": torch.zeros(10, 10)},
            "scaler": {"scale": 1.0},
            "optimizer": {"param_groups": [{"lr": 1.2e-6}],
                          "state": {0: {"step": torch.tensor(3000)},
                                    1: {"step": 3000}}}}
    row = forensics_row(_mrow(), ckpt)
    assert row["epoch"] == 299 and row["top1"] == 79.13
    assert row["best_top1"] == 79.19
    assert row["last_lr"] == 1.2e-6
    assert row["optimizer_step_count"] == 3000
    assert row["n_optimizer_state_entries"] == 2
    assert row["has_scaler"] is True and row["has_ema"] is False
    assert row["n_model_tensors"] == 1 and row["model_params_m"] == 0.0
    assert "epoch;model;optimizer" in row["top_level_keys"]


def test_forensics_missing_keys_and_raw_state_dict():
    row = forensics_row(_mrow(), {"model": {"w": torch.zeros(2)},
                                  "epoch": 5})
    assert row["top1"] == "MISSING" and row["best_top1"] == "MISSING"
    assert row["last_lr"] == "MISSING"
    assert row["optimizer_step_count"] == "MISSING"
    assert row["has_scaler"] is False

    raw = {"w1": torch.zeros(2), "w2": torch.ones(3)}   # raw state dict
    row = forensics_row(_mrow(), raw)
    assert row["top_level_keys"] == "<raw state_dict, 2 tensors>"
    assert row["epoch"] == "MISSING"


def test_per_cell_refuses_v1_default_out(tmp_path, monkeypatch):
    # --per-cell without an explicit --out must never write the v1 path
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--per-cell", "--diag-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="v1 default path"):
        compute_fixed_thr.main()


def test_v2_apply_leaves_no_tmp_file(tmp_path):
    stem = "e2_vit_small_nomix_saga_rlast_last"
    jp = tmp_path / f"{stem}.json"
    jp.write_text(json.dumps({"arch": "vit_small"}))
    np.savez(tmp_path / f"{stem}_norms.npz",
             last_block_patch_norms=np.ones((1, 3), dtype=np.float16))
    apply_to_file(jp, {"vit_small|nomix": 0.5}, version="v2")
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(jp.read_text())["sink_fixed_v2"] == 3.0
