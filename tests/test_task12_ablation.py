"""TASK-12 A3 — matched-init ablation: gate modes, matrix, launchers.

The claim these tests defend is the one the whole task rests on: **the six
arms differ only in `gate_mode` (and `gate_init_logit` for arm F)**. That is
checked structurally — resolved configs diffed key by key, weights compared
tensor by tensor, activations compared before the gate — never by assertion.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import timm
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "classification" / "tools"))
import train as trainer  # classification/tools/train.py

from saga.gate import (GATE_MODES, LAYERSCALE_INIT_DEIT3, LayerScaleGate,
                       SpatialGate, build_gate)
from saga.vit import GatedAttention, build_saga_vit
from tools.model_factory import build_model as factory_build

MATRIX = REPO / "configs" / "abl_matrix.yaml"
ARMS = ["abl_vits_mixup_baseline_s0",
        "abl_vits_mixup_const05_s0",
        "abl_vits_mixup_headscalar_s0",
        "abl_vits_mixup_layerscale_s0",
        "abl_vits_mixup_spatial_i0_s0",
        "abl_vits_mixup_spatial_i4_s0"]

# ViT-S/16: 12 layers, 6 heads, 196 positions, embed_dim 384.
# These are the counts TASK_12's design table promises a reviewer.
EXPECTED_COUNTS = {          # gate_mode: (registered, trainable)
    "none":       (0, 0),
    "const":      (14112, 0),
    "headscalar": (72, 72),
    "layerscale": (4608, 4608),
    "spatial":    (14112, 14112),
}

TINY = dict(arch="vit_tiny_patch16_224", img_size=96, num_classes=4)
TINY_GEOM = (12, 3, 36)      # layers, heads, patches at img 96 / patch 16


def tiny(mode, **kw):
    return build_saga_vit(gate=(mode != "none"), gate_mode=mode,
                          patch_size=16, pretrained=False, **TINY, **kw)


def gate_counts(model):
    total = trainable = 0
    for name, p in model.named_parameters():
        if ".attn.gate." in name:
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# A3.1 — shapes and parameter counts, pinned at the production geometry
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", GATE_MODES)
def test_param_counts_vit_small(mode):
    """The numbers the ablation reports per arm, at the real ViT-S/16."""
    model = build_saga_vit("vit_small_patch16_224", gate=(mode != "none"),
                           gate_mode=mode, num_classes=10, pretrained=False)
    assert gate_counts(model) == EXPECTED_COUNTS[mode]


def test_param_shapes():
    for mode, slots in (("const", 36), ("headscalar", 1), ("spatial", 36)):
        m = tiny(mode)
        for blk in m.blocks:
            assert blk.attn.gate.phi.shape == (TINY_GEOM[1], slots), mode
    ls = tiny("layerscale")
    for blk in ls.blocks:
        assert blk.attn.gate.gamma.shape == (3, 64)      # heads x head_dim
        assert not hasattr(blk.attn.gate, "phi")


def test_parameter_names_are_the_ones_every_tool_reads():
    """tools/extract_gate.py, figures/fig4_layer_analysis.py and the
    trainer's phi dump all key on `blocks.{i}.attn.gate.phi`."""
    for mode in ("const", "headscalar", "spatial"):
        names = [n for n, _ in tiny(mode).named_parameters()
                 if n.endswith(".attn.gate.phi")]
        assert names[0] == "blocks.0.attn.gate.phi"
        assert len(names) == TINY_GEOM[0], mode
    ls_names = [n for n, _ in tiny("layerscale").named_parameters()
                if ".attn.gate." in n]
    assert all(n.endswith(".attn.gate.gamma") for n in ls_names)


def test_layerscale_init_is_timms_deit3_value():
    """Provenance, mechanically: LAYERSCALE_INIT_DEIT3 must equal the value
    timm's own deit3 config uses, not a number typed into our docstring."""
    m = timm.create_model("deit3_small_patch16_224", pretrained=False,
                          depth=1, num_classes=2)
    # compare in fp32, which is what both sides actually store
    assert (m.blocks[0].ls1.gamma.detach()[0].item()
            == torch.tensor(LAYERSCALE_INIT_DEIT3).float().item())
    g = tiny("layerscale").blocks[0].attn.gate.gamma
    assert torch.allclose(g, torch.full_like(g, LAYERSCALE_INIT_DEIT3))


def test_layerscale_init_is_configurable_and_recorded():
    m = tiny("layerscale", layerscale_init=0.5)
    assert m.blocks[0].attn.gate.gamma.detach()[0, 0].item() == 0.5
    assert m.blocks[0].attn.gate.init_values == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# A3.2 — gate_mode 'none' is stock timm attention, bit for bit
# ─────────────────────────────────────────────────────────────────────────────

def test_mode_none_is_stock_timm_attention():
    """No gate module is inserted at all, so the block keeps timm's own
    fused-SDPA Attention and the logits are bit-identical to timm's."""
    torch.manual_seed(0)
    ours = tiny("none")
    torch.manual_seed(0)
    ref = timm.create_model(TINY["arch"], pretrained=False,
                            num_classes=TINY["num_classes"],
                            img_size=TINY["img_size"], dynamic_img_size=True)

    assert not isinstance(ours.blocks[0].attn, GatedAttention)
    assert type(ours.blocks[0].attn) is type(ref.blocks[0].attn)

    sd_a, sd_b = ours.state_dict(), ref.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k

    x = torch.randn(2, 3, TINY["img_size"], TINY["img_size"],
                    generator=torch.Generator().manual_seed(1))
    ours.eval(), ref.eval()
    with torch.no_grad():
        assert torch.equal(ours(x), ref(x))          # bit-identical


def test_spatial_is_unchanged_from_the_shipped_implementation():
    """The headline runs must stay reproducible: gate=True with no gate_mode
    builds exactly what gate_mode='spatial' builds."""
    torch.manual_seed(3)
    shipped = build_saga_vit(gate=True, patch_size=16, pretrained=False,
                             **TINY)
    torch.manual_seed(3)
    explicit = tiny("spatial")
    sd_a, sd_b = shipped.state_dict(), explicit.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k
    assert shipped.blocks[0].attn.gate.phi.shape == (3, 36)
    assert isinstance(shipped.blocks[0].attn.gate, SpatialGate)


# ─────────────────────────────────────────────────────────────────────────────
# A3.3 — const is frozen, headscalar broadcasts, spatial does neither
# ─────────────────────────────────────────────────────────────────────────────

def _one_optimizer_step(model):
    opt = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.05)
    x = torch.randn(2, 3, TINY["img_size"], TINY["img_size"],
                    generator=torch.Generator().manual_seed(5))
    y = torch.tensor([0, 1])
    model.train()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_const_phi_survives_a_training_step_bit_identically():
    m = tiny("const")
    before = [b.attn.gate.phi.detach().clone() for b in m.blocks]
    _one_optimizer_step(m)
    for blk, b0 in zip(m.blocks, before):
        assert torch.equal(blk.attn.gate.phi, b0)
        assert torch.all(blk.attn.gate.phi == 0.0)   # G = sigmoid(0) = 0.5
        assert blk.attn.gate.phi.grad is None
        assert not blk.attn.gate.phi.requires_grad
    # ... and it is in the optimizer's parameter iterable all the same, so
    # the param-group logic is exercised exactly as for the learning arms
    assert any(p is m.blocks[0].attn.gate.phi for p in m.parameters())


@pytest.mark.parametrize("mode", ["headscalar", "spatial"])
def test_learnable_gates_move(mode):
    m = tiny(mode)
    before = m.blocks[0].attn.gate.phi.detach().clone()
    _one_optimizer_step(m)
    assert not torch.equal(m.blocks[0].attn.gate.phi, before)
    assert m.blocks[0].attn.gate.phi.requires_grad


def test_headscalar_broadcasts_and_spatial_does_not():
    """Arm C must apply ONE value to every position of a (layer, head);
    arm E must be able to apply a different value per position."""
    hs, sp = tiny("headscalar"), tiny("spatial")
    with torch.no_grad():
        for i, blk in enumerate(hs.blocks):
            blk.attn.gate.phi.fill_(float(i) - 5.0)
        for blk in sp.blocks:
            blk.attn.gate.phi.copy_(
                torch.linspace(-3, 3, 36).expand(3, 36))

    # gate maps: headscalar is constant within a (layer, head), spatial is not
    hs_map = hs.blocks[0].attn.gate.get_gate_maps()
    sp_map = sp.blocks[0].attn.gate.get_gate_maps()
    assert hs_map.shape == sp_map.shape == (3, 6, 6)
    assert float(hs_map.reshape(3, -1).std(dim=1).max()) == 0.0
    assert float(sp_map.reshape(3, -1).std(dim=1).min()) > 0.0

    # and the applied tensor really is broadcast, not just the map
    sdpa = torch.ones(2, 3, 37, 8)                   # [B, H, 1+36, D]
    out = hs.blocks[0].attn.gate(sdpa)
    patch = out[:, :, 1:, :]
    assert torch.allclose(patch.std(dim=2), torch.zeros_like(patch.std(dim=2)))
    assert torch.equal(out[:, :, 0, :], sdpa[:, :, 0, :])   # CLS never gated
    sp_out = sp.blocks[0].attn.gate(sdpa)
    assert sp_out[:, :, 1:, :].std(dim=2).min().detach().item() > 0.0


def test_layerscale_scales_every_token_including_cls():
    """Documented deviation: LayerScale is the standard formulation, so
    unlike the gate it does touch CLS. If that ever changes silently, arm D
    stops being the control it claims to be."""
    g = LayerScaleGate(num_heads=3, head_dim=8, init_values=0.25)
    sdpa = torch.ones(2, 3, 37, 8)
    out = g(sdpa)
    assert torch.allclose(out, torch.full_like(out, 0.25))
    assert torch.allclose(out[:, :, 0, :], torch.full((2, 3, 8), 0.25))


def test_const_and_spatial_at_init_are_the_same_function():
    """Both start at G = 0.5 everywhere, so at initialisation arms B and E
    must be bit-identical — they differ only in what happens afterwards."""
    torch.manual_seed(11)
    b = tiny("const")
    torch.manual_seed(11)
    e = tiny("spatial")
    x = torch.randn(2, 3, TINY["img_size"], TINY["img_size"],
                    generator=torch.Generator().manual_seed(12))
    b.eval(), e.eval()
    with torch.no_grad():
        assert torch.equal(b(x), e(x))


def test_build_gate_rejects_nonsense():
    with pytest.raises(ValueError, match="gate_mode must be one of"):
        build_gate("bogus", grid_h=6, grid_w=6, num_heads=3, head_dim=8)
    with pytest.raises(ValueError, match="SpatialGate mode must be one of"):
        SpatialGate(grid_h=6, grid_w=6, num_heads=3, mode="layerscale")
    assert build_gate("none", grid_h=6, grid_w=6, num_heads=3,
                      head_dim=8) is None
    with pytest.raises(ValueError, match="contradicts"):
        build_saga_vit(gate=False, gate_mode="spatial", patch_size=16,
                       **TINY)
    with pytest.raises(ValueError, match="contradicts"):
        build_saga_vit(gate=True, gate_mode="none", patch_size=16, **TINY)


# ─────────────────────────────────────────────────────────────────────────────
# A3.4 — two arms differing only in gate_mode: same seed, same data order,
#        same activations up to the gate
# ─────────────────────────────────────────────────────────────────────────────

def flatten(d, prefix=()):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


ALLOWED_DIFFS = {("run_id",), ("variant",),
                 ("model", "gate"), ("model", "gate_mode"),
                 ("knobs", "gate_init_logit"),
                 ("instrumentation", "log_grad_phi")}


def test_the_six_arms_differ_only_where_they_are_allowed_to():
    cfgs = {a: trainer.resolve_run_config(str(MATRIX), a) for a in ARMS}
    ref_id, ref = ARMS[0], flatten(cfgs[ARMS[0]])
    for arm in ARMS[1:]:
        flat = flatten(cfgs[arm])
        keys = set(ref) | set(flat)
        bad = sorted(k for k in keys
                     if k not in ALLOWED_DIFFS and ref.get(k) != flat.get(k))
        assert not bad, f"{arm} vs {ref_id}: unexpected diffs {bad}"
    # the shared settings the ablation depends on
    for arm, cfg in cfgs.items():
        assert cfg["seed"] == 0, arm
        assert cfg["recipe"] == "mixup", arm
        assert cfg["train"]["epochs"] == 100, arm
        assert cfg["model"]["arch"] == "vit_small_patch16_224", arm
        assert cfg["augmentation"]["mixup_alpha"] == 0.8, arm
        assert cfg["augmentation"]["cutmix_alpha"] == 1.0, arm
    # ... and gate_init_logit differs for exactly ONE arm (F)
    nonzero = {a for a, c in cfgs.items()
               if c["knobs"]["gate_init_logit"] != 0.0}
    assert nonzero == {"abl_vits_mixup_spatial_i4_s0"}
    assert cfgs["abl_vits_mixup_spatial_i4_s0"]["knobs"][
        "gate_init_logit"] == 4.0


def test_matrix_gate_modes_are_the_six_designed_arms():
    matrix = yaml.safe_load(MATRIX.read_text())
    assert sorted(matrix["runs"]) == sorted(ARMS)
    modes = {a: trainer.resolve_run_config(str(MATRIX), a)["model"]["gate_mode"]
             for a in ARMS}
    assert modes == {
        "abl_vits_mixup_baseline_s0": "none",
        "abl_vits_mixup_const05_s0": "const",
        "abl_vits_mixup_headscalar_s0": "headscalar",
        "abl_vits_mixup_layerscale_s0": "layerscale",
        "abl_vits_mixup_spatial_i0_s0": "spatial",
        "abl_vits_mixup_spatial_i4_s0": "spatial",
    }


def test_arms_share_weights_and_activations_up_to_the_gate():
    """Same seed -> the gate modules consume no RNG, so every non-gate
    weight is identical and everything upstream of G1 computes the same
    numbers. Only what happens AT the gate may differ."""
    models = {}
    for mode in ("none", "const", "headscalar", "layerscale", "spatial"):
        trainer.set_seed(0)
        models[mode] = tiny(mode)

    ref = models["none"].state_dict()
    for mode, m in models.items():
        sd = m.state_dict()
        shared = {k: v for k, v in sd.items() if ".attn.gate." not in k}
        assert shared.keys() == ref.keys(), mode
        for k in shared:
            assert torch.equal(shared[k], ref[k]), f"{mode}:{k}"

    # activations strictly upstream of the gate: the qkv projection output
    x = torch.randn(2, 3, TINY["img_size"], TINY["img_size"],
                    generator=torch.Generator().manual_seed(7))
    captured = {}

    def grab(mode):
        def hook(_mod, _inp, out):
            captured[mode] = out.detach().clone()
        return hook

    for mode, m in models.items():
        m.eval()
        attn = m.blocks[0].attn
        h = attn.qkv.register_forward_hook(grab(mode))
        with torch.no_grad():
            m(x)
        h.remove()
    for mode in models:
        assert torch.equal(captured[mode], captured["none"]), mode


def test_arms_see_the_same_data_in_the_same_order(tmp_path, monkeypatch):
    """The real trainer, the real loader construction, the real seeding —
    only train_one_epoch is replaced, by a recorder."""
    data_root = _fake_imagenet(tmp_path / "data")
    matrix = _tiny_two_arm_matrix(tmp_path, data_root)

    seen = {}

    def recorder(model, loader, *a, **kw):
        run = kw.get("_run") or getattr(recorder, "run", None)
        seen[run] = [t.tolist() for _, t in loader]
        return 0.0, 0

    for run in ("tinyA", "tinyB"):
        recorder.run = run
        monkeypatch.setattr(trainer, "train_one_epoch", recorder)
        trainer.run_training(matrix, run, data_root,
                             out_root=tmp_path / "runs", resume="auto",
                             max_epochs=1, device_str="cpu")
    assert seen["tinyA"] and seen["tinyA"] == seen["tinyB"]


def test_a_zero_duration_epoch_does_not_kill_the_run(tmp_path, monkeypatch):
    """A patched-out epoch can measure 0.0 s on Windows' 15.6 ms clock, and
    the img_per_sec division then raised ZeroDivisionError AFTER the epoch's
    work was done. Real epochs are minutes long, so this only ever bit
    instrumented runs — but it bit them at the worst possible moment."""
    data_root = _fake_imagenet(tmp_path / "data")
    matrix = _tiny_two_arm_matrix(tmp_path, data_root)
    monkeypatch.setattr(trainer, "train_one_epoch",
                        lambda *a, **kw: (0.0, 0))
    monkeypatch.setattr(trainer.time, "time", lambda: 1234.0)  # frozen clock
    run_dir = trainer.run_training(matrix, "tinyA", data_root,
                                   out_root=tmp_path / "runs",
                                   resume="auto", max_epochs=1,
                                   device_str="cpu")
    rows = list(csv.DictReader(open(run_dir / "log.csv", newline="")))
    assert len(rows) == 1 and rows[0]["img_per_sec"] not in ("", None)


# ─────────────────────────────────────────────────────────────────────────────
# trainer wiring: knob refusals, phi dump, grad logging
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_init_logit_refuses_to_be_a_silent_no_op():
    with pytest.raises(ValueError, match="FROZEN gate"):
        trainer.apply_knobs(tiny("const"), {"gate_init_logit": 4.0})
    with pytest.raises(ValueError, match="no gate phi"):
        trainer.apply_knobs(tiny("layerscale"), {"gate_init_logit": 4.0})
    with pytest.raises(ValueError, match="no gate phi"):
        trainer.apply_knobs(tiny("none"), {"gate_init_logit": 4.0})
    # on a learnable phi gate it does what it says
    m = tiny("spatial")
    trainer.apply_knobs(m, {"gate_init_logit": 4.0})
    assert torch.allclose(m.blocks[0].attn.gate.phi,
                          torch.full((3, 36), 4.0))
    assert torch.sigmoid(m.blocks[0].attn.gate.phi.detach())[0, 0] > 0.98


def test_apply_knobs_is_still_a_no_op_at_the_legacy_defaults():
    m = tiny("spatial")
    before = {k: v.clone() for k, v in m.state_dict().items()}
    trainer.apply_knobs(m, {"gate_init_logit": 0.0, "wd_phi_zero": False,
                            "drop_path_rate": 0.0})
    for k, v in m.state_dict().items():
        assert torch.equal(before[k], v)


@pytest.mark.parametrize("mode,slots", [("const", 36), ("headscalar", 1),
                                        ("spatial", 36), ("layerscale", None),
                                        ("none", None)])
def test_dump_phi_writes_exactly_for_the_phi_arms(tmp_path, mode, slots):
    trainer.dump_phi(tiny(mode), 0, tmp_path / mode)
    files = sorted((tmp_path / mode).glob("phi_e*.npz"))
    if slots is None:
        assert not files
    else:
        assert len(files) == 1
        assert np.load(files[0])["phi"].shape == (12, 3, slots)


def test_grad_phi_logger_tolerates_a_gate_without_phi(tmp_path):
    log = trainer.GradPhiLogger(tmp_path / "grad_phi.csv", every=1,
                                max_epoch=10)
    m = tiny("layerscale")
    _one_optimizer_step(m)
    log.maybe_log(m, 0, 0)                       # must not raise
    rows = list(csv.DictReader(open(tmp_path / "grad_phi.csv", newline="")))
    assert rows == []
    m2 = tiny("spatial")
    _one_optimizer_step(m2)
    log.maybe_log(m2, 0, 0)
    rows = list(csv.DictReader(open(tmp_path / "grad_phi.csv", newline="")))
    assert {int(r["layer"]) for r in rows} == set(range(12))


def test_gate_param_counts_helper():
    assert trainer.gate_param_counts(tiny("const")) == (12 * 3 * 36, 0)
    assert trainer.gate_param_counts(tiny("none")) == (0, 0)


def test_pre_task12_matrices_resolve_exactly_as_before():
    """e2r runs must keep building what they trained: no gate_mode key in
    the matrix -> 'spatial' for saga, 'none' otherwise, `gate` unchanged."""
    e2r = REPO / "configs" / "e2r_matrix.yaml"
    for run, want_gate, want_mode in (
            ("e2r_vits_mixup_saga_s1", True, "spatial"),
            ("e2r_vits_mixup_baseline_s1", False, "none"),
            ("e2r_vits_mixup_registers_s1", False, "none")):
        cfg = trainer.resolve_run_config(str(e2r), run)
        assert cfg["model"]["gate"] is want_gate
        assert cfg["model"]["gate_mode"] == want_mode
        assert cfg["knobs"]["gate_init_logit"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# model_factory: a checkpoint can only be rebuilt with its own gate_mode
# ─────────────────────────────────────────────────────────────────────────────

def test_factory_round_trips_each_arm_and_refuses_the_wrong_mode(tmp_path):
    for mode in ("const", "headscalar", "layerscale", "spatial", "none"):
        model = build_saga_vit("vit_small_patch16_224",
                               gate=(mode != "none"), gate_mode=mode,
                               num_classes=10, pretrained=False)
        sd = model.state_dict()
        rebuilt = factory_build("vit_small",
                                "saga" if mode != "none" else "baseline",
                                num_classes=10, gate_mode=mode)
        rebuilt.load_state_dict(sd, strict=True)          # exact match

    # headscalar weights must NOT load into the default (spatial) build
    hs = build_saga_vit("vit_small_patch16_224", gate=True,
                        gate_mode="headscalar", num_classes=10,
                        pretrained=False).state_dict()
    with pytest.raises(RuntimeError):
        factory_build("vit_small", "saga", num_classes=10).load_state_dict(
            hs, strict=True)


def test_factory_default_is_the_pre_task12_behaviour():
    for variant, mode in (("saga", "spatial"), ("baseline", "none")):
        a = factory_build("vit_small", variant, num_classes=10)
        b = factory_build("vit_small", variant, num_classes=10,
                          gate_mode=mode)
        assert a.state_dict().keys() == b.state_dict().keys()
    with pytest.raises(ValueError, match="no gate"):
        factory_build("vit_small", "registers", num_classes=10,
                      gate_mode="spatial")


# ─────────────────────────────────────────────────────────────────────────────
# A2 — launchers
# ─────────────────────────────────────────────────────────────────────────────

def _job(name):
    return (REPO / "scripts" / "jobs" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("arm", ARMS)
def test_job_and_submit_files_exist_and_name_the_right_run(arm):
    src = _job(f"{arm}.sbatch")
    assert f"--run {arm} \\\n" in src
    assert "--matrix configs/abl_matrix.yaml" in src
    assert "--partition=a100" in src and "--gres=gpu:a100:4" in src
    assert "--resume auto" in src
    assert "--max_epochs" not in src          # production runs the full 100
    assert "results/runs_smoke" not in src
    submit = (REPO / "scripts" / f"submit_{arm}.sh").read_text(
        encoding="utf-8")
    assert "--dependency=singleton" in submit
    assert f"scripts/jobs/{arm}.sbatch" in submit


def _committed_bytes(rel_path: str) -> bytes:
    """The bytes git actually ships — i.e. what the HPC checks out.

    NOT the Windows working-tree bytes: this clone has core.autocrlf=true,
    so a `.sh` comes back out of a checkout with CRLF while its blob (and
    every Linux checkout of it) stays LF. `.gitattributes` pins `*.sbatch`
    to LF for that reason; `*.sh` is not pinned, so the working tree is the
    wrong thing to compare against."""
    return subprocess.run(["git", "show", f"HEAD:{rel_path}"],
                          cwd=str(REPO), check=True,
                          capture_output=True).stdout


def test_job_files_are_exactly_what_the_generator_emits(tmp_path):
    """A hand-edited job file is how a run stops matching its matrix."""
    out = tmp_path / "scripts"
    (out / "jobs").mkdir(parents=True)
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_slurm_chain.py"),
         "--matrix", "configs/abl_matrix.yaml", "--out-dir", str(out),
         "--base-port", "29770", "--only", ",".join(sorted(ARMS))],
        cwd=str(REPO), check=True, capture_output=True)
    for arm in ARMS:
        # *.sbatch is pinned to LF by .gitattributes, so it compares byte for
        # byte in any checkout. *.sh is not pinned, and core.autocrlf=true
        # hands it back as CRLF on Windows — compare its CONTENT and leave
        # the line endings to test_committed_launchers_ship_with_unix_...
        assert (out / "jobs" / f"{arm}.sbatch").read_bytes() == \
            (REPO / "scripts" / "jobs" / f"{arm}.sbatch").read_bytes(), arm
        assert (out / f"submit_{arm}.sh").read_bytes() == \
            (REPO / "scripts" / f"submit_{arm}.sh"
             ).read_bytes().replace(b"\r\n", b"\n"), arm


def test_committed_launchers_ship_with_unix_line_endings():
    """A CRLF script fails on the HPC with `$'\\r': command not found`.
    What ships is the blob, so that is what is checked."""
    for arm in ARMS:
        for rel in (f"scripts/jobs/{arm}.sbatch",
                    f"scripts/submit_{arm}.sh"):
            assert b"\r\n" not in _committed_bytes(rel), rel
    assert b"\r\n" not in _committed_bytes("scripts/jobs/abl_smoke.sbatch")


def test_generator_template_still_produces_the_committed_e2r_files():
    """TASK-12 templated the matrix path into JOB_TEMPLATE. That must be a
    no-op for the e2r jobs, whose runs are finished and whose files are
    provenance. (Their ports come from the ORIGINAL 10-run sorted index, so
    a full --all regen would reshuffle them — hence the generator's guard.)"""
    sys.path.insert(0, str(REPO / "scripts"))
    import gen_slurm_chain
    committed = _job("e2r_vits_mixup_saga_s1.sbatch")
    assert gen_slurm_chain.JOB_TEMPLATE.format(
        run_id="e2r_vits_mixup_saga_s1", port=29704,
        matrix="configs/e2r_matrix.yaml", ckpt_root_arg="") == committed


def test_ablation_ports_collide_with_nothing_in_the_repo():
    sys.path.insert(0, str(REPO / "tests"))
    from test_task09_dense import _launcher_ports
    ports = _launcher_ports()
    for port in list(range(29770, 29776)) + [29769]:
        owners = {o for o in ports.get(port, set())
                  if "abl_" in o}
        others = ports.get(port, set()) - owners
        assert not others, f"port {port} also used by {others}"
    # and every ablation launcher's port is one of ours
    for arm in ARMS:
        line = [l for l in _job(f"{arm}.sbatch").splitlines()
                if "--master_port=" in l][0]
        assert 29770 <= int(line.split("=")[1].strip().rstrip("\\").strip()) \
            <= 29775


def test_production_jobs_send_checkpoints_to_the_matrix_ckpt_root():
    matrix = yaml.safe_load(MATRIX.read_text())
    root = matrix["ckpt_root"]
    assert root and str(root).startswith("/home/woody/"), \
        "the bulk filesystem per How to Run.md §1, not hpc or vault"
    for arm in ARMS:
        assert f"--ckpt_root {root}\n" in _job(f"{arm}.sbatch"), arm


def test_smoke_and_production_never_share_a_ckpt_root():
    """They share run_ids. A shared root would leave the smoke's 2-epoch
    last.pth exactly where the 100-epoch run's `--resume auto` looks, and
    the trainer would resume from it with only a schedule warning."""
    production = str(yaml.safe_load(MATRIX.read_text())["ckpt_root"])
    src = _job("abl_smoke.sbatch")
    smoke = [l.split(":-", 1)[1].rstrip("}") for l in src.splitlines()
             if l.startswith("SMOKE_CKPT_ROOT=")][0]
    assert smoke != production
    assert not smoke.startswith(production.rstrip("/") + "/")
    assert "--ckpt_root $SMOKE_CKPT_ROOT" in src
    assert production not in src, \
        "the production root must not appear anywhere in the smoke job"


def test_e2r_matrix_has_no_ckpt_root_so_its_jobs_are_unchanged():
    """Those runs are finished; their job files are provenance."""
    e2r = yaml.safe_load((REPO / "configs" / "e2r_matrix.yaml").read_text())
    assert "ckpt_root" not in e2r
    assert "--ckpt_root" not in _job("e2r_vits_mixup_saga_s1.sbatch")


def test_smoke_job_covers_every_arm_and_gates_before_staging():
    src = _job("abl_smoke.sbatch")
    for arm in ARMS:
        assert arm in src, arm
    assert "--out_root results/runs_smoke" in src
    assert "--max_epochs 2" in src
    assert "tests/test_task12_ablation.py" in src
    assert "tools/check_abl_smoke.py" in src

    lines = src.splitlines()

    def first(pred):
        return next(i for i, l in enumerate(lines) if pred(l))

    i_src = first(lambda l: l.strip().startswith("source ")
                  and "env_alex.sh" in l)
    i_setu = first(lambda l: l.strip() == "set -u")
    i_py = first(lambda l: "SAGA_PY=" in l)
    i_torch = first(lambda l: "import torch, timm" in l)
    i_trap = first(lambda l: l.strip() == "trap cleanup_imagenet EXIT")
    i_stage = first(lambda l: l.strip() == "stage_imagenet")
    assert i_setu > i_src, "set -u must not precede the env sourcing"
    assert i_py < i_torch < i_trap < i_stage, \
        "the interpreter preflight must be fatal BEFORE staging, and the " \
        "trap armed before staging can be killed"
    assert src.count("cleanup_imagenet") == 1     # the trap is the only caller
    # TASK-10: an interpreter resolved off $PATH silently became
    # /usr/bin/python once the conda module was renamed. Executable lines
    # only — the comment above it explains exactly this.
    code = [l for l in lines if not l.lstrip().startswith("#")]
    assert not [l for l in code if "which python" in l]
    assert "/home/vault/iwi5/iwi5359h/envs/saga/bin/python" in src


# ─────────────────────────────────────────────────────────────────────────────
# tools/check_abl_smoke.py — it must FAIL on a broken smoke, not just pass
# ─────────────────────────────────────────────────────────────────────────────

def _write_smoke_tree(root: Path, break_const=False, break_config=False,
                      init_drift=0.0, break_init=False):
    matrix = yaml.safe_load(MATRIX.read_text())
    for run_id, run in matrix["runs"].items():
        mode = run.get("gate_mode",
                       "spatial" if run["variant"] == "saga" else "none")
        d = root / run_id
        (d / "ckpt").mkdir(parents=True)
        (d / "ckpt" / "last.pth").write_bytes(b"x")
        (d / "meta.json").write_text(json.dumps(
            {"run_id": run_id, "seed": 0, "end_time": "2026-09-13T00:00:00"}))
        cfg = trainer.resolve_run_config(str(MATRIX), run_id)
        if break_config and run_id.endswith("headscalar_s0"):
            cfg["train"]["epochs"] = 42          # an arm on its own schedule
        (d / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
        with open(d / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trainer.LOG_FIELDS)
            w.writeheader()
            for e in (0, 1):
                w.writerow({"epoch": e, "lr": 1e-4, "train_loss": 7.0 - e,
                            "val_top1_full": 1.0 + e, "val_top5_full": 5.0,
                            "val_loss": 6.9, "img_per_sec": 900.0,
                            "wall_time": 420.0})
        slots = {"const": 196, "headscalar": 1, "spatial": 196}.get(mode)
        if slots:
            (d / "gates").mkdir()
            init = float(run.get("gate_init_logit", 0.0))
            if init:
                # epoch-0 dumps land AFTER one trained epoch, so a real run's
                # phi has drifted a little (weight decay on phi, mostly)
                init = 0.0 if break_init else init - init_drift
            base = np.full((12, 6, slots), init, dtype=np.float32)
            for e in (0, 1):
                phi = base.copy()
                if mode != "const" or break_const:
                    phi += 0.01 * e
                np.savez(d / "gates" / f"phi_e{e:03d}.npz", phi=phi)
        if run.get("log_grad_phi"):
            (d / "grads").mkdir()
            with open(d / "grads" / "grad_phi.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "iter", "layer", "grad_phi_norm"])
                w.writerow([0, 0, 0, 1e-4])


def _run_checker(root: Path):
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / "check_abl_smoke.py"),
         "--runs-root", str(root), "--matrix", str(MATRIX)],
        cwd=str(REPO), capture_output=True, text=True)


def test_checker_passes_a_good_smoke(tmp_path):
    _write_smoke_tree(tmp_path)
    res = _run_checker(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert ", 0 failed" in res.stdout


def test_checker_accepts_the_real_epoch0_drift_but_not_a_lost_init(tmp_path):
    """The 2026-09-13 smoke read 3.9926 for arm F's epoch-0 phi, because the
    dump happens after the epoch trains and AdamW's decoupled wd=0.05 has
    already contracted phi by prod(1 - lr*wd) = 0.99838 over epoch 0's LR
    ramp. That must PASS; an init that never reached the model must not."""
    _write_smoke_tree(tmp_path, init_drift=0.0074)      # the measured value
    res = _run_checker(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr

    _write_smoke_tree(tmp_path / "broken", break_init=True)
    broken = _run_checker(tmp_path / "broken")
    assert broken.returncode == 1
    assert "the init reached the model" in broken.stdout


def test_checker_fails_a_thawed_const_arm(tmp_path):
    _write_smoke_tree(tmp_path, break_const=True)
    res = _run_checker(tmp_path)
    assert res.returncode == 1
    assert "BIT-IDENTICAL" in res.stdout


def test_checker_fails_an_arm_on_its_own_schedule(tmp_path):
    _write_smoke_tree(tmp_path, break_config=True)
    res = _run_checker(tmp_path)
    assert res.returncode == 1
    assert "train.epochs" in res.stdout


def test_checker_fails_on_a_missing_run(tmp_path):
    _write_smoke_tree(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / ARMS[3])
    res = _run_checker(tmp_path)
    assert res.returncode == 1
    assert ARMS[3] in res.stdout


# ─────────────────────────────────────────────────────────────────────────────
# fixtures for the data-order test (tiny fake ImageNet, CPU)
# ─────────────────────────────────────────────────────────────────────────────

IMG, N_CLASSES = 96, 4


def _fake_imagenet(root: Path, n_per_class=2):
    from PIL import Image
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        for c in range(N_CLASSES):
            d = root / split / f"n{c:08d}"
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = rng.randint(0, 255, (IMG, IMG, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"im{i}_n{c:08d}.JPEG")
    return root


def _tiny_two_arm_matrix(tmp_path: Path, data_root: Path) -> Path:
    """Two arms that differ ONLY in gate_mode — the A3.4 configuration."""
    matrix = {
        "config_dir": (REPO / "classification" / "configs").as_posix(),
        "overrides": {
            "model": {"img_size": IMG, "num_classes": N_CLASSES},
            "train": {"epochs": 1, "batch_size": 2, "warmup_epochs": 1,
                      "amp": False},
            "data": {"input_size": IMG, "num_workers": 0, "pin_memory": False},
            "augmentation": {"rand_aug": False, "random_erase_prob": 0.0},
            "logging": {"log_freq": 1000},
        },
        "defaults": {"diag_freq": 100,
                     "diag_split": (tmp_path / "nope.json").as_posix(),
                     "diag_n_effrank": 8},
        "runs": {
            "tinyA": {"arch": "vit_tiny_patch16_224", "recipe": "mixup",
                      "variant": "saga", "seed": 3, "gate_mode": "headscalar"},
            "tinyB": {"arch": "vit_tiny_patch16_224", "recipe": "mixup",
                      "variant": "saga", "seed": 3, "gate_mode": "spatial"},
            "tinyConst": {"arch": "vit_tiny_patch16_224", "recipe": "mixup",
                          "variant": "saga", "seed": 3, "gate_mode": "const"},
            "tinyLS": {"arch": "vit_tiny_patch16_224", "recipe": "mixup",
                       "variant": "saga", "seed": 3,
                       "gate_mode": "layerscale"},
        },
    }
    p = tmp_path / "matrix.yaml"
    p.write_text(yaml.safe_dump(matrix))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# end to end through the REAL trainer: timm's create_optimizer_v2, the grad
# scaler and the clip path all see these gates for the first time
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_const_arm_never_learns(tmp_path):
    data_root = _fake_imagenet(tmp_path / "data")
    m = _tiny_two_arm_matrix(tmp_path, data_root)
    run_dir = trainer.run_training(m, "tinyConst", data_root,
                                   out_root=tmp_path / "runs",
                                   resume="auto", max_epochs=2,
                                   device_str="cpu")
    dumps = sorted((run_dir / "gates").glob("phi_e*.npz"))
    assert len(dumps) == 2
    phis = [np.load(p)["phi"] for p in dumps]
    assert phis[0].shape == (12, 3, 36)
    assert np.array_equal(phis[0], phis[1])
    assert np.all(phis[0] == 0.0)
    # the rest of the model DID train
    rows = list(csv.DictReader(open(run_dir / "log.csv", newline="")))
    assert len(rows) == 2
    ckpt = torch.load(run_dir / "ckpt" / "last.pth", map_location="cpu",
                      weights_only=False)
    assert torch.all(ckpt["model"]["blocks.0.attn.gate.phi"] == 0.0)


def test_ckpt_root_moves_only_the_checkpoints(tmp_path):
    """--ckpt_root keeps multi-GB checkpoints off the code filesystem while
    every small artifact stays under out_root, where sync_results.sh globs
    for it. Resume must follow, and meta.json must record where they went."""
    data_root = _fake_imagenet(tmp_path / "data")
    m = _tiny_two_arm_matrix(tmp_path, data_root)
    out_root, ckpt_root = tmp_path / "runs", tmp_path / "bulk"

    run_dir = trainer.run_training(m, "tinyA", data_root, out_root=out_root,
                                   resume="auto", max_epochs=1,
                                   device_str="cpu", ckpt_root=ckpt_root)
    moved = ckpt_root / "tinyA" / "ckpt"
    assert (moved / "last.pth").exists() and (moved / "best.pth").exists()
    assert not (run_dir / "ckpt").exists(), "no ckpt/ in the run dir"
    for name in ("meta.json", "config.resolved.yaml", "log.csv"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "gates").is_dir()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["ckpt_dir"] == moved.as_posix()

    # and --resume auto finds it there (epoch 1 continues, does not restart)
    trainer.run_training(m, "tinyA", data_root, out_root=out_root,
                         resume="auto", max_epochs=2, device_str="cpu",
                         ckpt_root=ckpt_root)
    rows = list(csv.DictReader(open(run_dir / "log.csv", newline="")))
    assert [int(r["epoch"]) for r in rows] == [0, 1]
    ckpt = torch.load(moved / "last.pth", map_location="cpu",
                      weights_only=False)
    assert ckpt["epoch"] == 1


def test_ckpt_root_default_is_unchanged(tmp_path):
    data_root = _fake_imagenet(tmp_path / "data")
    m = _tiny_two_arm_matrix(tmp_path, data_root)
    run_dir = trainer.run_training(m, "tinyA", data_root,
                                   out_root=tmp_path / "runs",
                                   resume="auto", max_epochs=1,
                                   device_str="cpu")
    assert (run_dir / "ckpt" / "last.pth").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["ckpt_dir"] == (run_dir / "ckpt").as_posix()


def test_posix_ckpt_root_is_not_rebased_under_the_repo():
    """TASK-09's lesson: '/home/woody/...' is not is_absolute() on Windows,
    so a naive Path() would silently put HPC checkpoints inside the repo."""
    from tools.dense_runtime import repo_path
    assert repo_path("/home/woody/iwi5/iwi5359h/SAGA/abl_ckpt").as_posix() \
        == "/home/woody/iwi5/iwi5359h/SAGA/abl_ckpt"


def test_end_to_end_layerscale_arm_trains_and_dumps_no_phi(tmp_path):
    data_root = _fake_imagenet(tmp_path / "data")
    m = _tiny_two_arm_matrix(tmp_path, data_root)
    run_dir = trainer.run_training(m, "tinyLS", data_root,
                                   out_root=tmp_path / "runs",
                                   resume="auto", max_epochs=1,
                                   device_str="cpu")
    assert not (run_dir / "gates").exists()
    ckpt = torch.load(run_dir / "ckpt" / "last.pth", map_location="cpu",
                      weights_only=False)
    gamma = ckpt["model"]["blocks.0.attn.gate.gamma"]
    assert gamma.shape == (3, 64)
    assert not torch.allclose(
        gamma, torch.full_like(gamma, LAYERSCALE_INIT_DEIT3)), \
        "LayerScale gamma must actually receive gradient"
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert cfg["model"]["gate_mode"] == "layerscale"
    assert cfg["model"]["layerscale_init"] == LAYERSCALE_INIT_DEIT3


def test_matrix_wide_overrides_apply_to_every_run(tmp_path):
    data_root = _fake_imagenet(tmp_path / "data")
    m = _tiny_two_arm_matrix(tmp_path, data_root)
    for run in ("tinyA", "tinyB"):
        cfg = trainer.resolve_run_config(m, run)
        assert cfg["train"]["epochs"] == 1 and cfg["data"]["num_workers"] == 0
