"""
tests/test_I0_frozen.py
=======================
TASK I0 D7 — the frozen-intervention framework, on CPU with tiny fake models
and fake data.

Two model shapes are exercised everywhere it matters:
  - a tiny SAGA ViT (1 prefix token, a SpatialGate in every block)
  - a tiny timm 4-REGISTER ViT (5 prefix tokens), built DIRECTLY and never
    wrapped in SAGAViT, which refuses num_prefix_tokens != 1 by design

The register model is not decoration: a patch intervention that slid one row
would be invisible on a CLS-only model and would silently invalidate every
register comparison (plan §13.3).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from saga.frozen import records as rec

from saga.frozen.edits import (DIHEDRAL_OPS, EditError, build_edited_gate_map,
                               check_mask, frozen_state, gate_edit, gate_map,
                               receiver_perturbation, ring_of, state_hash,
                               terminal_gate_override)
from saga.frozen.stages import (HIST_STAGE, HIST_STAGE_CITATION, STAGES,
                                StageError, capture_stages,
                                forward_with_stages, stage_block_index)
from saga.metrics import compute_diagnostics, infer_num_prefix_tokens
from saga.vit import build_saga_vit

REPO = Path(__file__).resolve().parents[1]

DEPTH = 4
DIM = 24
HEADS = 3
N_PATCHES = 196


def _install_structured_gate(model, seed=0):
    """Give every gate a SPATIALLY STRUCTURED phi, deterministically.

    A freshly built SAGA gate has phi = 0, so sigmoid(phi) = 0.5 at every
    position and EVERY permutation of it is a no-op — a permutation test on
    an untrained gate would pass without testing anything. The trained
    checkpoints have real spatial structure; these tests need the same
    property, so it is written in here as fake data (no training, no
    optimizer: this is `torch.no_grad()` assignment of a fixed pattern).
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for blk in model.blocks:
            phi = blk.attn.gate.phi
            phi.copy_(torch.randn(phi.shape, generator=g) * 1.5)
    return model


@pytest.fixture(scope="module")
def saga_model():
    torch.manual_seed(0)
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=10,
                           depth=DEPTH, embed_dim=DIM, num_heads=HEADS).eval()
    return _install_structured_gate(model)


@pytest.fixture(scope="module")
def baseline_model():
    torch.manual_seed(0)
    return build_saga_vit("vit_tiny_patch16_224", gate=False, num_classes=10,
                          depth=DEPTH, embed_dim=DIM, num_heads=HEADS).eval()


@pytest.fixture(scope="module")
def reg4_model():
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=10, reg_tokens=4, depth=DEPTH,
                             embed_dim=DIM, num_heads=HEADS).eval()


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(1)
    return torch.randn(3, 3, 224, 224)


# ── stages ──────────────────────────────────────────────────────────────────

def test_hist_stage_matches_the_historical_hook(saga_model, images):
    """The `hist` alias must be the stage saga/metrics.py actually uses.

    Not an assertion about a constant: the real `compute_diagnostics` runs
    here, and the patch tensor it would have measured is compared against
    `capture_stages`' output at HIST_STAGE. If someone moved the historical
    hook past the final norm, this test fails and the alias is wrong.
    """
    captured = {}
    blocks = saga_model.blocks
    handle = blocks[-1].register_forward_hook(
        lambda m, i, out: captured.__setitem__("block_out", out.detach()))
    try:
        with torch.no_grad():
            saga_model(images)
    finally:
        handle.remove()

    p = infer_num_prefix_tokens(saga_model)
    historical_patch = captured["block_out"][:, p:, :].float()

    _, store = forward_with_stages(saga_model, images, (HIST_STAGE, "hist"))
    assert torch.equal(store[HIST_STAGE], historical_patch)
    assert torch.equal(store["hist"], historical_patch)

    # and it is NOT the post-norm stage (the thing the alias could be wrong
    # about); the final LayerNorm is not the identity on these tensors
    _, store2 = forward_with_stages(saga_model, images, ("s12_post_norm",))
    assert not torch.allclose(store2["s12_post_norm"], historical_patch)

    assert HIST_STAGE == "s12_pre_norm"
    assert "saga/metrics.py" in HIST_STAGE_CITATION
    assert "tools/diagnose.py" in HIST_STAGE_CITATION


def test_compute_diagnostics_still_runs_on_the_same_model(saga_model, images):
    """The historical diagnostics path is untouched by this task: it still
    runs and still reports the model's own prefix count."""
    loader = [(images, torch.zeros(images.shape[0], dtype=torch.long))]
    out = compute_diagnostics(saga_model, loader, torch.device("cpu"),
                              with_attn=False, n_effrank=1)
    assert out["num_prefix_tokens"] == infer_num_prefix_tokens(saga_model) == 1
    assert out["block_idx"] == -1
    assert out["n_images"] == images.shape[0]


@pytest.mark.parametrize("model_name", ["saga_model", "baseline_model",
                                        "reg4_model"])
def test_every_stage_returns_patch_rows_only(model_name, images, request):
    """Smoke check 8: [B, N - n_prefix, d] at every stage, on all three
    model shapes — 1 prefix token for saga/baseline, 5 for reg4."""
    model = request.getfixturevalue(model_name)
    p = infer_num_prefix_tokens(model)
    seen = {}
    handle = model.blocks[-1].register_forward_hook(
        lambda m, i, out: seen.__setitem__("n", out.shape[1]))
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()

    logits, store = forward_with_stages(model, images, STAGES)
    expect = seen["n"] - p
    assert expect == N_PATCHES
    for stage in STAGES:
        assert store[stage].shape == (images.shape[0], expect, DIM), stage
        assert store[stage].dtype == torch.float32
    assert logits.shape == (images.shape[0], 10)


def test_reg4_has_five_prefix_tokens_and_is_not_wrapped(reg4_model):
    assert infer_num_prefix_tokens(reg4_model) == 5
    from saga.vit import SAGAViT
    with pytest.raises(ValueError, match="single-prefix-token"):
        SAGAViT(reg4_model)


def test_s11_out_is_block_index_10_on_a_12_block_model():
    """The stage names are named for the 12-block production models; the
    indices are derived from depth so a tiny model works too. Pin the
    equality that makes the two descriptions the same thing."""
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=2,
                           depth=12, embed_dim=DIM, num_heads=HEADS)
    assert len(model.blocks) == 12
    assert stage_block_index(model, "s11_out") == 10
    assert stage_block_index(model, "s12_pre_norm") == 11
    assert stage_block_index(model, "hist") == 11


def test_unknown_stage_is_refused(saga_model):
    with pytest.raises(StageError, match="unknown stage"):
        with capture_stages(saga_model, ("s13_out",)):
            pass
    with pytest.raises(StageError, match="not a block output"):
        stage_block_index(saga_model, "s12_post_norm")


def test_hooks_are_removed_after_the_block(saga_model, images):
    before = [len(b._forward_hooks) for b in saga_model.blocks]
    with capture_stages(saga_model, STAGES):
        with torch.no_grad():
            saga_model(images)
    assert [len(b._forward_hooks) for b in saga_model.blocks] == before
    assert len(saga_model.norm._forward_hooks) == 0


# ── state hashing ───────────────────────────────────────────────────────────

def test_state_hash_is_deterministic_and_sensitive(saga_model):
    h1 = state_hash(saga_model)
    assert h1 == state_hash(saga_model)
    with torch.no_grad():
        saga_model.blocks[0].attn.gate.phi[0, 0] += 1e-4
    assert state_hash(saga_model) != h1
    with torch.no_grad():
        saga_model.blocks[0].attn.gate.phi[0, 0] -= 1e-4
    assert state_hash(saga_model) == h1


def test_frozen_state_raises_when_state_is_not_restored(saga_model):
    with pytest.raises(EditError, match="NOT restored"):
        with frozen_state(saga_model, "deliberate"):
            with torch.no_grad():
                saga_model.blocks[0].attn.gate.phi.add_(1.0)
    with torch.no_grad():                       # put it back for later tests
        saga_model.blocks[0].attn.gate.phi.sub_(1.0)


# ── terminal gate override (I2) ─────────────────────────────────────────────

@pytest.mark.parametrize("value", [0.25, 0.5, 0.75, 1.0])
def test_terminal_gate_leaves_cls_logits_unchanged(saga_model, images, value):
    """Smoke check 3 / plan §5.6 Proposition 2: a CLS-only readout cannot see
    the terminal PATCH gate."""
    with torch.no_grad():
        native = saga_model(images).float()
    with terminal_gate_override(saga_model, value) as info:
        with torch.no_grad():
            edited = saga_model(images).float()
    assert info["layer"] == len(saga_model.blocks) - 1
    assert info["n_prefix"] == 1
    assert (edited - native).abs().max().item() < 1e-4


def test_terminal_gate_actually_changes_the_patch_tokens(saga_model, images):
    """The invariance above must not come from an edit that did nothing."""
    _, native = forward_with_stages(saga_model, images, (HIST_STAGE,))
    with terminal_gate_override(saga_model, 0.25):
        _, edited = forward_with_stages(saga_model, images, (HIST_STAGE,))
    assert not torch.allclose(edited[HIST_STAGE], native[HIST_STAGE])


def test_terminal_gate_refuses_a_model_without_a_gate(baseline_model,
                                                      reg4_model):
    for model in (baseline_model, reg4_model):
        with pytest.raises(EditError, match="no gate|no spatial map"):
            with terminal_gate_override(model, 0.5):
                pass


def test_terminal_gate_refuses_an_undeclared_value(saga_model):
    with pytest.raises(EditError, match="not one of the declared"):
        with terminal_gate_override(saga_model, 0.6):
            pass


def test_terminal_gate_restores_state(saga_model, images):
    before = state_hash(saga_model)
    with terminal_gate_override(saga_model, 0.5):
        with torch.no_grad():
            saga_model(images)
    assert state_hash(saga_model) == before


# ── gate edits (I3) ─────────────────────────────────────────────────────────

def test_identity_gate_edit_is_bit_exact(saga_model, images):
    """Smoke check 2: mode='original' rebuilds the gate's own map and applies
    it through the replacement module, so bit equality also proves the
    replacement path is exact."""
    with torch.no_grad():
        native = saga_model(images).float()
    with gate_edit(saga_model, DEPTH - 1, "original"):
        with torch.no_grad():
            edited = saga_model(images).float()
    assert torch.equal(edited, native)


def test_mean_collapse_keeps_per_head_means(saga_model):
    layer = 1
    g = gate_map(saga_model.blocks[layer].attn.gate, N_PATCHES)
    edited = build_edited_gate_map(g, "mean")
    assert torch.allclose(edited.mean(dim=1), g.mean(dim=1))
    assert torch.allclose(edited.std(dim=1), torch.zeros(HEADS), atol=1e-7)


def test_mean_plus_alpha_delta_interpolates(saga_model):
    g = gate_map(saga_model.blocks[1].attn.gate, N_PATCHES)
    mu = g.mean(dim=1, keepdim=True)
    for alpha in (0.0, 0.5, 1.0):
        edited = build_edited_gate_map(g, "mean_plus_alpha_delta", alpha=alpha)
        assert torch.allclose(edited, mu + alpha * (g - mu), atol=1e-6)
    assert torch.allclose(
        build_edited_gate_map(g, "mean_plus_alpha_delta", alpha=0.0),
        build_edited_gate_map(g, "mean"), atol=1e-6)
    with pytest.raises(EditError, match="requires alpha"):
        build_edited_gate_map(g, "mean_plus_alpha_delta")


def test_permutation_preserves_means_and_histograms(saga_model):
    """Smoke check 5: a permutation is a relabelling — per-head mean AND the
    per-head multiset of values are preserved exactly."""
    g = gate_map(saga_model.blocks[1].attn.gate, N_PATCHES)
    perm = np.random.RandomState(0).permutation(N_PATCHES)
    edited = build_edited_gate_map(g, "permute", perm=perm)
    assert torch.allclose(edited.mean(dim=1), g.mean(dim=1), atol=1e-7)
    assert torch.equal(edited.sort(dim=1).values, g.sort(dim=1).values)
    assert not torch.equal(edited, g)


def test_permutation_is_shared_across_heads(saga_model):
    g = gate_map(saga_model.blocks[1].attn.gate, N_PATCHES)
    perm = np.random.RandomState(3).permutation(N_PATCHES)
    edited = build_edited_gate_map(g, "permute", perm=perm)
    for h in range(g.shape[0]):
        assert torch.equal(edited[h], g[h][torch.as_tensor(perm)])


def test_permute_requires_an_explicit_permutation(saga_model):
    g = gate_map(saga_model.blocks[1].attn.gate, N_PATCHES)
    with pytest.raises(EditError, match="requires an explicit permutation"):
        build_edited_gate_map(g, "permute")
    with pytest.raises(EditError, match="not a bijection"):
        build_edited_gate_map(g, "permute", perm=np.zeros(N_PATCHES, int))


def test_within_ring_permutation_preserves_ring_counts(saga_model):
    """Smoke check 6, and the refusal that makes it meaningful."""
    rings = ring_of(N_PATCHES)
    rng = np.random.RandomState(0)
    perm = np.arange(N_PATCHES)
    for r in np.unique(rings):
        idx = np.flatnonzero(rings == r)
        perm[idx] = idx[rng.permutation(idx.size)]
    assert np.array_equal(rings[perm], rings)

    g = gate_map(saga_model.blocks[1].attn.gate, N_PATCHES)
    edited = build_edited_gate_map(g, "permute_within_ring", perm=perm)
    for r in np.unique(rings):
        idx = torch.as_tensor(np.flatnonzero(rings == r))
        assert torch.equal(edited[:, idx].sort(dim=1).values,
                           g[:, idx].sort(dim=1).values)

    cross = perm.copy()
    a = int(np.flatnonzero(rings == 0)[0])
    b = int(np.flatnonzero(rings == rings.max())[0])
    cross[[a, b]] = cross[[b, a]]
    with pytest.raises(EditError, match="across rings"):
        build_edited_gate_map(g, "permute_within_ring", perm=cross)


def test_ring_definition_is_imported_not_rederived():
    """THE ring definition lives in analysis/address_analysis.py; a second
    derivation could drift from the geometry TASK-07/TASK-13 used."""
    from analysis.address_analysis import border_distance_map, ring_indices
    rings = ring_of(N_PATCHES)
    assert np.array_equal(rings, border_distance_map(14).reshape(-1))
    assert np.array_equal(np.flatnonzero(rings == 1), ring_indices(14, 1))
    assert int((rings == 1).sum()) == 44          # the TASK-07 ring


def test_dihedral_preserves_the_value_multiset(saga_model):
    g = gate_map(saga_model.blocks[1].attn.gate, N_PATCHES)
    seen = set()
    for k in range(len(DIHEDRAL_OPS)):
        edited = build_edited_gate_map(g, "dihedral", perm=k)
        assert torch.equal(edited.sort(dim=1).values, g.sort(dim=1).values)
        seen.add(tuple(edited[0].tolist()))
    assert len(seen) == len(DIHEDRAL_OPS)         # 8 distinct transforms
    assert torch.equal(build_edited_gate_map(g, "dihedral", perm=0), g)
    with pytest.raises(EditError, match="out of range"):
        build_edited_gate_map(g, "dihedral", perm=99)


def test_gate_edit_refuses_unknown_layer_and_gateless_models(saga_model,
                                                             baseline_model,
                                                             reg4_model):
    with pytest.raises(EditError, match="out of range"):
        with gate_edit(saga_model, 99, "mean"):
            pass
    with pytest.raises(EditError, match="out of range"):
        with gate_edit(saga_model, -1, "mean"):
            pass
    for model in (baseline_model, reg4_model):
        with pytest.raises(EditError, match="has no gate"):
            with gate_edit(model, 1, "mean"):
                pass


@pytest.mark.parametrize("mode,kw", [
    ("original", {}), ("mean", {}),
    ("mean_plus_alpha_delta", {"alpha": 0.5}),
    ("permute", {"perm": np.random.RandomState(7).permutation(N_PATCHES)}),
    ("dihedral", {"perm": 5}),
])
def test_gate_edit_restores_state(saga_model, images, mode, kw):
    """Smoke check 9, for every mode."""
    before = state_hash(saga_model)
    with gate_edit(saga_model, 1, mode, **kw):
        with torch.no_grad():
            saga_model(images)
    assert state_hash(saga_model) == before


def test_gate_edit_leaves_prefix_rows_alone(saga_model, images):
    captured = []
    handle = saga_model.blocks[1].register_forward_hook(
        lambda m, i, out: captured.append(out[:, :1, :].detach().clone()))
    try:
        with torch.no_grad():
            saga_model(images)
        native = captured[-1]
        with gate_edit(saga_model, 1, "mean"):
            with torch.no_grad():
                saga_model(images)
        edited = captured[-1]
    finally:
        handle.remove()
    assert torch.equal(edited, native)


# ── receiver perturbation (I4) ──────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["saga_model", "baseline_model",
                                        "reg4_model"])
def test_perturbation_leaves_prefix_rows_bit_identical(model_name, images,
                                                       request):
    """Smoke check 4, on all three shapes — the register model has FIVE
    prefix rows, which is the case this guard exists for."""
    model = request.getfixturevalue(model_name)
    p = infer_num_prefix_tokens(model)
    layer = 1
    captured = []
    handle = model.blocks[layer].register_forward_hook(
        lambda m, i, out: captured.append(out[:, :p, :].detach().clone()))
    try:
        with torch.no_grad():
            model(images)
        native = captured[-1]
        mask = np.array([0, 1, 2, 7, 13, 42, 100, 195])
        with receiver_perturbation(model, layer, mask, 0.25) as rec:
            with torch.no_grad():
                model(images)
        edited = captured[-1]
    finally:
        handle.remove()
    assert rec["n_prefix"] == p
    assert rec["n_masked_coords"] == mask.size
    assert torch.equal(edited, native)
    assert max(rec["measured_perturbation_norm"]) > 0     # it did something


@pytest.mark.parametrize("model_name", ["saga_model", "baseline_model",
                                        "reg4_model"])
def test_perturbation_changes_only_masked_patch_rows(model_name, images,
                                                     request):
    model = request.getfixturevalue(model_name)
    p = infer_num_prefix_tokens(model)
    layer = 1
    captured = []

    def grab(_m, _i, out):
        captured.append(out.detach().clone())

    # NOTE the hook ordering. A forward hook that returns a value REPLACES
    # the output for every hook registered after it, so a capture hook
    # registered BEFORE receiver_perturbation's would see the unedited
    # tensor and this test would pass while measuring nothing. The capture
    # is therefore registered INSIDE the `with`.
    handle = model.blocks[layer].attn.register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(images)
        native = captured[-1]
    finally:
        handle.remove()

    mask = np.array([3, 17, 190])
    with receiver_perturbation(model, layer, mask, 0.5):
        handle = model.blocks[layer].attn.register_forward_hook(grab)
        try:
            with torch.no_grad():
                model(images)
            edited = captured[-1]
        finally:
            handle.remove()

    diff = (edited - native).abs().amax(dim=(0, 2))       # [N]
    changed = torch.nonzero(diff > 0).flatten().tolist()
    assert changed == sorted((mask + p).tolist())


def test_perturbation_scales_the_bias_free_update(baseline_model, images):
    """The edit is (1-alpha)*U + b with U = out - b (plan §5.5), not
    (1-alpha)*out — the projection bias is excluded by definition."""
    layer = 1
    attn = baseline_model.blocks[layer].attn
    bias = attn.proj.bias
    assert bias is not None
    captured = []

    def grab(_m, _i, out):
        captured.append(out.detach().clone())

    handle = attn.register_forward_hook(grab)       # sees the native output
    try:
        with torch.no_grad():
            baseline_model(images)
        native = captured[-1]
    finally:
        handle.remove()

    # registered INSIDE the `with`, so it runs AFTER the edit's hook and sees
    # the replaced output (see the note in the previous test)
    with receiver_perturbation(baseline_model, layer, np.array([5]), 0.5):
        handle = attn.register_forward_hook(grab)
        try:
            with torch.no_grad():
                baseline_model(images)
            edited = captured[-1]
        finally:
            handle.remove()

    row = 5 + infer_num_prefix_tokens(baseline_model)
    u = native[:, row, :] - bias
    assert torch.allclose(edited[:, row, :], 0.5 * u + bias, atol=1e-6)


def test_energy_matching_hits_the_declared_norm(baseline_model, images):
    """Smoke check 7 / plan §5.5."""
    layer = 1
    mask = np.array([0, 5, 60, 195])
    with receiver_perturbation(baseline_model, layer, mask, 0.1) as native_rec:
        with torch.no_grad():
            baseline_model(images)
    native_norms = np.asarray(native_rec["native_masked_norm"])
    kappa = 0.1 * native_norms
    with receiver_perturbation(baseline_model, layer, mask, 0.1,
                               energy_target=kappa.tolist()) as rec:
        with torch.no_grad():
            baseline_model(images)
    achieved = np.asarray(rec["measured_perturbation_norm"])
    assert np.allclose(achieved, kappa, rtol=1e-5)
    assert rec["energy_matched"] is True
    assert rec["n_zero_norm_images"] == 0


def test_energy_matching_handles_a_zero_norm_without_dropping_the_image(
        baseline_model, images):
    """A zero-norm image gets zero perturbation and is COUNTED — never
    silently discarded (plan §5.5)."""
    layer = 1
    n = images.shape[0]
    with receiver_perturbation(baseline_model, layer, np.array([7]), 0.1,
                               energy_target=[0.0] * n) as rec:
        with torch.no_grad():
            baseline_model(images)
    assert len(rec["measured_perturbation_norm"]) == n
    assert all(v == pytest.approx(0.0) for v in rec["alpha"])


def test_perturbation_refuses_a_token_length_mask(reg4_model, images):
    """The mask-length guard: 201 entries is the TOKEN count, and a mask
    written against token indices would perturb a register row."""
    with pytest.raises(EditError, match="TOKEN count"):
        with receiver_perturbation(reg4_model, 1,
                                   np.zeros(201, dtype=bool), 0.1):
            with torch.no_grad():
                reg4_model(images)


def test_perturbation_refuses_out_of_range_and_negative_indices(reg4_model,
                                                                images):
    for mask, match in ((np.array([196]), "outside the 196 patch"),
                        (np.array([-1]), "negative")):
        with pytest.raises(EditError, match=match):
            with receiver_perturbation(reg4_model, 1, mask, 0.1):
                with torch.no_grad():
                    reg4_model(images)


def test_perturbation_refuses_unknown_layer_and_bad_epsilon(baseline_model):
    with pytest.raises(EditError, match="out of range"):
        with receiver_perturbation(baseline_model, 99, np.array([0]), 0.1):
            pass
    with pytest.raises(EditError, match=r"epsilon must lie in"):
        with receiver_perturbation(baseline_model, 1, np.array([0]), 1.5):
            pass


@pytest.mark.parametrize("model_name", ["saga_model", "baseline_model",
                                        "reg4_model"])
def test_perturbation_restores_state(model_name, images, request):
    model = request.getfixturevalue(model_name)
    before = state_hash(model)
    with receiver_perturbation(model, 1, np.array([0, 5, 195]), 0.1):
        with torch.no_grad():
            model(images)
    assert state_hash(model) == before
    assert len(model.blocks[1].attn._forward_hooks) == 0


def test_check_mask_accepts_boolean_and_index_forms():
    boolean = np.zeros(N_PATCHES, dtype=bool)
    boolean[[3, 9]] = True
    assert np.array_equal(check_mask(boolean, N_PATCHES, 1),
                          check_mask(np.array([3, 9]), N_PATCHES, 1))
    assert check_mask(np.array([], dtype=int), N_PATCHES, 5).sum() == 0


# ── records: schema, append-safety, atomicity ───────────────────────────────

def _prov(**kw):
    base = dict(run_id="r", arch="vit_small", recipe_actual="mixup",
                variant="saga", ckpt_kind="last", ckpt_sha256="a" * 64,
                n_prefix=1, stage=HIST_STAGE, split_name="evaluation",
                split_sha256="b" * 64, precision="fp32", git_sha="c" * 40,
                git_dirty=True)
    base.update(kw)
    return rec.provenance(**base)


def _record_row(image_id, condition_id, **kw):
    row = dict(_prov(), image_id=image_id, condition_id=condition_id,
               edit_type="native", edit_params="{}", layer=rec.MISSING,
               epsilon=rec.MISSING, measured_perturbation_norm=rec.MISSING,
               nll=1.0, correct=1, top1=3, label=3,
               max_abs_logit_diff_vs_native=0.0)
    row.update(kw)
    return row


def test_every_record_row_carries_the_provenance_columns():
    row = _record_row("n01/x", "native")
    for col in rec.PROVENANCE_COLUMNS:
        assert col in row, col
    assert set(row) == set(rec.RECORD_COLUMNS)
    rec.check_rows([row], "records")


def test_a_row_with_a_missing_or_extra_column_is_refused():
    row = _record_row("n01/x", "native")
    del row["nll"]
    with pytest.raises(ValueError, match="missing"):
        rec.check_rows([row], "records")
    row = _record_row("n01/x", "native", )
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        rec.check_rows([row], "records")


def test_append_is_idempotent_on_the_condition_image_key(tmp_path):
    rows = [_record_row(f"n01/{i}", "native") for i in range(4)]
    assert rec.append_rows(tmp_path, "records", rows) == (4, 0)
    # a resubmitted job adds what is missing and rewrites nothing
    assert rec.append_rows(tmp_path, "records", rows) == (0, 4)
    more = rows + [_record_row("n01/9", "native"),
                   _record_row("n01/0", "terminal_gate_0.50")]
    assert rec.append_rows(tmp_path, "records", more) == (2, 4)
    path = rec.records_path(tmp_path, "records")
    if path.suffix == ".csv":
        import csv as _csv
        with open(path, newline="", encoding="utf-8") as f:
            assert len(list(_csv.DictReader(f))) == 6


def test_the_completion_marker_is_written_last_and_is_checkpoint_keyed(
        tmp_path):
    rows = [_record_row("n01/0", "native")]
    rec.append_rows(tmp_path, "records", rows)
    assert not rec.is_done(tmp_path, "records", "a" * 64)
    rec.mark_done(tmp_path, "records", {"ckpt_sha256": "a" * 64})
    assert rec.is_done(tmp_path, "records", "a" * 64)
    assert not rec.is_done(tmp_path, "records", "z" * 64)


def test_no_tmp_files_are_left_behind(tmp_path):
    rec.append_rows(tmp_path, "records", [_record_row("n01/0", "native")])
    rec.mark_done(tmp_path, "records", {"ckpt_sha256": "a" * 64})
    assert not list(tmp_path.glob("*.tmp"))


# ── runner: manifest row, checkpoint verification, conditions ───────────────

def test_the_runner_refuses_a_row_whose_checkpoint_hash_is_missing(tmp_path):
    from saga.frozen.runner import RunnerError, verify_checkpoint
    with pytest.raises(RunnerError, match="frozen_manifest_hashes"):
        verify_checkpoint({"run_id": "r", "ckpt_kind": "last",
                           "ckpt_sha256": "MISSING", "ckpt_path": "x"})


def test_the_runner_refuses_a_checkpoint_that_moved(tmp_path):
    from saga.run_registry import file_sha256
    from saga.frozen.runner import RunnerError, verify_checkpoint
    ckpt = tmp_path / "last.pth"
    ckpt.write_bytes(b"not really a checkpoint")
    row = {"run_id": "r", "ckpt_kind": "last", "ckpt_path": str(ckpt),
           "ckpt_sha256": file_sha256(ckpt)}
    assert verify_checkpoint(row) == row["ckpt_sha256"]
    ckpt.write_bytes(b"something else")
    with pytest.raises(RunnerError, match="refusing to evaluate a checkpoint "
                                          "that moved"):
        verify_checkpoint(row)


def test_the_shipped_smoke_conditions_file_is_valid():
    from saga.frozen.runner import load_conditions
    doc = load_conditions(REPO / "configs" / "frozen" / "smoke.yaml")
    assert doc["work_package"] == "I0_smoke"
    assert doc["stage"] == "hist"
    assert doc["precision"] == "fp32"
    ids = [c["id"] for c in doc["conditions"]]
    assert ids[0] == "native"
    assert len(set(ids)) == len(ids)
    values = {c["params"]["value"] for c in doc["conditions"]
              if c["edit_type"] == "terminal_gate_override"}
    assert values == {0.25, 0.5, 0.75, 1.0}


def test_conditions_with_an_unknown_edit_or_duplicate_id_are_refused(tmp_path):
    import yaml
    from saga.frozen.runner import RunnerError, load_conditions
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "work_package": "X", "stage": "hist",
        "conditions": [{"id": "a", "edit_type": "teleport"}]}))
    with pytest.raises(RunnerError, match="unknown edit_type"):
        load_conditions(p)
    p.write_text(yaml.safe_dump({
        "work_package": "X", "stage": "hist",
        "conditions": [{"id": "a", "edit_type": "native"},
                       {"id": "a", "edit_type": "native"}]}))
    with pytest.raises(RunnerError, match="duplicate condition id"):
        load_conditions(p)
    p.write_text(yaml.safe_dump({
        "work_package": "X", "stage": "s99",
        "conditions": [{"id": "a", "edit_type": "native"}]}))
    with pytest.raises(StageError, match="unknown stage"):
        load_conditions(p)


def test_run_work_package_writes_traceable_rows_and_restores_state(
        saga_model, tmp_path, monkeypatch):
    """The whole loop on CPU with a fake model and fake data: every row
    carries its provenance, the two files share a key, and the model comes
    out of the sweep bit-identical."""
    import saga.frozen.runner as runner
    monkeypatch.setattr(runner, "build_from_row",
                        lambda row, ckpt_path=None, device="cpu": saga_model)

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            torch.manual_seed(i)
            return torch.randn(3, 224, 224), i % 10

    items = [[f"val/n{c:08d}/IMG_{c}.JPEG", c] for c in range(4)]
    conditions = {
        "work_package": "I0_smoke", "stage": "hist", "precision": "fp32",
        "conditions": [
            {"id": "native", "edit_type": "native"},
            {"id": "terminal_gate_0.50", "edit_type": "terminal_gate_override",
             "params": {"value": 0.5}},
        ]}
    row = {"run_id": "fake_run", "arch": "vit_small", "variant": "saga",
           "recipe_actual": "mixup", "ckpt_kind": "last",
           "ckpt_sha256": "a" * 64, "gate_mode": "spatial",
           "git_dirty": 1, "patch_file": "MISSING",
           "ckpt_path": "results/runs/fake_run/ckpt/last.pth"}

    before = state_hash(saga_model)
    summary = runner.run_work_package(
        row=row, conditions=conditions, dataset=FakeDataset(items),
        out_dir=tmp_path, device="cpu", batch_size=2,
        split_name="evaluation", split_sha="b" * 64, git_sha="c" * 40,
        git_dirty=row["git_dirty"], patch_file=row["patch_file"])

    assert summary["state_restored"] is True
    assert state_hash(saga_model) == before
    assert summary["n_records"] == 8            # 4 images x 2 conditions

    import csv as _csv
    with open(rec.records_path(tmp_path, "records"), newline="",
              encoding="utf-8") as f:
        got = list(_csv.DictReader(f))
    assert len(got) == 8
    for r in got:
        assert r["ckpt_sha256"] == "a" * 64
        assert r["split_sha256"] == "b" * 64
        assert r["stage"] == "hist"
        assert r["n_prefix"] == "1"
        assert r["precision"] == "fp32"
        assert r["image_id"].startswith("n0")
        assert float(r["nll"]) > 0
    # the terminal gate leaves the CLS logits alone (Proposition 2), and the
    # runner MEASURES that rather than asserting it
    edited = [r for r in got if r["condition_id"] == "terminal_gate_0.50"]
    assert len(edited) == 4
    assert max(float(r["max_abs_logit_diff_vs_native"]) for r in edited) < 1e-4
    # ... while the patch diagnostics at the same stage DO move
    with open(rec.records_path(tmp_path, "diag"), newline="",
              encoding="utf-8") as f:
        diag = list(_csv.DictReader(f))
    assert len(diag) == 8
    by_cond = {}
    for r in diag:
        by_cond.setdefault(r["condition_id"], {})[r["image_id"]] = \
            float(r["patch_norm_median"])
    assert by_cond["native"] != by_cond["terminal_gate_0.50"]

    marker = json.loads(
        (tmp_path / "records.done.json").read_text(encoding="utf-8"))
    assert marker["ckpt_sha256"] == "a" * 64
    assert marker["hist_stage"] == HIST_STAGE
    assert rec.is_done(tmp_path, "records", "a" * 64)


def test_no_optimizer_or_training_import_in_the_frozen_package():
    """TASK I0 §11: no training, no fine-tuning, no optimizer, no probe
    fitting in any new file.

    Checked on the parsed AST, not on the raw text: several of these files
    say "no optimizer" in their own docstrings, and a substring search would
    flag the very comments that state the rule.
    """
    import ast

    banned_modules = ("torch.optim", "torch.optim.lr_scheduler")
    banned_names = {"AdamW", "Adam", "SGD", "backward", "step",
                    "LabelSmoothingCrossEntropy", "Mixup"}
    targets = list((REPO / "saga" / "frozen").glob("*.py"))
    targets += [REPO / "tools" / "frozen_eval.py",
                REPO / "tools" / "frozen_smoke.py",
                REPO / "tools" / "frozen_manifest_hashes.py",
                REPO / "tools" / "build_frozen_splits.py",
                REPO / "analysis" / "build_I0_manifest.py"]
    assert len(targets) >= 9

    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("torch.optim"), \
                        f"{path.name} imports {a.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod not in banned_modules, \
                    f"{path.name} imports from {mod}"
                for a in node.names:
                    assert a.name not in banned_names, \
                        f"{path.name} imports {a.name} from {mod}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in ("backward", "zero_grad"), \
                    f"{path.name} calls .{node.attr}()"
