"""
tests/test_I3_gate_edits.py
===========================
TASK B / I3 Phase A — the conditions file, the dihedral set, the native
reference, the energy strata and the contrast module, on CPU with tiny fake
models and fake data.

Everything here runs without a GPU, without a checkpoint and without a
dataset. What it pins:

  - the 61 conditions, by family and by layer, and the refusal of anything
    the conditions file does not declare;
  - that the permutations come from `configs/frozen/permutations_14x14.json`
    and from nowhere else, and that the file's digest is what LOCKED D3 says;
  - that `mean` leaves the per-head mean exactly, that the 8 dihedral
    transforms are ring-preserving bijections, and that `dihedral0` is
    bit-identical to `original` through a REAL forward pass;
  - that `delta_update_norm` equals a hand computation, and is exactly 0 on
    the native condition;
  - that the decile stratification is deterministic and exhaustive;
  - that C1 and C2 land in every interpretation branch of TASK B §4 on
    synthetic records, and that the branch TEXT is byte-identical to the task
    file.
"""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from analysis import i34_contrasts as ic
from analysis.frozen_I3_analysis import (assign_decile, build_a, build_c,
                                         decile_edges, scope_of)
from analysis.address_analysis import border_distance_map, ring_indices
from saga.frozen import records as rec
from saga.frozen.edits import (DIHEDRAL_NAMES, DIHEDRAL_OPS, EditError,
                               build_edited_gate_map, dihedral_permutation,
                               gate_edit, gate_map, ring_of)
from saga.frozen.reference import (ReferenceError, UpdateReference,
                                   attention_branch_update, condition_layer,
                                   watched_layers)
from saga.frozen.runner import (RunnerError, condition_diag, condition_stages,
                                conditions_for, eligible_cohort,
                                load_conditions, run_work_package_stages)
from saga.vit import build_saga_vit

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = REPO / "configs" / "frozen" / "I3_gate_edits.yaml"
PERM_FILE = REPO / "configs" / "frozen" / "permutations_14x14.json"
TASK_FILE = REPO / "docs" / "TASK_B_I3_I4.md"
JOB_FILE = REPO / "scripts" / "jobs" / "frozen_I3.sbatch"
MANIFEST = REPO / "results" / "frozen" / "I0_manifest" / "manifest.json"

SIDE = 14
N_POSITIONS = SIDE * SIDE
LAYERS = (7, 8)

DEPTH = 4
DIM = 24
HEADS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _install_structured_gate(model, seed=0):
    """A SPATIALLY STRUCTURED phi, deterministically.

    A freshly built gate has phi = 0, so sigma(phi) = 0.5 everywhere and every
    permutation of it is a no-op — a permutation test on an untrained gate
    would pass without testing anything. No training and no optimizer: this is
    `torch.no_grad()` assignment of a fixed pattern, exactly as
    `tests/test_I0_frozen.py` does it.
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
def images():
    torch.manual_seed(1)
    return torch.randn(3, 3, 224, 224)


@pytest.fixture(scope="module")
def conditions():
    return load_conditions(CONDITIONS)


def _gate_map(seed=0, heads=3, n=N_POSITIONS):
    g = torch.Generator().manual_seed(seed)
    return torch.sigmoid(torch.randn(heads, n, generator=g) * 1.5)


# ─────────────────────────────────────────────────────────────────────────────
# The conditions file — 61, and nothing undeclared
# ─────────────────────────────────────────────────────────────────────────────

def test_the_conditions_file_declares_exactly_61_conditions(conditions):
    """TASK B §3: 1 + 2 x 30. The count is the experiment."""
    conds = conditions["conditions"]
    assert len(conds) == 61
    assert len({c["id"] for c in conds}) == 61
    assert conds[0]["id"] == "original"
    # the runner measures max_abs_logit_diff_vs_native against the FIRST
    # condition applying to the variant, so `original` must be native and first
    assert conds[0]["edit_type"] == "native"
    assert conditions["work_package"] == "I3_gate_edits"
    assert conditions["stage"] == "s11_out"
    assert conditions["precision"] == "fp32"
    assert conditions["bootstrap_seed"] == 0
    assert conditions["bootstrap_resamples"] == 10000
    assert conditions["measure_update_norm"] is True


def test_the_condition_families_are_the_declared_counts(conditions):
    counts = {}
    for c in conditions["conditions"]:
        if c["id"] == "original":
            continue
        parsed = ic.parse_condition_id(c["id"])
        assert parsed is not None, c["id"]
        counts[(parsed["family"], parsed["layer"])] = \
            counts.get((parsed["family"], parsed["layer"]), 0) + 1
    for layer in LAYERS:
        assert counts[("mean", layer)] == 1
        assert counts[("mean_half", layer)] == 1
        assert counts[("perm", layer)] == 10
        assert counts[("ringperm", layer)] == 10
        assert counts[("dihedral", layer)] == 8
    assert sum(counts.values()) == 60
    assert {k[1] for k in counts} == set(LAYERS)


def test_every_condition_is_saga_only_and_edits_one_layer(conditions):
    """A gate edit needs a gate; a baseline or a register model is refused
    rather than treated as 'already flat'. And no condition edits two layers:
    joint edits are explicitly out of scope (TASK B §2)."""
    for c in conditions["conditions"]:
        assert c["applies_to"] == ["saga"], c["id"]
        layer = condition_layer(c)
        if c["id"] == "original":
            assert layer is None
        else:
            assert layer in LAYERS, c["id"]
    assert len(conditions_for(conditions, "saga")) == 61
    with pytest.raises(RunnerError, match="no declared condition applies"):
        conditions_for(conditions, "baseline")
    assert watched_layers(conditions) == LAYERS


def test_only_three_conditions_record_the_patch_diagnostics(conditions):
    """TASK B §3: `original`, `mean_L7`, `mean_L8` and nothing else."""
    recording = [c["id"] for c in conditions["conditions"]
                 if condition_diag(conditions, c)]
    assert recording == ["original", "mean_L7", "mean_L8"]
    for c in conditions["conditions"]:
        if condition_diag(conditions, c):
            assert condition_stages(conditions, c) == ("s11_out",), c["id"]
        else:
            assert "stages" not in c, c["id"]


def test_an_undeclared_condition_is_refused(tmp_path):
    """The conditions contract: a sweep runs what the file declares and
    nothing else, and a document naming an unknown edit is a parse error on
    the login node rather than a surprise on a GPU."""
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["conditions"].append({"id": "improvised", "edit_type": "sharpen"})
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RunnerError, match="unknown edit_type"):
        load_conditions(p)


def test_a_duplicate_condition_id_is_refused(tmp_path):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["conditions"].append(dict(doc["conditions"][3]))
    p = tmp_path / "dup.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RunnerError, match="duplicate condition id"):
        load_conditions(p)


def test_a_non_recording_condition_may_not_declare_a_stage(tmp_path):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    for c in doc["conditions"]:
        if c["id"] == "perm0_L7":
            c["stages"] = ["s11_out"]
    p = tmp_path / "lying.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RunnerError, match="declare `stages` and"):
        load_conditions(p)


def test_a_non_boolean_diag_flag_is_refused(tmp_path):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["conditions"][5]["diag"] = "sometimes"
    p = tmp_path / "diag.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RunnerError, match="expected true or false"):
        load_conditions(p)


# ─────────────────────────────────────────────────────────────────────────────
# Permutations: from the committed file, never drawn
# ─────────────────────────────────────────────────────────────────────────────

def test_every_permutation_comes_from_the_committed_file(conditions):
    """The 20 permutation conditions resolve to the 20 committed lists — not
    to something equal to them, to THOSE, element by element."""
    doc = json.loads(PERM_FILE.read_text(encoding="utf-8"))
    n = 0
    for c in conditions["conditions"]:
        spec = (c.get("params") or {}).get("perm")
        if not isinstance(spec, dict):
            continue
        n += 1
        want = np.asarray(doc[spec["source"]][spec["index"]], dtype=np.int64)
        assert np.array_equal(c["_perm"], want), c["id"]
        assert len(c["_perm"]) == N_POSITIONS
    assert n == 40, f"{n} permutation conditions, expected 2 layers x 20"


def test_the_resolved_permutation_stays_out_of_the_recorded_params(conditions):
    """`edit_params` is written on every one of 610,000 record rows. The
    declared {source, index} goes there; the 196 indices do not."""
    c = [x for x in conditions["conditions"] if x["id"] == "perm3_L7"][0]
    blob = json.dumps(c["params"], sort_keys=True, separators=(",", ":"),
                      default=str)
    assert blob == '{"layer":7,"mode":"permute","perm":{"index":3,"source":"position"}}'
    assert len(blob) < 100


def test_the_permutations_digest_is_the_one_LOCKED_D3_closed(conditions):
    digest = hashlib.sha256(
        PERM_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert conditions["permutations_sha256"] == digest
    locked = (REPO / "docs" / "LOCKED_ANALYSIS.md").read_text(encoding="utf-8")
    assert digest in locked


def test_a_permutation_reference_without_a_file_is_refused(tmp_path):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    del doc["permutations_file"]
    p = tmp_path / "nofile.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RunnerError, match="declares no `permutations_file`"):
        load_conditions(p)


def test_an_out_of_range_permutation_index_is_refused(tmp_path):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    for c in doc["conditions"]:
        if c["id"] == "perm0_L7":
            c["params"]["perm"] = {"source": "position", "index": 10}
    p = tmp_path / "oob.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RunnerError, match="holds 10 of them"):
        load_conditions(p)


def test_an_unknown_permutation_source_is_refused(tmp_path):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    for c in doc["conditions"]:
        if c["id"] == "perm0_L7":
            c["params"]["perm"] = {"source": "whatever", "index": 0}
    p = tmp_path / "src.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(RunnerError, match="permutation source"):
        load_conditions(p)


# ─────────────────────────────────────────────────────────────────────────────
# The edits themselves
# ─────────────────────────────────────────────────────────────────────────────

def test_mean_leaves_the_per_head_mean_exactly():
    """`mean` replaces the arrangement and keeps mu_h. The per-head mean of
    the edited map is the constant it was built from, exactly."""
    g = _gate_map()
    edited = build_edited_gate_map(g, "mean")
    mu = g.mean(dim=1, keepdim=True)
    assert torch.equal(edited, mu.expand_as(g).contiguous())
    for h in range(g.shape[0]):
        assert torch.equal(edited[h], mu[h].expand(g.shape[1]))
        # exactly flat: max == min. NOT `std() == 0` — the variance of 196
        # identical fp32 values is 6e-08, an artifact of the reduction, not a
        # property of the edit.
        assert float(edited[h].amax() - edited[h].amin()) == 0.0


def test_mean_half_is_mu_plus_half_delta():
    g = _gate_map()
    edited = build_edited_gate_map(g, "mean_plus_alpha_delta", alpha=0.5)
    mu = g.mean(dim=1, keepdim=True)
    assert torch.allclose(edited, mu + 0.5 * (g - mu), rtol=0, atol=0)
    # halfway: the spatial std is halved, per head
    for h in range(g.shape[0]):
        assert float(edited[h].std()) == pytest.approx(
            float(g[h].std()) * 0.5, rel=1e-6)


@pytest.mark.parametrize("t", range(8))
def test_every_dihedral_transform_is_a_ring_preserving_bijection(t):
    """TASK B §3/§8: the 8 symmetries of the 14x14 grid are bijections that
    move no coordinate across a Chebyshev ring. Checked against
    `analysis.address_analysis.ring_indices` — THE ring definition — and not
    against a second derivation here."""
    idx = dihedral_permutation(t, N_POSITIONS)
    assert idx.shape == (N_POSITIONS,)
    assert np.array_equal(np.sort(idx), np.arange(N_POSITIONS))
    rings = border_distance_map(SIDE).reshape(-1)
    assert np.array_equal(rings[idx], rings), DIHEDRAL_NAMES[t]
    assert np.array_equal(rings, ring_of(N_POSITIONS))
    # and the ring MEMBERSHIP is permuted within itself, not merely preserved
    # as a count
    for k in range(SIDE // 2):
        members = ring_indices(SIDE, k)
        assert set(idx[members].tolist()) == set(members.tolist())


def test_dihedral_index_0_is_the_identity_permutation():
    assert DIHEDRAL_OPS[0] == (0, False)
    assert DIHEDRAL_NAMES[0] == "identity"
    assert np.array_equal(dihedral_permutation(0, N_POSITIONS),
                          np.arange(N_POSITIONS))
    g = _gate_map()
    assert torch.equal(build_edited_gate_map(g, "dihedral", perm=0), g)


def test_the_eight_transforms_are_distinct_and_only_one_is_the_identity():
    seen = {tuple(dihedral_permutation(t, N_POSITIONS).tolist())
            for t in range(8)}
    assert len(seen) == 8
    identity = tuple(range(N_POSITIONS))
    assert sum(1 for s in seen if s == identity) == 1


def test_an_out_of_range_dihedral_index_is_refused():
    with pytest.raises(EditError, match="out of range"):
        dihedral_permutation(8, N_POSITIONS)
    with pytest.raises(EditError, match="not a square grid"):
        dihedral_permutation(1, 195)


def test_dihedral0_is_bit_identical_to_original_through_a_real_forward(
        saga_model, images):
    """TASK B §7, the reference-condition identity — asserted on LOGITS from a
    real forward pass, not on the gate map.

    `original` is the unedited model through its own SpatialGate;
    `dihedral0` is the identity transform of the same map pushed through the
    REPLACEMENT module. They must agree bit for bit, which is what makes
    `dihedral0` the bit-exactness control LOCKED §3 item 1 asks for.
    """
    with torch.no_grad():
        native = saga_model(images).float()
        for layer in (1, 2):
            with gate_edit(saga_model, layer, "dihedral", perm=0):
                edited = saga_model(images).float()
            assert torch.equal(native, edited), f"layer {layer}"
            assert float((native - edited).abs().max()) == 0.0


def test_a_non_identity_dihedral_actually_moves_the_logits(saga_model, images):
    """Otherwise the test above would pass on an edit that did nothing."""
    with torch.no_grad():
        native = saga_model(images).float()
        moved = 0
        for t in range(1, 8):
            with gate_edit(saga_model, 1, "dihedral", perm=t):
                edited = saga_model(images).float()
            if float((native - edited).abs().max()) > 0:
                moved += 1
    assert moved == 7


def test_gate_edit_still_refuses_to_draw_a_permutation():
    g = _gate_map()
    for mode in ("permute", "permute_within_ring"):
        with pytest.raises(EditError, match="requires an explicit"):
            build_edited_gate_map(g, mode, perm=None)


# ─────────────────────────────────────────────────────────────────────────────
# delta_update_norm
# ─────────────────────────────────────────────────────────────────────────────

def test_the_native_update_is_the_blocks_own_expression(saga_model, images):
    """`attention_branch_update` must reproduce what the block ADDS to its
    input, bit for bit — otherwise every delta is measured against three
    quarters of an expression.

    Captured from a real forward with a hook on `drop_path1` and recomputed
    from the block's input, then compared as TENSORS.
    """
    blk = saga_model.blocks[1]
    cap = {}
    h_in = blk.register_forward_pre_hook(
        lambda m, args: cap.__setitem__("x", args[0]))
    h_out = blk.drop_path1.register_forward_hook(
        lambda m, i, out: cap.__setitem__("u", out))
    try:
        with torch.no_grad():
            saga_model(images)
    finally:
        h_in.remove()
        h_out.remove()
    with torch.no_grad():
        recomputed = attention_branch_update(blk, cap["x"])
    assert torch.equal(recomputed, cap["u"])


def test_delta_update_norm_is_exactly_zero_on_the_native_condition(
        saga_model, images):
    with UpdateReference(saga_model, layers=(1, 2)) as ref:
        with torch.no_grad():
            saga_model(images)
        assert ref.delta_update_norm(1, images.shape[0]) == [0.0] * 3
        assert ref.delta_update_norm(2, images.shape[0]) == [0.0] * 3
    # and the native CONDITION (layer None) is zero without measuring
    with UpdateReference(saga_model, layers=(1,)) as ref:
        assert ref.delta_update_norm(None, 4) == [0.0] * 4


def test_delta_update_norm_is_zero_under_dihedral0(saga_model, images):
    """The identity edit injects no energy. If this were nonzero, every
    decile edge would be built on an offset."""
    with UpdateReference(saga_model, layers=(1,)) as ref:
        with gate_edit(saga_model, 1, "dihedral", perm=0):
            with torch.no_grad():
                saga_model(images)
        assert ref.delta_update_norm(1, 3) == [0.0, 0.0, 0.0]


def test_delta_update_norm_equals_a_hand_computation(saga_model, images):
    """TASK B §8: verified against a hand computation on a fake model.

    The hand computation captures the block input under the NATIVE forward,
    evaluates the attention branch twice — once with the gate as it is, once
    under the edit — and takes the per-image Frobenius norm of the difference
    over the patch rows. No part of `reference.py` is used to produce it.
    """
    layer, n_prefix = 1, 1
    blk = saga_model.blocks[layer]

    with UpdateReference(saga_model, layers=(layer,)) as ref:
        with gate_edit(saga_model, layer, "mean"):
            with torch.no_grad():
                saga_model(images)
        measured = ref.delta_update_norm(layer, images.shape[0])

    cap = {}
    h = blk.register_forward_pre_hook(
        lambda m, args: cap.__setitem__("x", args[0]))
    try:
        with torch.no_grad():
            saga_model(images)
    finally:
        h.remove()
    x = cap["x"]
    with torch.no_grad():
        u_native = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
        with gate_edit(saga_model, layer, "mean"):
            u_edit = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
    d = (u_edit[:, n_prefix:, :] - u_native[:, n_prefix:, :]).float()
    hand = d.flatten(1).norm(dim=1).tolist()

    assert measured == hand
    assert all(v > 0 for v in hand), "the mean collapse must inject energy"


def test_delta_update_norm_is_measured_at_the_layer_the_condition_edits(
        saga_model, images):
    """An edit at layer 1 changes the layer-1 update; it also changes layer 2,
    because layer 2's INPUT has moved. The column is defined at the EDITED
    block, and this pins which one that is."""
    with UpdateReference(saga_model, layers=(1, 2)) as ref:
        with gate_edit(saga_model, 1, "mean"):
            with torch.no_grad():
                saga_model(images)
        at_1 = ref.delta_update_norm(1, 3)
        at_2 = ref.delta_update_norm(2, 3)
    assert all(v > 0 for v in at_1)
    assert at_1 != at_2


def test_an_unwatched_layer_is_refused(saga_model, images):
    with UpdateReference(saga_model, layers=(1,)) as ref:
        with torch.no_grad():
            saga_model(images)
        with pytest.raises(ReferenceError, match="is not watched"):
            ref.delta_update_norm(2, 3)


def test_a_measurement_without_a_forward_is_refused(saga_model):
    with UpdateReference(saga_model, layers=(1,)) as ref:
        ref.clear()
        with pytest.raises(ReferenceError, match="no forward pass"):
            ref.delta_update_norm(1, 3)


def test_the_reference_removes_its_hooks(saga_model, images):
    blk = saga_model.blocks[1]
    before = len(blk._forward_pre_hooks), len(blk.drop_path1._forward_hooks)
    with UpdateReference(saga_model, layers=(1,)):
        during = (len(blk._forward_pre_hooks),
                  len(blk.drop_path1._forward_hooks))
    after = len(blk._forward_pre_hooks), len(blk.drop_path1._forward_hooks)
    assert during == (before[0] + 1, before[1] + 1)
    assert after == before


# ─────────────────────────────────────────────────────────────────────────────
# The runner, end to end on a fake model
# ─────────────────────────────────────────────────────────────────────────────

class _FakeSet(torch.utils.data.Dataset):
    def __init__(self, n=4):
        torch.manual_seed(3)
        self.x = torch.randn(n, 3, 224, 224)
        self.y = torch.randint(0, 10, (n,))
        self.items = [[f"n{i:08d}/img_{i}.JPEG", int(self.y[i])]
                      for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _small_conditions(tmp_path, ids):
    """The real conditions file, cut down to `ids` — the same parse, the same
    resolution, a handful of forwards."""
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["conditions"] = [c for c in doc["conditions"] if c["id"] in ids]
    # a 4-block fake model has no block 7 or 8
    for c in doc["conditions"]:
        if (c.get("params") or {}).get("layer") is not None:
            c["params"]["layer"] = 1
    p = tmp_path / "small.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return load_conditions(p)


@pytest.fixture
def fake_row(tmp_path, saga_model):
    ckpt = tmp_path / "last.pth"
    torch.save({"model": saga_model.state_dict()}, ckpt)
    from saga.run_registry import file_sha256
    return {"run_id": "fake_saga", "arch": "vit_small",
            "recipe_actual": "mixup", "variant": "saga", "ckpt_kind": "last",
            "ckpt_sha256": file_sha256(ckpt), "ckpt_path": str(ckpt),
            "gate_mode": "spatial"}


def test_the_sweep_records_the_new_columns_and_restores_the_model(
        tmp_path, saga_model, fake_row, monkeypatch):
    """The whole loop on a fake model: the new `delta_update_norm` and
    `permutations_sha256` columns land on every record row, the diagnostics
    land only for the conditions that declare them, and the model is
    bit-identical afterwards."""
    conds = _small_conditions(
        tmp_path, {"original", "mean_L7", "perm0_L7", "dihedral0_L7"})
    monkeypatch.setattr("saga.frozen.runner.build_from_row",
                        lambda row, ckpt_path=None, device="cpu": saga_model)
    out = tmp_path / "out"
    summary = run_work_package_stages(
        row=fake_row, conditions=conds, dataset=_FakeSet(), out_dir=out,
        split_name="fake", split_sha="d" * 64, git_sha="e" * 40,
        git_dirty="0", batch_size=2,
        tau_cal_doc={"split_sha256": "d" * 64,
                     "tau_cal": {"vit_small|mixup": {"s11_out": 1.0}}},
        canon_doc={})

    assert summary["state_restored"] is True
    assert summary["measure_update_norm"] is True
    assert summary["update_reference"]["layers"] == [1]
    assert summary["update_reference"]["tail_module"] == {"1": "drop_path1"}

    rows = _read(out, "records")
    assert set(rows[0]) == set(rec.RECORD_UPDATE_COLUMNS)
    assert len(rows) == 4 * 4                     # 4 conditions x 4 images
    by = {}
    for r in rows:
        by.setdefault(r["condition_id"], []).append(r)
    assert set(by) == {"original", "mean_L7", "perm0_L7", "dihedral0_L7"}
    for cond, got in by.items():
        for r in got:
            assert r["permutations_sha256"] == conds["permutations_sha256"]
    # the reference conditions inject nothing; the real edits do
    for cond in ("original", "dihedral0_L7"):
        assert [float(r["delta_update_norm"]) for r in by[cond]] == [0.0] * 4
    for cond in ("mean_L7", "perm0_L7"):
        assert all(float(r["delta_update_norm"]) > 0 for r in by[cond])

    # diag rows ONLY for the two conditions that declare them
    diag = _read(out, "diag")
    assert {r["condition_id"] for r in diag} == {"original", "mean_L7"}

    marker = json.loads((out / "records.done.json").read_text(encoding="utf-8"))
    assert marker["permutations_sha256"] == conds["permutations_sha256"]
    assert marker["split_sha256"] == "d" * 64


def _read(out_dir, kind):
    import csv
    path = rec.records_path(out_dir, kind)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_a_sweep_without_the_flag_keeps_the_old_record_schema(
        tmp_path, saga_model, fake_row, monkeypatch):
    """I1's and I2's conditions files must run exactly as they did: the two
    new columns appear only when a document asks for them."""
    conds = _small_conditions(tmp_path, {"original", "mean_L7"})
    conds["measure_update_norm"] = False
    monkeypatch.setattr("saga.frozen.runner.build_from_row",
                        lambda row, ckpt_path=None, device="cpu": saga_model)
    out = tmp_path / "out2"
    run_work_package_stages(
        row=fake_row, conditions=conds, dataset=_FakeSet(2), out_dir=out,
        split_name="fake", split_sha="d" * 64, git_sha="e" * 40,
        git_dirty="0", batch_size=2,
        tau_cal_doc={"split_sha256": "d" * 64,
                     "tau_cal": {"vit_small|mixup": {"s11_out": 1.0}}},
        canon_doc={})
    rows = _read(out, "records")
    assert set(rows[0]) == set(rec.RECORD_COLUMNS)
    assert "delta_update_norm" not in rows[0]


# ─────────────────────────────────────────────────────────────────────────────
# The energy strata
# ─────────────────────────────────────────────────────────────────────────────

def test_the_decile_edges_are_deterministic_and_exhaustive():
    rng = np.random.RandomState(0)
    vals = rng.lognormal(size=997)
    e1, e2 = decile_edges(vals), decile_edges(vals)
    assert np.array_equal(e1, e2)
    assert np.array_equal(e1, decile_edges(rng.permutation(vals)))
    assert e1[0] == -np.inf and e1[-1] == np.inf
    assert len(e1) == 11
    bins = assign_decile(vals, e1)
    assert bins.min() == 0 and bins.max() == 9
    assert len(bins) == len(vals)
    # every observation lands in exactly one bin, and the bins are balanced
    counts = np.bincount(bins, minlength=10)
    assert counts.sum() == vals.size
    assert counts.min() >= vals.size // 10 - 2


def test_the_deciles_are_pooled_over_conditions_not_per_family(tmp_path):
    """TASK B §3: ONE energy axis per checkpoint. Per-family deciles would put
    every family's own median in bin 5 and make the comparison vacuous."""
    records = _synthetic_records("fake_saga", c1_shift=0.3, c2_shift=0.1,
                                 n=60, energy={"mean": 5.0, "mean_half": 2.5,
                                               "perm": 1.0, "ringperm": 0.5,
                                               "dihedral": 0.2})
    rows = build_c("fake_saga", records, resamples=200)
    # the low-energy families cannot reach the top decile of a pooled axis
    top = {r["family"] for r in rows if r["decile"] == 9}
    assert "mean" in top
    assert "dihedral" not in top
    assert {r["family"] for r in rows} <= set(
        ("mean", "mean_half", "perm", "ringperm", "dihedral"))
    assert all(r["n_pairs"] > 0 for r in rows)


def test_the_strata_refuse_a_sweep_with_no_energy_column():
    records = _synthetic_records("fake_saga", 0.3, 0.1, n=20, energy=None)
    with pytest.raises(Exception, match="measure_update_norm"):
        build_c("fake_saga", records, resamples=50)


# ─────────────────────────────────────────────────────────────────────────────
# C1 / C2 and the interpretation branches
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_records(run_id, c1_shift, c2_shift, n=200, seed=0,
                       energy=None, arch="vit_small", recipe="mixup"):
    """Fake I3 record rows: 61 conditions x n images.

    `energy` maps a family to the `delta_update_norm` it injects; None leaves
    the column MISSING, which is what a sweep run without the flag looks like.
    """
    rng = np.random.RandomState(seed)
    ids = [f"n{i // 10:08d}/img_{i}" for i in range(n)]
    base = rng.rand(n) * 2.0
    rows = []

    def add(cond, shift, family):
        for i, img in enumerate(ids):
            nll = base[i] + shift + rng.randn() * 0.01
            rows.append({
                "run_id": run_id, "arch": arch, "recipe_actual": recipe,
                "variant": "saga", "ckpt_kind": "last", "ckpt_sha256": "a" * 64,
                "split_name": "evaluation", "split_sha256": "b" * 64,
                "stage": "s11_out", "precision": "fp32", "git_sha": "c" * 40,
                "permutations_sha256": "d" * 64, "condition_id": cond,
                "image_id": img, "nll": nll,
                "correct": 1 if nll < 1.5 else 0,
                "max_abs_logit_diff_vs_native": 0.0 if family is None else 0.5,
                "delta_update_norm": ("MISSING" if energy is None or
                                      family is None
                                      else energy[family] * (1 + 0.3 * rng.rand())),
            })

    add("original", 0.0, None)
    for layer in LAYERS:
        add(f"mean_L{layer}", c1_shift, "mean")
        add(f"mean_half_L{layer}", c1_shift / 2, "mean_half")
        for k in range(10):
            add(f"perm{k}_L{layer}", c2_shift, "perm")
        for k in range(10):
            add(f"ringperm{k}_L{layer}", c2_shift * 0.5, "ringperm")
        for t in range(8):
            add(f"dihedral{t}_L{layer}", 0.0, "dihedral")
    return rows


def _decide(c1_shift, c2_shift, layer=7, resamples=400):
    recs = {r: _synthetic_records(r, c1_shift, c2_shift, seed=s)
            for s, r in enumerate(ic.FRESH_SAGA)}
    return ic.i3_contrasts(recs, layers=(layer,), resamples=resamples)[layer]


@pytest.mark.parametrize("c1_shift,c2_shift,branch", [
    (0.0, 0.0, "neither"),
    (0.25, 0.0, "ring_only"),
    (0.25, 0.12, "arrangement"),
    (-0.25, 0.0, "unanticipated"),
    (0.0, 0.12, "unanticipated"),
])
def test_every_interpretation_branch_is_reachable(c1_shift, c2_shift, branch):
    """TASK B §4's three branches, plus the pattern it does not name. The
    module decides; nothing is re-framed by inspection."""
    got = _decide(c1_shift, c2_shift)
    assert got["interpretation"]["branch"] == branch, (
        got["C1"]["verdict"], got["C2"]["verdict"])
    assert got["interpretation"]["anticipated"] is (branch != "unanticipated")


def test_the_branch_text_is_byte_identical_to_the_task_file():
    """The interpretation guide was fixed before any result existed. If a
    branch is ever softened, the task file has to be edited and the diff
    shows it."""
    text = TASK_FILE.read_text(encoding="utf-8")
    for key in ("neither", "ring_only", "arrangement"):
        assert ic.I3_BRANCHES[key] in text, key
    assert ic.I3_GUIDE_CLOSE in text
    # the unanticipated branch is NOT in the task file, and says so
    assert ic.I3_BRANCHES["unanticipated"] not in text
    assert "TASK B §4 names" in ic.I3_BRANCHES["unanticipated"]


def test_only_the_fresh_pair_decides():
    """LOCKED §10.1. A legacy checkpoint with a huge effect changes no
    verdict; its row is still in the output."""
    recs = {r: _synthetic_records(r, 0.0, 0.0, seed=s)
            for s, r in enumerate(ic.FRESH_SAGA)}
    recs["legacy_e2_vit_small_mixupdir_saga"] = _synthetic_records(
        "legacy_e2_vit_small_mixupdir_saga", 5.0, 5.0, seed=9)
    got = ic.i3_contrasts(recs, layers=(7,), resamples=400)[7]
    assert got["C1"]["verdict"] == "null"
    assert got["interpretation"]["branch"] == "neither"
    assert got["C1"]["reported_not_deciding"] == [
        "legacy_e2_vit_small_mixupdir_saga"]
    assert "legacy_e2_vit_small_mixupdir_saga" in got["C1"]["per_checkpoint"]


def test_a_missing_fresh_checkpoint_is_insufficient_not_null():
    """An absent checkpoint is an ABSENCE. Calling it a null would report a
    measurement that was never made."""
    recs = {"legacy_e2_vit_small_mixupdir_saga": _synthetic_records(
        "legacy_e2_vit_small_mixupdir_saga", 0.3, 0.1)}
    got = ic.i3_contrasts(recs, layers=(7,), resamples=200)[7]
    assert got["C1"]["verdict"] == "insufficient"
    assert got["C1"]["missing_deciding"] == list(ic.FRESH_SAGA)
    assert got["interpretation"]["branch"] == "unanticipated"


def test_a_direction_disagreement_between_the_fresh_pair_is_a_null():
    a, b = ic.FRESH_SAGA
    recs = {a: _synthetic_records(a, 0.30, 0.0, seed=0),
            b: _synthetic_records(b, -0.30, 0.0, seed=1)}
    got = ic.i3_contrasts(recs, layers=(7,), resamples=400)[7]
    assert got["C1"]["verdict"] == "null"
    assert "direction is not consistent" in got["C1"]["reason"]


def test_c2_excludes_the_identity_dihedral():
    """`dihedral0` is bit-identical to `original`, so its delta is exactly 0.
    Including it would shrink C2 by 1/8 for a reason unrelated to
    arrangement."""
    recs = _synthetic_records("x", 0.2, 0.1)
    grouped = ic.by_condition(recs)
    out = ic.c2(grouped, 7, resamples=200)
    assert out["n_perm"] == 10 and out["n_dihedral"] == 7
    assert "dihedral0_L7" not in out["reference"]
    assert "dihedral1_L7" in out["reference"]


def test_c2_refuses_a_partial_condition_list():
    recs = [r for r in _synthetic_records("x", 0.2, 0.1)
            if r["condition_id"] not in ("perm9_L7", "dihedral3_L7")]
    grouped = ic.by_condition(recs)
    with pytest.raises(ic.ContrastError, match="found 9 and 6"):
        ic.c2(grouped, 7, resamples=50)


def test_duplicate_rows_are_refused():
    recs = _synthetic_records("x", 0.2, 0.1, n=5)
    with pytest.raises(ic.ContrastError, match="two rows for image"):
        ic.by_condition(recs + recs[:1])


def test_c3_belongs_to_i4_and_is_no_longer_a_stub():
    """Through I3 Phase A this asserted `NotImplementedError("C3 is I4 Phase
    A′")`: C3 reads columns and mask ids D5 had not defined, and a
    substitute contract would have been a guess written into the decision
    module. I4 Phase A′ implemented it, so the assertion is inverted rather
    than deleted — the phase boundary was real and this records that it has
    been crossed.

    C3 is exercised in `tests/test_I4_perturbation.py`; what belongs HERE is
    only that I3's two contrasts do not depend on it.
    """
    assert not isinstance(ic.c3, type(NotImplemented))
    with pytest.raises(TypeError):
        ic.c3()                       # it now REQUIRES records and a layer
    # and I3's own contrasts still stand entirely on their own
    recs = _synthetic_records("x", 0.2, 0.1, n=40)
    grouped = ic.by_condition(recs)
    assert ic.c1(grouped, 7, resamples=100)["contrast"] == "C1"
    assert ic.c2(grouped, 7, resamples=100)["contrast"] == "C2"


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def test_scope_labels_the_four_vits_mixup_checkpoints_and_the_rest():
    assert scope_of("e2r_vits_mixup_saga_s1", "vit_small", "mixup") == "fresh"
    assert scope_of("e2r_vits_mixup_saga_s2", "vit_small", "mixup") == "fresh"
    assert scope_of("legacy_e2_vit_small_mixupdir_saga", "vit_small",
                    "mixup") == "legacy"
    assert scope_of("e2r_vitb_mixup_saga_s1", "vit_base",
                    "mixup") == "exploratory"
    assert scope_of("e2r_vits_nomix_saga_s1", "vit_small",
                    "nomix") == "exploratory"


def test_T_I3a_covers_every_edited_condition_and_carries_the_energy():
    records = _synthetic_records("e2r_vits_mixup_saga_s1", 0.3, 0.1, n=40,
                                 energy={"mean": 5.0, "mean_half": 2.5,
                                         "perm": 1.0, "ringperm": 0.5,
                                         "dihedral": 0.2})
    rows = build_a("e2r_vits_mixup_saga_s1", records, resamples=200)
    assert len(rows) == 60
    assert all(r["scope"] == "fresh" for r in rows)
    assert all(r["mean_delta_update_norm"] != "MISSING" for r in rows)
    fam = {r["family"] for r in rows}
    assert fam == {"mean", "mean_half", "perm", "ringperm", "dihedral"}
    assert {r["layer"] for r in rows} == set(LAYERS)


def test_the_tables_use_descriptive_wording_only(tmp_path):
    """TASK B §10: no functional language. A Delta-NLL is a Delta-NLL."""
    from analysis.frozen_I3_analysis import (T_A_FIELDS, T_C_FIELDS,
                                             write_csv)
    records = _synthetic_records("e2r_vits_mixup_saga_s1", 0.3, 0.1, n=30,
                                 energy={"mean": 5.0, "mean_half": 2.5,
                                         "perm": 1.0, "ringperm": 0.5,
                                         "dihedral": 0.2})
    write_csv(tmp_path / "a.csv", T_A_FIELDS,
              build_a("e2r_vits_mixup_saga_s1", records, resamples=100))
    write_csv(tmp_path / "c.csv", T_C_FIELDS,
              build_c("e2r_vits_mixup_saga_s1", records, resamples=100))
    banned = ("relocat", "empties", "emptying", "sink function",
              "sink_function", "the model needs", "needs the")
    for path in sorted(tmp_path.glob("*.csv")):
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{path.name} says {word!r}"


def test_the_new_source_files_use_descriptive_wording_only():
    banned = ("relocat", "empties", "emptying", "sink function",
              "sink_function", "the model needs")
    for name in ("saga/frozen/reference.py", "analysis/i34_contrasts.py",
                 "analysis/frozen_I3_analysis.py",
                 "configs/frozen/I3_gate_edits.yaml",
                 "scripts/jobs/frozen_I3.sbatch"):
        text = (REPO / name).read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{name} says {word!r}"


# ─────────────────────────────────────────────────────────────────────────────
# The job file
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not MANIFEST.exists(),
                    reason="the I0 manifest is not in this checkout")
def test_the_job_file_lists_the_saga_cohort_in_array_order():
    """An array index must mean the same run in the job file and in the
    manifest. The job file checks this at run time too; this is the check
    that fails in CI instead of on a node."""
    want = [r["run_id"] for r in eligible_cohort(str(MANIFEST))
            if r["variant"] == "saga"]
    text = JOB_FILE.read_text(encoding="utf-8")
    block = text.split("RUN_IDS=(", 1)[1].split("\n)", 1)[0]
    have = [line.split('"')[1] for line in block.strip().splitlines()
            if '"' in line]
    assert have == want, f"job file {have}\nmanifest {want}"
    assert len(have) == 8


def test_the_job_file_fixes_the_evaluation_split_and_takes_no_split_argument():
    """TASK B §2/§6: nothing in Track B runs on discovery or calibration."""
    text = JOB_FILE.read_text(encoding="utf-8")
    assert "SPLIT=results/frozen/splits/evaluation.json" in text
    assert "results/frozen/splits/calibration.json" not in text
    assert "val_diag_split.json" not in text
    assert '${1:?' not in text, "the split must not be a positional argument"
    assert "--array=0-7" in text
    assert "configs/frozen/I3_gate_edits.yaml" in text
    # and it verifies the permutations digest before staging anything
    assert "permutations sha256" in text
    assert "LOCKED_ANALYSIS.md" in text


def test_the_job_file_does_not_recalibrate_thresholds():
    """tau_cal[s11_out] is CONSUMED from I2. Running the threshold tool with
    I3's conditions file would write a second calibration into I2's
    directory."""
    body = [line for line in JOB_FILE.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")]
    assert not [line for line in body if "frozen_I2_thresholds" in line], \
        "the job file INVOKES the threshold tool"
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    assert "calibration_stages" not in doc
    assert doc["thresholds_cal"] == \
        "results/frozen/I2_terminal/{split_name}/thresholds_cal.json"


# ─────────────────────────────────────────────────────────────────────────────
# The sampling guard, from the other side
# ─────────────────────────────────────────────────────────────────────────────

def test_the_sampling_guard_would_catch_a_planted_draw(tmp_path):
    """A guard that has never been shown to fail is not a guard. This plants
    each banned form and asserts the detector finds it."""
    from tests.test_I0_frozen import _sampling_sites

    for snippet, needle in (
            ("import numpy as np\ndef f():\n    return np.random.rand(3)\n",
             "np.random.rand"),
            ("import torch\ndef f():\n    return torch.randperm(4)\n",
             "torch.randperm"),
            ("import random\ndef f():\n    return random.shuffle([1])\n",
             "random.shuffle"),
            ("import numpy\ndef f():\n"
             "    return numpy.random.RandomState(0).permutation(4)\n",
             "numpy.random.RandomState")):
        p = tmp_path / "planted.py"
        p.write_text(snippet, encoding="utf-8")
        sites = _sampling_sites(p)
        assert [s[1] for s in sites] == [needle], snippet
        assert sites[0][0] == "f"
    # and a clean file yields nothing
    p = tmp_path / "clean.py"
    p.write_text("import numpy as np\ndef f():\n    return np.zeros(3)\n",
                 encoding="utf-8")
    assert _sampling_sites(p) == []
    assert ast.parse(p.read_text(encoding="utf-8")) is not None
