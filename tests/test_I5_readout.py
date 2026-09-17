"""
tests/test_I5_readout.py
========================
TASK C / I5 §8 Phase A — the correspondence readout, on CPU with tiny fake
models and fake data.

Three model shapes are exercised where it matters:
  - a tiny SAGA ViT (1 prefix token, a SpatialGate in every block)
  - a tiny plain ViT (1 prefix token, no gate)
  - a tiny timm 4-REGISTER ViT (5 prefix tokens), built DIRECTLY and never
    wrapped in SAGAViT, which refuses num_prefix_tokens != 1 by design

The register model is not decoration here either: every descriptor and every
correspondence table is indexed on PATCH rows, and a prefix row that slipped
into the matcher would shift the whole answer key by one position and
silently make every accuracy wrong.

The models are `vit_tiny_patch16_224` at depth 4, so the patch grid is the
REAL 14x14 — the tables under test are the tables that run on the cluster.
"""

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from saga.frozen import features as feat
from saga.frozen import records as rec
from saga.frozen import transforms as TR
from saga.frozen.correspondence import (DESCRIPTORS, CorrespondenceError,
                                        descriptor, exceedance_flags, match,
                                        split_by_exceedance)
from saga.frozen.edits import state_hash
from saga.frozen.runner import load_conditions
from saga.frozen.stages import capture_stages
from saga.metrics import infer_num_prefix_tokens, sink_counts_mad, token_norms
from saga.vit import build_saga_vit

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = REPO / "configs" / "frozen" / "I5_readout.yaml"

DEPTH = 4
DIM = 24
HEADS = 3
GRID = 14
N_PATCHES = GRID * GRID

#: The shared-set sizes task file §3 quotes, and the reason they are what
#: they are. Asserted against the DERIVED tables, never used to build them.
EXPECTED_SHARED = {"T0": 196, "T1": 196, "T2": 182, "T3": 168}


def _structured_gate(model, seed=0):
    """A spatially structured phi, so the terminal-gate bypass is not a no-op.

    A freshly built SAGA gate has phi = 0, hence sigmoid(phi) = 0.5
    everywhere, and an override to 1.0 would still change the output — but a
    STRUCTURED map is what the trained checkpoints have, and it is what makes
    `term_1.00` move the later stages by different amounts at different
    positions. No training, no optimizer: `torch.no_grad()` assignment of a
    fixed pattern.
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
    return _structured_gate(model)


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


def _fake_split(tmp_path, n=4):
    """n fake JPEGs in an ImageNet-shaped tree, plus the split's item list.

    Deliberately NON-SQUARE (260 x 300): the T2/T3 pipeline resizes to a
    square before cropping, and a square source would hide a bug in that
    resize.
    """
    from PIL import Image
    items = []
    for i in range(n):
        rel = f"val/n0000000{i % 2}/img_{i}.JPEG"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = (np.random.RandomState(i).rand(260, 300, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(path)
        items.append([rel, i % 2])
    return items


def _fake_row(variant="saga"):
    return {"run_id": f"fake_{variant}", "arch": "vit_tiny",
            "recipe_actual": "mixup", "variant": variant, "ckpt_kind": "last",
            "ckpt_sha256": "a" * 64, "git_dirty": 0, "patch_file": "MISSING"}


# ── the correspondence tables ────────────────────────────────────────────────

@pytest.mark.parametrize("tid", ["T0", "T1", "T2", "T3"])
def test_every_correspondence_table_is_a_bijection_of_the_declared_size(tid):
    """§3: 196 / 182 / 168, and a one-to-one map on the shared set.

    The sizes are ASSERTED against tables derived from the geometry, so this
    fails if the derivation drifts — it is not a restatement of a constant
    the code also reads.
    """
    t_idx, o_idx = TR.correspondence(tid, GRID)
    assert t_idx.size == EXPECTED_SHARED[tid]
    assert TR.n_shared(tid, GRID) == EXPECTED_SHARED[tid]
    assert TR.is_bijection(t_idx, o_idx, GRID)
    assert len(set(t_idx.tolist())) == t_idx.size
    assert len(set(o_idx.tolist())) == o_idx.size


def test_t1_is_the_horizontal_flip_it_claims_to_be():
    """(r, c) <-> (r, G-1-c), checked position by position."""
    t_idx, o_idx = TR.correspondence("T1", GRID)
    tr, tc = TR.row_col(t_idx, GRID)
    orr, oc = TR.row_col(o_idx, GRID)
    assert np.array_equal(tr, orr)
    assert np.array_equal(oc, GRID - 1 - tc)


@pytest.mark.parametrize("tid,dx", [("T2", 1), ("T3", 2)])
def test_translation_tables_shift_by_exactly_the_declared_patch_count(tid, dx):
    """The right crop's column c' is the left crop's column c' + dx, and the
    rows never move."""
    t_idx, o_idx = TR.correspondence(tid, GRID)
    tr, tc = TR.row_col(t_idx, GRID)
    orr, oc = TR.row_col(o_idx, GRID)
    assert np.array_equal(tr, orr)
    assert np.array_equal(oc, tc + dx)
    assert tc.max() == GRID - 1 - dx
    assert TR.transform_spec(tid).resize == 224 + dx * 16


@pytest.mark.parametrize("tid", ["T1", "T2", "T3"])
@pytest.mark.parametrize("kind", list(DESCRIPTORS))
def test_the_matcher_scores_100_percent_on_an_image_with_unique_patches(
        tid, kind):
    """§8: the answer key, end to end, on a synthetic image.

    The "transformed" descriptors are the original's, REARRANGED by the
    table itself. If the table and the matcher disagree about which index
    means what, this is not 1.0 — which is exactly the failure a
    hand-tabulated correspondence would produce.
    """
    g = torch.Generator().manual_seed(3)
    original = torch.randn(2, N_PATCHES, DIM, generator=g)
    t_idx, o_idx = TR.correspondence(tid, GRID)
    transformed = torch.zeros_like(original)
    transformed[:, t_idx, :] = original[:, o_idx, :]

    result = match(original, transformed, tid, kind, grid=GRID)
    assert result["n_shared"] == EXPECTED_SHARED[tid]
    assert np.allclose(result["acc_exact"], 1.0)
    assert np.allclose(result["acc_1"], 1.0)


def test_a_shuffled_image_scores_near_chance_not_near_one():
    """The 100% test above would also pass a matcher that ignored its input
    and returned the key. This one cannot: the transformed patches are a
    RANDOM permutation, so the correspondence is wrong by construction."""
    g = torch.Generator().manual_seed(11)
    original = torch.randn(8, N_PATCHES, DIM, generator=g)
    perm = torch.randperm(N_PATCHES, generator=g)
    transformed = original[:, perm, :]
    result = match(original, transformed, "T1", "l2", grid=GRID)
    assert result["acc_exact"].mean() < 0.15


def test_chance_is_computed_and_below_the_quoted_bound():
    """§3: chance = 1/196 exact, and at most 9/196 for the 1-tolerance."""
    assert TR.chance_exact(GRID) == pytest.approx(1.0 / 196)
    for tid in ("T1", "T2", "T3"):
        c1 = TR.chance_within_1(tid, GRID)
        assert 0 < c1 <= 9.0 / 196 + 1e-12
        # strictly below the bound, because the shared set touches the border
        assert c1 < 9.0 / 196


def test_an_undeclared_transform_or_descriptor_is_refused():
    """§11: no transform without exact grid correspondence, no extra
    descriptor. The refusal is the contract, so it is tested."""
    with pytest.raises(TR.TransformError):
        TR.correspondence("rotate90", GRID)
    with pytest.raises(TR.TransformError):
        TR.transform_spec("T4")
    with pytest.raises(CorrespondenceError):
        descriptor(torch.zeros(1, N_PATCHES, DIM), "cosine_whitened")


def test_a_non_square_token_count_is_refused():
    with pytest.raises(TR.TransformError):
        TR.grid_size_for(197)


# ── descriptors ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", list(DESCRIPTORS))
def test_descriptors_are_unit_norm(kind):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(3, N_PATCHES, DIM, generator=g)
    d = descriptor(x, kind)
    assert torch.allclose(d.norm(dim=-1), torch.ones(3, N_PATCHES), atol=1e-5)


def test_the_centred_descriptor_actually_subtracts_the_image_mean():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(2, N_PATCHES, DIM, generator=g)
    centred = x - x.mean(dim=1, keepdim=True)
    expect = centred / centred.norm(dim=-1, keepdim=True)
    assert torch.allclose(descriptor(x, "centred_l2"), expect, atol=1e-6)
    # and it is NOT the same as the raw descriptor, or the secondary
    # descriptor would be measuring the primary one
    assert not torch.allclose(descriptor(x, "l2"),
                              descriptor(x, "centred_l2"), atol=1e-3)


def test_a_zero_norm_descriptor_row_is_zero_not_nan():
    """A patch equal to the image mean centres to zero. It must not become a
    NaN that an argmax would then resolve arbitrarily."""
    x = torch.ones(1, N_PATCHES, DIM)
    d = descriptor(x, "centred_l2")
    assert torch.isfinite(d).all()
    assert float(d.abs().max()) == 0.0


def test_primary_descriptor_is_first_and_is_l2():
    """D9 names `l2` primary. Several tables report DESCRIPTORS[0]."""
    assert DESCRIPTORS[0] == "l2"


# ── MAD exceedance, on the fly ───────────────────────────────────────────────

def test_on_the_fly_mad_flags_equal_saga_metrics_exactly():
    """§8: the flags must be the metric the paper reports, not a lookalike.

    Compared against `saga.metrics.sink_counts_mad` itself — the function
    every historical sink count in this project came from — so the readout
    cannot silently use a different threshold from the diagnostics it is put
    beside.
    """
    g = torch.Generator().manual_seed(7)
    for scale in (1.0, 17.0):
        patches = torch.randn(6, N_PATCHES, DIM, generator=g) * scale
        flags = exceedance_flags(patches)
        counts = sink_counts_mad(token_norms(patches), k=5.0)
        assert torch.equal(flags.sum(dim=1).float(), counts)


def test_exceedance_split_reports_missing_for_an_empty_half():
    """I0 handoff §8.2: MISSING is a value. An image with no exceedance
    position has no accuracy there — not a zero."""
    g = torch.Generator().manual_seed(8)
    original = torch.randn(2, N_PATCHES, DIM, generator=g)
    t_idx, o_idx = TR.correspondence("T1", GRID)
    transformed = torch.zeros_like(original)
    transformed[:, t_idx, :] = original[:, o_idx, :]
    result = match(original, transformed, "T1", "l2", grid=GRID)

    none_exceed = torch.zeros(2, N_PATCHES, dtype=torch.bool)
    split = split_by_exceedance(result, none_exceed)
    assert list(split["n_shared_exc"]) == [0, 0]
    assert split["acc_exact_exc"] == ["MISSING", "MISSING"]
    assert split["acc_exact_nonexc"] == [1.0, 1.0]

    all_exceed = torch.ones(2, N_PATCHES, dtype=torch.bool)
    split = split_by_exceedance(result, all_exceed)
    assert split["acc_exact_nonexc"] == ["MISSING", "MISSING"]
    assert split["acc_exact_exc"] == [1.0, 1.0]


def test_the_split_halves_add_up_to_the_whole():
    """The two halves partition the shared set; nothing is double counted
    and nothing is dropped."""
    g = torch.Generator().manual_seed(9)
    original = torch.randn(4, N_PATCHES, DIM, generator=g)
    transformed = torch.randn(4, N_PATCHES, DIM, generator=g)
    result = match(original, transformed, "T2", "l2", grid=GRID)
    exc = exceedance_flags(original)
    split = split_by_exceedance(result, exc)
    total = np.asarray(split["n_shared_exc"]) + \
        np.asarray(split["n_shared_nonexc"])
    assert (total == result["n_shared"]).all()


# ── prefix rows ──────────────────────────────────────────────────────────────

def test_prefix_rows_are_excluded_by_the_models_own_count(reg4_model, images,
                                                          saga_model):
    """A 5-prefix register model and a 1-prefix SAGA model must both yield
    exactly 196 patch rows into the matcher.

    This is the bug that would be invisible: a hard-coded 1 on the register
    model leaves 4 register rows at the FRONT of the token sequence, every
    patch index shifts by 4, and every accuracy in the table is wrong while
    nothing raises.
    """
    assert infer_num_prefix_tokens(reg4_model) == 5
    assert infer_num_prefix_tokens(saga_model) == 1
    for model in (reg4_model, saga_model):
        with torch.no_grad(), capture_stages(model, ("s11_out",)) as store:
            model(images)
            patches = store["s11_out"]
        assert patches.shape[1] == N_PATCHES
        assert TR.grid_size_for(int(patches.shape[1])) == GRID


def test_the_readout_runs_on_a_five_prefix_register_model(reg4_model, images):
    """End to end on the register model: the answer key applies unchanged."""
    with torch.no_grad(), capture_stages(reg4_model, ("s11_out",)) as store:
        reg4_model(images)
        patches = store["s11_out"].clone()
    result = match(patches, patches, "T1", "l2")
    assert result["n_shared"] == 196
    assert np.isfinite(result["acc_exact"]).all()


# ── the two controls (§7) ────────────────────────────────────────────────────

def test_t0_reproduces_the_plain_forward_bit_for_bit(saga_model, tmp_path):
    """§7: T0 IS the evaluation forward — not almost.

    Compared tensor against tensor at every declared stage, with
    `torch.equal`, not a tolerance.
    """
    items = _fake_split(tmp_path)
    ds = feat.CorrespondencePairDataset(tmp_path, items, "T0")
    original, transformed, _ = ds[0]
    assert torch.equal(original, transformed), \
        "T0's pair must be the same tensor twice"

    batch = original.unsqueeze(0)
    stages = ("s11_out", "hist", "s12_post_norm")
    pair_side, other = feat.capture_pair(saga_model, batch, batch, stages)
    plain = feat.plain_forward_stages(saga_model, batch, stages)
    for s in stages:
        assert torch.equal(pair_side[s], plain[s]), f"T0 differs at {s}"
        assert torch.equal(other[s], plain[s])


def test_term_1_00_leaves_s11_out_unchanged(saga_model, images):
    """§7: the bypass swaps the LAST block's gate, so the output of the
    SECOND-TO-LAST block cannot move.

    Asserted with `torch.equal` on a model whose gates are structured, so the
    edit is demonstrably not a no-op: `hist` DOES move, and the test checks
    that too. A test where nothing moved anywhere would pass while measuring
    nothing.
    """
    from saga.frozen.edits import terminal_gate_override

    stages = ("s11_out", "hist")
    native = feat.plain_forward_stages(saga_model, images, stages)
    with terminal_gate_override(saga_model, 1.0):
        bypass = feat.plain_forward_stages(saga_model, images, stages)

    assert torch.equal(native["s11_out"], bypass["s11_out"])
    assert not torch.equal(native["hist"], bypass["hist"]), \
        "the bypass changed nothing at all — the control is vacuous"


def test_the_sweep_records_a_zero_s11_difference_under_the_bypass(
        saga_model, tmp_path):
    """The same control, as the COLUMN Phase C asserts on."""
    items = _fake_split(tmp_path)
    conditions = load_conditions(CONDITIONS)
    out = tmp_path / "out"
    feat.run_correspondence_package(
        row=_fake_row("saga"), conditions=conditions,
        dataset_for=lambda t: feat.CorrespondencePairDataset(tmp_path, items,
                                                             t),
        out_dir=out, image_ids=[f"img/{i}" for i in range(len(items))],
        batch_size=2, split_name="fake", split_sha="b" * 64,
        git_sha="c" * 40, git_dirty=0, model=saga_model)

    rows = _read(out, "corr_records")
    bypass = [r for r in rows if r["condition_id"] == "term_1.00"]
    assert bypass, "the SAGA sweep must run the bypass"
    assert {r["max_abs_s11_diff_vs_native"] for r in bypass} == {"0.0"}
    native = [r for r in rows if r["condition_id"] == "native"]
    assert {r["max_abs_s11_diff_vs_native"] for r in native} == {"MISSING"}

    # AND the edit was really in force for the sweep's own forwards, not
    # only for the isolated control above: the two LATER stages must move
    # while s11_out does not. Without this, a bug that dropped the edit
    # context would leave every assertion in this test passing.
    def sim(condition, stage):
        return [r["mean_nn_sim"] for r in rows
                if r["condition_id"] == condition and r["stage"] == stage
                and r["transform"] == "T1" and r["descriptor"] == "l2"]

    assert sim("native", "s11_out") == sim("term_1.00", "s11_out")
    assert sim("native", "hist") != sim("term_1.00", "hist")
    assert sim("native", "s12_post_norm") != sim("term_1.00", "s12_post_norm")


# ── the sweep ────────────────────────────────────────────────────────────────

def _read(out_dir, kind):
    path = rec.records_path(out_dir, kind)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_sweep_writes_one_row_per_condition_transform_stage_descriptor(
        saga_model, tmp_path):
    items = _fake_split(tmp_path)
    conditions = load_conditions(CONDITIONS)
    out = tmp_path / "out"
    summary = feat.run_correspondence_package(
        row=_fake_row("saga"), conditions=conditions,
        dataset_for=lambda t: feat.CorrespondencePairDataset(tmp_path, items,
                                                             t),
        out_dir=out, image_ids=[f"img/{i}" for i in range(len(items))],
        batch_size=2, split_name="fake", split_sha="b" * 64,
        git_sha="c" * 40, git_dirty=0, model=saga_model)

    rows = _read(out, "corr_records")
    # 2 conditions x 4 transforms x 3 stages x 2 descriptors x 4 images
    assert len(rows) == 2 * 4 * 3 * 2 * len(items)
    assert summary["n_records"] == len(rows)
    assert set(rows[0]) == set(rec.CORR_COLUMNS)
    assert summary["state_restored"] is True
    # every stage of the T0 control is bit-exact against the plain forward
    for stage, diff in summary["t0_matches_plain_forward"].items():
        if stage in ("pair_members_identical", "n_images"):
            continue
        assert diff == 0.0, f"T0 differs at {stage}: {diff}"


def test_a_baseline_run_declares_only_the_native_condition(baseline_model,
                                                           tmp_path):
    """`applies_to` gates the bypass: a baseline has no terminal gate, and
    saga/frozen/edits.py refuses to override one that does not exist."""
    items = _fake_split(tmp_path)
    conditions = load_conditions(CONDITIONS)
    out = tmp_path / "out"
    summary = feat.run_correspondence_package(
        row=_fake_row("baseline"), conditions=conditions,
        dataset_for=lambda t: feat.CorrespondencePairDataset(tmp_path, items,
                                                             t),
        out_dir=out, image_ids=[f"img/{i}" for i in range(len(items))],
        batch_size=2, split_name="fake", split_sha="b" * 64,
        git_sha="c" * 40, git_dirty=0, model=baseline_model)
    assert set(summary["conditions"]) == {"native"}
    assert summary["skipped_conditions"] == ["term_1.00"]


def test_the_writer_is_append_safe_and_idempotent(saga_model, tmp_path):
    """The standing rule after the legacy resume-overwrite incident: a
    resubmitted job adds what is missing and rewrites nothing."""
    items = _fake_split(tmp_path)
    conditions = load_conditions(CONDITIONS)
    out = tmp_path / "out"
    kw = dict(row=_fake_row("saga"), conditions=conditions,
              dataset_for=lambda t: feat.CorrespondencePairDataset(
                  tmp_path, items, t),
              out_dir=out, image_ids=[f"img/{i}" for i in range(len(items))],
              batch_size=2, split_name="fake", split_sha="b" * 64,
              git_sha="c" * 40, git_dirty=0, model=saga_model)
    first = feat.run_correspondence_package(**kw)
    second = feat.run_correspondence_package(**kw)
    assert second["n_records"] == 0
    assert second["n_skipped"] == first["n_records"]
    assert len(_read(out, "corr_records")) == first["n_records"]
    assert rec.is_done(out, "corr_records", "a" * 64)


def test_the_model_state_is_restored_after_the_sweep(saga_model, tmp_path):
    items = _fake_split(tmp_path)
    conditions = load_conditions(CONDITIONS)
    before = state_hash(saga_model)
    feat.run_correspondence_package(
        row=_fake_row("saga"), conditions=conditions,
        dataset_for=lambda t: feat.CorrespondencePairDataset(tmp_path, items,
                                                             t),
        out_dir=tmp_path / "out",
        image_ids=[f"img/{i}" for i in range(len(items))],
        batch_size=2, split_name="fake", split_sha="b" * 64,
        git_sha="c" * 40, git_dirty=0, model=saga_model)
    assert state_hash(saga_model) == before


# ── the conditions contract (the runner hook) ────────────────────────────────

def test_the_committed_conditions_file_parses_and_declares_D9(  # noqa: N802
        ):
    """The YAML is the only place the transforms, stages and descriptors are
    defined, and it must match D9."""
    doc = load_conditions(CONDITIONS)
    assert doc["work_package"] == "I5_readout"
    assert doc["readout"] == "correspondence"
    assert doc["stage"] == "s11_out"                 # D1
    assert doc["transforms"] == ["T0", "T1", "T2", "T3"]
    assert doc["descriptors"] == ["l2", "centred_l2"]
    assert doc["bootstrap_seed"] == 0                # D6
    assert doc["bootstrap_resamples"] == 10000       # LOCKED_ANALYSIS §8
    ids = [c["id"] for c in doc["conditions"]]
    assert ids[0] == "native", "native must be declared first"
    assert ids == ["native", "term_1.00"]
    for c in doc["conditions"]:
        assert c["stages"] == ["s11_out", "hist", "s12_post_norm"]


def test_the_runner_refuses_an_undeclared_transform(tmp_path):
    """§7, the conditions contract: the hook in saga/frozen/runner.py is what
    makes an unlisted transform unrunnable rather than merely undocumented."""
    import yaml
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["transforms"] = ["T0", "T1", "rotate90"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(feat.FeatureError, match="rotate90"):
        load_conditions(bad)


def test_the_runner_refuses_an_undeclared_descriptor(tmp_path):
    import yaml
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["descriptors"] = ["l2", "whitened"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(feat.FeatureError, match="whitened"):
        load_conditions(bad)


def test_the_runner_refuses_a_readout_that_scores_nothing(tmp_path):
    """T0 alone is the reference forward and scores no correspondence."""
    import yaml
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["transforms"] = ["T0"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(feat.FeatureError, match="scores nothing"):
        load_conditions(bad)


def test_the_hook_leaves_every_other_conditions_file_alone():
    """Tracks A and B are adding their own hooks to load_conditions. Mine
    must be a no-op for every work package that declares no `readout`."""
    for name in ("smoke", "I2_terminal", "I1_spatial", "I6_external"):
        doc = load_conditions(REPO / "configs" / "frozen" / f"{name}.yaml")
        assert doc.get("readout") is None
        assert feat.check_readout_block("x", doc) is doc


# ── I5b, the conditional segmentation evaluator ──────────────────────────────

def test_i5b_gate_reads_the_manifest_and_nothing_else():
    """§2: I5b runs only on matched-epoch weights PRESENT IN THE MANIFEST.

    Against the real committed manifest, so this test states what I5b's
    status actually is — and will fail loudly if the manifest changes such
    that the pair stops being matched.
    """
    from tools.frozen_I5b_seg_eval import matched_seg_rows

    manifest = REPO / "results" / "frozen" / "I0_manifest" / "manifest.json"
    pair, reason = matched_seg_rows(manifest)
    assert pair is not None, f"I5b would be DROPPED: {reason}"
    assert set(pair) == {"baseline", "saga"}
    assert pair["baseline"]["run_id"] == "seg_vitb_baseline_s1"
    assert pair["saga"]["run_id"] == "seg_vitb_saga_s1"
    assert (pair["baseline"]["epochs_completed"]
            == pair["saga"]["epochs_completed"] == 80)
    for row in pair.values():
        assert len(row["ckpt_sha256"]) == 64
        assert row["arch"] == "vit_base"


def test_i5b_gate_refuses_an_unmatched_or_unhashed_pair(tmp_path):
    """The refusal is the point: no retraining, no re-fetching, no
    substituting a different epoch."""
    from tools.frozen_I5b_seg_eval import matched_seg_rows

    def _manifest(rows):
        p = tmp_path / f"m{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        return p

    def _row(variant, epochs=80, sha="d" * 64):
        return {"family": "dense_seg", "ckpt_kind": "last",
                "status": "eligible", "variant": variant,
                "run_id": f"seg_{variant}", "epochs_completed": epochs,
                "ckpt_sha256": sha, "arch": "vit_base"}

    pair, reason = matched_seg_rows(_manifest([_row("baseline")]))
    assert pair is None and "saga" in reason

    pair, reason = matched_seg_rows(
        _manifest([_row("baseline"), _row("saga", epochs=40)]))
    assert pair is None and "MATCHED epoch" in reason

    pair, reason = matched_seg_rows(
        _manifest([_row("baseline"), _row("saga", sha="MISSING")]))
    assert pair is None and "not identified" in reason


def test_i5b_refuses_a_training_mode_call():
    """§4, §11: evaluation of existing heads only."""
    from tools.frozen_I5b_seg_eval import SegEvalError, assert_eval_only

    model = torch.nn.Linear(4, 4)
    model.train()
    with torch.no_grad():
        with pytest.raises(SegEvalError, match="TRAINING mode"):
            assert_eval_only(model)

    model.eval()
    with pytest.raises(SegEvalError, match="gradients are enabled"):
        assert_eval_only(model)

    with torch.no_grad():
        with pytest.raises(SegEvalError, match="require a gradient"):
            assert_eval_only(model)
        for p in model.parameters():
            p.requires_grad_(False)
        assert assert_eval_only(model) is True


def test_per_image_confusion_matrices_sum_to_the_pooled_one():
    """The claim the I5b wrapper rests on: per-image records do not change
    the dataset metric, because integer confusion counts add.

    Run against the TRAINER's own `update_confusion`, so this is a statement
    about the committed metric and not about a copy of it.
    """
    from segmentation.tools.train import summarize_confusion, update_confusion

    rng = np.random.RandomState(0)
    n_classes = 6
    pooled = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    per_image_sum = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for _ in range(5):
        gt = torch.as_tensor(rng.randint(0, n_classes, size=(8, 8)))
        gt[0, 0] = 255                                  # an ignored pixel
        pred = torch.as_tensor(rng.randint(0, n_classes, size=(8, 8)))
        update_confusion(pooled, pred, gt, n_classes)
        one = torch.zeros(n_classes, n_classes, dtype=torch.int64)
        update_confusion(one, pred, gt, n_classes)
        per_image_sum += one
    assert torch.equal(pooled, per_image_sum)
    assert summarize_confusion(pooled)[0] == summarize_confusion(
        per_image_sum)[0]
    # and the ignored pixels really were excluded
    assert int(pooled.sum()) == 5 * (64 - 1)


# ── the analysis ─────────────────────────────────────────────────────────────

def test_the_bootstrap_is_deterministic_and_seeded_at_zero():
    """D6: seed 0, 10,000 resamples. Same input, same interval, every time."""
    from analysis.frozen_I5_analysis import (BOOTSTRAP_RESAMPLES,
                                             BOOTSTRAP_SEED, bootstrap_ci)

    assert (BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES) == (0, 10000)
    values = list(np.random.RandomState(4).rand(60))
    a = bootstrap_ci(values)
    b = bootstrap_ci(values)
    assert a == b
    assert a[1] < a[0] < a[2]


def test_missing_is_never_averaged():
    """I0 handoff §8.2, as code: MISSING is dropped and COUNTED, never
    coerced to zero and never guessed."""
    from analysis.frozen_I5_analysis import MISSING, mean_or_missing, value

    mean, n, n_missing = mean_or_missing([1.0, MISSING, 3.0])
    assert (mean, n, n_missing) == (2.0, 2, 1)
    assert mean_or_missing([MISSING, MISSING]) == (MISSING, 0, 2)
    assert value({"acc_exact_exc": "MISSING"}, "acc_exact_exc") is MISSING
    assert value({"acc_exact_exc": "0.25"}, "acc_exact_exc") == 0.25


def test_the_tables_are_byte_identical_when_rebuilt(saga_model, baseline_model,
                                                    tmp_path):
    """A generated file under results/ is REGENERATED, never hand-merged
    (I0 handoff §8.6), so building it twice from the same records must give
    the same bytes."""
    from analysis.frozen_I5_analysis import build_all

    items = _fake_split(tmp_path)
    conditions = load_conditions(CONDITIONS)
    root = tmp_path / "runs"
    manifest = _cohort_manifest(tmp_path)
    for variant, model in (("saga", saga_model), ("baseline", baseline_model)):
        feat.run_correspondence_package(
            row=_fake_row(variant), conditions=conditions,
            dataset_for=lambda t: feat.CorrespondencePairDataset(
                tmp_path, items, t),
            out_dir=root / f"fake_{variant}",
            image_ids=[f"img/{i}" for i in range(len(items))],
            batch_size=2, split_name="fake", split_sha="b" * 64,
            git_sha="c" * 40, git_dirty=0, model=model)

    first = build_all(root, manifest, tmp_path / "t1")
    second = build_all(root, manifest, tmp_path / "t2")
    assert first["n_runs_loaded"] == 2
    for name, _fields, _builder in _TABLES():
        a = (tmp_path / "t1" / f"{name}.csv").read_bytes()
        b = (tmp_path / "t2" / f"{name}.csv").read_bytes()
        assert a == b, f"{name}.csv is not reproducible"
        assert a.count(b"\r") == 0, f"{name}.csv has CRLF line endings"


def _TABLES():  # noqa: N802
    from analysis.frozen_I5_analysis import TABLES
    return TABLES


def _cohort_manifest(tmp_path):
    """A two-row manifest the cohort loader accepts, paired on seed tag."""
    rows = []
    for variant in ("baseline", "saga"):
        rows.append({
            "run_id": f"fake_{variant}", "family": "e2r_300ep",
            "status": "eligible", "ckpt_kind": "last", "arch": "vit_tiny",
            "recipe_actual": "mixup", "variant": variant,
            "provenance_tag": "s1", "seed_controlled": 1,
            "ckpt_sha256": "a" * 64})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def test_every_table_stamps_its_endpoint_class(saga_model, baseline_model,
                                               tmp_path):
    """§7 / D7: every I5 comparison is secondary or descriptive, and the
    table says which on EVERY row — not in a caption."""
    from analysis.frozen_I5_analysis import DESCRIPTIVE, SECONDARY, build_all

    items = _fake_split(tmp_path)
    conditions = load_conditions(CONDITIONS)
    root = tmp_path / "runs"
    for variant, model in (("saga", saga_model), ("baseline", baseline_model)):
        feat.run_correspondence_package(
            row=_fake_row(variant), conditions=conditions,
            dataset_for=lambda t: feat.CorrespondencePairDataset(
                tmp_path, items, t),
            out_dir=root / f"fake_{variant}",
            image_ids=[f"img/{i}" for i in range(len(items))],
            batch_size=2, split_name="fake", split_sha="b" * 64,
            git_sha="c" * 40, git_dirty=0, model=model)
    build_all(root, _cohort_manifest(tmp_path), tmp_path / "tables")

    for name, _f, _b in _TABLES():
        path = tmp_path / "tables" / f"{name}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"{name} is empty"
        assert "endpoint_class" in rows[0]
        assert {r["endpoint_class"] for r in rows} <= {SECONDARY, DESCRIPTIVE}


def test_chance_is_on_every_readout_table(saga_model, tmp_path):
    """§3: chance is printed on every readout table. An accuracy without it
    is unreadable."""
    from analysis.frozen_I5_analysis import (A_FIELDS, C_FIELDS, D_FIELDS,
                                             B_FIELDS)
    for fields in (A_FIELDS, B_FIELDS, C_FIELDS, D_FIELDS):
        assert "chance_exact" in fields


# ── guards ───────────────────────────────────────────────────────────────────

def test_no_optimizer_or_training_import_in_the_i5_files():
    """§11 / I0 handoff §8.7, extended to this track's files.

    Checked on the parsed AST, not on the raw text: these files say "no
    training" in their own docstrings and a substring search would flag the
    very comments that state the rule.
    """
    banned_modules = ("torch.optim", "torch.optim.lr_scheduler")
    banned_names = {"AdamW", "Adam", "SGD", "backward", "step",
                    "LabelSmoothingCrossEntropy", "Mixup"}
    targets = [
        REPO / "saga" / "frozen" / "features.py",
        REPO / "saga" / "frozen" / "transforms.py",
        REPO / "saga" / "frozen" / "correspondence.py",
        REPO / "tools" / "frozen_I5_corr.py",
        REPO / "tools" / "frozen_I5b_seg_eval.py",
        REPO / "analysis" / "frozen_I5_analysis.py",
    ]
    assert all(t.exists() for t in targets)

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


def test_the_runner_hook_is_one_line():
    """§7: `features.py` is a new module and the only change to runner.py is
    a single hook line plus its import. A second call site would make the
    three tracks' hooks harder to keep separable in a merge."""
    src = (REPO / "saga" / "frozen" / "runner.py").read_text(encoding="utf-8")
    calls = [ln for ln in src.splitlines()
             if "_c_check_readout_block(" in ln and "import" not in ln]
    assert len(calls) == 1, f"expected one hook call, found {calls}"
    assert "# TASK C / I5" in calls[0]


def test_the_job_file_lists_the_cohort_in_array_order():
    """An array index must mean the same run in the job file and in the
    manifest — the job's own runtime check, pinned here at commit time."""
    from saga.frozen.runner import eligible_run_ids

    text = (REPO / "scripts" / "jobs"
            / "frozen_I5a.sbatch").read_text(encoding="utf-8")
    block = text.split("RUN_IDS=(", 1)[1].split(")", 1)[0]
    listed = [ln.split('"')[1] for ln in block.splitlines()
              if ln.strip().startswith('"')]
    want = eligible_run_ids(
        REPO / "results" / "frozen" / "I0_manifest" / "manifest.json")
    assert listed == want
    assert f"--array=0-{len(want) - 1}" in text


def test_the_job_file_fixes_the_split_to_sub2k():
    """D9 declares sub2k. The job must not be able to run another split."""
    text = (REPO / "scripts" / "jobs"
            / "frozen_I5a.sbatch").read_text(encoding="utf-8")
    assert "SPLIT=results/frozen/splits/sub2k.json" in text
    assert "frozen_I5_corr.py" in text


def test_the_declared_split_is_the_one_d9_names():
    """sub2k, 2,000 images, and the sha LOCKED_ANALYSIS §11 records."""
    split = json.loads(
        (REPO / "results" / "frozen" / "splits" / "sub2k.json").read_text(
            encoding="utf-8"))
    assert split["n"] == 2000
    assert split["name"] == "sub2k"
    assert split["sha256"] == (
        "d2fc8b5b4c5ab40928c1ea861f21a3b561c0654a93479048773802568d338117")
    locked = (REPO / "docs" / "LOCKED_ANALYSIS.md").read_text(encoding="utf-8")
    assert split["sha256"] in locked


def test_the_word_sink_does_not_appear_in_the_i5_modules():
    """§11: descriptive wording in all outputs. "sink" is reserved for I7's
    wording module and appears nowhere else in this track."""
    for name in ("features.py", "transforms.py", "correspondence.py"):
        text = (REPO / "saga" / "frozen" / name).read_text(encoding="utf-8")
        assert "sink" not in text.lower().replace("sink_counts_mad", ""), name
