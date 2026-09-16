"""
tests/test_I1_spatial.py
========================
TASK A / I1 A11 — CPU, tiny fake models, fake data, in the style of
`tests/test_I2_terminal.py`.

What these tests are actually for, in order of how much damage they prevent:

  1. `in_b07` and `in_b08` must be the residual stream entering blocks[7]
     and blocks[8]. Off by one, and every I4 perturbation acts at a
     different depth than the map that chose its coordinates — silently,
     and the result would still look like a result.
  2. No mask may be built from evaluation data. The guard is an allow-list
     and the test drives it with a real evaluation sha.
  3. The `*_addr.json` round trip must reproduce a COMMITTED map byte for
     byte. The contract reads TASK-07's history; it may not rewrite it.
  4. The MAD count must be EXACTLY invariant under rescaling. The
     scale-only null's whole argument rests on that identity, and an
     `allclose` here would hide an implementation that broke it.
  5. `ring_indices` must be right at side 16 and unchanged at side 14. I6's
     grids are 16x16 and TASK-07's answers were written against 14x14.
  6. A table rebuild from the same inputs must produce the same bytes.

Two model shapes are exercised where it matters: a tiny SAGA ViT (1 prefix
token) and a tiny timm 4-REGISTER ViT (5 prefix tokens), built directly and
never wrapped in SAGAViT.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from analysis.address_analysis import border_distance_map, ring_indices
from saga.frozen import norms as fnorms
from saga.frozen import prevalence as P
from saga.frozen.stages import (ALL_STAGES, BLOCK_INPUT_STAGES, STAGES,
                                StageError, capture_stages,
                                stage_block_index)
from saga.vit import build_saga_vit

REPO = Path(__file__).resolve().parents[1]
CONDITIONS_YAML = REPO / "configs" / "frozen" / "I1_spatial.yaml"
JOB_FILE = REPO / "scripts" / "jobs" / "frozen_I1.sbatch"
MANIFEST = REPO / "results" / "frozen" / "I0_manifest" / "manifest.json"
GITIGNORE = REPO / ".gitignore"

DEPTH = 12          # the production depth: in_b07/in_b08 need >= 8 blocks
DIM = 24
HEADS = 3
SIDE = 14
N_PATCHES = SIDE * SIDE
MISSING = "MISSING"
PENDING = "PENDING B"

#: An evaluation-split sha, for driving the selection guard.
EVAL_SHA = "7fdf5f9f2ace98ef03a6267455daa1104b92b6689c510ab8b1e5420340f27014"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def plain_model():
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=10, depth=DEPTH, embed_dim=DIM,
                             num_heads=HEADS).eval()


@pytest.fixture(scope="module")
def saga_model():
    torch.manual_seed(0)
    return build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=10,
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
    return torch.randn(4, 3, 224, 224)


def a_map(freq, *, basis="fixed_cal", split_sha=P.DISCOVERY_SPLIT_SHA256,
          stage="in_b07", n_images=1000, threshold=1.0):
    freq = np.asarray(freq, dtype=np.float64)
    return P.PrevalenceMap(
        freq=freq, n_images=n_images,
        n_exceed_total=int(round(freq.sum() * n_images)), basis=basis,
        threshold=threshold, stage=stage, split_sha=split_sha,
        split_name="test", run_id="fake_run", ckpt_sha256="a" * 64,
        grid_side=P.grid_side_of(freq.size), n_prefix=1, git_sha="c" * 40)


# ─────────────────────────────────────────────────────────────────────────────
# 1. The block-input stages
# ─────────────────────────────────────────────────────────────────────────────

def test_in_b07_is_the_output_of_block_6_and_in_b08_of_block_7(plain_model,
                                                               images):
    """THE off-by-one test. Tensors, not indices: the hook has to land on
    blocks[6]/blocks[7], and a comparison of block indices would pass even if
    `capture_stages` registered the hook somewhere else."""
    with capture_stages(plain_model, ("in_b07", "in_b08")) as store:
        plain_model(images)
        got = {k: v.clone() for k, v in store.items()}

    with torch.no_grad():
        x = plain_model.patch_embed(images)
        x = plain_model._pos_embed(x)
        x = plain_model.norm_pre(x)
        manual = {}
        for i, blk in enumerate(plain_model.blocks):
            x = blk(x)
            if i == 6:
                manual["in_b07"] = x[:, 1:, :].clone()
            if i == 7:
                manual["in_b08"] = x[:, 1:, :].clone()

    for stage in ("in_b07", "in_b08"):
        assert torch.equal(got[stage], manual[stage]), \
            f"{stage} is not the output of blocks[{BLOCK_INPUT_STAGES[stage] - 1}]"


def test_the_block_input_indices_are_absolute_not_depth_relative(plain_model):
    """`in_b07` names blocks[7] on a model of ANY depth. `s11_out` is
    depth-relative; conflating the two would move the block-input stages on a
    model that is not 12 deep."""
    assert stage_block_index(plain_model, "in_b07") == 6
    assert stage_block_index(plain_model, "in_b08") == 7
    assert stage_block_index(plain_model, "s11_out") == DEPTH - 2

    torch.manual_seed(0)
    shallow = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                                num_classes=10, depth=9, embed_dim=DIM,
                                num_heads=HEADS).eval()
    assert stage_block_index(shallow, "in_b07") == 6      # unmoved
    assert stage_block_index(shallow, "s11_out") == 7     # moved with depth


def test_a_model_too_shallow_for_a_block_input_stage_is_refused():
    torch.manual_seed(0)
    tiny = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=10, depth=4, embed_dim=DIM,
                             num_heads=HEADS).eval()
    with pytest.raises(StageError, match="only 4 block"):
        stage_block_index(tiny, "in_b08")


def test_the_block_input_stages_are_additive(plain_model, images, saga_model):
    """`STAGES` is UNTOUCHED — it is the I0 contract for the terminal region,
    and several I0 tests capture every member of it on a 4-block fake model,
    where a block-input stage does not exist. The new names are reachable
    through `ALL_STAGES`, which is what `resolve_stage` validates against."""
    assert STAGES == ("s11_out", "s12_pre_norm", "s12_post_norm", "hist")
    assert ALL_STAGES[:6] == STAGES + ("in_b07", "in_b08")
    # I6 appends its two external analog names; tests/test_I6_external.py
    # owns those. Slicing rather than comparing the whole tuple keeps this
    # test about I1's addition.
    assert set(ALL_STAGES) >= set(STAGES) | {"in_b07", "in_b08"}
    for model in (plain_model, saga_model):
        with capture_stages(model, ("s11_out", "hist", "in_b07")) as store:
            model(images)
            assert set(store) >= {"s11_out", "hist", "s12_pre_norm", "in_b07"}
            assert torch.equal(store["hist"], store["s12_pre_norm"])


def test_the_block_input_stages_drop_the_registers_prefix(reg4_model, images):
    """5 prefix tokens, not a hard-coded 1 (I0 audit bug B2)."""
    with capture_stages(reg4_model, ("in_b07", "in_b08")) as store:
        reg4_model(images)
        for stage in ("in_b07", "in_b08"):
            assert store[stage].shape[1] == N_PATCHES


# ─────────────────────────────────────────────────────────────────────────────
# 2. The selection guard — tested before the mask builders it protects
# ─────────────────────────────────────────────────────────────────────────────

def test_the_selection_guard_refuses_the_evaluation_split():
    pm = a_map(np.linspace(0, 1, N_PATCHES), split_sha=EVAL_SHA)
    with pytest.raises(P.PrevalenceError, match="EVALUATION"):
        P.assert_selection_split(pm)
    with pytest.raises(P.PrevalenceError, match="EVALUATION"):
        P.topk_mask(pm)


def test_the_selection_guard_is_an_allow_list_not_a_deny_list():
    """A split nobody has named is refused as firmly as evaluation is."""
    for sha in ("f" * 64, "", None, "6b707eb39f934274a5ea753d613af54dc9aad015"
                                    "10808f246b70506a96003003"):
        with pytest.raises(P.PrevalenceError):
            P.assert_selection_split(a_map(np.ones(N_PATCHES) * 0.5,
                                           split_sha=sha))


def test_the_selection_guard_accepts_the_discovery_split():
    pm = a_map(np.linspace(0, 1, N_PATCHES))
    assert P.assert_selection_split(pm) == P.DISCOVERY_SPLIT_SHA256
    assert len(P.topk_mask(pm, k=16)) == 16


def test_the_discovery_sha_matches_the_locked_split_file():
    """The constant is a copy of a sha; pin it to the file it copies."""
    import hashlib
    path = REPO / "results" / "diagsplit" / "val_diag_split.json"
    if not path.exists():
        pytest.skip("the discovery split file is not in this checkout")
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n"))
    assert digest.hexdigest() == P.DISCOVERY_SPLIT_SHA256


# ─────────────────────────────────────────────────────────────────────────────
# 3. Masks
# ─────────────────────────────────────────────────────────────────────────────

def test_topk_mask_takes_the_k_largest_and_breaks_ties_by_index():
    freq = np.zeros(N_PATCHES)
    freq[[10, 20, 30]] = [0.9, 0.8, 0.7]
    freq[[100, 101, 102, 103]] = 0.5          # a four-way tie
    mask = P.topk_mask(a_map(freq), k=5)
    assert mask == [10, 20, 30, 100, 101]
    assert mask == sorted(mask)


def test_ring_matched_controls_match_the_ring_composition_and_are_deterministic():
    freq = np.zeros(N_PATCHES)
    ring0, ring1 = ring_indices(SIDE, 0), ring_indices(SIDE, 1)
    freq[ring0[:6]] = 0.9
    freq[ring1[:10]] = 0.8
    primary = P.topk_mask(a_map(freq), k=16)
    wanted = P.ring_composition(primary, SIDE)
    assert wanted == {0: 6, 1: 10}

    controls = P.ring_matched_controls(primary, SIDE)
    assert len(controls) == P.N_CONTROL_MASKS
    for mask, info in controls:
        assert P.ring_composition(mask, SIDE) == wanted
        assert mask == sorted(set(mask))
        assert info["seed"] in P.CONTROL_SEEDS
        assert 0 <= info["overlap"] <= len(primary)
    again = P.ring_matched_controls(primary, SIDE)
    assert [m for m, _ in controls] == [m for m, _ in again]


def test_ring_matched_controls_may_overlap_the_primary_mask_by_default():
    """LOCKED_ANALYSIS §5: draws are never rejected to amplify contrast. The
    task file asks for the opposite; both are implemented and the DEFAULT is
    the signed document's rule."""
    freq = np.zeros(N_PATCHES)
    freq[ring_indices(SIDE, 1)[:16]] = 0.9
    primary = P.topk_mask(a_map(freq), k=16)
    overlaps = [i["overlap"]
                for _, i in P.ring_matched_controls(primary, SIDE)]
    assert max(overlaps) > 0, "ring 1 has 44 members and 16 are taken — some " \
                              "control must overlap unless draws are rejected"
    excluded = P.ring_matched_controls(primary, SIDE, exclude_primary=True)
    assert all(i["overlap"] == 0 for _, i in excluded)


def test_mask_coordinates_carry_row_col_and_ring():
    coords = P.mask_coordinates([0, 15, 195], SIDE)
    assert coords[0] == {"index": 0, "row": 0, "col": 0, "ring": 0}
    assert coords[1] == {"index": 15, "row": 1, "col": 1, "ring": 1}
    assert coords[2] == {"index": 195, "row": 13, "col": 13, "ring": 0}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rings at 14 and at 16
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("side,expected_sizes",
                         [(14, [52, 44, 36, 28, 20, 12, 4]),
                          (16, [60, 52, 44, 36, 28, 20, 12, 4])])
def test_ring_indices_are_correct_at_both_grid_sides(side, expected_sizes):
    """`analysis/address_analysis.ring_indices` already takes `side`; it is
    imported and NOT re-derived, and no optional argument was added to it.
    This pins both the 14x14 geometry TASK-07's answers were written against
    and the 16x16 geometry I6's patch-14 models need."""
    sizes = [ring_indices(side, k).size for k in range(side // 2)]
    assert sizes == expected_sizes
    assert sum(sizes) == side * side

    # independent derivation of the Chebyshev distance from the border
    for k in range(side // 2):
        manual = [r * side + c for r in range(side) for c in range(side)
                  if min(r, side - 1 - r, c, side - 1 - c) == k]
        assert list(ring_indices(side, k)) == manual
    assert border_distance_map(side).shape == (side, side)


def test_address_analysis_was_not_modified_for_the_grid_size():
    """TASK A §6 allows an optional `side` argument; none was needed. Assert
    the functions still take it POSITIONALLY and have no side default, so a
    later edit that hard-codes 14 fails here."""
    import inspect
    for fn in (ring_indices, border_distance_map):
        params = inspect.signature(fn).parameters
        assert list(params)[0] == "side"
        assert params["side"].default is inspect.Parameter.empty


# ─────────────────────────────────────────────────────────────────────────────
# 5. The PrevalenceMap contract
# ─────────────────────────────────────────────────────────────────────────────

def test_prevalence_map_round_trips_through_npz(tmp_path):
    counts = np.random.RandomState(0).randint(0, 500, size=N_PATCHES)
    pm = P.from_counts(counts, n_images=500, basis="fixed_cal", threshold=3.5,
                       stage="in_b07", split_sha=P.DISCOVERY_SPLIT_SHA256,
                       run_id="r", ckpt_sha256="a" * 64, n_prefix=1,
                       git_sha="c" * 40, split_name="discovery")
    back = P.PrevalenceMap.load(pm.save(tmp_path / "m.npz"))
    assert np.array_equal(back.freq, pm.freq)
    assert np.array_equal(back.counts, pm.counts)
    for field in ("n_images", "n_exceed_total", "basis", "threshold", "stage",
                  "split_sha", "run_id", "ckpt_sha256", "grid_side",
                  "n_prefix", "git_sha", "condition_id"):
        assert getattr(back, field) == getattr(pm, field)


@pytest.mark.parametrize("break_it,match", [
    (lambda m: setattr(m, "freq", np.full(N_PATCHES, np.nan)), "non-finite"),
    (lambda m: setattr(m, "freq", np.full(N_PATCHES, 1.5)), r"\[0, 1\]"),
    (lambda m: setattr(m, "freq", np.zeros(N_PATCHES + 1)), "grid_side"),
    (lambda m: setattr(m, "n_images", 0), "n_images"),
    (lambda m: setattr(m, "git_sha", ""), "provenance"),
    (lambda m: setattr(m, "ckpt_sha256", MISSING), "provenance"),
    (lambda m: setattr(m, "basis", "canon"), "basis"),
    (lambda m: setattr(m, "n_prefix", 0), "prefix"),
])
def test_validate_refuses_a_map_that_cannot_carry_a_number(break_it, match):
    pm = a_map(np.full(N_PATCHES, 0.25))
    pm.counts = None
    break_it(pm)
    with pytest.raises(P.PrevalenceError, match=match):
        pm.validate()


def test_validate_refuses_counts_that_disagree_with_freq():
    pm = a_map(np.full(N_PATCHES, 0.25), n_images=1000)
    pm.counts = np.full(N_PATCHES, 999, dtype=np.int64)
    with pytest.raises(P.PrevalenceError, match="counts / n_images"):
        pm.validate()


ADDR_FILES = sorted((REPO / "results" / "runs").glob(
    "*/diag/diag_final_last_addr.json"))


@pytest.mark.skipif(not ADDR_FILES, reason="no committed *_addr.json here")
def test_the_addr_json_round_trip_reproduces_a_committed_map_byte_for_byte():
    """The contract READS TASK-07's history and never rewrites it.

    Byte identity alone would be satisfied by carrying the document blindly,
    so the SECOND assertion is the one with teeth: the carried values must
    still pass `address_analysis.verify_stored_concentration`, which
    re-derives every concentration statistic through an independent
    implementation.
    """
    path = ADDR_FILES[0]
    raw = path.read_bytes()
    for basis in ("canon", "mad"):
        pm = P.from_addr_json(path, basis)
        doc = P.to_addr_json(pm)
        assert P.dumps_addr_json(doc).encode() == raw, \
            f"{path.name}: the {basis} round trip changed the bytes"
        P.verify_addr_document(doc, basis, label=path.name)
        assert pm.n_positions == 196
        assert np.isclose(pm.freq.sum(), doc[f"mass_{basis}"])


def test_to_addr_json_builds_a_fresh_document_for_a_new_map():
    freq = np.abs(np.random.RandomState(3).normal(size=N_PATCHES)) / 20.0
    pm = a_map(np.clip(freq, 0, 1), basis="mad")
    doc = P.to_addr_json(pm)
    assert doc["schema"] == P.ADDR_SCHEMA
    assert doc["n_positions"] == N_PATCHES
    assert doc["stage"] == "in_b07"
    P.verify_addr_document(doc, "mad", label="fresh")


def test_cell_mean_map_weights_every_checkpoint_equally():
    a = a_map(np.full(N_PATCHES, 0.2), n_images=100)
    b = a_map(np.full(N_PATCHES, 0.4), n_images=9000)
    assert np.allclose(P.cell_mean_map([a, b]), 0.3)
    with pytest.raises(P.PrevalenceError, match="one stage and one basis"):
        P.cell_mean_map([a, a_map(np.full(N_PATCHES, 0.4), stage="in_b08")])


def test_a_non_square_token_count_is_refused():
    with pytest.raises(P.PrevalenceError, match="not a square grid"):
        P.grid_side_of(197)


# ─────────────────────────────────────────────────────────────────────────────
# 6. The reused statistics
# ─────────────────────────────────────────────────────────────────────────────

def test_the_ring_adjusted_residual_is_zero_for_a_pure_ring_profile_map():
    """A map that is constant within every ring shares its whole structure
    with its ring profile, so the residual has nothing left in it."""
    rings = border_distance_map(SIDE).reshape(-1)
    pure = np.array([0.05 * (r + 1) for r in rings], dtype=np.float64)
    assert np.allclose(P.ring_adjusted(pure), 0.0, atol=1e-12)


def test_ring_adjustment_removes_the_agreement_a_shared_ring_profile_buys():
    """THE reason the residual correlation is reported beside the raw one.

    Two maps built from the SAME ring profile with INDEPENDENT within-ring
    arrangements share nothing but their border structure. The raw rho is
    large anyway; the ring-adjusted rho is not, and it is the one that says
    whether two runs put the mass in the same PLACES.
    """
    rings = border_distance_map(SIDE).reshape(-1)
    profile = 0.02 + 0.10 * (rings == 1) + 0.04 * (rings == 0)
    rng = np.random.RandomState(0)
    a = profile + rng.uniform(0, 0.004, size=N_PATCHES)
    b = profile + rng.uniform(0, 0.004, size=N_PATCHES)

    raw = P.spatial_rho(a, b)[0]
    resid = P.spatial_rho(P.ring_adjusted(a), P.ring_adjusted(b))[0]
    assert raw > 0.8, f"the shared ring profile should carry a large rho ({raw})"
    assert abs(resid) < 0.2, f"independent arrangements should leave little " \
                             f"residual agreement ({resid})"


def test_the_ring_adjusted_residual_keeps_within_ring_structure():
    rng = np.random.RandomState(0)
    rings = border_distance_map(SIDE).reshape(-1)
    freq = 0.05 * (rings + 1) + rng.uniform(0, 0.01, size=N_PATCHES)
    resid = P.ring_adjusted(freq)
    for k in range(SIDE // 2):
        assert abs(resid[ring_indices(SIDE, k)].mean()) < 1e-12
    assert resid.std() > 0


@pytest.mark.parametrize("freq,expected", [
    (np.full(N_PATCHES, 0.3), 0.0),                     # no positional variance
    (np.concatenate([np.ones(98), np.zeros(98)]), 1.0),  # all of it
])
def test_eta2_has_its_known_values(freq, expected):
    """eta^2 = Var_P f(P) / (f_bar (1 - f_bar)): 0 when every position has the
    same frequency, 1 when each position is deterministic."""
    assert np.isclose(P.eta2_positional(freq * 1.0), expected)


def test_eta2_is_scale_free_in_the_way_raw_variance_is_not():
    """Two maps with the same SHAPE but different mass have the same eta^2
    only if the shape is scaled the way Bernoulli variance is; the point of
    the ratio is that it does not simply track mass."""
    a = np.full(N_PATCHES, 0.5)
    a[:98] = 0.9
    a[98:] = 0.1
    assert P.eta2_positional(a) > 0.5
    with pytest.raises(P.PrevalenceError, match="undefined"):
        P.eta2_positional(np.zeros(N_PATCHES))


def test_ring_share_sums_to_one_over_the_rings():
    freq = np.abs(np.random.RandomState(7).normal(size=N_PATCHES)) + 0.01
    total = sum(P.ring_share(a_map(freq / freq.max() / 2), k)
                for k in range(SIDE // 2))
    assert np.isclose(total, 1.0)


def test_spatial_rho_carries_the_permutation_reference():
    rng = np.random.RandomState(0)
    a = rng.uniform(0, 1, size=N_PATCHES)
    rho, p, sd, n = P.spatial_rho(a, a)
    assert np.isclose(rho, 1.0)
    assert n == 8 * N_PATCHES          # 8 dihedral x every torus roll
    assert p >= 1.0 / n                # the identity is in the set
    assert sd > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Norms, indicators and the writers
# ─────────────────────────────────────────────────────────────────────────────

class _FakeSplit(torch.utils.data.Dataset):
    def __init__(self, n):
        torch.manual_seed(2)
        self.x = torch.randn(n, 3, 224, 224)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], 0


@pytest.fixture(scope="module")
def extracted(plain_model):
    ds = _FakeSplit(6)
    ids = [f"img{i:03d}" for i in range(6)]
    tau = {s: 1.0 for s in ("s11_out", "in_b07", "in_b08", "hist")}
    return fnorms.extract_stage_norms(
        plain_model, ds, ("s11_out", "in_b07", "in_b08", "hist"),
        device="cpu", batch_size=4, image_ids=ids, tau_cal=tau), ids


def test_the_norm_writer_has_the_declared_shapes(extracted, tmp_path):
    acc, ids = extracted
    a = acc["in_b07"]
    assert a.norms().shape == (6, N_PATCHES)
    assert a.norms().dtype == np.float32
    assert a.mad_thresholds().shape == (6,)
    assert a.image_ids == ids
    assert a.n_positions == N_PATCHES

    meta = {"run_id": "r", "ckpt_sha256": "a" * 64, "split_sha256": "b" * 64,
            "git_sha": "c" * 40, "n_prefix": 1}
    path = fnorms.write_norms_npz(tmp_path, a, meta)
    with np.load(path, allow_pickle=False) as z:
        assert z["norms"].shape == (6, N_PATCHES)
        assert list(z["image_ids"]) == ids
        m = json.loads(str(z["meta_json"]))
    for key in ("run_id", "ckpt_sha256", "split_sha256", "git_sha", "stage",
                "n_images", "n_positions"):
        assert key in m, f"norms_<stage>.npz does not record {key}"
    assert m["stage"] == "in_b07"


def test_the_indicator_packbits_round_trip_is_exact(extracted, tmp_path):
    acc, ids = extracted
    a = acc["in_b07"]
    meta = {"run_id": "r", "ckpt_sha256": "a" * 64, "split_sha256": "b" * 64,
            "git_sha": "c" * 40, "n_prefix": 1}
    path = fnorms.write_stage_maps_npz(tmp_path, [a], meta)
    for basis in ("fixed_cal", "mad"):
        back = fnorms.read_indicators(path, "native", basis)
        assert back.shape == (6, N_PATCHES)
        assert np.array_equal(back, a.indicators(basis)), \
            f"{basis} indicators did not survive packbits"
    with np.load(path, allow_pickle=False) as z:
        assert z["ind__native|mad"].shape == (6, (N_PATCHES + 7) // 8)
        assert np.array_equal(z["counts__native|mad"],
                              a.indicators("mad").sum(axis=0))
        assert np.allclose(z["freq__native|mad"],
                           z["counts__native|mad"] / 6.0)
        assert list(z["image_ids"]) == ids


def test_the_indicators_cost_25_bytes_per_image_not_196(extracted, tmp_path):
    """What makes `maps_<stage>.npz` committable: 196 bits packed into 25
    bytes. The npz file size is dominated by its own headers at 6 images, so
    the claim is checked on the ARRAY, which is where it lives."""
    acc, _ = extracted
    meta = {"run_id": "r", "ckpt_sha256": "a" * 64, "split_sha256": "b" * 64,
            "git_sha": "c" * 40, "n_prefix": 1}
    path = fnorms.write_stage_maps_npz(tmp_path, [acc["in_b07"]], meta)
    with np.load(path, allow_pickle=False) as z:
        packed = z["ind__native|mad"]
    assert packed.dtype == np.uint8
    assert packed.nbytes / packed.shape[0] == 25
    # 10,000 images x 25 B x 2 bases is well inside what git should carry
    assert 10000 * packed.nbytes / packed.shape[0] * 2 < 600_000


def test_the_norm_dumps_are_git_ignored_and_the_maps_are_not():
    """A `maps_*` pattern would ignore the committed artifact Phase C reads.
    Comments may mention it; only the RULES matter."""
    rules = [line.strip() for line in
             GITIGNORE.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.strip().startswith("#")]
    assert "results/frozen/**/norms_*.npz" in rules
    assert not [r for r in rules if "maps_" in r]


# ─────────────────────────────────────────────────────────────────────────────
# 8. The scale-only null
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def norm_field():
    """A norm field with border structure, so thresholds actually bite."""
    rng = np.random.RandomState(0)
    rings = border_distance_map(SIDE).reshape(-1)
    base = 10.0 + 6.0 * (rings == 1) + 2.0 * (rings == 0)
    return base[None, :] * rng.uniform(0.8, 1.2, size=(200, N_PATCHES))


def test_the_mad_count_is_exactly_invariant_under_rescaling(norm_field):
    """median(cv) + k*MAD(cv) = c*(median(v) + k*MAD(v)), so the comparison
    has the same truth value. EXACT equality, not allclose: the scale null's
    whole argument rests on this identity."""
    reference = fnorms.mad_counts(norm_field)
    for c in (0.1, 0.5, 1.0, 2.0, 37.5, 1000.0):
        assert np.array_equal(fnorms.mad_counts(norm_field, c=c), reference), \
            f"the MAD count moved under c={c}"


def test_the_fixed_count_is_monotone_in_the_scale(norm_field):
    tau = float(np.quantile(norm_field, 0.9))
    counts = [fnorms.mean_fixed_count(norm_field, tau, c)
              for c in np.linspace(0.2, 3.0, 40)]
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert counts[0] < counts[-1]


def test_the_matched_scale_is_found_and_is_the_unique_infimum(norm_field):
    tau = float(np.quantile(norm_field, 0.9))
    target = fnorms.mean_fixed_count(norm_field, tau, 0.85)
    found = fnorms.match_scale(norm_field, tau, target)
    assert found["achieved"] >= target
    assert np.isclose(found["achieved"], target)
    # the infimum: just below it the target is not reached
    assert fnorms.mean_fixed_count(norm_field, tau,
                                   found["c"] * (1 - 1e-9)) < target
    # and it is deterministic
    assert fnorms.match_scale(norm_field, tau, target)["c"] == found["c"]


def test_an_unreachable_target_is_refused_rather_than_clipped(norm_field):
    tau = float(np.quantile(norm_field, 0.9))
    with pytest.raises(fnorms.NormsError, match="no scale"):
        fnorms.match_scale(norm_field, tau, N_PATCHES + 1.0)


def test_the_scaled_frequency_map_agrees_with_the_counts(norm_field):
    tau = float(np.quantile(norm_field, 0.9))
    m = fnorms.scaled_frequency_map(norm_field, tau, 1.3)
    assert np.isclose(m.sum(),
                      fnorms.mean_fixed_count(norm_field, tau, 1.3))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Split-half reliability
# ─────────────────────────────────────────────────────────────────────────────

def test_the_split_half_is_deterministic_and_exhaustive():
    a, b = fnorms.split_half_indices(10000, seed=0)
    assert len(a) == len(b) == 5000
    assert set(a).isdisjoint(b)
    assert sorted(np.concatenate([a, b]).tolist()) == list(range(10000))
    again = fnorms.split_half_indices(10000, seed=0)
    assert np.array_equal(a, again[0]) and np.array_equal(b, again[1])
    assert not np.array_equal(a, fnorms.split_half_indices(10000, seed=1)[0])


def test_split_half_frequency_returns_two_maps_of_the_right_halves():
    rng = np.random.RandomState(0)
    ind = rng.uniform(size=(101, N_PATCHES)) < 0.1
    fa, fb = fnorms.split_half_frequency(ind, seed=0)
    a, b = fnorms.split_half_indices(101, seed=0)
    assert np.allclose(fa, ind[a].mean(axis=0))
    assert np.allclose(fb, ind[b].mean(axis=0))
    assert len(a) == 51 and len(b) == 50


# ─────────────────────────────────────────────────────────────────────────────
# 10. The conditions file and the job file
# ─────────────────────────────────────────────────────────────────────────────

def test_the_conditions_file_declares_exactly_two_conditions():
    from saga.frozen.runner import load_conditions
    doc = load_conditions(CONDITIONS_YAML)
    assert doc["work_package"] == "I1_spatial"
    assert doc["stage"] == "s11_out", "D1 is CLOSED at s11_out"
    ids = [c["id"] for c in doc["conditions"]]
    assert ids == ["native", "term_1.00"]
    by_id = {c["id"]: c for c in doc["conditions"]}
    assert by_id["native"]["stages"] == ["s11_out", "in_b07", "in_b08", "hist"]
    assert by_id["term_1.00"]["stages"] == ["hist"]
    assert by_id["term_1.00"]["applies_to"] == ["saga"]
    assert doc["calibration_stages"] == ["s11_out", "in_b07", "in_b08", "hist"]


def test_the_thresholds_path_is_a_per_split_template_and_never_the_canon_file():
    from saga.frozen import diag as fdiag
    from saga.frozen.runner import load_conditions
    doc = load_conditions(CONDITIONS_YAML)
    path = fdiag.thresholds_cal_path(doc["thresholds_cal"], "discovery")
    assert path == Path("results/frozen/I1_spatial/discovery/"
                        "thresholds_cal.json")
    assert doc["thresholds_canon"] == fdiag.CANON_THRESHOLDS
    with pytest.raises(fdiag.DiagError, match="forbids recalibrating"):
        fdiag.write_thresholds_cal(fdiag.CANON_THRESHOLDS, {})


@pytest.mark.skipif(not MANIFEST.exists(), reason="no manifest in this checkout")
def test_the_job_file_lists_the_cohort_in_array_order():
    from saga.frozen.runner import eligible_run_ids
    text = JOB_FILE.read_text(encoding="utf-8")
    listed = [line.split('"')[1] for line in text.splitlines()
              if line.strip().startswith('"') and line.count('"') >= 2]
    assert listed == eligible_run_ids(str(MANIFEST))
    assert "--array=0-18" in text


def test_the_job_file_takes_the_split_as_an_argument_with_no_default():
    text = JOB_FILE.read_text(encoding="utf-8")
    assert 'SPLIT="${1:?' in text, "the split must have no default"
    assert "val_diag_split.json" in text and "evaluation.json" in text


JOB_FILES = sorted((REPO / "scripts" / "jobs").glob("frozen_*.sbatch"))


def _working_bash():
    """A bash that can actually run, or None.

    `shutil.which("bash")` on Windows finds WSL's stub first, which on this
    machine cannot start at all (`execvpe(/bin/bash) failed`). Git Bash is
    the one that works. Probe rather than trust the PATH, so the test skips
    where there is no usable bash instead of failing on a broken one.
    """
    candidates = [shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe",
                  "/bin/bash", "/usr/bin/bash"]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            probe = subprocess.run([candidate, "-c", "exit 0"],
                                   capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return None


BASH = _working_bash()


@pytest.mark.skipif(BASH is None, reason="no usable bash on this machine")
@pytest.mark.parametrize("job", JOB_FILES, ids=lambda p: p.name)
def test_every_frozen_job_file_is_valid_bash(job):
    """`bash -n` on every job file.

    An sbatch script is submitted, queued, allocated a GPU and only THEN
    parsed. A syntax error costs a full scheduling round trip per array task
    and shows up as a 14-second FAILED with nothing written — which is
    exactly what happened to job 4262675 on 2026-09-16.
    """
    result = subprocess.run([BASH, "-n", str(job)], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, \
        f"{job.name} is not valid bash:\n{result.stderr}"


@pytest.mark.parametrize("job", JOB_FILES, ids=lambda p: p.name)
def test_no_apostrophe_inside_a_parameter_default_message(job):
    """The portable half of the test above, for a machine without bash.

    Inside `${var:?word}` bash RE-PARSES `word`, so a single quote there
    opens a quoted section even within double quotes. "D5's masks" in a
    usage message turned a whole job file into a syntax error at EOF.
    """
    text = job.read_text(encoding="utf-8")
    for match in re.finditer(r"\$\{\w+:[?-]((?:[^{}]|\n)*?)\}", text):
        assert "'" not in match.group(1), (
            f"{job.name}: an apostrophe inside ${{...:?...}} makes the file "
            f"a syntax error:\n  {match.group(1)[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. The tables
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_i2_results(tmp_path):
    """Fake maps for real cohort run_ids, in the layout
    `analysis/frozen_I1_spatial.py` reads.

    BOTH recipes are present: `T_I1c` is the mixup-vs-true-nomix contrast and
    a fixture with only mixup runs would let it be written empty without any
    test noticing.
    """
    from saga.frozen.diag import MapAccumulator, write_maps_npz

    root = tmp_path / "I2_terminal"
    rng = np.random.RandomState(0)
    rings = border_distance_map(SIDE).reshape(-1)
    shape = 0.02 + 0.10 * (rings == 1) + 0.04 * (rings == 0)
    runs = {"e2r_vits_mixup_baseline_s1": 1.0,
            "e2r_vits_mixup_baseline_s2": 0.9,
            "e2r_vits_nomix_baseline_s1": 0.3,
            "e2r_vits_nomix_baseline_s2": 0.25,
            "legacy_e2_vit_small_mixupdir_registers": 0.4,
            "e2r_vits_mixup_saga_s1": 0.6}
    for run_id, scale in runs.items():
        acc = MapAccumulator()
        for cond in ("native", "term_1.00"):
            if cond == "term_1.00" and "saga" not in run_id:
                continue
            for stage in ("s11_out", "hist"):
                if cond == "term_1.00" and stage != "hist":
                    continue
                p = np.clip(shape * scale, 0, 1)
                ind = rng.uniform(size=(200, N_PATCHES)) < p[None, :]
                acc.add(cond, stage, {"fixed_cal": ind, "mad": ind})
        write_maps_npz(root / "evaluation" / run_id, acc, {
            "run_id": run_id, "ckpt_sha256": "a" * 64, "git_sha": "c" * 40,
            "split_name": "evaluation", "split_sha256": EVAL_SHA,
            "tau_cal": {"s11_out": 18.7, "hist": 20.9}})
    return root


def test_the_tables_are_byte_identical_on_a_rebuild(fake_i2_results, tmp_path):
    """Two builds from the same inputs must produce the same bytes — no
    timestamp, no dict iteration order, no re-seeded null."""
    from analysis.frozen_I1_spatial import build

    out = {}
    for i in (1, 2):
        d = tmp_path / f"tables{i}"
        build("evaluation", manifest=str(MANIFEST), out_dir=d,
              git_sha_value="c" * 40, i2_root=fake_i2_results,
              i1_root=tmp_path / "absent", pack_dir=tmp_path / "absent")
        out[i] = {p.name: p.read_bytes() for p in sorted(d.glob("*.csv"))}
    assert out[1] == out[2]
    assert set(out[1]) == {
        "T_I1a_prevalence.csv", "T_I1b_scale_null.csv", "T_I1c_recipe.csv",
        "T_I1d_seed_stability.csv", "T_I1e_eta2.csv",
        "T_I1f_blockinput.csv", "T_I1g_variant_maps.csv"}


def test_every_table_carries_the_provenance_columns(fake_i2_results, tmp_path):
    import csv

    from analysis.frozen_I1_spatial import PROVENANCE, build

    d = tmp_path / "tables"
    build("evaluation", manifest=str(MANIFEST), out_dir=d,
          git_sha_value="c" * 40, i2_root=fake_i2_results,
          i1_root=tmp_path / "absent", pack_dir=tmp_path / "absent")
    for path in sorted(d.glob("*.csv")):
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        assert rows, f"{path.name} has no rows"
        for col in PROVENANCE:
            assert col in rows[0], f"{path.name} is missing {col}"
        assert all(r["split_sha256"] == EVAL_SHA for r in rows)


def test_the_pending_tables_name_the_artifact_they_are_waiting_for(
        fake_i2_results, tmp_path):
    """`PENDING B` is a VALUE, like MISSING: it is never averaged and never
    replaced by a guess, and it says what would fill it."""
    import csv

    from analysis.frozen_I1_spatial import build

    d = tmp_path / "tables"
    build("evaluation", manifest=str(MANIFEST), out_dir=d,
          git_sha_value="c" * 40, i2_root=fake_i2_results,
          i1_root=tmp_path / "absent", pack_dir=tmp_path / "absent")
    for name, needle in (("T_I1b_scale_null.csv", "norms_"),
                         ("T_I1f_blockinput.csv", "maps_in_b07.npz")):
        rows = list(csv.DictReader(open(d / name, encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["status"] == PENDING
        assert needle in rows[0]["needs"]

    eta = list(csv.DictReader(open(d / "T_I1e_eta2.csv", encoding="utf-8")))
    assert all(r["ci_lo"] == PENDING and r["ci_hi"] == PENDING for r in eta)
    assert all(float(r["eta2_pos"]) >= 0 for r in eta)


def test_the_tables_use_descriptive_wording_only(fake_i2_results, tmp_path):
    """TASK A §12: a correlation between two maps is a residual map
    correlation, not a relocation, an emptying or a sink function."""
    from analysis.frozen_I1_spatial import build

    d = tmp_path / "tables"
    build("evaluation", manifest=str(MANIFEST), out_dir=d,
          git_sha_value="c" * 40, i2_root=fake_i2_results,
          i1_root=tmp_path / "absent", pack_dir=tmp_path / "absent")
    banned = ("relocat", "empties", "emptying", "sink function",
              "sink_function")
    for path in sorted(d.glob("*.csv")):
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{path.name} says {word!r}"


@pytest.mark.skipif(
    not (REPO / "results/frozen/I2_terminal/evaluation").exists(),
    reason="I2's evaluation maps are not in this checkout")
def test_the_committed_tables_regenerate_from_the_committed_i2_maps(tmp_path):
    """The real tables, rebuilt from the real inputs, must match what is
    committed — apart from `git_sha`, which records the commit that generated
    them and necessarily moves."""
    import csv

    from analysis.frozen_I1_spatial import TABLE_DIR, build

    build("evaluation", out_dir=tmp_path, git_sha_value="PINNED")
    for path in sorted(tmp_path.glob("*.csv")):
        committed = TABLE_DIR / path.name
        assert committed.exists(), f"{path.name} is not committed"
        fresh = list(csv.DictReader(open(path, encoding="utf-8")))
        old = list(csv.DictReader(open(committed, encoding="utf-8")))
        assert len(fresh) == len(old), f"{path.name}: row count moved"
        for a, b in zip(fresh, old):
            a.pop("git_sha"), b.pop("git_sha")
            assert a == b, f"{path.name}: a row changed"
