"""
tests/test_I4_perturbation.py
=============================
TASK B / I4 Phase A′ — the conditions file, the D5 mask contract, the freeze
guard, per-image energy matching and C3, on CPU with tiny fake models and
fake data.

NOTHING HERE TOUCHES THE REAL MASK FILE OR THE REAL LOCKED DOCUMENT.
`configs/frozen/I4_masks.json` does not exist yet and D5 is still OPEN, which
is exactly the state the guard must refuse — so the loader is exercised
against a FAKE mask file written into `tmp_path`, and the freeze guard
against fake LOCKED documents in both states.

What it pins:

  - the 65 conditions, by role and by layer, and the refusal of a control
    declared before the primary it is matched to;
  - that masks load only through the contract, only from a file whose sha256
    the LOCKED document records, and that a control is refused unless it is
    ring-matched and the same size as its primary;
  - that prefix rows are untouched on a fake 5-prefix register model;
  - that a per-image `energy_target` is achieved within 1%, and that matched
    and unmatched controls differ ONLY in energy;
  - the freeze guard in both states, with the refusal naming what is missing;
  - C3 and every interpretation branch of TASK B §6 on synthetic records.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch
import yaml

from analysis import i34_contrasts as ic
from analysis.frozen_I4_analysis import (_condition_meta, _tag_of, build_a,
                                         build_c, scope_of)
from saga.frozen import masks as fmasks
from saga.frozen import records as rec
from saga.frozen.edits import receiver_perturbation
from saga.frozen.prevalence import (N_CONTROL_MASKS, PRIMARY_MASK_K,
                                    ring_composition, ring_indices)
from saga.frozen.runner import (RunnerError, load_conditions,
                                run_work_package_stages)
from saga.vit import build_saga_vit
from tools.frozen_eval import assert_not_embargoed

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = REPO / "configs" / "frozen" / "I4_perturbation.yaml"
JOB_FILE = REPO / "scripts" / "jobs" / "frozen_I4.sbatch"
MANIFEST = REPO / "results" / "frozen" / "I0_manifest" / "manifest.json"
TASK_FILE = REPO / "docs" / "TASK_B_I3_I4.md"

SIDE = 14
N_POSITIONS = SIDE * SIDE
LAYERS = (7, 8)

DEPTH = 4
DIM = 24
HEADS = 3


# ─────────────────────────────────────────────────────────────────────────────
# A fake D5 mask file, and a fake freeze
# ─────────────────────────────────────────────────────────────────────────────

#: The discovery split's CANONICAL digest, which is what
#: `analysis/build_D5_masks.py` stamps into the file. TASK A / I1 Phase C
#: found that the discovery split predates `build_frozen_splits.py` and
#: records no sha of its own, so the FILE digest (`0a686340…`) and this one
#: differ; their allow-list accepts both.
DISCOVERY_CANONICAL = (
    "bcb2a4c5a8f5335a71f7abef3c3a5dc028acd4b96aa76177e66360422186f8a5")


def _fake_masks_doc(side=SIDE, k=PRIMARY_MASK_K, n_controls=N_CONTROL_MASKS):
    """A fake mask file in the REAL `i4_masks_v1` schema.

    Keyed by STAGE with `primary_mask.index` and `controls[j].index`, the
    shape `analysis/build_D5_masks.py` actually writes — not the flat
    `{mask_id: [...]}` this fixture used before the real file landed on
    2026-09-17. A fixture in a schema nothing produces tests nothing.

    The primary takes the first `k` positions of ring 1 — the ring TASK-07
    found the sinks on — and each control takes a different slice of the SAME
    ring, so every control is ring-matched without the selection machinery.
    """
    ring1 = ring_indices(side, 1).tolist()
    assert len(ring1) >= k * 2, "ring 1 is too small for this fixture"
    primary = sorted(ring1[:k])
    rest = ring1[k:]
    masks = {}
    for layer in LAYERS:
        stage = fmasks.STAGE_FOR_LAYER[layer]
        controls = []
        for j in range(n_controls):
            start = (j * 2) % max(1, len(rest) - k)
            take = rest[start:start + k]
            take = take + rest[:k - len(take)] if len(take) < k else take
            controls.append({"index": sorted(take), "seed": 200 + j,
                             "overlap_with_primary": 0})
        masks[stage] = {
            "stage": stage, "grid_side": side,
            "block_entered_0based": layer, "paper_block_1based": layer + 1,
            "discovery_split_sha256": DISCOVERY_CANONICAL,
            "tau_cal": 14.0, "n_images": 10000, "n_positions": side * side,
            "n_prefix": 1, "source_run_ids": ["fake_baseline_s1"],
            "primary_mask": {"index": primary,
                             "ring_composition": {"1": k}},
            "controls": controls,
        }
    return {"schema": fmasks.SCHEMA, "work_package": "I1_spatial",
            "masks": masks, "cell": "vit_small|mixup", "basis": "fixed_cal",
            "git_sha": "c" * 40, "k": k, "n_controls": n_controls,
            "control_seeds": list(range(200, 200 + n_controls)),
            "discovery_split_name": "val_diag_split",
            "generated_by": "tests/test_I4_perturbation.py"}


@pytest.fixture
def fake_masks(tmp_path):
    p = tmp_path / "I4_masks.json"
    p.write_text(json.dumps(_fake_masks_doc(), indent=2, sort_keys=True),
                 encoding="utf-8", newline="\n")
    return p


def _index_of(doc, mask_id):
    """The `index` list of one mask id inside the nested `i4_masks_v1` doc."""
    for stage, block in doc["masks"].items():
        layer = fmasks.BLOCK_INPUT_STAGES[stage]
        if mask_id == fmasks.primary_id(layer):
            return block["primary_mask"]
        for j, ctrl in enumerate(block["controls"]):
            if mask_id == fmasks.control_id(j, layer):
                return ctrl
    raise KeyError(mask_id)


def _rewrite(tmp_path, doc, name):
    """Write a mutated doc and a matching frozen LOCKED for it."""
    p = tmp_path / name
    p.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
                 newline="\n")
    return p, _locked(tmp_path, frozen=True, masks_path=p,
                      name=name + ".LOCKED.md")


def _locked(tmp_path, *, frozen: bool, masks_path=None, name="LOCKED.md"):
    """A fake LOCKED_ANALYSIS.md in one of the two states."""
    if not frozen:
        body = ("# LOCKED\n\n> ## STATUS: **DRAFT — NOT YET FROZEN**\n>\n"
                "> Date frozen: `PENDING`\n"
                "> Git sha at freeze: `PENDING`\n\n"
                "| D5 | 5 | the map | **OPEN** — blocks I4 |\n")
    else:
        digest = fmasks.file_sha256(masks_path)
        body = ("# LOCKED\n\n> ## STATUS: FROZEN\n>\n"
                "> Date frozen: `2026-09-17`\n"
                "> Git sha at freeze: `abc1234`\n\n"
                f"| ~~D5~~ | 5 | closed | **CLOSED** — "
                f"`configs/frozen/I4_masks.json`, sha256 `{digest}` |\n")
    p = tmp_path / name
    p.write_text(body, encoding="utf-8", newline="\n")
    return p


@pytest.fixture
def frozen_locked(tmp_path, fake_masks):
    return _locked(tmp_path, frozen=True, masks_path=fake_masks)


# ─────────────────────────────────────────────────────────────────────────────
# The freeze guard — both states
# ─────────────────────────────────────────────────────────────────────────────

def test_the_guard_refuses_while_the_document_is_a_draft(tmp_path, fake_masks):
    draft = _locked(tmp_path, frozen=False)
    with pytest.raises(fmasks.MaskError) as exc:
        fmasks.assert_frozen(draft, masks_path=fake_masks)
    msg = str(exc.value)
    assert "EMBARGOED" in msg
    # the refusal NAMES what is missing, all of it at once
    assert "STATUS: FROZEN" in msg
    assert "Date frozen" in msg
    assert "Git sha at freeze" in msg
    assert "~~D5~~" in msg


def test_the_guard_passes_once_the_document_is_frozen(frozen_locked,
                                                      fake_masks):
    state = fmasks.assert_frozen(frozen_locked, masks_path=fake_masks)
    assert state["missing"] == []
    assert state["frozen"] is True
    assert state["d5_closed"] is True
    assert state["date"] == "2026-09-17"
    assert state["git_sha"] == "abc1234"
    assert state["masks_sha"] == fmasks.file_sha256(fake_masks)


def test_the_guard_refuses_a_mask_file_that_changed_after_the_freeze(
        tmp_path, fake_masks, frozen_locked):
    """The failure this check exists for is not a missing file — it is a file
    quietly regenerated between the freeze and the run."""
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    entry = _index_of(doc, fmasks.primary_id(7))
    entry["index"] = sorted(entry["index"][:-1] + [195])
    fake_masks.write_text(json.dumps(doc, indent=2, sort_keys=True),
                          encoding="utf-8", newline="\n")
    with pytest.raises(fmasks.MaskError, match="does not record the digest"):
        fmasks.assert_frozen(frozen_locked, masks_path=fake_masks)


def test_the_guard_refuses_a_missing_mask_file(tmp_path):
    locked = tmp_path / "L.md"
    locked.write_text("STATUS: FROZEN\nDate frozen: `2026-09-17`\n"
                      "Git sha at freeze: `abc1234`\n~~D5~~\n",
                      encoding="utf-8")
    with pytest.raises(fmasks.MaskError, match="does not exist"):
        fmasks.assert_frozen(locked, masks_path=tmp_path / "absent.json")


def test_frozen_eval_embargoes_I4_and_only_I4(tmp_path, fake_masks,
                                              frozen_locked):
    """TASK B §7: the guard is keyed on the FILE NAME, before the document is
    parsed, so a malformed I4 file is refused for the right reason."""
    assert assert_not_embargoed("configs/frozen/I3_gate_edits.yaml") is None
    assert assert_not_embargoed("configs/frozen/I2_terminal.yaml") is None
    assert assert_not_embargoed("configs/frozen/smoke.yaml") is None
    state = assert_not_embargoed("configs/frozen/I4_perturbation.yaml",
                                 locked_path=frozen_locked,
                                 masks_path=fake_masks)
    assert state["missing"] == []
    with pytest.raises(fmasks.MaskError, match="EMBARGOED"):
        assert_not_embargoed("configs/frozen/I4_anything.yaml",
                             locked_path=_locked(tmp_path, frozen=False,
                                                 name="D.md"),
                             masks_path=fake_masks)


REAL_MASKS = REPO / "configs" / "frozen" / "I4_masks.json"


@pytest.mark.skipif(not REAL_MASKS.exists(),
                    reason="D5's mask file is not in this checkout")
def test_the_REAL_d5_file_satisfies_this_contract():
    """The cross-track check. TASK A / I1 Phase C writes this file and I4
    reads it; nothing else verifies that the two agree.

    It landed on 2026-09-17 keyed by STAGE (`in_b07`, `in_b08`) with
    `primary_mask.index` and `controls[j].index`, not by the flat
    `{mask_id: [...]}` this loader first assumed — so `_flatten` exists, and
    this test is what would have caught the mismatch before an HPC job did.
    """
    loaded = fmasks.load_masks(REAL_MASKS, require_freeze=False)
    assert set(loaded["masks"]) == set(fmasks.expected_mask_ids())
    assert loaded["grid_side"] == SIDE
    for mask_id, idx in loaded["masks"].items():
        assert idx.size == PRIMARY_MASK_K, mask_id
        assert idx.min() >= 0 and idx.max() < N_POSITIONS, mask_id
    # every control ring-matched to its primary — enforced by the loader,
    # asserted here against the REAL draws
    for layer in LAYERS:
        primary = loaded["ring_composition"][fmasks.primary_id(layer)]
        assert primary == {1: 14, 2: 2}, primary
        for j in range(N_CONTROL_MASKS):
            assert loaded["ring_composition"][
                fmasks.control_id(j, layer)] == primary


@pytest.mark.skipif(not REAL_MASKS.exists(),
                    reason="D5's mask file is not in this checkout")
def test_the_REAL_d5_file_reproduces_track_As_reported_numbers():
    """Independent reproduction of the three numbers TASK A's Phase C log
    reports for D5. Two modules built from the same file by different code
    must agree, or one of them is wrong."""
    loaded = fmasks.load_masks(REAL_MASKS, require_freeze=False)
    # 1. the digest LOCKED will be signed against
    assert loaded["sha256"] == (
        "72612357be7dde3925b3a312b20964c826234396c165580d55f21e10e6bfe96e")
    # 2. both primaries are the SAME 16 coordinates, so I4's two sites differ
    #    in DEPTH alone with the address held fixed
    assert loaded["primaries_identical"] is True
    assert loaded["masks"]["P_L7"].tolist() == [
        15, 16, 17, 20, 25, 26, 29, 30, 39, 40, 43, 54, 166, 169, 179, 180]
    # 3. control overlap with the primary sits in 2..6 around the analytic
    #    expectation 4.566 — reported, never used to reject a draw (LOCKED §5)
    for layer in LAYERS:
        overlap = list(fmasks.mask_overlap(loaded["masks"], layer).values())
        assert len(overlap) == N_CONTROL_MASKS
        assert min(overlap) >= 2 and max(overlap) <= 6, overlap


@pytest.mark.skipif(not REAL_MASKS.exists(),
                    reason="D5's mask file is not in this checkout")
def test_the_real_file_is_still_refused_while_locked_is_a_draft():
    """The file existing is NOT the freeze. D5 must also be closed in
    LOCKED_ANALYSIS with this digest, and the header signed."""
    with pytest.raises(fmasks.MaskError, match="EMBARGOED"):
        fmasks.load_masks(REAL_MASKS, require_freeze=True)


def test_a_file_in_an_unknown_schema_is_refused(tmp_path, fake_masks):
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    doc["schema"] = "i4_masks_v2"
    p, locked = _rewrite(tmp_path, doc, "v2.json")
    with pytest.raises(fmasks.MaskError, match="declares schema"):
        fmasks.load_masks(p, locked_path=locked)


def test_a_stage_that_disagrees_about_its_block_is_refused(tmp_path,
                                                           fake_masks):
    """The off-by-one that would silently perturb the wrong block."""
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    doc["masks"]["in_b07"]["block_entered_0based"] = 8
    p, locked = _rewrite(tmp_path, doc, "offby1.json")
    with pytest.raises(fmasks.MaskError, match="says it enters block"):
        fmasks.load_masks(p, locked_path=locked)


def test_a_mask_file_built_from_reporting_data_is_refused(tmp_path,
                                                          fake_masks):
    """Masks are SELECTED on discovery. A file naming the evaluation split
    is refused here as firmly as prevalence.py would refuse to build it."""
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    doc["masks"]["in_b07"]["discovery_split_sha256"] = (
        "7fdf5f9f2ace98ef03a6267455daa1104b92b6689c510ab8b1e5420340f27014")
    p, locked = _rewrite(tmp_path, doc, "evalsplit.json")
    with pytest.raises(fmasks.MaskError, match="EVALUATION"):
        fmasks.load_masks(p, locked_path=locked)


def test_the_real_repository_is_still_embargoed_today():
    """D5 is OPEN and LOCKED is a DRAFT on this checkout. If this test ever
    fails, the freeze has happened and I4 Phase B may run."""
    state = fmasks.locked_state()
    assert state["missing"], (
        "docs/LOCKED_ANALYSIS.md appears to be FROZEN with D5 closed — "
        "update this test and run I4 Phase B")
    assert state["frozen"] is False


# ─────────────────────────────────────────────────────────────────────────────
# The mask contract
# ─────────────────────────────────────────────────────────────────────────────

def test_the_masks_load_through_the_contract(fake_masks, frozen_locked):
    loaded = fmasks.load_masks(fake_masks, locked_path=frozen_locked)
    assert set(loaded["masks"]) == set(fmasks.expected_mask_ids())
    assert len(loaded["masks"]) == 2 * (1 + N_CONTROL_MASKS) == 22
    assert loaded["grid_side"] == SIDE
    assert loaded["sha256"] == fmasks.file_sha256(fake_masks)
    for mask_id, idx in loaded["masks"].items():
        assert idx.size == PRIMARY_MASK_K, mask_id
        assert np.array_equal(idx, np.sort(idx)), mask_id
        assert idx.min() >= 0 and idx.max() < N_POSITIONS


def test_every_control_is_ring_matched_to_its_primary(fake_masks,
                                                      frozen_locked):
    loaded = fmasks.load_masks(fake_masks, locked_path=frozen_locked)
    comps = loaded["ring_composition"]
    for layer in LAYERS:
        primary = comps[fmasks.primary_id(layer)]
        for j in range(N_CONTROL_MASKS):
            assert comps[fmasks.control_id(j, layer)] == primary


def test_a_control_that_is_not_ring_matched_is_refused(tmp_path, fake_masks):
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    centre = [q for q in range(N_POSITIONS)
              if ring_composition([q], SIDE) == {6: 1}]
    entry = _index_of(doc, fmasks.control_id(0, 7))
    entry["index"] = sorted(entry["index"][:-1] + centre[:1])
    p, locked = _rewrite(tmp_path, doc, "bad_masks.json")
    with pytest.raises(fmasks.MaskError, match="not ring-matched"):
        fmasks.load_masks(p, locked_path=locked)


def test_a_control_of_a_different_size_is_refused(tmp_path, fake_masks):
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    entry = _index_of(doc, fmasks.control_id(1, 8))
    entry["index"] = entry["index"][:-1]
    p, locked = _rewrite(tmp_path, doc, "short.json")
    with pytest.raises(fmasks.MaskError, match="coordinates but"):
        fmasks.load_masks(p, locked_path=locked)


def test_a_missing_mask_id_is_refused(tmp_path, fake_masks):
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    doc["masks"]["in_b08"]["controls"].pop()
    p, locked = _rewrite(tmp_path, doc, "gap.json")
    with pytest.raises(fmasks.MaskError, match="9 controls, expected 10"):
        fmasks.load_masks(p, locked_path=locked)


def test_a_token_indexed_mask_is_refused(tmp_path, fake_masks):
    """The exact bug the guard exists for: a mask written against TOKEN
    indices would silently perturb a CLS or register row."""
    doc = json.loads(fake_masks.read_text(encoding="utf-8"))
    entry = _index_of(doc, fmasks.primary_id(7))
    entry["index"] = sorted(entry["index"][:-1] + [N_POSITIONS + 3])
    p, locked = _rewrite(tmp_path, doc, "tok.json")
    with pytest.raises(fmasks.MaskError, match="PATCH coordinates"):
        fmasks.load_masks(p, locked_path=locked)


def test_the_overlap_with_the_primary_is_reported_not_rejected(fake_masks,
                                                               frozen_locked):
    """LOCKED §5: draws are never rejected to amplify contrast."""
    loaded = fmasks.load_masks(fake_masks, locked_path=frozen_locked)
    overlap = fmasks.mask_overlap(loaded["masks"], 7)
    assert set(overlap) == {fmasks.control_id(j, 7)
                            for j in range(N_CONTROL_MASKS)}
    assert all(isinstance(v, int) and v >= 0 for v in overlap.values())


# ─────────────────────────────────────────────────────────────────────────────
# The conditions file
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def conditions(fake_masks, frozen_locked):
    """The REAL conditions file, resolved against the FAKE mask file."""
    return load_conditions(CONDITIONS, masks_path=fake_masks,
                           locked_path=frozen_locked)


def test_the_conditions_file_declares_exactly_65_conditions(conditions):
    conds = conditions["conditions"]
    assert len(conds) == 65
    assert len({c["id"] for c in conds}) == 65
    assert conds[0]["id"] == "native"
    assert conds[0]["edit_type"] == "native"
    assert conditions["work_package"] == "I4_perturbation"
    assert conditions["stage"] == "s11_out"
    assert conditions["bootstrap_seed"] == 0
    assert conditions["bootstrap_resamples"] == 10000
    assert conditions["energy_rel_error_tol"] == 0.01


def test_the_condition_roles_are_the_declared_counts(conditions):
    counts = {}
    for c in conditions["conditions"]:
        meta = _condition_meta(c["id"])
        assert meta is not None, c["id"]
        key = (meta["role"], meta["layer"], meta["eps"], meta["kind"])
        counts[key] = counts.get(key, 0) + 1
    for layer in LAYERS:
        assert counts[("primary", layer, "10", "fixed_epsilon")] == 1
        assert counts[("primary", layer, "25", "fixed_epsilon")] == 1
        assert counts[("control", layer, "10", "energy_matched")] == 10
        assert counts[("control", layer, "25", "energy_matched")] == 10
        assert counts[("control", layer, "10", "fixed_epsilon")] == 10
    assert sum(counts.values()) == 65


def test_only_three_conditions_record_the_patch_diagnostics(conditions):
    from saga.frozen.runner import condition_diag
    recording = [c["id"] for c in conditions["conditions"]
                 if condition_diag(conditions, c)]
    assert recording == ["native", "prim_e10_L7", "prim_e10_L8"]


def test_every_condition_applies_to_every_variant(conditions):
    """Unlike a gate edit, a receiver perturbation needs no gate: the cell's
    4 baseline, 4 SAGA and 2 register checkpoints run the same 65."""
    for c in conditions["conditions"]:
        assert c["applies_to"] == ["baseline", "registers", "saga"], c["id"]


def test_every_mask_resolves_to_the_committed_coordinates(conditions,
                                                          fake_masks,
                                                          frozen_locked):
    loaded = fmasks.load_masks(fake_masks, locked_path=frozen_locked)
    n = 0
    for c in conditions["conditions"]:
        mask_id = (c.get("params") or {}).get("mask")
        if mask_id is None:
            continue
        n += 1
        assert np.array_equal(c["_mask"], loaded["masks"][mask_id]), c["id"]
    assert n == 64
    assert conditions["masks_sha256"] == loaded["sha256"]


def test_the_resolved_mask_stays_out_of_the_recorded_params(conditions):
    c = [x for x in conditions["conditions"] if x["id"] == "ctrl3_e10m_L7"][0]
    blob = json.dumps(c["params"], sort_keys=True, separators=(",", ":"),
                      default=str)
    assert blob == ('{"energy_match":"prim_e10_L7","epsilon":0.1,'
                    '"layer":7,"mask":"K3_L7"}')


def test_a_control_matched_to_a_later_primary_is_refused(tmp_path, fake_masks,
                                                         frozen_locked):
    """The target is the primary's MEASURED norm, so the primary must run
    first — declaration order is what guarantees it."""
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    conds = doc["conditions"]
    i = next(k for k, c in enumerate(conds) if c["id"] == "prim_e10_L7")
    j = next(k for k, c in enumerate(conds) if c["id"] == "ctrl0_e10m_L7")
    conds[i], conds[j] = conds[j], conds[i]
    p = tmp_path / "order.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(RunnerError, match="declared later"):
        load_conditions(p, masks_path=fake_masks, locked_path=frozen_locked)


def test_a_control_matched_to_nothing_is_refused(tmp_path, fake_masks,
                                                 frozen_locked):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    for c in doc["conditions"]:
        if c["id"] == "ctrl0_e10m_L7":
            c["params"]["energy_match"] = "prim_e99_L7"
    p = tmp_path / "ghost.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(RunnerError, match="not declared at all"):
        load_conditions(p, masks_path=fake_masks, locked_path=frozen_locked)


def test_the_conditions_file_cannot_be_parsed_without_the_freeze(tmp_path):
    """The embargo fires at PARSE time, on the login node, not after a job
    has staged 50,000 images."""
    with pytest.raises(RunnerError, match="EMBARGOED"):
        load_conditions(CONDITIONS,
                        masks_path=tmp_path / "nope.json",
                        locked_path=_locked(tmp_path, frozen=False,
                                            name="Draft.md"))


# ─────────────────────────────────────────────────────────────────────────────
# The perturbation itself
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def saga_model():
    torch.manual_seed(0)
    return build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=10,
                          depth=DEPTH, embed_dim=DIM, num_heads=HEADS).eval()


@pytest.fixture(scope="module")
def reg4_model():
    """The timm 4-register model: 5 prefix rows, NOT wrapped in SAGAViT."""
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=10, reg_tokens=4, depth=DEPTH,
                             embed_dim=DIM, num_heads=HEADS).eval()


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(1)
    return torch.randn(4, 3, 224, 224)


def test_prefix_rows_are_untouched_on_a_five_prefix_model(reg4_model, images):
    """TASK B §5/§8. The register rows must be bit-identical while a real
    amount of energy is injected into the patch rows — a PASS that came from
    an edit doing nothing would be worthless, so the injected norm is
    asserted positive in the same test."""
    from saga.metrics import infer_num_prefix_tokens
    assert infer_num_prefix_tokens(reg4_model) == 5

    cap = {}
    blk = reg4_model.blocks[1]
    h = blk.register_forward_hook(
        lambda m, i, out: cap.setdefault("out", []).append(out.detach().clone()))
    try:
        with torch.no_grad():
            reg4_model(images)
            with receiver_perturbation(reg4_model, 1,
                                       np.arange(8), 0.5) as record:
                reg4_model(images)
    finally:
        h.remove()

    native, edited = cap["out"][0], cap["out"][1]
    assert torch.equal(native[:, :5, :], edited[:, :5, :]), \
        "a prefix row moved"
    assert float((native[:, 5:, :] - edited[:, 5:, :]).abs().max()) > 0
    assert min(record["measured_perturbation_norm"]) > 0
    assert record["n_masked_coords"] == 8


def test_a_per_image_energy_target_is_achieved_within_one_percent(saga_model,
                                                                  images):
    """TASK B §5/§8: the matched controls are only controls if they inject
    what they were asked for."""
    mask_primary = np.arange(16)
    mask_control = np.arange(30, 46)

    with torch.no_grad():
        with receiver_perturbation(saga_model, 1, mask_primary, 0.10) as prim:
            saga_model(images)
        target = list(prim["measured_perturbation_norm"])
        assert len(target) == images.shape[0]
        assert all(t > 0 for t in target)

        with receiver_perturbation(saga_model, 1, mask_control, 0.10,
                                   energy_target=target) as ctrl:
            saga_model(images)

    got = ctrl["measured_perturbation_norm"]
    rel = [abs(g - t) / t for g, t in zip(got, target)]
    assert max(rel) < 0.01, rel
    # and it is not trivially matched because the masks are the same size:
    # the UNMATCHED control at the same epsilon injects something different
    with torch.no_grad():
        with receiver_perturbation(saga_model, 1, mask_control, 0.10) as fixed:
            saga_model(images)
    assert max(abs(f - t) / t for f, t in
               zip(fixed["measured_perturbation_norm"], target)) > 0.01


def test_matched_and_unmatched_controls_differ_only_in_energy(saga_model,
                                                              images):
    """Same mask, same layer, same site — only alpha differs, and alpha is a
    scalar per image. So the ratio of the two injected norms is the ratio of
    the two alphas, exactly."""
    mask = np.arange(30, 46)
    with torch.no_grad():
        with receiver_perturbation(saga_model, 1, np.arange(16), 0.10) as prim:
            saga_model(images)
        target = list(prim["measured_perturbation_norm"])
        with receiver_perturbation(saga_model, 1, mask, 0.10,
                                   energy_target=target) as m:
            saga_model(images)
        with receiver_perturbation(saga_model, 1, mask, 0.10) as f:
            saga_model(images)
    for i in range(images.shape[0]):
        ratio_norm = (m["measured_perturbation_norm"][i]
                      / f["measured_perturbation_norm"][i])
        ratio_alpha = m["alpha"][i] / f["alpha"][i]
        assert ratio_norm == pytest.approx(ratio_alpha, rel=1e-5)
        assert m["native_masked_norm"][i] == pytest.approx(
            f["native_masked_norm"][i], rel=1e-6)


def test_a_zero_norm_image_gets_zero_perturbation_and_is_counted(saga_model,
                                                                 images):
    """LOCKED §6: zero-norm images receive zero perturbation and their
    frequency is REPORTED; they are never discarded."""
    with torch.no_grad():
        with receiver_perturbation(saga_model, 1, np.arange(16), 0.10,
                                   energy_target=[0.0] * images.shape[0]) as r:
            saga_model(images)
    assert all(a == 0.0 for a in r["alpha"])
    assert all(v == 0.0 for v in r["measured_perturbation_norm"])
    assert "n_zero_norm_images" in r


def test_energy_target_already_accepts_a_per_image_tensor():
    """B2b asked for `energy_target` to be extended from a scalar to a
    per-image tensor. I0 already shipped both, so nothing was changed — this
    pins that it stays true."""
    from saga.frozen.edits import EditError
    import inspect
    src = inspect.getsource(receiver_perturbation)
    assert "kappa.numel() not in (1, norm.numel())" in src
    assert "kappa.expand_as(norm) if kappa.numel() == 1 else kappa" in src
    assert EditError is not None


# ─────────────────────────────────────────────────────────────────────────────
# The runner, end to end
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


def _read(out_dir, kind):
    import csv
    path = rec.records_path(out_dir, kind)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_sweep_records_the_energy_columns_and_restores_the_model(
        tmp_path, saga_model, fake_masks, frozen_locked, monkeypatch):
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    keep = {"native", "prim_e10_L7", "ctrl0_e10m_L7", "ctrl0_e10f_L7"}
    doc["conditions"] = [c for c in doc["conditions"] if c["id"] in keep]
    for c in doc["conditions"]:                  # a 4-block fake model
        if (c.get("params") or {}).get("layer") is not None:
            c["params"]["layer"] = 1
    p = tmp_path / "small.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    conds = load_conditions(p, masks_path=fake_masks,
                            locked_path=frozen_locked)

    from saga.run_registry import file_sha256
    ckpt = tmp_path / "last.pth"
    torch.save({"model": saga_model.state_dict()}, ckpt)
    row = {"run_id": "fake_saga", "arch": "vit_small",
           "recipe_actual": "mixup", "variant": "saga", "ckpt_kind": "last",
           "ckpt_sha256": file_sha256(ckpt), "ckpt_path": str(ckpt),
           "gate_mode": "spatial"}
    monkeypatch.setattr("saga.frozen.runner.build_from_row",
                        lambda r, ckpt_path=None, device="cpu": saga_model)

    out = tmp_path / "out"
    summary = run_work_package_stages(
        row=row, conditions=conds, dataset=_FakeSet(), out_dir=out,
        split_name="fake", split_sha="d" * 64, git_sha="e" * 40,
        git_dirty="0", batch_size=2,
        tau_cal_doc={"split_sha256": "d" * 64,
                     "tau_cal": {"vit_small|mixup": {"s11_out": 1.0}}},
        canon_doc={})

    assert summary["state_restored"] is True
    assert summary["masks_sha256"] == conds["masks_sha256"]
    assert summary["energy_rel_error_max"] < 0.01

    rows = _read(out, "records")
    assert set(rows[0]) == set(rec.RECORD_PERTURBATION_COLUMNS)
    by = {}
    for r in rows:
        by.setdefault(r["condition_id"], []).append(r)
    assert set(by) == keep

    # native: no mask, no epsilon, no target
    for r in by["native"]:
        assert r["epsilon"] == "MISSING"
        assert r["measured_perturbation_norm"] == "MISSING"
        assert r["energy_target"] == "MISSING"
        assert r["mask_id"] == "MISSING"
    # the primary measures a norm but has no target
    for r in by["prim_e10_L7"]:
        assert float(r["measured_perturbation_norm"]) > 0
        assert r["energy_target"] == "MISSING"
        assert r["mask_id"] == "P_L7"
        assert r["n_masked_coords"] == str(PRIMARY_MASK_K)
    # the MATCHED control carries a target and hits it
    tgt = {r["image_id"]: float(r["measured_perturbation_norm"])
           for r in by["prim_e10_L7"]}
    for r in by["ctrl0_e10m_L7"]:
        assert float(r["energy_target"]) == pytest.approx(
            tgt[r["image_id"]], rel=1e-9)
        assert float(r["energy_rel_error"]) < 0.01
        assert r["energy_match"] == "prim_e10_L7"
    # the UNMATCHED control has neither
    for r in by["ctrl0_e10f_L7"]:
        assert r["energy_target"] == "MISSING"
        assert r["energy_rel_error"] == "MISSING"
        assert r["energy_match"] == "MISSING"
        assert float(r["measured_perturbation_norm"]) > 0

    diag = _read(out, "diag")
    assert {r["condition_id"] for r in diag} == {"native", "prim_e10_L7"}


# ─────────────────────────────────────────────────────────────────────────────
# C3 and the interpretation branches
# ─────────────────────────────────────────────────────────────────────────────

def _synth(run, variant, prim_shift, ctrl_shift, n=220, seed=0):
    rng = np.random.RandomState(seed)
    ids = [f"n{i // 10:08d}/img_{i}" for i in range(n)]
    base = rng.rand(n) * 2.0
    rows = []

    def add(cond, shift, mask_id, injected):
        for i, img in enumerate(ids):
            nll = base[i] + shift + rng.randn() * 0.01
            rows.append({
                "run_id": run, "arch": "vit_small", "recipe_actual": "mixup",
                "variant": variant, "ckpt_kind": "last",
                "ckpt_sha256": "a" * 64, "split_name": "evaluation",
                "split_sha256": "b" * 64, "stage": "s11_out",
                "precision": "fp32", "git_sha": "c" * 40,
                "masks_sha256": "d" * 64, "condition_id": cond,
                "image_id": img, "nll": nll,
                "correct": 1 if nll < 1.5 else 0,
                "max_abs_logit_diff_vs_native": 0.0 if injected is None else 0.4,
                "mask_id": mask_id, "n_masked_coords": "16",
                "measured_perturbation_norm": ("MISSING" if injected is None
                                               else injected),
                "energy_target": ("MISSING" if injected is None else injected),
                "energy_rel_error": ("MISSING" if injected is None
                                     else 0.0001),
            })

    add("native", 0.0, "MISSING", None)
    for layer in LAYERS:
        add(f"prim_e10_L{layer}", prim_shift, f"P_L{layer}", 2.0)
        add(f"prim_e25_L{layer}", prim_shift * 2, f"P_L{layer}", 5.0)
        for j in range(10):
            add(f"ctrl{j}_e10m_L{layer}", ctrl_shift + 0.002 * j,
                f"K{j}_L{layer}", 2.0)
            add(f"ctrl{j}_e25m_L{layer}", ctrl_shift * 2, f"K{j}_L{layer}", 5.0)
            add(f"ctrl{j}_e10f_L{layer}", ctrl_shift * 0.8,
                f"K{j}_L{layer}", 1.7)
    return rows


def _runs(b_prim, s_prim, ctrl=0.05):
    out = {}
    for i, r in enumerate(ic.FRESH_BASELINE):
        out[r] = _synth(r, "baseline", b_prim, ctrl, seed=10 + i)
    for i, r in enumerate(ic.FRESH_SAGA):
        out[r] = _synth(r, "saga", s_prim, ctrl, seed=20 + i)
    return out


@pytest.mark.parametrize("b_prim,s_prim,branch", [
    (0.059, 0.059, "inside_band"),
    (0.40, 0.40, "all_methods"),
    (0.059, 0.40, "method_specific"),
    (0.0, 0.40, "unanticipated"),
])
def test_every_i4_interpretation_branch_is_reachable(b_prim, s_prim, branch):
    got = ic.i4_contrasts(_runs(b_prim, s_prim), layers=(7,),
                          resamples=400)[7]
    assert got["interpretation"]["branch"] == branch, \
        got["interpretation"]["verdicts"]
    assert got["interpretation"]["anticipated"] is (branch != "unanticipated")


def test_the_i4_branch_text_is_byte_identical_to_the_task_file():
    text = TASK_FILE.read_text(encoding="utf-8")
    for key in ("inside_band", "method_specific", "all_methods"):
        assert ic.I4_BRANCHES[key] in text, key
    assert ic.I4_GUIDE_CLOSE in text
    # I3's close is a DIFFERENT sentence and must not be reused
    assert ic.I4_GUIDE_CLOSE != ic.I3_GUIDE_CLOSE


def test_c3_reports_the_control_band_and_whether_the_primary_sits_in_it():
    grouped = ic.by_condition(_synth("x", "saga", 0.40, 0.05))
    out = ic.c3(grouped, 7, resamples=300)
    assert out["n_controls"] == 10
    assert out["control_band_lo"] <= out["theta_control_mean"] \
        <= out["control_band_hi"]
    assert out["primary_inside_band"] is False
    assert out["theta_primary"] > out["control_band_hi"]
    assert len(out["theta_control_per_mask"]) == 10

    inside = ic.by_condition(_synth("y", "saga", 0.059, 0.05))
    assert ic.c3(inside, 7, resamples=300)["primary_inside_band"] is True


def test_c3_refuses_a_partial_control_set():
    rows = [r for r in _synth("x", "saga", 0.3, 0.05)
            if r["condition_id"] != "ctrl7_e10m_L7"]
    with pytest.raises(ic.ContrastError, match="missing"):
        ic.c3(ic.by_condition(rows), 7, resamples=50)


def test_registers_are_labelled_and_never_decide():
    runs = _runs(0.059, 0.40)
    runs["legacy_e2_vit_small_mixupdir_registers"] = _synth(
        "legacy_e2_vit_small_mixupdir_registers", "registers", 9.0, 0.05,
        seed=99)
    got = ic.i4_contrasts(runs, layers=(7,), resamples=400)[7]
    assert got["registers"] == ["legacy_e2_vit_small_mixupdir_registers"]
    for method in ("baseline", "saga"):
        assert "legacy_e2_vit_small_mixupdir_registers" not in \
            got["C3"][method]["per_checkpoint"]
    assert got["interpretation"]["branch"] == "method_specific"


def test_a_run_whose_rows_disagree_about_its_variant_is_refused():
    rows = _synth("x", "saga", 0.3, 0.05, n=5)
    rows[0]["variant"] = "baseline"
    with pytest.raises(ic.ContrastError, match="one checkpoint is one"):
        ic.i4_contrasts({"x": rows}, layers=(7,), resamples=50)


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def test_T_I4a_covers_every_condition_and_flags_the_energy_tolerance():
    rows = build_a("e2r_vits_mixup_saga_s1",
                   _synth("e2r_vits_mixup_saga_s1", "saga", 0.3, 0.05, n=60),
                   resamples=200)
    assert len(rows) == 64                    # 65 minus `native`
    assert all(r["scope"] == "fresh" for r in rows)
    roles = {r["role"] for r in rows}
    assert roles == {"primary", "control"}
    matched = [r for r in rows if r["control_kind"] == "energy_matched"]
    assert len(matched) == 40
    assert all(r["energy_within_tolerance"] is True for r in matched)
    assert {r["layer"] for r in rows} == set(LAYERS)


def test_scope_labels_the_cell_correctly():
    assert scope_of("e2r_vits_mixup_baseline_s1", "baseline") == "fresh"
    assert scope_of("e2r_vits_mixup_saga_s2", "saga") == "fresh"
    assert scope_of("legacy_e2_vit_small_mixupdir_saga", "saga") == "legacy"
    assert scope_of("legacy_e2_vit_small_mixupdir_registers",
                    "registers") == "registers"


def test_the_cross_method_table_pairs_by_provenance_tag():
    assert _tag_of("e2r_vits_mixup_baseline_s1") == "s1"
    assert _tag_of("e2r_vits_mixup_saga_s2") == "s2"
    assert _tag_of("legacy_e2_vit_small_mixupdir_saga") == "legacy-mixupdir"
    loaded = {r: (rows, []) for r, rows in _runs(0.06, 0.30).items()}
    rows = build_c(loaded, layers=(7,), resamples=200)
    paired = [r for r in rows if r["comparison"] == "baseline_minus_saga"]
    assert {r["provenance_tag"] for r in paired} == {"s1", "s2"}
    for r in paired:
        assert r["paired_with"].endswith(r["provenance_tag"])
        assert r["paired_difference"] < 0      # saga theta is larger here


def test_the_i4_tables_use_descriptive_wording_only(tmp_path):
    from analysis.frozen_I4_analysis import T_A_FIELDS
    from analysis.frozen_I3_analysis import write_csv
    write_csv(tmp_path / "a.csv", T_A_FIELDS,
              build_a("e2r_vits_mixup_saga_s1",
                      _synth("e2r_vits_mixup_saga_s1", "saga", 0.3, 0.05,
                             n=30), resamples=100))
    banned = ("relocat", "empties", "emptying", "sink function",
              "sink_function", "the model needs")
    text = (tmp_path / "a.csv").read_text(encoding="utf-8").lower()
    for word in banned:
        assert word not in text, word


def test_the_new_i4_source_files_use_descriptive_wording_only():
    banned = ("relocat", "empties", "emptying", "sink function",
              "sink_function", "the model needs")
    for name in ("saga/frozen/masks.py", "analysis/frozen_I4_analysis.py",
                 "configs/frozen/I4_perturbation.yaml",
                 "scripts/jobs/frozen_I4.sbatch"):
        text = (REPO / name).read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{name} says {word!r}"


# ─────────────────────────────────────────────────────────────────────────────
# The job file
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not MANIFEST.exists(), reason="no manifest in this checkout")
def test_the_job_file_lists_the_cell_in_array_order():
    from saga.frozen.runner import eligible_cohort
    want = [r["run_id"] for r in eligible_cohort(str(MANIFEST))
            if (r["arch"], r["recipe_actual"]) == ("vit_small", "mixup")]
    text = JOB_FILE.read_text(encoding="utf-8")
    block = text.split("RUN_IDS=(", 1)[1].split("\n)", 1)[0]
    have = [line.split('"')[1] for line in block.strip().splitlines()
            if '"' in line]
    assert have == want, f"job file {have}\nmanifest {want}"
    assert len(have) == 10


def test_the_job_file_carries_the_embargo_and_the_fixed_split():
    text = JOB_FILE.read_text(encoding="utf-8")
    assert "EMBARGOED" in text
    assert "SPLIT=results/frozen/splits/evaluation.json" in text
    assert "--array=0-9" in text
    assert "configs/frozen/I4_perturbation.yaml" in text
    assert "assert_not_embargoed" in text
    assert '${1:?' not in text, "the split must not be a positional argument"
    # and the guard runs BEFORE staging
    assert text.index("assert_not_embargoed") < text.index(
        "stage_probe_imagenet_val")
