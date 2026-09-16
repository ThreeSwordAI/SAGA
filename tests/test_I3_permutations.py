"""
tests/test_I3_permutations.py
=============================
LOCKED_ANALYSIS D3 — the ten fixed position permutations and the ten fixed
within-ring permutations of the 14x14 patch grid, and the file that holds
them.

`saga/frozen/edits.gate_edit` REFUSES to draw a permutation: both `permute`
and `permute_within_ring` require an explicit `perm`, so no code path can
quietly sample a fresh one per checkpoint. That guarantee is only worth
something if the lists exist, are committed, and are shared across every
checkpoint on the same grid — which is what `configs/frozen/permutations_14x14.json`
is for, and what this file pins.

**This module is also the generator.** D3 was scoped to "a new config file
and its test only", so the recipe lives here rather than in a script that
would then be a second place to keep in sync:

    python -c "import sys; sys.path.insert(0,'.'); \
               from tests.test_I3_permutations import write_permutations; \
               write_permutations()"

`test_the_committed_file_is_byte_identical_on_regeneration` runs
`build_permutations()` and compares the serialization to what is committed,
so the file cannot drift from the recipe and the recipe cannot drift from
the file.

The within-ring rings come from `analysis.address_analysis.ring_indices` —
THE ring definition, the one TASK-07's sink-address answers and TASK-13's
ring ablation were both written against. It is imported here, never
re-derived.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.address_analysis import border_distance_map, ring_indices
from saga.frozen.edits import (EditError, build_edited_gate_map,
                               check_permutation)

REPO = Path(__file__).resolve().parents[1]
PERM_FILE = REPO / "configs" / "frozen" / "permutations_14x14.json"

SIDE = 14
N_POSITIONS = SIDE * SIDE                      # 196
N_DRAWS = 10

#: LOCKED_ANALYSIS §4: `RandomState(s).permutation(196)` for s = 0..9.
POSITION_SEEDS = tuple(range(N_DRAWS))
#: The same ten draws for the within-ring family, offset so that a position
#: list and a within-ring list can never come from the same stream.
RING_SEEDS = tuple(range(100, 100 + N_DRAWS))

DEFINITION = (
    "LOCKED_ANALYSIS D3, 14x14 patch grid. `position`: "
    "numpy.random.RandomState(s).permutation(196) for s in 0..9. "
    "`within_ring`: for s in 100..109, start from the identity and permute "
    "the flat indices INSIDE each Chebyshev border ring among themselves, "
    "rings taken in order k = 0..6 from "
    "analysis.address_analysis.ring_indices(14, k) — THE ring definition "
    "(TASK-07 Q1b, TASK-13). One list per (grid, kind), generated once, "
    "committed, and shared across every checkpoint on the same grid; "
    "saga/frozen/edits.gate_edit refuses to draw one, so no code path can "
    "sample a fresh permutation per checkpoint.")


def build_permutations() -> dict:
    """The committed document, as a pure function of the seeds and the rings.

    Deterministic and dependency-light: two `RandomState` streams and
    `ring_indices`. Nothing here reads a checkpoint, a split or a result.
    """
    position = [np.random.RandomState(s).permutation(N_POSITIONS).tolist()
                for s in POSITION_SEEDS]

    within_ring = []
    for s in RING_SEEDS:
        rng = np.random.RandomState(s)
        perm = np.arange(N_POSITIONS)
        for k in range(SIDE // 2):
            idx = ring_indices(SIDE, k)
            # position idx[i] receives the value at idx[shuffled[i]]: every
            # destination stays inside the ring it came from
            perm[idx] = idx[rng.permutation(idx.size)]
        within_ring.append(perm.tolist())

    ring_of = border_distance_map(SIDE).reshape(-1)
    return {
        "definition": DEFINITION,
        "grid": [SIDE, SIDE],
        "n_positions": N_POSITIONS,
        "generator": "numpy.random.RandomState(seed).permutation",
        "ring_source": "analysis.address_analysis.ring_indices",
        "ring_sizes": {str(k): int(ring_indices(SIDE, k).size)
                       for k in range(SIDE // 2)},
        "ring_of_position": ring_of.tolist(),
        "position_seeds": list(POSITION_SEEDS),
        "within_ring_seeds": list(RING_SEEDS),
        "position": position,
        "within_ring": within_ring,
    }


def serialize(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def write_permutations(path=PERM_FILE) -> Path:
    """Write the file. Idempotent: the content is a function of the seeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize(build_permutations()), encoding="utf-8",
                    newline="\n")
    return path


def _doc():
    if not PERM_FILE.exists():                           # pragma: no cover
        pytest.skip(f"{PERM_FILE} has not been generated")
    return json.loads(PERM_FILE.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# The file
# ─────────────────────────────────────────────────────────────────────────────

def test_the_file_exists_and_declares_its_own_recipe():
    doc = _doc()
    assert doc["grid"] == [SIDE, SIDE]
    assert doc["n_positions"] == N_POSITIONS == 196
    assert doc["position_seeds"] == list(range(10))
    assert doc["within_ring_seeds"] == list(range(100, 110))
    assert "RandomState" in doc["definition"]
    assert doc["ring_source"] == "analysis.address_analysis.ring_indices"
    assert len(doc["position"]) == len(doc["within_ring"]) == N_DRAWS


def test_the_committed_file_is_byte_identical_on_regeneration():
    """A generated file is REGENERATED, never hand-edited (I0 handoff §8.6).
    If someone tweaks one index by hand, this fails."""
    assert PERM_FILE.read_text(encoding="utf-8") == serialize(
        build_permutations())
    # ... and the recipe itself is deterministic
    assert serialize(build_permutations()) == serialize(build_permutations())


@pytest.mark.parametrize("kind", ["position", "within_ring"])
def test_every_list_is_a_bijection_over_the_196_patch_coordinates(kind):
    for i, perm in enumerate(_doc()[kind]):
        assert len(perm) == N_POSITIONS, (kind, i)
        arr = np.asarray(perm)
        assert np.array_equal(np.sort(arr), np.arange(N_POSITIONS)), (kind, i)
        # and the project's own validator accepts it
        assert check_permutation(arr, N_POSITIONS).shape == (N_POSITIONS,)


def test_the_ten_draws_are_distinct_within_and_across_families():
    doc = _doc()
    seen = {}
    for kind in ("position", "within_ring"):
        for i, perm in enumerate(doc[kind]):
            key = tuple(perm)
            assert key not in seen, \
                f"{kind}[{i}] repeats {seen[key]} — a repeated draw is not a " \
                f"second intervention"
            seen[key] = f"{kind}[{i}]"
    assert len(seen) == 2 * N_DRAWS


def test_within_ring_permutations_preserve_ring_membership_exactly():
    """The whole point of the family: it destroys arrangement INSIDE a ring
    while leaving the ring profile untouched, so a difference from the
    unrestricted family cannot be a difference in ring composition."""
    doc = _doc()
    rings = np.asarray(doc["ring_of_position"])
    assert np.array_equal(rings, border_distance_map(SIDE).reshape(-1))
    for i, perm in enumerate(doc["within_ring"]):
        p = np.asarray(perm)
        assert np.array_equal(rings[p], rings), \
            f"within_ring[{i}] moves a coordinate across a ring"
    # ring 1 of the 14x14 grid has 44 members — the ring TASK-07 found the
    # sinks and SAGA's gate suppression sitting on
    assert doc["ring_sizes"]["1"] == 44 == ring_indices(SIDE, 1).size
    assert sum(doc["ring_sizes"].values()) == N_POSITIONS


def test_the_unrestricted_family_actually_crosses_rings():
    """Otherwise the two families would be the same experiment twice."""
    doc = _doc()
    rings = np.asarray(doc["ring_of_position"])
    crossing = [int(np.count_nonzero(rings[np.asarray(p)] != rings))
                for p in doc["position"]]
    assert all(c > 0 for c in crossing), crossing
    assert min(crossing) > N_POSITIONS // 2, \
        f"the position permutations barely cross rings: {crossing}"


# ─────────────────────────────────────────────────────────────────────────────
# gate_edit accepts these lists, and still refuses to invent one
# ─────────────────────────────────────────────────────────────────────────────

def _gate_map(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.sigmoid(torch.randn(3, N_POSITIONS, generator=g) * 1.5)


@pytest.mark.parametrize("kind,mode", [("position", "permute"),
                                       ("within_ring", "permute_within_ring")])
def test_build_edited_gate_map_accepts_every_committed_list(kind, mode):
    g = _gate_map()
    for i, perm in enumerate(_doc()[kind]):
        edited = build_edited_gate_map(g, mode, perm=np.asarray(perm))
        assert edited.shape == g.shape
        # a permutation moves values without creating or destroying any:
        # the per-head MULTISET is invariant, which is the property TASK I0's
        # smoke check asserts with zero tolerance
        for h in range(g.shape[0]):
            assert torch.equal(torch.sort(edited[h]).values,
                               torch.sort(g[h]).values), (kind, i, h)
        assert not torch.equal(edited, g), (kind, i)


def test_a_position_permutation_is_refused_by_the_within_ring_mode():
    """The ring guard is real: an unrestricted list passed to the
    ring-preserving mode must be rejected, not silently applied."""
    g = _gate_map()
    doc = _doc()
    with pytest.raises(EditError, match="across rings"):
        build_edited_gate_map(g, "permute_within_ring",
                              perm=np.asarray(doc["position"][0]))


def test_gate_edit_still_refuses_to_draw_its_own_permutation():
    """D3 exists because `gate_edit` will not sample one. If that ever
    changed, committing the lists would stop meaning anything."""
    g = _gate_map()
    for mode in ("permute", "permute_within_ring"):
        with pytest.raises(EditError, match="requires an explicit"):
            build_edited_gate_map(g, mode, perm=None)


def test_the_files_sha256_is_recorded_in_LOCKED_ANALYSIS():
    """D3 is closed in LOCKED_ANALYSIS by naming this file AND its digest;
    a file that changed would leave that record false."""
    locked = (REPO / "docs" / "LOCKED_ANALYSIS.md").read_text(encoding="utf-8")
    # `~~D3~~` is this document's own convention for a CLOSED decision (D8
    # was struck through the same way). The path alone is not the guard: §4
    # has named the file since I0, back when it did not exist.
    if "~~D3~~" not in locked:                           # pragma: no cover
        pytest.skip("D3 has not been closed in LOCKED_ANALYSIS.md yet")
    digest = hashlib.sha256(
        PERM_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest in locked, (
        f"docs/LOCKED_ANALYSIS.md does not record the committed digest "
        f"{digest} for configs/frozen/permutations_14x14.json")
