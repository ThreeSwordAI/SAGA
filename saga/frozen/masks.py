"""
saga/frozen/masks.py
====================
TASK B / I4 — the D5 mask contract, and the freeze it is gated on.

I4 perturbs the attention output at 16 patch coordinates and compares the
effect against 10 ring-matched control masks. WHICH 16 coordinates is the
whole experiment, so this module exists to make "the masks were fixed before
the evaluation outcomes were inspected" a property of the repository:

    configs/frozen/I4_masks.json   written by TASK A / I1 Phase C from the
                                   DISCOVERY split (D5), never by I4
    docs/LOCKED_ANALYSIS.md        must carry `STATUS: FROZEN` with a date
                                   and a git sha, and must close D5 with a
                                   sha256 that matches that file

Neither exists yet. D5 is still OPEN and the LOCKED document is still a
DRAFT, so `load_masks` refuses today and will keep refusing until the human
closes both. That refusal IS the deliverable: the loader, the guard and
their tests are written now, against a schema TASK A §8 fixed, so that
closing D5 is the only thing left to do.

THE TWO FILES ARE CHECKED AGAINST EACH OTHER, NOT JUST READ
------------------------------------------------------------
A masks file with no LOCKED entry is an unfixed mask. A LOCKED entry whose
sha does not match the file is a mask that changed after it was fixed. Both
are refused by name, and the refusal says which of the two it is — the
failure mode this guards against is not a missing file, it is a file that
was quietly regenerated between the freeze and the run.

Mask ids, from TASK B §5:

    P_L7,  P_L8            the primary masks, 16 coordinates each
    K0_L7 .. K9_L7         the 10 ring-matched controls at layer 7
    K0_L8 .. K9_L8         the 10 ring-matched controls at layer 8

`P_L7` is selected on the prevalence map at `in_b07` — the residual stream
ENTERING blocks[7], which is the block I4 perturbs — and `P_L8` at
`in_b08`. The layer suffix is the 0-BASED block index, matching
`saga/frozen/stages.BLOCK_INPUT_STAGES` and LOCKED §2 ("paper blocks 8 and
9 = 0-based block indices 7 and 8").

No training, no optimizer, no sampling: this module reads a committed file
and verifies digests. The masks themselves are DRAWN in
`saga/frozen/prevalence.ring_matched_controls`, on the discovery split,
which refuses any other split by sha.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from saga.frozen.prevalence import (N_CONTROL_MASKS, PRIMARY_MASK_K,
                                    ring_composition)
from saga.frozen.stages import BLOCK_INPUT_STAGES

MISSING = "MISSING"

#: Where D5's masks live. Written by TASK A / I1 Phase C, read here.
MASKS_FILE = "configs/frozen/I4_masks.json"

#: The document that has to be frozen before any I4 Phase-B job may run.
LOCKED_FILE = "docs/LOCKED_ANALYSIS.md"

#: The layers I4 edits, and the block-INPUT stage each mask was selected on.
#: Imported from `stages` rather than restated: `in_b07` means "the input to
#: blocks[7]" in exactly one place in this project.
MASK_LAYERS = tuple(sorted(BLOCK_INPUT_STAGES.values()))
STAGE_FOR_LAYER = {v: k for k, v in BLOCK_INPUT_STAGES.items()}

#: The header line the freeze writes. Matched case-sensitively: `FROZEN` is
#: a state, and a file that says `frozen` in prose has not been signed.
FROZEN_MARKER = "STATUS: FROZEN"

#: The document's own convention for a CLOSED decision — a struck-through id,
#: as D1/D2/D3/D4/D6/D7/D8 already use it. `tests/test_I3_permutations.py`
#: relies on the same convention for D3.
CLOSED_RE = re.compile(r"~~\s*D5\s*~~")

#: A 64-hex digest anywhere in the D5 material.
SHA_RE = re.compile(r"\b([0-9a-f]{64})\b")


class MaskError(ValueError):
    """A mask file, or a freeze, this module refuses to accept."""


# ─────────────────────────────────────────────────────────────────────────────
# Digests
# ─────────────────────────────────────────────────────────────────────────────

def file_sha256(path) -> str:
    """sha256 with LINE ENDINGS NORMALISED to LF.

    The same rule `saga/frozen/diag.canon_sha256` and the D3 permutation test
    use, and for the same reason: this repository is checked out with CRLF on
    Windows and LF on the cluster, so a raw-byte digest would disagree with
    itself across the two machines that have to agree.
    """
    return hashlib.sha256(
        Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# The freeze
# ─────────────────────────────────────────────────────────────────────────────

def locked_state(locked_path=LOCKED_FILE, *, masks_path=MASKS_FILE) -> dict:
    """What the LOCKED document says, as data. Never raises on content.

    Returns `{frozen, date, git_sha, d5_closed, d5_sha, masks_sha, missing}`
    where `missing` is the list of human-readable reasons the freeze is not
    in force. An empty `missing` means an I4 job may run.

    Separated from the refusal so that the guard can NAME every problem at
    once rather than failing on the first one and hiding the second.
    """
    p = Path(locked_path)
    state = {"locked_file": str(p), "frozen": False, "date": MISSING,
             "git_sha": MISSING, "d5_closed": False, "d5_sha": MISSING,
             "masks_file": str(masks_path), "masks_sha": MISSING,
             "missing": []}

    if not p.exists():
        state["missing"].append(f"{p} does not exist")
        return state
    text = p.read_text(encoding="utf-8")

    state["frozen"] = FROZEN_MARKER in text
    if not state["frozen"]:
        state["missing"].append(
            f"{p} does not carry the header line '{FROZEN_MARKER}' — it is "
            f"still a DRAFT and nothing in it is locked")

    date = re.search(r"Date frozen:\s*`?([^`\n]+)`?", text)
    date = (date.group(1).strip() if date else MISSING)
    state["date"] = date
    if date in (MISSING, "PENDING", ""):
        state["missing"].append(
            f"{p} has no freeze DATE (the 'Date frozen:' line is "
            f"{date!r})")

    sha = re.search(r"Git sha at freeze:\s*`?([^`\n]+)`?", text)
    sha = (sha.group(1).strip() if sha else MISSING)
    state["git_sha"] = sha
    if sha in (MISSING, "PENDING", ""):
        state["missing"].append(
            f"{p} has no git sha at freeze (the 'Git sha at freeze:' line is "
            f"{sha!r})")

    state["d5_closed"] = bool(CLOSED_RE.search(text))
    if not state["d5_closed"]:
        state["missing"].append(
            f"D5 is not closed in {p} (no struck-through '~~D5~~', which is "
            f"that document's own convention for a closed decision). D5 is "
            f"the prevalence map at the input to blocks 7/8, and TASK A / I1 "
            f"owns it")

    m = Path(masks_path)
    if not m.exists():
        state["missing"].append(
            f"{m} does not exist — TASK A / I1 Phase C writes it from the "
            f"DISCOVERY split")
        return state
    state["masks_sha"] = file_sha256(m)

    digests = set(SHA_RE.findall(text))
    if state["masks_sha"] in digests:
        state["d5_sha"] = state["masks_sha"]
    else:
        state["missing"].append(
            f"{p} does not record the digest {state['masks_sha']} for {m}. "
            f"Either D5 was closed against a different file, or the file has "
            f"been regenerated since it was frozen — which is the failure "
            f"this check exists for, not a missing file")
    return state


def assert_frozen(locked_path=LOCKED_FILE, *, masks_path=MASKS_FILE) -> dict:
    """Raise unless the freeze is in force. Returns the state on success.

    The message names EVERY missing piece, so one run tells the human the
    whole list instead of one item per attempt.
    """
    state = locked_state(locked_path, masks_path=masks_path)
    if state["missing"]:
        bullets = "\n  - ".join(state["missing"])
        raise MaskError(
            f"I4 is EMBARGOED: the analysis parameters are not frozen.\n"
            f"  - {bullets}\n"
            f"TASK B §7: no I4 Phase-B job runs until "
            f"`docs/LOCKED_ANALYSIS.md` carries '{FROZEN_MARKER}' with a "
            f"date and a git sha, and D5 is closed with a sha256 matching "
            f"{masks_path}.")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# The masks
# ─────────────────────────────────────────────────────────────────────────────

def primary_id(layer: int) -> str:
    return f"P_L{int(layer)}"


def control_id(j: int, layer: int) -> str:
    return f"K{int(j)}_L{int(layer)}"


def expected_mask_ids(layers=MASK_LAYERS, n_controls=N_CONTROL_MASKS) -> tuple:
    """Every mask id I4 declares, in a fixed order."""
    out = []
    for layer in layers:
        out.append(primary_id(layer))
        out += [control_id(j, layer) for j in range(int(n_controls))]
    return tuple(out)


def load_masks(masks_path=MASKS_FILE, *, locked_path=LOCKED_FILE,
               require_freeze: bool = True, expect_sha=None) -> dict:
    """The D5 masks, verified, as `{mask_id: int64 index array}` plus meta.

    Returns `{"masks": {...}, "grid_side": int, "sha256": str,
              "provenance": {...}, "ring_composition": {...}}`.

    `require_freeze` is True on every real path. It is False ONLY for the
    tests that exercise the schema against a fake mask file, and for
    `tools/frozen_eval.py`'s own error reporting — never for a job.

    `expect_sha` is an independent digest the caller already knows (the
    runner passes the one it read out of LOCKED), checked in addition to the
    freeze so that the file cannot be swapped between the two reads.
    """
    m = Path(masks_path)
    if require_freeze:
        assert_frozen(locked_path, masks_path=masks_path)
    if not m.exists():
        raise MaskError(
            f"{m} does not exist. TASK A / I1 Phase C writes it from the "
            f"DISCOVERY split ({', '.join(STAGE_FOR_LAYER[l] for l in MASK_LAYERS)}), "
            f"and I4 never builds a mask of its own — a mask chosen by I4 "
            f"would be a mask chosen after the outcome was visible.")
    sha = file_sha256(m)
    if expect_sha not in (None, MISSING) and sha != expect_sha:
        raise MaskError(
            f"{m} hashes to {sha} but the caller expected {expect_sha} — "
            f"refusing to perturb coordinates that changed since they were "
            f"fixed")

    doc = json.loads(m.read_text(encoding="utf-8"))
    side = int(doc.get("grid_side", 0))
    if side <= 0:
        raise MaskError(f"{m} declares grid_side={doc.get('grid_side')!r}")
    n_positions = side * side

    raw = doc.get("masks")
    if not isinstance(raw, dict):
        raise MaskError(
            f"{m} has no `masks` object; expected "
            f"{{mask_id: [coordinate, ...]}} with ids "
            f"{list(expected_mask_ids())}")

    want = set(expected_mask_ids())
    got = set(raw)
    if got != want:
        raise MaskError(
            f"{m} declares masks {sorted(got)};\nI4 needs exactly "
            f"{sorted(want)}.\n  missing: {sorted(want - got)}\n"
            f"  unexpected: {sorted(got - want)}")

    masks, comps = {}, {}
    for mask_id in expected_mask_ids():
        idx = np.asarray(raw[mask_id], dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise MaskError(f"{m}: mask {mask_id!r} is empty")
        if len(set(idx.tolist())) != idx.size:
            raise MaskError(f"{m}: mask {mask_id!r} repeats a coordinate")
        if idx.min() < 0 or idx.max() >= n_positions:
            raise MaskError(
                f"{m}: mask {mask_id!r} has a coordinate outside "
                f"0..{n_positions - 1} for a {side}x{side} grid. These are "
                f"PATCH coordinates, never token indices — the prefix rows "
                f"are not maskable.")
        masks[mask_id] = np.sort(idx)
        comps[mask_id] = ring_composition(masks[mask_id].tolist(), side)

    _check_sizes_and_rings(m, masks, comps)

    return {
        "masks": masks, "grid_side": side, "sha256": sha,
        "ring_composition": comps,
        "masks_file": str(m),
        "provenance": {k: doc.get(k, MISSING) for k in (
            "split_sha256", "split_name", "stage_by_layer", "basis",
            "source_run_ids", "thresholds", "git_sha", "generated_by",
            "generated_at", "control_seeds", "n_controls", "k")},
    }


def _check_sizes_and_rings(path, masks: dict, comps: dict):
    """The two invariants that make a control a control.

    A control with a different NUMBER of coordinates injects a different
    amount of energy before any matching is applied; a control with a
    different RING COMPOSITION differs from the primary in two ways at once
    — where the mass is and how far from the border it is — and I4 could not
    say which one moved the loss (LOCKED §5).
    """
    for layer in MASK_LAYERS:
        pid = primary_id(layer)
        if masks[pid].size != PRIMARY_MASK_K:
            raise MaskError(
                f"{path}: primary mask {pid!r} has {masks[pid].size} "
                f"coordinates, expected {PRIMARY_MASK_K} (LOCKED §5)")
        for j in range(N_CONTROL_MASKS):
            cid = control_id(j, layer)
            if masks[cid].size != masks[pid].size:
                raise MaskError(
                    f"{path}: control {cid!r} has {masks[cid].size} "
                    f"coordinates but {pid!r} has {masks[pid].size} — a "
                    f"control must inject over the same number of positions")
            if comps[cid] != comps[pid]:
                raise MaskError(
                    f"{path}: control {cid!r} is not ring-matched to {pid!r}\n"
                    f"  {pid}: {comps[pid]}\n  {cid}: {comps[cid]}\n"
                    f"A control that ignored the rings would differ from the "
                    f"primary in two ways at once (LOCKED §5).")


def mask_overlap(masks: dict, layer: int) -> dict:
    """`{control_id: how many coordinates it shares with the primary}`.

    LOCKED §5: "random masks MAY overlap the high-prevalence mask. The
    overlap is REPORTED; draws are never rejected to amplify contrast." This
    is the reporting half, and `analysis/frozen_I4_analysis.py` puts it in
    T_I4b beside every control.
    """
    pid = primary_id(layer)
    primary = set(masks[pid].tolist())
    return {control_id(j, layer): int(len(primary
                                         & set(masks[control_id(j, layer)]
                                               .tolist())))
            for j in range(N_CONTROL_MASKS)}
