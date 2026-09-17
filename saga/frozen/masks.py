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

Both are now in place. D5 was closed and the document signed on 2026-09-17
at git `5c1b737`, and `load_masks` opens. Until then it refused, and that
refusal was the deliverable — the loader, the guard and their tests were
written against a schema TASK A §8 had fixed, before the file existed, so
that closing D5 was the only thing left to do. The refusal path is still
live and still tested: it is what a later rebuild of the masks, or an edit
to the signed document, would run into.

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

#: ...but the document is MARKDOWN, and `**` is presentation. The draft head
#: was `STATUS: **DRAFT — NOT YET FROZEN**`, so a freeze written in the same
#: style would say `STATUS: **FROZEN**` — which does NOT contain the literal
#: above, because the asterisks sit between the colon and the word.
#:
#: That bit on 2026-09-17: this module accepted `STATUS: FROZEN` while
#: `analysis/build_C_handoff.py` tested for `STATUS: **FROZEN**`, so one of
#: the two had to be wrong about a frozen document no matter which style the
#: human used. Detection is now emphasis-insensitive in both places; the
#: STATE is what matters and the asterisks are not part of it.
FROZEN_RE = re.compile(r"STATUS:\s*(?:\*+\s*)?FROZEN")


def says_frozen(text: str) -> bool:
    """True when the document's header declares the freeze, in either style."""
    return bool(FROZEN_RE.search(text))

#: The document's own convention for a CLOSED decision — a struck-through id,
#: as D1/D2/D3/D4/D6/D7/D8 already use it. `tests/test_I3_permutations.py`
#: relies on the same convention for D3.
CLOSED_RE = re.compile(r"~~\s*D5\s*~~")

#: A 64-hex digest anywhere in the D5 material.
SHA_RE = re.compile(r"\b([0-9a-f]{64})\b")

#: The schema string `analysis/build_D5_masks.py` (TASK A / I1 Phase C)
#: stamps into the file. Checked, not assumed: a future D5 rebuild under a
#: different shape must announce itself rather than be silently misread.
SCHEMA = "i4_masks_v1"


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

#: Values that are a PROMISE of a signature rather than one. `PENDING` is the
#: draft's own word; the angle-bracket forms are what a prepared-but-unsigned
#: edit leaves behind, and `docs/LOCKED_FREEZE_DRAFT.patch` uses exactly those.
#:
#: This matters more than it looks. A staged freeze with `<DATE>` still in it
#: would otherwise satisfy the header check, and the repository would read as
#: FROZEN while nobody had signed anything — the one failure mode a freeze
#: guard must not have.
UNSIGNED = ("PENDING", "TBD", "TODO", "")


def _unsigned(value: str) -> bool:
    """True when a signature line is empty, pending, or a placeholder."""
    v = str(value or "").strip()
    if v in UNSIGNED or v == MISSING:
        return True
    if v.startswith("<") and v.endswith(">"):
        return True
    return "PLACEHOLDER" in v.upper()


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

    state["frozen"] = says_frozen(text)
    if not state["frozen"]:
        state["missing"].append(
            f"{p} does not carry the header line '{FROZEN_MARKER}' — it is "
            f"still a DRAFT and nothing in it is locked")

    date = re.search(r"Date frozen:\s*`?([^`\n]+)`?", text)
    date = (date.group(1).strip() if date else MISSING)
    state["date"] = date
    if _unsigned(date):
        state["missing"].append(
            f"{p} has no freeze DATE (the 'Date frozen:' line is "
            f"{date!r})")

    sha = re.search(r"Git sha at freeze:\s*`?([^`\n]+)`?", text)
    sha = (sha.group(1).strip() if sha else MISSING)
    state["git_sha"] = sha
    if _unsigned(sha):
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
    raw, side, provenance = _flatten(m, doc)
    n_positions = side * side

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
        # TASK A / I1 Phase C found that BOTH primary masks select the same
        # 16 coordinates (the two cell mean maps correlate at rho = 0.9985).
        # That is not a bug and it is not this module's to fix, but it
        # changes what I4's two sites mean — with the address held fixed,
        # comparing layer 7 against layer 8 is a comparison of DEPTH alone.
        # Surfaced here so a table can say so without re-reading the file.
        "primaries_identical": bool(np.array_equal(
            masks[primary_id(MASK_LAYERS[0])],
            masks[primary_id(MASK_LAYERS[1])])),
        "provenance": provenance,
    }


def _flatten(path, doc: dict):
    """(`{mask_id: [index]}`, grid_side, provenance) from an `i4_masks_v1` doc.

    THE FILE IS KEYED BY STAGE, NOT BY MASK ID. TASK A / I1 Phase C writes

        masks:
          in_b07:
            grid_side, block_entered_0based, discovery_split_sha256, ...
            primary_mask: {index: [16], coordinates: [...], ...}
            controls: [{index: [16], seed, overlap_with_primary, ...} x10]
          in_b08: the same

    because a stage is what a mask was SELECTED on. I4 names its masks by
    layer (`P_L7`, `K3_L8`), because a layer is what a condition EDITS.
    Those are two names for one thing — `in_b07` is the residual stream
    entering `blocks[7]` — and this function is the one place the two
    vocabularies meet. `block_entered_0based` is cross-checked against
    `saga/frozen/stages.BLOCK_INPUT_STAGES` so the mapping cannot drift.
    """
    schema = doc.get("schema")
    if schema != SCHEMA:
        raise MaskError(
            f"{path} declares schema {schema!r}; this loader reads "
            f"{SCHEMA!r} (TASK A / I1 Phase C, analysis/build_D5_masks.py)")
    by_stage = doc.get("masks")
    if not isinstance(by_stage, dict):
        raise MaskError(f"{path} has no `masks` object")

    want_stages = {STAGE_FOR_LAYER[l] for l in MASK_LAYERS}
    if set(by_stage) != want_stages:
        raise MaskError(
            f"{path} declares masks at stage(s) {sorted(by_stage)}; I4 needs "
            f"exactly {sorted(want_stages)} — the block-INPUT stages the two "
            f"edited layers are entered at")

    flat, sides = {}, set()
    for stage, block in sorted(by_stage.items()):
        layer = BLOCK_INPUT_STAGES[stage]
        declared = block.get("block_entered_0based")
        if declared is not None and int(declared) != layer:
            raise MaskError(
                f"{path}: stage {stage!r} says it enters block "
                f"{declared}, but saga/frozen/stages.py defines {stage!r} as "
                f"the input to blocks[{layer}]. One of the two is wrong and "
                f"an off-by-one here would perturb the wrong block.")
        side = block.get("grid_side")
        if not side:
            raise MaskError(f"{path}: stage {stage!r} declares no grid_side")
        sides.add(int(side))

        primary = (block.get("primary_mask") or {}).get("index")
        if primary is None:
            raise MaskError(
                f"{path}: stage {stage!r} has no `primary_mask.index`")
        flat[primary_id(layer)] = primary

        controls = block.get("controls") or []
        if len(controls) != N_CONTROL_MASKS:
            raise MaskError(
                f"{path}: stage {stage!r} has {len(controls)} controls, "
                f"expected {N_CONTROL_MASKS} (LOCKED §5)")
        for j, ctrl in enumerate(controls):
            if "index" not in ctrl:
                raise MaskError(
                    f"{path}: stage {stage!r} control {j} has no `index`")
            flat[control_id(j, layer)] = ctrl["index"]

    if len(sides) != 1:
        raise MaskError(
            f"{path}: the two stages declare different grids {sorted(sides)}")

    missing = sorted(set(expected_mask_ids()) - set(flat))
    if missing:                                          # pragma: no cover
        raise MaskError(f"{path}: could not resolve {missing}")

    # Masks are SELECTED on discovery and reported on evaluation. The split
    # the file names is checked through TASK A's own allow-list, so a mask
    # file built from reporting data is refused here as firmly as
    # `prevalence.py` would have refused to build it.
    shas = {b.get("discovery_split_sha256") for b in by_stage.values()}
    for sha in sorted(s for s in shas if s):
        try:
            from saga.frozen.prevalence import assert_selection_split
            assert_selection_split(sha, what="mask file")
        except ImportError:                              # pragma: no cover
            break
        except Exception as exc:
            raise MaskError(f"{path}: {exc}") from exc

    prov = {k: doc.get(k, MISSING) for k in (
        "schema", "work_package", "cell", "basis", "k", "n_controls",
        "control_seeds", "control_overlap_policy", "discovery_split_name",
        "git_sha", "generated_by")}
    prov["by_stage"] = {
        s: {k: b.get(k, MISSING) for k in (
            "stage", "block_entered_0based", "paper_block_1based", "tau_cal",
            "discovery_split_sha256", "source_run_ids", "n_images",
            "control_overlap_observed", "control_overlap_expected")}
        for s, b in sorted(by_stage.items())}
    return flat, sides.pop(), prov


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
