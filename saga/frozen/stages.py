"""
saga/frozen/stages.py
=====================
Named feature stages, captured by forward hooks, with the prefix rows
already removed using the MODEL'S OWN prefix count.

Four names from TASK I0, and no fifth for the terminal region anywhere in
the project:

    s11_out        output of the second-to-last block
    s12_pre_norm   output of the LAST block, BEFORE the final LayerNorm
    s12_post_norm  after the final LayerNorm (model.norm)
    hist           alias for whichever of the above the HISTORICAL
                   diagnostics were computed at -> see HIST_STAGE

On the 12-block production models (ViT-S/16 and ViT-B/16 both have 12
blocks) `s11_out` is the output of block INDEX 10 and `s12_pre_norm` the
output of block index 11, which is how TASK I0 §6 names them. The indices
are derived from the depth (L-2, L-1) rather than hard-coded, so a tiny
fake model in the tests exercises this exact code path; a test pins the
equality at L = 12.

THE BLOCK-INPUT STAGES (TASK A / I1 §3 — added for D5)
------------------------------------------------------
Two more names, ADDITIVE, for the residual stream ENTERING a block:

    in_b07         the residual stream entering blocks[7]
    in_b08         the residual stream entering blocks[8]

`in_bNN` is read "the input to blocks[NN]", and NN is a 0-BASED absolute
block index — unlike `s11_out`/`s12_pre_norm`, which are relative to the
depth. Because a pre-norm ViT block is a residual update, the stream
entering blocks[NN] IS the output of blocks[NN-1]:

    in_b07 == output of blocks[6]
    in_b08 == output of blocks[7]

and that is how they are captured (a forward hook on the PRECEDING block),
which is exactly equivalent and needs no pre-hook.

BOTH NUMBERING CONVENTIONS, because the off-by-one here is the one that
would silently invalidate I4:

    code, 0-based   | in_b07 enters blocks[7]  | in_b08 enters blocks[8]
    paper, 1-based  | input to block 8         | input to block 9

`docs/LOCKED_ANALYSIS.md` §2 fixes the pair as "paper blocks 8 and 9 =
0-based block indices 7 and 8", and §5 asks for the prevalence map "at the
input to the edited block (or the preceding block's output)". These two
stages are that input, for those two blocks.
`tests/test_I1_spatial.py` pins `in_b07 == blocks[6](x)` and
`in_b08 == blocks[7](x)` on a fake model by comparing TENSORS, so the
convention cannot drift from this docstring.

A THIRD ONE (TASK C / I7 §5 — added for the attention alignment)
----------------------------------------------------------------
    in_b10         the residual stream entering blocks[10]
                   == output of blocks[9]

Same convention, same capture path, added for one reason: the attention
INSIDE `blocks[10]` is computed from that block's INPUT tokens, so the
exceedance it must be compared against is the exceedance of those tokens —
not of the block's output. I7 pairs `blocks[10]` with `in_b10` and
`blocks[11]` with `s11_out` (which IS the output of `blocks[10]` on a
12-block model, i.e. the input to `blocks[11]`). Comparing a block's
incoming attention with the norms of its own OUTPUT would put the effect one
block after its cause. `tests/test_I7_attention.py` pins
`in_b10 == blocks[9](x)` by comparing TENSORS.

THE HISTORICAL STAGE (TASK I0 §2.5 — identified, not assumed)
-------------------------------------------------------------
Every historical patch diagnostic in this project was computed on the
output of the last transformer block, BEFORE the final LayerNorm, i.e.
`s12_pre_norm`. Read in this order:

  saga/metrics.py:8-9    module docstring: "token tensors captured by
                         forward hooks on EVERY ``model.blocks[k]``
                         (pre-final-norm block outputs)"
  saga/metrics.py:287-291  compute_diagnostics registers a forward hook on
                         each `model.blocks[i]`, storing `out` — a block's
                         own output, so the hook fires BEFORE `model.norm`
  saga/metrics.py:324    `x_last = _fp32(feats[L - 1])` — the LAST block
  saga/metrics.py:325    `patch = x_last[:, P:, :]` — prefix rows dropped
                         with P = infer_num_prefix_tokens(model)
  saga/vit.py:238-241    SAGAViT.forward runs the block loop first and only
                         then `x = self.norm(x)`, so nothing a block hook
                         sees has passed through the final norm
  tools/diagnose.py:89-95  the historical driver calls compute_diagnostics
                         and writes its output verbatim

`tests/test_I0_frozen.py::test_hist_stage_matches_the_historical_hook`
re-derives this by running the real `compute_diagnostics` hook shape
against `capture_stages`, so the alias cannot drift from the code above.
"""

from contextlib import contextmanager

import torch

from saga.metrics import infer_num_prefix_tokens

# The four TASK I0 stage names for the TERMINAL region. `hist` is an alias,
# resolved by resolve_stage(). This tuple is UNCHANGED by TASK A: it is the
# I0 contract, several tests enumerate it and capture every member on a
# 4-block fake model, and a block-input stage does not exist on a model that
# shallow. The new names live in BLOCK_INPUT_STAGES and in ALL_STAGES.
STAGES = ("s11_out", "s12_pre_norm", "s12_post_norm", "hist")

#: The block-input stages of TASK A / I1, as {name: the 0-based index of the
#: block whose INPUT this is}. `in_b07` enters blocks[7]; see the module
#: docstring for both numbering conventions.
#:
#: THIS TUPLE IS I4'S CONTRACT, NOT A LIST OF EVERY BLOCK-INPUT STAGE.
#: `saga/frozen/masks.py` derives the layers I4 EDITS from it
#: (`MASK_LAYERS = tuple(sorted(BLOCK_INPUT_STAGES.values()))`), so adding a
#: name here silently requires a new mask family in
#: `configs/frozen/I4_masks.json` and changes another work package's declared
#: experiment. A new block-input stage that I4 does not edit goes in
#: `EXTRA_BLOCK_INPUT_STAGES` below. (Measured: putting `in_b10` here made 15
#: of Track B's I4 tests fail by demanding `*_L10` masks that were never
#: drawn.)
BLOCK_INPUT_STAGES = {"in_b07": 7, "in_b08": 8}

#: Block-input stages that exist for CAPTURE ONLY — no work package edits
#: them, so they are not part of any mask or layer contract.
#:
#: `in_b10` (TASK C / I7 §5): the attention inside `blocks[10]` is computed
#: from that block's INPUT tokens, so I7 compares it against the exceedance
#: of those tokens. I7 edits nothing.
EXTRA_BLOCK_INPUT_STAGES = {"in_b10": 10}

#: Every block-input stage this module can capture, whatever declared it.
#: `resolve_stage` and the capture machinery use THIS; a contract that is
#: about which blocks a work package EDITS uses its own tuple.
ALL_BLOCK_INPUT_STAGES = {**BLOCK_INPUT_STAGES, **EXTRA_BLOCK_INPUT_STAGES}

#: Which real stage the historical diagnostics used. See the module docstring
#: for the code citation; a test pins it.
HIST_STAGE = "s12_pre_norm"

#: External-model ANALOG stages (TASK A / I6), as {name: the stage it
#: aliases}. They are the same two block outputs `s11_out` and
#: `s12_pre_norm` name, captured by the same code, under DIFFERENT NAMES.
#:
#: The separate names are the point. A public checkpoint differs from this
#: project's cohort in training data, augmentation recipe, input resolution
#: and patch size; DINOv2 is natively a 518-pixel model on a 14-pixel patch
#: and is run here at 224 with its position embedding interpolated. Two
#: numbers that both said `s11_out` would invite a reader to compare them as
#: if they were the same measurement. `ext_s11_out` says "the second-to-last
#: block of some other model", which is all I6 claims.
EXTERNAL_STAGES = {"ext_s11_out": "s11_out", "ext_hist": HIST_STAGE}

#: Every stage name this module accepts. `resolve_stage` validates against
#: THIS, so a conditions YAML may spell any of the eight; `STAGES` keeps its
#: I0 meaning for everything that enumerates the terminal four.
ALL_STAGES = (STAGES + tuple(ALL_BLOCK_INPUT_STAGES)
              + tuple(EXTERNAL_STAGES))

HIST_STAGE_CITATION = (
    "saga/metrics.py:287-291 (forward hooks on model.blocks[i], storing the "
    "block's own output), saga/metrics.py:324-325 (feats[L-1][:, P:, :]), "
    "saga/vit.py:238-241 (SAGAViT.forward applies self.norm AFTER the block "
    "loop), tools/diagnose.py:89-95 (the historical driver)"
)

#: Stages that are a block output (value = block index counted from the end).
_BLOCK_STAGES = {"s11_out": -2, "s12_pre_norm": -1}

#: Block-input stages, as {name: the 0-based index of the block whose OUTPUT
#: carries them}. `in_b07` enters blocks[7], so it is captured on blocks[6].
#: ABSOLUTE indices — a block-input stage names one specific block, and
#: rebasing it on the depth would move it on a model of another depth.
_BLOCK_INPUT_SOURCE = {name: idx - 1
                       for name, idx in ALL_BLOCK_INPUT_STAGES.items()}


class StageError(ValueError):
    """An unknown stage, or a model this module cannot hook."""


def resolve_stage(stage: str) -> str:
    """Map an alias onto the real stage it names; validate every other name.

    `hist` -> HIST_STAGE, and the two external analogs onto the two block
    outputs they are (see EXTERNAL_STAGES for why they have their own names
    at all).
    """
    if stage == "hist":
        return HIST_STAGE
    if stage in EXTERNAL_STAGES:
        return EXTERNAL_STAGES[stage]
    if stage not in ALL_STAGES:
        raise StageError(
            f"unknown stage {stage!r}; expected one of {ALL_STAGES}")
    return stage


def model_blocks(model):
    """The transformer block list of a SAGAViT, a plain timm ViT, or a timm
    register model. Raises rather than guessing."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise StageError(
            f"{type(model).__name__} exposes no non-empty `.blocks`; this "
            f"module hooks transformer blocks and refuses to guess at an "
            f"unknown architecture")
    return blocks


def final_norm(model):
    """The final LayerNorm. timm and SAGAViT both call it `norm`."""
    norm = getattr(model, "norm", None)
    if norm is None:
        raise StageError(
            f"{type(model).__name__} exposes no `.norm`; stage "
            f"'s12_post_norm' cannot be captured")
    return norm


def stage_block_index(model, stage: str) -> int:
    """0-based index of the block whose OUTPUT carries `stage`.

    Raises for `s12_post_norm` (not a block output). On the 12-block
    production models this returns 10 for s11_out and 11 for s12_pre_norm;
    6 for in_b07 and 7 for in_b08 on ANY model deep enough, since a
    block-input stage names one specific block rather than a position
    relative to the end.
    """
    real = resolve_stage(stage)
    depth = len(model_blocks(model))
    if real in _BLOCK_INPUT_SOURCE:
        idx = _BLOCK_INPUT_SOURCE[real]
        if idx >= depth:
            raise StageError(
                f"stage {real!r} is the input to "
                f"blocks[{ALL_BLOCK_INPUT_STAGES[real]}], "
                f"i.e. the output of blocks[{idx}], but the model has only "
                f"{depth} block(s) — this stage does not exist on it")
        return idx
    if real not in _BLOCK_STAGES:
        raise StageError(f"stage {stage!r} ({real}) is not a block output")
    idx = depth + _BLOCK_STAGES[real]
    if idx < 0:
        raise StageError(
            f"stage {real!r} needs at least {-_BLOCK_STAGES[real]} blocks, "
            f"model has {depth}")
    return idx


def num_prefix_tokens(model) -> int:
    """The model's own prefix-token count — 1 for baseline/SAGA, 5 for the
    timm 4-register models. NEVER a hard-coded 1 (audit bug B2)."""
    p = infer_num_prefix_tokens(model)
    if not isinstance(p, int) or p < 1:
        raise StageError(
            f"prefix count for {type(model).__name__} is {p!r}, which is not "
            f"a positive integer — refusing to slice patch rows")
    return p


def _patch_rows(x: torch.Tensor, n_prefix: int) -> torch.Tensor:
    """[B, N, D] -> [B, N - n_prefix, D], fp32, detached."""
    if x.ndim != 3:
        raise StageError(
            f"expected a [B, N, D] token tensor, got shape {tuple(x.shape)}")
    if x.shape[1] <= n_prefix:
        raise StageError(
            f"token sequence has {x.shape[1]} rows but the model reports "
            f"{n_prefix} prefix tokens — no patch rows would remain")
    return x[:, n_prefix:, :].detach().float()


@contextmanager
def capture_stages(model, stages=("s12_pre_norm",)):
    """Capture patch tokens at `stages` for every forward pass in the block.

    Yields a dict that fills in on each forward: ``store[stage]`` is a
    ``[B, N_patches, D]`` fp32 tensor with the prefix rows ALREADY removed
    using ``infer_num_prefix_tokens(model)``. `hist` is stored under its own
    key as well as under the real stage it aliases, so a caller may ask for
    either. Clear the dict between batches (the runner does).

    Works unchanged for SAGAViT, a plain timm ViT, and the timm 4-register
    model — register models are NOT wrapped in SAGAViT (it refuses
    num_prefix_tokens != 1 by design; saga/vit.py:125-136).
    """
    wanted = tuple(stages)
    for s in wanted:
        resolve_stage(s)                      # validates, raises on unknown
    real = {resolve_stage(s) for s in wanted}
    # real -> EVERY requested name that resolves to it. A dict of one name
    # would drop `hist` whenever `s12_pre_norm` was also asked for (or vice
    # versa, depending on iteration order).
    aliases: dict = {}
    for s in wanted:
        aliases.setdefault(resolve_stage(s), []).append(s)

    n_prefix = num_prefix_tokens(model)
    blocks = model_blocks(model)
    store: dict = {}
    handles = []

    def make_hook(real_name):
        def hook(_mod, _inp, out):
            tokens = out[0] if isinstance(out, (tuple, list)) else out
            patches = _patch_rows(tokens, n_prefix)
            store[real_name] = patches
            for alias in aliases.get(real_name, ()):
                store[alias] = patches
        return hook

    try:
        for name in sorted(real & (set(_BLOCK_STAGES) | set(_BLOCK_INPUT_SOURCE))):
            idx = stage_block_index(model, name)
            handles.append(blocks[idx].register_forward_hook(make_hook(name)))
        if "s12_post_norm" in real:
            handles.append(
                final_norm(model).register_forward_hook(
                    make_hook("s12_post_norm")))
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def forward_with_stages(model, x, stages=("s12_pre_norm",)):
    """(logits, {stage: [B, N_patches, D]}) for one batch.

    `logits` are the model's NATIVE head output — SAGAViT.forward and timm's
    forward both return class logits, so nothing here re-implements a
    readout.
    """
    was_training = model.training
    model.eval()
    try:
        with capture_stages(model, stages) as store:
            logits = model(x)
            captured = dict(store)
    finally:
        if was_training:
            model.train()
    missing = [s for s in stages if s not in captured]
    if missing:
        raise StageError(
            f"stages {missing} were not captured — the forward pass did not "
            f"reach the hooked module(s)")
    return logits, captured
