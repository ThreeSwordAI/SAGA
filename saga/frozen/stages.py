"""
saga/frozen/stages.py
=====================
Named feature stages, captured by forward hooks, with the prefix rows
already removed using the MODEL'S OWN prefix count.

Four names, and no fifth anywhere in the project:

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

# The four stage names. `hist` is an alias, resolved by resolve_stage().
STAGES = ("s11_out", "s12_pre_norm", "s12_post_norm", "hist")

#: Which real stage the historical diagnostics used. See the module docstring
#: for the code citation; a test pins it.
HIST_STAGE = "s12_pre_norm"

HIST_STAGE_CITATION = (
    "saga/metrics.py:287-291 (forward hooks on model.blocks[i], storing the "
    "block's own output), saga/metrics.py:324-325 (feats[L-1][:, P:, :]), "
    "saga/vit.py:238-241 (SAGAViT.forward applies self.norm AFTER the block "
    "loop), tools/diagnose.py:89-95 (the historical driver)"
)

#: Stages that are a block output (value = block index counted from the end).
_BLOCK_STAGES = {"s11_out": -2, "s12_pre_norm": -1}


class StageError(ValueError):
    """An unknown stage, or a model this module cannot hook."""


def resolve_stage(stage: str) -> str:
    """Map `hist` onto the real stage it aliases; validate every other name."""
    if stage == "hist":
        return HIST_STAGE
    if stage not in STAGES:
        raise StageError(f"unknown stage {stage!r}; expected one of {STAGES}")
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
    """0-based index of the block whose OUTPUT is `stage`.

    Raises for `s12_post_norm` (not a block output). On the 12-block
    production models this returns 10 for s11_out and 11 for s12_pre_norm.
    """
    real = resolve_stage(stage)
    if real not in _BLOCK_STAGES:
        raise StageError(f"stage {stage!r} ({real}) is not a block output")
    depth = len(model_blocks(model))
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
        for name in sorted(real & set(_BLOCK_STAGES)):
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
