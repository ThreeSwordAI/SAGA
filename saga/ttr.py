"""
saga/ttr.py
===========
Test-time registers (TTR) — Jiang, Dravid, Efros & Gandelsman, *Vision
Transformers Don't Need Trained Registers* (NeurIPS 2025 Spotlight),
arXiv:2506.08010.

PROVENANCE: this is a REIMPLEMENTATION, not vendored code. The authors'
implementation was read (github.com/nickjiang2378/test-time-registers at
commit 860df43515c8d8e9e90952af25a46c26e4469570) but carries NO license file
of its own, so it could not be vendored under `third_party/ttr/` as TASK-10
A1 prescribes. See `third_party/ttr/PROVENANCE.md` for the full record and
for every point where this implementation deviates from theirs.

The method, in two steps
------------------------
1. ``find_register_neurons`` — a sparse set of mid-layer MLP neurons ("register
   neurons") is responsible for the high-norm outlier tokens. Score every
   hidden neuron by its activation on the outlier tokens; rank descending.
2. ``apply_ttr`` — at inference, append untrained token(s) initialised to
   ZEROS and, at every layer holding selected neurons, write those neurons'
   activation into the extra token(s) while removing it from the patch
   tokens. The outliers move off the image grid; attention maps clean up.
   No training, no weight change.

Token layout contract (TASK-10 A2.2)
------------------------------------
The extra tokens are inserted as PREFIX tokens, directly after CLS::

    [CLS] [EXTRA x n] [PATCH x 196]

so ``num_prefix_tokens`` grows by ``n_extra_tokens`` and the patch tokens keep
their count and raster ordering. ``saga/metrics.py``, ``tools/diagnose.py``
and ``tools/sink_address.py`` therefore work on the patched model unchanged —
they all slice patches as ``x[:, P:, :]`` with
``P = infer_num_prefix_tokens(model)``.

The authors instead APPEND their registers ( ``[CLS][PATCH][REG]`` ). Self-
attention is permutation-equivariant and the extra tokens carry no positional
embedding, so the two placements give identical patch outputs; the prefix
placement is the one that satisfies the contract above. ``test_task10_ttr.py``
pins that equivalence numerically.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

import torch

#: Keys every neurons.json must carry. Pinned by a test so the schema cannot
#: drift out from under Phase C's collectors.
NEURONS_JSON_KEYS = (
    "schema", "criterion", "criterion_description", "outlier_tau",
    "tau_key", "tau_source", "seed", "arch", "variant", "ckpt",
    "ckpt_sha256", "git_sha", "timestamp", "n_images_seen", "n_images_scored",
    "layer_range", "num_layers_scanned", "num_neurons_per_layer",
    "num_prefix_tokens", "detect_outliers_layer", "top_n_stored", "neurons",
)

NEURONS_JSON_SCHEMA = "saga.ttr.neurons/v1"

# ─────────────────────────────────────────────────────────────────────────────
# Scoring criteria — the string is recorded in neurons.json, never implied
# ─────────────────────────────────────────────────────────────────────────────

#: The authors' criterion (shared/algorithms.py::find_register_neurons):
#: per image, mean |activation| over the outlier token positions; averaged
#: over the images that HAVE at least one outlier.
CRITERION_MEAN_ABS = "mean_abs_act_at_outliers_v1"

#: The contrast criterion suggested by TASK-10 A2.1: mean |activation| on
#: outlier tokens MINUS mean |activation| on the remaining patch tokens.
CRITERION_CONTRAST = "mean_abs_act_outliers_minus_rest_v1"

CRITERIA = {
    CRITERION_MEAN_ABS: (
        "per image: mean over outlier patch tokens of |h_l,j| for each MLP "
        "hidden neuron j of layer l, where outlier tokens are the patch "
        "tokens whose last-block L2 norm is STRICTLY above tau; averaged over "
        "the images that contain at least one outlier. Matches the authors' "
        "shared/algorithms.py (their apply_sparsity_filter=False path)."
    ),
    CRITERION_CONTRAST: (
        "per image: mean |h_l,j| over outlier patch tokens MINUS mean "
        "|h_l,j| over the non-outlier patch tokens; averaged over the images "
        "that contain at least one outlier AND at least one non-outlier. "
        "The contrast variant suggested by TASK-10 A2.1; NOT the authors'."
    ),
}

NORMAL_VALUES = ("zero", "mean", "same")


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def mlp_act_module(block) -> torch.nn.Module:
    """The module whose OUTPUT is the MLP hidden activation of `block`.

    timm's ``Mlp`` is fc1 -> act -> drop1 -> norm -> fc2 -> drop2, so the
    activation function's output is the hidden state that fc2 consumes (in
    eval mode drop1 is identity and norm is Identity for the stock ViT).
    This is the same hook point the authors use
    (``Dinov2HookManager.neuron_activation_component`` -> ``mlp.act``).
    """
    mlp = getattr(block, "mlp", None)
    act = getattr(mlp, "act", None)
    if act is None:
        raise TypeError(
            f"block {type(block).__name__} has no .mlp.act — TTR needs the "
            f"MLP hidden activation module (timm Mlp layout)")
    return act


def mlp_hidden_dim(block) -> int:
    """Number of MLP hidden neurons in `block` (= fc1.out_features)."""
    return int(block.mlp.fc1.out_features)


def _signed_max_per_image(sel: torch.Tensor) -> torch.Tensor:
    """The authors' ``sign_max``, generalised over a batch.

    ``sign_max(t)`` returns the element of ``t`` with the largest ABSOLUTE
    value, keeping its sign (their utils.py: ``pos_max if abs(pos_max) >
    abs(neg_max) else neg_max``). They reduce over the whole
    ``output[0, :, neuron_indices]`` slice — i.e. over tokens AND selected
    neurons jointly — producing ONE scalar per (image, layer), which is then
    broadcast to every selected neuron.

    They always run batch size 1; here the reduction is per batch element,
    which is the faithful generalisation.

    sel: [B, T, n_sel] -> [B]
    """
    flat = sel.reshape(sel.shape[0], -1)
    pos = flat.max(dim=1).values
    neg = flat.min(dim=1).values
    return torch.where(pos.abs() > neg.abs(), pos, neg)


def neurons_to_dict(neurons) -> dict:
    """Normalise a neuron selection to ``{layer: [neuron_idx, ...]}``.

    Accepts a list of ``(layer, neuron)`` or ``(layer, neuron, score)``
    tuples, or an already-grouped dict. Order within a layer follows the
    input order; duplicates are dropped (first occurrence wins).
    """
    if isinstance(neurons, dict):
        return {int(l): [int(n) for n in idxs] for l, idxs in neurons.items()
                if len(idxs)}
    grouped: dict = {}
    for entry in neurons:
        layer, neuron = int(entry[0]), int(entry[1])
        bucket = grouped.setdefault(layer, [])
        if neuron not in bucket:
            bucket.append(neuron)
    return grouped


def top_k(neurons: Sequence, k: int) -> list:
    """The first `k` entries of a descending-score ranking.

    ``find_register_neurons`` returns the ranking already sorted, so the
    ``--n-neurons`` sweep is a prefix of ONE scan — never a rescan per k.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return list(neurons[:k])


def _has_spatial_gate(model) -> bool:
    for blk in model.blocks:
        if getattr(getattr(blk, "attn", None), "gate", None) is not None:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — find the register neurons
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def find_register_neurons(
    model,
    loader,
    k: Optional[int] = None,
    layer_range: Optional[tuple] = None,
    *,
    outlier_tau: float,
    device=None,
    seed: int = 0,
    criterion: str = CRITERION_MEAN_ABS,
    detect_outliers_layer: int = -1,
    max_images: Optional[int] = 500,
) -> list:
    """Rank MLP hidden neurons by their activation on high-norm patch tokens.

    Args:
        model:   an eval-mode ViT exposing ``.blocks`` (SAGAViT or timm ViT).
        loader:  a NON-shuffling loader over the scan images. Determinism is
                 the caller's to preserve: the same loader order and the same
                 seed give a bit-identical ranking.
        k:       return only the top-k entries (None = the full ranking).
        layer_range: ``(lo, hi)`` half-open block range to scan, as in
                 ``range(lo, hi)``; None scans every block. Register neurons
                 live in MID layers, so a range both focuses the search and
                 cuts the activation memory.
        outlier_tau: absolute last-block PATCH-norm threshold defining an
                 outlier token. Pass the cell's canon tau — never a value
                 recalibrated on a patched model (TASK-10 acceptance).
        seed:    recorded, and used to seed torch for reproducibility.
        criterion: one of the CRITERION_* strings above.
        detect_outliers_layer: block whose output defines the norms
                 (default -1 = the last block, the authors' default and the
                 block every sink number in this repo is computed on).
        max_images: stop after this many images (the authors use 500).

    Returns:
        ``[(layer, neuron_idx, score), ...]`` sorted by score DESCENDING.

    Note (deviation from the authors, deliberate): outliers are detected among
    PATCH tokens only — ``x[:, P:]`` — never CLS or an existing register. The
    authors threshold the whole token sequence, but every tau in this repo is
    calibrated on ``last_block_patch_norms``, so applying it to CLS would be a
    category error (CLS norm is a different distribution; cf. cls_norm_ratio).
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {sorted(CRITERIA)}, "
                         f"got {criterion!r}")
    if _has_spatial_gate(model):
        raise ValueError(
            "find_register_neurons: this model carries SAGA SpatialGates. "
            "TTR is a baseline-model intervention (TASK-10 applies it to the "
            "four BASELINE checkpoints); the gate also hard-codes "
            "n_patches = N-1, which extra prefix tokens break. Refusing "
            "rather than producing misaligned gate maps.")

    torch.manual_seed(seed)
    model.eval()
    from saga.metrics import infer_num_prefix_tokens, token_norms
    P = infer_num_prefix_tokens(model)

    n_blocks = len(model.blocks)
    lo, hi = (0, n_blocks) if layer_range is None else layer_range
    lo, hi = int(lo), int(hi)
    if not (0 <= lo < hi <= n_blocks):
        raise ValueError(
            f"layer_range {(lo, hi)} is not a valid half-open range within "
            f"[0, {n_blocks})")
    layers = list(range(lo, hi))
    hidden = mlp_hidden_dim(model.blocks[0])

    if device is None:
        device = next(model.parameters()).device

    acts: dict = {}
    outs: dict = {}
    hooks = [
        mlp_act_module(model.blocks[i]).register_forward_hook(
            lambda mod, inp, out, i=i: acts.__setitem__(i, out.detach()))
        for i in layers
    ]
    det_idx = detect_outliers_layer % n_blocks
    hooks.append(model.blocks[det_idx].register_forward_hook(
        lambda mod, inp, out: outs.__setitem__("x", out.detach())))

    score_sum = torch.zeros(len(layers), hidden, dtype=torch.float64,
                            device=device)
    n_scored = 0          # images that contributed a score
    n_seen = 0            # images run through the model

    try:
        for batch in loader:
            if max_images is not None and n_seen >= max_images:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            if max_images is not None:
                images = images[: max_images - n_seen]
            images = images.to(device, non_blocking=True)
            n_seen += images.shape[0]

            model(images)

            patch = outs["x"][:, P:, :]
            norms = token_norms(patch)                    # [B, N]
            is_out = norms > float(outlier_tau)           # [B, N]
            n_out = is_out.sum(dim=1)                     # [B]
            n_rest = (~is_out).sum(dim=1)

            usable = n_out > 0
            if criterion == CRITERION_CONTRAST:
                usable = usable & (n_rest > 0)
            if not bool(usable.any()):
                acts.clear(); outs.clear()
                continue

            w_out = (is_out.float() / n_out.clamp_min(1).unsqueeze(1))
            if criterion == CRITERION_CONTRAST:
                w_rest = ((~is_out).float()
                          / n_rest.clamp_min(1).unsqueeze(1))

            for row, layer in enumerate(layers):
                a = acts[layer][:, P:, :].abs().to(torch.float64)   # [B,N,H]
                s = torch.bmm(w_out.to(a.dtype).unsqueeze(1), a).squeeze(1)
                if criterion == CRITERION_CONTRAST:
                    s = s - torch.bmm(
                        w_rest.to(a.dtype).unsqueeze(1), a).squeeze(1)
                score_sum[row] += s[usable].sum(dim=0)

            n_scored += int(usable.sum())
            acts.clear(); outs.clear()
    finally:
        for h in hooks:
            h.remove()

    if n_scored == 0:
        raise RuntimeError(
            f"no image produced an outlier token at tau={outlier_tau} over "
            f"{n_seen} images — lower the threshold or scan more images "
            f"(the authors' algorithms.py raises the same way)")

    mean_scores = (score_sum / n_scored).flatten()
    order = torch.argsort(mean_scores, descending=True)
    ranking = [(layers[int(i) // hidden], int(i) % hidden,
                float(mean_scores[i]))
               for i in order.tolist()]
    ranking = ranking if k is None else ranking[:k]

    find_register_neurons.last_scan_stats = {      # for the caller's JSON
        "n_images_seen": n_seen,
        "n_images_scored": n_scored,
        "num_layers_scanned": len(layers),
        "layer_range": [lo, hi],
        "num_neurons_per_layer": hidden,
        "num_prefix_tokens": P,
        "detect_outliers_layer": int(detect_outliers_layer),
    }
    return ranking


# ─────────────────────────────────────────────────────────────────────────────
# neurons.json — the committed artifact
# ─────────────────────────────────────────────────────────────────────────────

def write_neurons_json(path, neurons: Sequence, *, criterion: str,
                       outlier_tau: float, tau_key: str, tau_source: str,
                       seed: int, arch: str, variant: str, ckpt: str,
                       ckpt_sha256: str, scan_stats: dict,
                       top_n_stored: int = 256) -> dict:
    """Write the neuron ranking with everything needed to reproduce it.

    Only the top `top_n_stored` entries are stored: the full ranking is
    num_layers x hidden_dim (18 432 entries for ViT-S, 36 864 for ViT-B),
    which has no business in git when the largest sweep value is 64. The cut
    is recorded so a reader knows the file is a prefix, not the whole ranking.

    Written atomically (tmp + os.replace) — a results JSON must never be
    found truncated, the rule TASK-02C's apply_fixed_thr fix established.
    """
    from datetime import datetime, timezone
    from saga.run_registry import git_sha

    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")

    payload = {
        "schema": NEURONS_JSON_SCHEMA,
        "criterion": criterion,
        "criterion_description": CRITERIA[criterion],
        "outlier_tau": float(outlier_tau),
        "tau_key": tau_key,
        "tau_source": tau_source,
        "seed": int(seed),
        "arch": arch,
        "variant": variant,
        "ckpt": ckpt,
        "ckpt_sha256": ckpt_sha256,
        "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images_seen": int(scan_stats["n_images_seen"]),
        "n_images_scored": int(scan_stats["n_images_scored"]),
        "layer_range": list(scan_stats["layer_range"]),
        "num_layers_scanned": int(scan_stats["num_layers_scanned"]),
        "num_neurons_per_layer": int(scan_stats["num_neurons_per_layer"]),
        "num_prefix_tokens": int(scan_stats["num_prefix_tokens"]),
        "detect_outliers_layer": int(scan_stats["detect_outliers_layer"]),
        "top_n_stored": int(top_n_stored),
        "neurons": [[int(l), int(n), float(s)]
                    for l, n, s in neurons[:top_n_stored]],
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return payload


def load_neurons_json(path) -> dict:
    """Read a neurons.json, validating the schema and returning the payload.

    ``payload['neurons']`` comes back as a list of ``(layer, neuron, score)``
    tuples, ready for ``top_k`` / ``apply_ttr``.
    """
    with open(path) as f:
        payload = json.load(f)
    missing = [k for k in NEURONS_JSON_KEYS if k not in payload]
    if missing:
        raise ValueError(f"{path}: neurons.json missing keys {missing}")
    if payload["schema"] != NEURONS_JSON_SCHEMA:
        raise ValueError(f"{path}: schema {payload['schema']!r}, expected "
                         f"{NEURONS_JSON_SCHEMA!r}")
    payload["neurons"] = [(int(l), int(n), float(s))
                          for l, n, s in payload["neurons"]]
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — apply the test-time registers
# ─────────────────────────────────────────────────────────────────────────────

def _make_insert_hook(n_extra: int):
    """Pre-hook on block 0: splice n_extra ZERO tokens in after CLS.

    Position embeddings have already been added by this point, so the extra
    tokens carry none — matching the authors' "extra tokens initialised to
    zeros" (their README, *Adding New Models*). Inserting here rather than
    rewriting the model forward keeps ``head(x[:, 0])`` reading CLS and lets
    every existing hook-based tool see the widened sequence.
    """
    def hook(module, args):
        x = args[0]
        extra = x.new_zeros(x.shape[0], n_extra, x.shape[2])
        return (torch.cat([x[:, :1], extra, x[:, 1:]], dim=1),) + args[1:]
    return hook


def _make_intervene_hook(neuron_idx, n_extra: int, scale: float,
                         normal_values: str):
    """Forward hook on ``blocks[l].mlp.act`` implementing the redirection.

    Mirrors the authors' ``shared/hook_fn.py::activate_on_registers`` with
    their default ``normal_values='zero'``: the extra tokens receive
    ``scale * sign_max(...)`` at the selected neurons, and those neurons are
    zeroed on the patch tokens. CLS is never touched (their slice is
    ``[0, 1:-num_registers]``; ours is ``[:, 1+n_extra:]`` — the same tokens
    under the prefix layout).

    Out-of-place: the hook returns a new tensor rather than mutating the
    activation, so the intervention is safe under autograd too.
    """
    idx = list(neuron_idx)

    def hook(module, inputs, output):
        out = output.clone()
        sel = out[:, :, idx]                              # [B, T, n_sel]
        fill = scale * _signed_max_per_image(sel)         # [B]
        out[:, 1:1 + n_extra, idx] = fill[:, None, None].to(out.dtype)
        if normal_values == "zero":
            out[:, 1 + n_extra:, idx] = 0
        elif normal_values == "mean":
            patch = out[:, 1 + n_extra:, idx]
            out[:, 1 + n_extra:, idx] = patch.mean(dim=1, keepdim=True)
        elif normal_values == "same":
            pass
        else:
            raise ValueError(f"invalid normal_values: {normal_values!r}")
        return out
    return hook


class TTRModel:
    """Proxy presenting the PATCHED prefix-token count to the outside world.

    Why a proxy and not a mutated attribute: ``num_prefix_tokens`` has two
    readers with incompatible needs during a TTR forward.

    * ``SAGAViT._interpolate_pos_embed`` reads it to describe the pos-embed
      layout, and hard-errors unless it is exactly 1 (the B6 tripwire). It
      runs BEFORE block 0, i.e. before the extra tokens exist — so from its
      point of view the answer must stay 1, and it is right to insist.
    * ``saga.metrics.infer_num_prefix_tokens`` reads it to slice patch
      tokens, and must see ``1 + n_extra_tokens``.

    Setting the attribute satisfies the second and trips the first. The proxy
    satisfies both: the wrapped model is left BIT-IDENTICAL — no attribute of
    it is written at all — while every tool that asks the proxy gets the
    patched count. ``.blocks`` is delegated to the same ModuleList object, so
    hook-based tools (compute_diagnostics, capture_attention) attach to the
    real blocks.
    """

    def __init__(self, inner, num_prefix_tokens: int):
        object.__setattr__(self, "_ttr_inner", inner)
        object.__setattr__(self, "num_prefix_tokens", int(num_prefix_tokens))

    @property
    def ttr_inner(self):
        """The wrapped, unmodified model."""
        return object.__getattribute__(self, "_ttr_inner")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_ttr_inner"), name)

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, "_ttr_inner")(*args, **kwargs)

    def eval(self):
        object.__getattribute__(self, "_ttr_inner").eval()
        return self

    def train(self, mode: bool = True):
        object.__getattribute__(self, "_ttr_inner").train(mode)
        return self

    def to(self, *args, **kwargs):
        object.__getattribute__(self, "_ttr_inner").to(*args, **kwargs)
        return self

    def __repr__(self):
        inner = object.__getattribute__(self, "_ttr_inner")
        return (f"TTRModel(num_prefix_tokens={self.num_prefix_tokens}, "
                f"inner={type(inner).__name__})")


@contextmanager
def apply_ttr(model, neurons, n_extra_tokens: int = 1, *,
              scale: float = 1.0, normal_values: str = "zero"):
    """Patch `model` with test-time registers for the duration of the block.

    Args:
        model:   an eval-mode ViT exposing ``.blocks`` (SAGAViT or timm ViT).
        neurons: the selection — a ranking of ``(layer, neuron[, score])``
                 tuples or a ``{layer: [idx]}`` dict.
        n_extra_tokens: how many untrained zero tokens to append (paper: 1).
        scale:   multiplier on the redirected activation (authors' default 1).
        normal_values: what the selected neurons become on the patch tokens —
                 ``'zero'`` (authors' default), ``'mean'``, or ``'same'``
                 (append the token but do not clean the patches).

    Yields:
        a ``TTRModel`` proxy over `model`, reporting
        ``num_prefix_tokens = 1 + n_extra_tokens`` and forwarding everything
        else to the real model. On exit every hook is removed; `model` itself
        is never written to, so it is exactly as it was found.

    EMPTY SELECTION IS AN EXACT NO-OP: with no register neurons there is
    nothing to redirect, so no token is appended, no hook is installed, and
    the ORIGINAL model is yielded unwrapped. It is then bit-identical to the
    unpatched one — the A4 sanity check depends on this, and appending an
    unused zero token would NOT be a no-op (other tokens would attend to it).
    """
    grouped = neurons_to_dict(neurons)

    if not grouped:
        yield model
        return

    if normal_values not in NORMAL_VALUES:
        raise ValueError(f"normal_values must be one of {NORMAL_VALUES}, "
                         f"got {normal_values!r}")
    if n_extra_tokens < 1:
        raise ValueError(
            f"n_extra_tokens must be >= 1 for a non-empty neuron selection, "
            f"got {n_extra_tokens}")
    if getattr(model, "_ttr_active", False):
        raise RuntimeError("apply_ttr is already active on this model; "
                           "nesting would append tokens twice")
    if _has_spatial_gate(model):
        raise ValueError(
            "apply_ttr: this model carries SAGA SpatialGates, whose forward "
            "hard-codes n_patches = N-1. Extra prefix tokens would be gated "
            "as if they were patches (or raise a grid mismatch). TTR is a "
            "baseline-model intervention here — refusing.")

    n_blocks = len(model.blocks)
    hidden = mlp_hidden_dim(model.blocks[0])
    for layer, idxs in grouped.items():
        if not (0 <= layer < n_blocks):
            raise ValueError(f"layer {layer} out of range [0, {n_blocks})")
        bad = [i for i in idxs if not (0 <= i < hidden)]
        if bad:
            raise ValueError(
                f"neuron indices {bad[:5]} out of range [0, {hidden}) "
                f"for layer {layer}")

    base_prefix = int(getattr(model, "num_prefix_tokens", 1))
    handles = []
    try:
        handles.append(model.blocks[0].register_forward_pre_hook(
            _make_insert_hook(n_extra_tokens)))
        for layer, idxs in grouped.items():
            handles.append(
                mlp_act_module(model.blocks[layer]).register_forward_hook(
                    _make_intervene_hook(idxs, n_extra_tokens, scale,
                                         normal_values)))
        model._ttr_active = True
        yield TTRModel(model, base_prefix + n_extra_tokens)
    finally:
        for h in handles:
            h.remove()
        model._ttr_active = False
