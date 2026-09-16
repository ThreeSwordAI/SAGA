"""
saga/frozen/reference.py
========================
TASK B / I3 B2a — the NATIVE REFERENCE of an edited forward pass.

Two quantities, both defined against "what this batch would have done
without the edit":

    max_abs_logit_diff_vs_native   the I0 record column; the runner already
                                   caches native logits per batch for it
    delta_update_norm              NEW: per image, the size of the change the
                                   edit made to the attention-branch residual
                                   update at the edited block

`delta_update_norm` is what makes I3's energy stratification possible. A gate
collapsed to its per-head mean and a gate permuted within its rings are not
the same size of intervention, and comparing their NLL effects without saying
so would compare an effect with an amplitude. TASK B §3 therefore reports
ΔNLL within deciles of this quantity, so the four condition families are
compared at equal injected energy.

THE DEFINITION
--------------
A pre-norm ViT block is two residual updates:

    x = x + drop_path1(ls1(attn(norm1(x))))        <- the ATTENTION branch
    x = x + drop_path2(ls2(mlp(norm2(x))))

`u` is the first of those, restricted to the PATCH rows (the model's own
prefix count, never a hard-coded 1), and

    delta_update_norm_i = || u_edit_i - u_native_i ||_F

the Frobenius norm over [n_patches, D] for image i. `original` records 0 by
definition: it IS the native forward.

WHY THE NATIVE UPDATE IS RECOMPUTED AND NOT CACHED
--------------------------------------------------
TASK B §3 says "computed against the cached native forward of the same
batch". Cached across the whole sweep it cannot be: the runner's loop is
CONDITION-MAJOR (every batch of condition 1, then every batch of condition
2), so a cache of u_native would have to hold every batch at once —
10,000 images x 196 patches x 384 dims x 4 B x 2 layers = 6.0 GB for ViT-S
and 12.0 GB for ViT-B, in a job whose memory request this repository does not
document. That is a silent OOM waiting for the largest checkpoint.

It does not have to be cached, because of what a single-layer edit is. Every
I3 condition edits ONE block, so the residual stream ENTERING that block is
bit-identical to the native one — nothing upstream of block l has changed.
So u_native can be recomputed from the block's own input, inside the same
batch, by evaluating the block's own attention branch once more after the
edit context has restored the gate:

    u_native = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))

That is the block's own expression, with the block's own modules, not a
hand-derived matrix identity. It costs one attention branch (~1/24 of a
forward) instead of 6-12 GB, and
`tests/test_I3_gate_edits.py::test_the_recomputed_native_update_is_the_blocks_own`
pins it by asserting the recomputation reproduces a HOOKED native update
bit-identically on a real forward.

Both hooks sit on modules that no edit ever swaps — the block itself
(forward pre-hook, for `x`) and `drop_path1` (forward hook, for `u_edit`) —
so they are registered ONCE around the whole condition sweep. `gate_edit`
replaces `attn.gate`; a hook on the gate would have to be re-registered
inside every edit, and a hook on `attn` would be ordered against
`receiver_perturbation`'s returning hook (I0 handoff §5: a returning forward
hook replaces the output only for hooks registered after it). `drop_path1` is
downstream of both and sees what the block actually adds.

NO TRAINING, NO OPTIMIZER, NO SAMPLING. Every tensor here is read from a
forward pass under `torch.no_grad()`; nothing in this module draws a random
number, and `tests/test_I0_frozen.py` enforces that at the AST level.
"""

from contextlib import contextmanager

import torch

from saga.frozen.edits import EditError
from saga.frozen.stages import model_blocks, num_prefix_tokens

MISSING = "MISSING"

#: The modules the attention branch is composed of, in the order a pre-norm
#: timm `Block` composes them. `blocks[l].norm1 -> attn -> ls1 -> drop_path1`,
#: and the block adds the result to its input. Named here so that a model
#: whose blocks are built differently is REFUSED with the list it failed,
#: rather than measured against three quarters of an expression.
ATTENTION_BRANCH = ("norm1", "attn", "ls1", "drop_path1")


class ReferenceError(ValueError):
    """A reference measurement this module refuses to make."""


def attention_branch_tail(block):
    """The module whose output is what the block adds to its input.

    `drop_path1` in current timm, `ls1` if a block predates it, `attn` if a
    block has neither. Returned as (module, name) so the caller can record
    WHICH module it measured — a block without `ls1` and a block whose `ls1`
    is `nn.Identity` are different situations and only one of them is a
    silent approximation.
    """
    for name in ("drop_path1", "ls1", "attn"):
        mod = getattr(block, name, None)
        if mod is not None:
            return mod, name
    raise ReferenceError(
        f"{type(block).__name__} has none of drop_path1 / ls1 / attn, so the "
        f"attention-branch residual update cannot be measured on it. "
        f"delta_update_norm is defined only for a pre-norm block of the shape "
        f"x = x + drop_path1(ls1(attn(norm1(x)))).")


def attention_branch_update(block, x: torch.Tensor) -> torch.Tensor:
    """`drop_path1(ls1(attn(norm1(x))))` — the block's OWN expression.

    Evaluated with the block's own modules in the block's own order, so this
    tracks `timm.models.vision_transformer.Block.forward` rather than
    restating it as arithmetic. A block missing any of the four is refused:
    silently dropping `ls1` would scale every measured delta by the
    LayerScale gamma this project's ViT-S checkpoints do not even use, and
    the error would be invisible in the tables.
    """
    missing = [n for n in ATTENTION_BRANCH if getattr(block, n, None) is None]
    if missing:
        raise ReferenceError(
            f"{type(block).__name__} is missing {missing} of the attention "
            f"branch {list(ATTENTION_BRANCH)}; refusing to recompute the "
            f"native update from part of the expression")
    return block.drop_path1(block.ls1(block.attn(block.norm1(x))))


class UpdateReference:
    """Per-image ||u_edit - u_native|| at each watched block.

    Built ONCE per checkpoint, BEFORE any edit is entered, and used for the
    whole condition sweep:

        with UpdateReference(model, layers=(7, 8)) as ref:
            for cond in conditions:
                layer = ref.layer_of(cond)          # None -> native condition
                for images, targets in loader:
                    run_condition_multistage(...)   # the edited forward
                    deltas = ref.delta_update_norm(layer)   # [B] floats

    `delta_update_norm` must be called AFTER the edited forward has returned
    and its edit context has exited, because that is when the model is native
    again and `attention_branch_update` recomputes the unedited branch. The
    runner calls it exactly there; calling it while an edit is installed
    would silently measure zero, so the class refuses a call it was not armed
    for.
    """

    def __init__(self, model, layers=(7, 8)):
        blocks = model_blocks(model)
        depth = len(blocks)
        self.layers = tuple(sorted({int(l) for l in layers}))
        for layer in self.layers:
            if not 0 <= layer < depth:
                raise ReferenceError(
                    f"layer {layer} is out of range for a {depth}-block "
                    f"model (valid 0..{depth - 1})")
        self.model = model
        self.n_prefix = num_prefix_tokens(model)
        self.tail_names = {}
        for layer in self.layers:
            _, name = attention_branch_tail(blocks[layer])
            self.tail_names[layer] = name
        self._x = {}                  # layer -> block input of the last forward
        self._u = {}                  # layer -> u_edit of the last forward
        self._handles = []
        self._n_forwards = 0

    # ── the two hooks, on modules no edit ever swaps ─────────────────────────

    def _make_input_hook(self, layer):
        def pre_hook(_mod, args, *_rest):
            if args:
                self._x[layer] = args[0]
        return pre_hook

    def _make_update_hook(self, layer):
        def hook(_mod, _inp, out):
            self._u[layer] = out[0] if isinstance(out, (tuple, list)) else out
            self._n_forwards += 1
        return hook

    def __enter__(self):
        blocks = model_blocks(self.model)
        try:
            for layer in self.layers:
                self._handles.append(
                    blocks[layer].register_forward_pre_hook(
                        self._make_input_hook(layer), with_kwargs=True))
                tail, _ = attention_branch_tail(blocks[layer])
                self._handles.append(
                    tail.register_forward_hook(self._make_update_hook(layer)))
        except Exception:                                # pragma: no cover
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._x.clear()
        self._u.clear()

    # ── the measurement ──────────────────────────────────────────────────────

    def clear(self):
        """Drop the captured tensors of the last forward."""
        self._x.clear()
        self._u.clear()

    def _patch_rows(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 3:
            raise ReferenceError(
                f"expected a [B, N, D] token tensor, got {tuple(t.shape)}")
        if t.shape[1] <= self.n_prefix:
            raise ReferenceError(
                f"token sequence has {t.shape[1]} rows but the model reports "
                f"{self.n_prefix} prefix tokens — no patch rows would remain")
        return t[:, self.n_prefix:, :]

    @torch.no_grad()
    def delta_update_norm(self, layer, n_images=None):
        """`[n_images]` floats — ||u_edit - u_native||_F per image.

        `layer is None` is the native condition: the edit is the empty edit,
        so the answer is exactly 0.0 and no recomputation is done (it would
        return 0.0 the long way round and cost an attention branch per batch
        for 1 of the 61 conditions).

        MUST be called with the model in its NATIVE state — the runner calls
        it after `run_condition_multistage` has returned and the edit context
        has restored the gate. Under an installed edit the recomputed
        "native" branch would be the edited one and every delta would be 0.
        """
        if layer is None:
            if n_images is None:                         # pragma: no cover
                raise ReferenceError(
                    "the native condition needs n_images to size its zeros")
            return [0.0] * int(n_images)
        layer = int(layer)
        if layer not in self.layers:
            raise ReferenceError(
                f"layer {layer} is not watched; this reference was built for "
                f"{list(self.layers)}. The watched layers come from the "
                f"conditions file and are fixed before the job runs.")
        x = self._x.get(layer)
        u_edit = self._u.get(layer)
        if x is None or u_edit is None:
            raise ReferenceError(
                f"no forward pass through blocks[{layer}] has been captured "
                f"since the last clear(); delta_update_norm is measured on "
                f"the forward that has just run")
        u_native = attention_branch_update(model_blocks(self.model)[layer], x)
        d = self._patch_rows(u_edit).float() - self._patch_rows(u_native).float()
        out = d.flatten(1).norm(dim=1).tolist()
        if n_images is not None and len(out) != int(n_images):
            raise ReferenceError(
                f"captured {len(out)} image(s) at blocks[{layer}] but the "
                f"batch has {n_images} — the hook saw a different forward")
        return out

    def info(self) -> dict:
        """What was measured, for `run_meta.json`."""
        return {
            "layers": list(self.layers),
            "n_prefix": self.n_prefix,
            "tail_module": {str(k): v for k, v in self.tail_names.items()},
            "attention_branch": list(ATTENTION_BRANCH),
            "definition": (
                "delta_update_norm = ||u_edit - u_native||_F over the PATCH "
                "rows of the attention-branch residual update "
                "drop_path1(ls1(attn(norm1(x)))) at the edited block; "
                "u_native is recomputed from the block's own input after the "
                "edit context has restored the gate."),
            "n_forwards_seen": self._n_forwards,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Which layer a declared condition edits
# ─────────────────────────────────────────────────────────────────────────────

def condition_layer(condition: dict):
    """The 0-based block a declared condition edits, or None for a native one.

    Read from the condition's own `params`, never inferred from its `id`: an
    id is a label and a params block is the instruction, and TASK B's ids
    carry the layer only as a human-readable suffix.
    """
    if condition.get("edit_type") in ("native", "identity", None):
        return None
    params = condition.get("params") or {}
    layer = params.get("layer")
    return None if layer is None else int(layer)


def watched_layers(conditions: dict) -> tuple:
    """Every layer the declared conditions edit, sorted and deduplicated."""
    out = {condition_layer(c) for c in conditions.get("conditions", ())}
    return tuple(sorted(l for l in out if l is not None))


@contextmanager
def update_reference(model, conditions: dict, enabled: bool = True):
    """`UpdateReference` for the layers a conditions document edits, or None.

    The null case yields None rather than a no-op object, so a caller that
    forgets to check gets an AttributeError at the first use instead of a
    column full of zeros that looks like a measured result.
    """
    layers = watched_layers(conditions)
    if not enabled or not layers:
        yield None
        return
    with UpdateReference(model, layers) as ref:
        yield ref
