"""
saga/frozen/edits.py
====================
Every intervention on a frozen model, as a context manager that proves it
was temporary: the model's parameter/buffer state is hashed on entry and on
exit, and the two MUST be equal (TASK I0 §2.8).

    with terminal_gate_override(model, 0.5):
        logits = model(images)       # edited
    # here the model is bit-identical to what it was before the `with`

Three interventions, one per downstream work package:

    terminal_gate_override(model, value)                 I2  (§6/D4)
    gate_edit(model, layer, mode, alpha=..., perm=...)   I3
    receiver_perturbation(model, layer, mask, epsilon)   I4

All three
- act on PATCH ROWS ONLY, excluding prefix rows with the model's own
  `infer_num_prefix_tokens` — 1 for baseline/SAGA, 5 for the timm
  4-register models (audit bug B2, plan §13.3);
- work on the timm 4-register model WITHOUT wrapping it in SAGAViT (which
  refuses num_prefix_tokens != 1 by design, saga/vit.py:125-136);
- refuse an unknown layer, a mask that reaches a prefix row, and a model
  whose prefix count cannot be inferred.

Nothing here mutates a parameter in place. An edit swaps a MODULE (the gate)
or attaches a forward hook, and restoring means putting the original object
back / removing the hook — so the restored state is the original tensors,
not a copy that merely compares equal.

No optimizer, no gradient, no training: every entry point runs under
`torch.no_grad()` semantics at the call site and touches nothing that a
backward pass would.
"""

import hashlib
import json
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from saga.frozen.stages import StageError, model_blocks, num_prefix_tokens

#: The constants I2 declares for the terminal patch gate. 1.0 is bypass.
TERMINAL_GATE_VALUES = (0.25, 0.5, 0.75, 1.0)

#: gate_edit modes (TASK I0 §6 / plan §6.6).
GATE_EDIT_MODES = ("original", "mean", "mean_plus_alpha_delta", "permute",
                   "permute_within_ring", "dihedral")

#: The 8 elements of the dihedral group of a square grid, as (k_rot90, flip).
#: Index 0 is (0, False) — THE IDENTITY. TASK B §3 requires `dihedral0` to be
#: bit-identical to `original`, so the ordering of this tuple is a contract
#: and not a convenience; `tests/test_I3_gate_edits.py` pins it.
DIHEDRAL_OPS = tuple((k, f) for k in range(4) for f in (False, True))

#: Human-readable name per DIHEDRAL_OPS index, for tables and notes.
DIHEDRAL_NAMES = ("identity", "flip_lr", "rot90", "rot90_flip_lr",
                  "rot180", "rot180_flip_lr", "rot270", "rot270_flip_lr")


class EditError(ValueError):
    """An intervention that cannot be applied as asked."""


# ─────────────────────────────────────────────────────────────────────────────
# State hashing — the proof that an edit was temporary
# ─────────────────────────────────────────────────────────────────────────────

def state_hash(model) -> str:
    """sha256 over every parameter and buffer: name, dtype, shape and bytes.

    Deterministic and order-independent (names are sorted). Tensors are moved
    to CPU and made contiguous first, so a device or stride difference cannot
    masquerade as a value difference.
    """
    h = hashlib.sha256()
    items = [("P", n, t) for n, t in model.named_parameters()]
    items += [("B", n, t) for n, t in model.named_buffers()]
    for kind, name, t in sorted(items, key=lambda it: (it[0], it[1])):
        h.update(kind.encode("utf-8"))
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        flat = t.detach().to("cpu").contiguous().reshape(-1)
        if flat.numel():
            # reinterpret as bytes: works for every dtype including bool and
            # bfloat16, which numpy cannot represent directly
            h.update(flat.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextmanager
def frozen_state(model, label: str = "edit"):
    """Assert the model's state is unchanged across the block.

    Every intervention in this module wraps itself in this, so a restore bug
    is a hard failure at the point of the edit and not a silently wrong
    number three work packages later.
    """
    before = state_hash(model)
    try:
        yield
    finally:
        after = state_hash(model)
        if before != after:
            raise EditError(
                f"{label}: model state was NOT restored "
                f"(before {before[:12]}, after {after[:12]}). Every frozen "
                f"intervention must leave the checkpoint bit-identical.")


# ─────────────────────────────────────────────────────────────────────────────
# Shared validation
# ─────────────────────────────────────────────────────────────────────────────

def check_layer(model, layer: int) -> int:
    """Validate a 0-based block index against the model's actual depth."""
    blocks = model_blocks(model)
    depth = len(blocks)
    if not isinstance(layer, (int, np.integer)) or isinstance(layer, bool):
        raise EditError(f"layer must be an int, got {layer!r}")
    layer = int(layer)
    if not 0 <= layer < depth:
        raise EditError(
            f"layer {layer} is out of range for a {depth}-block model "
            f"(valid 0..{depth - 1})")
    return layer


def block_gate(model, layer: int):
    """The gate module of block `layer`, or a clear refusal."""
    layer = check_layer(model, layer)
    attn = getattr(model_blocks(model)[layer], "attn", None)
    gate = getattr(attn, "gate", None) if attn is not None else None
    if gate is None:
        raise EditError(
            f"block {layer} of this {type(model).__name__} has no gate "
            f"(gate_mode='none', or a baseline/register model). "
            f"Gate interventions apply to SAGA checkpoints only.")
    return gate


def gate_map(gate, n_patches: int) -> torch.Tensor:
    """The gate's applied multiplier as [H, n_patches], fp32.

    `headscalar` gates carry one value per head and broadcast over
    positions; that broadcast IS the applied map, so it is expanded here
    rather than being reported as a length-1 spatial axis (the convention
    analysis/build_ablation_tables.py:126-132 already uses).
    """
    phi = getattr(gate, "phi", None)
    if phi is None:
        raise EditError(
            f"gate {type(gate).__name__} has no `phi` (LayerScaleGate is "
            f"per-channel, not spatial) — gate_edit applies to the "
            f"phi-parameterised gates only")
    g = torch.sigmoid(phi.detach().float())          # [H, n_slots]
    if g.shape[1] == 1:
        g = g.expand(g.shape[0], n_patches).contiguous()
    if g.shape[1] != n_patches:
        raise EditError(
            f"gate map has {g.shape[1]} positions but the token sequence has "
            f"{n_patches} patch rows")
    return g


def check_mask(mask, n_patches: int, n_prefix: int) -> np.ndarray:
    """Normalise a patch mask to a boolean array over PATCH coordinates.

    Accepts a boolean array of length n_patches or an integer index array.
    An index >= n_patches, a negative index, or a boolean array whose length
    is the TOKEN count (n_patches + n_prefix) is refused: the second is the
    exact shape of the bug this guard exists for — a mask written against
    token indices silently perturbs CLS or a register row.
    """
    arr = np.asarray(mask)
    if arr.dtype == bool:
        if arr.shape == (n_patches + n_prefix,):
            raise EditError(
                f"mask has {arr.size} entries, which is the TOKEN count "
                f"(n_patches={n_patches} + n_prefix={n_prefix}). Patch masks "
                f"are indexed over PATCH coordinates only — the prefix rows "
                f"are not maskable.")
        if arr.shape != (n_patches,):
            raise EditError(
                f"boolean mask has shape {arr.shape}, expected "
                f"({n_patches},) — one entry per patch coordinate")
        return arr.copy()
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise EditError(
            f"mask must be a boolean array over patch coordinates or an "
            f"integer index array, got dtype {arr.dtype}")
    idx = arr.astype(np.int64).reshape(-1)
    if idx.size and idx.min() < 0:
        raise EditError(
            f"mask index {int(idx.min())} is negative; patch coordinates are "
            f"0..{n_patches - 1} and negative indexing is refused because it "
            f"would wrap onto another patch")
    if idx.size and idx.max() >= n_patches:
        raise EditError(
            f"mask index {int(idx.max())} is outside the {n_patches} patch "
            f"coordinates. Indices are PATCH coordinates, not token indices: "
            f"the {n_prefix} prefix row(s) are never maskable.")
    out = np.zeros(n_patches, dtype=bool)
    out[idx] = True
    return out


def check_permutation(perm, n_patches: int) -> np.ndarray:
    perm = np.asarray(perm)
    if perm.shape != (n_patches,):
        raise EditError(
            f"permutation has shape {perm.shape}, expected ({n_patches},)")
    if not np.array_equal(np.sort(perm), np.arange(n_patches)):
        raise EditError(
            "permutation is not a bijection over the patch coordinates")
    return perm.astype(np.int64)


def dihedral_permutation(t: int, n_patches: int) -> np.ndarray:
    """The position permutation of dihedral transform `t` on a square grid.

    ONE definition of the 8 symmetries, shared by `build_edited_gate_map`
    (which applies them to a gate map) and by TASK B's I3 conditions (which
    name them and test them as ring-preserving bijections). A second
    derivation in the conditions file or the analysis layer could drift from
    this one, and the drift would be invisible: `dihedral{t}` would still be
    *a* symmetry, just not the one the tables say it is.

    Returns `idx` such that `g[:, idx]` is the transformed map, i.e. the
    value that lands at position p comes from position `idx[p]`. t = 0 is the
    identity and returns `arange(n_patches)` exactly.
    """
    if not isinstance(t, (int, np.integer)) or isinstance(t, bool):
        raise EditError(f"dihedral index must be an int, got {t!r}")
    t = int(t)
    if not 0 <= t < len(DIHEDRAL_OPS):
        raise EditError(
            f"dihedral index {t} out of range 0..{len(DIHEDRAL_OPS) - 1}")
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise EditError(
            f"{n_patches} patches is not a square grid — the dihedral group "
            f"is undefined")
    rot, flip = DIHEDRAL_OPS[t]
    grid = np.rot90(np.arange(n_patches).reshape(side, side), rot)
    if flip:
        grid = np.fliplr(grid)
    return np.ascontiguousarray(grid.reshape(-1))


def ring_of(n_patches: int) -> np.ndarray:
    """Ring label per patch coordinate, from THE ring definition.

    Imported from analysis/address_analysis.py (border_distance_map) rather
    than re-derived: TASK-07's answers and TASK-13's ring ablation were
    written against that one function, and a second derivation here could
    drift from the geometry they used. The import is lazy so that `saga`
    stays importable without the analysis layer.
    """
    try:
        from analysis.address_analysis import border_distance_map
    except ImportError as exc:                       # pragma: no cover
        raise EditError(
            f"ring edits need analysis.address_analysis.border_distance_map "
            f"(THE ring definition, TASK-07/TASK-13) and it is not "
            f"importable: {exc}. Run from the repo root.") from exc

    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise EditError(
            f"{n_patches} patches is not a square grid — rings are undefined")
    return border_distance_map(side).reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Replacement gate modules (used only inside a `with`)
# ─────────────────────────────────────────────────────────────────────────────

class _MapGate(nn.Module):
    """Applies a FIXED multiplier map to patch rows of an SDPA output.

    Input/output shape matches SpatialGate: [B, H, N, D]. The prefix rows
    (0 .. n_prefix-1) are left exactly as they came in. `g` is a plain
    tensor, registered as a buffer under a name no checkpoint uses, and this
    module never survives the `with` block that installs it.
    """

    def __init__(self, g: torch.Tensor, n_prefix: int):
        super().__init__()
        self.n_prefix = int(n_prefix)
        self.register_buffer("_frozen_gate_map", g.detach().clone(),
                             persistent=False)

    def set_current_grid_size(self, grid_size):
        """No-op: the map is already resolved for the grid it was built on.
        Present so SAGAViT._set_gate_grid can treat every gate alike."""
        return

    def forward(self, sdpa_out: torch.Tensor) -> torch.Tensor:
        B, H, N, D = sdpa_out.shape
        n_patches = N - self.n_prefix
        g = self._frozen_gate_map
        if g.shape != (H, n_patches):
            raise EditError(
                f"frozen gate map {tuple(g.shape)} does not match this "
                f"forward's [H={H}, n_patches={n_patches}]")
        out = sdpa_out.clone()
        out[:, :, self.n_prefix:, :] = (
            sdpa_out[:, :, self.n_prefix:, :]
            * g.to(sdpa_out.dtype).unsqueeze(0).unsqueeze(-1))
        return out


@contextmanager
def _swapped_gate(model, layer: int, new_gate: nn.Module, label: str):
    """Install `new_gate` on block `layer`, restore the ORIGINAL OBJECT."""
    layer = check_layer(model, layer)
    attn = model_blocks(model)[layer].attn
    original = attn.gate
    with frozen_state(model, label):
        attn.gate = new_gate
        try:
            yield new_gate
        finally:
            attn.gate = original


# ─────────────────────────────────────────────────────────────────────────────
# I2 — terminal patch-gate override
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def terminal_gate_override(model, value: float):
    """Set the LAST block's patch gate to the constant `value`.

    `value` = 1.0 is a bypass of that gate. SAGA (phi-parameterised)
    checkpoints only — a baseline or register model has no gate to override
    and is refused rather than silently treated as "already 1.0".

    Under plan §5.6 Proposition 2 the classification logits of a CLS-only
    readout are invariant under this edit; measuring that invariance is
    smoke check 3, not an assumption made here.
    """
    if value not in TERMINAL_GATE_VALUES:
        raise EditError(
            f"terminal gate value {value!r} is not one of the declared "
            f"constants {TERMINAL_GATE_VALUES}; I0 does not accept an "
            f"unlisted operating point")
    depth = len(model_blocks(model))
    layer = depth - 1
    gate = block_gate(model, layer)
    n_prefix = num_prefix_tokens(model)
    phi = getattr(gate, "phi", None)
    if phi is None:
        raise EditError(
            f"the last block's gate is {type(gate).__name__}, which has no "
            f"spatial map to override (terminal_gate_override is for the "
            f"phi-parameterised SAGA gate)")
    n_heads = phi.shape[0]
    n_patches = gate.n_patches if hasattr(gate, "n_patches") else phi.shape[1]
    g = torch.full((n_heads, int(n_patches)), float(value),
                   dtype=torch.float32, device=phi.device)
    with _swapped_gate(model, layer, _MapGate(g, n_prefix),
                       f"terminal_gate_override(value={value})"):
        yield {"layer": layer, "value": float(value), "n_prefix": n_prefix,
               "n_heads": int(n_heads), "n_patches": int(n_patches)}


# ─────────────────────────────────────────────────────────────────────────────
# I3 — gate mean / arrangement / dihedral edits
# ─────────────────────────────────────────────────────────────────────────────

def build_edited_gate_map(g: torch.Tensor, mode: str, *, alpha=None,
                          perm=None) -> torch.Tensor:
    """The edited multiplier map, as a pure function of the original map.

    g: [H, n_patches] (already sigmoid'd, prefix-free). mu and delta are
    PER HEAD: mu_h = mean_p g[h, p], delta_h(p) = g[h, p] - mu_h.
    Permutations are SHARED ACROSS HEADS (plan §6.6).
    """
    if mode not in GATE_EDIT_MODES:
        raise EditError(f"mode must be one of {GATE_EDIT_MODES}, got {mode!r}")
    n_patches = g.shape[1]

    if mode == "original":
        return g.clone()

    if mode == "mean":
        return g.mean(dim=1, keepdim=True).expand_as(g).contiguous()

    if mode == "mean_plus_alpha_delta":
        if alpha is None:
            raise EditError("mode 'mean_plus_alpha_delta' requires alpha")
        mu = g.mean(dim=1, keepdim=True)
        return (mu + float(alpha) * (g - mu)).contiguous()

    if mode in ("permute", "permute_within_ring"):
        if perm is None:
            raise EditError(f"mode {mode!r} requires an explicit permutation "
                            f"(permutations are FIXED and shared across "
                            f"checkpoints on the same grid, never drawn here)")
        p = check_permutation(perm, n_patches)
        if mode == "permute_within_ring":
            rings = ring_of(n_patches)
            moved = rings[p]
            if not np.array_equal(moved, rings):
                bad = int(np.count_nonzero(moved != rings))
                raise EditError(
                    f"permutation moves {bad} coordinate(s) across rings; "
                    f"mode 'permute_within_ring' requires ring membership to "
                    f"be preserved exactly")
        return g[:, torch.as_tensor(p, device=g.device)].contiguous()

    # dihedral
    if perm is None:
        raise EditError(
            "mode 'dihedral' requires `perm` = the index of the transform in "
            f"DIHEDRAL_OPS (0..{len(DIHEDRAL_OPS) - 1})")
    k = int(np.asarray(perm).reshape(-1)[0]) if np.ndim(perm) else int(perm)
    idx = torch.as_tensor(dihedral_permutation(k, n_patches), device=g.device)
    return g[:, idx].contiguous()


@contextmanager
def gate_edit(model, layer: int, mode: str, *, alpha=None, perm=None):
    """Edit the spatial gate of block `layer` for the duration of the block.

    mode in GATE_EDIT_MODES; see build_edited_gate_map for the definitions.
    `perm` is a FIXED index array (permute / permute_within_ring) or a
    DIHEDRAL_OPS index (dihedral) supplied by the caller — I0 draws no
    permutations, so the same saved indices are used across checkpoints on
    the same grid (plan §6.6).
    """
    layer = check_layer(model, layer)
    gate = block_gate(model, layer)
    n_prefix = num_prefix_tokens(model)
    n_patches = int(getattr(gate, "n_patches", 0)) or int(gate.phi.shape[1])
    g = gate_map(gate, n_patches)
    edited = build_edited_gate_map(g, mode, alpha=alpha, perm=perm)
    label = f"gate_edit(layer={layer}, mode={mode})"
    with _swapped_gate(model, layer, _MapGate(edited, n_prefix), label):
        yield {
            "layer": layer, "mode": mode, "alpha": alpha,
            "n_prefix": n_prefix, "n_patches": n_patches,
            "n_heads": int(g.shape[0]),
            "gate_mean_before": float(g.mean()),
            "gate_mean_after": float(edited.mean()),
            "gate_per_head_mean_before": [float(v) for v in g.mean(dim=1)],
            "gate_per_head_mean_after": [float(v) for v in edited.mean(dim=1)],
        }


# ─────────────────────────────────────────────────────────────────────────────
# I4 — receiver-output perturbation at masked coordinates
# ─────────────────────────────────────────────────────────────────────────────

def _proj_bias(attn):
    """The attention output projection's bias, or None.

    Plan §5.5 defines the patch attention update EXCLUDING the projection
    bias: U_ip = sum_h Zhat_ihp W^O_h. The module's output is U + b, so the
    bias is subtracted before scaling and added back after — which is what
    makes `(1 - alpha) * U + b` the edit the mathematics describes.
    """
    proj = getattr(attn, "proj", None)
    return None if proj is None else getattr(proj, "bias", None)


class _PerturbationRecord(dict):
    """Per-forward log for receiver_perturbation (a dict so the caller can
    read it after the `with` block as plain data)."""


@contextmanager
def receiver_perturbation(model, layer: int, mask, epsilon: float, *,
                          energy_target=None):
    """Attenuate the patch attention output at masked coordinates.

    At block `layer`, after any native SAGA gate and BEFORE the attention
    residual addition (the module output is exactly that tensor), the
    bias-free update U is multiplied by (1 - alpha) at the masked PATCH
    coordinates; prefix rows are untouched.

    alpha:
      - `energy_target` None      -> alpha = epsilon for every image.
      - `energy_target` = kappa   -> per-image alpha_i = kappa_i / ||M .* U_i||_F
        (plan §5.5), so every candidate mask injects the same Frobenius
        energy for that image. kappa may be a scalar or one value per image.
        A zero denominator gives alpha_i = 0 and is COUNTED, never dropped.

    The yielded record fills in on each forward with the MEASURED injected
    norm per image (``||dY_i||_F``), the native ``||M .* U_i||_F``, the
    applied alphas, and the number of zero-norm images.
    """
    layer = check_layer(model, layer)
    if not 0.0 <= float(epsilon) <= 1.0:
        raise EditError(
            f"epsilon must lie in [0, 1] (a weak attenuation), got {epsilon!r}")
    n_prefix = num_prefix_tokens(model)
    attn = model_blocks(model)[layer].attn
    bias = _proj_bias(attn)

    record = _PerturbationRecord(
        layer=layer, epsilon=float(epsilon), n_prefix=n_prefix,
        energy_matched=energy_target is not None,
        measured_perturbation_norm=[], native_masked_norm=[], alpha=[],
        n_zero_norm_images=0, n_masked_coords=None, n_batches=0)

    state = {"mask": None}

    def hook(_mod, _inp, out):
        tokens = out[0] if isinstance(out, (tuple, list)) else out
        if tokens.ndim != 3:
            raise EditError(
                f"attention output has shape {tuple(tokens.shape)}, expected "
                f"[B, N, C]")
        n_patches = tokens.shape[1] - n_prefix
        if n_patches <= 0:
            raise EditError(
                f"token sequence has {tokens.shape[1]} rows but the model "
                f"reports {n_prefix} prefix tokens")
        if state["mask"] is None:
            state["mask"] = check_mask(mask, n_patches, n_prefix)
            record["n_masked_coords"] = int(state["mask"].sum())
        m = torch.as_tensor(state["mask"], device=tokens.device)

        patches = tokens[:, n_prefix:, :]
        u = patches if bias is None else patches - bias.to(patches.dtype)
        u_masked = u * m.view(1, -1, 1).to(u.dtype)
        norm = u_masked.flatten(1).float().norm(dim=1)          # [B]

        if energy_target is None:
            alpha = torch.full_like(norm, float(epsilon))
        else:
            kappa = torch.as_tensor(energy_target, dtype=torch.float32,
                                    device=norm.device).reshape(-1)
            if kappa.numel() not in (1, norm.numel()):
                raise EditError(
                    f"energy_target has {kappa.numel()} values but the batch "
                    f"has {norm.numel()} images")
            kappa = kappa.expand_as(norm) if kappa.numel() == 1 else kappa
            alpha = torch.where(norm > 0, kappa / norm.clamp_min(1e-30),
                                torch.zeros_like(norm))

        delta = u_masked * alpha.view(-1, 1, 1).to(u_masked.dtype)

        record["n_zero_norm_images"] += int((norm == 0).sum())
        record["native_masked_norm"].extend(norm.tolist())
        record["alpha"].extend(alpha.tolist())
        # MEASURED, from the tensor actually subtracted — not alpha*norm,
        # which would only restate the arithmetic the hook was asked to do
        record["measured_perturbation_norm"].extend(
            delta.flatten(1).float().norm(dim=1).tolist())
        record["n_batches"] += 1

        new = tokens.clone()
        new[:, n_prefix:, :] = patches - delta
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        if isinstance(out, list):
            return [new] + list(out[1:])
        return new

    label = (f"receiver_perturbation(layer={layer}, epsilon={epsilon}, "
             f"energy_matched={energy_target is not None})")
    with frozen_state(model, label):
        handle = attn.register_forward_hook(hook)
        try:
            yield record
        finally:
            handle.remove()


def edit_params_json(params: dict) -> str:
    """Canonical JSON for the `edit_params` record column."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"),
                      default=float)
