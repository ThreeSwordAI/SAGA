"""
saga/frozen/attention.py
========================
TASK C / I7 §5 — incoming attention at the exceedance positions.

    check_attention_block(...)     the runner's single hook
    attention_summaries(...)       one forward -> the per-image summaries
    run_attention_package(...)     the loop that writes incoming_mass.npz

THE QUESTION
------------
This project measures NORM outliers. The term the literature attaches to
such tokens is a claim about INCOMING ATTENTION, and that claim has never
been measured at the positions this paper reports. I7 measures it, against a
wording rule declared in advance (D10, `analysis/i7_wording.py`), so the term
is earned or dropped by a number rather than by habit.

Nothing in THIS module decides the wording, and the term itself is not
written here: `analysis/i7_wording.py` is the one file in this repository
allowed to spell it, and `tests/test_I7_attention.py` enforces that (§11).
What this module produces is the measurement the rule consumes.

HOW THE MAP IS OBTAINED — AND A CORRECTION TO THE TASK FILE
-----------------------------------------------------------
Task file §5 says "with fused attention disabled for the dump (as
`tools/dump_attention.py` does)". That is NOT what this repository does, and
the difference matters enough to write down.

`tools/dump_attention.py` uses `saga.attn_extract.capture_attention`, which
does not disable anything. It monkey-patches each attention module's
`forward` so that the explicit `softmax(q·kᵀ·scale)` map is ALSO computed and
stashed, while the ORIGINAL fused forward still produces the module's output.
Model outputs are therefore bit-identical with and without capture — which
`tests/test_attn_extract.py` already pins — and there is no `fused_attn` flag
anywhere in this repository to toggle: `saga/vit.py` calls
`F.scaled_dot_product_attention` unconditionally, and the timm register
models carry their own `fused_attn` attribute which this code never touches.

So this module REUSES `capture_attention` rather than adding a second
mechanism. What the task file actually wants guaranteed — that the model is
exactly as it was afterwards — is guaranteed and tested three ways:
`state_hash` before/after, every `fused_attn` flag that exists before/after,
and that no attention module is left with a patched instance `forward`.

WHAT IS STORED, AND WHAT IS NOT
-------------------------------
NO ATTENTION MAPS. A [B, H, T, T] map for two blocks over 1,000 images is
tens of gigabytes; §5 stores per-image, per-KEY-position summaries only, and
they are computed inside the forward and discarded with it:

    in_mass_patchq[k]   mean over heads AND PATCH QUERIES of A[..., q, k]
    in_mass_clsq[k]     mean over heads of A[..., cls, k]
    value_norm[k]       ||v_k|| averaged over heads (the Fesser et al.
                        NOP/broadcast diagnostic; EXPLORATORY)

The key axis is the FULL token axis (197 for a 1-prefix model, 201 for a
4-register model), NOT just the patches, because the mass landing on each
PREFIX token is one of the things §5 asks for and a register model's four
register tokens are prefix positions. `n_prefix` is recorded so a reader
slices the two regions without guessing.

`in_mass_patchq` is a mean over queries, not a sum, so it is comparable
across models with different token counts; each row of the underlying map
sums to 1 (tested), so the mean over keys of `in_mass_patchq` is 1/T.

THE ALIGNMENT (§5) — THE OFF-BY-ONE THAT WOULD INVERT THE RESULT
----------------------------------------------------------------
Attention inside `blocks[l]` is computed from that block's INPUT tokens, so
it is compared with the exceedance of THOSE tokens:

    blocks[10]  <->  exceedance at `in_b10`   (= output of blocks[9])
    blocks[11]  <->  exceedance at `s11_out`  (= output of blocks[10])

Comparing a block's incoming attention with the norms of its own OUTPUT
would place the effect one block after its cause, and would do so silently.
`in_b10` was added to `saga/frozen/stages.py` for exactly this, additively,
and a test pins it against `blocks[9]` by comparing tensors.

Exceedance is recorded on TWO bases: MAD (PRIMARY, threshold-free, the
paper's own per-image rule) and I2's `tau_cal[s11_out]` for the cell where it
is defined (SECONDARY). Both are stored per image and per position; neither
is chosen after looking at the result.
"""

import json
from pathlib import Path

import numpy as np
import torch

from saga.frozen.correspondence import exceedance_flags
from saga.frozen.stages import capture_stages, model_blocks, resolve_stage

MISSING = "MISSING"

#: The two blocks I7 measures, and the stage each is aligned against (§5).
#: A block's incoming attention is compared with the exceedance of the tokens
#: that ENTER it. Fixed here and in D10; the runner refuses anything else.
BLOCK_ALIGNMENT = {10: "in_b10", 11: "s11_out"}

#: The per-key quantities stored per image, per block.
QUANTITIES = ("in_mass_patchq", "in_mass_clsq", "value_norm")

#: The two exceedance bases. MAD is PRIMARY (D10); `tau_cal` is I2's
#: per-cell calibrated threshold at `s11_out` and is SECONDARY.
EXCEEDANCE_BASES = ("mad", "tau_cal")


class AttentionError(ValueError):
    """An attention configuration this module refuses to run."""


# ─────────────────────────────────────────────────────────────────────────────
# The runner hook (§7) — the conditions contract, extended to I7's keys
# ─────────────────────────────────────────────────────────────────────────────

def check_attention_block(conditions_yaml, doc: dict):
    """Validate a conditions document that declares `capture: attention`.

    Called from `saga/frozen/runner.py::load_conditions`, the project's ONE
    conditions parser, so I7's blocks are refused on the login node by the
    same contract that already refuses an unknown edit_type or stage. A
    document without the key is untouched — which is every work package but
    this one.
    """
    if doc.get("capture") is None:
        return doc
    if doc["capture"] != "attention":
        raise AttentionError(
            f"{conditions_yaml}: capture={doc['capture']!r}; the only "
            f"declared capture is 'attention' (TASK C / I7 §5)")

    blocks = doc.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise AttentionError(
            f"{conditions_yaml}: an attention capture must declare a "
            f"non-empty `blocks` list, got {blocks!r}")
    unknown = [b for b in blocks if b not in BLOCK_ALIGNMENT]
    if unknown:
        raise AttentionError(
            f"{conditions_yaml}: blocks {unknown} are not declared. D10 fixes "
            f"the set at {sorted(BLOCK_ALIGNMENT)}; a third block would be a "
            f"measurement chosen after the fact")
    if len(set(blocks)) != len(blocks):
        raise AttentionError(
            f"{conditions_yaml}: `blocks` repeats an entry: {blocks}")

    declared = doc.get("alignment") or {}
    for b in blocks:
        want = BLOCK_ALIGNMENT[b]
        got = declared.get(b, declared.get(str(b)))
        if got is not None and got != want:
            raise AttentionError(
                f"{conditions_yaml}: block {b} is declared aligned with "
                f"{got!r}, but §5 fixes it to {want!r} — the attention inside "
                f"a block is computed from that block's INPUT tokens")
        resolve_stage(want)                    # raises on an unknown stage
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Restoration guards
# ─────────────────────────────────────────────────────────────────────────────

def fused_attn_flags(model) -> dict:
    """{block index: bool} for every attention module that HAS the flag.

    timm's `Attention` carries `fused_attn`; this repository's
    `GatedAttention` does not (it calls SDPA unconditionally), so this
    returns an empty dict for a SAGA or baseline model built through
    `saga/vit.py` and a populated one for a timm register model. Either way
    the dict before and after a capture must be equal — that is the property
    the task file asks for, and it is checked rather than assumed.
    """
    out = {}
    for i, blk in enumerate(model_blocks(model)):
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(attn, "fused_attn"):
            out[i] = bool(attn.fused_attn)
    return out


def patched_forwards(model) -> list:
    """Indices of attention modules left carrying an INSTANCE `forward`.

    `capture_attention` patches `mod.forward` and deletes it on exit. A
    non-empty list after the context means a capture leaked into the model,
    which would make every later forward pay for a map nobody reads.
    """
    return [i for i, blk in enumerate(model_blocks(model))
            if getattr(blk, "attn", None) is not None
            and "forward" in blk.attn.__dict__]


# ─────────────────────────────────────────────────────────────────────────────
# The summaries
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def value_norms(attn_module, x: torch.Tensor) -> torch.Tensor:
    """||v_k|| averaged over heads, from the module's own qkv projection.

    [B, N, C] -> [B, N]. Mirrors the q/k/v path of
    `saga.attn_extract._explicit_attention_map` exactly — same projection,
    same reshape — because a value norm computed from a different unbind
    order would be a different quantity wearing the same name.
    """
    B, N, C = x.shape
    H = attn_module.num_heads
    head_dim = getattr(attn_module, "head_dim", C // H)
    qkv = attn_module.qkv(x).reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
    _q, _k, v = qkv.unbind(0)                       # [B, H, N, head_dim]
    return v.float().norm(dim=-1).mean(dim=1)       # [B, N]


@torch.no_grad()
def summarize_attention(attn_map: torch.Tensor, n_prefix: int,
                        cls_index: int = 0) -> dict:
    """[B, H, T, T] post-softmax map -> the per-image, per-key summaries.

    `in_mass_patchq` averages over heads AND over PATCH queries only: the
    CLS row is a different question (it is what a CLS-only classifier reads)
    and is kept separately as `in_mass_clsq`, which is what lets the wording
    rule be stated for patch queries specifically, as D10 states it.

    The map is consumed here and never returned: §7 stores no attention maps.
    """
    if attn_map.ndim != 4 or attn_map.shape[-1] != attn_map.shape[-2]:
        raise AttentionError(
            f"expected a square [B, H, T, T] attention map, got "
            f"{tuple(attn_map.shape)}")
    t = attn_map.shape[-1]
    if t <= n_prefix:
        raise AttentionError(
            f"the map has {t} tokens but the model reports {n_prefix} prefix "
            f"tokens — no patch queries would remain")
    a = attn_map.float()
    return {
        "in_mass_patchq": a[:, :, n_prefix:, :].mean(dim=(1, 2)),   # [B, T]
        "in_mass_clsq": a[:, :, cls_index, :].mean(dim=1),          # [B, T]
    }


@torch.no_grad()
def attention_summaries(model, images, blocks, *, n_prefix, tau_cal=None):
    """One forward: the per-key summaries and the aligned exceedance flags.

    Returns {block: {quantity: [B, T] or [B, P] tensor}}. The attention maps
    and the stage tensors are both released when this returns; only the
    summaries survive, which is the §7 storage rule as code.

    The stage captures and the attention capture happen in the SAME forward,
    so an exceedance flag and the attention it is compared with come from one
    computation and cannot drift.
    """
    from saga.attn_extract import capture_attention

    blocks = sorted(int(b) for b in blocks)
    unknown = [b for b in blocks if b not in BLOCK_ALIGNMENT]
    if unknown:
        raise AttentionError(
            f"blocks {unknown} have no declared alignment; §5 fixes "
            f"{BLOCK_ALIGNMENT}")
    depth = len(model_blocks(model))
    too_deep = [b for b in blocks if b >= depth]
    if too_deep:
        raise AttentionError(
            f"blocks {too_deep} do not exist on a {depth}-block model")

    stages = tuple(dict.fromkeys(BLOCK_ALIGNMENT[b] for b in blocks))
    inputs = {}

    hooks = []
    for b in blocks:
        mod = model_blocks(model)[b].attn

        def pre_hook(_m, args, _b=b):
            inputs[_b] = args[0]
        hooks.append(mod.register_forward_pre_hook(pre_hook))

    try:
        with capture_attention(model, blocks=blocks) as maps, \
                capture_stages(model, stages) as store:
            model(images)
            captured_maps = {b: maps[b] for b in blocks}
            captured_stages = {s: store[s].clone() for s in stages}
            captured_inputs = {b: inputs[b] for b in blocks}
    finally:
        for h in hooks:
            h.remove()

    out = {}
    for b in blocks:
        summary = summarize_attention(captured_maps[b], n_prefix)
        summary["value_norm"] = value_norms(model_blocks(model)[b].attn,
                                            captured_inputs[b])
        stage = BLOCK_ALIGNMENT[b]
        patches = captured_stages[stage]                    # [B, P, D]
        summary["exceedance_mad"] = exceedance_flags(patches)
        summary["exceedance_tau_cal"] = (
            _tau_flags(patches, tau_cal) if isinstance(tau_cal, float)
            else None)
        summary["stage"] = stage
        out[b] = summary
    return out


@torch.no_grad()
def _tau_flags(patches: torch.Tensor, tau: float) -> torch.Tensor:
    """Exceedance against I2's absolute calibrated threshold (SECONDARY)."""
    from saga.metrics import token_norms
    return token_norms(patches) > float(tau)


# ─────────────────────────────────────────────────────────────────────────────
# The sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_attention_package(*, row, conditions, dataset, out_dir, image_ids,
                          device="cpu", batch_size=16, split_name, split_sha,
                          git_sha, tau_cal=None, model=None, ckpt_path=None,
                          max_images=None):
    """Walk the split once, write `incoming_mass.npz`, return a summary.

    One npz per run rather than a row file: every quantity here is indexed by
    (image, block, KEY POSITION), and a long-format row per position would be
    1,000 x 2 x 201 = 402,000 rows per run carrying the provenance columns
    twice over. The npz carries the same provenance in `meta_json`.
    """
    from torch.utils.data import DataLoader

    from saga.frozen.edits import state_hash
    from saga.metrics import infer_num_prefix_tokens

    if model is None:
        from saga.frozen.runner import build_from_row
        model = build_from_row(row, ckpt_path=ckpt_path, device=device)
    model.eval()

    n_prefix = infer_num_prefix_tokens(model)
    blocks = sorted(int(b) for b in conditions["blocks"])
    hash_before = state_hash(model)
    fused_before = fused_attn_flags(model)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    acc = {q: {b: [] for b in blocks} for q in QUANTITIES}
    exc = {basis: {b: [] for b in blocks} for basis in EXCEEDANCE_BASES}
    seen = 0
    for images, _targets in loader:
        if max_images is not None:
            if seen >= max_images:
                break
            if seen + images.shape[0] > max_images:
                images = images[:max_images - seen]
        images = images.to(device)
        summaries = attention_summaries(model, images, blocks,
                                        n_prefix=n_prefix, tau_cal=tau_cal)
        for b in blocks:
            for q in QUANTITIES:
                acc[q][b].append(summaries[b][q].cpu().numpy().astype(
                    np.float32))
            exc["mad"][b].append(
                summaries[b]["exceedance_mad"].cpu().numpy())
            flags = summaries[b]["exceedance_tau_cal"]
            if flags is not None:
                exc["tau_cal"][b].append(flags.cpu().numpy())
        seen += images.shape[0]

    hash_after = state_hash(model)
    fused_after = fused_attn_flags(model)
    leaked = patched_forwards(model)
    if hash_before != hash_after:
        raise AttentionError(
            f"{row['run_id']}: model state changed across the attention "
            f"capture ({hash_before[:12]} -> {hash_after[:12]})")
    if fused_before != fused_after:
        raise AttentionError(
            f"{row['run_id']}: fused_attn flags changed across the capture: "
            f"{fused_before} -> {fused_after}")
    if leaked:
        raise AttentionError(
            f"{row['run_id']}: attention modules {leaked} still carry a "
            f"patched instance forward after the capture")

    arrays = {}
    for q in QUANTITIES:
        arrays[q] = np.stack(
            [np.concatenate(acc[q][b], axis=0) for b in blocks], axis=1)
    for basis in EXCEEDANCE_BASES:
        if any(exc[basis][b] for b in blocks):
            arrays[f"exceedance_{basis}"] = np.stack(
                [np.concatenate(exc[basis][b], axis=0) for b in blocks],
                axis=1)
    arrays["blocks"] = np.asarray(blocks, dtype=np.int64)
    arrays["aligned_stage"] = np.asarray(
        [BLOCK_ALIGNMENT[b] for b in blocks])
    arrays["image_ids"] = np.asarray(image_ids[:seen])

    meta = {
        "run_id": row["run_id"], "arch": row["arch"],
        "recipe_actual": row["recipe_actual"], "variant": row["variant"],
        "ckpt_kind": row["ckpt_kind"], "ckpt_sha256": row["ckpt_sha256"],
        "n_prefix": int(n_prefix), "n_images": int(seen),
        "blocks": blocks,
        "alignment": {str(b): BLOCK_ALIGNMENT[b] for b in blocks},
        "split_name": split_name, "split_sha256": split_sha,
        "precision": conditions.get("precision", "fp32"),
        "work_package": conditions["work_package"],
        "tau_cal_s11_out": (tau_cal if isinstance(tau_cal, float)
                            else MISSING),
        "exceedance_primary": "mad",
        "git_sha": git_sha,
        "state_hash_before": hash_before, "state_hash_after": hash_after,
        "state_restored": True,
        "fused_attn_flags": {str(k): v for k, v in fused_before.items()},
        "capture_mechanism": (
            "saga.attn_extract.capture_attention — the explicit post-softmax "
            "map is recomputed alongside the fused forward, which still "
            "produces the output; no fused_attn flag is toggled and none "
            "exists in saga/vit.py"),
    }
    arrays["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "incoming_mass.npz"
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)

    marker = dict(meta, n_records_appended=int(seen),
                  outputs={"incoming_mass.npz": int(path.stat().st_size)})
    (out_dir / "incoming_mass.done.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n")
    return dict(meta, out_dir=str(out_dir), path=str(path),
                bytes=int(path.stat().st_size))


def is_done(out_dir, ckpt_sha256: str) -> bool:
    """True iff this run completed for this exact checkpoint (resubmit-safe)."""
    marker = Path(out_dir) / "incoming_mass.done.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(
            encoding="utf-8")).get("ckpt_sha256") == ckpt_sha256
    except (OSError, ValueError):
        return False
