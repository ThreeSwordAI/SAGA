#!/usr/bin/env python3
"""
tools/frozen_smoke.py
=====================
TASK I0 D5 — the nine correctness checks from plan §13.3, on a fixed
32-image subset, before any expensive inference runs.

    python tools/frozen_smoke.py --run-id e2r_vits_mixup_baseline_s1 \
        --data $STAGE_DIR

Exit 0 when every applicable check PASSes, 3 otherwise (the project's gate
convention, as tools/ttr_validate.py uses). Writes
`results/frozen/I0_manifest/smoke_<run_id>.json` with the MEASURED quantity
beside every verdict — a PASS with no number is not evidence.

The checks (plan §13.3, TASK I0 §7):

 1 native_matches_recorded   the native loader reproduces the recorded eval
                             logits/accuracy on the shared subset within a
                             documented tolerance
 2 identity_is_bit_exact     an identity edit reproduces the native output
                             bit for bit
 3 terminal_gate_invariance  the terminal patch-gate override leaves the CLS
                             logits unchanged (Proposition 2; SAGA only)
 4 prefix_rows_untouched     every patch edit leaves the prefix rows exactly
                             as they were — checked on the REGISTER model in
                             particular, where there are five of them
 5 permutation_preserves     gate permutations preserve per-head means and
                             the value histogram
 6 ring_permutation_counts   within-ring permutations preserve ring counts
 7 energy_matched_norm       energy-matched edits achieve the declared
                             injected norm within tolerance
 8 stage_shapes              every stage hook returns [B, N - n_prefix, d]
 9 state_restored            the parameter/buffer hash is restored after
                             every edit

A check that cannot apply to this checkpoint (a gate check on a baseline,
say) is SKIP with the reason — never a silent PASS.

No training, no optimizer, no probe: this file imports none.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen.edits import (DIHEDRAL_OPS, EditError, gate_edit,  # noqa: E402
                               gate_map, receiver_perturbation, ring_of,
                               state_hash, terminal_gate_override)
from saga.frozen.runner import (build_from_row, image_id_for,  # noqa: E402
                                load_manifest_row, verify_checkpoint)
from saga.frozen.stages import (HIST_STAGE, STAGES, forward_with_stages,  # noqa: E402
                                num_prefix_tokens)
from saga.metrics import infer_num_prefix_tokens  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"
N_SMOKE = 32

#: fp32 tolerances, stated rather than tuned after looking at the failures.
#: TOL_LOGIT_INVARIANCE is the scale at which two fp32 forward passes that
#: differ only in a multiplication the CLS row never sees may still disagree,
#: on logits of order 10.
TOL_LOGIT_INVARIANCE = 1e-3
#: check 1 compares a 32-image mean NLL against a 50k-image recorded loss, so
#: it is a SANITY band on the loader, not an equality (the subsets differ).
TOL_TOP1_VS_RECORDED = 25.0
#: relative tolerance on the achieved injected Frobenius norm
TOL_ENERGY_REL = 1e-4
#: A permutation preserves a head's gate values EXACTLY (the multiset check
#: below has no tolerance), but the per-head MEAN is a sum over 196 fp32
#: values in a different order, and fp32 addition is not associative. One ULP
#: at magnitude 1 is 2^-24 = 5.96e-08; 8 ULP leaves room for the 196-term
#: reduction without admitting a real change. Measured on the Phase-B
#: ViT-S/mixup SAGA checkpoint: 5.96e-08, i.e. exactly one ULP.
TOL_PERM_MEAN = 8 * 2 ** -24


def result(name, status, measured, detail=""):
    return {"check": name, "status": status, "measured": measured,
            "detail": detail}


def atomic_write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def smoke_items(split_json, n=N_SMOKE):
    """The FIXED 32-image subset: the first n items of the frozen split, in
    the split's own order. Fixed by construction — no seed, no draw, so two
    runs on two checkpoints see exactly the same images."""
    doc = json.loads(Path(split_json).read_text(encoding="utf-8"))
    return doc["items"][:n], doc.get("sha256", MISSING), doc.get("name",
                                                                 MISSING)


def load_batch(data_root, items, device="cpu"):
    from tools.eval import build_val_transform
    from PIL import Image
    tf = build_val_transform(224)
    xs, ys = [], []
    for rel, label in items:
        img = Image.open(Path(data_root) / rel).convert("RGB")
        xs.append(tf(img))
        ys.append(int(label))
    return (torch.stack(xs).to(device),
            torch.tensor(ys, dtype=torch.long, device=device))


# ── the nine checks ─────────────────────────────────────────────────────────

def check_native(model, images, targets, row):
    """1. The native loader reproduces the recorded evaluation."""
    import torch.nn.functional as F
    with torch.no_grad():
        logits = model(images).float()
    top1 = 100.0 * (logits.argmax(1) == targets).float().mean().item()
    nll = F.cross_entropy(logits, targets).item()
    recorded = row.get("recorded_top1", MISSING)
    if recorded in (MISSING, "", None):
        return result("native_matches_recorded", "SKIP",
                      {"subset_top1": top1, "subset_nll": nll},
                      "no recorded top-1 in the manifest row for this "
                      "checkpoint; the loader ran and produced finite logits")
    delta = abs(top1 - float(recorded))
    ok = delta <= TOL_TOP1_VS_RECORDED and np.isfinite(nll)
    return result(
        "native_matches_recorded", "PASS" if ok else "FAIL",
        {"subset_top1": top1, "recorded_full_val_top1": float(recorded),
         "abs_diff": delta, "subset_nll": nll},
        f"32-image subset top-1 vs the recorded 50k top-1; band "
        f"{TOL_TOP1_VS_RECORDED} pts — a LOADER sanity check, not an "
        f"equality (the two image sets differ)")


def check_identity(model, images):
    """2. An identity edit reproduces the native output bit for bit.

    `terminal_gate_override(model, 1.0)` is the declared identity on a SAGA
    gate only if that gate is already 1.0, which it is not — so the identity
    edit tested here is `gate_edit(..., mode='original')`, which rebuilds the
    gate's own map and applies it through the replacement module. Bit
    equality therefore also proves the replacement path itself is exact.
    """
    with torch.no_grad():
        native = model(images).float()
    try:
        with gate_edit(model, len(model.blocks) - 1, "original"):
            with torch.no_grad():
                edited = model(images).float()
    except EditError as exc:
        return result("identity_is_bit_exact", "SKIP", MISSING, str(exc))
    diff = (edited - native).abs().max().item()
    n_exact = int((edited == native).all(dim=1).sum())
    return result(
        "identity_is_bit_exact", "PASS" if diff == 0.0 else "FAIL",
        {"max_abs_logit_diff": diff, "n_images_bit_exact": n_exact,
         "n_images": int(native.shape[0])},
        "gate_edit(mode='original') rebuilds the gate's own map and applies "
        "it through the replacement module; anything but 0.0 means the "
        "replacement path is not the identity it claims to be")


def check_terminal_gate(model, images):
    """3. The terminal patch gate preserves CLS-only logits (Proposition 2)."""
    with torch.no_grad():
        native = model(images).float()
    measured = {}
    try:
        for value in (0.25, 0.5, 0.75, 1.0):
            with terminal_gate_override(model, value):
                with torch.no_grad():
                    edited = model(images).float()
            measured[str(value)] = (edited - native).abs().max().item()
    except EditError as exc:
        return result("terminal_gate_invariance", "SKIP", MISSING, str(exc))
    worst = max(measured.values())
    return result(
        "terminal_gate_invariance", "PASS" if worst <= TOL_LOGIT_INVARIANCE
        else "FAIL",
        {"max_abs_logit_diff_by_value": measured, "worst": worst,
         "tolerance": TOL_LOGIT_INVARIANCE},
        "plan §5.6 Proposition 2: a CLS-only readout cannot see the terminal "
        "PATCH gate. A failure is a readout/hook/implementation problem and "
        "is investigated as one before it is reported as a finding.")


def check_prefix_untouched(model, images):
    """4. Prefix rows are untouched by every patch edit.

    Run on the REGISTER model in particular: it has five prefix rows, and a
    patch intervention that slid one row would be invisible on a model with
    only CLS.

    The comparison is made on the edited BLOCK's output, not on the attention
    module's output. That is the stronger place to look and it does not
    depend on forward-hook ordering: row p of the attention output only feeds
    token p through the residual and the tokenwise MLP, so if the patch edit
    is confined to patch rows then the block's prefix rows must come out bit
    identical. (The NEXT block's prefix rows legitimately change — CLS
    attends to the perturbed patches — which is the intervention working, not
    a leak.)
    """
    n_prefix = num_prefix_tokens(model)
    captured = []
    layer = max(0, len(model.blocks) - 2)

    def grab(_m, _i, out):
        tokens = out[0] if isinstance(out, (tuple, list)) else out
        captured.append(tokens[:, :n_prefix, :].detach().float().clone())

    handle = model.blocks[layer].register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(images)
        native_prefix = captured[-1]
        mask = np.array([0, 1, 2, 7, 13, 42, 100, 195])
        with receiver_perturbation(model, layer, mask, 0.25) as rec:
            with torch.no_grad():
                model(images)
        edited_prefix = captured[-1]
    finally:
        handle.remove()

    diff = (edited_prefix - native_prefix).abs().max().item()
    injected = float(np.max(rec["measured_perturbation_norm"]))
    ok = diff == 0.0 and injected > 0.0
    return result(
        "prefix_rows_untouched", "PASS" if ok else "FAIL",
        {"n_prefix": n_prefix, "max_abs_prefix_diff": diff,
         "max_injected_norm": injected, "layer": layer,
         "n_masked_coords": int(mask.size)},
        f"receiver_perturbation at epsilon=0.25 on {mask.size} patch "
        f"coordinates; the {n_prefix} prefix row(s) of block {layer}'s output "
        f"must be bit-identical. The injected norm is reported alongside so a "
        f"PASS cannot come from an edit that did nothing.")


def check_permutation_preserves(model):
    """5. Gate permutations preserve per-head means and value histograms."""
    layer = max(0, len(model.blocks) - 2)
    try:
        gate = model.blocks[layer].attn.gate
        if gate is None:
            raise EditError("no gate on this model")
        n_patches = int(getattr(gate, "n_patches", gate.phi.shape[1]))
        before = gate_map(gate, n_patches)
    except (AttributeError, EditError) as exc:
        return result("permutation_preserves", "SKIP", MISSING, str(exc))

    perm = np.random.RandomState(0).permutation(n_patches)
    with gate_edit(model, layer, "permute", perm=perm) as info:
        after = model.blocks[layer].attn.gate._frozen_gate_map
    d_mean = float(np.abs(np.asarray(info["gate_per_head_mean_before"])
                          - np.asarray(info["gate_per_head_mean_after"])).max())
    hist_same = bool(torch.equal(before.sort(dim=1).values,
                                 after.sort(dim=1).values))
    ok = d_mean <= TOL_PERM_MEAN and hist_same
    return result(
        "permutation_preserves", "PASS" if ok else "FAIL",
        {"max_abs_per_head_mean_diff": d_mean,
         "mean_tolerance": TOL_PERM_MEAN,
         "per_head_sorted_values_identical": hist_same,
         "layer": layer, "n_patches": n_patches},
        "a position permutation is a relabelling. The per-head MULTISET of "
        "gate values is checked for BIT equality — that is the invariant, and "
        "it has no tolerance. The per-head MEAN is a reduction over the "
        f"permuted vector, so fp32 non-associativity moves it by up to a few "
        f"ULP ({TOL_PERM_MEAN:g} at gate magnitude ~1); demanding bit "
        f"equality of the mean would be demanding that float addition "
        f"commute, which it does not.")


def check_ring_permutation(model):
    """6. Within-ring permutations preserve ring counts."""
    layer = max(0, len(model.blocks) - 2)
    try:
        gate = model.blocks[layer].attn.gate
        if gate is None:
            raise EditError("no gate on this model")
        n_patches = int(getattr(gate, "n_patches", gate.phi.shape[1]))
        rings = ring_of(n_patches)
    except (AttributeError, EditError) as exc:
        return result("ring_permutation_counts", "SKIP", MISSING, str(exc))

    rng = np.random.RandomState(0)
    perm = np.arange(n_patches)
    for r in np.unique(rings):
        idx = np.flatnonzero(rings == r)
        perm[idx] = idx[rng.permutation(idx.size)]
    counts_before = {int(r): int((rings == r).sum()) for r in np.unique(rings)}
    counts_after = {int(r): int((rings[perm] == r).sum())
                    for r in np.unique(rings)}
    moved = int(np.count_nonzero(rings[perm] != rings))

    with gate_edit(model, layer, "permute_within_ring", perm=perm):
        pass
    # and the guard must REFUSE a permutation that does cross rings
    cross = perm.copy()
    a = int(np.flatnonzero(rings == 0)[0])
    b = int(np.flatnonzero(rings == rings.max())[0])
    cross[[a, b]] = cross[[b, a]]
    refused = False
    try:
        with gate_edit(model, layer, "permute_within_ring", perm=cross):
            pass
    except EditError:
        refused = True

    ok = counts_before == counts_after and moved == 0 and refused
    return result(
        "ring_permutation_counts", "PASS" if ok else "FAIL",
        {"ring_counts_before": counts_before, "ring_counts_after": counts_after,
         "n_coords_changing_ring": moved,
         "cross_ring_permutation_refused": refused, "n_patches": n_patches},
        "rings from analysis.address_analysis.border_distance_map (THE ring "
        "definition, TASK-07/TASK-13); a within-ring permutation must move no "
        "coordinate between rings, and one that does must be refused")


def check_energy_matched(model, images):
    """7. Energy-matched edits achieve the declared injected norm."""
    layer = max(0, len(model.blocks) - 2)
    mask = np.array([0, 1, 2, 7, 13, 42, 100, 195])
    with receiver_perturbation(model, layer, mask, 0.10) as native_rec:
        with torch.no_grad():
            model(images)
    native_norms = np.asarray(native_rec["native_masked_norm"])
    if native_norms.size == 0:
        return result("energy_matched_norm", "FAIL", MISSING,
                      "the hook never fired")
    # kappa_i = epsilon * ||M .* U_i||_F for the single mask here (plan §5.5
    # takes the min over the candidate family; with one candidate that IS it)
    kappa = 0.10 * native_norms
    with receiver_perturbation(model, layer, mask, 0.10,
                               energy_target=kappa.tolist()) as rec:
        with torch.no_grad():
            model(images)
    achieved = np.asarray(rec["measured_perturbation_norm"])
    denom = np.where(kappa > 0, kappa, 1.0)
    rel = float(np.abs(achieved - kappa).max() / denom.max()) if kappa.size \
        else float("nan")
    ok = bool(rel <= TOL_ENERGY_REL) and rec["n_zero_norm_images"] == 0
    return result(
        "energy_matched_norm", "PASS" if ok else "FAIL",
        {"max_abs_rel_error": rel, "tolerance": TOL_ENERGY_REL,
         "kappa_mean": float(kappa.mean()),
         "achieved_mean": float(achieved.mean()),
         "n_zero_norm_images": int(rec["n_zero_norm_images"]),
         "layer": layer},
        "plan §5.5: alpha_iM = kappa_i / ||M .* U_i||_F, so every candidate "
        "mask injects the same Frobenius energy for that image. The achieved "
        "norm is measured from the tensor actually subtracted.")


def check_stage_shapes(model, images):
    """8. Every stage returns [B, N - n_prefix, d]."""
    n_prefix = infer_num_prefix_tokens(model)
    logits, store = forward_with_stages(model, images, STAGES)
    tokens = {}

    def grab(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        tokens["n"] = int(t.shape[1])

    handle = model.blocks[-1].register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()

    n_tokens = tokens["n"]
    expect = n_tokens - n_prefix
    shapes = {s: list(store[s].shape) for s in STAGES}
    ok = all(v[0] == images.shape[0] and v[1] == expect for v in shapes.values())
    ok = ok and store["hist"].shape == store[HIST_STAGE].shape
    return result(
        "stage_shapes", "PASS" if ok else "FAIL",
        {"n_tokens": n_tokens, "n_prefix": n_prefix,
         "expected_n_patches": expect, "shapes": shapes,
         "hist_alias_of": HIST_STAGE, "n_logits": int(logits.shape[1])},
        f"N_patches = N - n_prefix = {n_tokens} - {n_prefix} = {expect} at "
        f"every stage, including on a register model where n_prefix is 5")


def check_state_restored(model, images):
    """9. The state hash is restored after every edit."""
    before = state_hash(model)
    seen = {}
    layer = max(0, len(model.blocks) - 2)
    n_patches = 196

    def run(label, cm):
        try:
            with cm:
                with torch.no_grad():
                    model(images)
        except EditError as exc:
            seen[label] = f"SKIP: {exc}"
            return
        seen[label] = state_hash(model) == before

    run("receiver_perturbation",
        receiver_perturbation(model, layer, np.array([0, 5, 195]), 0.1))
    try:
        gate = model.blocks[layer].attn.gate
        n_patches = int(getattr(gate, "n_patches", gate.phi.shape[1]))
    except AttributeError:
        gate = None
    if gate is not None:
        run("gate_edit_mean", gate_edit(model, layer, "mean"))
        run("gate_edit_dihedral", gate_edit(model, layer, "dihedral", perm=5))
        run("gate_edit_mean_plus_alpha_delta",
            gate_edit(model, layer, "mean_plus_alpha_delta", alpha=0.5))
        run("gate_edit_permute",
            gate_edit(model, layer, "permute",
                      perm=np.random.RandomState(1).permutation(n_patches)))
    if getattr(model.blocks[-1], "attn", None) is not None and \
            getattr(model.blocks[-1].attn, "gate", None) is not None:
        run("terminal_gate_override", terminal_gate_override(model, 0.5))

    after = state_hash(model)
    restored = [v for v in seen.values() if v is True]
    failed = {k: v for k, v in seen.items() if v is False}
    ok = not failed and after == before
    return result(
        "state_restored", "PASS" if ok else "FAIL",
        {"hash_before": before, "hash_after": after,
         "per_edit": {k: (v if isinstance(v, str) else bool(v))
                      for k, v in seen.items()},
         "n_edits_checked": len(restored)},
        "sha256 over every parameter and buffer (name, dtype, shape, bytes); "
        "an edit that did not restore is a hard failure at the edit, not a "
        "wrong number three work packages later")


# ── driver ──────────────────────────────────────────────────────────────────

def run_checks(model, images, targets, row):
    checks = [
        check_native(model, images, targets, row),
        check_identity(model, images),
        check_terminal_gate(model, images),
        check_prefix_untouched(model, images),
        check_permutation_preserves(model),
        check_ring_permutation(model),
        check_energy_matched(model, images),
        check_stage_shapes(model, images),
        check_state_restored(model, images),
    ]
    return checks


def main():
    p = argparse.ArgumentParser(
        description="TASK I0 D5 — nine frozen-framework correctness checks.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ckpt-kind", default="last")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--data", required=True, metavar="ROOT",
                   help="ImageNet root containing val/")
    p.add_argument("--split",
                   default="results/frozen/splits/evaluation.json",
                   help="the frozen split the 32 smoke images come from")
    p.add_argument("--ckpt", default=None,
                   help="override the manifest's ckpt_path (same sha still "
                        "required)")
    p.add_argument("--out-dir", default="results/frozen/I0_manifest")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or not args.device.startswith("cuda") else "cpu")

    row = load_manifest_row(args.manifest, args.run_id, args.ckpt_kind)
    sha = verify_checkpoint(row, args.ckpt)
    items, split_sha, split_name = smoke_items(args.split)
    if len(items) < N_SMOKE:
        sys.exit(f"{args.split} has only {len(items)} items, need {N_SMOKE}")

    print(f"frozen_smoke: {args.run_id}/{args.ckpt_kind}  "
          f"{row['arch']}/{row['variant']}  ckpt_sha256={sha[:16]}…")
    print(f"              split={split_name} sha={str(split_sha)[:16]}… "
          f"first {N_SMOKE} images, device={device}")

    model = build_from_row(row, ckpt_path=args.ckpt, device=device)
    images, targets = load_batch(args.data, items, device=device)
    checks = run_checks(model, images, targets, row)

    n_pass = sum(c["status"] == "PASS" for c in checks)
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_skip = sum(c["status"] == "SKIP" for c in checks)

    doc = {
        "run_id": args.run_id, "ckpt_kind": args.ckpt_kind,
        "ckpt_sha256": sha, "arch": row["arch"], "variant": row["variant"],
        "recipe_actual": row["recipe_actual"],
        "gate_mode": row.get("gate_mode", MISSING),
        "n_prefix": int(infer_num_prefix_tokens(model)),
        "hist_stage": HIST_STAGE,
        "split_name": split_name, "split_sha256": split_sha,
        "n_images": N_SMOKE,
        "image_ids": [image_id_for(rel) for rel, _ in items],
        "device": str(device), "seed": args.seed,
        "precision": "fp32",
        "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "checks": checks,
    }
    out = Path(args.out_dir) / f"smoke_{args.run_id}.json"
    atomic_write_json(doc, out)

    print()
    for c in checks:
        print(f"  {c['status']:4s}  {c['check']}")
        if c["status"] == "FAIL":
            print(f"        measured: {c['measured']}")
    print(f"\n{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP -> {out}")
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
