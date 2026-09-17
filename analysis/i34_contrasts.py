#!/usr/bin/env python3
"""
analysis/i34_contrasts.py
=========================
TASK B B4 — the three primary contrasts of Track B, as CODE.

D7 named them in advance and `docs/LOCKED_ANALYSIS.md` fixes the rule that
decides them. This module is where both live, so that Phase C copies an
OUTPUT into the notes instead of reading a table and choosing a sentence:

    C1   I3: `permute` - `original`, per-image NLL          (LOCKED §9)
    C2   I3: `permute` - `permute_within_ring`               (LOCKED §9)
    C3   I4: theta(primary mask) - mean theta(10 matched controls), eps = 0.10

    S1   I3: `mean` - `original`             pre-declared SECONDARY
    S2   I3: `permute` - `dihedral` (t != 0) pre-declared SECONDARY

C1 and C2 were the S1/S2 pair through I3 Phase A, following the D7 table of
`docs/TASK_B_I3_I4.md` §0. On 2026-09-17 the human resolved that table's
disagreement with LOCKED §9 in favour of the signed document, and §4's
interpretation guide was amended to match; the superseded guide is kept,
struck through, in the task file.

The decision rule, applied per contrast (LOCKED_ANALYSIS §10, TASK B §0):

  1. the direction is consistent across the FRESH ViT-S/mixup checkpoints
     (`e2r_vits_mixup_saga_s1` and `_s2`);
  2. the image-level bootstrap CI (10,000 resamples, seed 0) excludes zero in
     each of them;
  3. the effect size is stated in nats and in top-1 points.

Legacy checkpoints and the exploratory cells are computed and reported and
DO NOT VOTE. A contrast that fails is reported as a null with its CI;
nothing is re-framed.

WHY THIS IS A MODULE AND NOT A PARAGRAPH IN A NOTEBOOK
------------------------------------------------------
I3 and I4 are the two experiments in this project that could be tuned into a
result. The interpretation branches of TASK B §4 and §6 were written before
any evaluation number existed, and §4's amendment of 2026-09-17 carries a
RECORDED DEVIATION saying exactly what had been computed when it was written;
`interpret_i3` returns one of them VERBATIM,
and `tests/test_I3_gate_edits.py` asserts the returned string appears
byte-for-byte in `docs/TASK_B_I3_I4.md`. A branch cannot be softened without
editing the task file, and editing the task file is visible in the diff.

No training, no optimizer, no probe fitting, and no discretion.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.build_I2_tables import (BOOTSTRAP_RESAMPLES,  # noqa: E402
                                      BOOTSTRAP_SEED, paired_bootstrap)

MISSING = "MISSING"

#: The two FRESH ViT-S/mixup SAGA checkpoints. They, and only they, decide
#: (LOCKED_ANALYSIS §10.1). The pair is spelled out rather than derived from
#: a `seed_controlled` flag so that a manifest change cannot quietly enlarge
#: the set of checkpoints that vote.
FRESH_SAGA = ("e2r_vits_mixup_saga_s1", "e2r_vits_mixup_saga_s2")

#: The condition-id families of the I3 sweep, as (family, prefix). `dihedral`
#: is listed with its identity member; `c2` drops it explicitly.
I3_FAMILIES = ("mean", "mean_half", "perm", "ringperm", "dihedral")

#: `dihedral0` is the identity transform and is bit-identical to `original`.
#: C2 contrasts the permutations against the SEVEN non-identity symmetries:
#: including the identity would put a known-zero delta into the comparison
#: side and shrink the contrast by 1/8 for a reason that has nothing to do
#: with arrangement.
DIHEDRAL_IDENTITY = 0

#: The verdicts a contrast can carry. `null` is a RESULT, not a failure.
VERDICTS = ("positive", "negative", "null", "insufficient")

# ─────────────────────────────────────────────────────────────────────────────
# The interpretation branches, VERBATIM from TASK B §4 and §6
# ─────────────────────────────────────────────────────────────────────────────

#: The closing sentence of the §4 guide, carried with every I3 interpretation.
I3_GUIDE_CLOSE = "Each is reportable; none is a failure."

#: The §4 guide as AMENDED on 2026-09-17 to the signed LOCKED §9 contrasts.
#: Four patterns, exhaustive over the verdicts `decide` can return for the
#: primary pair. The first three carry the task file's text VERBATIM (its
#: non-breaking hyphens included) and a test pins each against
#: `docs/TASK_B_I3_I4.md`; the fourth is the anomaly slot.
#:
#: The SUPERSEDED guide is still in the task file, struck through, because
#: it was the pre-registered text and the paper trail matters more than a
#: clean page. It is deliberately NOT reproduced here: this module returns
#: the guide in force, and holding both would let a caller pick.
I3_BRANCHES = {
    "no_arrangement": "the frozen model does not use its gate arrangement "
                      "beyond the per‑head mean",
    "ring_profile": "it uses the ring profile, and within‑ring permutation "
                    "hurts less",
    "within_ring": "it uses the arrangement within rings, and ring "
                   "membership adds nothing beyond it",
    "anomaly": "anomaly. Both intervals are reported and no mechanism "
               "statement is made.",
}

#: What C2 means when C1 is null: nothing. With no effect to decompose, the
#: split between ring profile and within-ring arrangement is not
#: interpretable, so the branch says so rather than reading a null C2 as
#: evidence of anything.
I3_C2_NOT_READ = ("C2 is reported without a reading: with no effect to "
                  f"decompose, the split between ring profile and within‑ring "
                  "arrangement is not interpretable.")

#: A deciding checkpoint is absent. Not one of §4's four patterns — it is a
#: statement about the RECORDS, not about the model — so it gets its own
#: slot rather than being folded into `anomaly`.
I3_INSUFFICIENT = ("a deciding checkpoint is absent from these records; both "
                   "intervals are reported and no mechanism statement is "
                   "made.")


#: TASK B §6, the I4 interpretation guide. Defined here with I3's so that both
#: branch sets live in one place; `interpret_i4` is completed in I4 Phase A'.
I4_BRANCHES = {
    "inside_band": "high‑prevalence coordinates are not special beyond "
                   "their ring composition and injected energy",
    "method_specific": "the methods differ in how those coordinates are used, "
                       "which is the cross‑method row",
    "all_methods": "coordinate specificity is a property of the data "
                   "position, not of the intervention",
}

#: The closing sentence of the §6 guide. I4's guide closes with "Each is
#: reportable." and I3's with "Each is reportable; none is a failure." —
#: they are different sentences and each is quoted from its own section.
I4_GUIDE_CLOSE = "Each is reportable."


class ContrastError(ValueError):
    """A contrast that cannot be computed from the records as given."""


# ─────────────────────────────────────────────────────────────────────────────
# Per-image deltas out of the record rows
# ─────────────────────────────────────────────────────────────────────────────

def by_condition(records) -> dict:
    """`{condition_id: {image_id: row}}` for one checkpoint's record rows.

    A duplicate (condition, image) is refused rather than silently kept once:
    `records.append_rows` is idempotent on that key, so a duplicate here means
    two files were concatenated and the mean would be weighted by whichever
    images appear twice.
    """
    out = {}
    for row in records:
        cond = out.setdefault(str(row["condition_id"]), {})
        img = str(row["image_id"])
        if img in cond:
            raise ContrastError(
                f"condition {row['condition_id']!r} has two rows for image "
                f"{img!r} — these records are not a single sweep")
        cond[img] = row
    return out


def paired_delta(grouped: dict, condition: str, reference: str,
                 field: str = "nll"):
    """`(image_ids, deltas)` for `condition - reference`, on common images.

    Sorted by image_id, so every contrast in this module resamples the same
    images in the same order under one bootstrap seed.
    """
    a, b = grouped.get(condition), grouped.get(reference)
    if not a or not b:
        missing = [n for n, g in ((condition, a), (reference, b)) if not g]
        raise ContrastError(
            f"no rows for condition(s) {missing} — a contrast is not computed "
            f"from a partial sweep")
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ContrastError(
            f"conditions {condition!r} and {reference!r} share no image")
    d = np.asarray([float(a[i][field]) - float(b[i][field]) for i in ids],
                   dtype=np.float64)
    return ids, d


def family_conditions(grouped: dict, family: str, layer=None,
                      exclude=()) -> list:
    """Every condition id of one I3 family, sorted, optionally for one layer.

    Ids are matched on the DECLARED naming of `configs/frozen/I3_gate_edits.yaml`
    (`<family><k>_L<layer>`, and `mean_L<layer>` / `mean_half_L<layer>` which
    carry no index). `mean` never matches `mean_half`, which is why the
    prefixes are compared against a split id rather than with `startswith`.
    """
    out = []
    for cond in sorted(grouped):
        parsed = parse_condition_id(cond)
        if parsed is None or parsed["family"] != family:
            continue
        if layer is not None and parsed["layer"] != int(layer):
            continue
        if cond in exclude or parsed["index"] in exclude:
            continue
        out.append(cond)
    return sorted(out, key=lambda c: (parse_condition_id(c)["index"], c))


def parse_condition_id(cond: str):
    """`{family, index, layer}` for an I3 condition id, or None.

    `original` and anything the I3 YAML does not declare return None.
    """
    if "_L" not in cond:
        return None
    head, _, tail = cond.rpartition("_L")
    if not tail.isdigit():
        return None
    layer = int(tail)
    for family in ("mean_half", "mean", "ringperm", "perm", "dihedral"):
        if head == family:
            return {"family": family, "index": 0, "layer": layer}
        if head.startswith(family) and head[len(family):].isdigit():
            return {"family": family, "index": int(head[len(family):]),
                    "layer": layer}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# One contrast on one checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _summary(deltas, top1_deltas=None, *, resamples=BOOTSTRAP_RESAMPLES,
             seed=BOOTSTRAP_SEED) -> dict:
    """mean, CI, direction and effect size for one per-image delta vector."""
    mean, lo, hi = paired_bootstrap([deltas], resamples=resamples, seed=seed)
    if mean is MISSING:
        return {"n_images": 0, "mean_nats": MISSING, "ci_lo": MISSING,
                "ci_hi": MISSING, "excludes_zero": False, "direction": MISSING,
                "delta_top1_points": MISSING, "frac_positive": MISSING,
                "bootstrap_resamples": resamples, "bootstrap_seed": seed}
    excludes = bool(lo > 0.0 or hi < 0.0)
    return {
        "n_images": int(np.asarray(deltas).size),
        "mean_nats": float(mean), "ci_lo": float(lo), "ci_hi": float(hi),
        "excludes_zero": excludes,
        "direction": "+" if mean > 0 else ("-" if mean < 0 else "0"),
        # LOCKED_ANALYSIS §10.3: the magnitude is reported, not only its
        # significance. Top-1 is a rate difference, so it is stated in POINTS.
        "delta_top1_points": (MISSING if top1_deltas is None else
                              float(np.asarray(top1_deltas,
                                               dtype=np.float64).mean() * 100)),
        "frac_positive": float(np.mean(np.asarray(deltas) > 0)),
        "bootstrap_resamples": resamples, "bootstrap_seed": seed,
    }


def _family_delta(grouped: dict, family: str, layer: int, ids, field,
                  exclude_index=None):
    """Mean over one family's draws, WITHIN IMAGE, as a [len(ids)] vector.

    LOCKED_ANALYSIS §8: permutation draws are repeated interventions on one
    checkpoint, never replication — so the ten draws are averaged inside
    each image before any contrast, and the resampling unit stays the image.
    """
    conds = [c for c in family_conditions(grouped, family, layer)
             if exclude_index is None
             or parse_condition_id(c)["index"] != exclude_index]
    if not conds:
        raise ContrastError(f"no {family!r} condition at layer {layer}")
    return conds, np.mean([[float(grouped[c][i][field]) for i in ids]
                           for c in conds], axis=0)


def _common_images(grouped: dict, conds) -> list:
    missing = [c for c in conds if c not in grouped]
    if missing:
        raise ContrastError(
            f"no rows for condition(s) {missing} — a contrast is not "
            f"computed from a partial sweep")
    ids = sorted(set.intersection(*[set(grouped[c]) for c in conds]))
    if not ids:
        raise ContrastError(f"conditions {list(conds)} share no image")
    return ids


def _expect(name, layer, got, want, what):
    if len(got) != want:
        raise ContrastError(
            f"{name} at layer {layer} needs {want} {what}; found {len(got)}. "
            f"The condition list is fixed in "
            f"configs/frozen/I3_gate_edits.yaml and a partial sweep is not a "
            f"contrast.")


def c1(grouped: dict, layer: int, *, reference="original",
       resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> dict:
    """C1 (PRIMARY, LOCKED §9.1) — the 10 `perm` conditions minus `original`.

    "`original` vs `permute`", as LOCKED_ANALYSIS §9 names it. A position
    permutation preserves each head's gate MULTISET exactly and destroys only
    the arrangement, so this asks whether the frozen model uses WHERE its
    gate values sit at all.

    Direction that would support "arrangement matters": Delta-NLL > 0.

    THIS DEFINITION CHANGED on 2026-09-17. Through I3 Phase A, C1 was
    `mean` - `original`, following the D7 table of `docs/TASK_B_I3_I4.md`
    §0, which disagreed with LOCKED §9. The human resolved it in favour of
    the signed document; the old pair is kept, pre-declared, as `s1`/`s2`.
    """
    layer = int(layer)
    perms = family_conditions(grouped, "perm", layer)
    _expect("C1", layer, perms, 10, "`perm` conditions")
    ids = _common_images(grouped, perms + [reference])
    _c, a = _family_delta(grouped, "perm", layer, ids, "nll")
    _c, atop = _family_delta(grouped, "perm", layer, ids, "correct")
    b = np.asarray([float(grouped[reference][i]["nll"]) for i in ids])
    btop = np.asarray([float(grouped[reference][i]["correct"]) for i in ids])
    out = _summary(a - b, atop - btop, resamples=resamples, seed=seed)
    out.update(contrast="C1", layer=layer, role="primary",
               condition="+".join(perms), reference=reference,
               n_perm=len(perms),
               definition="mean_i[ mean_k nll(perm k) - nll(original) ], "
                          "paired per image (LOCKED §9.1)")
    return out


def c2(grouped: dict, layer: int, *, resamples=BOOTSTRAP_RESAMPLES,
       seed=BOOTSTRAP_SEED) -> dict:
    """C2 (PRIMARY, LOCKED §9.1) — `perm` minus `permute_within_ring`.

    Both families destroy arrangement; only the unrestricted one crosses
    rings. So this isolates the RING composition from the arrangement inside
    a ring, on the same checkpoint and the same images.

    Direction: > 0 means an unrestricted permutation costs MORE than one that
    keeps every coordinate in its ring — i.e. ring membership carries
    something the within-ring arrangement does not.

    NOTE, and it matters for reading `interpret_i3`: the §4 interpretation
    guide of `docs/TASK_B_I3_I4.md` was written against the OTHER C2 (the one
    now called `s2`). Its middle branch does not transfer to this definition;
    `interpret_i3` flags that rather than silently re-labelling it.
    """
    layer = int(layer)
    perms = family_conditions(grouped, "perm", layer)
    rings = family_conditions(grouped, "ringperm", layer)
    _expect("C2", layer, perms, 10, "`perm` conditions")
    _expect("C2", layer, rings, 10, "`ringperm` conditions")
    ids = _common_images(grouped, perms + rings)
    _c, a = _family_delta(grouped, "perm", layer, ids, "nll")
    _c, b = _family_delta(grouped, "ringperm", layer, ids, "nll")
    _c, atop = _family_delta(grouped, "perm", layer, ids, "correct")
    _c, btop = _family_delta(grouped, "ringperm", layer, ids, "correct")
    out = _summary(a - b, atop - btop, resamples=resamples, seed=seed)
    out.update(contrast="C2", layer=layer, role="primary",
               condition="+".join(perms), reference="+".join(rings),
               n_perm=len(perms), n_ringperm=len(rings),
               definition="mean_i[ mean_k nll(perm k) - mean_k "
                          "nll(ringperm k) ], paired per image (LOCKED §9.1)")
    return out


def s1(grouped: dict, layer: int, *, reference="original",
       resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> dict:
    """S1 (SECONDARY, pre-declared) — `mean_L{layer}` minus `original`.

    The whole arrangement removed and the per-head mean kept. This was C1
    through I3 Phase A; it is now reported beside the primaries under the
    same decision rule and labelled secondary in every table, never as the
    headline.
    """
    layer = int(layer)
    cond = f"mean_L{layer}"
    _ids, d = paired_delta(grouped, cond, reference, "nll")
    _ids2, dtop = paired_delta(grouped, cond, reference, "correct")
    out = _summary(d, dtop, resamples=resamples, seed=seed)
    out.update(contrast="S1", layer=layer, role="secondary", condition=cond,
               reference=reference,
               definition="mean(nll[mean_L{l}]) - mean(nll[original]), "
                          "paired per image (secondary)")
    return out


def s2(grouped: dict, layer: int, *, resamples=BOOTSTRAP_RESAMPLES,
       seed=BOOTSTRAP_SEED) -> dict:
    """S2 (SECONDARY, pre-declared) — `perm` minus the 7 non-identity
    `dihedral` transforms.

    `dihedral0` is excluded because it is bit-identical to `original`, so its
    delta is exactly 0 and including it would shrink the contrast by 1/8 for
    a reason unrelated to arrangement. This was C2 through I3 Phase A.
    """
    layer = int(layer)
    perms = family_conditions(grouped, "perm", layer)
    dihs = [c for c in family_conditions(grouped, "dihedral", layer)
            if parse_condition_id(c)["index"] != DIHEDRAL_IDENTITY]
    _expect("S2", layer, perms, 10, "`perm` conditions")
    _expect("S2", layer, dihs, 7, "non-identity `dihedral` conditions")
    ids = _common_images(grouped, perms + dihs)
    _c, a = _family_delta(grouped, "perm", layer, ids, "nll")
    _c, atop = _family_delta(grouped, "perm", layer, ids, "correct")
    b = np.mean([[float(grouped[c][i]["nll"]) for i in ids] for c in dihs],
                axis=0)
    btop = np.mean([[float(grouped[c][i]["correct"]) for i in ids]
                    for c in dihs], axis=0)
    out = _summary(a - b, atop - btop, resamples=resamples, seed=seed)
    out.update(contrast="S2", layer=layer, role="secondary",
               condition="+".join(perms), reference="+".join(dihs),
               n_perm=len(perms), n_dihedral=len(dihs),
               definition="mean_i[ mean_k nll(perm k) - mean_t nll(dihedral "
                          "t, t != 0) ], paired per image (secondary)")
    return out


#: I4 condition ids (TASK B §5). `native` is the reference of every theta.
I4_REFERENCE = "native"
I4_PRIMARY = "prim_e{eps}_L{layer}"
I4_CONTROL = "ctrl{j}_e{eps}{kind}_L{layer}"

#: The energy regimes a control can be in: `m` = per-image energy-matched to
#: the primary, `f` = fixed epsilon and UNMATCHED. Both are reported; the
#: difference between them is the size of the energy confound.
I4_MATCHED, I4_FIXED = "m", "f"


def i4_condition_ids(layer: int, eps: str = "10", kind: str = I4_MATCHED,
                     n_controls: int = 10) -> tuple:
    """`(primary id, [control ids])` for one layer and one epsilon."""
    primary = I4_PRIMARY.format(eps=eps, layer=int(layer))
    controls = [I4_CONTROL.format(j=j, eps=eps, kind=kind, layer=int(layer))
                for j in range(int(n_controls))]
    return primary, controls


def theta(grouped: dict, condition: str, *, reference=I4_REFERENCE,
          resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> dict:
    """theta for ONE I4 condition: mean paired Delta-NLL vs `native`.

    LOCKED §7: the primary endpoint is the paired per-image NLL difference.
    A positive theta means the perturbation at those coordinates worsens the
    loss.
    """
    _ids, d = paired_delta(grouped, condition, reference, "nll")
    _ids2, dtop = paired_delta(grouped, condition, reference, "correct")
    out = _summary(d, dtop, resamples=resamples, seed=seed)
    out.update(condition=condition, reference=reference)
    return out


def c3(grouped: dict, layer: int, *, eps: str = "10", kind: str = I4_MATCHED,
       n_controls: int = 10, resamples=BOOTSTRAP_RESAMPLES,
       seed=BOOTSTRAP_SEED) -> dict:
    """C3 — theta(primary) minus the mean theta of the matched controls.

    LOCKED §7: theta_m = E_i[ Delta_i^high - (1/R) sum_r Delta_i^control(r) ].
    The R = 10 controls are averaged WITHIN IMAGE before the contrast
    (LOCKED §8: repeated interventions on one checkpoint are never
    replication), so the resampling unit stays the image.

    Also returns the CONTROL BAND — the min and max of the 10 individual
    control thetas — because TASK B §6's interpretation turns on whether the
    primary sits inside it, not on the contrast alone.
    """
    layer = int(layer)
    primary, controls = i4_condition_ids(layer, eps, kind, n_controls)
    missing = [c for c in [primary] + controls if c not in grouped]
    if missing:
        raise ContrastError(
            f"C3 at layer {layer} (eps {eps}, kind {kind!r}) needs "
            f"{primary!r} and {n_controls} controls; missing {missing}. The "
            f"condition list is fixed in configs/frozen/I4_perturbation.yaml "
            f"and a partial sweep is not a contrast.")
    ids = sorted(set.intersection(
        *[set(grouped[c]) for c in [primary, I4_REFERENCE] + controls]))
    if not ids:
        raise ContrastError(
            f"C3 at layer {layer}: the conditions share no image")

    def _delta(cond, field):
        return np.asarray([float(grouped[cond][i][field])
                           - float(grouped[I4_REFERENCE][i][field])
                           for i in ids], dtype=np.float64)

    d_primary = _delta(primary, "nll")
    per_control = [_delta(c, "nll") for c in controls]
    d = d_primary - np.mean(per_control, axis=0)
    dtop = (_delta(primary, "correct")
            - np.mean([_delta(c, "correct") for c in controls], axis=0))

    out = _summary(d, dtop, resamples=resamples, seed=seed)
    band = [float(np.mean(x)) for x in per_control]
    theta_primary = float(np.mean(d_primary))
    out.update(
        contrast="C3", layer=layer, epsilon=eps, control_kind=kind,
        condition=primary, reference="+".join(controls),
        n_controls=len(controls),
        theta_primary=theta_primary,
        theta_control_mean=float(np.mean(band)),
        control_band_lo=min(band), control_band_hi=max(band),
        theta_control_per_mask={c: b for c, b in zip(controls, band)},
        # TASK B §6's first branch is "theta(primary) inside the control
        # band", which is a statement about the BAND and not about the CI —
        # so it is computed here, beside the interval, rather than inferred
        # from it later.
        primary_inside_band=bool(min(band) <= theta_primary <= max(band)),
        definition="mean_i[ (nll(primary) - nll(native)) - mean_r (nll(ctrl r)"
                   " - nll(native)) ], paired per image")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The decision rule
# ─────────────────────────────────────────────────────────────────────────────

def decide(per_checkpoint: dict, *, deciding=FRESH_SAGA) -> dict:
    """The LOCKED §10 verdict for ONE contrast over the deciding checkpoints.

    `per_checkpoint` is `{run_id: summary}` from `c1`/`c2`. Only `deciding`
    votes; everything else is carried through in `reported` and is visible in
    the tables, exactly as LOCKED §10 says ("Legacy and cross-recipe results
    are shown, never converted into extra training replication").
    """
    votes = {r: per_checkpoint[r] for r in deciding if r in per_checkpoint}
    missing = [r for r in deciding if r not in per_checkpoint]
    reported = sorted(set(per_checkpoint) - set(deciding))

    if missing:
        verdict, why = "insufficient", (
            f"the deciding checkpoints {list(missing)} are absent from these "
            f"records; the rule votes on {list(deciding)} and on nothing else")
    else:
        dirs = {v["direction"] for v in votes.values()}
        all_excl = all(v["excludes_zero"] for v in votes.values())
        consistent = len(dirs) == 1 and dirs <= {"+", "-"}
        if consistent and all_excl:
            verdict = "positive" if dirs == {"+"} else "negative"
            why = (f"direction {dirs.pop()} in both fresh checkpoints and "
                   f"both bootstrap CIs exclude zero")
        elif not consistent:
            verdict, why = "null", (
                f"the direction is not consistent across the fresh pair "
                f"({ {r: v['direction'] for r, v in votes.items()} })")
        else:
            verdict, why = "null", (
                f"at least one fresh checkpoint's CI includes zero "
                f"({ {r: (v['ci_lo'], v['ci_hi']) for r, v in votes.items()} })")

    return {
        "verdict": verdict, "reason": why,
        "deciding": list(deciding), "missing_deciding": missing,
        "reported_not_deciding": reported,
        "per_checkpoint": per_checkpoint,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def interpret_i3(c1_verdict: str, c2_verdict: str) -> dict:
    """The TASK B §4 branch a (C1, C2) pair lands in, with its VERBATIM text.

    Four patterns, exhaustive over the verdicts `decide` returns for the
    SIGNED pair (C1 = permute - original, C2 = permute - permute_within_ring):

        C1 ~ 0                 -> no_arrangement  (+ C2 carries no reading)
        C1 > 0 and C2 > 0      -> ring_profile
        C1 > 0 and C2 ~ 0      -> within_ring
        C1 < 0 or  C2 < 0      -> anomaly, intervals only

    plus `insufficient` when a deciding checkpoint is absent, which is a
    statement about the records rather than about the model.
    """
    for name, v in (("C1", c1_verdict), ("C2", c2_verdict)):
        if v not in VERDICTS:
            raise ContrastError(
                f"{name} verdict {v!r} is not one of {list(VERDICTS)}")
    # ORDER MATTERS. A negative contrast is flagged as an anomaly even when
    # C1 is null, where §4's first pattern would otherwise absorb it into
    # "C2 reported without a reading". The two are not in conflict — the
    # anomaly branch is strictly more informative and still makes no
    # mechanism statement — and flagging a sign nobody predicted is the
    # safer of the two readings.
    if "insufficient" in (c1_verdict, c2_verdict):
        branch, text = "insufficient", I3_INSUFFICIENT
    elif "negative" in (c1_verdict, c2_verdict):
        branch, text = "anomaly", I3_BRANCHES["anomaly"]
    elif c1_verdict == "null":
        branch, text = "no_arrangement", I3_BRANCHES["no_arrangement"]
    elif c2_verdict == "positive":
        branch, text = "ring_profile", I3_BRANCHES["ring_profile"]
    else:
        branch, text = "within_ring", I3_BRANCHES["within_ring"]

    return {
        "branch": branch, "text": text, "close": I3_GUIDE_CLOSE,
        "c1_verdict": c1_verdict, "c2_verdict": c2_verdict,
        # §4: when C1 is null, C2 carries no reading of its own.
        "c2_note": (I3_C2_NOT_READ if branch == "no_arrangement" else None),
        "anticipated": branch in I3_BRANCHES,
        "source": "docs/TASK_B_I3_I4.md §4 (amended 2026-09-17)",
    }


#: The method groups C3 decides SEPARATELY (TASK B §6, LOCKED §10.1).
#: Registers are n = 2 and never decide; they are labelled in every table.
FRESH_BASELINE = ("e2r_vits_mixup_baseline_s1", "e2r_vits_mixup_baseline_s2")
REGISTERS = ("legacy_e2_vit_small_mixupdir_registers",
             "legacy_e2_vit_small_nomixdir_registers")


def interpret_i4(per_method: dict) -> dict:
    """The TASK B §6 branch an I4 result lands in, with its VERBATIM text.

    `per_method` is `{"baseline": decision, "saga": decision}` from `decide`,
    each carrying the per-checkpoint C3 summaries. The guide turns on TWO
    things, and both are read from the data rather than inferred from the
    verdict alone:

      * whether theta(primary) sits INSIDE the control band, per checkpoint;
      * whether the sign is consistent in one method but not the other.

    Branches, from §6:
      inside the band for every method          -> `inside_band`
      outside, consistent in one method only    -> `method_specific`
      outside, same sign in all methods         -> `all_methods`

    Anything else — outside the band with inconsistent signs everywhere, or a
    method that could not be decided — is `unanticipated`, which says so
    rather than borrowing the nearest named sentence, exactly as
    `interpret_i3` does.
    """
    methods = sorted(per_method)
    if not methods:
        raise ContrastError("interpret_i4 needs at least one method")

    inside, verdicts, signs = {}, {}, {}
    for m, dec in per_method.items():
        votes = [dec["per_checkpoint"][r] for r in dec["deciding"]
                 if r in dec["per_checkpoint"]]
        inside[m] = bool(votes) and all(
            v.get("primary_inside_band") for v in votes)
        verdicts[m] = dec["verdict"]
        signs[m] = {v["direction"] for v in votes}

    decided = {m: v for m, v in verdicts.items()
               if v in ("positive", "negative")}
    if all(inside.values()):
        branch = "inside_band"
    elif not decided:
        branch = "unanticipated"
    elif len(decided) == len(methods) and len({tuple(sorted(signs[m]))
                                               for m in decided}) == 1:
        branch = "all_methods"
    elif len(decided) < len(methods):
        branch = "method_specific"
    else:
        branch = "unanticipated"

    text = I4_BRANCHES.get(branch)
    if text is None:
        text = ("the theta/control-band pattern is not one of the three "
                "branches TASK B §6 names; it is reported with the band, the "
                "intervals and no interpretation supplied")
    return {
        "branch": branch, "text": text, "close": I4_GUIDE_CLOSE,
        "anticipated": branch != "unanticipated",
        "verdicts": verdicts, "primary_inside_band": inside,
        "source": "docs/TASK_B_I3_I4.md §6",
    }


def i4_contrasts(records_by_run: dict, *, layers=(7, 8), eps="10",
                 kind=I4_MATCHED, resamples=BOOTSTRAP_RESAMPLES,
                 seed=BOOTSTRAP_SEED) -> dict:
    """C3 at each layer, per checkpoint, decided per METHOD.

    `records_by_run` is `{run_id: [record rows]}`. The fresh baselines and
    the fresh SAGA checkpoints are decided SEPARATELY (TASK B §6): the
    question is whether the methods differ in how those coordinates are used,
    and pooling them would answer a different one.
    """
    grouped = {run: by_condition(rows) for run, rows in records_by_run.items()}
    # The METHOD comes from the record rows' own `variant` column, which the
    # manifest wrote — never from a substring of the run_id. `..._saga_s1`
    # and `..._baseline_s1` happen to be greppable; a run named otherwise
    # would be silently dropped from its own method, and a null built from a
    # missing checkpoint is the one failure this module must not produce.
    variant = {}
    for run, rows in records_by_run.items():
        seen = {str(r.get("variant", MISSING)) for r in rows}
        if len(seen) != 1:
            raise ContrastError(
                f"{run}: record rows carry {sorted(seen)} in `variant`; one "
                f"checkpoint is one variant")
        variant[run] = seen.pop()

    out = {}
    for layer in layers:
        per_run = {run: c3(g, layer, eps=eps, kind=kind, resamples=resamples,
                           seed=seed)
                   for run, g in grouped.items()}
        per_method = {
            m: decide({r: s for r, s in per_run.items() if variant[r] == m},
                      deciding=d)
            for m, d in (("baseline", FRESH_BASELINE), ("saga", FRESH_SAGA))
        }
        out[int(layer)] = {
            "C3": per_method, "per_checkpoint": per_run,
            "variant": dict(variant),
            "registers": sorted(r for r, v in variant.items()
                                if v == "registers"),
            "interpretation": interpret_i4(per_method),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The whole I3 decision, from one checkpoint's records upward
# ─────────────────────────────────────────────────────────────────────────────

def i3_contrasts(records_by_run: dict, *, layers=(7, 8),
                 resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> dict:
    """C1 and C2 at each layer, per checkpoint, plus the verdict and branch.

    `records_by_run` is `{run_id: [record rows]}`. Returns
    `{layer: {"C1": decision, "C2": decision, "interpretation": {...}}}`.
    A checkpoint whose records cannot supply a contrast raises — a table
    built from "whatever was loadable" is not a pre-registered contrast.
    """
    grouped = {run: by_condition(rows) for run, rows in records_by_run.items()}
    out = {}
    for layer in layers:
        per = {name: {run: fn(g, layer, resamples=resamples, seed=seed)
                      for run, g in grouped.items()}
               for name, fn in (("C1", c1), ("C2", c2), ("S1", s1), ("S2", s2))}
        decided = {name: decide(v) for name, v in per.items()}
        out[int(layer)] = {
            # LOCKED §9's pair decides; the pre-declared secondaries are
            # computed under the SAME rule and reported beside them, never as
            # the headline (LOCKED §9: "everything else is labelled
            # exploratory and reported as such").
            "C1": decided["C1"], "C2": decided["C2"],
            "S1": decided["S1"], "S2": decided["S2"],
            "primary": ("C1", "C2"), "secondary": ("S1", "S2"),
            "interpretation": interpret_i3(decided["C1"]["verdict"],
                                           decided["C2"]["verdict"]),
        }
    return out
