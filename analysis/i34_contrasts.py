#!/usr/bin/env python3
"""
analysis/i34_contrasts.py
=========================
TASK B B4 — the three primary contrasts of Track B, as CODE.

D7 named them in advance and `docs/LOCKED_ANALYSIS.md` fixes the rule that
decides them. This module is where both live, so that Phase C copies an
OUTPUT into the notes instead of reading a table and choosing a sentence:

    C1   I3: `mean` collapse - `original`, per-image NLL
    C2   I3: mean over the 10 `perm` conditions
             - mean over the 7 NON-IDENTITY `dihedral` conditions
    C3   I4: theta(primary mask) - mean theta(10 matched controls), eps = 0.10

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
any evaluation number existed; `interpret_i3` returns one of them VERBATIM,
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

#: TASK B §4, the I3 interpretation guide, fixed before any result existed.
#: These four strings are quoted out of the task file; the first three are the
#: branches it names and a test pins each one byte-for-byte against
#: `docs/TASK_B_I3_I4.md`. The fourth is NOT in the task file, and says so: a
#: pattern the guide did not anticipate is reported as unanticipated rather
#: than bent into the nearest named branch.
#: NOTE: the task file writes "per‑head", "within‑ring",
#: "high‑prevalence" and "cross‑method" with U+2011 NON-BREAKING
#: HYPHENS, and these literals carry the same characters. A plain "-" here
#: would make the byte-identity test fail, which is the test doing its job.
I3_BRANCHES = {
    "neither": "the frozen model does not use its gate arrangement beyond "
               "the per‑head mean",
    "ring_only": "it uses the ring/symmetry structure but not the "
                 "within‑ring arrangement",
    "arrangement": "it uses the specific arrangement",
    "unanticipated": "the C1/C2 pattern is not one of the three branches "
                     "TASK B §4 names; it is reported with both "
                     "intervals and no interpretation is supplied",
}

#: The closing sentence of the §4 guide, carried with every I3 interpretation.
I3_GUIDE_CLOSE = "Each is reportable; none is a failure."

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


def c1(grouped: dict, layer: int, *, reference="original",
       resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> dict:
    """C1 — `mean_L{layer}` minus `original`, per-image NLL.

    Direction that would support "arrangement matters": Delta-NLL > 0, i.e.
    collapsing the gate to its per-head mean makes the loss worse.
    """
    cond = f"mean_L{int(layer)}"
    _ids, d = paired_delta(grouped, cond, reference, "nll")
    # `delta_top1_points` is the top-1 change THE EDIT CAUSED — accuracy
    # under the edit minus accuracy under the reference, in points. It is the
    # same subtraction as the NLL delta and is NOT sign-flipped to "look
    # like" a loss: a condition that hurts shows a positive Delta-NLL and a
    # negative Delta-top-1, and the table says both.
    _ids2, dtop = paired_delta(grouped, cond, reference, "correct")
    out = _summary(d, dtop, resamples=resamples, seed=seed)
    out.update(contrast="C1", condition=cond, reference=reference,
               layer=int(layer),
               definition="mean(nll[mean_L{l}]) - mean(nll[original]), "
                          "paired per image")
    return out


def c2(grouped: dict, layer: int, *, resamples=BOOTSTRAP_RESAMPLES,
       seed=BOOTSTRAP_SEED) -> dict:
    """C2 — the 10 `perm` conditions minus the 7 non-identity `dihedral` ones.

    Both sides are averaged WITHIN IMAGE before the contrast
    (LOCKED_ANALYSIS §8: repeated interventions on one checkpoint are never
    treated as replication), so the resampling unit stays the image.

    Direction that would support "arrangement beyond ring/symmetry structure":
    > 0.
    """
    layer = int(layer)
    perms = family_conditions(grouped, "perm", layer)
    dihs = [c for c in family_conditions(grouped, "dihedral", layer)
            if parse_condition_id(c)["index"] != DIHEDRAL_IDENTITY]
    if len(perms) != 10 or len(dihs) != 7:
        raise ContrastError(
            f"C2 at layer {layer} needs the 10 `perm` conditions and the 7 "
            f"non-identity `dihedral` conditions; found {len(perms)} and "
            f"{len(dihs)}. The condition list is fixed in "
            f"configs/frozen/I3_gate_edits.yaml and a partial sweep is not a "
            f"contrast.")
    ids = sorted(set.intersection(*[set(grouped[c]) for c in perms + dihs]))
    if not ids:
        raise ContrastError(
            f"C2 at layer {layer}: the 17 conditions share no image")

    def _mean_over(conds, field):
        return np.mean([[float(grouped[c][i][field]) for i in ids]
                        for c in conds], axis=0)

    d = _mean_over(perms, "nll") - _mean_over(dihs, "nll")
    dtop = _mean_over(perms, "correct") - _mean_over(dihs, "correct")
    out = _summary(d, dtop, resamples=resamples, seed=seed)
    out.update(contrast="C2", layer=layer, condition="+".join(perms),
               reference="+".join(dihs),
               n_perm=len(perms), n_dihedral=len(dihs),
               definition="mean_i[ mean_k nll(perm k) - mean_t nll(dihedral t, "
                          "t != 0) ], paired per image")
    return out


def c3(*_args, **_kwargs):
    """C3 — I4's primary contrast: theta(primary mask) minus the mean theta of
    the 10 ring-matched, energy-matched controls at eps = 0.10.

    Not implemented in I3 Phase A. C3 reads `measured_perturbation_norm` and
    the mask ids that `configs/frozen/I4_masks.json` (D5) will define, and D5
    is still OPEN in `docs/LOCKED_ANALYSIS.md`. Writing a substitute contract
    for it here — guessing the mask naming, the control count or the energy
    columns — is exactly the failure mode the phase split exists to prevent.
    """
    raise NotImplementedError("C3 is I4 Phase A′")


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

    The three branches the guide names are keyed on "C1 > 0" versus "C1 ~ 0"
    and likewise for C2. A `negative` verdict is NOT "> 0" and is NOT "~ 0":
    the guide did not anticipate it, so it lands in `unanticipated` and the
    text says so rather than borrowing a sentence that would mean the
    opposite of what was measured.
    """
    for name, v in (("C1", c1_verdict), ("C2", c2_verdict)):
        if v not in VERDICTS:
            raise ContrastError(
                f"{name} verdict {v!r} is not one of {list(VERDICTS)}")
    pair = (c1_verdict, c2_verdict)
    if pair == ("null", "null"):
        branch = "neither"
    elif pair == ("positive", "null"):
        branch = "ring_only"
    elif pair == ("positive", "positive"):
        branch = "arrangement"
    else:
        branch = "unanticipated"
    return {
        "branch": branch, "text": I3_BRANCHES[branch],
        "close": I3_GUIDE_CLOSE,
        "c1_verdict": c1_verdict, "c2_verdict": c2_verdict,
        "anticipated": branch != "unanticipated",
        "source": "docs/TASK_B_I3_I4.md §4",
    }


def interpret_i4(*_args, **_kwargs):
    """The TASK B §6 branch an I4 result lands in. Completed in I4 Phase A′,
    with C3."""
    raise NotImplementedError("C3 is I4 Phase A′")


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
        per_c1 = {run: c1(g, layer, resamples=resamples, seed=seed)
                  for run, g in grouped.items()}
        per_c2 = {run: c2(g, layer, resamples=resamples, seed=seed)
                  for run, g in grouped.items()}
        d1, d2 = decide(per_c1), decide(per_c2)
        out[int(layer)] = {
            "C1": d1, "C2": d2,
            "interpretation": interpret_i3(d1["verdict"], d2["verdict"]),
        }
    return out
