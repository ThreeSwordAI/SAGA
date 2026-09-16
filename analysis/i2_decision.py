#!/usr/bin/env python3
"""
analysis/i2_decision.py
=======================
TASK I2 §7 — the PRE-DECLARED decision rule for D1, as code.

D1 is "which feature stage do all new patch diagnostics use?". It is the one
open decision in `docs/LOCKED_ANALYSIS.md` that I2 exists to close, and it is
exactly the kind of decision that must not be made by looking at a table:
`decide()` takes the measured gaps and returns the branch, the per-diagnostic
survival values and the sentence to paste into `LOCKED_ANALYSIS.md`. It takes
no options, has no discretion, and is unit-tested on synthetic gaps for all
three branches.

The rule, verbatim from the task file §7, computed on the CALIBRATION split
only, for the ViT-S/mixup cell (4 pairs), on the primary diagnostics:

  1. Terminal gate is a MINOR CONTRIBUTOR if, for every primary diagnostic,
     sign(gap_term_1.00) == sign(gap_native) and survival >= 0.70 (paired
     means).                                              -> D1 = `hist`
  2. Otherwise, if for every primary diagnostic
     sign(gap_s11) == sign(gap_native) and survival_s11 >= 0.70
                                                          -> D1 = `s11_out`
  3. Otherwise                                            -> D1 = `s11_out`,
     and I2 is promoted to a main finding: the historical SAGA-vs-baseline
     diagnostic distinction was substantially a terminal-gate effect.

The cutoff 0.70 is CONVENTIONAL and is declared as such — it is named once,
here, and the complete tables are reported whatever the branch. The
ViT-S/nomix and ViT-B cells are computed and reported alongside but DO NOT
VOTE (n = 2).

**The module proposes. The human signs D1 (name, date, git sha) before any
B2 run and before any I1/I3/I4/I5 Phase-B run.**

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

MISSING = "MISSING"

#: The conventional survival cutoff. One name, one place (task §7).
SURVIVAL_CUTOFF = 0.70

#: The diagnostics the rule votes on (task §7).
PRIMARY_DIAGNOSTICS = ("cos_all", "cos_nosink_mad", "eff_rank",
                       "count_fixed_cal", "count_mad")

#: The only cell that votes: ViT-S/mixup, 4 pairs. The others are reported
#: beside it and are explicitly non-voting at n = 2.
DECISION_CELL = "vit_small|mixup"

#: Setting names as `analysis/build_I2_tables.py` writes them.
SETTING_NATIVE = "hist_native"
SETTING_TERM = "hist_term_1.00"
SETTING_S11 = "s11_out_native"

BRANCH_MINOR = 1
BRANCH_STAGE_SHIFT = 2
BRANCH_MAIN_FINDING = 3

#: branch -> the D1 value it selects.
BRANCH_STAGE = {BRANCH_MINOR: "hist",
                BRANCH_STAGE_SHIFT: "s11_out",
                BRANCH_MAIN_FINDING: "s11_out"}


class DecisionError(ValueError):
    """Gaps the rule cannot be evaluated on."""


def _sign(x) -> int:
    """1, -1 or 0. A gap of exactly 0 has sign 0 and matches nothing but 0."""
    if x == MISSING or x is None:
        return 0
    return (x > 0) - (x < 0)


def survival_of(gap_native, gap_other):
    """gap_other / gap_native, or MISSING when the ratio is not defined.

    MISSING is returned when either gap is MISSING and when the denominator
    is exactly 0 — a survival fraction of a zero gap is not a small number,
    it is not a number. A diagnostic whose survival is MISSING can never
    satisfy "survival >= 0.70", so it fails its branch test rather than
    being quietly dropped from the conjunction.

    This is THE definition of survival: `analysis/build_I2_tables.py` imports
    it rather than writing the division a second time.
    """
    if gap_native == MISSING or gap_other == MISSING:
        return MISSING
    if gap_native is None or gap_other is None:
        return MISSING
    if float(gap_native) == 0.0:
        return MISSING
    return float(gap_other) / float(gap_native)


def _passes(gap_native, gap_other):
    """(survival, sign_match, passes) for one diagnostic at one setting."""
    surv = survival_of(gap_native, gap_other)
    sign_match = bool(_sign(gap_native) != 0
                      and _sign(gap_native) == _sign(gap_other))
    ok = bool(sign_match and surv != MISSING and surv >= SURVIVAL_CUTOFF)
    return surv, sign_match, ok


def decide(gaps: dict, *, cell: str = DECISION_CELL, n_pairs=MISSING) -> dict:
    """Apply task §7 to one cell's gaps and return the whole verdict.

    `gaps` is `{diagnostic: {setting: gap}}` with the three setting names
    above; a gap may be the literal MISSING. Every primary diagnostic must be
    present — a rule that silently skipped one would be a different rule.

    Returns the branch, the D1 stage it selects, the per-diagnostic survival
    and sign-match values for BOTH tests, and the sentence for
    `docs/LOCKED_ANALYSIS.md`.
    """
    absent = [d for d in PRIMARY_DIAGNOSTICS if d not in gaps]
    if absent:
        raise DecisionError(
            f"the decision rule needs every primary diagnostic "
            f"{list(PRIMARY_DIAGNOSTICS)}; {absent} were not supplied")

    survival, survival_s11 = {}, {}
    sign_match, sign_match_s11 = {}, {}
    pass_term, pass_s11 = {}, {}
    for d in PRIMARY_DIAGNOSTICS:
        row = gaps[d]
        for setting in (SETTING_NATIVE, SETTING_TERM, SETTING_S11):
            if setting not in row:
                raise DecisionError(
                    f"diagnostic {d!r} has no {setting!r} gap; the rule "
                    f"compares native against term_1.00 and s11_out")
        base = row[SETTING_NATIVE]
        survival[d], sign_match[d], pass_term[d] = _passes(
            base, row[SETTING_TERM])
        survival_s11[d], sign_match_s11[d], pass_s11[d] = _passes(
            base, row[SETTING_S11])

    if all(pass_term.values()):
        branch = BRANCH_MINOR
    elif all(pass_s11.values()):
        branch = BRANCH_STAGE_SHIFT
    else:
        branch = BRANCH_MAIN_FINDING

    out = {
        "branch": branch,
        "branch_name": {BRANCH_MINOR: "terminal gate is a minor contributor",
                        BRANCH_STAGE_SHIFT: "report at s11_out",
                        BRANCH_MAIN_FINDING: "I2 is a main finding"}[branch],
        "d1_stage": BRANCH_STAGE[branch],
        "cell": cell,
        "n_pairs": n_pairs,
        "cutoff": SURVIVAL_CUTOFF,
        "primary_diagnostics": list(PRIMARY_DIAGNOSTICS),
        "gap_native": {d: gaps[d][SETTING_NATIVE] for d in PRIMARY_DIAGNOSTICS},
        "gap_term": {d: gaps[d][SETTING_TERM] for d in PRIMARY_DIAGNOSTICS},
        "gap_s11": {d: gaps[d][SETTING_S11] for d in PRIMARY_DIAGNOSTICS},
        "survival": survival,
        "survival_s11": survival_s11,
        "sign_match": sign_match,
        "sign_match_s11": sign_match_s11,
        "passes_term": pass_term,
        "passes_s11": pass_s11,
        "failing_diagnostics_term": sorted(d for d, ok in pass_term.items()
                                           if not ok),
        "failing_diagnostics_s11": sorted(d for d, ok in pass_s11.items()
                                          if not ok),
    }
    out["sentence"] = sentence_for(out)
    return out


def _fmt(v) -> str:
    return MISSING if v == MISSING or v is None else f"{float(v):.3f}"


def sentence_for(verdict: dict) -> str:
    """The line the human pastes into `docs/LOCKED_ANALYSIS.md` §1 on signing.

    One sentence per branch, with the measured survival values inlined so the
    line carries its own evidence and the 0.70 cutoff is named as the
    convention it is.
    """
    surv = ", ".join(f"{d} {_fmt(verdict['survival'][d])}"
                     for d in PRIMARY_DIAGNOSTICS)
    surv11 = ", ".join(f"{d} {_fmt(verdict['survival_s11'][d])}"
                       for d in PRIMARY_DIAGNOSTICS)
    cell, cut = verdict["cell"], verdict["cutoff"]
    n = verdict["n_pairs"]
    if verdict["branch"] == BRANCH_MINOR:
        return (
            f"D1 = `hist` (= `s12_pre_norm`): on the calibration split, in "
            f"the {cell} cell (n = {n} pairs), every primary diagnostic keeps "
            f"the sign of its native SAGA-vs-baseline gap when the terminal "
            f"patch gate is bypassed and retains at least the conventional "
            f"{cut:.2f} of it (survival: {surv}), so the untrained terminal "
            f"constant is a minor contributor and the historical stage stays "
            f"the reporting stage; `term_1.00` and `s11_out` are reported as "
            f"appendix controls.")
    if verdict["branch"] == BRANCH_STAGE_SHIFT:
        failed = ", ".join(verdict["failing_diagnostics_term"])
        return (
            f"D1 = `s11_out`: on the calibration split, in the {cell} cell "
            f"(n = {n} pairs), the terminal-gate bypass does not preserve the "
            f"native gap for {failed} (survival: {surv}), while the "
            f"second-to-last block does for every primary diagnostic "
            f"(survival_s11: {surv11}, all at or above the conventional "
            f"{cut:.2f}), so all new patch diagnostics are reported at "
            f"`s11_out` and the final-block diagnostics are reported only as "
            f"a readout-contaminated control, stated as such.")
    failed = ", ".join(verdict["failing_diagnostics_term"])
    failed11 = ", ".join(verdict["failing_diagnostics_s11"])
    return (
        f"D1 = `s11_out`, and I2 is a main finding: on the calibration split, "
        f"in the {cell} cell (n = {n} pairs), neither the terminal-gate "
        f"bypass ({failed} below the conventional {cut:.2f} or sign-flipped; "
        f"survival: {surv}) nor the second-to-last block ({failed11}; "
        f"survival_s11: {surv11}) preserves the native SAGA-vs-baseline gap, "
        f"so the historical final-block diagnostic distinction was "
        f"substantially a terminal-gate effect and the claim ladder narrows "
        f"accordingly.")


# ─────────────────────────────────────────────────────────────────────────────
# Reading the gaps table
# ─────────────────────────────────────────────────────────────────────────────

def _num(v):
    if v in ("", None, MISSING):
        return MISSING
    try:
        return float(v)
    except (TypeError, ValueError):
        return MISSING


def load_gaps(table_path) -> dict:
    """`{cell: ({diagnostic: {setting: gap}}, n_pairs)}` from T_I2c_gaps.csv."""
    path = Path(table_path)
    if not path.exists():
        raise DecisionError(
            f"{path} not found — run analysis/build_I2_tables.py on the "
            f"calibration results first")
    out, n_pairs = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell = row["cell"]
            out.setdefault(cell, {}).setdefault(
                row["diagnostic"], {})[row["setting"]] = _num(row["gap_mean"])
            # the MAXIMUM over rows: a row whose gap is MISSING carries
            # n_pairs = 0, and the cell's pair count is not 0 because one
            # diagnostic could not be computed
            got = _num(row["n_pairs"])
            if got != MISSING:
                n_pairs[cell] = max(n_pairs.get(cell, 0), int(got))
    return {c: (d, n_pairs.get(c, MISSING)) for c, d in out.items()}


def main():
    p = argparse.ArgumentParser(
        description="Apply TASK I2 §7's pre-declared rule and propose D1.")
    p.add_argument("--gaps",
                   default="results/frozen/I2_terminal/calibration/tables/T_I2c_gaps.csv")
    p.add_argument("--cell", default=DECISION_CELL,
                   help="the VOTING cell; the others are reported but do not "
                        "vote (n = 2)")
    p.add_argument("--out", default=None,
                   help="also write the verdict as JSON")
    args = p.parse_args()

    by_cell = load_gaps(args.gaps)
    if args.cell not in by_cell:
        raise SystemExit(
            f"{args.gaps} has no rows for the voting cell {args.cell!r}; "
            f"it has {sorted(by_cell)}")
    gaps, n_pairs = by_cell[args.cell]
    verdict = decide(gaps, cell=args.cell, n_pairs=n_pairs)

    print(f"branch {verdict['branch']} — {verdict['branch_name']}")
    print(f"D1 = {verdict['d1_stage']}   (cutoff {verdict['cutoff']}, cell "
          f"{verdict['cell']}, n_pairs {verdict['n_pairs']})")
    print(f"{'diagnostic':<18}{'gap_native':>13}{'gap_term':>13}"
          f"{'survival':>11}{'gap_s11':>13}{'survival_s11':>14}")
    for d in PRIMARY_DIAGNOSTICS:
        print(f"{d:<18}{_fmt(verdict['gap_native'][d]):>13}"
              f"{_fmt(verdict['gap_term'][d]):>13}"
              f"{_fmt(verdict['survival'][d]):>11}"
              f"{_fmt(verdict['gap_s11'][d]):>13}"
              f"{_fmt(verdict['survival_s11'][d]):>14}")
    print("\nnon-voting cells (n = 2): " + ", ".join(
        sorted(c for c in by_cell if c != args.cell)))
    print("\nPROPOSED LINE FOR docs/LOCKED_ANALYSIS.md §1 — THE HUMAN SIGNS:")
    print(verdict["sentence"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8", newline="\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
