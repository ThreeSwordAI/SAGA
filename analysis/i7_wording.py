#!/usr/bin/env python3
"""
analysis/i7_wording.py
======================
TASK C / I7 — D10, the pre-declared wording rule, and the ONLY file in this
project where the word this module decides about is allowed to appear.

THE RULE, AS D10 STATES IT
--------------------------
    The paper uses "attention sink" for the exceedance tokens ONLY IF, in
    every fresh ViT-S/mixup baseline checkpoint (s1, s2), at both blocks, for
    PATCH queries, the ratio of mean incoming mass per exceedance token to
    mean per non-exceedance token is >= 3 with the bootstrap CI excluding 3.
    Otherwise the paper says "high-norm outlier tokens" throughout and
    reports the measured ratio. The 3x is conventional and declared as such.

Four conditions, all of which must hold, and each of which is checked
separately so a failure says WHICH one failed:

  1. every REQUIRED measurement exists          (a missing one cannot pass)
  2. query group is `patch`                     (the CLS row is a different
                                                 question and is reported
                                                 separately)
  3. ratio >= 3.0                               in every required cell
  4. the bootstrap CI EXCLUDES 3.0              in every required cell

Condition 4 is `ci_lo > 3.0`, strictly. A CI whose lower bound is exactly
3.0 CONTAINS 3 and therefore does not exclude it; `test_the_ci_edge` pins
that, because "excluding" is the whole content of the criterion and an
`>=` there would quietly weaken it.

WHY A MODULE AND NOT A SENTENCE IN THE PAPER
--------------------------------------------
The rule was written before the measurement existed. Putting it in code that
returns the verdict AND the sentence means the wording cannot be adjusted
after seeing the number: `decide()` is called once in Phase C and its
sentence is COPIED into the handoff, never edited. If the rule fails, the
paper reports the measured ratio and drops the term — which is a result, not
a disappointment.

THE 3x IS A CONVENTION. It is not derived from anything; D10 says so and so
does the sentence this module returns.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

MISSING = "MISSING"

#: D10's conventional threshold. Declared, not derived.
RATIO_THRESHOLD = 3.0

#: The checkpoints the rule is evaluated on: the two FRESH (seeded, e2r)
#: ViT-S/mixup baselines. The legacy repeats are not "fresh" and the SAGA and
#: register checkpoints are not baselines; D10 names these two.
REQUIRED_RUN_IDS = ("e2r_vits_mixup_baseline_s1", "e2r_vits_mixup_baseline_s2")

#: Both measured blocks, and the only query group the rule reads.
REQUIRED_BLOCKS = (10, 11)
REQUIRED_QUERY_GROUP = "patch"

#: The two verdicts. Nothing else is returnable.
VERDICT_SINK = "attention_sink"
VERDICT_OUTLIER = "high_norm_outlier"


@dataclass
class Verdict:
    """The rule's output: the verdict, the sentence, and why."""

    verdict: str
    sentence: str
    term: str
    passed: bool
    threshold: float = RATIO_THRESHOLD
    min_ratio: object = MISSING
    min_ci_lo: object = MISSING
    failures: list = field(default_factory=list)
    checked: list = field(default_factory=list)
    missing: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "term": self.term, "passed": self.passed,
            "sentence": self.sentence, "threshold": self.threshold,
            "min_ratio": self.min_ratio, "min_ci_lo": self.min_ci_lo,
            "failures": self.failures, "checked": self.checked,
            "missing": self.missing,
            "rule": ("D10: the paper uses 'attention sink' only if, in every "
                     "fresh ViT-S/mixup baseline, at both blocks, for patch "
                     "queries, mean incoming mass per exceedance token is at "
                     "least 3x that per non-exceedance token with the "
                     "bootstrap CI excluding 3. The 3x is conventional."),
        }


def _key(run_id, block):
    return f"{run_id}|block{int(block)}"


def _fmt(v):
    return MISSING if v is MISSING or v is None else f"{float(v):.2f}"


def decide(measurements, *, threshold=RATIO_THRESHOLD,
           required_run_ids=REQUIRED_RUN_IDS,
           required_blocks=REQUIRED_BLOCKS) -> Verdict:
    """Apply D10. `measurements` is an iterable of dicts carrying

        run_id, block, query_group, ratio, ci_lo, ci_hi

    Returns a `Verdict` with the verdict, the sentence to copy into the
    paper, and the list of cells that failed. Measurements for other runs,
    other blocks or the CLS query group are IGNORED — the rule names its own
    evidence, so a passing measurement somewhere else cannot rescue it and a
    failing one cannot sink it.
    """
    by_key = {}
    for m in measurements:
        if str(m.get("query_group")) != REQUIRED_QUERY_GROUP:
            continue
        by_key[_key(m["run_id"], m["block"])] = m

    checked, failures, missing = [], [], []
    ratios, ci_los = [], []
    for run_id in required_run_ids:
        for block in required_blocks:
            key = _key(run_id, block)
            m = by_key.get(key)
            if m is None:
                missing.append(key)
                failures.append(f"{key}: no measurement")
                continue
            ratio = m.get("ratio", MISSING)
            ci_lo = m.get("ci_lo", MISSING)
            if ratio in (MISSING, None) or ci_lo in (MISSING, None):
                missing.append(key)
                failures.append(f"{key}: ratio or CI is MISSING")
                continue
            ratio, ci_lo = float(ratio), float(ci_lo)
            ratios.append(ratio)
            ci_los.append(ci_lo)
            checked.append({"cell": key, "ratio": ratio, "ci_lo": ci_lo,
                            "ci_hi": m.get("ci_hi", MISSING)})
            if ratio < threshold:
                failures.append(
                    f"{key}: ratio {ratio:.2f} < {threshold:g}")
            elif ci_lo <= threshold:
                # NOT `<`: a CI whose lower bound IS the threshold contains
                # it, and D10 requires the interval to EXCLUDE 3.
                failures.append(
                    f"{key}: CI lower bound {ci_lo:.2f} does not exclude "
                    f"{threshold:g}")

    passed = not failures and len(checked) == len(required_run_ids) * len(
        required_blocks)
    min_ratio = min(ratios) if ratios else MISSING
    min_ci_lo = min(ci_los) if ci_los else MISSING

    if passed:
        term = "attention sink"
        sentence = (
            f"Across both fresh ViT-S/mixup baseline checkpoints and both "
            f"measured blocks, patch queries send at least "
            f"{_fmt(min_ratio)}x more attention mass to high-norm outlier "
            f"tokens than to other tokens (smallest bootstrap CI lower bound "
            f"{_fmt(min_ci_lo)}, above the pre-declared and conventional "
            f"threshold of {threshold:g}x), so this paper calls these tokens "
            f"attention sinks.")
        verdict = VERDICT_SINK
    else:
        term = "high-norm outlier tokens"
        if ratios:
            measured = (f"the smallest measured ratio across the four "
                        f"required cells is {_fmt(min_ratio)}x (bootstrap CI "
                        f"lower bound {_fmt(min_ci_lo)})")
        else:
            measured = ("no required cell was measured, so no ratio can be "
                        "reported")
        sentence = (
            f"The pre-declared criterion for the term attention sink — mean "
            f"incoming attention mass per high-norm outlier token at least "
            f"{threshold:g}x that per other token, with the bootstrap CI "
            f"excluding {threshold:g}, in both fresh ViT-S/mixup baselines at "
            f"both measured blocks — is not met: {measured}. This paper "
            f"therefore says high-norm outlier tokens throughout and does not "
            f"use the term attention sink.")
        verdict = VERDICT_OUTLIER

    return Verdict(verdict=verdict, sentence=sentence, term=term,
                   passed=passed, threshold=float(threshold),
                   min_ratio=min_ratio, min_ci_lo=min_ci_lo,
                   failures=failures, checked=checked, missing=missing)


def from_table(path, **kw) -> Verdict:
    """Apply D10 to a committed `T_I7a_incoming.csv`.

    Reads only the columns the rule names, so a table that gained a column
    still decides identically.
    """
    import csv

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "run_id": r.get("run_id"),
                "block": int(r["block"]) if str(
                    r.get("block", "")).strip().isdigit() else -1,
                "query_group": r.get("query_group"),
                "ratio": r.get("ratio", MISSING),
                "ci_lo": r.get("ratio_ci_lo", r.get("ci_lo", MISSING)),
                "ci_hi": r.get("ratio_ci_hi", r.get("ci_hi", MISSING)),
            })
    return decide(rows, **kw)


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="TASK C / I7 — apply D10 and print the verdict and the "
                    "sentence. Run ONCE; the sentence is copied into the "
                    "handoff, never edited.")
    p.add_argument("--table", required=True,
                   help="results/frozen/I7_attention/<split>/tables/"
                        "T_I7a_incoming.csv")
    p.add_argument("--out", default=None,
                   help="also write the verdict as JSON here")
    args = p.parse_args()

    verdict = from_table(args.table)
    print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    print()
    print(verdict.sentence)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(verdict.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
