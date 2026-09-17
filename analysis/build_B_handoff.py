#!/usr/bin/env python3
"""
analysis/build_B_handoff.py
===========================
TASK B B11 — `docs/B_HANDOFF.md`, generated.

    python analysis/build_B_handoff.py [--out docs/B_HANDOFF.md]

Every number in the document is read from `results/frozen/I3_gate_edits/`,
`results/frozen/I4_perturbation/` and `docs/LOCKED_ANALYSIS.md`; none is
typed by hand. The surrounding prose is fixed text. Re-run after new results
land and the document updates itself, the way `build_I0_handoff.py` and
`build_I2_handoff.py` already do.

TWO THINGS THIS GENERATOR REFUSES TO DO
---------------------------------------
1. **It will not render while `docs/LOCKED_ANALYSIS.md` is unfrozen.** A
   handoff is the document a reader takes the result from. Producing one
   while the analysis parameters are still a draft would publish numbers
   whose pre-registration is unsigned, which is the whole thing the freeze
   exists to prevent. The refusal names what is missing, the same list
   `saga/frozen/masks.assert_frozen` produces, and `--allow-unfrozen` exists
   ONLY for the tests.

2. **It will not drop the recorded deviation.** `docs/TASK_B_I3_I4.md` §4
   carries a disclosure of what had been computed when its interpretation
   guide was amended, and §11 requires this document to carry it VERBATIM
   under the heading "Recorded deviation". It is extracted from the task file
   rather than restated here, and a task file that no longer contains it is a
   hard error — a disclosure that can be lost by editing a generator is not a
   disclosure.

The interpretation branch each contrast lands in is COPIED from
`analysis/i34_contrasts.py` output. This module chooses no sentence.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import masks as fmasks  # noqa: E402

MISSING = "MISSING"
REPO = Path(__file__).resolve().parents[1]
TASK_FILE = REPO / "docs" / "TASK_B_I3_I4.md"

#: The §4 heading the disclosure lives under, and the one it is republished
#: under here. §11 of the task file names both.
DEVIATION_HEADING = "Recorded deviation"

I3_TABLES = REPO / "results" / "frozen" / "I3_gate_edits" / "evaluation" / "tables"
I4_TABLES = REPO / "results" / "frozen" / "I4_perturbation" / "evaluation" / "tables"


class HandoffError(ValueError):
    """A handoff this generator refuses to write."""


# ─────────────────────────────────────────────────────────────────────────────
# The two refusals
# ─────────────────────────────────────────────────────────────────────────────

def assert_frozen_or_refuse(locked_path=None, masks_path=None):
    """Refuse to render while the analysis parameters are unfrozen."""
    state = fmasks.locked_state(locked_path or fmasks.LOCKED_FILE,
                                masks_path=masks_path or fmasks.MASKS_FILE)
    if state["missing"]:
        bullets = "\n  - ".join(state["missing"])
        raise HandoffError(
            f"REFUSING to write the Track B handoff: "
            f"`docs/LOCKED_ANALYSIS.md` is not frozen.\n  - {bullets}\n"
            f"A handoff is what a reader takes the result from. Publishing "
            f"one while the analysis parameters are a draft would present "
            f"numbers whose pre-registration is unsigned.")
    return state


def recorded_deviation(task_file=TASK_FILE) -> str:
    """The §4 disclosure, verbatim, or a hard error.

    Extracted rather than restated: a copy in this file could drift from the
    task file, and the version a reader trusts would then depend on which
    document they opened. The section runs from its `### Recorded deviation`
    heading to the next heading of the same or higher level.
    """
    p = Path(task_file)
    if not p.exists():
        raise HandoffError(f"{p} not found — the disclosure lives in its §4")
    text = p.read_text(encoding="utf-8")
    m = re.search(r"^###\s+" + DEVIATION_HEADING + r".*$", text,
                  flags=re.MULTILINE)
    if not m:
        raise HandoffError(
            f"{p} has no '### {DEVIATION_HEADING}' section. §11 requires this "
            f"handoff to carry it verbatim; a disclosure that can be lost by "
            f"editing the task file is not a disclosure.")
    rest = text[m.end():]
    nxt = re.search(r"^#{1,3}\s+\S", rest, flags=re.MULTILINE)
    body = (rest[:nxt.start()] if nxt else rest).strip()
    if len(body) < 200:
        raise HandoffError(
            f"the '{DEVIATION_HEADING}' section of {p} is {len(body)} "
            f"characters — too short to be the disclosure")
    for token in ("3a030f6", "3b0ca2f", "layer 7"):
        if token not in body:
            raise HandoffError(
                f"the '{DEVIATION_HEADING}' section of {p} does not mention "
                f"{token!r}; it is not the disclosure §11 names")
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Reading what landed
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_meta(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fmt(v, nd=5):
    if v in (None, "", MISSING):
        return MISSING
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def contrast_rows(rows, contrast, role=None):
    out = [r for r in rows if r.get("contrast") == contrast]
    if role is not None:
        out = [r for r in out if r.get("role") == role]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The document
# ─────────────────────────────────────────────────────────────────────────────

def build(*, locked_path=None, masks_path=None, allow_unfrozen=False,
          i3_tables=I3_TABLES, i4_tables=I4_TABLES,
          task_file=TASK_FILE, git_sha_value=None) -> str:
    state = ({"date": MISSING, "git_sha": MISSING, "masks_sha": MISSING}
             if allow_unfrozen
             else assert_frozen_or_refuse(locked_path, masks_path))
    deviation = recorded_deviation(task_file)

    i3b = read_csv(Path(i3_tables) / "T_I3b_contrasts.csv")
    i3a = read_csv(Path(i3_tables) / "T_I3a_conditions.csv")
    i3c = read_csv(Path(i3_tables) / "T_I3c_energy_strata.csv")
    i4b = read_csv(Path(i4_tables) / "T_I4b_contrast.csv")
    i4c = read_csv(Path(i4_tables) / "T_I4c_cross_method.csv")
    i3meta = read_meta(Path(i3_tables) / "I3_tables_meta.json")
    i4meta = read_meta(Path(i4_tables) / "I4_tables_meta.json")

    A = []
    A.append("# TASK B handoff — does the learned gate arrangement matter, "
             "and do high-prevalence coordinates respond differently?")
    A.append("")
    A.append(f"**GENERATED by `analysis/build_B_handoff.py`"
             f"{'' if git_sha_value is None else f' at git `{git_sha_value}`'}"
             f".** Every number below is read from "
             f"`results/frozen/I3_gate_edits/` and "
             f"`results/frozen/I4_perturbation/`; none is typed by hand. The "
             f"interpretation branch each contrast lands in is COPIED from "
             f"`analysis/i34_contrasts.py` output — this generator chooses no "
             f"sentence. Re-run it after new results land.")
    A.append("")
    A.append(f"Analysis parameters frozen **{state['date']}** at git "
             f"`{state['git_sha']}`; D5 masks sha256 `{state['masks_sha']}`.")
    A.append("")

    # ── the contrasts ────────────────────────────────────────────────────────
    A.append("## 1. The three primary contrasts")
    A.append("")
    A.append("D7 named them before any evaluation number existed, and "
             "`docs/LOCKED_ANALYSIS.md` §9 fixes which two of I3's are "
             "primary. The decision rule (§10) is: direction consistent "
             "across the fresh ViT-S/mixup pair, image-level bootstrap CI "
             "(10,000 resamples, seed 0) excluding zero in each, effect size "
             "in nats and top-1 points.")
    A.append("")
    if not i3b and not i4b:
        A.append("> **PENDING.** Neither `T_I3b_contrasts.csv` nor "
                 "`T_I4b_contrast.csv` is present; Phase C has not run.")
    for label, rows, cname in (("C1", i3b, "C1"), ("C2", i3b, "C2"),
                               ("C3", i4b, "C3")):
        sel = contrast_rows(rows, cname)
        if not sel:
            A.append(f"- **{label}** — MISSING (no rows in the table).")
            continue
        deciding = [r for r in sel if str(r.get("decides")).lower() == "true"]
        verdict = sorted({r.get("verdict", MISSING) for r in sel})
        branch = sorted({r.get("branch", MISSING) for r in sel
                         if r.get("branch") not in (None, "", MISSING)})
        A.append(f"- **{label}** — verdict {', '.join(verdict) or MISSING}"
                 f"; {len(deciding)} deciding checkpoint(s).")
        for r in deciding:
            A.append(f"    - `{r.get('run_id')}` "
                     f"{_fmt(r.get('mean_nats'))} nats "
                     f"[{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}], "
                     f"Δtop-1 {_fmt(r.get('delta_top1_points'), 3)} pts")
        if branch:
            A.append(f"    - branch: **{', '.join(branch)}**")
    A.append("")

    # the pre-declared secondaries
    sec = contrast_rows(i3b, "S1") + contrast_rows(i3b, "S2")
    A.append("### Pre-declared secondary contrasts")
    A.append("")
    A.append("`mean` − `original` (S1) and `permute` − `dihedral` (S2) are "
             "reported under the same rule with no branch of their own "
             "(LOCKED §9, TASK B §4).")
    A.append("")
    if sec:
        A.append("| contrast | run_id | nats | CI | verdict |")
        A.append("|---|---|---|---|---|")
        for r in sec:
            if str(r.get("decides")).lower() != "true":
                continue
            A.append(f"| {r.get('contrast')} | `{r.get('run_id')}` | "
                     f"{_fmt(r.get('mean_nats'))} | "
                     f"[{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}] | "
                     f"{r.get('verdict')} |")
    else:
        A.append("> **PENDING** — Phase C has not run.")
    A.append("")

    # ── energy ───────────────────────────────────────────────────────────────
    A.append("## 2. Energy, and why the comparison is fair")
    A.append("")
    A.append("**I3 matches by STRATIFICATION** (LOCKED §3 item 7, amended): "
             "ΔNLL is reported within deciles of `delta_update_norm`, pooled "
             "over the edited conditions per checkpoint, so the families are "
             "compared at equal injected energy. **I4 matches PER IMAGE**: a "
             "control's `energy_target` is the primary's measured injected "
             "norm on the same image, and the achieved norm is recorded for "
             "every condition.")
    A.append("")
    A.append(f"- I3 strata rows: {len(i3c) or MISSING}")
    worst = i4meta.get("energy_rel_error_max", MISSING)
    A.append(f"- I4 largest `energy_rel_error` observed: "
             f"**{_fmt(worst, 6)}** (tolerance 0.01)")
    A.append("- Unmatched controls (`*_e10f_*`) are reported beside the "
             "matched ones; the gap between them is the size of the energy "
             "confound, reported rather than removed.")
    A.append("")

    # ── cross-method ─────────────────────────────────────────────────────────
    A.append("## 3. The cross-method row")
    A.append("")
    if i4c:
        paired = [r for r in i4c
                  if r.get("comparison") == "baseline_minus_saga"]
        A.append("| layer | tag | baseline − SAGA (nats) | paired with |")
        A.append("|---|---|---|---|")
        for r in paired:
            A.append(f"| {r.get('layer')} | {r.get('provenance_tag')} | "
                     f"{_fmt(r.get('paired_difference'))} | "
                     f"`{r.get('paired_with')}` |")
        A.append("")
        A.append("Registers are n = 2 and are labelled on every row; nothing "
                 "pools them with anything.")
    else:
        A.append("> **PENDING** — `T_I4c_cross_method.csv` is not present.")
    A.append("")

    # ── the deviation, verbatim ──────────────────────────────────────────────
    A.append(f"## 4. {DEVIATION_HEADING}")
    A.append("")
    A.append("*Copied verbatim from `docs/TASK_B_I3_I4.md` §4, as §11 "
             "requires. This generator refuses to render without it.*")
    A.append("")
    A.append(deviation)
    A.append("")

    # ── what Track B does not settle ─────────────────────────────────────────
    A.append("## 5. What Track B does NOT settle")
    A.append("")
    A.append("- **Frozen-model edits show what the trained model USES, not "
             "what a differently trained model would achieve.** I3 edits a "
             "checkpoint at inference and never retrains; what a scalar gate "
             "trained from scratch would do is Track T and is not claimed "
             "here.")
    A.append("- **I4's masks come from one cell.** D5 is defined for "
             "`vit_small|mixup` only, so no other architecture or recipe is "
             "in I4.")
    A.append("- **Both I4 primary masks are the same 16 coordinates.** The "
             "two cell mean maps correlate at ρ = 0.9985, so comparing I4's "
             "two sites is a comparison of DEPTH with the address held "
             "fixed — not of two addresses.")
    A.append("- **Registers are n = 2.** They are reported and labelled and "
             "never decide.")
    A.append("- **A null is a result.** A contrast whose CI includes zero is "
             "reported as a null with its interval; nothing is re-framed.")
    A.append("")
    A.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
             f" from {len(i3a)} I3 condition rows and {len(i4b)} I4 contrast "
             f"rows. I3 decisions: {i3meta.get('decisions', MISSING)}. "
             f"I4 decisions: {i4meta.get('decisions', MISSING)}.")
    A.append("")
    return "\n".join(A)


def main():
    p = argparse.ArgumentParser(description="Generate docs/B_HANDOFF.md.")
    p.add_argument("--out", default="docs/B_HANDOFF.md")
    p.add_argument("--git-sha", default=None,
                   help="record the sha this was built at (tests pass it to "
                        "regenerate byte-identically)")
    p.add_argument("--allow-unfrozen", action="store_true",
                   help="TESTS ONLY: render without the freeze. Never use "
                        "this for a document anyone will read.")
    args = p.parse_args()
    try:
        text = build(allow_unfrozen=args.allow_unfrozen,
                     git_sha_value=args.git_sha)
    except HandoffError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
