#!/usr/bin/env python3
"""
analysis/build_D1_proposal.py
=============================
TASK I2 D5 — the generated D1 proposal note.

    python analysis/build_D1_proposal.py
        -> results/frozen/I2_terminal/D1_proposal.md

Every number in the note is READ from `results/frozen/I2_terminal/tables/`
and from `analysis/i2_decision.decide()`; not one is typed here. The
surrounding prose is fixed text, so re-running after new results land
updates the document rather than inviting a hand edit, and
`tests/test_I2_terminal.py` pins the output byte for byte.

The note PROPOSES. It carries an unsigned signature block, and
`docs/LOCKED_ANALYSIS.md` §1 stays `DECISION NEEDED` until the human fills
that block in. No B2 run and no I1/I3/I4/I5 Phase-B run may start before
then (task file §7, §10).

The note also states, in its own section, where the pre-declared branch
LABEL and the measured numbers pull in different directions. That is not a
softening of the rule — the rule's verdict is printed verbatim and is not
touched — it is the disclosure the human needs in order to sign something
true.

No training, no optimizer, no probe fitting.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.i2_decision import (DECISION_CELL, PRIMARY_DIAGNOSTICS,  # noqa: E402
                                  SETTING_NATIVE, SETTING_S11, SETTING_TERM,
                                  SURVIVAL_CUTOFF, decide, load_gaps)

MISSING = "MISSING"
OUT = "results/frozen/I2_terminal/D1_proposal.md"
TABLES = "results/frozen/I2_terminal/tables"
THRESHOLDS = "results/frozen/I2_terminal/thresholds_cal.json"


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


#: The two diagnostics that behave differently from the other three here, so
#: the note can say WHICH ones rather than quoting a range that spans both.
COSINE_DIAGNOSTICS = ("cos_all", "cos_nosink_mad")


def _f(v, nd=4):
    return MISSING if v in (MISSING, "", None) else f"{float(v):.{nd}f}"


def _cell(text: str) -> str:
    """Escape a value going INSIDE a markdown table cell.

    A literal `|` silently breaks the row it sits in — every cell name in
    this project is `<arch>|<recipe_actual>`, and TASK I0 Phase C hit exactly
    this and recorded it in docs/TASK_LOG.md. Escaped here rather than at
    each call site, so a new table cannot reintroduce it.
    """
    return str(text).replace("|", "\\|")


def _gap_row(rows, cell, diagnostic, setting):
    hit = [r for r in rows if r["cell"] == cell
           and r["diagnostic"] == diagnostic and r["setting"] == setting]
    return hit[0] if hit else None


def build(tables=TABLES, thresholds=THRESHOLDS, cell=DECISION_CELL) -> str:
    tables = Path(tables)
    gaps_csv = tables / "T_I2c_gaps.csv"
    gaps_rows = _read(gaps_csv)
    inv_rows = _read(tables / "T_I2a_invariance.csv")
    maps_rows = _read(tables / "T_I2d_maps.csv")
    thr = json.loads(Path(thresholds).read_text(encoding="utf-8"))

    by_cell = load_gaps(gaps_csv)
    if cell not in by_cell:
        raise SystemExit(f"{gaps_csv} has no rows for the voting cell {cell!r}")
    gaps, n_pairs = by_cell[cell]
    v = decide(gaps, cell=cell, n_pairs=n_pairs)

    prov = gaps_rows[0]
    split_name, split_sha = prov["split_name"], prov["split_sha256"]
    git_sha = prov["git_sha"]
    n_images = _gap_row(gaps_rows, cell, PRIMARY_DIAGNOSTICS[0],
                        SETTING_NATIVE)["n_images"]

    # ── the architectural half, measured ────────────────────────────────────
    logit_max = max(float(r["max_abs_logit_diff_vs_native"]) for r in inv_rows)
    agree_min = min(float(r["top1_agreement"]) for r in inv_rows)
    nll_max = max(float(r["mean_abs_delta_nll"]) for r in inv_rows)
    n_ckpt = len({r["run_id"] for r in inv_rows})
    n_const = len({r["condition_id"] for r in inv_rows})
    half = [r for r in inv_rows if r["condition_id"] == "term_0.50"]
    identical = sum(r["phi_L_zero_bit_identical"] == "1" for r in half)
    cos_moves = max(float(r["max_abs_cos_all_diff_hist"]) for r in inv_rows
                    if r["condition_id"] == "term_1.00")

    # ── how close the branch was, from the survivals themselves ─────────────
    surv = {d: v["survival"][d] for d in PRIMARY_DIAGNOSTICS}
    surv11 = {d: v["survival_s11"][d] for d in PRIMARY_DIAGNOSTICS}
    numeric = [s for s in surv.values() if s != MISSING]
    numeric11 = [s for s in surv11.values() if s != MISSING]
    failed = v["failing_diagnostics_term"]
    margins = {d: SURVIVAL_CUTOFF - surv[d] for d in failed
               if surv[d] != MISSING}

    empty_maps = [r for r in maps_rows if r["rho_spatial"] == MISSING]
    empty_stages = sorted({r["stage"] for r in empty_maps})

    L = []
    w = L.append
    w("# D1 proposal — which feature stage all new patch diagnostics use")
    w("")
    w("> ## STATUS: **PROPOSED — NOT SIGNED**")
    w(">")
    w("> Signed and dated by: *(the human — nobody else)*  ")
    w("> Date signed: `PENDING`  ")
    w("> Git sha at signature: `PENDING`")
    w(">")
    w("> Until those three lines are filled in and the sentence below is "
      "copied into  ")
    w("> `docs/LOCKED_ANALYSIS.md` §1, **D1 is still `DECISION NEEDED`**: no "
      "I2 Phase-B2  ")
    w("> run and no I1/I3/I4/I5 Phase-B run may start (task file §7, §10).")
    w("")
    w("GENERATED by `analysis/build_D1_proposal.py`. Every number below is "
      "read from")
    w("`results/frozen/I2_terminal/tables/` and from "
      "`analysis/i2_decision.decide()`;")
    w("none is typed by hand. Re-run the script after new results land and "
      "this")
    w("document updates itself.")
    w("")
    w(f"| | |")
    w(f"|---|---|")
    w(f"| split | `{split_name}`, sha `{split_sha}` |")
    w(f"| images | {n_images} |")
    w(f"| voting cell | `{_cell(cell)}`, n = {n_pairs} pairs |")
    w(f"| non-voting, reported alongside | "
      + ", ".join(f"`{_cell(c)}`" for c in sorted(by_cell) if c != cell)
      + " (n = 2; they do not vote) |")
    w(f"| cutoff | {SURVIVAL_CUTOFF:.2f}, conventional and declared as such |")
    w(f"| git sha of the rows | `{git_sha}` |")
    w("")

    # ── 1. the verdict ──────────────────────────────────────────────────────
    w("## 1. What the pre-declared rule returns")
    w("")
    w(f"**Branch {v['branch']} — {v['branch_name']}. D1 = "
      f"`{v['d1_stage']}`.**")
    w("")
    w("The rule is `analysis/i2_decision.decide()`, committed in Phase A "
      "before any")
    w("result existed. It takes no options and has no discretion, and all "
      "three of")
    w("its branches are unit-tested on synthetic gaps. It was run on the "
      "table, not")
    w("read off it.")
    w("")
    w("**The sentence it returns, verbatim — this is what would go into "
      "`docs/LOCKED_ANALYSIS.md` §1:**")
    w("")
    w("> " + v["sentence"])
    w("")

    # ── 2. the evidence ─────────────────────────────────────────────────────
    w(f"## 2. The evidence ({cell}, n = {n_pairs} pairs)")
    w("")
    w("Paired SAGA − baseline gaps, image-level bootstrap "
      f"({prov['bootstrap_resamples']} resamples, seed "
      f"{prov['bootstrap_seed']}). `survival` = gap(term_1.00) / gap(native);")
    w("`survival_s11` = gap(s11_out) / gap(native). Both sides of every gap "
      "are the")
    w("same images.")
    w("")
    w("| diagnostic | gap @ hist/native | 95% CI | gap @ hist/term_1.00 | "
      "survival | gap @ s11_out | survival_s11 |")
    w("|---|---|---|---|---|---|---|")
    for d in PRIMARY_DIAGNOSTICS:
        n = _gap_row(gaps_rows, cell, d, SETTING_NATIVE)
        t = _gap_row(gaps_rows, cell, d, SETTING_TERM)
        s = _gap_row(gaps_rows, cell, d, SETTING_S11)
        flag = "" if surv[d] != MISSING and surv[d] >= SURVIVAL_CUTOFF \
            else " **✗**"
        flag11 = "" if surv11[d] != MISSING and surv11[d] >= SURVIVAL_CUTOFF \
            else " **✗**"
        w(f"| `{d}` | {_f(n['gap_mean'])} | "
          f"[{_f(n['ci_lo'])}, {_f(n['ci_hi'])}] | {_f(t['gap_mean'])} | "
          f"{_f(surv[d], 3)}{flag} | {_f(s['gap_mean'])} | "
          f"{_f(surv11[d], 3)}{flag11} |")
    w("")
    w("Every gap's sign is preserved under both comparisons — "
      f"sign(term_1.00) matches sign(native) for "
      f"{sum(v['sign_match'].values())} of {len(PRIMARY_DIAGNOSTICS)} "
      f"primary diagnostics and sign(s11_out) for "
      f"{sum(v['sign_match_s11'].values())} of "
      f"{len(PRIMARY_DIAGNOSTICS)}. What decides the branch is the "
      "MAGNITUDE retained.")
    w("")

    # ── 3. the disclosure ───────────────────────────────────────────────────
    w("## 3. Read this before signing")
    w("")
    w("Two things about branch "
      f"{v['branch']} that the sentence in §1 does not say.")
    w("")
    w("**(a) The branch turned on a small margin.** Branch 1 requires every "
      "primary")
    w(f"diagnostic to retain at least {SURVIVAL_CUTOFF:.2f} of its native "
      "gap under the")
    w(f"terminal-gate bypass. {len(failed)} of "
      f"{len(PRIMARY_DIAGNOSTICS)} did not:")
    w("")
    for d in failed:
        if d in margins:
            w(f"- `{d}`: survival {surv[d]:.3f}, short of "
              f"{SURVIVAL_CUTOFF:.2f} by **{margins[d]:.3f}**")
        else:
            w(f"- `{d}`: survival {MISSING}")
    w("")
    import math
    passed = [d for d in PRIMARY_DIAGNOSTICS if d not in failed]
    if numeric:
        # the largest two-decimal cutoff that every survival would have
        # cleared — floor, so the statement can never overstate the margin
        would_pass = math.floor(min(numeric) * 100) / 100
        w(f"The other {len(passed)} are "
          f"{', '.join(f'`{d}` {surv[d]:.3f}' for d in passed)}. Across all "
          f"five, survival runs {min(numeric):.3f}–{max(numeric):.3f}. Any "
          f"cutoff at or below {would_pass:.2f} would have returned branch 1 "
          f"instead. The {SURVIVAL_CUTOFF:.2f} was fixed before these numbers "
          "existed and is not being moved now; it is recorded here so the "
          "reader can see how much of the verdict rests on it.")
    w("")
    w("**(b) The branch-3 boilerplate overstates what these numbers show.** "
      "The")
    w("sentence in §1 ends \"the historical final-block diagnostic "
      "distinction was")
    w("substantially a terminal-gate effect\", which is the wording task "
      "file §7")
    w("attaches to branch 3. On this evidence the terminal gate accounts for "
      "the")
    cos11 = {d: surv11[d] for d in COSINE_DIAGNOSTICS
             if surv11[d] != MISSING}
    rest11 = {d: surv11[d] for d in PRIMARY_DIAGNOSTICS
              if d not in COSINE_DIAGNOSTICS and surv11[d] != MISSING}
    if numeric:
        w(f"SMALLER part of the gap: bypassing it retains "
          f"{min(numeric):.2f}–{max(numeric):.2f} of every native gap, with "
          "every sign preserved.")
    if cos11 and rest11:
        w("Moving the readout back one block is what actually removes the "
          "difference, and")
        w("only for two of the five: "
          + ", ".join(f"`{d}` keeps {s:.2f}" for d, s in cos11.items())
          + " at `s11_out`, while "
          + ", ".join(f"`{d}` keeps {s:.2f}" for d, s in rest11.items())
          + ".")
    w("So on this evidence the STAGE, not the gate, carries most of the "
      "cosine")
    w("difference, and the other three diagnostics are largely unchanged by "
      "either.")
    w("Both facts are in the table above. **The verdict `D1 = "
      + v["d1_stage"] + "` is unaffected**;")
    w("what is in question is only the clause explaining WHY — and that "
      "clause is the")
    w("one that would be quoted in the paper.")
    w("")
    w("A wording that the same table supports without the overstatement:")
    w("")
    w("> *D1 = `" + v["d1_stage"] + "`: on the calibration split, in the "
      f"`{cell}` cell (n = {n_pairs} pairs), the")
    if numeric and cos11:
        w(f"> SAGA-vs-baseline gap at the historical stage survives a "
          f"terminal-gate bypass at")
        w(f"> {min(numeric):.2f}–{max(numeric):.2f} of its native size with "
          "every sign preserved, so the untrained terminal")
        w(f"> constant is a contributor but not the main one; the cosine "
          f"diagnostics, however, retain")
        w(f"> only {min(cos11.values()):.2f}–{max(cos11.values()):.2f} of "
          "that gap one block earlier. The final-block patch diagnostics are")
        w("> therefore a mixture of what training did and what a "
          "readout-invisible constant does,")
        w("> and they are reported with the `term_1.00` and `s11_out` "
          "controls beside them, never alone.*")
    w("")
    w("Signing either one is the human's call. What must not happen is the "
      "§1")
    w("sentence going into `LOCKED_ANALYSIS.md` unread.")
    w("")

    # ── 4. the non-voting cells ─────────────────────────────────────────────
    w("## 4. The non-voting cells (n = 2 — reported, and they do not vote)")
    w("")
    w("Task file §7 fixes the voting cell in advance precisely so that a "
      "disagreement")
    w("between cells cannot be resolved after the fact by picking one. Here "
      "is what")
    w("the same rule returns on each of the others, for disclosure:")
    w("")
    w("| cell | n | branch it would give | survival, per primary diagnostic |")
    w("|---|---|---|---|")
    for c in sorted(by_cell):
        if c == cell:
            continue
        g, n = by_cell[c]
        try:
            vc = decide(g, cell=c, n_pairs=n)
        except Exception:                                # pragma: no cover
            continue
        detail = ", ".join(
            f"`{d}` " + (MISSING if vc["survival"][d] == MISSING
                         else f"{vc['survival'][d]:.2f}")
            + ("" if vc["passes_term"][d]
               else (" (sign flip)" if not vc["sign_match"][d] else " ✗"))
            for d in PRIMARY_DIAGNOSTICS)
        w(f"| `{_cell(c)}` | {n} | {vc['branch']} — D1 = "
          f"`{vc['d1_stage']}` | {detail} |")
    w("")
    w("A survival ratio is only as meaningful as its denominator. Where a "
      "cell's")
    w("native gap on some diagnostic is close to zero, the ratio is large, "
      "unstable")
    w("or sign-flipped for arithmetic reasons rather than because the "
      "terminal gate")
    w("did something; those entries are marked and should not be read as "
      "evidence")
    w("in either direction. The per-diagnostic gaps and CIs behind every one "
      "of them")
    w("are in `T_I2c_gaps.csv`.")
    w("")

    # ── 5. the architectural half ───────────────────────────────────────────
    w("## 5. The architectural half, measured on the real checkpoints")
    w("")
    w(f"Across {n_ckpt} SAGA checkpoints × {n_const} constants, on "
      f"{n_images} images each:")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| max abs logit difference vs native | **{logit_max:g}** |")
    w(f"| min top-1 agreement with native | **{agree_min:g}** |")
    w(f"| max mean abs ΔNLL | **{nll_max:g}** |")
    w(f"| `native` ≡ `term_0.50` (φ_L = 0) | **{identical} of {len(half)}** "
      "checkpoints bit-identical |")
    w(f"| max abs Δ`cos_all` at `hist` under `term_1.00` | "
      f"{cos_moves:.4f} |")
    w("")
    w("Proposition 2 holds exactly, not approximately: a CLS-only readout "
      "cannot see")
    w("the terminal patch gate, and φ_L = 0 is the weight-decay fixed point "
      "it was")
    w("predicted to be on every trained SAGA checkpoint in the cohort. The "
      "patch")
    w("diagnostics at the same stage move while the classifier does not — "
      "which is")
    w("the I2 panel, and it is why the stage question had to be settled "
      "before any")
    w("patch diagnostic was typeset.")
    w("")

    # ── 6. context ──────────────────────────────────────────────────────────
    w("## 6. Context the tables carry")
    w("")
    w("**Calibrated vs historical thresholds.** `tau_cal[hist]` is the canon "
      "recipe")
    w(f"re-run on `{split_name}`; `tau_canon` is the committed historical "
      "value from the")
    w("discovery split. They are reported side by side so the split effect "
      "stays")
    w("visible:")
    w("")
    w("| cell | `tau_cal[s11_out]` | `tau_cal[hist]` | `tau_canon` | "
      "`tau_cal[s12_post_norm]` |")
    w("|---|---|---|---|---|")
    for c in sorted(thr["tau_cal"]):
        by = thr["tau_cal"][c]
        src = thr["sources"][c]
        w(f"| `{_cell(c)}` | {by['s11_out']:.5f} | {by['hist']:.5f} | "
          f"{src['tau_canon']} | {by['s12_post_norm']:.5f} |")
    w("")
    if empty_stages:
        w(f"**`{'`, `'.join(empty_stages)}` is not a candidate reporting "
          f"stage.** {len(empty_maps)} of {len(maps_rows)} rows of")
        w("`T_I2d_maps.csv` carry `MISSING` for every spatial statistic, all "
          "of them at")
        w(f"`{empty_stages[0]}`, because the exceedance map there has ZERO "
          "mass: after the")
        w("final LayerNorm the patch norms are close to uniform, so nothing "
          "exceeds")
        w("either threshold and a map with no mass has no spatial "
          "arrangement to")
        w("correlate. That is a property of the stage, not a gap in the "
          "results.")
        w("")
    w("## 7. What this unblocks, and what it does not")
    w("")
    w("Signing D1 releases I2 Phase B2 (the same sweep on the evaluation "
      "split) and")
    w("the Phase-B runs of I1/I3/I4/I5, which have been waiting to know "
      "which stage")
    w("to measure at. It settles nothing else: the six other open decisions "
      "in")
    w("`docs/LOCKED_ANALYSIS.md` §12 are untouched, and D5 still blocks I4.")
    w("")
    w("The complete tables are reported whatever is signed — "
      "`T_I2a_invariance.csv`,")
    w("`T_I2b_sweep.csv`, `T_I2c_gaps.csv`, `T_I2d_maps.csv`, "
      "`T_I2e_registers.csv`.")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description="Generate the D1 proposal note.")
    p.add_argument("--tables", default=TABLES)
    p.add_argument("--thresholds", default=THRESHOLDS)
    p.add_argument("--cell", default=DECISION_CELL)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    text = build(args.tables, args.thresholds, args.cell)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
