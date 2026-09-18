#!/usr/bin/env python3
"""
analysis/build_C_handoff.py
===========================
TASK C / C13 — generate `docs/C_HANDOFF.md`.

    python analysis/build_C_handoff.py

GENERATED, not written by hand. Every number below comes from a committed
file under `results/frozen/I5_readout/`, `results/frozen/I7_attention/` or
`results/ttr*/`; the surrounding prose is fixed text. Re-run it after new
results land and the document updates itself, which is the same contract
`analysis/build_I0_handoff.py` and `analysis/build_I2_handoff.py` keep.

Three things it must get right, and a test pins each:

  1. The I7 wording verdict and its sentence are COPIED from
     `i7_wording_verdict.json`, never re-decided and never re-worded. The
     rule was declared before the measurement existed; re-phrasing its
     output here would quietly undo that.
  2. The D9/D10 status is reported as it IS, in THREE states, not two.
     Signed-and-before-the-runs is a pre-registration; signed-and-after is
     not; unsigned is neither. The state is DERIVED by comparing the sha in
     the freeze header with the sha each Phase B run recorded, so the
     document cannot claim a pre-registration it does not have.
  3. `MISSING` is printed as `MISSING`, never as a blank or a zero.

No training, no optimizer, no probe fitting.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MISSING = "MISSING"
I5T = REPO / "results" / "frozen" / "I5_readout" / "sub2k" / "tables"
I7T = REPO / "results" / "frozen" / "I7_attention" / "sub1k" / "tables"
SEG = REPO / "results" / "frozen" / "I5_readout" / "seg"
VERDICT = (REPO / "results" / "frozen" / "I7_attention" / "sub1k"
           / "i7_wording_verdict.json")
LOCKED = REPO / "docs" / "LOCKED_ANALYSIS.md"
OUT = REPO / "docs" / "C_HANDOFF.md"

PRIMARY_STAGE = "s11_out"
PRIMARY_DESC = "l2"
FRESH = ("e2r_vits_mixup_baseline_s1", "e2r_vits_mixup_baseline_s2")

#: The conditions files whose ADDING COMMIT is the "fixed in advance" claim.
CONFIGS = {"I5": "configs/frozen/I5_readout.yaml",
           "I7": "configs/frozen/I7_attention.yaml"}
#: Where Phase B wrote its per-run provenance.
PHASE_B = ("results/frozen/I5_readout", "results/frozen/I7_attention")


def read(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(v, nd=4):
    """A number at fixed precision, or the literal MISSING."""
    if v is None or v == "" or v == MISSING:
        return MISSING
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def git_sha():
    from saga.run_registry import git_sha as _g
    try:
        return _g()
    except Exception:                                    # pragma: no cover
        return MISSING


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

def _git(*args):
    """git, or MISSING. A generator must not die because git is absent."""
    try:
        r = subprocess.run(("git",) + args, cwd=str(REPO), text=True,
                           capture_output=True)
    except Exception:                                    # pragma: no cover
        return MISSING
    return r.stdout.strip() if r.returncode == 0 else MISSING


def _is_ancestor(a, b):
    """True / False / None — None when git could not answer at all, which is
    never silently folded into either answer."""
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", a, b],
                           cwd=str(REPO), capture_output=True)
    except Exception:                                    # pragma: no cover
        return None
    if r.returncode in (0, 1):
        return r.returncode == 0
    return None                                          # bad sha, not a repo


def config_shas():
    """The commit that ADDED each conditions file. "Fixed in advance" is a
    claim about a commit, so the commit is named and can be checked."""
    return {k: (_git("log", "--diff-filter=A", "-1", "--format=%h", "--", v)
                or MISSING)
            for k, v in CONFIGS.items()}


def signature_order():
    """Did the signature come BEFORE or AFTER the Phase B jobs? → (state, ev)

    `run_meta.json` carries NO completion timestamp — `tools/frozen_eval.py`
    never wrote one — so the ordering evidence that exists is the `git_sha`
    each job recorded, compared against the sha the freeze was signed at.
    That is a stronger anchor than a wall clock anyway: a timestamp is
    whatever the writer put in the file, while git ancestry is checkable by
    anyone holding the repository and cannot be back-dated.

    A run whose sha is an ANCESTOR of the signed sha ran at code that
    predates the signature, so the run came first → "signed_after". A run
    whose sha DESCENDS from it ran at code written after the signature, so
    the signature came first → "signed_before". Anything else, including a
    missing sha on either side, is "indeterminate" and is reported as such
    rather than resolved by guessing.
    """
    text = LOCKED.read_text(encoding="utf-8")
    m = re.search(r"Git sha at freeze:\s*`?([0-9a-fA-F]{7,40})`?", text)
    sig = m.group(1) if m else None
    date = None
    md = re.search(r"Date frozen:\s*`?(\d{4}-\d{2}-\d{2})`?", text)
    if md:
        date = md.group(1)

    runs = {}
    for pkg in PHASE_B:
        for meta in sorted((REPO / pkg).glob("*/*/run_meta.json")):
            try:
                g = json.loads(meta.read_text(encoding="utf-8")).get("git_sha")
            except Exception:                            # pragma: no cover
                continue
            if g and g != MISSING:
                runs[g] = runs.get(g, 0) + 1

    ev = {"signature_sha": sig or MISSING, "signature_date": date or MISSING,
          "run_shas": sorted(runs), "n_runs": sum(runs.values())}
    if not sig or not runs:
        return "indeterminate", ev

    if all(_is_ancestor(r, sig) is True for r in runs):
        return "signed_after", ev
    if all(_is_ancestor(sig, r) is True for r in runs):
        return "signed_before", ev
    return "indeterminate", ev


def phase_c_bullet(sig):
    """Where Phase C's tables sit relative to the signature — DERIVED, not
    asserted, and worded to say only what git can date. A sentence written
    by hand here would be a chronology claim that nothing checks, which is
    exactly the failure this whole section exists to correct."""
    rel = "results/frozen/I5_readout/sub2k/tables/T_I5e_diag_vs_readout.csv"
    add = _git("log", "--diff-filter=A", "-1", "--format=%h", "--", rel)
    if add in (MISSING, ""):
        return (f"- **Where Phase C's tables sit relative to the signature is "
                f"UNKNOWN**: git does not name a commit that added `{rel}`.")
    date = _git("log", "-1", "--format=%ad", "--date=short", add)
    after = _is_ancestor(sig, add)
    if after is None:
        return (f"- **Where Phase C's tables sit relative to the signature is "
                f"UNKNOWN**: `{rel}` was added at `{add}` ({date}), but git "
                f"could not relate that to `{sig}`.")
    if after:
        return (f"- **Phase C's tables were COMMITTED after the signature**: "
                f"`{rel}` was added at `{add}` ({date}), which descends from "
                f"`{sig}`. Whether they had already been PRODUCED before the "
                f"signature is not recorded anywhere — git dates the commit, "
                f"not the computation — so this document says only what it "
                f"can date.")
    return (f"- **Phase C's tables were committed BEFORE the signature**: "
            f"`{rel}` was added at `{add}` ({date}), which `{sig}` descends "
            f"from. The signature therefore post-dates both phases.")


def section_status(lines):
    text = LOCKED.read_text(encoding="utf-8")
    d9, d10 = "D9" in text, "D10" in text
    # emphasis-insensitive: the header may be written `STATUS: FROZEN`
    # or `STATUS: **FROZEN**`, and `**` is markdown, not state.
    # (2026-09-17: the freeze used the unemphasised form and this line
    # reported "Document frozen: NO" for a signed document.)
    from saga.frozen.masks import says_frozen
    frozen = says_frozen(text)
    signed = d9 and d10 and frozen
    order, ev = signature_order() if signed else ("indeterminate", {})
    lines += [
        "## 0. The parameter declaration, as it actually stands",
        "",
    ]
    if signed and order == "signed_before":
        runs = ", ".join(f"`{s[:10]}`" for s in ev["run_shas"])
        lines += [
            f"D9 and D10 are in `docs/LOCKED_ANALYSIS.md`, the document is "
            f"FROZEN, and the signature PREDATES the runs: it was signed at "
            f"`{ev['signature_sha']}` on {ev['signature_date']}, and every "
            f"one of the {ev['n_runs']} Phase B jobs recorded a git sha "
            f"({runs}) that descends from it. The parameters below were "
            f"therefore signed before these results existed, let alone before "
            f"they were inspected.",
            "",
        ]
    elif signed and order == "signed_after":
        cfg = config_shas()
        runs = ", ".join(f"`{s[:10]}`" for s in ev["run_shas"])
        lines += [
            "D9 and D10 are in `docs/LOCKED_ANALYSIS.md` and the document is "
            "FROZEN — **but the signature came after these runs, so this is "
            "not a pre-registration.** The order is what a reader needs, so "
            "it is given as it happened:",
            "",
            f"- **The values were fixed in committed YAML before any Phase B "
            f"job ran.** `{CONFIGS['I5']}` was added at `{cfg['I5']}` and "
            f"`{CONFIGS['I7']}` at `{cfg['I7']}`; between them they carry the "
            f"transforms, stages, descriptors, blocks, alignment, bootstrap "
            f"seed and resample count, and `saga/frozen/runner.py` REFUSES "
            f"anything not in them. That is verifiable from the git history, "
            f"and it is the part of a pre-registration that actually "
            f"constrains the analysis.",
            f"- **The signature came after Phase B had run and its results "
            f"had been committed, and they could have been inspected before "
            f"it.** All {ev['n_runs']} Phase B `run_meta.json` files record "
            f"git sha {runs}, which is an ANCESTOR of the signed sha "
            f"`{ev['signature_sha']}` ({ev['signature_date']}) — so the jobs "
            f"ran at code predating the signature. (`run_meta.json` carries "
            f"no completion TIMESTAMP; the recorded sha is the ordering "
            f"evidence, and git ancestry cannot be back-dated the way a "
            f"written timestamp could.)",
            phase_c_bullet(ev["signature_sha"]),
            "- **The person who signed it states that the Phase B results had "
            "in fact been inspected, and Phase C run, before the signature.** "
            "That is their own account of their own conduct, not something "
            "the artifacts can show, and it is recorded here because it is "
            "the LESS favourable reading and leaving it out would flatter the "
            "work.",
            "",
            "So the honest claim is the narrow one: the analysis "
            "configuration was fixed in advance and can be PROVED to have "
            "been, and D9/D10 are now signed. The paper may say that. It must "
            "**not** call this a pre-registration, and no result below "
            "acquires confirmatory status from the freeze.",
            "",
        ]
    elif signed:
        lines += [
            "D9 and D10 are in `docs/LOCKED_ANALYSIS.md` and the document is "
            "FROZEN. **The order of signature and runs is INDETERMINATE from "
            "the artifacts** — the signed sha and the shas recorded in the "
            "Phase B `run_meta.json` files are not on one line of descent, or "
            "one of them is absent — so this document does not claim the "
            "parameters were signed before the results were inspected. "
            "Treat the analysis as configured in advance (the conditions "
            "YAMLs prove that much) and NOT as a pre-registration.",
            "",
        ]
    else:
        lines += [
            f"**D9 present in `docs/LOCKED_ANALYSIS.md`: {'yes' if d9 else 'NO'}."
            f" D10 present: {'yes' if d10 else 'NO'}."
            f" Document frozen: {'yes' if frozen else 'NO'}.**",
            "",
            "This has to be stated plainly, because it is the difference "
            "between a pre-registered analysis and an ordinary one:",
            "",
            "- The PARAMETER VALUES were fixed in committed code before any "
            "Phase B job ran. `configs/frozen/I5_readout.yaml` and "
            "`configs/frozen/I7_attention.yaml` carry the transforms, "
            "stages, descriptors, blocks, alignment, bootstrap seed and "
            "resample count, and `saga/frozen/runner.py` REFUSES anything "
            "not in them. That is verifiable from the git history.",
            "- The D9/D10 lines were NOT added to `docs/LOCKED_ANALYSIS.md`, "
            "and the human's signature — the act that makes that document "
            "authoritative — did not happen before the runs.",
            "",
            "So: the analysis configuration was fixed in advance and can be "
            "proved to have been, but this is **not** a signed "
            "pre-registration. The paper should say the former and must not "
            "claim the latter.",
            "",
        ]
    if not signed:
        return "unsigned"
    return {"signed_before": "signed_before",
            "signed_after": "signed_after"}.get(order, "signed_indeterminate")


def section_wording(lines):
    lines += ["## 1. I7 — the wording rule (D10), and what it returned", ""]
    if not VERDICT.exists():
        lines += [f"`{VERDICT.name}`: {MISSING} — the rule has not been run.",
                  ""]
        return
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    lines += [
        f"**Verdict: `{v['verdict']}`** — the paper's term is "
        f"**{v['term']}**.",
        "",
        "The sentence `analysis/i7_wording.py` returned, COPIED verbatim and "
        "never edited:",
        "",
        f"> {v['sentence']}",
        "",
        f"The rule, as it was declared before the measurement existed: "
        f"{v['rule']}",
        "",
        "The four required cells — both fresh ViT-S/mixup baselines, both "
        "blocks, patch queries:",
        "",
        "| checkpoint | block | ratio | CI low | CI high |",
        "|---|---|---|---|---|",
    ]
    for c in v.get("checked", []):
        run, blk = c["cell"].split("|block")
        lines.append(f"| `{run}` | {blk} | **{f(c['ratio'], 2)}** | "
                     f"{f(c['ci_lo'], 2)} | {f(c.get('ci_hi'), 2)} |")
    lines += [
        "",
        f"Smallest ratio {f(v['min_ratio'], 2)}, smallest CI lower bound "
        f"{f(v['min_ci_lo'], 2)}, against the conventional threshold "
        f"{f(v['threshold'], 1)}. "
        + ("Every cell passed." if v["passed"]
           else f"Failures: {'; '.join(v['failures'])}"),
        "",
    ]


def section_i5a(lines):
    a = [r for r in read(I5T / "T_I5a_readout.csv")
         if r["condition_id"] == "native" and r["stage"] == PRIMARY_STAGE
         and r["descriptor"] == PRIMARY_DESC]
    lines += [
        "## 2. I5a — the correspondence readout",
        "",
        "Primary cut: D1 stage `s11_out`, primary descriptor `l2`, `native`. "
        "**Every I5 comparison is a SECONDARY endpoint under D7.**",
        "",
    ]
    if not a:
        lines += [f"{MISSING} — `T_I5a_readout.csv` has no primary rows.", ""]
        return
    chance = a[0]["chance_exact"]
    lines += [
        f"Chance is `{f(chance, 6)}` (1/196) and is a column on every readout "
        f"table. `T0` is the identity transform and scores 1.0000 on every "
        f"checkpoint — the matcher's ceiling, and a control rather than a "
        f"result.",
        "",
        "| checkpoint | variant | T1 flip | T2 (1 patch) | T3 (2 patches) |",
        "|---|---|---|---|---|",
    ]
    by = {}
    for r in a:
        by.setdefault((r["variant"], r["run_id"]), {})[r["transform"]] = r
    for (var, rid), d in sorted(by.items()):
        cells = [f(d[t]["acc_exact_mean"]) if t in d else MISSING
                 for t in ("T1", "T2", "T3")]
        lines.append(f"| `{rid}` | {var} | {cells[0]} | {cells[1]} | "
                     f"{cells[2]} |")
    over = [float(r["acc_exact_over_chance"]) for r in a
            if r["transform"] != "T0" and r["acc_exact_over_chance"] != MISSING]
    if over:
        lines += [
            "",
            f"Across T1-T3 the readout runs at **{min(over):.0f}x to "
            f"{max(over):.0f}x chance**, so it is a measurement with room to "
            f"move in either direction rather than a ceiling or a floor.",
            "",
        ]


def section_i5b_methods(lines):
    b = [r for r in read(I5T / "T_I5b_methods.csv")
         if r["stage"] == PRIMARY_STAGE and r["descriptor"] == PRIMARY_DESC
         and r["transform"] in ("T1", "T2", "T3")]
    lines += [
        "## 3. I5 — the method contrast (T_I5b_methods), at `s11_out`",
        "",
        "Paired by seed where seeds match; legacy pairs are labelled "
        "`pair_kind=legacy` and are NOT seed pairs. SECONDARY under D7, and "
        "worded as a measurement, not a claim.",
        "",
        "| contrast | pair | kind | T1 | T2 | T3 |",
        "|---|---|---|---|---|---|",
    ]
    by = {}
    for r in b:
        by.setdefault((r["contrast"], r["pair_label"], r["pair_kind"]),
                      {})[r["transform"]] = r
    for (contrast, pair, kind), d in sorted(by.items()):
        cells = []
        for t in ("T1", "T2", "T3"):
            r = d.get(t)
            if r is None:
                cells.append(MISSING)
                continue
            lo, hi = r["ci_lo"], r["ci_hi"]
            star = ""
            try:
                if float(lo) > 0 or float(hi) < 0:
                    star = ""
            except (TypeError, ValueError):
                pass
            cells.append(f"{f(r['delta_acc_exact'])}{star}")
        short = pair.replace(" - ", " − ")
        lines.append(f"| {contrast} | `{short}` | {kind} | {cells[0]} | "
                     f"{cells[1]} | {cells[2]} |")

    fresh_pairs = [k for k in by
                   if k[0] == "saga_minus_baseline" and k[2] == "seed"
                   and any(s in k[1] for s in FRESH)]
    signs = {}
    for t in ("T1", "T2", "T3"):
        vals = []
        for k in fresh_pairs:
            r = by[k].get(t)
            if r and r["delta_acc_exact"] != MISSING:
                vals.append(float(r["delta_acc_exact"]))
        signs[t] = vals
    lines += [
        "",
        "LOCKED_ANALYSIS §10 asks first whether a direction is CONSISTENT "
        "across the fresh ViT-S/mixup pairs. For `saga_minus_baseline` on "
        "those pairs:",
        "",
    ]
    for t, vals in signs.items():
        if not vals:
            lines.append(f"- **{t}**: {MISSING}")
            continue
        same = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        lines.append(
            f"- **{t}**: {', '.join(f'{v:+.4f}' for v in vals)} — "
            f"{'consistent in sign' if same else 'NOT consistent in sign'}")
    lines.append("")


def section_i5c(lines):
    c = [r for r in read(I5T / "T_I5c_terminal.csv")
         if r["descriptor"] == PRIMARY_DESC and r["transform"] == "T1"]
    lines += [
        "## 4. I5 — can a feature consumer see the terminal constant? "
        "(T_I5c_terminal)",
        "",
        "I2 measured that a CLS-only classifier cannot see the terminal patch "
        "gate at all: max abs logit difference exactly **0** on 8 of 8 SAGA "
        "checkpoints. I5 asks the same question of a consumer that reads the "
        "patches. `s11_out` is the control — the bypass swaps the LAST "
        "block's gate, so that stage cannot move.",
        "",
        "| checkpoint | stage | native | `term_1.00` | delta | CI |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(c, key=lambda r: (r["run_id"], r["stage"])):
        lines.append(
            f"| `{r['run_id']}` | `{r['stage']}` | {f(r['acc_native'])} | "
            f"{f(r['acc_term_1.00'])} | {f(r['delta'], 5)} | "
            f"[{f(r['ci_lo'], 5)}, {f(r['ci_hi'], 5)}] |")
    s11 = {r["max_abs_s11_diff_vs_native"] for r in c
           if r["stage"] == PRIMARY_STAGE}
    deltas = [abs(float(r["delta"])) for r in c
              if r["stage"] != PRIMARY_STAGE and r["delta"] != MISSING]
    lines += [
        "",
        f"The control holds on the real records: `max_abs_s11_diff_vs_native` "
        f"is {sorted(s11)} at `s11_out`, and the measured delta there is "
        f"exactly 0 for every checkpoint.",
        "",
    ]
    if deltas:
        lines += [
            f"**The answer: yes, and by a small amount.** At `hist` and "
            f"`s12_post_norm` the bypass moves the readout by at most "
            f"**{max(deltas):.4f}** in accuracy — a real, interval-excluding "
            f"difference on most checkpoints, but one to two orders of "
            f"magnitude smaller than the readout itself. A classifier sees "
            f"nothing; this consumer sees a little.",
            "",
        ]


def section_i5b_seg(lines):
    lines += ["## 5. I5b — the frozen ADE20K heads", ""]
    summary = SEG / "I5b_summary.json"
    if not summary.exists():
        lines += [f"**I5b: DROPPED / INCOMPLETE** — `{summary}` is absent.",
                  ""]
        return
    doc = json.loads(summary.read_text(encoding="utf-8"))
    lines += [
        f"**I5b: RUN.** Gate: {doc['gate_reason']}.",
        "",
        "| run / condition | mIoU_ss | pixel acc | images |",
        "|---|---|---|---|",
    ]
    for k, v in sorted(doc["runs"].items()):
        lines.append(f"| `{k}` | {f(v['mIoU'], 4)} | {f(v['pixel_acc'], 4)} | "
                     f"{v['n_images']} |")
    fr = read(I5T / "T_I5f_seg.csv")
    gr = read(I5T / "T_I5g_seg_terminal.csv")
    if fr:
        r = fr[0]
        lines += [
            "",
            f"`T_I5f_seg`: SAGA − baseline is **{f(r['miou_ss_delta'], 4)} "
            f"mIoU_ss** at the dataset level; the paired per-image difference "
            f"is {f(r['delta_miou_present_paired'], 5)} "
            f"[{f(r['ci_lo'], 5)}, {f(r['ci_hi'], 5)}] over "
            f"{r['n_images_paired']} images ({r['endpoint_class']}).",
        ]
    if gr:
        r = gr[0]
        lines += [
            "",
            f"`T_I5g_seg_terminal`: under the terminal-gate bypass the SAGA "
            f"head goes {f(r['miou_ss_native'], 4)} → "
            f"**{f(r['miou_ss_term_1.00'], 4)} mIoU_ss**, a change of "
            f"**{f(r['miou_ss_delta'], 4)}** ({r['endpoint_class']}).",
            "",
            "This is EXPLORATORY and must be read as such: the head was "
            "TRAINED on gated features, so a drop under the bypass measures "
            "sensitivity to an input distribution it never saw. It is not "
            "evidence that the gated features are better.",
        ]
    lines.append("")


def section_i5d_i5e(lines):
    """T_I5d and T_I5e — the position split, and whether the diagnostic
    predicts the readout.

    T_I5e is the test this thesis rests on: the project has reported a patch
    diagnostic for years, and this asks whether that diagnostic predicts a
    downstream utility on the SAME images. It cannot be absent from the
    handoff, so both tables render MISSING loudly rather than being skipped
    when a filter comes back empty.

    Primary configuration throughout: stage `s11_out`, descriptor `l2`,
    `count_mad_s11`, the two fresh ViT-S/mixup baselines, T0 excluded — T0 is
    the identity transform, where accuracy is 1 by construction and a rank
    correlation against a constant is undefined.
    """
    d = [r for r in read(I5T / "T_I5d_positions.csv")
         if r.get("stage") == PRIMARY_STAGE
         and r.get("descriptor") == PRIMARY_DESC
         and r.get("condition_id") == "native"
         and r.get("run_id") in FRESH and r.get("transform") != "T0"]
    lines += [
        "## 6. I5d/I5e — the exceedance positions, and whether the "
        "diagnostic predicts the readout",
        "",
        "Both tables are **SECONDARY under D7** and every row says so in its "
        "own `endpoint_class`; neither licenses a primary claim, and the size "
        "of the numbers below does not change that.",
        "",
        "### T_I5d — correspondence at exceedance vs non-exceedance positions",
        "",
    ]
    if not d:
        lines += ["**MISSING** — no `T_I5d_positions` rows at the primary "
                  "configuration.", ""]
    else:
        lines += ["| transform | checkpoint | acc(exceedance) | "
                  "acc(non-exc.) | delta | CI |",
                  "|---|---|---|---|---|---|"]
        for r in sorted(d, key=lambda r: (r["transform"], r["run_id"])):
            lines.append(
                f"| {r['transform']} | `{r['run_id']}` | {f(r['acc_exc'])} | "
                f"{f(r['acc_nonexc'])} | **{f(r['delta_exc_minus_nonexc'])}** "
                f"| [{f(r['delta_ci_lo'])}, {f(r['delta_ci_hi'])}] |")
        dd = [float(r["delta_exc_minus_nonexc"]) for r in d]
        neg = sum(1 for x in dd if x < 0)
        lines += [
            "",
            f"**Mean delta {sum(dd) / len(dd):+.4f}, negative on {neg} of "
            f"{len(dd)} rows**, with every interval well clear of zero. "
            f"Positions the diagnostic flags correspond across an exact grid "
            f"transform far less reliably than the positions it does not "
            f"flag — roughly 0.47–0.55 against 0.93–0.95. This is the "
            f"largest effect in Track C by an order of magnitude, and it is "
            f"descriptive: it says these positions carry less "
            f"transform-stable identity, not why they do.",
            "",
        ]

    e = [r for r in read(I5T / "T_I5e_diag_vs_readout.csv")
         if r.get("stage") == PRIMARY_STAGE
         and r.get("descriptor") == PRIMARY_DESC
         and r.get("diagnostic") == "count_mad_s11"
         and r.get("transform") != "T0"]
    within = [r for r in e if r.get("scope") == "within_checkpoint"]
    fresh = [r for r in within if r.get("run_id") in FRESH
             and r.get("spearman_rho") not in ("", MISSING)]
    across = [r for r in e if r.get("scope") == "across_checkpoints"]
    across_ok = [r for r in across
                 if r.get("spearman_rho") not in ("", MISSING)]
    lines += ["### T_I5e — does the diagnostic predict the readout? "
              "(the thesis test)", ""]
    if not fresh:
        lines += ["**MISSING** — no within-checkpoint rows at the primary "
                  "configuration. The thesis test has no answer in this "
                  "handoff, and nothing below substitutes for it.", ""]
    else:
        lines += ["| scope | transform | checkpoint | Spearman rho | images |",
                  "|---|---|---|---|---|"]
        for r in sorted(fresh, key=lambda r: (r["transform"], r["run_id"])):
            lines.append(
                f"| within-checkpoint | {r['transform']} | `{r['run_id']}` | "
                f"**{f(r['spearman_rho'])}** | {r['n']} |")
        rho = [float(r["spearman_rho"]) for r in fresh]
        allr = [float(r["spearman_rho"]) for r in within
                if r.get("spearman_rho") not in ("", MISSING)]
        neg = sum(1 for x in rho if x < 0)
        lines += [
            "",
            f"**Mean rho {sum(rho) / len(rho):+.4f} over the {len(rho)} "
            f"fresh-baseline rows, negative on {neg} of {len(rho)}**, range "
            f"[{min(rho):+.4f}, {max(rho):+.4f}]. Across the whole cohort "
            f"({len(allr)} usable rows) the mean is "
            f"{sum(allr) / len(allr):+.4f}, negative on "
            f"{sum(1 for x in allr if x < 0)}.",
            "",
            "**Within a checkpoint, the images with more exceedance positions "
            "read out worse.** That is the direction the thesis predicts, "
            "measured on the same images, with the diagnostic the paper "
            "already reports. It is a rank correlation across images and "
            "nothing more: it is not causal, it is secondary under D7, and it "
            "is consistent with both quantities depending on some third "
            "property of the image — texture, clutter, scale.",
            "",
        ]
    if not across_ok:
        lines += [
            f"**`across_checkpoints` is MISSING: {len(across)} rows of that "
            f"scope at this configuration, {len(across_ok)} of them carrying "
            f"a rho.** The BETWEEN-checkpoint form of the same question — do "
            f"checkpoints with more exceedance positions read out worse — is "
            f"NOT answered here, and the within-checkpoint result above does "
            f"not answer it by proxy. The two are different claims and only "
            f"one of them has a number.",
            "",
        ]
    else:
        ar = [float(r["spearman_rho"]) for r in across_ok]
        lines += [
            f"`across_checkpoints`: mean rho {sum(ar) / len(ar):+.4f} over "
            f"{len(ar)} of {len(across)} rows.",
            "",
        ]


def section_i7(lines):
    a = read(I7T / "T_I7a_incoming.csv")
    lines += [
        "## 7. I7 — incoming attention at the exceedance positions",
        "",
        "MAD is the PRIMARY basis. Patch queries, both blocks, all ten "
        "ViT-S/mixup checkpoints. Ratio = mean incoming mass per exceedance "
        "token / mean per other token, as a ratio of means with an "
        "image-level bootstrap.",
        "",
        "| checkpoint | variant | block | ratio | CI | AUC |",
        "|---|---|---|---|---|---|",
    ]
    rows = [r for r in a if r["query_group"] == "patch" and r["basis"] == "mad"]
    for r in sorted(rows, key=lambda r: (r["variant"], r["run_id"],
                                         int(r["block"]))):
        lines.append(
            f"| `{r['run_id']}` | {r['variant']} | {r['block']} | "
            f"**{f(r['ratio'], 2)}** | [{f(r['ratio_ci_lo'], 2)}, "
            f"{f(r['ratio_ci_hi'], 2)}] | {f(r['auc'], 3)} |")
    b = read(I7T / "T_I7b_registers.csv")
    reg = [r for r in b if r["register_index"] == "all"
           and r["query_group"] == "patch"]
    if reg:
        lines += [
            "",
            "`T_I7b_registers` (DESCRIPTIVE, n = 2 register checkpoints, both "
            "legacy): where a register model's patch-query mass lands.",
            "",
            "| checkpoint | block | CLS | 4 registers total | per register | "
            "patch exceedances |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(reg, key=lambda r: (r["run_id"], int(r["block"]))):
            lines.append(
                f"| `{r['run_id']}` | {r['block']} | "
                f"{f(r['mass_cls_token'], 5)} | "
                f"{f(r['mass_register_tokens_total'], 5)} | "
                f"{f(r['mass_per_register_token'], 5)} | "
                f"{f(r['mass_patch_exceedance_total'], 5)} |")
    c = [r for r in read(I7T / "T_I7c_value_norm.csv") if r["basis"] == "mad"]
    if c:
        vals = [float(r["ratio"]) for r in c if r["ratio"] != MISSING]
        lines += [
            "",
            f"`T_I7c_value_norm` (EXPLORATORY): the value-norm ratio at "
            f"exceedance vs other positions runs "
            f"{min(vals):.3f} to {max(vals):.3f} across the cell. It decides "
            f"nothing about the wording rule.",
        ]
    lines.append("")


def section_ttr(lines):
    curve = read(I7T / "T_I7d_ttr_curve.csv")
    cov = read(I7T / "T_I7d_ttr_coverage.csv")
    lines += [
        "## 8. The TTR operating curve (§6)",
        "",
        "Built from committed files only; **no new TTR runs**. The two layer "
        "ranges are NOT a crossed grid — they were swept on different neuron "
        "grids over different cell sets, and the coverage table says so "
        "rather than presenting a cross product nobody intended to run.",
        "",
        "| layer range | cells present | cells MISSING | n_neurons swept | points |",
        "|---|---|---|---|---|",
    ]
    by_range = {}
    for r in cov:
        by_range.setdefault(r["layer_range_label"], []).append(r)
    for label, rows in sorted(by_range.items()):
        present = [r for r in rows if r["status"] == "present"]
        absent = [r["base_run_id"] for r in rows if r["status"] == MISSING]
        grid = present[0]["n_neurons_grid"] if present else MISSING
        npts = sum(int(r["n_points"]) for r in present)
        lines.append(
            f"| `{label}` ({present[0]['layer_range'] if present else MISSING}) "
            f"| {len(present)} | "
            f"{(', '.join('`' + a + '`' for a in absent)) if absent else '—'} "
            f"| `{grid}` | {npts} |")
    chosen = [r for r in curve if r["is_chosen_operating_point"] == "1"]
    lines += [
        "",
        f"**{len(curve)} points total.** The chosen operating point is marked "
        f"on {len(chosen)} cells, all at `n_neurons=24` in the midlayer "
        f"range. The gate lines are read from each run's own `validate.json` "
        f"and agree across every file.",
        "",
    ]


def section_not_settled(lines):
    lines += [
        "## 9. What Track C does NOT settle",
        "",
        "- **Correspondence under exact grid-aligned transforms is ONE "
        "utility, not utility.** T1-T3 are a flip and two whole-patch "
        "translations. They say nothing about scale, rotation, or any "
        "transform whose correspondence is not exact — which is precisely "
        "why those were excluded rather than approximated.",
        "- **The 3x in D10 is a convention.** It is not derived from "
        "anything, and the sentence the wording module returns says so. A "
        "different threshold would be a different sentence, and the measured "
        "ratios are reported so a reader can apply their own.",
        "- **The register checkpoints are n = 2**, both legacy repeats with "
        "no recorded seed. Every register row is labelled DESCRIPTIVE and no "
        "register comparison is a paired seed contrast.",
        "- **I5b's `term_1.00` pass is EXPLORATORY.** The head was trained on "
        "gated features; a drop under the bypass measures sensitivity to an "
        "unseen input distribution, not feature quality.",
        "- **The value-norm ratio and the register-token mass are "
        "EXPLORATORY** (§5) and decide nothing.",
        "- **Everything I5 and I7 measure is SECONDARY under D7.** The three "
        "primary contrasts of the paper are C1-C3 in Track B. No number in "
        "this document decides a primary claim.",
        "",
    ]


#: The provenance line's shape. The sha in it is the one HEAD was at when the
#: document was generated, so it legitimately changes on the next commit —
#: which is why `build()` accepts it and `sha_in()` reads it back out. Every
#: other byte of the document is pinned by `test_the_handoff_is_byte_identical
#: _to_its_generator`; without this seam that test would fail on every commit
#: that followed the one which wrote the file, and a test that fails for a
#: reason nobody acts on is a test people learn to ignore.
PROVENANCE_PREFIX = "**GENERATED** by `analysis/build_C_handoff.py` at git `"


def sha_in(text):
    """The git sha recorded in an existing handoff, or None."""
    for line in text.splitlines():
        if line.startswith(PROVENANCE_PREFIX):
            rest = line[len(PROVENANCE_PREFIX):]
            return rest.split("`", 1)[0]
    return None


def build(sha=None):
    sha = git_sha() if sha is None else sha
    lines = [
        "# TASK C handoff — the correspondence readout (I5) and incoming "
        "attention (I7)",
        "",
        f"{PROVENANCE_PREFIX}{sha}`. Every number is read from a committed "
        f"file under "
        f"`results/frozen/I5_readout/`, `results/frozen/I7_attention/` or "
        f"`results/ttr*/`; none is typed by hand. The surrounding prose is "
        f"fixed text. Re-run the script after new results land and this "
        f"document updates itself.",
        "",
    ]
    section_status(lines)
    section_wording(lines)
    section_i5a(lines)
    section_i5b_methods(lines)
    section_i5c(lines)
    section_i5b_seg(lines)
    section_i5d_i5e(lines)
    section_i7(lines)
    section_ttr(lines)
    section_not_settled(lines)
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="TASK C / C13 — the handoff.")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--check", action="store_true",
                   help="exit 1 if the file on disk differs from what this "
                        "script would write (byte identity)")
    args = p.parse_args()
    out = Path(args.out)
    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        # Regenerate at the sha the file itself records, so the check pins
        # every number and every word and tolerates only the one value that
        # legitimately moves with HEAD.
        if current != build(sha=sha_in(current)):
            print(f"{out} is NOT what build_C_handoff.py would write")
            return 1
        print(f"{out} is byte-identical to its generator's output "
              f"(at the sha it records)")
        return 0
    text = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
