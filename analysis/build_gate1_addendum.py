#!/usr/bin/env python3
"""
analysis/build_gate1_addendum.py
================================
TASK-02B Phase A3 (+C2): generate results/notes/gate1_addendum.md.

Every number is read from files under results/ — nothing hand-typed. Facts
and tables only; no framing decisions (the human decides Gate 1).

The norm-scale section fills itself from results/legacy/diag/*_normstats.json
when those exist (Phase B output); until then it prints PENDING PHASE B.
Re-run this script after pulling Phase-B results to finalize the memo.

    python analysis/build_gate1_addendum.py \
        [--table results/tables/legacy_e2_corrected.csv]
        [--robustness results/tables/sink_robustness.csv]
        [--verdict results/tables/sink_robustness_verdict.csv]
        [--f6 results/figures_data/F6_legacy.csv]
        [--diag-dir results/legacy/diag]
        [--out results/notes/gate1_addendum.md]
"""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ARCH_NAME = {"vit_small": "ViT-S/16", "vit_base": "ViT-B/16"}
THRESHOLDS = ["sink_mad_k5", "sink_mu2s", "sink_mu3s", "sink_mu4s",
              "sink_mu5s", "sink_mu6s", "sink_fixed_thr"]
BEST_LAST_FLAG = 0.25


def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def read_csv(path, skip_comments=False):
    with open(path, newline="") as f:
        lines = [ln for ln in f if not (skip_comments and ln.startswith("#"))]
    return list(csv.DictReader(lines))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def sec_table(t):
    L = ["## 1. Corrected e2 table (from `results/tables/legacy_e2_corrected.csv`)",
         "",
         "Diagnostics from `last.pth`; per-checkpoint sha256 provenance is in "
         "the CSV. Δ = top1_last − same-cell baseline.",
         "",
         "| arch | recipe | variant | top1_best | top1_last | Δ | sink_mad_k5 "
         "| sink_mu3s | sink_fixed_thr | over_pair | over_nosink | "
         "nosink_excl | eff_rank | reg_norm_mean |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t:
        L.append(
            f"| {ARCH_NAME[r['arch']]} | {r['recipe']} | {r['variant']} "
            f"| {r['top1_best']} | {r['top1_last']} "
            f"| {r['delta_top1_last_vs_baseline'] or '—'} "
            f"| {fmt(float(r['sink_mad_k5']))} | {fmt(float(r['sink_mu3s']))} "
            f"| {fmt(float(r['sink_fixed_thr']))} "
            f"| {fmt(float(r['oversmooth_pairwise']))} "
            f"| {fmt(float(r['oversmooth_pairwise_nosink']))} "
            f"| {fmt(float(r['nosink_excluded_mean']))} "
            f"| {fmt(float(r['eff_rank']), 2)} "
            f"| {fmt(float(r['reg_norm_mean'])) if r['reg_norm_mean'] else '—'} |")
    L.append("")
    return L


N_PATCHES = 196
SATURATION_FRAC = 0.95


def sec_robustness(verdict_rows, cells, robustness_rows):
    L = ["## 2. Sink-threshold robustness (from "
         "`results/tables/sink_robustness{,_verdict}.csv`)",
         "",
         "Entries compare mean per-image sink counts at each threshold "
         "definition. CAVEAT: the fixed τ is per-arch (calibrated on that "
         "arch's nomix baseline) — fixed-threshold comparisons are valid "
         "within an arch, never across arches.",
         "",
         "Prediction each hypothesis makes for ViT-B/nomix: **H1 (metric "
         "artifact)** — SAGA beats baseline at the FIXED τ while losing at "
         "MAD (threshold shrinks with the compressed bulk); **H2 (real)** — "
         "SAGA does not beat baseline regardless of definition.",
         ""]
    header = "| comparison | threshold | " + " | ".join(
        f"{ARCH_NAME[a]}/{r}" for a, r in cells) + " |"
    L += [header, "|---|---|" + "---|" * len(cells)]
    for row in verdict_rows:
        L.append(f"| {row['comparison']} | {row['threshold']} | "
                 + " | ".join(row[f"{a}/{r}"] for a, r in cells) + " |")
    L.append("")
    def reading(comparison, label, lt, gt, eq):
        out = []
        for a, r in cells:
            buckets = {lt: [], gt: [], eq: [], "MISSING": []}
            for row in verdict_rows:
                if row["comparison"] != comparison:
                    continue
                mark = row[f"{a}/{r}"]
                key = mark if mark in buckets else "MISSING"
                buckets[key].append(row["threshold"])
            line = (f"- {ARCH_NAME[a]}/{r}: {label} < baseline under "
                    f"{', '.join(buckets[lt]) if buckets[lt] else 'none'}; "
                    f"{label} > baseline under "
                    f"{', '.join(buckets[gt]) if buckets[gt] else 'none'}")
            if buckets[eq]:
                line += f"; equal under {', '.join(buckets[eq])}"
            if buckets["MISSING"]:
                line += f"; MISSING under {', '.join(buckets['MISSING'])}"
            out.append(line + ".")
        return out

    L.append("Factual reading per cell (SAGA vs baseline):")
    L += reading("saga_vs_baseline", "SAGA", "S<B", "S>B", "S=B")
    L.append("")
    L.append("Registers vs baseline, same reading:")
    L += reading("registers_vs_baseline", "registers", "R<B", "R>B", "R=B")
    L.append("")

    # computed saturation note: when the per-arch tau sits below a model's
    # norm bulk, the fixed count approaches all 196 patch tokens and the
    # ordering carries no information. Spans are quoted PER ARCH — each
    # arch's counts share one tau; cross-arch spans would be incommensurable.
    fixed = [(r["arch"], r["recipe"], r["variant"], float(r["sink_fixed_thr"]))
             for r in robustness_rows]
    saturated = [x for x in fixed if x[3] >= SATURATION_FRAC * N_PATCHES]
    L.append("Saturation note (computed). Per-arch sink_fixed_thr spans at "
             "that arch's single τ (norm scales differ strongly across "
             "recipes/variants within each arch):")
    for arch in sorted({x[0] for x in fixed}):
        sub = [x for x in fixed if x[0] == arch]
        lo = min(sub, key=lambda x: x[3])
        hi = max(sub, key=lambda x: x[3])
        L.append(f"- {ARCH_NAME[arch]}: {fmt(lo[3])} ({lo[1]} {lo[2]}) to "
                 f"{fmt(hi[3])} ({hi[1]} {hi[2]}) of {N_PATCHES} patch tokens.")
    L.append(f"Counts at ≥{SATURATION_FRAC:.0%} of all tokens (τ below the "
             f"bulk of that model's norm distribution; the ordering there is "
             f"not informative):")
    for a, rec, v, val in saturated:
        L.append(f"- {ARCH_NAME[a]}/{rec} {v}: {fmt(val)} / {N_PATCHES}")
    if not saturated:
        L.append("- none")
    L.append("")
    return L


def sec_f6(f6_rows, cells):
    last_block = max(int(r["block"]) for r in f6_rows)
    by = {(r["arch"], r["recipe"], r["variant"], int(r["block"])): r
          for r in f6_rows}
    L = ["## 3. F6 relocation observations (from "
         "`results/figures_data/F6_legacy.csv`, final block "
         f"= block {last_block})",
         "",
         "The relocation question: does SAGA raise CLS norm/attention share "
         "relative to baseline — i.e. does outlier mass relocate to the "
         "ungated CLS token — and is the effect strongest where SAGA's "
         "patch-outlier count did not fall (ViT-B/nomix)?",
         "",
         "| cell | metric | baseline | registers | saga |",
         "|---|---|---|---|---|"]
    for a, rec in cells:
        for key, label in (("cls_norm_ratio", "CLS-norm ratio"),
                           ("cls_attn_share", "CLS attn share")):
            vals = {v: by[(a, rec, v, last_block)][key]
                    for v in ("baseline", "registers", "saga")}
            L.append(f"| {ARCH_NAME[a]}/{rec} | {label} (final block) | "
                     + " | ".join(fmt(float(vals[v]))
                                  for v in ("baseline", "registers", "saga"))
                     + " |")
    L.append("")
    for a, rec in cells:
        ratios = {v: float(by[(a, rec, v, last_block)]["cls_norm_ratio"])
                  for v in ("baseline", "saga")}
        shares = {v: float(by[(a, rec, v, last_block)]["cls_attn_share"])
                  for v in ("baseline", "saga")}
        L.append(f"- {ARCH_NAME[a]}/{rec}: final-block SAGA CLS-norm ratio "
                 f"{fmt(ratios['saga'])} vs baseline {fmt(ratios['baseline'])} "
                 f"({'higher' if ratios['saga'] > ratios['baseline'] else 'lower'}); "
                 f"CLS attn share {fmt(shares['saga'])} vs "
                 f"{fmt(shares['baseline'])} "
                 f"({'higher' if shares['saga'] > shares['baseline'] else 'lower'}).")
    L.append("")
    L.append("Draft figure: `results/figures/F6_draft.pdf`.")
    L.append("")
    return L


def sec_best_last(t):
    L = ["## 4. best.pth vs last.pth recap",
         ""]
    flagged = [r for r in t
               if r["top1_best_minus_last"] not in ("", "MISSING")
               and abs(float(r["top1_best_minus_last"])) > BEST_LAST_FLAG]
    for r in flagged:
        L.append(f"- {ARCH_NAME[r['arch']]}/{r['recipe']}/{r['variant']}: "
                 f"top1_best − top1_last = {float(r['top1_best_minus_last']):+.3f} "
                 f"(> {BEST_LAST_FLAG} flag).")
    L.append("")
    L.append("Note: the legacy resume path OVERWROTE history JSONs, so the "
             "training curves for these runs are partial (last segment only) "
             "— late-training instability behind these gaps cannot be "
             "inspected for the legacy runs.")
    L.append("")
    return L


def sec_normscale(diag_dir: Path, t):
    """C2: relative SAGA-vs-baseline change of median norm, MAD threshold,
    p99.9 and max — from Phase-B _normstats.json when present."""
    L = ["## 5. Norm-scale analysis (H1 vs H2, per cell)", ""]
    stats = {}
    for p in sorted(diag_dir.glob("e2_*_last_normstats.json")):
        parts = p.name.replace("_last_normstats.json", "").split("_")
        arch = f"{parts[1]}_{parts[2]}"
        stats[(arch, parts[3], parts[4])] = json.load(open(p))
    if not stats:
        L += ["**PENDING PHASE B** — requires "
              "`results/legacy/diag/*_normstats.json` "
              "(`python tools/summarize_norms.py` on the HPC, where the "
              "git-ignored `_norms.npz` live). Re-run "
              "`python analysis/build_gate1_addendum.py` after pulling.", ""]
        return L

    cells = sorted({(r["arch"], r["recipe"]) for r in t})
    L += ["Relative change, SAGA vs baseline (negative = SAGA smaller), from "
          "`results/legacy/diag/*_last_normstats.json`:",
          "",
          "| cell | median norm | MAD threshold (k=5) | p99.9 | max |",
          "|---|---|---|---|---|"]
    readings = []
    for a, rec in cells:
        s = stats.get((a, rec, "saga"))
        b = stats.get((a, rec, "baseline"))
        if not s or not b:
            L.append(f"| {ARCH_NAME[a]}/{rec} | MISSING | MISSING | MISSING "
                     f"| MISSING |")
            continue
        def rel(key):
            return (s[key] - b[key]) / b[key] * 100
        L.append(f"| {ARCH_NAME[a]}/{rec} | {rel('median_of_medians'):+.1f}% "
                 f"| {rel('mean_threshold_mad_k5'):+.1f}% "
                 f"| {rel('p999'):+.1f}% | {rel('max'):+.1f}% |")
        thr_shrink = rel("mean_threshold_mad_k5")
        tail_shrink = max(rel("p999"), rel("max"))   # least-shrunk extreme
        # The two signatures are NOT mutually exclusive; report each on its
        # own definition rather than forcing a binary:
        #   H1 signature: threshold shrank, and by more than the extremes
        #                 (MAD count can rise for scale reasons).
        #   H2 signature: extremes did not shrink.
        h1 = thr_shrink < 0 and thr_shrink < tail_shrink
        h2 = tail_shrink >= 0
        if h1 and h2:
            pattern = ("both signatures (threshold shrank while extremes "
                       "did not)")
        elif h1:
            pattern = "H1 (threshold shrank more than the extremes)"
        elif h2:
            pattern = "H2 (extremes did not shrink)"
        else:
            pattern = ("neither (extremes shrank at least as much as the "
                       "threshold)")
        readings.append(f"- {ARCH_NAME[a]}/{rec}: threshold "
                        f"{thr_shrink:+.1f}%, p99.9 {rel('p999'):+.1f}%, max "
                        f"{rel('max'):+.1f}% → pattern: {pattern}.")
    L.append("")
    L += readings
    L += ["", "Histogram draft: `results/figures/A1_norm_hist_draft.pdf`.", ""]
    return L


def sec_open_questions():
    return [
        "## 6. Open questions for the human",
        "",
        "1. Primary sink metric: keep per-image median+5·MAD (scale-"
        "sensitive; see §2 and §5), switch to the per-arch fixed τ, or "
        "report both?",
        "2. ViT-S/nomix registers improving oversmoothing (§1, and TASK-02 "
        "gate1_report §4): treat as single-repeat noise (no seed control "
        "existed) or investigate further before Gate 1?",
        "3. ViT-B/mixup best-vs-last gaps (§4): report last.pth only per the "
        "TASK-02 memo, or additionally show best.pth with a caveat?",
        "4. If the F6 relocation pattern (§3) is paper-relevant, should a "
        "later task extend diagnostics (e.g. CLS-norm histograms, per-head "
        "shares) beyond the legacy checkpoints?",
        "5. Does Gate 1 proceed on these single-repeat legacy numbers, or "
        "wait for the clean-protocol reruns?",
        "",
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate gate1_addendum.md.")
    parser.add_argument("--table",
                        default="results/tables/legacy_e2_corrected.csv")
    parser.add_argument("--robustness",
                        default="results/tables/sink_robustness.csv")
    parser.add_argument("--verdict",
                        default="results/tables/sink_robustness_verdict.csv")
    parser.add_argument("--f6", default="results/figures_data/F6_legacy.csv")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--out", default="results/notes/gate1_addendum.md")
    args = parser.parse_args()

    t = read_csv(args.table)
    robustness_rows = read_csv(args.robustness)
    verdict_rows = read_csv(args.verdict, skip_comments=True)
    f6_rows = read_csv(args.f6)
    cells = sorted({(r["arch"], r["recipe"]) for r in t})

    L = ["# Gate-1 addendum — sink robustness & relocation (TASK-02B)",
         "",
         f"Generated by `analysis/build_gate1_addendum.py`; git "
         f"`{git_sha()[:12]}`, {datetime.now(timezone.utc).date().isoformat()}. "
         "All numbers verbatim from files under `results/`. Facts only — "
         "framing and the Gate-1 decision are the human's.",
         ""]
    L += sec_table(t)
    L += sec_robustness(verdict_rows, cells, robustness_rows)
    L += sec_f6(f6_rows, cells)
    L += sec_best_last(t)
    L += sec_normscale(Path(args.diag_dir), t)
    L += sec_open_questions()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
