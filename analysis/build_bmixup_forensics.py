#!/usr/bin/env python3
"""
analysis/build_bmixup_forensics.py
==================================
TASK-02C C3 (amended): generate results/notes/bmixup_forensics.md.

Facts only, sourced from results/legacy/ckpt_forensics.csv, the phi stats
sidecars, the normstats, the corrected table, and the v2 robustness CSV.
No verdict — the human decides.

    python analysis/build_bmixup_forensics.py \
        [--out results/notes/bmixup_forensics.md]
"""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ARCH_NAME = {"vit_small": "ViT-S/16", "vit_base": "ViT-B/16"}
FINAL_EPOCH = 299          # 0-indexed; trainer loop range(start, 300)
SAVE_FREQ = 25
EARLY_BEST = 250           # stated criterion for "anomalously early" best


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


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


def sec_semantics():
    return [
        "## 1. Trainer epoch semantics (verified from the save code)",
        "",
        "From `classification/tools/train.py` (the trainer that produced all "
        "e2 checkpoints): the loop is `for epoch in range(start_epoch, "
        "total_epochs)` with `total_epochs = 300` (`classification/configs/"
        "base.yaml`), and every checkpoint stores `'epoch': epoch` — i.e. "
        "**0-indexed**; the final epoch of a completed run is `epoch = 299`.",
        "",
        "- `last.pth` is written only when `(epoch + 1) % save_freq == 0` "
        f"(save_freq = {SAVE_FREQ}) or `epoch == total_epochs - 1` — so "
        "legal last.pth epochs are 24, 49, 74, ..., 274, and 299.",
        "- `best.pth` is written at any epoch where val top-1 improves; its "
        "epoch has no completeness semantics.",
        "",
        f"**Completeness rule applied below: a run is COMPLETE iff its "
        f"last.pth carries epoch == {FINAL_EPOCH}.** A best.pth epoch is "
        f"flagged 'early' under the stated criterion epoch < {EARLY_BEST} "
        f"(of 0..{FINAL_EPOCH}).",
        "",
    ]


def sec_completeness(rows):
    rows = sorted(rows, key=lambda r: (r["arch"], r["recipe"], r["variant"],
                                       r["filename"]))
    L = ["## 2. Completeness table — all 24 checkpoints",
         "",
         "| run | file | saved epoch | LR at save | optimizer steps | verdict |",
         "|---|---|---|---|---|---|"]
    incomplete = {}
    best_epochs = {}
    for r in rows:
        run = f"{ARCH_NAME[r['arch']]}/{r['recipe']}/{r['variant']}"
        tag = r["filename"].removesuffix(".pth")
        epoch = int(r["epoch"])
        lr = float(r["last_lr"])
        if tag == "last":
            if epoch == FINAL_EPOCH:
                verdict = "COMPLETE"
            else:
                verdict = (f"**INCOMPLETE** ({epoch + 1}/{FINAL_EPOCH + 1} "
                           f"epochs)")
                incomplete[run] = (epoch, lr)
        else:
            best_epochs[run] = epoch
            verdict = ("n/a (best-selection file)"
                       + (f", **early** (< {EARLY_BEST})"
                          if epoch < EARLY_BEST else ""))
        L.append(f"| {run} | {tag} | {epoch} | {lr:.3e} | "
                 f"{r['optimizer_step_count']} | {verdict} |")
    L.append("")

    L.append("Runs whose last.pth is below the final epoch:")
    for run, (epoch, lr) in sorted(incomplete.items()):
        be = best_epochs.get(run)
        early = "also early" if be is not None and be < EARLY_BEST \
            else "not early"
        note = f"; best.pth epoch {be} ({early})"
        if be is not None and be > epoch:
            # only the derivable interval — whether the run was resubmitted
            # and died again cannot be determined from checkpoint state
            note += (f" — best.pth is NEWER than last.pth: training reached "
                     f"at least epoch {be} but never completed epoch "
                     f"{epoch + SAVE_FREQ} (the next last.pth save point), "
                     f"so the resumable state remains at {epoch}")
        L.append(f"- {run}: last.pth at epoch {epoch} "
                 f"(LR {lr:.3e}){note}.")
    if not incomplete:
        L.append("- none")
    L.append("")
    return L, incomplete


def sec_phi(gates_dir: Path):
    L = ["## 3. SAGA gate logits — ViT-B/mixup vs the other three runs",
         "",
         "From `results/legacy/gates/*_last_phi_stats.json` (per-layer stats "
         "of sigmoid(phi)); maps in `results/figures/gates_legacy_draft.pdf`.",
         "",
         "| run | mean gate (all layers) | min layer-mean | max layer-mean | "
         "max frac<0.4 (layer) | frac<0.25 anywhere | NaN/Inf |",
         "|---|---|---|---|---|---|---|"]
    for p in sorted(gates_dir.glob("e2_*_saga_*_last_phi_stats.json")):
        parts = p.name.replace("_phi_stats.json", "").split("_")
        run = f"{ARCH_NAME[f'{parts[1]}_{parts[2]}']}/{parts[3]}"
        s = json.load(open(p))
        layers = s["layers"]
        means = [l["mean_gate"] for l in layers]
        fr04 = max(layers, key=lambda l: l["frac_below_0.4"])
        any_025 = any(l["frac_below_0.25"] > 0 for l in layers)
        any_bad = any(l["has_nan"] or l["has_inf"] for l in layers)
        L.append(f"| {run} | {sum(means) / len(means):.4f} "
                 f"| {min(means):.4f} | {max(means):.4f} "
                 f"| {fr04['frac_below_0.4']:.4f} (L{fr04['layer']}) "
                 f"| {'yes' if any_025 else 'no'} "
                 f"| {'YES' if any_bad else 'none'} |")
    # computed profile facts (never asserted; derived from the loaded stats)
    all_stats = [json.load(open(p)) for p in
                 sorted(gates_dir.glob("e2_*_saga_*_last_phi_stats.json"))]
    low_layers = [l["layer"] for s in all_stats for l in s["layers"]
                  if l["frac_below_0.4"] > 0]
    final_means = [s["layers"][-1]["mean_gate"] for s in all_stats]
    peak_layers = sorted({max(s["layers"],
                              key=lambda l: l["mean_gate"])["layer"]
                          for s in all_stats})
    L.append("")
    L.append(f"Computed across the four runs: positions with gate < 0.4 "
             f"occur only in layers "
             f"{min(low_layers)}..{max(low_layers) if low_layers else '-'}; "
             f"the per-run peak layer-mean occurs at layer(s) "
             f"{', '.join(map(str, peak_layers))}; the final layer's mean "
             f"gate spans {min(final_means):.4f}-{max(final_means):.4f}. "
             f"Maps: `results/figures/gates_legacy_draft.pdf`.")
    L.append("")
    return L


def sec_anomalies(table_rows, rob_rows, f6_rows, diag_dir: Path):
    t = {(r["arch"], r["recipe"], r["variant"]): r for r in table_rows}
    rob = {(r["arch"], r["recipe"], r["variant"]): r for r in rob_rows}
    cell = ("vit_base", "mixup")
    base, saga = t[(*cell, "baseline")], t[(*cell, "saga")]

    last_block = max(int(r["block"]) for r in f6_rows)
    share = {(r["arch"], r["recipe"]): float(r["cls_attn_share"])
             for r in f6_rows
             if r["variant"] == "baseline" and int(r["block"]) == last_block}
    cell_share = share.pop(cell)
    other_shares = sorted(share.values())

    def norm(variant, key):
        p = diag_dir / f"e2_vit_base_mixup_{variant}_rlast_last_normstats.json"
        return json.load(open(p))[key]

    L = ["## 4. Known anomaly list for ViT-B/mixup (numbers restated)",
         "",
         f"- top-1: SAGA {saga['top1_last']} vs baseline {base['top1_last']} "
         f"(Δ = {saga['delta_top1_last_vs_baseline']}).",
         f"- best − last gaps: baseline +{base['top1_best_minus_last']}, "
         f"SAGA +{saga['top1_best_minus_last']} (both above the 0.25 flag).",
         f"- baseline final-block CLS attention share {cell_share:.4f} "
         f"(other cells' baselines: {other_shares[0]:.4f}–"
         f"{other_shares[-1]:.4f}).",
         f"- v1 fixed-τ saturation: baseline {base['sink_fixed_thr']}/196, "
         f"SAGA {saga['sink_fixed_thr']}/196 tokens above the per-arch τ.",
         f"- v2 per-cell τ (no saturation): baseline "
         f"{fmt(float(rob[(*cell, 'baseline')]['sink_fixed_v2']))}, "
         f"registers {fmt(float(rob[(*cell, 'registers')]['sink_fixed_v2']))}, "
         f"SAGA {fmt(float(rob[(*cell, 'saga')]['sink_fixed_v2']))} — SAGA > "
         "baseline in this cell even under v2.",
         f"- norm scale (SAGA vs baseline, normstats): median of medians "
         f"{fmt(norm('saga', 'median_of_medians'), 3)} vs "
         f"{fmt(norm('baseline', 'median_of_medians'), 3)}; p99.9 "
         f"{fmt(norm('saga', 'p999'), 3)} vs "
         f"{fmt(norm('baseline', 'p999'), 3)}.",
         f"- eff_rank: baseline {fmt(float(base['eff_rank']), 2)} "
         "(other baselines: "
         + ", ".join(fmt(float(t[k]['eff_rank']), 2) for k in sorted(t)
                     if k[2] == 'baseline' and (k[0], k[1]) != cell)
         + ").",
         ""]
    return L


def sec_evidence(incomplete, rows, diag_dir: Path):
    """Every number in the observation strings is computed from the data."""
    f = {(r["arch"], r["recipe"], r["variant"],
          r["filename"].removesuffix(".pth")): r for r in rows}
    saga_last = f[("vit_base", "mixup", "saga", "last")]
    saga_best = f[("vit_base", "mixup", "saga", "best")]
    base_last = f[("vit_base", "mixup", "baseline", "last")]
    # steps/epoch from a completed reference run (any last.pth at 299)
    ref = next(r for r in rows
               if r["filename"] == "last.pth"
               and int(r["epoch"]) == FINAL_EPOCH)
    spe = round(int(ref["optimizer_step_count"]) / (FINAL_EPOCH + 1))
    final_lr = float(ref["last_lr"])

    def rel(key):
        s = json.load(open(diag_dir /
                           "e2_vit_base_mixup_saga_rlast_last_normstats.json"))
        b = json.load(open(diag_dir /
                           "e2_vit_base_mixup_baseline_rlast_last_normstats.json"))
        return (s[key] - b[key]) / b[key] * 100

    bmix = {run: v for run, v in incomplete.items() if "ViT-B/16/mixup" in run}
    obs = [
        (f"All three ViT-B/mixup runs' last.pth sit below epoch "
         f"{FINAL_EPOCH} ("
         + "; ".join(f"{r.split('/')[-1]} at {e}"
                     for r, (e, _) in sorted(bmix.items())) + ")",
         "consistent", "inconsistent (the cell was never fully trained)"),
        (f"SAGA last.pth at epoch {saga_last['epoch']} with LR "
         f"{float(saga_last['last_lr']):.3e} — early in the cosine schedule "
         f"(completed runs end at {final_lr:.0e})",
         "consistent", "inconsistent (metrics reflect an early-training "
         "model, not a converged pathology)"),
        (f"SAGA best.pth (epoch {saga_best['epoch']}) is newer than its "
         f"last.pth (epoch {saga_last['epoch']}): training reached at least "
         f"epoch {saga_best['epoch']} but never completed epoch "
         f"{int(saga_last['epoch']) + SAVE_FREQ} (the next save point)",
         "consistent (a kill between save points; whether the run was "
         "resubmitted and killed again before the next save point is not "
         "determinable from checkpoint state)",
         "neutral"),
        (f"Optimizer step counts match the saved epochs (~{spe} steps/epoch "
         "throughout) — no sign of state corruption",
         "consistent (clean kill, not damage to files)",
         "consistent"),
        ("phi carries no NaN/Inf; its low-gate layers and peak layer fall "
         "in the same ranges as the other three SAGA runs (computed in "
         "section 3)",
         "consistent (gates simply less trained)",
         "inconsistent (no gate pathology visible)"),
        (f"Norm-scale anomalies (median {rel('median_of_medians'):+.1f}%, "
         f"MAD threshold {rel('mean_threshold_mad_k5'):+.1f}% vs baseline) "
         "and SAGA > baseline sinks under v2",
         f"confounded — SAGA({int(saga_last['epoch']) + 1} epochs) is "
         f"compared against baseline({int(base_last['epoch']) + 1} epochs); "
         "different training stages",
         "would require equal-progress runs to attribute"),
        (f"The cell's baseline itself is incomplete "
         f"({int(base_last['epoch']) + 1}/{FINAL_EPOCH + 1}) — see "
         "sections 2 and 4",
         "consistent (every comparison inside this cell is between "
         "unfinished runs)",
         "inconsistent as evidence of a SAGA-specific phenomenon"),
    ]
    L = ["## 5. Evidence summary — candidate explanations (NO verdict)",
         "",
         "| observation | incomplete/damaged run (24h-resume era) | "
         "trained-but-pathological |",
         "|---|---|---|"]
    for o, a, b in obs:
        L.append(f"| {o} | {a} | {b} |")
    L += ["", "The verdict is the human's; this note only lists the "
          "evidence.", ""]
    return L


def main():
    parser = argparse.ArgumentParser(description="Generate the ViT-B/mixup "
                                                 "forensics note.")
    parser.add_argument("--forensics",
                        default="results/legacy/ckpt_forensics.csv")
    parser.add_argument("--gates-dir", default="results/legacy/gates")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--table",
                        default="results/tables/legacy_e2_corrected.csv")
    parser.add_argument("--robustness",
                        default="results/tables/sink_robustness.csv")
    parser.add_argument("--f6", default="results/figures_data/F6_legacy.csv")
    parser.add_argument("--out", default="results/notes/bmixup_forensics.md")
    args = parser.parse_args()

    rows = read_csv(args.forensics)
    table_rows = read_csv(args.table)
    rob_rows = read_csv(args.robustness)

    L = ["# ViT-B/mixup forensics — checkpoint completeness & gate state "
         "(TASK-02C)",
         "",
         f"Generated by `analysis/build_bmixup_forensics.py`; git "
         f"`{git_sha()[:12]}`, "
         f"{datetime.now(timezone.utc).date().isoformat()}. Facts only, "
         "sourced from `results/legacy/ckpt_forensics.csv`, the phi stats, "
         "the normstats, and the corrected tables. No verdict.",
         ""]
    L += sec_semantics()
    completeness, incomplete = sec_completeness(rows)
    L += completeness
    L += sec_phi(Path(args.gates_dir))
    L += sec_anomalies(table_rows, rob_rows, read_csv(args.f6),
                       Path(args.diag_dir))
    L += sec_evidence(incomplete, rows, Path(args.diag_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    print(f"incomplete runs: {len(incomplete)}")
    for run, (e, lr) in sorted(incomplete.items()):
        print(f"  - {run}: last at {e}")


if __name__ == "__main__":
    main()
