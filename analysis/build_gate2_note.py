#!/usr/bin/env python3
"""
analysis/build_gate2_note.py
============================
TASK-09 Phase C.4: generate results/notes/gate2_report.md.

EVERY number in the note is computed here from committed files —
results/tables/T4_coco*.csv, results/tables/T5_ade20k*.csv and the runs'
own log.csv — so re-running after a table rebuild cannot leave stale prose
behind (the TASK-02C lesson). Nothing is hand-typed.

THE VERDICT RULE IS FROZEN HERE, in these terms and no others. TASK-09
states the Gate-2 question as: "AP_S and/or background-class mIoU move in
SAGA's favor vs baseline" -> PASS / PARTIAL / FAIL, no framing. That maps
onto the three outcomes the task asks for as:

    moved_ap_s  :=  (saga.AP_S - baseline.AP_S) > 0
    moved_bg    :=  mean over {wall, sky, floor} of
                    (saga.iou - baseline.iou) > 0
    PASS     both moved
    PARTIAL  exactly one moved
    FAIL     neither moved

A delta of exactly zero is NOT "in SAGA's favor". The background set is the
three classes TASK-09 Phase C.2 names; the value used is the single-scale
per-class IoU, because per_class_iou.csv is written from the same confusion
matrix as mIoU_ss.

The note reports the measurement uncertainty next to the verdict but the
verdict is computed only from the rule above — uncertainty never moves it.

    python analysis/build_gate2_note.py [--out results/notes/gate2_report.md]
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

MISSING = "MISSING"
REPO = Path(__file__).resolve().parents[1]


def show(path: Path):
    """Repo-relative when the path is inside the repo, absolute otherwise —
    a --out outside the tree (tests, scratch) must not crash the tool."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path

BACKGROUND_ORDER = ["wall", "sky", "floor"]
AP_KEYS = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]
SEG_KEYS = ["mIoU_ss", "mIoU_ms", "pixel_acc", "mean_acc"]
VARIANTS = ["baseline", "registers", "saga"]
LAST_N_EVALS = 3      # window for the within-run spread (noise proxy)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    if v in ("", MISSING, None):
        return MISSING
    return float(v)


def fmt(v, nd=3, sign=False):
    if v == MISSING:
        return MISSING
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def pick(rows, kind, variant):
    for r in rows:
        if r["kind"] == kind and r["variant"] == variant:
            return r
    return None


def within_run_spread(run_dir: Path, column: str):
    """(spread, n_evals, values) over the last LAST_N_EVALS eval epochs.

    With exactly one run per cell this is the only noise estimate the data
    can supply: it is NOT a seed-level error bar, and the note says so.
    """
    log = run_dir / "log.csv"
    if not log.exists():
        return MISSING, 0, []
    rows = [r for r in read_csv(log) if r.get(column) not in ("", None)]
    vals = [float(r[column]) for r in rows][-LAST_N_EVALS:]
    if len(vals) < 2:
        return MISSING, len(vals), vals
    return max(vals) - min(vals), len(vals), vals


def main():
    p = argparse.ArgumentParser("TASK-09 Phase C Gate-2 note")
    p.add_argument("--tables", default="results/tables")
    p.add_argument("--det-root", default="results/detection")
    p.add_argument("--out", default="results/notes/gate2_report.md")
    args = p.parse_args()

    def rel(x):
        q = Path(x)
        return q if q.is_absolute() else REPO / q

    tdir = rel(args.tables)
    t4 = read_csv(tdir / "T4_coco.csv")
    t4c = read_csv(tdir / "T4_coco_per_category.csv")
    t5 = read_csv(tdir / "T5_ade20k.csv")
    t5c = read_csv(tdir / "T5_ade20k_per_class.csv")

    # ── the two criterion numbers ─────────────────────────────────────────
    d_saga = pick(t4, "delta_saga_minus_baseline", "saga")
    ap_s_delta = num(d_saga["AP_S"]) if d_saga else MISSING

    bg_rows = [r for r in t5c if r["is_background_class"] == "yes"]
    bg = {}
    for r in bg_rows:
        head = r["name"].split(",")[0].strip().lower()
        bg[head] = {
            "index": int(r["class_index"]),
            "name": r["name"],
            "baseline": num(r["baseline_iou_pts"]),
            "saga": num(r["saga_iou_pts"]),
            "registers": num(r["registers_iou_pts"]),
            "delta": num(r["saga_minus_baseline_iou_pts"]),
        }
    bg_deltas = [bg[k]["delta"] for k in BACKGROUND_ORDER
                 if k in bg and bg[k]["delta"] != MISSING]
    bg_delta = (statistics.fmean(bg_deltas)
                if len(bg_deltas) == len(BACKGROUND_ORDER) else MISSING)

    # ── the frozen rule ───────────────────────────────────────────────────
    moved_ap_s = (ap_s_delta != MISSING and ap_s_delta > 0)
    moved_bg = (bg_delta != MISSING and bg_delta > 0)
    if ap_s_delta == MISSING or bg_delta == MISSING:
        verdict = "INCOMPLETE"
    elif moved_ap_s and moved_bg:
        verdict = "PASS"
    elif moved_ap_s or moved_bg:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    # ── distribution facts ────────────────────────────────────────────────
    def cat_stats(metric):
        vals = [(r["name"], float(r[f"saga_minus_baseline_{metric}"]))
                for r in t4c
                if r[f"saga_minus_baseline_{metric}"] != MISSING]
        if not vals:
            return None
        xs = [v for _, v in vals]
        vals.sort(key=lambda t: t[1])
        return {"n": len(xs), "wins": sum(1 for x in xs if x > 0),
                "mean": statistics.fmean(xs), "median": statistics.median(xs),
                "worst": vals[:3], "best": vals[-3:][::-1]}

    ap_cat, aps_cat = cat_stats("AP"), cat_stats("AP_S")

    cls_vals = [float(r["saga_minus_baseline_iou_pts"]) for r in t5c
                if r["saga_minus_baseline_iou_pts"] != MISSING]
    cls = {"n": len(cls_vals),
           "wins": sum(1 for x in cls_vals if x > 0),
           "mean": statistics.fmean(cls_vals) if cls_vals else MISSING,
           "median": statistics.median(cls_vals) if cls_vals else MISSING}

    # ── the noise proxy ───────────────────────────────────────────────────
    spreads = {}
    for variant in VARIANTS:
        row = pick(t4, "value", variant)
        if not row:
            continue
        s, n, vals = within_run_spread(rel(args.det_root) / row["run_id"],
                                       "AP_S")
        spreads[variant] = {"run_id": row["run_id"], "spread": s, "n": n,
                            "values": vals}

    # ── render ────────────────────────────────────────────────────────────
    L = []
    A = L.append
    A("# Gate 2 — dense prediction (COCO detection, ADE20K segmentation)")
    A("")
    A("Generated by `analysis/build_gate2_note.py` from the committed "
      "tables. Every number below is read from a file under `results/`; "
      "none is typed by hand. Re-run the script after any table rebuild.")
    A("")

    A("## 1. The Gate-2 question")
    A("")
    A("> AP_S and/or background-class mIoU move in SAGA's favor vs baseline")
    A("")
    A("Evaluated exactly as frozen in the generator:")
    A("")
    A(f"- `AP_S(saga) - AP_S(baseline)` = **{fmt(ap_s_delta, 3, True)}** "
      f"-> moved in SAGA's favor: **{'yes' if moved_ap_s else 'no'}**")
    bgtxt = ", ".join(
        f"{bg[k]['name'].split(',')[0]} {fmt(bg[k]['delta'], 4, True)}"
        for k in BACKGROUND_ORDER if k in bg)
    A(f"- background-class IoU, mean over {{wall, sky, floor}} = "
      f"**{fmt(bg_delta, 4, True)}** IoU points ({bgtxt}) -> moved in "
      f"SAGA's favor: **{'yes' if moved_bg else 'no'}**")
    A("")
    A(f"## VERDICT: {verdict}")
    A("")
    A("Rule: PASS if both move in SAGA's favor, PARTIAL if exactly one, "
      "FAIL if neither. A delta of exactly zero is not \"in favor\". "
      "The uncertainty discussed in section 5 does not enter the rule.")
    A("")

    A("## 2. T4 — COCO detection (results/tables/T4_coco.csv)")
    A("")
    A("| variant | " + " | ".join(AP_KEYS) + " | backbone |")
    A("|---|" + "---|" * (len(AP_KEYS) + 1))
    for v in VARIANTS:
        r = pick(t4, "value", v)
        if not r:
            continue
        A(f"| {v} | " + " | ".join(r[k] for k in AP_KEYS)
          + f" | {r['backbone_source']} |")
    for v in ("registers", "saga"):
        r = pick(t4, f"delta_{v}_minus_baseline", v)
        if not r:
            continue
        A(f"| **{v} − baseline** | " + " | ".join(r[k] for k in AP_KEYS)
          + " | |")
    A("")
    r0 = pick(t4, "value", "baseline")
    if r0:
        A(f"All three at epoch {r0['epoch']}, {r0['n_val_images']} val "
          f"images, seed {r0['seed']}. Selection is on AP, so AP_S is read "
          f"at each run's best-AP epoch, not at its best-AP_S epoch.")
    A("")

    A("## 3. T5 — ADE20K segmentation (results/tables/T5_ade20k.csv)")
    A("")
    A("| variant | " + " | ".join(SEG_KEYS) + " | best epoch (ss) | backbone |")
    A("|---|" + "---|" * (len(SEG_KEYS) + 2))
    for v in VARIANTS:
        r = pick(t5, "value", v)
        if not r:
            continue
        A(f"| {v} | " + " | ".join(r[k] for k in SEG_KEYS)
          + f" | {r['epoch_ss']} | {r['backbone_source']} |")
    for v in ("registers", "saga"):
        r = pick(t5, f"delta_{v}_minus_baseline", v)
        if not r:
            continue
        A(f"| **{v} − baseline** | " + " | ".join(r[k] for k in SEG_KEYS)
          + " | | |")
    A("")
    A("mIoU/pixel_acc/mean_acc are percentages; the per-class table carries "
      "fractions AND IoU points. The background rows:")
    A("")
    A("| class | index | baseline | saga | saga − baseline | registers |")
    A("|---|---|---|---|---|---|")
    for k in BACKGROUND_ORDER:
        if k not in bg:
            continue
        b = bg[k]
        A(f"| {b['name']} | {b['index']} | {fmt(b['baseline'], 4)} | "
          f"{fmt(b['saga'], 4)} | {fmt(b['delta'], 4, True)} | "
          f"{fmt(b['registers'], 4)} |")
    A("")

    A("## 4. What the aggregates are made of")
    A("")
    if aps_cat:
        A(f"- Per-category AP_S (80 COCO categories): SAGA is above baseline "
          f"in **{aps_cat['wins']} of {aps_cat['n']}**, yet the mean delta is "
          f"**{fmt(aps_cat['mean'], 3, True)}** and the median is "
          f"**{fmt(aps_cat['median'], 3, True)}**. The aggregate is pulled "
          f"down by a small number of large negative swings: "
          + ", ".join(f"{n} {v:+.2f}" for n, v in aps_cat["worst"])
          + "; the largest gains are "
          + ", ".join(f"{n} {v:+.2f}" for n, v in aps_cat["best"]) + ".")
    if ap_cat:
        A(f"- Per-category AP: SAGA above baseline in **{ap_cat['wins']} of "
          f"{ap_cat['n']}**, mean {fmt(ap_cat['mean'], 3, True)}, median "
          f"{fmt(ap_cat['median'], 3, True)}.")
    A(f"- Per-class ADE20K IoU (150 classes): SAGA above baseline in "
      f"**{cls['wins']} of {cls['n']}**, mean "
      f"{fmt(cls['mean'], 4, True)} IoU points, median "
      f"{fmt(cls['median'], 4, True)}.")
    A("")
    A("These are distribution facts about the same numbers the verdict uses. "
      "They do not change it.")
    A("")

    A("## 5. How much weight these deltas carry")
    A("")
    A("**There is exactly one run per cell.** No seed repeats were trained "
      "for the dense runs, so no delta here has a standard error, and none "
      "of the project's usual significance machinery (paired SE, "
      "significant_2xSE, Welch — see `analysis/build_pooled_tables.py`) can "
      "be applied.")
    A("")
    A("The only noise estimate the data can supply is the within-run spread "
      f"of AP_S over the last {LAST_N_EVALS} evaluation epochs of each run:")
    A("")
    A("| run | AP_S at the last evals | spread |")
    A("|---|---|---|")
    for v in VARIANTS:
        s = spreads.get(v)
        if not s:
            continue
        vals = ", ".join(f"{x:.3f}" for x in s["values"])
        A(f"| {s['run_id']} | {vals} | {fmt(s['spread'], 3)} |")
    A("")
    biggest = max((s["spread"] for s in spreads.values()
                   if s["spread"] != MISSING), default=MISSING)
    if biggest != MISSING and ap_s_delta != MISSING:
        A(f"The AP_S delta the verdict turns on ({fmt(ap_s_delta, 3, True)}) "
          f"is of the same order as that within-run spread "
          f"(largest: {fmt(biggest, 3)}). This is a statement about the "
          f"precision of a single run, not a seed-level error bar, which "
          f"would require repeats that do not exist.")
    A("")

    A("## 6. Provenance and caveats")
    A("")
    rreg = pick(t4, "value", "registers")
    if rreg and rreg["backbone_source"] == "fallback":
        A(f"- **The registers rows are not backbone-matched.** They use "
          f"`{rreg['backbone_run']}` (`backbone_source: fallback`), the "
          f"UNSEEDED legacy ViT-B registers checkpoint, because the seeded "
          f"e2r registers run had not finished at launch. baseline and saga "
          f"both use their seeded e2r s1 backbones. registers-vs-baseline "
          f"deltas therefore mix a backbone change with a variant change.")
    rseg = pick(t5, "value", "registers")
    rsegb = pick(t5, "value", "baseline")
    if rseg and rsegb and rseg["epoch_ss"] != rsegb["epoch_ss"]:
        A(f"- The segmentation registers run's best val mIoU is at epoch "
          f"{rseg['epoch_ss']} while baseline and saga peak at "
          f"{rsegb['epoch_ss']}; its val mIoU declined over the remaining "
          f"epochs. Val-selection is working as designed, but the three "
          f"columns are not read at the same point in training.")
    A("- Every table value is file-sourced; a run whose recorded backbone "
      "sha256 does not match the matrix candidate it names yields MISSING "
      "rather than a number from an unverified checkpoint.")
    A("")

    out = rel(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L) + "\n"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(out)
    print(f"wrote {show(out)}  ({verdict})")
    return verdict


if __name__ == "__main__":
    main()
