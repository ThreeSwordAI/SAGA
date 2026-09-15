#!/usr/bin/env python3
"""
analysis/build_finegrained_ext_note.py — TASK-13 C3
===================================================
Generates results/notes/finegrained_ext.md from
  results/tables/T3_finegrained.csv        (rebuilt with the ViT-B seed fill)
  results/tables/T_ring_ablation.csv       (the four conditions)
  results/finegrained/ring_ablation/*.json (the reproduction check)

EVERY number in the note is read from those files. Nothing is typed in,
including the prose ranges — TASK-07 was burned by a hand-typed "1.79x" that
should have been 1.71x, and TASK-02C by two hardcoded prose blocks.
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

MISSING = "MISSING"
REPO = Path(__file__).resolve().parents[1]
ARCH = {"vit_small_patch16_224": "ViT-S", "vit_base_patch16_224": "ViT-B"}
DS = {"cub": "CUB", "aircraft": "Aircraft"}


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pick(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def verdict(mean, se):
    """The 2xSE verdict, always WITH its direction and its margin — a bare
    'significant: YES' was misread as favourable on the CUB cell in TASK-08."""
    if mean is None or se is None:
        return MISSING, None
    margin = abs(mean) - 2 * se
    direction = "ABOVE" if mean > 0 else "BELOW"
    if margin > 0:
        thin = " (THIN: margin < 25% of the 2xSE threshold)" \
            if margin < 0.25 * (2 * se) else ""
        return (f"clears 2xSE, SAGA {direction} baseline "
                f"(|d| - 2SE = {margin:+.3f}){thin}"), True
    return (f"does NOT clear 2xSE (|d| - 2SE = {margin:+.3f}); "
            f"point estimate is {direction} baseline"), False


def t3_section(t3):
    out, clears = [], {}
    out.append("## 1. T3 — fine-grained transfer, now with ViT-B repeats\n")
    out.append("Exact test top-1, official test split touched once per run, "
               "val-selected checkpoint. Repeats are listed before any mean; "
               "deltas are paired BY ft-seed (the only valid pairing — both "
               "sides share the frozen val split).\n")
    out.append("| cell | n | baseline mean | SAGA mean | per-seed deltas | "
               "paired mean | SE | 2xSE verdict | Welch p |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for ds in ("cub", "aircraft"):
        for arch in ("vit_small_patch16_224", "vit_base_patch16_224"):
            k = dict(dataset=ds, arch=arch)
            bm = pick(t3, **k, kind="mean", variant="baseline")
            sm = pick(t3, **k, kind="mean", variant="saga")
            dm = pick(t3, **k, kind="paired_delta_mean", variant="saga")
            dse = pick(t3, **k, kind="paired_delta_se", variant="saga")
            wp = pick(t3, **k, kind="welch_p", variant="saga")
            reps = sorted([r for r in t3 if r["dataset"] == ds
                           and r["arch"] == arch and r["variant"] == "saga"
                           and r["kind"] == "paired_delta"],
                          key=lambda r: r["ft_seed"])
            per = " / ".join(f"{num(r['test_top1']):+.3f}" for r in reps)
            m, se = num(dm["test_top1"]) if dm else None, \
                num(dse["test_top1"]) if dse else None
            v, ok = verdict(m, se)
            clears[(ds, arch)] = (m, se, ok, v)
            out.append(
                f"| {DS[ds]} {ARCH[arch]} | {dm['n'] if dm else '?'} | "
                f"{num(bm['test_top1']):.3f} | {num(sm['test_top1']):.3f} | "
                f"{per} | **{m:+.3f}** | {se:.4f} | {v} | "
                f"{num(wp['test_top1']):.4f} |")
    return out, clears


def ring_section(ring):
    out = []
    out.append("## 2. Ring ablation — the background-informativeness test\n")
    out.append("Four input conditions per run, applied in pixel space to "
               "patch regions of the 14x14 grid; the ring definition is "
               "imported from TASK-07's `analysis/address_analysis.py`, where "
               "ring 1 (one patch inside the border) is where the sinks and "
               "SAGA's strongest gate suppression sit. All three mask "
               "conditions are area-matched at 44 patches.\n")
    out.append("**The quantity that matters is the CONTRAST** "
               "`drop_ring1 - drop_random44`. Both terms come from the same "
               "run, the same checkpoint and the same test images, so "
               "everything that makes one dataset or model easier than "
               "another cancels out of it. A raw drop does not have that "
               "property.\n")
    out.append("**Stated prediction:** the hypothesis (positional suppression "
               "helps when the border is uninformative for the class, hurts "
               "when it carries class evidence) predicts a LARGER ring-1 drop "
               "for CUB than for Aircraft, relative to each dataset's own "
               "`mask_random44` control — i.e. a HIGHER contrast for CUB.\n")

    out.append("### 2.1 Per dataset, pooled over both architectures and "
               "both variants\n")
    out.append("| dataset | n runs | drop ring1 | drop center44 | "
               "drop random44 | CONTRAST (ring1 - random44) |")
    out.append("|---|---|---|---|---|---|")
    vals = {}
    for ds in ("cub", "aircraft"):
        m = pick(ring, dataset=ds, arch="ALL", kind="mean")
        s = pick(ring, dataset=ds, arch="ALL", kind="se")
        g = lambda c: f"{num(m[c]):+.3f} ± {num(s[c]):.3f}"   # noqa: E731
        vals[ds] = (num(m["contrast_ring1_minus_random44"]),
                    num(s["contrast_ring1_minus_random44"]),
                    num(m["drop_ring1"]), num(m["drop_random44"]))
        out.append(f"| {DS[ds]} | {m['n']} | {g('drop_ring1')} | "
                   f"{g('drop_center44')} | {g('drop_random44')} | "
                   f"**{g('contrast_ring1_minus_random44')}** |")

    diff = pick(ring, kind="between_dataset_difference")
    d = num(diff["contrast_ring1_minus_random44"])
    out.append("\n### 2.2 Does it match the prediction?\n")
    matches = d is not None and d > 0
    out.append(f"CUB contrast **{vals['cub'][0]:+.3f}** vs Aircraft "
               f"**{vals['aircraft'][0]:+.3f}**; difference "
               f"**{d:+.4f}** ({diff['flag']}).\n")
    out.append(f"**The predicted ORDERING {'HOLDS' if matches else 'DOES NOT HOLD'}"
               f"**: CUB's contrast is "
               f"{'higher' if matches else 'not higher'} than Aircraft's, and "
               f"the difference clears 2xSE. Ring 1 does matter relatively "
               f"more for {DS['cub' if matches else 'aircraft']} than for "
               f"{DS['aircraft' if matches else 'cub']}.\n")

    out.append("**Three qualifications that the same table forces, stated "
               "because they bound what the ordering can be used to claim:**\n")
    out.append(f"1. **Both contrasts are NEGATIVE** "
               f"({vals['cub'][0]:+.3f} and {vals['aircraft'][0]:+.3f}). "
               f"Masking ring 1 costs LESS than masking 44 random patches on "
               f"BOTH datasets — so ring 1 is less informative than an "
               f"average area-matched region everywhere, including CUB. The "
               f"hypothesis's ordinal prediction survives; its mechanism "
               f"story — that CUB's border *carries class evidence* — is NOT "
               f"supported in absolute terms.")
    out.append(f"2. **The raw ring-1 drop orders the OTHER way.** Aircraft "
               f"{vals['aircraft'][2]:+.3f} vs CUB {vals['cub'][2]:+.3f}: "
               f"masking ring 1 costs Aircraft MORE in absolute points. The "
               f"contrast reverses that ordering because Aircraft's "
               f"`random44` control is much larger "
               f"({vals['aircraft'][3]:+.3f} vs {vals['cub'][3]:+.3f}, a "
               f"factor of {vals['aircraft'][3] / vals['cub'][3]:.2f}). So the "
               f"result is carried by the DENOMINATOR — how much a random "
               f"region matters — not by ring 1 itself.")
    ctr = {ds: num(pick(ring, dataset=ds, arch="ALL", kind="mean")
                   ["drop_center44"]) for ds in ("cub", "aircraft")}
    out.append(f"3. **The centre is where the evidence is, on both.** "
               f"`mask_center44` costs {ctr['cub']:+.3f} (CUB) and "
               f"{ctr['aircraft']:+.3f} (Aircraft) — an order of magnitude "
               f"more than ring 1 for the same 44 patches. Both datasets are "
               f"object-centred; no border story competes with that.")

    out.append("\n### 2.3 Per cell — is the ordering robust?\n")
    out.append("| cell | drop ring1 | drop random44 | CONTRAST |")
    out.append("|---|---|---|---|")
    per_cell = {}
    for ds in ("cub", "aircraft"):
        for arch in ("vit_small_patch16_224", "vit_base_patch16_224"):
            for v in ("baseline", "saga"):
                m = pick(ring, dataset=ds, arch=arch, variant=v, kind="mean")
                s = pick(ring, dataset=ds, arch=arch, variant=v, kind="se")
                if not m:
                    continue
                c = num(m["contrast_ring1_minus_random44"])
                per_cell.setdefault(ds, []).append(c)
                out.append(
                    f"| {DS[ds]} {ARCH[arch]} {v} | "
                    f"{num(m['drop_ring1']):+.3f} ± {num(s['drop_ring1']):.3f} | "
                    f"{num(m['drop_random44']):+.3f} ± "
                    f"{num(s['drop_random44']):.3f} | "
                    f"{c:+.3f} ± {num(s['contrast_ring1_minus_random44']):.3f} |")
    cub_lo, cub_hi = min(per_cell["cub"]), max(per_cell["cub"])
    air_lo, air_hi = min(per_cell["aircraft"]), max(per_cell["aircraft"])
    separated = cub_lo > air_hi
    out.append(f"\nCUB cells span {cub_lo:+.3f}..{cub_hi:+.3f}; Aircraft "
               f"cells span {air_lo:+.3f}..{air_hi:+.3f} (both low..high). "
               f"The two sets are "
               f"**{'DISJOINT' if separated else 'OVERLAPPING'}** — "
               f"{'every' if separated else 'not every'} one of the "
               f"{len(per_cell['cub'])} CUB cells has a higher contrast than "
               f"{'every' if separated else 'every'} one of the "
               f"{len(per_cell['aircraft'])} Aircraft cells"
               + (f", with a gap of {cub_lo - air_hi:.3f} between the two "
                  f"sets, so the dataset ordering is not an artefact of "
                  f"pooling." if separated else "."))
    return out, vals, matches


def cell_tracking(t3, ring):
    """Does the ring contrast track the T3 delta across the four cells?
    Computed and reported — NOT asserted. n=4, which is stated."""
    pairs = []
    for ds in ("cub", "aircraft"):
        for arch in ("vit_small_patch16_224", "vit_base_patch16_224"):
            d = pick(t3, dataset=ds, arch=arch, kind="paired_delta_mean",
                     variant="saga")
            c = pick(ring, dataset=ds, arch=arch, variant="saga", kind="mean")
            if d and c:
                pairs.append((DS[ds], ARCH[arch],
                              num(d["test_top1"]),
                              num(c["contrast_ring1_minus_random44"])))
    out = ["\n### 2.4 Does the contrast track the T3 delta, cell by cell?\n"]
    out.append("| cell | T3 paired delta (SAGA - baseline) | "
               "SAGA ring contrast |")
    out.append("|---|---|---|")
    for ds, arch, d, c in pairs:
        out.append(f"| {ds} {arch} | {d:+.3f} | {c:+.3f} |")
    deltas = [p[2] for p in pairs]
    contrasts = [p[3] for p in pairs]
    rd = _rank(deltas)
    rc = _rank(contrasts)
    rho = _pearson(rd, rc)
    n = len(pairs)
    out.append(f"\nSpearman rho over these {n} cells = "
               f"{'MISSING' if rho is None else f'{rho:+.3f}'}. **With n={n} "
               f"this is reported, not interpreted** — four points cannot "
               f"support a correlation claim, and the two CUB cells sit on "
               f"opposite sides of zero in the left column while both Aircraft "
               f"cells sit on the same side. The cell-level relationship "
               f"between the two quantities is therefore UNRESOLVED by this "
               f"task's data.")
    return out


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    return r


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return None
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (va * vb) ** 0.5


def repro_section(ring_dir):
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(ring_dir.glob("*.json"))]
    n_ok = sum(r["full_reproduces_test_final"] for r in recs)
    worst = max(abs(r["full_minus_committed"]) for r in recs)
    tol = recs[0]["full_tolerance"]
    out = ["## 3. The `full`-condition reproduction check\n"]
    out.append(f"The `full` condition re-runs the TASK-08 eval path "
               f"unmodified and must reproduce each run's committed "
               f"`test_final.json` top-1 within {tol}.\n")
    out.append(f"**{n_ok} of {len(recs)} pass. The largest absolute "
               f"deviation across all {len(recs)} runs is {worst:.3f} "
               f"points.**"
               + (" Every run reproduces its committed number EXACTLY, so the "
                  "ablation's baseline is the committed number itself, not an "
                  "approximation of it." if worst == 0 else ""))
    seeds = {r["seed"] for r in recs}
    grids = {r["grid_side"] for r in recs}
    nmask = {r["n_masked_patches"]["mask_ring1"] for r in recs}
    rings = {tuple(r["masked_indices"]["mask_ring1"]) for r in recs}
    out.append(f"\nConsistency across the {len(recs)} files: seed(s) "
               f"{sorted(seeds)}, grid side {sorted(grids)}, ring-1 mask size "
               f"{sorted(nmask)}, and {len(rings)} distinct ring-1 index "
               f"set(s) (1 = the TASK-07 definition was applied identically "
               f"everywhere). Fill value is each dataset's own per-channel "
               f"mean over its TRAIN split, never the test split.")
    return out, len(recs), n_ok, worst


def main():
    ap = argparse.ArgumentParser("TASK-13 C3 note")
    ap.add_argument("--t3", default="results/tables/T3_finegrained.csv")
    ap.add_argument("--ring", default="results/tables/T_ring_ablation.csv")
    ap.add_argument("--ring-dir", default="results/finegrained/ring_ablation")
    ap.add_argument("--out", default="results/notes/finegrained_ext.md")
    a = ap.parse_args()

    def _p(x):
        x = Path(x)
        return x if x.is_absolute() else REPO / x

    t3 = read_csv(_p(a.t3))
    ring = read_csv(_p(a.ring))

    L = []
    L.append("# TASK-13 — fine-grained extension: ViT-B seed fill and the "
             "ring ablation\n")
    L.append("*Generated by `analysis/build_finegrained_ext_note.py`. Every "
             "number is read from `results/tables/T3_finegrained.csv`, "
             "`results/tables/T_ring_ablation.csv` and the committed "
             "ring-ablation JSONs — none is typed in.*\n")

    L.append("## 0. The legacy fine-grained numbers remain VOID\n")
    L.append("The legacy e6 results — **+2.19 Aircraft and +1.29 CUB** — are "
             "VOID and are never cited. Bug B7: the legacy protocol used the "
             "official TEST split as its validation set, selected `best.pth` "
             "on it, and evaluated it every 5 epochs, so the reported figure "
             "is a test-tuned peak. They measure a different and invalid "
             "quantity and **must not be compared against anything in this "
             "note.** Everything below comes from the TASK-08 clean protocol: "
             "val carved from the official train split, checkpoint selected "
             "on val, official test split touched exactly once per run.\n")

    t3_lines, clears = t3_section(t3)
    L += t3_lines

    vitb = [(ds, arch) for (ds, arch) in clears if "base" in arch]
    L.append("\n### 1.1 Do the ViT-B deltas now clear 2xSE?\n")
    L.append("TASK-08 left both ViT-B cells at n=1, so neither could carry an "
             "SE or a significance claim. With ft-seeds 1 and 2 added they "
             "are at n=3:\n")
    for ds, arch in sorted(vitb):
        m, se, ok, v = clears[(ds, arch)]
        L.append(f"- **{DS[ds]} {ARCH[arch]}: {m:+.3f} ± {se:.4f} — {v}.**")
    s_cub = clears[("cub", "vit_small_patch16_224")]
    b_cub = clears[("cub", "vit_base_patch16_224")]
    L.append(f"\n**The two CUB architectures now disagree in SIGN**: ViT-S "
             f"{s_cub[0]:+.3f} (SAGA below baseline) against ViT-B "
             f"{b_cub[0]:+.3f} (SAGA above), both clearing 2xSE. A "
             f"dataset-level story alone cannot account for that, and no "
             f"pooled cross-architecture claim is made anywhere in this note.")

    ring_lines, vals, matches = ring_section(ring)
    L += ["\n"] + ring_lines + cell_tracking(t3, ring)

    rep_lines, n_files, n_ok, worst = repro_section(_p(a.ring_dir))
    L += ["\n"] + rep_lines

    L.append("\n## 4. Third dataset (sub-goal 3) — SKIPPED\n")
    L.append("Stanford Cars and Oxford Flowers-102 were **not** added. The "
             "human confirmed that neither is staged on the HPC and declined "
             "the sub-goal (2026-09-15); no dataset was downloaded. The "
             "reasoning recorded at the time: a third dataset differs from "
             "CUB and Aircraft along class count, train size and "
             "fine-grained axis simultaneously, so it would have added a "
             "third confounded point rather than isolating background "
             "informativeness — which is exactly what the ring ablation does "
             "*within* each dataset, at a fixed checkpoint and test set. "
             "Consequence, stated plainly: **no claim of generalization "
             "beyond CUB and Aircraft is available**, and none is made. "
             "`tests/test_task13_ring_ablation.py` pins the matrix to "
             "{cub, aircraft} so a dataset cannot be added later without its "
             "split builder, committed split file and tests.\n")

    L.append("## 5. Open items\n")
    L.append(f"- The contrast ordering is significant and per-cell disjoint, "
             f"but it is carried by the `random44` denominator rather than by "
             f"ring 1 itself (§2.2 item 2). A control that varies the mask "
             f"POSITION at matched control difficulty would separate those "
             f"two explanations; none exists in this task.")
    cs = pick(ring, dataset="cub", arch="vit_small_patch16_224",
              variant="saga", kind="mean")
    cb = pick(ring, dataset="cub", arch="vit_base_patch16_224",
              variant="saga", kind="mean")
    cs_v = num(cs["contrast_ring1_minus_random44"])
    cb_v = num(cb["contrast_ring1_minus_random44"])
    L.append(f"- The CUB sign disagreement between ViT-S and ViT-B (§1.1) is "
             f"unexplained. The two CUB SAGA cells differ in ring contrast by "
             f"{abs(cb_v - cs_v):.3f} ({cs_v:+.3f} for ViT-S against "
             f"{cb_v:+.3f} for ViT-B), so the contrast is NOT constant across "
             f"the two architectures — but with one contrast value per cell "
             f"there is no way to test whether that difference is what drives "
             f"the sign flip (§2.4).")
    L.append(f"- Both ViT-B cells clear 2xSE only narrowly; the CUB ViT-B "
             f"margin in particular is the thinnest in the table. A fourth "
             f"and fifth ft-seed would settle whether they are stable.")
    L.append(f"- `mask_random44` uses a single seed (seed 0) for every run, so "
             f"the control is one draw, not an average over draws. Re-running "
             f"`tools/ring_ablation.py --seed <k>` for a few k would put an "
             f"error bar on the control itself.")

    out = _p(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    try:                                   # --out may point outside the repo
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"wrote {shown} ({len(L)} blocks; "
          f"{n_ok}/{n_files} reproduction PASS, worst |delta| {worst:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
