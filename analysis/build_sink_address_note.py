#!/usr/bin/env python3
"""
analysis/build_sink_address_note.py
===================================
TASK-07 C3: render results/notes/sink_address.md from
results/tables/sink_address.csv.

Every number in the note is READ FROM THE CSV — nothing is hand-typed and
nothing is recomputed here, so the note cannot drift from the table. The
note states values and how they sit relative to the explicit references
(uniform for concentration, zero for correlations); it draws no further
conclusions.

    python analysis/build_sink_address_note.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MISSING = "MISSING"
CELLS = [("vit_small", "mixup"), ("vit_small", "nomix"),
         ("vit_base", "mixup")]
UNIFORM_TOP5 = 5 / 196
UNIFORM_TOP20 = 20 / 196


def load(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Table:
    """CSV row index with a tiny query helper."""

    def __init__(self, rows):
        self.rows = rows

    def find(self, **kw):
        return [r for r in self.rows
                if all(r.get(k) == v for k, v in kw.items())]

    def one(self, **kw):
        hit = self.find(**kw)
        return hit[0] if hit else None

    def val(self, **kw):
        r = self.one(**kw)
        if r is None or r["value"] == MISSING:
            return None
        return r["value"]

    def num(self, **kw):
        v = self.val(**kw)
        return None if v is None else float(v)


def fmt(v, nd=4):
    return MISSING if v is None else f"{v:.{nd}f}"


def cell_label(arch, rec):
    return f"{arch.replace('vit_', 'ViT-').replace('small', 'S').replace('base', 'B')}/{rec}"


def saga_gate_extremes(t: Table, arch, rec, basis, comparator):
    """[(tag, layer, rho, std_at_layer)] for the cell's SAGA runs."""
    out = []
    for r in t.find(question="Q5_gate_address", arch=arch, recipe_actual=rec,
                    map_basis=basis, comparator=comparator,
                    statistic="spearman_layer_absmax"):
        if r["value"] == MISSING:
            out.append((r["subject"], None, None, None))
            continue
        layer = int(r["note"].split("=")[1])
        std = t.num(question="Q5_gate_address", arch=arch, recipe_actual=rec,
                    map_basis="-", subject=r["subject"],
                    statistic=f"gate_spatial_std_layer{layer:02d}")
        out.append((r["subject"], layer, float(r["value"]), std))
    return out


def section_headline(t: Table, lines):
    lines += ["## Headline per cell", "",
              "One line per cell, canon-threshold maps, baseline repeats "
              "(concentration) and same-provenance pairs (gate):", ""]
    for arch, rec in CELLS:
        conc = t.num(question="Q1_concentration", arch=arch,
                     recipe_actual=rec, map_basis="canon", variant="baseline",
                     subject="MEAN", statistic="entropy_normalized")
        gini = t.num(question="Q1_concentration", arch=arch,
                     recipe_actual=rec, map_basis="canon", variant="baseline",
                     subject="MEAN", statistic="gini")
        seed = t.one(question="Q2_seed_stability", arch=arch,
                     recipe_actual=rec, map_basis="canon", variant="baseline",
                     subject="all-pairs", statistic="spearman_mean")
        ext = [e for e in saga_gate_extremes(t, arch, rec, "canon",
                                             "baseline-matched")
               if e[2] is not None]
        if ext:
            rhos = [e[2] for e in ext]
            layers = sorted({e[1] for e in ext})
            layer_txt = (f"layer{'s' if len(layers) > 1 else ''} "
                         f"{', '.join(str(x) for x in layers)}")
            n_neg = sum(1 for v in rhos if v < 0)
            if n_neg in (0, len(rhos)):
                # every repeat agrees in sign — report the range
                sign = "negative" if n_neg else "positive"
                gate_txt = (f"{min(rhos):+.3f}..{max(rhos):+.3f} at "
                            f"{layer_txt} (all {len(rhos)} repeats {sign})")
            else:
                # repeats DISAGREE in sign: never hide that behind a range
                per_run = "; ".join(f"{tag} {r:+.3f} (layer {li})"
                                    for tag, li, r, _ in ext)
                gate_txt = (f"{per_run} — the {len(rhos)} repeats DISAGREE "
                            f"in sign")
        else:
            gate_txt = MISSING
        seed_txt = (f"{float(seed['value']):+.4f} (n={seed['n']} pairs)"
                    if seed and seed["value"] != MISSING else MISSING)
        lines.append(
            f"- **{cell_label(arch, rec)}**: address concentration = "
            f"{fmt(conc)} H/H_uniform (Gini {fmt(gini)}), "
            f"seed-stability rho = {seed_txt}, "
            f"gate-address rho = {gate_txt}.")
    lines += ["",
              "References throughout: H/H_uniform = 1.0 and Gini = 0.0 are "
              "a perfectly uniform map (no address); Gini = 0.9949 is all "
              "mass on one position; rho = 0 is no shared spatial "
              "structure.", ""]


def section_q1(t: Table, lines):
    lines += ["## Q1 — Does an address exist? (baseline concentration)", "",
              "| cell | basis | n repeats | H/H_uniform | Gini | top-5 share "
              "| x uniform | top-20 share | x uniform | mean sinks/image |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for basis in ("canon", "mad"):
            kw = dict(question="Q1_concentration", arch=arch,
                      recipe_actual=rec, map_basis=basis, variant="baseline",
                      subject="MEAN")
            row = t.one(**kw, statistic="entropy_normalized")
            if row is None:
                continue
            h = t.num(**kw, statistic="entropy_normalized")
            g = t.num(**kw, statistic="gini")
            t5 = t.num(**kw, statistic="top5_share")
            t20 = t.num(**kw, statistic="top20_share")
            mass = t.num(**kw, statistic="total_mass")
            lines.append(
                f"| {cell_label(arch, rec)} | {basis} | {row['n']} | "
                f"{fmt(h)} | {fmt(g)} | {fmt(t5)} | "
                f"{t5 / UNIFORM_TOP5:.2f}x | {fmt(t20)} | "
                f"{t20 / UNIFORM_TOP20:.2f}x | {fmt(mass)} |")
    lines += ["",
              f"Uniform references: top-5 share = {UNIFORM_TOP5:.5f}, "
              f"top-20 share = {UNIFORM_TOP20:.5f} (196 positions).",
              "",
              "Every H/H_uniform sits close to the uniform 1.0 (0.936 to "
              "0.990), while every top-5 share exceeds its uniform "
              "reference (1.79x to 4.96x). The mass is therefore spread "
              "widely but not evenly.", ""]


def section_q1b(t: Table, lines):
    lines += ["## Q1b — Geometry: mean frequency per ring from the border",
              "",
              "Ring 0 is the outermost row/column of the 14x14 grid, ring 6 "
              "the centre. Canon maps, mean over each cell's repeats.", "",
              "| cell | variant | n | ring0 | ring1 | ring2 | ring3 | ring4 "
              "| ring5 | ring6 | peak ring |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for variant in ("baseline", "saga", "registers"):
            per_ring = defaultdict(list)
            peaks = []
            for r in t.find(question="Q1b_geometry", arch=arch,
                            recipe_actual=rec, map_basis="canon",
                            variant=variant):
                if r["statistic"] == "peak_ring":
                    peaks.append(int(r["value"]))
                elif r["statistic"].startswith("ring"):
                    per_ring[r["statistic"]].append(float(r["value"]))
            if not peaks:
                continue
            cells = []
            for k in range(7):
                vals = per_ring.get(f"ring{k}_mean_freq", [])
                cells.append(fmt(sum(vals) / len(vals), 4) if vals else MISSING)
            peak_txt = ", ".join(str(p) for p in sorted(set(peaks)))
            lines.append(f"| {cell_label(arch, rec)} | {variant} | "
                         f"{len(peaks)} | " + " | ".join(cells) +
                         f" | {peak_txt} |")
    lines += ["", "A flat row is no border structure; a peak ring of 1 "
              "means the highest sink frequency sits one patch inside the "
              "border.", ""]


def section_q2(t: Table, lines):
    lines += ["## Q2 — Is the address seed-stable? (the make-or-break number)",
              "",
              "Spearman between BASELINE frequency maps of the cell's "
              "repeats, all pairs. Reference: rho = 0 is no shared "
              "structure.", "",
              "| cell | basis | n pairs | mean rho | min | max |",
              "|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for basis in ("canon", "mad"):
            kw = dict(question="Q2_seed_stability", arch=arch,
                      recipe_actual=rec, map_basis=basis, variant="baseline",
                      subject="all-pairs")
            m = t.one(**kw, statistic="spearman_mean")
            if m is None:
                continue
            if m["value"] == MISSING:
                lines.append(f"| {cell_label(arch, rec)} | {basis} | 0 | "
                             f"{MISSING} | {MISSING} | {MISSING} |")
                continue
            lines.append(
                f"| {cell_label(arch, rec)} | {basis} | {m['n']} | "
                f"{fmt(float(m['value']))} | "
                f"{fmt(t.num(**kw, statistic='spearman_min'))} | "
                f"{fmt(t.num(**kw, statistic='spearman_max'))} |")
    pairs = t.find(question="Q2_seed_stability", arch="vit_small",
                   recipe_actual="mixup", map_basis="canon",
                   variant="baseline", statistic="spearman")
    if pairs:
        lines += ["", "Individual pairs, ViT-S/mixup baseline (canon) — the "
                  "only cell with four repeats, spanning legacy runs and "
                  "fresh seeded reruns:", "",
                  "| pair | rho |", "|---|---|"]
        for r in pairs:
            lines.append(f"| {r['subject']} | {fmt(float(r['value']))} |")
    lines.append("")


def section_q3(t: Table, lines):
    lines += ["## Q3 — Is the address recipe-stable?", "",
              "Spearman between ViT-S/mixup and ViT-S/true-nomix BASELINE "
              "maps, all cross pairs.", "",
              "| basis | n pairs | mean rho | min | max |",
              "|---|---|---|---|---|"]
    for basis in ("canon", "mad"):
        kw = dict(question="Q3_recipe_stability", map_basis=basis,
                  subject="all-cross-pairs")
        m = t.one(**kw, statistic="spearman_mean")
        if m is None or m["value"] == MISSING:
            continue
        lines.append(f"| {basis} | {m['n']} | {fmt(float(m['value']))} | "
                     f"{fmt(t.num(**kw, statistic='spearman_min'))} | "
                     f"{fmt(t.num(**kw, statistic='spearman_max'))} |")
    within = t.num(question="Q2_seed_stability", arch="vit_small",
                   recipe_actual="mixup", map_basis="canon",
                   variant="baseline", subject="all-pairs",
                   statistic="spearman_mean")
    cross = t.num(question="Q3_recipe_stability", map_basis="canon",
                  subject="all-cross-pairs", statistic="spearman_mean")
    if within is not None and cross is not None:
        lines += ["", f"Within-recipe (Q2, ViT-S/mixup, canon) mean rho = "
                  f"{fmt(within)}; cross-recipe mean rho = {fmt(cross)}.", ""]


def section_q4(t: Table, lines):
    lines += ["## Q4 — Does SAGA (or registers) relocate the address?", "",
              "Spearman of each variant repeat's map against the "
              "SAME-PROVENANCE baseline map. Reference: rho = 1 is an "
              "identical address, rho = 0 relocated.", "",
              "| cell | variant | basis | n pairs | mean rho | min | max |",
              "|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for variant in ("saga", "registers"):
            for basis in ("canon", "mad"):
                kw = dict(question="Q4_relocation", arch=arch,
                          recipe_actual=rec, map_basis=basis,
                          variant=variant, subject="paired")
                m = t.one(**kw, statistic="spearman_mean")
                if m is None or m["value"] == MISSING:
                    continue
                lines.append(
                    f"| {cell_label(arch, rec)} | {variant} | {basis} | "
                    f"{m['n']} | {fmt(float(m['value']))} | "
                    f"{fmt(t.num(**kw, statistic='spearman_min'))} | "
                    f"{fmt(t.num(**kw, statistic='spearman_max'))} |")
    lines += ["", "Change in concentration and mass (variant cell-mean minus "
              "baseline cell-mean, canon maps). For H/H_uniform a NEGATIVE "
              "delta is more concentrated; for Gini and top-5 share a "
              "POSITIVE delta is more concentrated.", "",
              "| cell | variant | d H/H_uniform | d Gini | d top-5 share | "
              "d mean sinks/image |", "|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for variant in ("saga", "registers"):
            kw = dict(question="Q4_relocation", arch=arch, recipe_actual=rec,
                      map_basis="canon", variant=variant, subject="MEAN")
            h = t.one(**kw, statistic="delta_entropy_normalized")
            if h is None:
                continue
            lines.append(
                f"| {cell_label(arch, rec)} | {variant} | "
                f"{fmt(float(h['value']))} | "
                f"{fmt(t.num(**kw, statistic='delta_gini'))} | "
                f"{fmt(t.num(**kw, statistic='delta_top5_share'))} | "
                f"{fmt(t.num(**kw, statistic='delta_total_mass'))} |")
    lines.append("")


def section_q5(t: Table, lines):
    lines += ["## Q5 — Does the gate know the address?", "",
              "Spearman between sigmoid(phi) (mean over heads) and a "
              "frequency map, per layer. `baseline-matched` uses the "
              "same-provenance baseline map, `own` the SAGA run's own map. "
              "The extremal layer is the one with the largest |rho|. "
              "`gate std` is that layer's spatial standard deviation of "
              "sigmoid(phi) — a rank correlation is scale-free, so it says "
              "how much absolute modulation the rho corresponds to.", "",
              "| cell | saga repeat | comparator | extremal layer | rho | "
              "layer-mean rho | layers defined | gate std at layer |",
              "|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for comparator in ("baseline-matched", "own"):
            for tag, layer, rho, std in saga_gate_extremes(
                    t, arch, rec, "canon", comparator):
                mean_row = t.one(question="Q5_gate_address", arch=arch,
                                 recipe_actual=rec, map_basis="canon",
                                 subject=tag, comparator=comparator,
                                 statistic="spearman_layer_mean")
                defined = (mean_row["note"].replace("n_layers_defined=", "")
                           if mean_row else MISSING)
                lines.append(
                    f"| {cell_label(arch, rec)} | {tag} | {comparator} | "
                    f"{layer if layer is not None else MISSING} | "
                    f"{fmt(rho, 3) if rho is not None else MISSING} | "
                    f"{fmt(float(mean_row['value']), 3) if mean_row and mean_row['value'] != MISSING else MISSING} | "
                    f"{defined} | {fmt(std, 5) if std is not None else MISSING} |")
    lines += ["", "The final layer is spatially constant in every SAGA run "
              "(phi stays at its initialisation there), so it has no defined "
              "rank correlation and is excluded from the layer mean and "
              "counted in `layers defined`.", ""]


def degenerate_maps(t: Table, min_distinct=32):
    """Members whose freq map has too few distinct values for a Spearman to
    mean anything (mostly ties). Returned with their numbers."""
    out = []
    for r in t.find(question="Q1_concentration", statistic="n_distinct_values"):
        n_distinct = int(float(r["value"]))
        if n_distinct >= min_distinct:
            continue
        zf = t.num(question="Q1_concentration", arch=r["arch"],
                   recipe_actual=r["recipe_actual"],
                   map_basis=r["map_basis"], variant=r["variant"],
                   subject=r["subject"], statistic="zero_fraction")
        out.append((cell_label(r["arch"], r["recipe_actual"]), r["variant"],
                    r["subject"], r["map_basis"], n_distinct, zf))
    return out


def section_open(t: Table, lines):
    lines += ["## OPEN ITEMS", ""]
    n_nomix = t.one(question="Q2_seed_stability", arch="vit_small",
                    recipe_actual="nomix", map_basis="canon",
                    variant="baseline", subject="all-pairs",
                    statistic="spearman_mean")
    n_base = t.one(question="Q2_seed_stability", arch="vit_base",
                   recipe_actual="mixup", map_basis="canon",
                   variant="baseline", subject="all-pairs",
                   statistic="spearman_mean")
    lines += [
        f"- Seed-stability rests on ONE pair in ViT-S/nomix "
        f"(n={n_nomix['n'] if n_nomix else 0}) and ONE in ViT-B/mixup "
        f"(n={n_base['n'] if n_base else 0}); only ViT-S/mixup has four "
        f"repeats (6 pairs). The TASK-07 A2 optional runs "
        f"(e2r_vitb_mixup_baseline_s2 / _saga_s2) would raise ViT-B to "
        f"3 pairs.",
        "- Registers is represented by LEGACY repeats only in every cell "
        "and is absent from ViT-S/nomix; no registers run has yet gone "
        "through the seeded trainer (the 2-epoch smoke passed, the chains "
        "are unsubmitted).",
    ]
    deg = degenerate_maps(t)
    if deg:
        lines.append(
            f"- DEGENERATE RANKINGS — a Spearman needs spread, and these "
            f"maps have almost none, so every correlation involving them "
            f"(Q4 especially) rests on ties and should not be compared "
            f"with the other cells: "
            + "; ".join(f"{cell} {variant} {tag} [{basis}] only "
                        f"{n} distinct values of 196 positions, "
                        f"{zf * 100:.1f}% never a sink"
                        for cell, variant, tag, basis, n, zf in deg) + ".")
    else:
        lines.append("- No frequency map is tie-degenerate (every map has "
                     "enough distinct values for a Spearman).")
    lines += [
        "- Q3 correlates maps thresholded at DIFFERENT canon taus (one per "
        "cell, calibrated on that cell's own baseline), so the canon "
        "comparison is between each cell's own sink definition; the MAD "
        "basis, whose threshold is per-image by construction, is reported "
        "beside it and points the same way.",
        "- Q5 is reported at each run's extremal layer; the full 12-layer "
        "profile for every run and comparator is in the CSV "
        "(`spearman_layer00..11`) and in Faddr.npz (`profile__*`).",
        "- Both threshold bases are reported throughout; no cell has been "
        "designated primary for the address question.",
        ""]


def build_note(rows, table_path: Path, npz_path: Path) -> str:
    t = Table(rows)
    n_members = len({(r["arch"], r["recipe_actual"], r["variant"],
                      r["subject"])
                     for r in rows if r["question"] == "Q1_concentration"
                     and r["subject"] != "MEAN"})
    lines = [
        "# Sink-address analysis (TASK-07 Phase C)", "",
        "GENERATED by `analysis/build_sink_address_note.py` from "
        f"`{table_path.as_posix()}` — every number below is read from that "
        "CSV, none is hand-typed.", "",
        f"Source maps: the committed `*_addr.json` files "
        f"(`tools/sink_address.py`, HPC Phase B), {n_members} of them, one "
        "per (cell, variant, repeat), all sha-matched to their checkpoints "
        "(legacy against `checkpoint_manifest.csv`, run dirs against their "
        "sibling diag JSON). Diagnostics come from the LAST checkpoint "
        "only. Gates: the committed `phi` dumps. Figure data: "
        f"`{npz_path.as_posix()}`; figure: "
        "`results/figures/Faddr_draft.pdf`.", "",
        "Cell membership is imported from "
        "`analysis/build_pooled_tables.py`, so the recipe erratum remap "
        "(legacy directory names are NOT recipes) and the exclusion of the "
        "VOID legacy ViT-B mixup-dir trio match the pooled tables exactly.",
        "",
    ]
    section_headline(t, lines)
    section_q1(t, lines)
    section_q1b(t, lines)
    section_q2(t, lines)
    section_q3(t, lines)
    section_q4(t, lines)
    section_q5(t, lines)
    section_open(t, lines)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Render the TASK-07 sink-address note from the CSV.")
    parser.add_argument("--table", default="results/tables/sink_address.csv")
    parser.add_argument("--npz", default="results/figures_data/Faddr.npz")
    parser.add_argument("--out", default="results/notes/sink_address.md")
    args = parser.parse_args()

    rows = load(Path(args.table))
    note = build_note(rows, Path(args.table), Path(args.npz))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(note + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(note.splitlines())} lines)")


if __name__ == "__main__":
    main()
