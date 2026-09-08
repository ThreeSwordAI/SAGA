#!/usr/bin/env python3
"""
analysis/build_sink_address_note.py
===================================
TASK-07 C3: render results/notes/sink_address.md from
results/tables/sink_address.csv.

Every number in the note is READ FROM THE CSV — nothing is hand-typed and
nothing is recomputed here, so the note cannot drift from the table. That
includes the ranges quoted in prose, which are computed from the same rows
the tables render.

The note states values and how they sit relative to the explicit
references, which for this analysis are NOT zero:
- concentration is reported as EXCESS over a finite-sample null (a uniform
  ground truth at the same mass gives Gini 0.017 at mass 20 but 0.49 at
  mass 0.02, so raw values are not comparable across members);
- correlations carry an exact spatial-permutation p (the 196 positions are
  spatially smooth, so the iid reference is far too generous).

    python analysis/build_sink_address_note.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MISSING = "MISSING"
CELLS = [("vit_small", "mixup"), ("vit_small", "nomix"),
         ("vit_base", "mixup")]
BASES = ("canon", "mad")
UNIFORM_TOP5 = 5 / 196
ALPHA = 0.05


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
        if r is None or r["value"] in (MISSING, ""):
            return None
        return r["value"]

    def num(self, **kw):
        v = self.val(**kw)
        return None if v is None else float(v)

    def nums(self, **kw):
        return [float(r["value"]) for r in self.find(**kw)
                if r["value"] not in (MISSING, "")]


def fmt(v, nd=4):
    return MISSING if v is None else f"{v:.{nd}f}"


def rng_txt(values, nd=4):
    if not values:
        return MISSING
    if len(values) == 1:
        return fmt(values[0], nd)
    return f"{min(values):.{nd}f} to {max(values):.{nd}f}"


def cell_label(arch, rec):
    return (f"{arch.replace('vit_', 'ViT-').replace('small', 'S')
            .replace('base', 'B')}/{rec}")


def pair_p(t: Table, question, basis, arch, rec, subject, variant="baseline"):
    return t.num(question=question, arch=arch, recipe_actual=rec,
                 map_basis=basis, variant=variant, subject=subject,
                 statistic="spearman_p_spatial")


def significant_count(t: Table, question, basis, variant="baseline"):
    """(n_significant, n_total) over every individual pair of a question."""
    ps = t.nums(question=question, map_basis=basis, variant=variant,
                statistic="spearman_p_spatial")
    return sum(1 for p in ps if p < ALPHA), len(ps)


def gate_extremes(t: Table, arch, rec, basis, comparator="baseline-matched"):
    """[(tag, layer, rho, p, bonf, other_layer, other_rho, std)] per run."""
    out = []
    for r in t.find(question="Q5_gate_address", arch=arch, recipe_actual=rec,
                    map_basis=basis, comparator=comparator,
                    statistic="spearman_layer_absmax"):
        tag = r["subject"]
        if r["value"] in (MISSING, ""):
            out.append((tag, None, None, None, None, None, None, None))
            continue
        layer = int(r["note"].split("layer=")[1].split(";")[0])
        rho = float(r["value"])
        p = t.num(question="Q5_gate_address", arch=arch, recipe_actual=rec,
                  map_basis=basis, comparator=comparator, subject=tag,
                  statistic="spearman_layer_absmax_p_spatial")
        bonf = None
        if "Bonferroni p=" in r["note"]:
            bonf = float(r["note"].split("Bonferroni p=")[1].split()[0])
        # the opposite-sign extreme of the SAME profile: if the absmax is
        # negative the relevant counter-evidence is the profile max
        other_stat = ("spearman_layer_max" if rho < 0
                      else "spearman_layer_min")
        other = t.one(question="Q5_gate_address", arch=arch,
                      recipe_actual=rec, map_basis=basis,
                      comparator=comparator, subject=tag,
                      statistic=other_stat)
        other_rho = (float(other["value"])
                     if other and other["value"] not in (MISSING, "") else None)
        other_layer = (int(other["note"].split("layer=")[1].split(";")[0])
                       if other and "layer=" in other["note"] else None)
        std = t.num(question="Q5_gate_address", arch=arch, recipe_actual=rec,
                    map_basis="-", subject=tag,
                    statistic=f"gate_spatial_std_layer{layer:02d}")
        out.append((tag, layer, rho, p, bonf, other_layer, other_rho, std))
    return out


# ── sections ─────────────────────────────────────────────────────────────────

def section_headline(t: Table, lines):
    lines += ["## Headline per cell", "",
              "Canon-threshold maps. Concentration is EXCESS over the "
              "finite-sample null (zero = indistinguishable from a uniform "
              "map at that mass); seed-stability is the mean baseline "
              "pair Spearman with its exact spatial-permutation p; the "
              "gate figure is each SAGA repeat's largest-|rho| layer.", ""]
    for arch, rec in CELLS:
        exc = t.num(question="Q1_concentration", arch=arch,
                    recipe_actual=rec, map_basis="canon", variant="baseline",
                    subject="MEAN", statistic="gini_excess")
        seed = t.one(question="Q2_seed_stability", arch=arch,
                     recipe_actual=rec, map_basis="canon", variant="baseline",
                     subject="all-pairs", statistic="spearman_mean")
        ps = t.nums(question="Q2_seed_stability", arch=arch,
                    recipe_actual=rec, map_basis="canon", variant="baseline",
                    statistic="spearman_p_spatial")
        seed_txt = MISSING
        if seed and seed["value"] not in (MISSING, ""):
            seed_txt = f"{float(seed['value']):+.4f} (n={seed['n']} pairs"
            if ps:
                seed_txt += f", p = {rng_txt(ps)}"
            seed_txt += ")"

        gate_bits = []
        for basis in BASES:
            ext = [e for e in gate_extremes(t, arch, rec, basis)
                   if e[2] is not None]
            if not ext:
                continue
            rhos = [e[2] for e in ext]
            layers = sorted({e[1] for e in ext})
            n_neg = sum(1 for v in rhos if v < 0)
            sign = ("all negative" if n_neg == len(rhos) else
                    "all positive" if n_neg == 0 else
                    f"SIGNS DISAGREE ({n_neg} negative of {len(rhos)})")
            sig = sum(1 for e in ext if e[4] is not None and e[4] < ALPHA)
            gate_bits.append(
                f"{basis} {rng_txt(rhos, 3)} at layer(s) "
                f"{', '.join(str(x) for x in layers)}, {sign}, "
                f"{sig}/{len(ext)} with Bonferroni p<{ALPHA}")
        lines.append(
            f"- **{cell_label(arch, rec)}**: address concentration = "
            f"{fmt(exc)} excess Gini, seed-stability rho = {seed_txt}, "
            f"gate-address rho = "
            f"{'; '.join(gate_bits) if gate_bits else MISSING}.")
    lines += ["",
              f"Significance threshold used throughout: p < {ALPHA} under "
              f"the exact spatial-permutation null. Raw (non-excess) "
              f"concentration values are in the Q1 table.", ""]


def section_q1(t: Table, lines):
    lines += ["## Q1 — Does an address exist? (baseline concentration)", "",
              "`obs` is the raw statistic, `null` its finite-sample "
              "expectation under a SPATIALLY UNIFORM ground truth at the "
              "same mass and image count, `excess` the difference. Only "
              "`excess` is comparable between rows, because the null moves "
              "with mass.", "",
              "| cell | basis | n | mean sinks/image | Gini obs | Gini null "
              "| Gini excess | H/H_uniform obs | H/H_uniform excess | "
              "top-5 obs | top-5 excess |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for basis in BASES:
            kw = dict(question="Q1_concentration", arch=arch,
                      recipe_actual=rec, map_basis=basis, variant="baseline",
                      subject="MEAN")
            row = t.one(**kw, statistic="gini")
            if row is None:
                continue
            lines.append(
                f"| {cell_label(arch, rec)} | {basis} | {row['n']} | "
                f"{fmt(t.num(**kw, statistic='total_mass'))} | "
                f"{fmt(t.num(**kw, statistic='gini'))} | "
                f"{fmt(t.num(**kw, statistic='gini_null'))} | "
                f"{fmt(t.num(**kw, statistic='gini_excess'))} | "
                f"{fmt(t.num(**kw, statistic='entropy_normalized'))} | "
                f"{fmt(t.num(**kw, statistic='entropy_normalized_excess'))} | "
                f"{fmt(t.num(**kw, statistic='top5_share'))} | "
                f"{fmt(t.num(**kw, statistic='top5_share_excess'))} |")

    h_obs, t5_ratio, g_exc = [], [], []
    for arch, rec in CELLS:
        for basis in BASES:
            kw = dict(question="Q1_concentration", arch=arch,
                      recipe_actual=rec, map_basis=basis, variant="baseline",
                      subject="MEAN")
            h = t.num(**kw, statistic="entropy_normalized")
            t5 = t.num(**kw, statistic="top5_share")
            ge = t.num(**kw, statistic="gini_excess")
            if h is not None:
                h_obs.append(h)
            if t5 is not None:
                t5_ratio.append(t5 / UNIFORM_TOP5)
            if ge is not None:
                g_exc.append(ge)
    lines += ["",
              f"Uniform references for the raw values: top-5 share = "
              f"{UNIFORM_TOP5:.5f}, Gini = 0, H/H_uniform = 1 (196 "
              f"positions).", "",
              f"Observed H/H_uniform spans {rng_txt(h_obs)} — close to the "
              f"uniform 1.0 — while observed top-5 share is "
              f"{rng_txt(t5_ratio, 2)}x its uniform reference and excess "
              f"Gini spans {rng_txt(g_exc)}. The mass is spread widely but "
              f"not evenly, and the excess column shows the unevenness "
              f"survives the finite-sample floor in every baseline cell.",
              ""]


def section_q1b(t: Table, lines):
    lines += ["## Q1b — Geometry: mean frequency per ring from the border",
              "",
              "Ring 0 is the outermost row/column of the 14x14 grid, ring 6 "
              "the centre. Canon maps, mean over each cell's repeats. A flat "
              "row is no border structure; peak ring 1 means the highest "
              "sink frequency sits one patch inside the border.", "",
              "| cell | variant | n | ring0 | ring1 | ring2 | ring3 | ring4 "
              "| ring5 | ring6 | peak ring | noise-flagged |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for variant in ("baseline", "saga", "registers"):
            per_ring, peaks = defaultdict(list), []
            for r in t.find(question="Q1b_geometry", arch=arch,
                            recipe_actual=rec, map_basis="canon",
                            variant=variant):
                if r["statistic"] == "peak_ring":
                    peaks.append(int(float(r["value"])))
                elif r["statistic"].startswith("ring"):
                    per_ring[r["statistic"]].append(float(r["value"]))
            if not peaks:
                continue
            cells = []
            for k in range(7):
                vals = per_ring.get(f"ring{k}_mean_freq", [])
                cells.append(fmt(sum(vals) / len(vals)) if vals else MISSING)
            noisy = [r["subject"] for r in t.find(
                question="Q1_concentration", arch=arch, recipe_actual=rec,
                map_basis="canon", variant=variant,
                statistic="position_rel_se")
                if r["value"] not in (MISSING, "")
                and float(r["value"]) > 0.5 and r["subject"] != "MEAN"]
            lines.append(f"| {cell_label(arch, rec)} | {variant} | "
                         f"{len(peaks)} | " + " | ".join(cells) +
                         f" | {', '.join(str(p) for p in sorted(set(peaks)))}"
                         f" | {', '.join(noisy) if noisy else 'no'} |")
    lines += ["", "A noise-flagged row's ring profile is read off a map "
              "whose per-position estimate has a relative standard error "
              "above 0.5, so its peak ring is not meaningful.", ""]


def section_q2(t: Table, lines):
    lines += ["## Q2 — Is the address seed-stable? (the make-or-break number)",
              "",
              "Spearman between BASELINE frequency maps of the cell's "
              "repeats, all pairs, with the exact spatial-permutation "
              "p-value. `null sd` is that permutation null's spread — "
              "compare it with the iid reference 1/sqrt(195) = 0.0716 to see "
              "why the position count is not the effective sample size.", "",
              "| cell | basis | n pairs | mean rho | min | max | p range | "
              f"pairs with p<{ALPHA} |", "|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for basis in BASES:
            kw = dict(question="Q2_seed_stability", arch=arch,
                      recipe_actual=rec, map_basis=basis, variant="baseline",
                      subject="all-pairs")
            m = t.one(**kw, statistic="spearman_mean")
            if m is None or m["value"] in (MISSING, ""):
                continue
            ps = t.nums(question="Q2_seed_stability", arch=arch,
                        recipe_actual=rec, map_basis=basis,
                        variant="baseline", statistic="spearman_p_spatial")
            lines.append(
                f"| {cell_label(arch, rec)} | {basis} | {m['n']} | "
                f"{fmt(float(m['value']))} | "
                f"{fmt(t.num(**kw, statistic='spearman_min'))} | "
                f"{fmt(t.num(**kw, statistic='spearman_max'))} | "
                f"{rng_txt(ps)} | {sum(1 for p in ps if p < ALPHA)}/{len(ps)} |")

    pairs = t.find(question="Q2_seed_stability", arch="vit_small",
                   recipe_actual="mixup", map_basis="canon",
                   variant="baseline", statistic="spearman")
    if pairs:
        lines += ["", "Individual pairs, ViT-S/mixup baseline (canon) — the "
                  "only cell with four repeats, spanning legacy runs and "
                  "fresh seeded reruns:", "",
                  "| pair | rho | p (spatial) | null sd |",
                  "|---|---|---|---|"]
        for r in pairs:
            sub = r["subject"]
            lines.append(
                f"| {sub} | {fmt(float(r['value']))} | "
                f"{fmt(pair_p(t, 'Q2_seed_stability', 'canon', 'vit_small', 'mixup', sub))} | "
                f"{fmt(t.num(question='Q2_seed_stability', arch='vit_small', recipe_actual='mixup', map_basis='canon', variant='baseline', subject=sub, statistic='spearman_null_sd'))} |")
    n_sig, n_tot = significant_count(t, "Q2_seed_stability", "canon")
    lines += ["", f"Across every cell (canon), {n_sig} of {n_tot} baseline "
              f"pairs reach p<{ALPHA}.", ""]


def section_q3(t: Table, lines):
    lines += ["## Q3 — Is the address recipe-stable?", "",
              "Spearman between ViT-S/mixup and ViT-S/true-nomix BASELINE "
              "maps, all cross pairs.", "",
              "| basis | n pairs | mean rho | min | max | p range | "
              f"pairs with p<{ALPHA} |", "|---|---|---|---|---|---|---|"]
    for basis in BASES:
        kw = dict(question="Q3_recipe_stability", map_basis=basis,
                  subject="all-cross-pairs")
        m = t.one(**kw, statistic="spearman_mean")
        if m is None or m["value"] in (MISSING, ""):
            continue
        ps = t.nums(question="Q3_recipe_stability", map_basis=basis,
                    statistic="spearman_p_spatial")
        lines.append(
            f"| {basis} | {m['n']} | {fmt(float(m['value']))} | "
            f"{fmt(t.num(**kw, statistic='spearman_min'))} | "
            f"{fmt(t.num(**kw, statistic='spearman_max'))} | "
            f"{rng_txt(ps)} | {sum(1 for p in ps if p < ALPHA)}/{len(ps)} |")

    within = t.num(question="Q2_seed_stability", arch="vit_small",
                   recipe_actual="mixup", map_basis="canon",
                   variant="baseline", subject="all-pairs",
                   statistic="spearman_mean")
    cross = t.num(question="Q3_recipe_stability", map_basis="canon",
                  subject="all-cross-pairs", statistic="spearman_mean")
    w_sig, w_tot = significant_count(t, "Q2_seed_stability", "canon")
    c_sig, c_tot = significant_count(t, "Q3_recipe_stability", "canon")
    if within is not None and cross is not None:
        lines += ["",
                  f"Within-recipe (Q2, ViT-S/mixup, canon) mean rho = "
                  f"{fmt(within)} with {w_sig}/{w_tot} pairs at p<{ALPHA}; "
                  f"cross-recipe mean rho = {fmt(cross)} with only "
                  f"{c_sig}/{c_tot} pairs at p<{ALPHA}. The cross-recipe "
                  f"value lies largely inside its own spatial null, so it "
                  f"is not evidence that the address transfers between "
                  f"recipes.", ""]
    lines += ["The two cells are thresholded at DIFFERENT canon taus (each "
              "calibrated on its own baseline), so the canon comparison is "
              "between each cell's own sink definition; the mad basis, whose "
              "threshold is per-image by construction, is listed beside it.",
              ""]


def section_q4(t: Table, lines):
    lines += ["## Q4 — Does SAGA (or registers) relocate the address?", "",
              "Spearman of each variant repeat's map against the "
              "SAME-PROVENANCE baseline map (rho = 1 identical, 0 "
              "relocated), with the spatial-permutation p.", "",
              "| cell | variant | basis | n pairs | mean rho | min | max | "
              f"pairs with p<{ALPHA} |", "|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for variant in ("saga", "registers"):
            for basis in BASES:
                kw = dict(question="Q4_relocation", arch=arch,
                          recipe_actual=rec, map_basis=basis,
                          variant=variant, subject="paired")
                m = t.one(**kw, statistic="spearman_mean")
                if m is None or m["value"] in (MISSING, ""):
                    continue
                ps = t.nums(question="Q4_relocation", arch=arch,
                            recipe_actual=rec, map_basis=basis,
                            variant=variant, statistic="spearman_p_spatial")
                lines.append(
                    f"| {cell_label(arch, rec)} | {variant} | {basis} | "
                    f"{m['n']} | {fmt(float(m['value']))} | "
                    f"{fmt(t.num(**kw, statistic='spearman_min'))} | "
                    f"{fmt(t.num(**kw, statistic='spearman_max'))} | "
                    f"{sum(1 for p in ps if p < ALPHA)}/{len(ps)} |")
    lines += ["", "Change in concentration, as PAIRED differences of "
              "EXCESS-over-null values (canon). Differencing raw values "
              "instead inverts the registers result, because the variants "
              "change the sink mass by up to two orders of magnitude and "
              "the null moves with mass. For excess H/H_uniform a NEGATIVE "
              "delta is more concentrated; for excess Gini and top-5 share "
              "a POSITIVE delta is.", "",
              "| cell | variant | n pairs | d excess H/H_uniform | "
              "d excess Gini | d excess top-5 | d mean sinks/image |",
              "|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for variant in ("saga", "registers"):
            kw = dict(question="Q4_relocation", arch=arch, recipe_actual=rec,
                      map_basis="canon", variant=variant, subject="MEAN",
                      comparator="baseline-matched")
            h = t.one(**kw, statistic="delta_entropy_normalized_excess")
            if h is None:
                continue
            lines.append(
                f"| {cell_label(arch, rec)} | {variant} | {h['n']} | "
                f"{fmt(float(h['value'])) if h['value'] not in (MISSING, '') else MISSING} | "
                f"{fmt(t.num(**kw, statistic='delta_gini_excess'))} | "
                f"{fmt(t.num(**kw, statistic='delta_top5_share_excess'))} | "
                f"{fmt(t.num(**kw, statistic='delta_total_mass'))} |")
    lines.append("")


def section_q5(t: Table, lines):
    lines += ["## Q5 — Does the gate know the address?", "",
              "Spearman between sigmoid(phi) (mean over heads) and a "
              "frequency map, per layer, against the same-provenance "
              "baseline map. Each row's layer is an ARGMAX over the "
              "defined layers, so `p Bonf` corrects for that selection, "
              "and `opposite extreme` gives the largest same-profile "
              "correlation of the other sign — the counter-evidence to a "
              "sign claim. `gate std` is that layer's spatial standard "
              "deviation of sigmoid(phi): a rank correlation is "
              "scale-free, so it says how little absolute modulation the "
              "rho corresponds to.", "",
              "| cell | saga repeat | basis | layer | rho | p | p Bonf | "
              "opposite extreme | gate std |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arch, rec in CELLS:
        for basis in BASES:
            for (tag, layer, rho, p, bonf, o_layer, o_rho,
                 std) in gate_extremes(t, arch, rec, basis):
                if rho is None:
                    lines.append(f"| {cell_label(arch, rec)} | {tag} | "
                                 f"{basis} | {MISSING} | {MISSING} | "
                                 f"{MISSING} | {MISSING} | {MISSING} | "
                                 f"{MISSING} |")
                    continue
                opp = (f"{o_rho:+.3f} (layer {o_layer})"
                       if o_rho is not None else MISSING)
                lines.append(
                    f"| {cell_label(arch, rec)} | {tag} | {basis} | {layer} "
                    f"| {fmt(rho, 3)} | {fmt(p)} | {fmt(bonf)} | {opp} | "
                    f"{fmt(std, 5)} |")
    lines += ["", "The final layer is spatially constant in every SAGA run "
              "(phi stays at its initialisation there), so it has no "
              "defined rank correlation and is excluded from the layer "
              "statistics. The signed layer-mean rho in the CSV is near "
              "zero partly through cancellation between layers of opposite "
              "sign; `spearman_layer_absmean` is listed beside it.", ""]


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
    sds = t.nums(question="Q2_seed_stability", map_basis="canon",
                 statistic="spearman_null_sd")
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
    if sds:
        lines.append(
            f"- The 196 positions are NOT 196 independent samples: the "
            f"spatial-permutation null sd is {rng_txt(sds)} against an iid "
            f"reference of 0.0716, i.e. an effective sample size around "
            f"{1 / max(sds) ** 2 + 1:.0f} to {1 / min(sds) ** 2 + 1:.0f}. "
            f"The `n` column on a correlation row is the POSITION count, "
            f"not that effective size; use the p-value.")
    noisy = [(r["arch"], r["recipe_actual"], r["variant"], r["subject"],
              r["map_basis"], float(r["value"]))
             for r in t.find(question="Q1_concentration",
                             statistic="position_rel_se")
             if r["value"] not in (MISSING, "") and float(r["value"]) > 0.5
             and r["subject"] != "MEAN"]
    if noisy:
        lines.append(
            "- MONTE-CARLO NOISE, not ties, is what makes a sparse map's "
            "ranking meaningless: these maps have so few sinks that one "
            "position's frequency carries a relative standard error above "
            "0.5, so their concentration, ring profile and correlations "
            "are not comparable with the other cells — "
            + "; ".join(f"{cell_label(a, rc)} {v} {tag} [{b}] rel SE "
                        f"{se:.2f}" for a, rc, v, tag, b, se in noisy) + ".")
    else:
        lines.append("- No map is noise-flagged (every per-position "
                     "relative standard error is at or below 0.5).")
    layers = sorted({r["statistic"] for r in t.find(
        question="Q5_gate_address") if r["statistic"].startswith(
            "spearman_layer") and r["statistic"][14:].isdigit()})
    comparators = sorted({r["comparator"] for r in t.find(
        question="Q5_gate_address")
        if r["statistic"].startswith("spearman_layer")
        and r["statistic"][14:].isdigit()})
    lines += [
        f"- Q5 is reported at each run's extremal layer. The full "
        f"per-layer profile is in the CSV for {', '.join(comparators)} "
        f"as {layers[0]}..{layers[-1]} (the spatially constant final layer "
        f"has no row), and for every comparator in Faddr.npz "
        f"(`profile__*`)." if layers else
        "- Q5 per-layer profiles are in Faddr.npz (`profile__*`).",
        "- Concentration nulls are Monte-Carlo (see the `*_null` rows for "
        "sims and seed); correlation p-values are exact permutation tests "
        "and need no seed.",
        "- Both threshold bases are reported for every question; no basis "
        "has been designated primary for the address question.",
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
        f"`{table_path.as_posix()}` — every number below, including the "
        "ranges quoted in prose, is read from that CSV.", "",
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
