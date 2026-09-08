#!/usr/bin/env python3
"""
plotting/plot_address.py
========================
TASK-07 C2: draw the sink-address maps.

Reads results/figures_data/Faddr.npz + results/tables/sink_address.csv
(both from analysis/address_analysis.py) and writes a multi-page
results/figures/Faddr_draft.pdf:

- one page per cell (arch x recipe_actual): a TOP row of 14x14 sink
  frequency maps — every baseline repeat, then SAGA, then registers — with
  ONE shared colorbar for the row (so the panels are comparable by eye);
  a BOTTOM row with, per SAGA repeat, the gate map sigmoid(phi) of that
  run's most (anti)correlated layer, shared 0..1 colorbar, titled with the
  layer index and its Spearman against the address.
- a final page of concentration bars (entropy vs uniform, Gini, top-5
  share) per cell and variant.

    python plotting/plot_address.py [--basis canon]

`*/figures/` is git-ignored, so the PDF is committed with `git add -f`
(same as every other draft figure in this project).
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

INK = "#3a3a3a"
MUTED = "#777777"
VARIANT_ORDER = ["baseline", "saga", "registers"]
CELLS = [("vit_small", "mixup"), ("vit_small", "nomix"),
         ("vit_base", "mixup")]


def load_npz(path: Path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def members_of(data, arch, rec):
    """{variant: [(tag, key)]} in VARIANT_ORDER, tags in npz order."""
    out = {v: [] for v in VARIANT_ORDER}
    for key in data["member_keys"]:
        k_arch, k_rec, k_var, k_tag = str(key).split("|")
        if (k_arch, k_rec) == (arch, rec) and k_var in out:
            out[k_var].append((k_tag, str(key)))
    return out


def extremal_layer(rows, arch, rec, tag, basis, comparator="baseline-matched"):
    """(layer, rho, p_bonferroni) of the most (anti)correlated gate layer."""
    for r in rows:
        if (r["question"] == "Q5_gate_address" and r["arch"] == arch
                and r["recipe_actual"] == rec and r["subject"] == tag
                and r["map_basis"] == basis
                and r["comparator"] == comparator
                and r["statistic"] == "spearman_layer_absmax"
                and r["value"] != "MISSING"):
            note = r["note"]
            layer = int(note.split("layer=")[1].split(";")[0])
            bonf = None
            if "Bonferroni p=" in note:
                bonf = float(note.split("Bonferroni p=")[1].split()[0])
            return layer, float(r["value"]), bonf
    return None


def noisy_subjects(rows, basis, threshold=0.5):
    """{(arch, rec, variant, tag)} whose per-position estimate is mostly
    Monte-Carlo noise — their concentration bars must be marked."""
    out = set()
    for r in rows:
        if (r["question"] == "Q1_concentration"
                and r["statistic"] == "position_rel_se"
                and r["map_basis"] == basis and r["subject"] != "MEAN"
                and r["value"] not in ("MISSING", "")
                and float(r["value"]) > threshold):
            out.add((r["arch"], r["recipe_actual"], r["variant"]))
    return out


def cell_figure(data, rows, arch, rec, basis):
    """One cell's page as a Figure, or None when the cell has no members."""
    members = members_of(data, arch, rec)
    freq_items = [(v, tag, key) for v in VARIANT_ORDER
                  for tag, key in members[v]]
    if not freq_items:
        return None
    gate_items = [(tag, key) for tag, key in members["saga"]
                  if f"gate__{key}" in data]

    ncols = max(len(freq_items), max(len(gate_items), 1))
    nrows = 2 if gate_items else 1
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(2.05 * ncols + 0.8, 2.6 * nrows + 0.6))

    maps = [data[f"freq_{basis}__{key}"] for _, _, key in freq_items]
    side = int(round(maps[0].size ** 0.5))
    vmax = float(max(m.max() for m in maps))

    for ax, (variant, tag, key), m in zip(axes[0], freq_items, maps):
        im = ax.imshow(m.reshape(side, side), vmin=0.0, vmax=vmax,
                       cmap="magma")
        ax.set_title(f"{variant}\n{tag}", fontsize=8, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axes[0][len(freq_items):]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.022, pad=0.01)
    cbar.set_label(f"P(position is a sink)  [{basis}]", fontsize=7,
                   color=MUTED)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)

    if gate_items:
        # the gate modulates by only ~0.005-0.02 around a ~0.5-0.65 base, so
        # a 0..1 colorbar renders every panel a flat block. Share ONE scale
        # across the row (panels stay comparable) but set it from the row's
        # true range, and print that range on the bar so the small absolute
        # amplitude behind a large rank correlation is never hidden.
        chosen = []
        for tag, key in gate_items:
            gate = data[f"gate__{key}"]
            ext = extremal_layer(rows, arch, rec, tag, basis)
            layer = ext[0] if ext else gate.shape[0] // 2
            chosen.append((tag, gate[layer], layer, ext))
        gmin = float(min(m.min() for _, m, _, _ in chosen))
        gmax = float(max(m.max() for _, m, _, _ in chosen))
        for ax, (tag, m, layer, ext) in zip(axes[1], chosen):
            gim = ax.imshow(m.reshape(side, side), vmin=gmin, vmax=gmax,
                            cmap="viridis")
            title = f"saga {tag}\nlayer {layer}"
            if ext:
                title += f", rho={ext[1]:+.3f}"
                # the layer is an argmax over 12, so show the corrected p
                if ext[2] is not None:
                    title += (f"\nBonf p={ext[2]:.3f}"
                              + ("" if ext[2] < 0.05 else " (n.s.)"))
            title += f"  std={m.std():.4f}"
            ax.set_title(title, fontsize=7.5, color=INK)
            ax.set_xticks([]), ax.set_yticks([])
        for ax in axes[1][len(chosen):]:
            ax.axis("off")
        gbar = fig.colorbar(gim, ax=axes[1].tolist(), fraction=0.022,
                            pad=0.01)
        gbar.set_label(f"sigmoid(phi), mean over heads\n"
                       f"row scale {gmin:.4f}..{gmax:.4f} (NOT 0..1)",
                       fontsize=7, color=MUTED)
        gbar.ax.tick_params(labelsize=7, colors=MUTED)

    fig.suptitle(f"{arch} | recipe_actual={rec} — sink frequency per "
                 f"position (top) and the most (anti)correlated gate layer "
                 f"(bottom)", fontsize=11, color=INK)
    return fig


def concentration_figure(rows, basis):
    """Concentration as EXCESS over the finite-sample null.

    Plotting the RAW statistics against a zero reference makes the
    sparsest map (ViT-B registers, 0.02 sinks/image) the tallest Gini bar
    in the figure, purely from Monte-Carlo noise. Excess subtracts each
    map's own null, and any map that is mostly noise is hatched.
    """
    stats = [("entropy_normalized_excess",
              "excess H / H_uniform  (0 = uniform-like)"),
             ("gini_excess", "excess Gini  (0 = uniform-like)"),
             ("top5_share_excess", "excess top-5 share")]
    fig, axes = plt.subplots(1, len(stats), figsize=(4.1 * len(stats), 3.6))
    colors = {"baseline": "#4c72b0", "saga": "#c44e52",
              "registers": "#55a868"}
    noisy = noisy_subjects(rows, basis)
    for ax, (stat, title) in zip(axes, stats):
        xs, heights, cs, hatches = [], [], [], []
        for arch, rec in CELLS:
            for variant in VARIANT_ORDER:
                hit = [r for r in rows
                       if r["question"] == "Q1_concentration"
                       and r["arch"] == arch and r["recipe_actual"] == rec
                       and r["variant"] == variant and r["subject"] == "MEAN"
                       and r["map_basis"] == basis
                       and r["statistic"] == stat]
                if not hit or hit[0]["value"] in ("MISSING", ""):
                    continue
                flagged = (arch, rec, variant) in noisy
                xs.append(f"{arch.replace('vit_', '')}|{rec}\n{variant} "
                          f"(n={hit[0]['n']})" + ("\nNOISY" if flagged else ""))
                heights.append(float(hit[0]["value"]))
                cs.append(colors[variant])
                hatches.append("///" if flagged else "")
        pos = np.arange(len(heights))
        bars = ax.bar(pos, heights, color=cs, width=0.72)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("white")
        ax.axhline(0.0, color=MUTED, ls="--", lw=1)
        ax.annotate("null", (0.02, 0.0), xycoords=("axes fraction", "data"),
                    fontsize=7, color=MUTED, va="bottom")
        ax.set_xticks(pos)
        ax.set_xticklabels(xs, fontsize=6, rotation=90, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Address concentration as EXCESS over each map's "
                 f"finite-sample null, mean over repeats  [{basis} map]\n"
                 f"hatched = per-position relative standard error > 0.5 "
                 f"(mostly Monte-Carlo noise)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def main():
    parser = argparse.ArgumentParser(description="TASK-07 C2 address maps.")
    parser.add_argument("--npz", default="results/figures_data/Faddr.npz")
    parser.add_argument("--table", default="results/tables/sink_address.csv")
    parser.add_argument("--out", default="results/figures/Faddr_draft.pdf")
    parser.add_argument("--basis", default="canon", choices=["canon", "mad"])
    args = parser.parse_args()

    data = load_npz(Path(args.npz))
    rows = load_rows(Path(args.table))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_pages = 0
    first_cell_fig = None
    with PdfPages(out) as pdf:
        for arch, rec in CELLS:
            fig = cell_figure(data, rows, arch, rec, args.basis)
            if fig is None:
                print(f"  skip {arch}|{rec}: no members")
                continue
            pdf.savefig(fig)
            n_pages += 1
            if first_cell_fig is None:
                first_cell_fig = fig
            else:
                plt.close(fig)
        conc = concentration_figure(rows, args.basis)
        pdf.savefig(conc)
        n_pages += 1
        plt.close(conc)

    # the PDF is the artifact; a png of the first cell page is the quick
    # look (same convention as the project's other draft figures)
    if first_cell_fig is not None:
        first_cell_fig.savefig(out.with_suffix(".png"), dpi=130)
        plt.close(first_cell_fig)
    print(f"wrote {out} ({n_pages} pages) + "
          f"{out.with_suffix('.png').name} preview")


if __name__ == "__main__":
    main()
