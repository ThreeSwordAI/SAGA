#!/usr/bin/env python3
"""
plotting/plot_F3.py
===================
TASK-02 Phase 2 (2.4): draft of figure F3 from results/figures_data/F3_legacy.csv.

Two panels: x = sink_mad_k5; y = oversmooth_pairwise (left) and
oversmooth_pairwise_nosink (right). Color by variant, marker by arch,
recipe annotated at each point. One point per run (no seeds exist for the
legacy checkpoints).

    python plotting/plot_F3.py \
        [--data results/figures_data/F3_legacy.csv]
        [--out results/figures/F3_draft.pdf]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# colors pinned by TASK_02 spec 2.4
VARIANT_COLOR = {"baseline": "#888780", "registers": "#1D9E75",
                 "saga": "#7F77DD"}
ARCH_MARKER = {"vit_small": "o", "vit_base": "s"}
INK = "#3a3a3a"
MUTED = "#777777"


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k in ("sink_mad_k5", "oversmooth_pairwise",
                  "oversmooth_pairwise_nosink"):
            r[k] = float(r[k])
    return rows


def draw_panel(ax, rows, ykey, title):
    for r in rows:
        ax.scatter(r["sink_mad_k5"], r[ykey],
                   c=VARIANT_COLOR[r["variant"]],
                   marker=ARCH_MARKER[r["arch"]],
                   s=70, edgecolors="white", linewidths=1.2, zorder=3)
        # fixed per-recipe offsets so near-coincident points (e.g. the two
        # ViT-S registers runs) get non-colliding labels
        dx, dy = (5, 5) if r["recipe"] == "nomix" else (5, -11)
        ax.annotate(r["recipe"], (r["sink_mad_k5"], r[ykey]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7, color=MUTED, zorder=4)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("sink_mad_k5  (mean sink tokens / image, median + 5·MAD)",
                  fontsize=8, color=INK)
    ax.grid(True, color="#e3e3e3", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    parser = argparse.ArgumentParser(description="Render the F3 draft PDF.")
    parser.add_argument("--data", default="results/figures_data/F3_legacy.csv")
    parser.add_argument("--out", default="results/figures/F3_draft.pdf")
    args = parser.parse_args()

    rows = load(args.data)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    draw_panel(axes[0], rows, "oversmooth_pairwise",
               "Oversmoothing (pairwise)")
    draw_panel(axes[1], rows, "oversmooth_pairwise_nosink",
               "Oversmoothing (pairwise, sink-excluded)")
    axes[0].set_ylabel("mean pairwise cosine (patch tokens)",
                       fontsize=8, color=INK)

    legend_handles = (
        [Line2D([], [], marker="o", linestyle="", color=c, markersize=8,
                markeredgecolor="white", label=v)
         for v, c in VARIANT_COLOR.items()]
        + [Line2D([], [], marker=m, linestyle="", color="#555555",
                  markersize=8, markeredgecolor="white",
                  label=a.replace("vit_", "ViT-"))
           for a, m in ARCH_MARKER.items()]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("F3 draft — legacy e2 runs, corrected metrics "
                 "(one surviving repeat per cell)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png} (preview)")


if __name__ == "__main__":
    main()
