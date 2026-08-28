#!/usr/bin/env python3
"""
plotting/plot_F6.py
===================
TASK-02B Phase A2: F6 relocation draft from results/figures_data/F6_legacy.csv.

2x2 outer grid of panels (arch x recipe); inside each, two stacked sub-rows
sharing the block-index x axis: CLS norm-ratio (top) and CLS attention-share
(bottom). Three curves per sub-row (variant colors). The figure answers:
does SAGA raise CLS norm/attention share relative to baseline — i.e. does
outlier mass relocate to the ungated CLS token?

    python plotting/plot_F6.py \
        [--data results/figures_data/F6_legacy.csv]
        [--out results/figures/F6_draft.pdf]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VARIANT_COLOR = {"baseline": "#888780", "registers": "#1D9E75",
                 "saga": "#7F77DD"}
ARCH_NAME = {"vit_small": "ViT-S/16", "vit_base": "ViT-B/16"}
INK = "#3a3a3a"
MUTED = "#777777"


def load(path):
    """(arch, recipe, variant) -> {'block': [...], 'ratio': [...], 'share': [...]}"""
    series = defaultdict(lambda: {"block": [], "ratio": [], "share": []})
    for r in csv.DictReader(open(path, newline="")):
        s = series[(r["arch"], r["recipe"], r["variant"])]
        s["block"].append(int(r["block"]))
        s["ratio"].append(float(r["cls_norm_ratio"]))
        s["share"].append(float(r["cls_attn_share"])
                          if r["cls_attn_share"] != "" else None)
    return series


def style(ax, ylabel=None, xlabel=None):
    ax.grid(True, color="#e3e3e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK)


def main():
    parser = argparse.ArgumentParser(description="Render the F6 draft PDF.")
    parser.add_argument("--data", default="results/figures_data/F6_legacy.csv")
    parser.add_argument("--out", default="results/figures/F6_draft.pdf")
    args = parser.parse_args()

    series = load(args.data)
    archs = ["vit_small", "vit_base"]
    recipes = ["mixup", "nomix"]

    # outer 2x2 (arch x recipe); each panel = 2 stacked sub-rows -> 4x2 axes
    fig, axes = plt.subplots(4, 2, figsize=(9.5, 10.0), sharex=True)
    for ai, arch in enumerate(archs):
        for ri, recipe in enumerate(recipes):
            ax_ratio = axes[2 * ai][ri]
            ax_share = axes[2 * ai + 1][ri]
            for variant, color in VARIANT_COLOR.items():
                s = series.get((arch, recipe, variant))
                if not s:
                    continue
                ax_ratio.plot(s["block"], s["ratio"], color=color,
                              linewidth=2, marker="o", markersize=3.5,
                              markeredgecolor="white", markeredgewidth=0.6)
                if all(v is not None for v in s["share"]):
                    ax_share.plot(s["block"], s["share"], color=color,
                                  linewidth=2, marker="o", markersize=3.5,
                                  markeredgecolor="white",
                                  markeredgewidth=0.6)
            ax_ratio.set_title(f"{ARCH_NAME[arch]} / {recipe}",
                               fontsize=10, color=INK)
            style(ax_ratio, ylabel="CLS-norm ratio" if ri == 0 else None)
            style(ax_share,
                  ylabel="CLS attn share" if ri == 0 else None,
                  xlabel="block index" if ai == len(archs) - 1 else None)

    handles = [Line2D([], [], color=c, linewidth=2, marker="o",
                      markersize=5, markeredgecolor="white", label=v)
               for v, c in VARIANT_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("F6 draft — CLS relocation, legacy e2 runs (last.pth): "
                 "per-block CLS-norm ratio and CLS attention share",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png} (preview)")


if __name__ == "__main__":
    main()
