#!/usr/bin/env python3
"""
plotting/plot_A1.py
===================
TASK-02B Phase C1: paper-figure-A1 draft — overlaid log-x histograms of
last-block patch norms per (arch, recipe) panel, three variants each, from
the Phase-B `*_last_normstats.json` (identical bin edges within an arch).

Vertical lines: solid dark = the per-arch fixed τ
(results/diagsplit/fixed_thresholds.json); dashed, in the variant color =
that variant's mean per-image MAD threshold (median + 5·MAD).

This makes H1 visible: does SAGA's dashed line sit far left of baseline's
while its right tail sits at-or-left of baseline's?

    python plotting/plot_A1.py \
        [--diag-dir results/legacy/diag]
        [--thresholds results/diagsplit/fixed_thresholds.json]
        [--out results/figures/A1_norm_hist_draft.pdf]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

VARIANT_COLOR = {"baseline": "#888780", "registers": "#1D9E75",
                 "saga": "#7F77DD"}
ARCH_NAME = {"vit_small": "ViT-S/16", "vit_base": "ViT-B/16"}
INK = "#3a3a3a"
MUTED = "#777777"


def main():
    parser = argparse.ArgumentParser(description="Render the A1 draft PDF.")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--thresholds",
                        default="results/diagsplit/fixed_thresholds.json")
    parser.add_argument("--out",
                        default="results/figures/A1_norm_hist_draft.pdf")
    args = parser.parse_args()

    taus = json.load(open(args.thresholds))
    diag_dir = Path(args.diag_dir)
    archs = ["vit_small", "vit_base"]
    recipes = ["mixup", "nomix"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ai, arch in enumerate(archs):
        for ri, recipe in enumerate(recipes):
            ax = axes[ai][ri]
            for variant, color in VARIANT_COLOR.items():
                p = diag_dir / (f"e2_{arch}_{recipe}_{variant}"
                                f"_rlast_last_normstats.json")
                d = json.load(open(p))
                edges = np.asarray(d["hist_bin_edges"])
                counts = np.asarray(d["hist_counts"], dtype=float)
                ax.stairs(counts, edges, color=color, linewidth=1.8,
                          fill=True, alpha=0.12)
                ax.stairs(counts, edges, color=color, linewidth=1.8)
                ax.axvline(d["mean_threshold_mad_k5"], color=color,
                           linestyle="--", linewidth=1.4)
            ax.axvline(float(taus[arch]), color=INK, linewidth=1.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
            ax.set_title(f"{ARCH_NAME[arch]} / {recipe}   "
                         f"(fixed τ = {float(taus[arch]):.2f})",
                         fontsize=10, color=INK)
            ax.grid(True, color="#e3e3e3", linewidth=0.6)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#bbbbbb")
            ax.tick_params(colors=MUTED, labelsize=8)
            if ri == 0:
                ax.set_ylabel("token count (log)", fontsize=8, color=INK)
            if ai == 1:
                ax.set_xlabel("last-block patch-token L2 norm (log)",
                              fontsize=8, color=INK)

    handles = ([Line2D([], [], color=c, linewidth=2, label=v)
                for v, c in VARIANT_COLOR.items()]
               + [Line2D([], [], color=INK, linewidth=1.6,
                         label="fixed τ (per arch)"),
                  Line2D([], [], color=MUTED, linewidth=1.4, linestyle="--",
                         label="mean MAD threshold (per variant)")])
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8, labelcolor=INK)
    fig.suptitle("A1 draft — last-block patch-norm distributions, legacy e2 "
                 "runs (last.pth), 10k diagnostic split", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png} (preview)")


if __name__ == "__main__":
    main()
