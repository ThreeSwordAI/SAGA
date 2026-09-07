#!/usr/bin/env python3
"""
plotting/plot_gradphi.py
========================
TASK-06B Part 3.4 (amended): per-layer ||dL/dphi|| trajectories for the two
instrumented SAGA runs (mixup s1, nomix s1), epochs 0-30, log-y, plus the
layer-summed trajectories of both runs overlaid.

    python plotting/plot_gradphi.py \
        [--data results/figures_data/gradphi.csv]
        [--out results/figures/gradphi_draft.pdf]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#3a3a3a"
MUTED = "#777777"
RECIPE_COLOR = {"mixup": "#7F77DD", "nomix": "#1D9E75"}
STEPS_PER_EPOCH = 1251          # measured; used only for the x coordinate


def load(path):
    """recipe -> layer -> (global_iter[], norm[])"""
    data = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for r in csv.DictReader(open(path, newline="")):
        g = int(r["epoch"]) * STEPS_PER_EPOCH + int(r["iter"])
        xs, ys = data[r["recipe"]][int(r["layer"])]
        xs.append(g)
        ys.append(float(r["grad_phi_norm"]))
    return data


def style(ax, xlabel, ylabel):
    ax.set_yscale("log")
    ax.grid(True, color="#e3e3e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)


def main():
    parser = argparse.ArgumentParser(description="Render the grad-phi draft.")
    parser.add_argument("--data", default="results/figures_data/gradphi.csv")
    parser.add_argument("--out", default="results/figures/gradphi_draft.pdf")
    args = parser.parse_args()

    data = load(args.data)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cmap = plt.get_cmap("viridis")

    for ax, recipe in zip(axes[:2], ("mixup", "nomix")):
        layers = data.get(recipe, {})
        n_layers = max(layers) + 1 if layers else 1
        for li in sorted(layers):
            xs, ys = layers[li]
            ax.plot(xs, ys, color=cmap(li / max(n_layers - 1, 1)),
                    linewidth=1.0)
        ax.set_title(f"{recipe} (per layer, L0 dark -> L11 light)",
                     fontsize=10, color=INK)
        style(ax, "training iteration (epochs 0-30)", "||dL/dphi||  (log)")

    ax = axes[2]
    for recipe, color in RECIPE_COLOR.items():
        layers = data.get(recipe, {})
        if not layers:
            continue
        # sum over layers at each logged iteration
        per_iter = defaultdict(float)
        for xs, ys in layers.values():
            for x, y in zip(xs, ys):
                per_iter[x] += y
        xs = sorted(per_iter)
        ax.plot(xs, [per_iter[x] for x in xs], color=color, linewidth=2,
                label=recipe)
    ax.set_title("layer-summed", fontsize=10, color=INK)
    style(ax, "training iteration (epochs 0-30)", "sum over layers  (log)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle("grad-phi trajectories — e2r ViT-S SAGA s1, mixup vs "
                 "true-nomix (epochs 0-30)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print(f"wrote {out} (+ png preview)")

    # factual magnitude summary for the note
    for recipe in ("mixup", "nomix"):
        layers = data.get(recipe, {})
        allv = np.array([y for _, ys in layers.values() for y in ys])
        if len(allv):
            print(f"{recipe}: n={len(allv)}  median={np.median(allv):.3e}  "
                  f"p10={np.percentile(allv, 10):.3e}  "
                  f"p90={np.percentile(allv, 90):.3e}")


if __name__ == "__main__":
    main()
