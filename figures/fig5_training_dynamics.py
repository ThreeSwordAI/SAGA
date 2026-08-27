#!/usr/bin/env python3
"""
figures/fig5_training_dynamics.py
=================================

Figure 5 — Training dynamics on ImageNet-1K (ViT-S)

Creates a two-panel figure:
    Left  : With MixUp/CutMix
    Right : Without MixUp/CutMix

Curves:
    - ViT
    - ViT + Registers
    - SAGA (ours)

Features:
    - Smoothed training curves (moving average)
    - Faint raw curves behind smoothed lines
    - Best-checkpoint markers
    - Right-side score annotations with automatic de-overlap
    - Custom legend in panel 2
    - "Scores < 60% omitted" note
    - Saves both PNG and PDF

Usage:
    python3 figures/fig5_training_dynamics.py \
        --results_dir /home/vault/iwi5/iwi5359h/SAGA/e2/results \
        --out_dir /home/woody/iwi5/iwi5359h/saga_figures/final \
        --basename fig5_training_dynamics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Serif"

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ------------------------------------------------------------------------------
# Visual style
# ------------------------------------------------------------------------------

NAVY = "#1B2A4A"
SLATE = "#6B7E9B"
TEAL = "#00A99D"
TEXT = "#111111"
GRID = "#6B7E9B"

# line colors
COLOR_VIT = SLATE
COLOR_REG = NAVY
COLOR_SAGA = TEAL

# figure size (good starting point for NeurIPS 2-column span)
FIG_W = 13.8
FIG_H = 5.2

TITLE_FSIZE = 25
AXIS_LABEL_FSIZE = 16
TICK_FSIZE = 11
NOTE_FSIZE = 10
SCORE_FSIZE = 13  # same neighborhood as axis tick labels

# plot styling
RAW_ALPHA = 0.12
RAW_LW = 1.3
SMOOTH_LW_VIT = 2.2
SMOOTH_LW_REG = 2.2
SMOOTH_LW_SAGA = 3.0

WINDOW = 7  # moving average

# axes
YMIN = 60
YMAX = 81
YTICKS = [60, 65, 70, 75, 80]
XTICKS = [0, 50, 100, 150, 200, 250, 300]

# right-side score labels
LABEL_X = 303.5
LABEL_MIN_GAP = 1.10
CONNECTOR_LW = 1.0

# note position
NOTE_X = 0.14
NOTE_Y = 0.03

# custom legend box in right panel (axes fraction coordinates)
LEG_X = 0.60
LEG_Y = 0.055
LEG_W = 0.35
LEG_H = 0.20

# internal relative placements inside legend box
LEG_LINE_X0 = 0.03
LEG_LINE_X1 = 0.25
LEG_TEXT_X = 0.30

# subplot spacing
LEFT = 0.07
RIGHT = 0.985
BOTTOM = 0.16
TOP = 0.93
WSPACE = 0.12


# ------------------------------------------------------------------------------
# Data mapping
# ------------------------------------------------------------------------------

PANEL_CFG = {
    "with_mix": {
        "title": "With MixUp/CutMix",
        "files": {
            "vit": "ViT-S_baseline.json",
            "registers": "ViT-S_registers.json",
            "saga": "ViT-S_SAGA.json",
        },
    },
    "without_mix": {
        "title": "Without MixUp/CutMix",
        "files": {
            "vit": "ViT-S_baseline_nomix.json",
            "registers": "ViT-S_registers_nomix.json",
            "saga": "ViT-S_SAGA_nomix.json",
        },
    },
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def smooth(vals, window=7):
    vals = np.asarray(vals, dtype=float)
    if window <= 1:
        return vals.copy()

    pad = window // 2
    padded = np.pad(vals, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def load_history(json_path: Path):
    d = json.load(open(json_path, "r"))
    history = d["history"]

    epochs = np.array([h["epoch"] for h in history], dtype=int)
    top1 = np.array([h["top1"] for h in history], dtype=float)

    best_idx = int(np.argmax(top1))
    best_epoch = int(epochs[best_idx])
    best_top1 = float(top1[best_idx])

    return {
        "epochs": epochs,
        "top1_raw": top1,
        "top1_smooth": smooth(top1, WINDOW),
        "best_epoch": best_epoch,
        "best_top1": best_top1,
    }


def load_all_results(results_dir: Path):
    data = {}
    for panel_key, panel_cfg in PANEL_CFG.items():
        data[panel_key] = {}
        for method_key, filename in panel_cfg["files"].items():
            json_path = results_dir / filename
            if not json_path.exists():
                raise FileNotFoundError(f"Missing JSON file: {json_path}")
            data[panel_key][method_key] = load_history(json_path)
    return data


def style_axis(ax, show_ylabel=False):
    ax.set_xlim(0, 300)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    ax.grid(axis="y", color=GRID, alpha=0.18, linewidth=0.8)
    ax.grid(axis="x", visible=False)

    ax.tick_params(axis="x", labelsize=TICK_FSIZE, colors=TEXT, width=1.1, length=5)
    ax.tick_params(axis="y", labelsize=TICK_FSIZE, colors=TEXT, width=1.1, length=5)

    if show_ylabel:
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=AXIS_LABEL_FSIZE, color=TEXT, labelpad=8)
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.text(
        NOTE_X, NOTE_Y,
        "Scores < 60% omitted",
        transform=ax.transAxes,
        fontsize=NOTE_FSIZE,
        color=NAVY,
        ha="left",
        va="bottom",
        fontstyle="italic"
    )


def plot_method(ax, result, color, linestyle, raw_lw, smooth_lw, zorder):
    e = result["epochs"]
    raw = result["top1_raw"]
    sm = result["top1_smooth"]

    ax.plot(
        e, raw,
        color=color,
        alpha=RAW_ALPHA,
        linewidth=raw_lw,
        linestyle=linestyle,
        zorder=zorder - 1,
    )

    ax.plot(
        e, sm,
        color=color,
        linewidth=smooth_lw,
        linestyle=linestyle,
        zorder=zorder,
    )

    ax.scatter(
        [result["best_epoch"]],
        [result["best_top1"]],
        s=72,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=zorder + 1,
    )


def compute_nonoverlap_positions(y_values, ymin, ymax, min_gap):
    """
    Returns adjusted y positions for labels.
    Keeps order by value and enforces a minimum vertical gap.
    """
    n = len(y_values)
    order = np.argsort(y_values)[::-1]  # highest first
    placed = np.zeros(n, dtype=float)

    top_limit = ymax - 0.6
    bottom_limit = ymin + 0.6

    current = top_limit
    for idx in order:
        y = min(y_values[idx], current)
        placed[idx] = y
        current = y - min_gap

    # if bottom goes below limit, shift the whole stack upward
    min_placed = placed[order[-1]]
    if min_placed < bottom_limit:
        shift = bottom_limit - min_placed
        placed += shift

    # if top exceeds limit again, shift downward
    max_placed = placed[order[0]]
    if max_placed > top_limit:
        shift = max_placed - top_limit
        placed -= shift

    return placed


def annotate_scores(ax, panel_results):
    """
    Adds right-side best-score labels and small connectors.
    """
    methods = ["vit", "registers", "saga"]
    y_true = np.array([panel_results[m]["best_top1"] for m in methods], dtype=float)
    y_disp = compute_nonoverlap_positions(
        y_values=y_true,
        ymin=YMIN,
        ymax=YMAX,
        min_gap=LABEL_MIN_GAP,
    )

    style_map = {
        "vit": {
            "color": COLOR_VIT,
            "ha": "left",
        },
        "registers": {
            "color": COLOR_REG,
            "ha": "left",
        },
        "saga": {
            "color": COLOR_SAGA,
            "ha": "left",
        },
    }

    for i, m in enumerate(methods):
        r = panel_results[m]
        x0 = r["best_epoch"]
        y0 = r["best_top1"]
        x1 = LABEL_X - 0.8
        y1 = y_disp[i]

        ax.plot(
            [x0 + 2.0, x1],
            [y0, y1],
            color=style_map[m]["color"],
            linewidth=CONNECTOR_LW,
            alpha=0.95,
            clip_on=False,
            zorder=10,
        )

        ax.text(
            LABEL_X,
            y1,
            f"{r['best_top1']:.2f}",
            fontsize=SCORE_FSIZE,
            color=style_map[m]["color"],
            ha="left",
            va="center",
            fontweight="bold" if m == "saga" else "regular",
            clip_on=False,
            zorder=11,
        )


def draw_custom_legend(ax):
    """
    Custom legend inside panel 2.
    """
    box = FancyBboxPatch(
        (LEG_X, LEG_Y),
        LEG_W,
        LEG_H,
        transform=ax.transAxes,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor="white",
        edgecolor=SLATE,
        linewidth=1.1,
        alpha=0.93,
        zorder=20,
    )
    ax.add_patch(box)

    bx, by, bw, bh = LEG_X, LEG_Y, LEG_W, LEG_H

    rows = [
        by + 0.76 * bh,
        by + 0.49 * bh,
        by + 0.22 * bh,
    ]

    # line samples
    lx0 = bx + LEG_LINE_X0 * bw
    lx1 = bx + LEG_LINE_X1 * bw
    tx = bx + LEG_TEXT_X * bw

    # SAGA
    ax.plot(
        [lx0, lx1], [rows[0], rows[0]],
        transform=ax.transAxes,
        color=COLOR_SAGA,
        linewidth=SMOOTH_LW_SAGA,
        solid_capstyle="butt",
        zorder=21,
    )
    ax.text(
        tx, rows[0],
        "SAGA",
        transform=ax.transAxes,
        fontsize=12.0,
        color=TEXT,
        va="center",
        ha="left",
        fontweight="bold",
        zorder=21,
    )
    ax.text(
        tx + + 0.25 * bw, rows[0],
        " (ours)",
        transform=ax.transAxes,
        fontsize=11.5,
        color=TEAL,
        va="center",
        ha="left",
        fontstyle="italic",
        fontweight="bold",
        zorder=21,
    )

    # ViT
    ax.plot(
        [lx0, lx1], [rows[1], rows[1]],
        transform=ax.transAxes,
        color=COLOR_VIT,
        linewidth=SMOOTH_LW_VIT,
        solid_capstyle="butt",
        zorder=21,
    )
    ax.text(
        tx, rows[1],
        "ViT",
        transform=ax.transAxes,
        fontsize=11.5,
        color=TEXT,
        va="center",
        ha="left",
        zorder=21,
    )

    # ViT + Registers
    ax.plot(
        [lx0, lx1], [rows[2], rows[2]],
        transform=ax.transAxes,
        color=COLOR_REG,
        linewidth=SMOOTH_LW_REG,
        linestyle="--",
        solid_capstyle="butt",
        zorder=21,
    )
    ax.text(
        tx, rows[2],
        "ViT + Registers",
        transform=ax.transAxes,
        fontsize=11.5,
        color=TEXT,
        va="center",
        ha="left",
        zorder=21,
    )


# ------------------------------------------------------------------------------
# Main plotting
# ------------------------------------------------------------------------------

def plot_figure(data, out_png: Path, out_pdf: Path):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)

    fig.subplots_adjust(
        left=LEFT,
        right=RIGHT,
        bottom=BOTTOM,
        top=TOP,
        wspace=WSPACE,
    )

    # left panel
    ax = axes[0]
    style_axis(ax, show_ylabel=True)
    ax.set_title(PANEL_CFG["with_mix"]["title"], fontsize=TITLE_FSIZE, color=NAVY, pad=14, fontweight="bold")

    plot_method(ax, data["with_mix"]["vit"], COLOR_VIT, "-", RAW_LW, SMOOTH_LW_VIT, zorder=3)
    plot_method(ax, data["with_mix"]["registers"], COLOR_REG, "--", RAW_LW, SMOOTH_LW_REG, zorder=4)
    plot_method(ax, data["with_mix"]["saga"], COLOR_SAGA, "-", RAW_LW, SMOOTH_LW_SAGA, zorder=5)
    annotate_scores(ax, data["with_mix"])

    # right panel
    ax = axes[1]
    style_axis(ax, show_ylabel=False)
    ax.set_title(PANEL_CFG["without_mix"]["title"], fontsize=TITLE_FSIZE, color=NAVY, pad=14, fontweight="bold")

    plot_method(ax, data["without_mix"]["vit"], COLOR_VIT, "-", RAW_LW, SMOOTH_LW_VIT, zorder=3)
    plot_method(ax, data["without_mix"]["registers"], COLOR_REG, "--", RAW_LW, SMOOTH_LW_REG, zorder=4)
    plot_method(ax, data["without_mix"]["saga"], COLOR_SAGA, "-", RAW_LW, SMOOTH_LW_SAGA, zorder=5)
    annotate_scores(ax, data["without_mix"])
    draw_custom_legend(ax)

    fig.supxlabel("Epoch", fontsize=AXIS_LABEL_FSIZE, color=TEXT, y=0.04)

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing ViT-S_*.json result files",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--basename",
        default="fig5_training_dynamics",
        help="Output basename without extension",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    data = load_all_results(results_dir)

    print("\nLoaded best top-1 values:")
    for panel_key in ["with_mix", "without_mix"]:
        print(f"  {panel_key}:")
        for method_key in ["vit", "registers", "saga"]:
            r = data[panel_key][method_key]
            print(
                f"    {method_key:<10}"
                f"best_top1={r['best_top1']:.3f} "
                f"at epoch={r['best_epoch']}"
            )

    out_png = out_dir / f"{args.basename}.png"
    out_pdf = out_dir / f"{args.basename}.pdf"

    plot_figure(data, out_png, out_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()