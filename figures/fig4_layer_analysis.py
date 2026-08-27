#!/usr/bin/env python3
"""
figures/fig4_layer_analysis.py
==============================

Figure 4 — Per-Layer Gate Analysis

Generates two outputs:
1) fig4_layer_analysis_grid.png/.pdf
2) fig4_layer_analysis_with_bar.png/.pdf

Design:
    - Uses ViT-B SAGA best.pth
    - Extracts raw gate logits phi from blocks.{i}.attn.gate.phi
    - Averages over 12 heads per layer
    - Spatially centers each layer map to highlight relative spatial priors
    - Adds an activity bar showing mean |phi| per layer
    - Marks Layer 12 with * because the final-block patch gates remain neutral
      under CLS-only classification readout

Usage:
    python3 figures/fig4_layer_analysis.py \
        --ckpt_path /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints/ViT-B_SAGA_nomix/best.pth \
        --out_dir /home/woody/iwi5/iwi5359h/saga_figures/final
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Serif"

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec


# ── Visual style ───────────────────────────────────────────────────────────────

NAVY = "#1B2A4A"
TEAL = "#00A99D"
SLATE = "#6B7E9B"
TEXT = "#111111"
POSITIVE = "#00A99D"

SAGA_GATE_CMAP = LinearSegmentedColormap.from_list(
    "saga_gate",
    [
        "#1B2A4A",  # strong negative
        #"#5E789C",  # mild negative
        #"#F2EFE8",  # neutral / 0.0
        #"#8FD4CD",  # mild positive
        #"#00A99D",  # strong positive / SAGA
        "#3E6F99",  # low attention
        "#22B8C8",  # low-mid
        "#FFC933",  # high-mid
        "#FFF2A6",  # peak
    ],
    N=256,
)


# ── Manual layout tuning values ────────────────────────────────────────────────
# Format for add_axes rectangles:
# [left, bottom, width, height]

# Grid-only colorbar
COLORBAR_RECT_GRID_ONLY = [0.925, 0.185, 0.020, 0.75]

# With-bar colorbar.
# Increase bottom to move it up.
# Increase height to make it longer.
# Example: [0.925, 0.315, 0.020, 0.445] aligns it mostly with heatmaps only.
COLORBAR_RECT_WITH_BAR = [0.925, 0.315, 0.020, 0.56]

# Shift only the activity bar to the right.
# Increase this to move the bar chart more right.
BAR_SHIFT_RIGHT = 0.1


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_state_dict(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def extract_layer_data(state):
    """
    Returns:
        raw_maps       list of 12 arrays [14, 14]
        centered_maps  list of 12 arrays [14, 14]
        activities     list of 12 floats
    """
    raw_maps = []
    centered_maps = []
    activities = []

    for i in range(12):
        key = f"blocks.{i}.attn.gate.phi"
        if key not in state:
            raise KeyError(f"Missing key: {key}")

        phi = state[key].detach().cpu().float()  # [12, 196]

        if phi.ndim != 2 or phi.shape != (12, 196):
            raise RuntimeError(
                f"Unexpected shape for {key}: {tuple(phi.shape)}; expected (12, 196)"
            )

        # Average over heads -> [196] -> [14, 14]
        phi_avg = phi.mean(dim=0).reshape(14, 14).numpy()

        # Spatial centering: removes layer-wise bias and reveals spatial structure
        phi_avg_centered = phi_avg - phi_avg.mean()

        # Activity magnitude from full phi tensor
        activity = phi.abs().mean().item()

        raw_maps.append(phi_avg)
        centered_maps.append(phi_avg_centered)
        activities.append(activity)

    return raw_maps, centered_maps, activities


def format_tick(v):
    av = abs(v)
    if av >= 1:
        return f"{v:.1f}"
    if av >= 0.1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def add_shared_colorbar(fig, im, rect, label, vmin, vmax):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax)

    cb.set_ticks([vmin, 0.0, vmax])
    cb.set_ticklabels([format_tick(vmin), "0.0", format_tick(vmax)])

    cb.set_label(label, fontsize=9.8, color=TEXT, labelpad=7)
    cb.ax.tick_params(labelsize=9.0, colors=TEXT, length=3)
    cb.outline.set_edgecolor(SLATE)

    for tick, ticklabel in zip(cb.get_ticks(), cb.ax.get_yticklabels()):
        if np.isclose(tick, 0.0):
            ticklabel.set_fontweight("bold")
            ticklabel.set_color(TEXT)


def compute_shared_symmetric_range(centered_maps):
    all_vals = np.concatenate([m.reshape(-1) for m in centered_maps])
    max_abs = float(np.max(np.abs(all_vals)))
    max_abs = max(max_abs, 1e-8)
    return -max_abs, +max_abs


def layer_title(idx: int) -> str:
    """
    Paper-facing layer labels are 1-indexed.
    Layer 12 is marked with * because it stays neutral under CLS-only readout.
    """
    if idx == 11:
        return f"Layer {idx + 1}*"
    return f"Layer {idx + 1}"


# ── Plotters ───────────────────────────────────────────────────────────────────

def plot_grid_only(centered_maps, out_png, out_pdf):
    vmin, vmax = compute_shared_symmetric_range(centered_maps)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 6, figsize=(7.05, 2.78))

    fig.subplots_adjust(
        left=0.025,
        right=0.895,
        top=0.885,
        bottom=0.065,
        wspace=0.13,
        hspace=0.20,
    )

    im = None

    for idx, ax in enumerate(axes.flat):
        im = ax.imshow(
            centered_maps[idx],
            cmap=SAGA_GATE_CMAP,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(
            layer_title(idx),
            fontsize=9.4,
            color=NAVY,
            pad=4,
            fontweight="regular",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    add_shared_colorbar(
        fig=fig,
        im=im,
        rect=COLORBAR_RECT_GRID_ONLY,
        label="Centered mean φ",
        vmin=vmin,
        vmax=vmax,
    )

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


def plot_grid_with_bar(centered_maps, activities, out_png, out_pdf):
    vmin, vmax = compute_shared_symmetric_range(centered_maps)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(7.05, 3.55))

    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.42],
        hspace=0.24,
        wspace=0.13,
    )

    heat_axes = []
    for r in range(2):
        for c in range(6):
            heat_axes.append(fig.add_subplot(gs[r, c]))

    im = None

    for idx, ax in enumerate(heat_axes):
        im = ax.imshow(
            centered_maps[idx],
            cmap=SAGA_GATE_CMAP,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(
            layer_title(idx),
            fontsize=9.4,
            color=NAVY,
            pad=4,
            fontweight="regular",
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    # Bottom activity bar
    ax_bar = fig.add_subplot(gs[2, :])

    x = np.arange(1, 13)
    ax_bar.bar(x, activities, color=POSITIVE, width=0.72)

    ax_bar.set_xlim(0.4, 12.6)
    ax_bar.set_xticks(x)

    bar_labels = [str(i) for i in x]
    bar_labels[-1] = "12*"

    ax_bar.set_xticklabels(bar_labels, fontsize=8.9, color=TEXT)

    for label in ax_bar.get_xticklabels():
        if label.get_text() == "12*":
            label.set_fontweight("bold")
            label.set_color(TEXT)

    ax_bar.tick_params(axis="x", labelsize=8.9, colors=TEXT)
    ax_bar.tick_params(axis="y", labelsize=8.9, colors=TEXT)

    ax_bar.set_ylabel("Mean |φ|", fontsize=9.4, color=TEXT)
    ax_bar.set_xlabel("Layer", fontsize=9.4, color=TEXT, labelpad=2)

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color(SLATE)
    ax_bar.spines["bottom"].set_color(SLATE)

    add_shared_colorbar(
        fig=fig,
        im=im,
        rect=COLORBAR_RECT_WITH_BAR,
        label="Centered mean φ",
        vmin=vmin,
        vmax=vmax,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.895,
        top=0.905,
        bottom=0.105,
    )
    # Shift only the activity bar to the right
    pos = ax_bar.get_position()
    ax_bar.set_position([
        pos.x0 + BAR_SHIFT_RIGHT,
        pos.y0,
        pos.width - BAR_SHIFT_RIGHT + 0.06,
        pos.height,
    ])

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to ViT-B SAGA best.pth",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save outputs",
    )

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    state = load_state_dict(ckpt_path)

    print("Extracting per-layer gate maps...")
    raw_maps, centered_maps, activities = extract_layer_data(state)

    raw_all = np.concatenate([m.reshape(-1) for m in raw_maps])
    centered_all = np.concatenate([m.reshape(-1) for m in centered_maps])

    raw_max_abs = float(np.max(np.abs(raw_all)))
    centered_max_abs = float(np.max(np.abs(centered_all)))

    print("\nPer-layer summary:")
    print(
        f"{'Layer':<8}"
        f"{'raw_min':>10}"
        f"{'raw_max':>10}"
        f"{'raw_mean':>11}"
        f"{'center_std':>12}"
        f"{'mean|phi|':>12}"
    )

    for i, (raw_m, cent_m, act) in enumerate(
        zip(raw_maps, centered_maps, activities),
        start=1,
    ):
        star = "*" if i == 12 else ""
        print(
            f"{str(i) + star:<8}"
            f"{raw_m.min():>10.4f}"
            f"{raw_m.max():>10.4f}"
            f"{raw_m.mean():>11.4f}"
            f"{cent_m.std():>12.4f}"
            f"{act:>12.4f}"
        )

    print(f"\nRaw map max abs:       {raw_max_abs:.4f}")
    print(f"Centered map max abs:  {centered_max_abs:.4f}")
    print(
        f"Shared centered range: "
        f"[{-centered_max_abs:+.4f}, {+centered_max_abs:+.4f}]"
    )
    print("\nNote: Layer 12* remains neutral under CLS-only classification readout.")

    out_grid_png = str(out_dir / "fig4_layer_analysis_grid.png")
    out_grid_pdf = str(out_dir / "fig4_layer_analysis_grid.pdf")
    out_bar_png = str(out_dir / "fig4_layer_analysis_with_bar.png")
    out_bar_pdf = str(out_dir / "fig4_layer_analysis_with_bar.pdf")

    print("\nGenerating centered grid-only version...")
    plot_grid_only(centered_maps, out_grid_png, out_grid_pdf)

    print("\nGenerating centered grid + activity-bar version...")
    plot_grid_with_bar(centered_maps, activities, out_bar_png, out_bar_pdf)

    print("\nDone.")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()