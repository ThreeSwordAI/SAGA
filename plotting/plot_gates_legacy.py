#!/usr/bin/env python3
"""
plotting/plot_gates_legacy.py
=============================
TASK-02C C2: gate-map draft from the Phase-B phi dumps.

    python plotting/plot_gates_legacy.py \
        [--gates-dir results/legacy/gates]
        [--out results/figures/gates_legacy_draft.pdf]

Multi-page PDF: one page per SAGA run (last.pth) — a [layers x heads] grid
of sigmoid(phi) maps reshaped to the 14x14 patch grid, shared color scale
0..1 — plus a final summary page: per-layer mean gate and fraction<0.4,
the four runs overlaid (labeled by cell; variant colors irrelevant here).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

INK = "#3a3a3a"
MUTED = "#777777"
RUN_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]  # cell labels, CVD-safe


def load_runs(gates_dir: Path):
    """[(cell_label, phi [L,H,N], stats dict)] for the 4 SAGA last runs."""
    runs = []
    for npz_path in sorted(gates_dir.glob("e2_*_saga_*_last_phi.npz")):
        parts = npz_path.name.replace("_phi.npz", "").split("_")
        label = f"{parts[1]}_{parts[2]}/{parts[3]}"       # e.g. vit_small/mixup
        phi = np.load(npz_path)["phi"]
        stats = json.load(open(npz_path.with_name(
            npz_path.name.replace("_phi.npz", "_phi_stats.json"))))
        runs.append((label, phi, stats))
    return runs


def gate_grid_page(pdf, label, phi):
    L, H, N = phi.shape
    side = int(round(N ** 0.5))
    gate = 1.0 / (1.0 + np.exp(-phi.astype(np.float64)))
    fig, axes = plt.subplots(L, H, figsize=(1.05 * H + 1.2, 1.05 * L + 1.0))
    for li in range(L):
        for hi in range(H):
            ax = axes[li][hi]
            im = ax.imshow(gate[li, hi].reshape(side, side),
                           vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_xticks([]), ax.set_yticks([])
            if li == 0:
                ax.set_title(f"h{hi}", fontsize=7, color=MUTED)
            if hi == 0:
                ax.set_ylabel(f"L{li}", fontsize=7, color=MUTED,
                              rotation=0, ha="right", va="center")
    fig.suptitle(f"sigmoid(phi) gate maps — {label} (last.pth), "
                 f"{side}x{side} patch grid", fontsize=11, color=INK)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf, runs, png_path=None):
    fig, (ax_mean, ax_frac) = plt.subplots(1, 2, figsize=(10, 4))
    for (label, phi, stats), color in zip(runs, RUN_COLORS):
        layers = stats["layers"]
        xs = [l["layer"] for l in layers]
        ax_mean.plot(xs, [l["mean_gate"] for l in layers], color=color,
                     linewidth=2, marker="o", markersize=4,
                     markeredgecolor="white", label=label)
        ax_frac.plot(xs, [l["frac_below_0.4"] for l in layers], color=color,
                     linewidth=2, marker="o", markersize=4,
                     markeredgecolor="white", label=label)
    for ax, ylab in ((ax_mean, "mean gate  sigmoid(phi)"),
                     (ax_frac, "fraction of positions with gate < 0.4")):
        ax.set_xlabel("layer", fontsize=9, color=INK)
        ax.set_ylabel(ylab, fontsize=9, color=INK)
        ax.grid(True, color="#e3e3e3", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    ax_mean.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax_mean.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Gate summary — per-layer mean gate and low-gate fraction, "
                 "4 SAGA runs (last.pth)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    if png_path:
        fig.savefig(png_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render the gate-map draft.")
    parser.add_argument("--gates-dir", default="results/legacy/gates")
    parser.add_argument("--out",
                        default="results/figures/gates_legacy_draft.pdf")
    args = parser.parse_args()

    runs = load_runs(Path(args.gates_dir))
    if not runs:
        raise SystemExit(f"no *_saga_*_last_phi.npz under {args.gates_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for label, phi, _ in runs:
            gate_grid_page(pdf, label, phi)
        summary_page(pdf, runs, png_path=out.with_suffix(".png"))
    print(f"wrote {out} ({len(runs)} map pages + summary; "
          f"PNG preview of the summary page alongside)")


if __name__ == "__main__":
    main()
