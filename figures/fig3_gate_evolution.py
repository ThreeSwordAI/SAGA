#!/usr/bin/env python3
"""
figures/fig3_gate_evolution.py
==============================

Figure 3: Temporal evolution of learned SAGA gate maps.

Default locked design:
    - Fixed layer: 2 (0-indexed) -> third transformer block
    - Fixed heads: 6, 7, 9 (0-indexed)
      displayed as Head 7, Head 8, Head 10
    - Columns: Init / 50 / 100 / 150 / 200 / 250 / 300
    - Auto-detects whether snapshots store:
        G = sigmoid(phi), or raw phi
    - If G is detected, plots G - 0.5
    - If phi is detected, plots phi directly
    - Navy -> soft neutral -> teal diverging colormap
    - Shared plot range across all non-init panels
    - Plot range uses percentile clipping by default
    - Saves both PNG and PDF

Usage:
    python3 figures/fig3_gate_evolution.py \
        --ckpt_dir /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints/ViT-B_SAGA_nomix \
        --out_dir /home/woody/iwi5/iwi5359h/saga_figures/final \
        --basename fig3_gate_evolution
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


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_LAYER = 2          # 0-indexed
DEFAULT_HEADS = [6, 7, 9]  # 0-indexed
EPOCHS = [50, 100, 150, 200, 250, 300]

NAVY = "#1B2A4A"
TEAL = "#00A99D"
SLATE = "#6B7E9B"
TEXT = "#111111"

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


# ── Loading helpers ────────────────────────────────────────────────────────────

def parse_layer_key(k):
    if isinstance(k, int):
        return k
    if isinstance(k, str) and k.isdigit():
        return int(k)
    return None


def normalize_layer_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a layer tensor to [heads, H, W].

    Supported:
        [heads, H, W]
        [heads, 196]
    """
    t = t.detach().cpu().float()

    if t.ndim == 3:
        return t

    if t.ndim == 2 and t.shape[1] == 196:
        return t.reshape(t.shape[0], 14, 14)

    raise RuntimeError(f"Unsupported layer tensor shape: {tuple(t.shape)}")


def load_snapshot(pt_path: Path) -> dict:
    """
    Load a snapshot and return:
        {layer_idx: tensor[heads, H, W]}
    """
    obj = torch.load(pt_path, map_location="cpu")
    layers = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            parsed = parse_layer_key(k)

            if parsed is not None and torch.is_tensor(v):
                layers[parsed] = normalize_layer_tensor(v)

            elif isinstance(v, dict):
                for kk, vv in v.items():
                    parsed_nested = parse_layer_key(kk)
                    if parsed_nested is not None and torch.is_tensor(vv):
                        layers[parsed_nested] = normalize_layer_tensor(vv)

            elif torch.is_tensor(v):
                t = v.detach().cpu().float()
                if t.ndim == 4:
                    for i in range(t.shape[0]):
                        layers[i] = normalize_layer_tensor(t[i])

    elif torch.is_tensor(obj):
        t = obj.detach().cpu().float()
        if t.ndim == 4:
            for i in range(t.shape[0]):
                layers[i] = normalize_layer_tensor(t[i])
        else:
            raise RuntimeError(f"Unsupported tensor snapshot shape: {tuple(t.shape)}")
    else:
        raise RuntimeError(f"Unsupported snapshot type: {type(obj)}")

    if len(layers) == 0:
        raise RuntimeError(f"No valid layer tensors found in {pt_path}")

    return layers


def detect_storage_mode(layers: dict):
    """
    Detect whether saved values are:
        G in [0,1], or raw phi logits.

    Returns:
        mode, colorbar_label, init_title
    """
    vals = torch.cat([v.flatten() for v in layers.values()])

    if vals.min().item() >= 0.0 and vals.max().item() <= 1.0:
        return "G", "Gate deviation (G - 0.5)", "Initial\n($G = 0.5$)"

    return "phi", r"$\phi_h$ (gate logit)", "Initial\n($\\phi = 0$)"


def transform_tensor(t: torch.Tensor, mode: str) -> np.ndarray:
    """
    Convert tensor to centered plotting values.

    If stored as G:
        plot G - 0.5

    If stored as phi:
        plot phi directly
    """
    arr = t.detach().cpu().float().numpy()

    if mode == "G":
        return arr - 0.5

    if mode == "phi":
        return arr

    raise ValueError(f"Unknown mode: {mode}")


def parse_heads(heads_str: str):
    return [int(x.strip()) for x in heads_str.split(",") if x.strip() != ""]


# ── Range / stats helpers ──────────────────────────────────────────────────────

def compute_panel_stats(name: str, epoch: int, panel: np.ndarray):
    flat = panel.reshape(-1)
    return {
        "row": name,
        "epoch": epoch,
        "min": float(flat.min()),
        "p1": float(np.percentile(flat, 1)),
        "mean_abs": float(np.mean(np.abs(flat))),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
    }


def print_panel_stats(stats):
    print("\nPer-panel statistics (excluding Init column):")
    print(f"{'row':<10}{'epoch':>7}{'min':>12}{'p1':>9}{'mean|.|':>11}{'p99':>10}{'max':>10}")
    for s in stats:
        print(
            f"{s['row']:<10}"
            f"{s['epoch']:>7d}"
            f"{s['min']:>12.4f}"
            f"{s['p1']:>9.4f}"
            f"{s['mean_abs']:>11.4f}"
            f"{s['p99']:>10.4f}"
            f"{s['max']:>10.4f}"
        )


def compute_plot_range(
    non_init_panels,
    lower_pct=1.0,
    upper_pct=99.0,
    clip=True,
    symmetric=False,
):
    """
    Compute plot range from all non-init panels.

    Returns:
        true_min, true_max, plot_vmin, plot_vmax
    """
    all_vals = np.concatenate([p.reshape(-1) for p in non_init_panels])

    true_min = float(all_vals.min())
    true_max = float(all_vals.max())

    if clip:
        vmin = float(np.percentile(all_vals, lower_pct))
        vmax = float(np.percentile(all_vals, upper_pct))
    else:
        vmin = true_min
        vmax = true_max

    if symmetric:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, +m

    # Needed for TwoSlopeNorm
    if vmin >= 0.0:
        vmin = -1e-8
    if vmax <= 0.0:
        vmax = 1e-8

    return true_min, true_max, vmin, vmax


# ── Plotting helpers ───────────────────────────────────────────────────────────

def bold_zero_tick(colorbar):
    ticks = colorbar.get_ticks()
    labels = colorbar.ax.get_yticklabels()

    for tick, label in zip(ticks, labels):
        if np.isclose(tick, 0.0):
            label.set_fontweight("bold")
            label.set_color(TEXT)


def plot_fig3(
    panel_maps,
    col_titles,
    row_labels,
    colorbar_label,
    plot_vmin,
    plot_vmax,
    out_png,
    out_pdf,
):
    n_rows = len(row_labels)
    n_cols = len(col_titles)

    norm = TwoSlopeNorm(vmin=plot_vmin, vcenter=0.0, vmax=plot_vmax)

    # Tuned for NeurIPS full-width figure placement
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.20, 3.20),
    )

    fig.subplots_adjust(
        left=0.083,
        right=0.885,
        top=0.86,
        bottom=0.07,
        wspace=0.045,
        hspace=0.085,
    )

    im = None

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]

            im = ax.imshow(
                panel_maps[r][c],
                cmap=SAGA_GATE_CMAP,
                norm=norm,
                interpolation="nearest",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column titles
            if r == 0:
                if c == 0:
                    ax.set_title(
                        col_titles[c],   # two-line title
                        fontsize=9.6,
                        color=SLATE,
                        fontstyle="italic",
                        pad=4,
                        linespacing=0.90,
                    )
                else:
                    ax.set_title(
                        col_titles[c],
                        fontsize=10.8,
                        color=NAVY,
                        fontweight="regular",
                        pad=7,
                    )

            # Row labels
            if c == 0:
                ax.set_ylabel(
                    row_labels[r],
                    fontsize=10.8,
                    color=NAVY,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                )

    # Taller and slightly wider colorbar
    # spans roughly from top of row 1 to bottom of row 3
    cax = fig.add_axes([0.905, 0.08, 0.022, 0.77])
    cb = fig.colorbar(im, cax=cax)

    tick_vals = [plot_vmin, 0.0, plot_vmax]
    cb.set_ticks(tick_vals)
    cb.set_ticklabels([
        f"{plot_vmin:.2f}",
        "0.0",
        f"{plot_vmax:.2f}",
    ])

    cb.set_label(
        colorbar_label,
        fontsize=10.6,
        color=TEXT,
        labelpad=8,
    )
    cb.ax.tick_params(labelsize=9.5, colors=TEXT, length=3)
    cb.outline.set_edgecolor(SLATE)
    bold_zero_tick(cb)

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Directory containing gate_maps_epochXXXX.pt",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--basename",
        default="fig3_gate_evolution",
        help="Output basename without extension",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=f"Fixed layer index (0-indexed). Default: {DEFAULT_LAYER}",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="6,7,9",
        help="Comma-separated 0-indexed heads. Default: 6,7,9",
    )
    parser.add_argument(
        "--lower_pct",
        type=float,
        default=1.0,
        help="Lower percentile for clipping plot range. Default: 1",
    )
    parser.add_argument(
        "--upper_pct",
        type=float,
        default=99.0,
        help="Upper percentile for clipping plot range. Default: 99",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Disable percentile clipping and use true min/max",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Force symmetric color range around 0",
    )

    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_idx = args.layer
    head_idxs = parse_heads(args.heads)

    ref_path = ckpt_dir / "gate_maps_epoch0300.pt"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference snapshot: {ref_path}")

    ref_layers = load_snapshot(ref_path)
    mode, colorbar_label, init_title = detect_storage_mode(ref_layers)

    print(f"Detected storage mode: {mode}")
    print(f"Using fixed layer:    {layer_idx} (0-indexed)")
    print(f"Using fixed heads:    {head_idxs} (0-indexed)")
    print(f"Displayed as:         {[f'Head {h + 1}' for h in head_idxs]}")

    if layer_idx not in ref_layers:
        raise KeyError(
            f"Layer {layer_idx} not found. Available layers: {sorted(ref_layers.keys())}"
        )

    n_rows = len(head_idxs)
    n_cols = 1 + len(EPOCHS)

    H, W = ref_layers[layer_idx].shape[-2:]
    init_map = np.zeros((H, W), dtype=np.float32)

    panel_maps = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    stats = []
    non_init_panels = []

    # Init column
    for r in range(n_rows):
        panel_maps[r][0] = init_map.copy()

    # Epoch columns
    for c, ep in enumerate(EPOCHS, start=1):
        pt_path = ckpt_dir / f"gate_maps_epoch{ep:04d}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing snapshot: {pt_path}")

        layers = load_snapshot(pt_path)

        if layer_idx not in layers:
            raise KeyError(
                f"Layer {layer_idx} missing in {pt_path}. "
                f"Available layers: {sorted(layers.keys())}"
            )

        plot_vals = transform_tensor(layers[layer_idx], mode)

        for r, h in enumerate(head_idxs):
            if h >= plot_vals.shape[0]:
                raise IndexError(
                    f"Head index {h} out of range for layer tensor shape {plot_vals.shape}"
                )

            panel = plot_vals[h]
            panel_maps[r][c] = panel
            non_init_panels.append(panel)

            row_name = f"Head {h + 1}"
            stats.append(compute_panel_stats(row_name, ep, panel))

    print_panel_stats(stats)

    true_min, true_max, plot_vmin, plot_vmax = compute_plot_range(
        non_init_panels=non_init_panels,
        lower_pct=args.lower_pct,
        upper_pct=args.upper_pct,
        clip=not args.no_clip,
        symmetric=args.symmetric,
    )

    print()
    print(f"True range:    [{true_min:+.4f}, {true_max:+.4f}]")
    print(
        f"Plot range:    [{plot_vmin:+.4f}, {plot_vmax:+.4f}]  "
        f"(clipped={not args.no_clip}, symmetric={args.symmetric})"
    )

    col_titles = [init_title] + [str(ep) for ep in EPOCHS]
    row_labels = [f"Head {h + 1}" for h in head_idxs]

    out_png = str(out_dir / f"{args.basename}.png")
    out_pdf = str(out_dir / f"{args.basename}.pdf")

    plot_fig3(
        panel_maps=panel_maps,
        col_titles=col_titles,
        row_labels=row_labels,
        colorbar_label=colorbar_label,
        plot_vmin=plot_vmin,
        plot_vmax=plot_vmax,
        out_png=out_png,
        out_pdf=out_pdf,
    )


if __name__ == "__main__":
    main()