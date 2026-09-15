#!/usr/bin/env python3
"""
plotting/plot_T2.py
===================
TASK-12 Phase C.3: results/figures/F_ablation_draft.pdf.

    python plotting/plot_T2.py [--table results/tables/T2_ablation.csv] \
        [--runs-root results/runs] [--out results/figures/F_ablation_draft.pdf]

Three panels:
  (a) top-1 by arm. The y-axis is labelled with WHICH quantity is plotted —
      the exact fp32 evaluation when it has been derived, otherwise the
      trainer's bf16 log value, with the bars hatched and the title saying
      so. A figure that silently swaps one for the other is how a log value
      becomes a paper number.
  (b) the final gate maps of the phi arms at one layer, on a SHARED
      colorbar: the visual form of "only the spatial arms have structure".
      The head-scalar arm's single value is expanded over the grid, which is
      exactly what it applies at every position.
  (c) within-layer spatial std of the gate against depth, for the same arms
      — the same claim across all 12 layers rather than at one of them.

Arms with no phi (baseline, LayerScale) are absent from (b) and (c) rather
than drawn as zeros; an absent arm is not a flat one.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

MISSING = "MISSING"


def num(v):
    if v in (None, "", MISSING):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def head_mean_gate(phi):
    sig = 1.0 / (1.0 + np.exp(-phi.astype(np.float64)))
    return sig.mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description="TASK-12 ablation figure.")
    ap.add_argument("--table", default="results/tables/T2_ablation.csv")
    ap.add_argument("--runs-root", default="results/runs")
    ap.add_argument("--out", default="results/figures/F_ablation_draft.pdf")
    args = ap.parse_args()

    with open(args.table, newline="", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    runs_root = Path(args.runs_root)

    fp32 = [num(r["top1_last"]) for r in table]
    use_fp32 = all(v is not None for v in fp32)
    vals = fp32 if use_fp32 else [num(r["top1_last_bf16_log"]) for r in table]
    src = ("exact fp32, full 50k val (tools/eval.py)" if use_fp32
           else "trainer bf16 per-epoch full-val — NOT CANONICAL, "
                "pending scripts/jobs/abl_derive.sbatch")

    # ── gate maps ────────────────────────────────────────────────────────
    gates = {}
    for r in table:
        dumps = sorted((runs_root / r["run_id"] / "gates").glob("phi_e*.npz"))
        if not dumps:
            continue
        g = head_mean_gate(np.load(dumps[-1])["phi"])          # [L, N or 1]
        if g.shape[1] == 1:                     # head scalar: what it applies
            g = np.repeat(g, 196, axis=1)       # at every position
        gates[r["arm"]] = g

    spatial_std = {a: g.std(axis=1) for a, g in gates.items()}
    # the layer where the learned spatial structure is strongest, if any
    layer = 0
    if "E" in spatial_std and spatial_std["E"].max() > 0:
        layer = int(np.argmax(spatial_std["E"]))

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, max(len(gates), 1), height_ratios=[1.1, 1.0, 0.9],
                          hspace=0.55, wspace=0.25)

    # (a) top-1 by arm
    ax = fig.add_subplot(gs[0, :])
    arms = [r["arm"] for r in table]
    finite = [v for v in vals if v is not None]
    bars = ax.bar(range(len(table)), [v if v is not None else 0 for v in vals],
                  color=["#4c72b0" if a not in ("E", "F") else "#c44e52"
                         for a in arms],
                  hatch="" if use_fp32 else "//",
                  edgecolor="black", linewidth=0.6)
    for i, (r, v) in enumerate(zip(table, vals)):
        ax.text(i, (v if v is not None else 0),
                f"{v:.3f}" if v is not None else MISSING,
                ha="center", va="bottom", fontsize=8)
    if finite:
        lo, hi = min(finite), max(finite)
        pad = max(0.15, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
    if any(a == "A" for a in arms) and vals[arms.index("A")] is not None:
        ax.axhline(vals[arms.index("A")], color="black", lw=0.9, ls="--",
                   label="arm A (no gate)")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([f"{r['arm']}\n{r['gate_mode']}\n"
                        f"{r['gate_params_trainable']} params"
                        for r in table], fontsize=8)
    ax.set_ylabel("top-1 (%)")
    ax.set_title("(a) ViT-S/16, mixup, seed 0, 100 EPOCHS — one seed per arm; "
                 "the 300-epoch headline is a different schedule\n"
                 f"source: {src}", fontsize=9)

    # (b) final gate maps, shared colorbar
    if gates:
        vmin = min(g[layer].min() for g in gates.values())
        vmax = max(g[layer].max() for g in gates.values())
        map_axes = []
        for i, (arm, g) in enumerate(sorted(gates.items())):
            axm = fig.add_subplot(gs[1, i])
            map_axes.append(axm)
            im = axm.imshow(g[layer].reshape(14, 14), vmin=vmin, vmax=vmax,
                            cmap="viridis")
            axm.set_title(f"arm {arm}\nspatial std {g[layer].std():.4f}",
                          fontsize=8)
            axm.set_xticks([])
            axm.set_yticks([])
            if i == len(gates) - 1:
                fig.colorbar(im, ax=axm, fraction=0.046)
        # placed from the axes' own bbox, so the caption cannot land on top
        # of the maps when the panel count or the figure size changes
        y = min(a.get_position().y0 for a in map_axes) - 0.045
        fig.text(0.5, y,
                 f"(b) final gate sigmoid(phi), head-mean, layer {layer} "
                 f"(where the spatial arm's structure is strongest) — "
                 f"shared colorbar",
                 ha="center", fontsize=9)

    # (c) spatial std vs depth
    axs = fig.add_subplot(gs[2, :])
    for arm, sd in sorted(spatial_std.items()):
        axs.plot(range(len(sd)), sd, marker="o", ms=3, label=f"arm {arm}")
    axs.set_xlabel("layer")
    axs.set_ylabel("within-layer spatial std")
    axs.set_title("(c) only the spatial arms can vary across positions — "
                  "the frozen and head-scalar arms are 0 by construction, "
                  "and the arms with no phi are absent, not flat",
                  fontsize=9)
    axs.legend(fontsize=8, ncol=4)
    axs.grid(alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (+ .png)")
    print(f"  panel (a) source: {src}")
    print(f"  panel (b) layer {layer}; arms with phi: "
          f"{', '.join(sorted(gates)) or 'none'}")


if __name__ == "__main__":
    main()
