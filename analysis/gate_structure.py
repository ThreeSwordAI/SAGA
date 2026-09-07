#!/usr/bin/env python3
"""
analysis/gate_structure.py
==========================
TASK-06B Part 3.5 (amended to all SIX ViT-S SAGA gate sets): repeat
agreement within and across the two true cells.

Gate sets (phi -> sigmoid, [12, 6, 196] each):
- mixup cell (4 repeats): legacy-mixupdir, legacy-nomixdir (both
  mixup-trained per the erratum), e2r s1, e2r s2;
- true-nomix cell (2 repeats): e2r nomix s1, s2.

Outputs:
- results/tables/gate_agreement.csv — per-pair Spearman (pooled over
  layers + per-layer mean/min/max) grouped {within-mixup, within-nomix,
  cross-recipe}, plus per-set per-layer within-layer spatial std.
- results/figures/gate_agreement_draft.pdf — layer-8 mean-over-heads gate
  maps of all six sets on one row, shared 0..1 colorbar.

    python analysis/gate_structure.py
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

INK = "#3a3a3a"
MUTED = "#777777"

GATE_SETS = {
    # label -> (cell, phi npz path)
    "mixup:legacy-mixupdir": ("mixup",
        "results/legacy/gates/e2_vit_small_mixup_saga_rlast_last_phi.npz"),
    "mixup:legacy-nomixdir": ("mixup",
        "results/legacy/gates/e2_vit_small_nomix_saga_rlast_last_phi.npz"),
    "mixup:s1": ("mixup",
        "results/runs/e2r_vits_mixup_saga_s1/gates/phi_e299.npz"),
    "mixup:s2": ("mixup",
        "results/runs/e2r_vits_mixup_saga_s2/gates/phi_e299.npz"),
    "nomix:s1": ("nomix",
        "results/runs/e2r_vits_nomix_saga_s1/gates/phi_e299.npz"),
    "nomix:s2": ("nomix",
        "results/runs/e2r_vits_nomix_saga_s2/gates/phi_e299.npz"),
}


def head_mean_gate(phi: np.ndarray) -> np.ndarray:
    """sigmoid(phi) [L, H, N] -> arithmetic mean over heads -> [L, N]."""
    sig = 1.0 / (1.0 + np.exp(-phi.astype(np.float64)))
    return sig.mean(axis=1)


def load_gates():
    """label -> gate maps [L, N] (sigmoid(phi), mean over heads)."""
    return {label: head_mean_gate(np.load(path)["phi"])
            for label, (cell, path) in GATE_SETS.items()}


def pair_group(a, b):
    ca, cb = a.split(":")[0], b.split(":")[0]
    if ca == cb:
        return f"within-{ca}"
    return "cross-recipe"


def agreement_rows(gates):
    rows = []
    import warnings
    for a, b in combinations(gates, 2):
        ga, gb = gates[a], gates[b]
        pooled = spearmanr(ga.ravel(), gb.ravel()).statistic
        # a spatially CONSTANT layer (e.g. the final layer, where phi stays
        # ~0) has no defined rank correlation — such layers are excluded and
        # counted, never averaged in as NaN
        per_layer = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for li in range(ga.shape[0]):
                v = spearmanr(ga[li], gb[li]).statistic
                if not np.isnan(v):
                    per_layer.append(v)
        rows.append({
            "pair": f"{a} vs {b}", "group": pair_group(a, b),
            "spearman_pooled": round(float(pooled), 4),
            "spearman_layer_mean": round(float(np.mean(per_layer)), 4),
            "spearman_layer_min": round(float(np.min(per_layer)), 4),
            "spearman_layer_max": round(float(np.max(per_layer)), 4),
            "n_layers_defined": len(per_layer),
        })
    return rows


def spatial_std_rows(gates):
    """Per set, per layer: within-layer spatial std of the mean-head gate —
    the 'did it learn spatial structure' scalar."""
    rows = []
    for label, g in gates.items():
        for li in range(g.shape[0]):
            rows.append({"set": label, "layer": li,
                         "spatial_std": round(float(g[li].std()), 5)})
    return rows


def figure(gates, out_path: Path, layer=8):
    fig, axes = plt.subplots(1, len(gates), figsize=(2.1 * len(gates), 2.9))
    side = int(round(gates[next(iter(gates))].shape[1] ** 0.5))
    for ax, (label, g) in zip(axes, gates.items()):
        im = ax.imshow(g[layer].reshape(side, side), vmin=0, vmax=1,
                       cmap="viridis")
        ax.set_title(label, fontsize=8, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle(f"Layer-{layer} gate maps (sigmoid(phi), mean over heads) — "
                 f"all six ViT-S SAGA repeats", fontsize=11, color=INK)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=130)
    print(f"wrote {out_path} (+ png preview)")


def main():
    parser = argparse.ArgumentParser(description="Gate repeat agreement.")
    parser.add_argument("--out", default="results/tables/gate_agreement.csv")
    parser.add_argument("--fig",
                        default="results/figures/gate_agreement_draft.pdf")
    parser.add_argument("--layer", type=int, default=8)
    args = parser.parse_args()

    gates = load_gates()
    agree = agreement_rows(gates)
    stds = spatial_std_rows(gates)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "group", "spearman_pooled",
                                          "spearman_layer_mean",
                                          "spearman_layer_min",
                                          "spearman_layer_max",
                                          "n_layers_defined"])
        w.writeheader()
        w.writerows(agree)
    std_out = out.with_name("gate_spatial_std.csv")
    with open(std_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "layer", "spatial_std"])
        w.writeheader()
        w.writerows(stds)
    print(f"wrote {out}: {len(agree)} pairs | {std_out}: {len(stds)} rows")

    for group in ("within-mixup", "within-nomix", "cross-recipe"):
        vals = [r["spearman_pooled"] for r in agree if r["group"] == group]
        if vals:
            print(f"  {group}: pooled Spearman "
                  f"{min(vals):.3f}..{max(vals):.3f} "
                  f"(mean {np.mean(vals):.3f}, {len(vals)} pairs)")

    figure(gates, Path(args.fig), layer=args.layer)


if __name__ == "__main__":
    main()
