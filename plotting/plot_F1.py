#!/usr/bin/env python3
"""
plotting/plot_F1.py
===================
Figure 1 (teaser) from the committed archives — no inference, no dataset,
no checkpoints:

    results/figures_data/F1_teaser.npz        (analysis/collect_F1.py, on HPC)
    results/figures_data/F1_localization.csv  (tools/localization_score.py)

    python plotting/plot_F1.py --images <id>,<id>,<id>

Layout
  rows    3 chosen curated images
  cols    Input | Baseline | Registers | TTR | SAGA
  under each model column: the MEAN localization score over the POPULATION
  (`boxes20` in-box mass, `random200` ring-1 mass) — never the three
  displayed images, which are illustrations and not evidence.
  bottom strip: baseline sink-frequency map | ring-1 mask | SAGA layer-7/8 gate
  — the address, and the gate that suppresses it, side by side.

A column with no dump renders as an explicit "pending" panel; it is never
silently dropped and never filled with zeros.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMN_ORDER = ("Baseline", "Registers", "TTR", "SAGA")
POPULATION_METRIC = {"boxes20": "inbox_mass", "random200": "ring1_mass"}


def load_archive(path):
    z = np.load(path, allow_pickle=False)
    index = json.loads(str(z["index"]))
    return z, index


def load_localization(path):
    """(run_id, group, metric) -> list of per-image values."""
    vals = defaultdict(list)
    if not Path(path).exists():
        return vals
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vals[(row["run_id"], row["group"], row["metric"])].append(
                float(row["value"]))
    return vals


def column_runs(index, explicit):
    """label -> run_id (or None = pending)."""
    if explicit:
        out = {}
        for part in explicit.split(","):
            label, _, run = part.partition("=")
            out[label.strip()] = run.strip() or None
        return out
    by_variant = {}
    for run_id, prov in index.get("provenance", {}).items():
        by_variant.setdefault(prov.get("variant"), run_id)
    return {"Baseline": by_variant.get("baseline"),
            "Registers": by_variant.get("registers"),
            "TTR": by_variant.get("ttr"),
            "SAGA": by_variant.get("saga")}


def population_caption(vals, run_id):
    if run_id is None:
        return "pending"
    bits = []
    for group, metric in sorted(POPULATION_METRIC.items()):
        v = vals.get((run_id, group, metric))
        if v:
            bits.append(f"{metric.replace('_', ' ')} "
                        f"{np.mean(v):.4f} (n={len(v)})")
    return "\n".join(bits) if bits else "no population rows"


def overlay(ax, thumb, attn, title=None):
    import matplotlib.pyplot as plt         # noqa: F401
    ax.imshow(thumb)
    if attn is not None:
        side = thumb.shape[0]
        ax.imshow(np.kron(attn, np.ones((side // attn.shape[0],) * 2)),
                  cmap="inferno", alpha=0.55, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def pending_panel(ax, label):
    ax.text(0.5, 0.5, f"{label}\npending", ha="center", va="center",
            fontsize=10, color="0.35", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linestyle(":"); s.set_color("0.6")


def main():
    p = argparse.ArgumentParser(description="Render the F1 teaser draft.")
    p.add_argument("--archive", default="results/figures_data/F1_teaser.npz")
    p.add_argument("--localization",
                   default="results/figures_data/F1_localization.csv")
    p.add_argument("--images", default=None,
                   help="comma-separated image_ids (default: first of each "
                        "curated criterion)")
    p.add_argument("--columns", default=None,
                   help="'Baseline=run,Registers=run,TTR=,SAGA=run'")
    p.add_argument("--gate-layer", type=int, default=8)
    p.add_argument("--out", default="results/figures/F1_teaser_draft.pdf")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, index = load_archive(args.archive)
    vals = load_localization(args.localization)
    cols = column_runs(index, args.columns)

    if args.images:
        chosen = [s.strip() for s in args.images.split(",") if s.strip()]
    else:
        seen, chosen = set(), []
        for im in index["images"]:
            if im["criterion"] not in seen:
                seen.add(im["criterion"]); chosen.append(im["image_id"])
        chosen = chosen[:3]
    by_id = {im["image_id"]: im for im in index["images"]}

    n_rows, n_cols = len(chosen), 1 + len(COLUMN_ORDER)
    fig = plt.figure(figsize=(2.1 * n_cols, 2.35 * n_rows + 2.9))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [1.05],
                          hspace=0.28, wspace=0.06)

    for r, image_id in enumerate(chosen):
        thumb = z[f"thumb/{image_id}"]
        meta = by_id.get(image_id, {})
        ax = fig.add_subplot(gs[r, 0])
        overlay(ax, thumb, None, "Input" if r == 0 else None)
        ax.set_ylabel(f"{meta.get('criterion', '?')}\n{image_id.split('/')[-1]}",
                      fontsize=6.5)
        for c, label in enumerate(COLUMN_ORDER, start=1):
            ax = fig.add_subplot(gs[r, c])
            run_id = cols.get(label)
            key = f"attn/{run_id}/{image_id}" if run_id else None
            if run_id is None or key not in z:
                pending_panel(ax, label if r == 0 else "")
                continue
            overlay(ax, thumb, z[key], label if r == 0 else None)

    for c, label in enumerate(COLUMN_ORDER, start=1):
        run_id = cols.get(label)
        fig.text(gs[n_rows - 1, c].get_position(fig).x0
                 + gs[n_rows - 1, c].get_position(fig).width / 2,
                 gs[n_rows - 1, c].get_position(fig).y0 - 0.018,
                 population_caption(vals, run_id),
                 ha="center", va="top", fontsize=6.2)

    # ── bottom strip: the address, the ring, the gate ────────────────────────
    baseline_run, saga_run = cols.get("Baseline"), cols.get("SAGA")
    strip = [
        (f"sink/{baseline_run}" if baseline_run else None,
         f"baseline sink freq\n({baseline_run})", "magma"),
        ("ring1_mask", "ring-1 address mask\n(TASK-07 definition)", "gray"),
        (f"gate/{saga_run}/{args.gate_layer}" if saga_run else None,
         f"SAGA gate layer {args.gate_layer}\n({saga_run})", "viridis"),
    ]
    for i, (key, title, cmap) in enumerate(strip):
        ax = fig.add_subplot(gs[n_rows, i + 1])
        if key is None or key not in z:
            pending_panel(ax, title)
            continue
        arr = np.asarray(z[key], dtype=np.float64)
        im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
            labelsize=5)

    absent = index.get("absent", [])
    fig.suptitle(
        "F1 teaser (draft) — panels are illustrations; the numbers under each "
        "column are population means"
        + (f"\n{len(absent)} absent artifact(s): {', '.join(absent[:3])}"
           f"{' …' if len(absent) > 3 else ''}" if absent else ""),
        fontsize=8.5, y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({n_rows} images x {len(COLUMN_ORDER)} models; "
          f"columns: {cols})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
