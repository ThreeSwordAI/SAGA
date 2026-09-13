#!/usr/bin/env python3
"""
plotting/plot_F7.py
===================
TASK-09 Phase C.3: the dense qualitative draft ->
results/figures/F7_dense_draft.pdf.

Two blocks:
  * ADE20K — the first N_ADE stems of the committed probe list
    (results/probe/ade20k_fixed20.json, frozen before any segmentation ran),
    one row each: input | ground truth | baseline | registers | saga. Every
    panel comes from a committed `preds_fixed20/` file, so this half needs
    no dataset.
  * COCO — the two small-object crops frozen by analysis/collect_F7.py and
    exported by tools/export_f7_crops.py, one row each: baseline | registers
    | saga, with that variant's small detections drawn over the crop and the
    ground-truth small boxes underneath for reference.

If the COCO export has not landed yet the figure still builds, with that
block drawn as an explicit MISSING placeholder naming the command that
fills it — the same self-filling convention the TASK-02B memo used.

Label colours are a deterministic palette (seeded permutation of a
perceptually spread colormap), NOT the official ADE20K colormap, which does
not ship with the archive; the caption says so.

    python plotting/plot_F7.py [--out results/figures/F7_dense_draft.pdf]
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]


def show(path: Path):
    """Repo-relative when the path is inside the repo, absolute otherwise —
    a --out outside the tree (tests, scratch) must not crash the tool."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path

N_ADE = 3                      # TASK-09 C.3: "3 fixed ADE20K images"
# 15 divides by both 5 (ADE columns) and 3 (COCO columns), so each
# block spans the full width instead of leaving empty cells
NCOL = 15
VARIANTS = ["baseline", "registers", "saga"]
VARIANT_COLOR = {"baseline": "#888780", "registers": "#1D9E75",
                 "saga": "#7F77DD"}
GT_COLOR = "#D08770"
INK = "#3a3a3a"
MUTED = "#777777"
IGNORE = 255
PALETTE_SEED = 0


def ade_palette(n=256):
    """Deterministic label colours. Index IGNORE renders white."""
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    rng = random.Random(PALETTE_SEED)
    cols = []
    for i in range(n):
        r, g, b = base[i % 20]
        j = (i // 20) + 1
        f = 0.55 + 0.45 * rng.random()
        cols.append([min(1.0, r * f * j ** 0.0 + 0.12 * rng.random()),
                     min(1.0, g * f + 0.12 * rng.random()),
                     min(1.0, b * f + 0.12 * rng.random())])
    pal = np.array(cols)
    pal[IGNORE] = [1.0, 1.0, 1.0]
    return pal


def colorize(label_png: Path, pal):
    from PIL import Image
    arr = np.array(Image.open(label_png))
    return pal[arr]


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
        sp.set_linewidth(0.6)


def draw_ade_block(fig, gs, stems, seg_root: Path, pal):
    cols = ["input", "ground truth"] + VARIANTS
    span = NCOL // len(cols)       # 5 panels across the 15-column grid
    for r, stem in enumerate(stems):
        for c, col in enumerate(cols):
            ax = fig.add_subplot(gs[r, c * span:(c + 1) * span])
            bare(ax)
            try:
                if col == "input":
                    from PIL import Image
                    src = (seg_root / f"seg_vitb_{VARIANTS[0]}_s1"
                           / "preds_fixed20" / f"{stem}_img.jpg")
                    ax.imshow(np.array(Image.open(src)))
                elif col == "ground truth":
                    src = (seg_root / f"seg_vitb_{VARIANTS[0]}_s1"
                           / "preds_fixed20" / f"{stem}_gt.png")
                    ax.imshow(colorize(src, pal))
                else:
                    src = (seg_root / f"seg_vitb_{col}_s1" / "preds_fixed20"
                           / f"{stem}_pred.png")
                    ax.imshow(colorize(src, pal))
            except (OSError, FileNotFoundError):
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center",
                        fontsize=7, color=MUTED, transform=ax.transAxes)
            if r == 0:
                ax.set_title(col, fontsize=8, color=INK, pad=4)
            if c == 0:
                ax.set_ylabel(stem.replace("ADE_val_", ""), fontsize=6.5,
                              color=MUTED)


def draw_coco_block(fig, gs, row0, crops, data_dir: Path):
    from PIL import Image
    for r, rec in enumerate(crops):
        img = np.array(Image.open(data_dir / rec["crop_file"]))
        gt_small = [g for g in rec["gt_in_crop"] if g.get("is_small")]
        # NOT `w`: the bbox unpacking below rebinds that name
        span = NCOL // len(VARIANTS)   # 3 panels across the same grid
        for c, variant in enumerate(VARIANTS):
            ax = fig.add_subplot(gs[row0 + r,
                                    c * span:(c + 1) * span])
            bare(ax)
            ax.imshow(img)
            for g in gt_small:
                x, y, w, h = g["bbox_crop"]
                ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                       edgecolor=GT_COLOR, linewidth=1.4,
                                       linestyle=(0, (3, 2))))
            dets = rec["detections_in_crop"].get(variant, [])
            for d in dets:
                x, y, w, h = d["bbox_crop"]
                ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                       edgecolor=VARIANT_COLOR[variant],
                                       linewidth=1.0))
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            n = rec["n_small"].get(variant, "?")
            ax.set_title(f"{variant}  ({n} small dets)", fontsize=8,
                         color=VARIANT_COLOR[variant], pad=4)
            if c == 0:
                ax.set_ylabel(f"COCO {rec['image_id']}", fontsize=6.5,
                              color=MUTED)


def draw_coco_missing(fig, gs, row0, nrows, selection):
    ax = fig.add_subplot(gs[row0:row0 + nrows, :])
    ax.axis("off")
    ids = ", ".join(str(im["image_id"]) for im in selection["images"])
    ax.text(0.5, 0.5,
            "COCO crops MISSING\n\n"
            f"selection is frozen (images {ids});\n"
            "the val2017 JPEGs live only on the HPC.\n"
            "Fill with:  python tools/export_f7_crops.py",
            ha="center", va="center", fontsize=9, color=MUTED,
            linespacing=1.6)
    ax.add_patch(Rectangle((0.02, 0.05), 0.96, 0.9, fill=False,
                           edgecolor="#cccccc", linewidth=0.8,
                           linestyle=(0, (4, 3)), transform=ax.transAxes))


def main():
    p = argparse.ArgumentParser("TASK-09 Phase C: F7 dense qualitative draft")
    p.add_argument("--seg-root", default="results/segmentation")
    p.add_argument("--probe", default="results/probe/ade20k_fixed20.json")
    p.add_argument("--selection",
                   default="results/figures_data/F7_coco_selection.json")
    p.add_argument("--coco-dir", default="results/figures_data/f7_coco")
    p.add_argument("--out", default="results/figures/F7_dense_draft.pdf")
    args = p.parse_args()

    def rel(x):
        q = Path(x)
        return q if q.is_absolute() else REPO / q

    probe = json.loads(rel(args.probe).read_text(encoding="utf-8"))
    stems = probe["stems"][:N_ADE]
    selection = json.loads(rel(args.selection).read_text(encoding="utf-8"))

    coco_dir = rel(args.coco_dir)
    crops_json = coco_dir / "crops.json"
    crops = None
    if crops_json.exists():
        payload = json.loads(crops_json.read_text(encoding="utf-8"))
        if all((coco_dir / r["crop_file"]).exists()
               for r in payload["images"]):
            crops = payload["images"]

    n_coco = len(selection["images"])
    nrows = N_ADE + n_coco
    fig = plt.figure(figsize=(12.5, 2.5 * nrows + 1.1))
    gs = fig.add_gridspec(nrows, NCOL, hspace=0.14, wspace=0.06,
                          top=0.90, bottom=0.06, left=0.045, right=0.99)

    draw_ade_block(fig, gs, stems, rel(args.seg_root), ade_palette())
    if crops:
        draw_coco_block(fig, gs, N_ADE, crops, coco_dir)
    else:
        draw_coco_missing(fig, gs, N_ADE, n_coco, selection)

    fig.suptitle("F7 (draft) — dense prediction, ViT-B/16: ADE20K semantic "
                 "segmentation and COCO small objects", fontsize=11,
                 color=INK, y=0.975)
    fig.text(0.045, 0.018,
             "ADE20K: the first 3 stems of the frozen 20-image probe list, "
             "rendered from each run's committed preds_fixed20/. Label "
             "colours are a deterministic palette, not the official ADE20K "
             "colormap (which the archive does not ship); unlabelled (255) "
             "is white.   COCO: dashed = ground-truth small boxes, solid = "
             "that variant's small detections (area < 32^2 px, score >= "
             "0.30).",
             fontsize=6.5, color=MUTED, va="bottom", wrap=True)

    out = rel(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"wrote {show(out)}")
    print(f"wrote {show(png)}")
    print(f"  ADE rows: {stems}")
    print(f"  COCO block: {'rendered' if crops else 'MISSING placeholder'}")


if __name__ == "__main__":
    main()
