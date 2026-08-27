#!/usr/bin/env python3
"""
figures/fig1_select_coco.py
============================
Automated image selection for Figure 1 teaser.

Samples N images from COCO val2017, runs all 3 models, scores each image
by how well it demonstrates SAGA's advantage over both ViT baseline and
registers. Saves top-30 ranked 4-column figures using the EXACT same
visual design as fig1_teaser_final.py.

Scoring (higher = better teaser candidate):
  sink_contrast   = max(attn_baseline) / mean(attn_baseline)
      → ViT has obvious concentrated sinks
  saga_focus      = mean(attn_saga in top-20% region) / mean(attn_saga)
      → SAGA concentrates attention (foreground focus)
  reg_saga_diff   = mean_abs_diff(attn_registers, attn_saga)
      → Registers and SAGA look visually different (makes comparison clear)
  combined        = sink_contrast × saga_focus × reg_saga_diff

Usage:
    python3 figures/fig1_select_coco.py \
        --paths_e2   /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --coco_zip   /home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip \
        --out_dir    /home/woody/iwi5/iwi5359h/saga_figures/selection \
        --n_sample   500 \
        --top_k      30
"""

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"]  = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_rgb
from PIL import Image
import torchvision.transforms.functional as TF
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga import build_saga_vit


# ── Visual style — identical to fig1_teaser_final.py ──────────────────────────
NAVY  = "#1B2A4A"
SLATE = "#6B7E9B"
TEXT  = "#111111"
TEAL  = "#00A99D"
AMBER = "#F5A623"

VEIL_BLUE      = "#2E5E88"
BLUE_VEIL_ALPHA = 0.42
ATTN_ALPHA_MAX  = 0.95
ATTN_POWER      = 1.20

ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attn_teal",
    ["#3E6F99", "#22B8C8", "#FFC933", "#FFF2A6"],
    N=256,
)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Staging ────────────────────────────────────────────────────────────────────

def stage_coco_val(zip_path, stage_dir, n_sample, seed=42):
    """
    Extract n_sample random images from COCO val2017.zip.
    Returns list of extracted image paths.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Opening COCO val2017.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_entries = [n for n in zf.namelist()
                       if n.startswith("val2017/") and n.endswith(".jpg")]
        print(f"  Total COCO val images: {len(all_entries)}")

        random.seed(seed)
        selected = random.sample(all_entries, min(n_sample, len(all_entries)))
        print(f"  Sampling {len(selected)} images...")

        paths = []
        for name in selected:
            base = Path(name).name
            out  = stage_dir / base
            out.write_bytes(zf.read(name))
            paths.append(str(out))

    print(f"  Staged {len(paths)} images to {stage_dir}")
    return paths


def cleanup(stage_dir):
    if Path(stage_dir).exists():
        shutil.rmtree(stage_dir)
        print(f"  Cleaned up: {stage_dir}")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(key, paths_e2, device):
    name_map = {
        "baseline":  "ViT-B_baseline_nomix",
        "registers": "ViT-B_registers_nomix",
        "saga":      "ViT-B_SAGA_nomix",
    }
    ckpt_path = Path(paths_e2) / name_map[key] / "best.pth"
    print(f"  Loading {key}...")
    if key == "registers":
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=False,
            num_classes=1000, reg_tokens=4, dynamic_img_size=True)
    else:
        model = build_saga_vit(
            "vit_base_patch16_224", gate=(key == "saga"),
            img_size=224, num_classes=1000, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = {k.replace("module.", ""): v
             for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(state, strict=False)
    print(f"  top-1={ckpt.get('top1','?')}%")
    return model.to(device).eval()


# ── Attention extraction ───────────────────────────────────────────────────────

class AttentionExtractor:
    def __init__(self, model, name):
        self.model  = model
        self.name   = name
        self._attn  = None
        self.num_prefix_tokens = self._infer_prefix(model)
        self._hook  = model.blocks[-1].attn.register_forward_hook(self._fn)

    def _infer_prefix(self, m):
        if hasattr(m, "num_prefix_tokens"):
            return int(m.num_prefix_tokens)
        if hasattr(m, "num_reg_tokens"):
            return 1 + int(m.num_reg_tokens)
        if hasattr(m, "reg_token") and m.reg_token is not None:
            if m.reg_token.ndim == 3:
                return 1 + int(m.reg_token.shape[1])
        return 1

    def _fn(self, module, inputs, output):
        x       = inputs[0]
        B, N, C = x.shape
        H       = module.num_heads
        D       = C // H
        with torch.no_grad():
            qkv     = module.qkv(x).reshape(B, N, 3, H, D).permute(2,0,3,1,4)
            q, k, _ = qkv.unbind(0)
            q_norm  = getattr(module, "q_norm", None)
            k_norm  = getattr(module, "k_norm", None)
            if q_norm: q = q_norm(q)
            if k_norm: k = k_norm(k)
            scale   = getattr(module, "scale", D**-0.5)
            attn    = ((q * scale) @ k.transpose(-2,-1)).softmax(dim=-1)
            cls_attn = attn[:,:,0, self.num_prefix_tokens:].mean(1).squeeze(0).cpu()
        self._attn = cls_attn

    def extract(self, tensor, device):
        self._attn = None
        with torch.no_grad():
            _ = self.model(tensor.to(device))
        a  = self._attn.numpy()
        gs = int(a.shape[0]**0.5)
        return a.reshape(gs, gs)

    def remove(self):
        self._hook.remove()


# ── Image scoring ──────────────────────────────────────────────────────────────

def score_image(attn_b, attn_r, attn_s):
    """
    Score one image for teaser quality.

    sink_contrast  : ViT has obvious concentrated sinks
    saga_focus     : SAGA concentrates on foreground (relative to its own mean)
    reg_saga_diff  : Registers and SAGA look visually different
    combined       : product of all three
    """
    # Flatten to 1D
    b = attn_b.flatten()
    r = attn_r.flatten()
    s = attn_s.flatten()

    # Score 1: sink contrast in ViT baseline
    sink_contrast = float(b.max() / (b.mean() + 1e-8))

    # Score 2: SAGA focus — how much attention is concentrated in top-20% patches
    thresh      = np.percentile(s, 80)
    saga_focus  = float(s[s >= thresh].mean() / (s.mean() + 1e-8))

    # Score 3: difference between registers and SAGA
    reg_saga_diff = float(np.abs(r - s).mean())

    combined = sink_contrast * saga_focus * reg_saga_diff

    return {
        "sink_contrast":  round(sink_contrast,  4),
        "saga_focus":     round(saga_focus,      4),
        "reg_saga_diff":  round(reg_saga_diff,   6),
        "combined":       round(combined,         6),
    }


# ── Visualisation — identical to fig1_teaser_final.py ─────────────────────────

def make_blue_veil(h, w):
    veil      = np.zeros((h, w, 4), dtype=np.float32)
    r, g, b   = to_rgb(VEIL_BLUE)
    veil[...,0] = r
    veil[...,1] = g
    veil[...,2] = b
    veil[...,3] = BLUE_VEIL_ALPHA
    return veil


def upsample_attn(attn_map, out_w, out_h):
    arr = np.clip(attn_map, 0, 1)
    up  = Image.fromarray((arr*255).astype(np.uint8)).resize(
        (out_w, out_h), Image.BILINEAR)
    return np.asarray(up).astype(np.float32) / 255.0


def save_candidate_figure(display, attn_b, attn_r, attn_s,
                           rank, score, filename, out_path):
    """
    Save one 1-row × 4-column candidate figure.
    Identical visual design to fig1_teaser_final.py.
    Adds rank, score and filename as suptitle for easy comparison.
    """
    h, w = display.shape[:2]

    all_maps = np.stack([attn_b, attn_r, attn_s])
    vmin = np.percentile(all_maps, 10)
    vmax = np.percentile(all_maps, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-8

    fig, axes = plt.subplots(1, 4, figsize=(7.15, 1.95))
    fig.subplots_adjust(left=0.04, right=0.91,
                        top=0.82, bottom=0.04,
                        wspace=0.045, hspace=0.07)

    raws = [None, attn_b, attn_r, attn_s]

    for c_idx in range(4):
        ax = axes[c_idx]
        ax.imshow(display)

        if c_idx > 0:
            raw    = raws[c_idx]
            norm   = np.clip((raw - vmin) / (vmax - vmin + 1e-8), 0, 1)
            attn_up = upsample_attn(norm, w, h)

            veil = make_blue_veil(h, w)
            ax.imshow(veil, interpolation="nearest")

            alpha_map = np.power(attn_up, ATTN_POWER)
            ax.imshow(attn_up, cmap=ATTN_CMAP,
                      alpha=ATTN_ALPHA_MAX * alpha_map,
                      vmin=0.0, vmax=1.0, interpolation="bilinear")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if c_idx == 3:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(AMBER)
                spine.set_linewidth(2.3)

    # Column headers
    headers = ["Input", "ViT", "ViT+Registers", "SAGA (ours)"]
    for i, title in enumerate(headers):
        ax = axes[i]
        if i == 3:
            ax.text(0.48, 1.06, "SAGA", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10.5,
                    fontweight="bold", color=TEXT)
            ax.text(0.50, 1.06, "(ours)", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=10.0,
                    fontstyle="italic", color=TEAL)
        else:
            ax.set_title(title, fontsize=10.5, fontweight="bold",
                         color=TEXT, pad=8)

    # Suptitle with rank + score + filename
    fig.suptitle(
        f"Rank {rank:02d}  |  combined={score['combined']:.4f}  "
        f"sink={score['sink_contrast']:.2f}  "
        f"focus={score['saga_focus']:.2f}  "
        f"diff={score['reg_saga_diff']:.4f}\n{filename}",
        fontsize=7.5, color=NAVY, y=0.98,
    )

    # Shared colorbar
    sm  = ScalarMappable(norm=Normalize(0, 1), cmap=ATTN_CMAP)
    sm.set_array(np.linspace(0, 1, 256))
    cax = fig.add_axes([0.925, 0.10, 0.018, 0.70])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["0", "0.5", "1"])
    cb.set_label("Attention", fontsize=8, color=TEXT, labelpad=5)
    cb.ax.tick_params(labelsize=7.5, colors=TEXT, length=2)
    cb.outline.set_edgecolor(SLATE)

    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_e2",  required=True)
    parser.add_argument("--coco_zip",  required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--n_sample",  type=int, default=500,
                        help="Number of COCO images to evaluate")
    parser.add_argument("--top_k",     type=int, default=30,
                        help="Save figures for top-K ranked images")
    parser.add_argument("--stage_base", default="/tmp/fig1_select_stage")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading models...")
    models     = {k: load_model(k, args.paths_e2, device)
                  for k in ["baseline", "registers", "saga"]}
    extractors = {k: AttentionExtractor(m, k) for k, m in models.items()}
    print()

    # ── Stage COCO images ──────────────────────────────────────────────────────
    stage_dir  = Path(args.stage_base)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    img_paths = stage_coco_val(args.coco_zip, stage_dir, args.n_sample, args.seed)
    print(f"\nScoring {len(img_paths)} images...\n")

    # ── Score all images ───────────────────────────────────────────────────────
    records = []

    for i, img_path in enumerate(img_paths):
        if i % 50 == 0:
            print(f"  [{i}/{len(img_paths)}]", flush=True)

        try:
            img     = Image.open(img_path).convert("RGB")
            img     = TF.resize(img, 256)
            img     = TF.center_crop(img, [224, 224])
            display = np.array(img)
            tensor  = TF.normalize(TF.to_tensor(img), MEAN, STD).unsqueeze(0)

            attn_b = extractors["baseline"].extract(tensor, device)
            attn_r = extractors["registers"].extract(tensor, device)
            attn_s = extractors["saga"].extract(tensor, device)

            sc = score_image(attn_b, attn_r, attn_s)
            sc["filename"] = Path(img_path).name
            sc["path"]     = img_path
            records.append((sc, display, attn_b, attn_r, attn_s))

        except Exception as e:
            print(f"  WARNING: skipping {Path(img_path).name}: {e}")

    # ── Sort by combined score ─────────────────────────────────────────────────
    records.sort(key=lambda x: x[0]["combined"], reverse=True)

    print(f"\n\nTop {args.top_k} candidates:")
    print(f"{'Rank':<6} {'combined':>10} {'sink':>8} {'focus':>8} "
          f"{'diff':>10}  filename")
    print("-" * 70)
    for rank, (sc, *_) in enumerate(records[:args.top_k], start=1):
        print(f"{rank:<6} {sc['combined']:>10.4f} {sc['sink_contrast']:>8.2f} "
              f"{sc['saga_focus']:>8.2f} {sc['reg_saga_diff']:>10.4f}  "
              f"{sc['filename']}")

    # ── Save top-K figures ─────────────────────────────────────────────────────
    print(f"\nSaving top-{args.top_k} candidate figures...")
    figs_dir = out_dir / "candidates"
    figs_dir.mkdir(exist_ok=True)

    for rank, (sc, display, attn_b, attn_r, attn_s) in \
            enumerate(records[:args.top_k], start=1):
        out_path = figs_dir / f"rank{rank:02d}_{sc['filename'].replace('.jpg','')}.png"
        save_candidate_figure(
            display, attn_b, attn_r, attn_s,
            rank=rank, score=sc, filename=sc["filename"],
            out_path=out_path,
        )

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "filename", "combined",
            "sink_contrast", "saga_focus", "reg_saga_diff"])
        writer.writeheader()
        for rank, (sc, *_) in enumerate(records, start=1):
            writer.writerow({
                "rank":          rank,
                "filename":      sc["filename"],
                "combined":      sc["combined"],
                "sink_contrast": sc["sink_contrast"],
                "saga_focus":    sc["saga_focus"],
                "reg_saga_diff": sc["reg_saga_diff"],
            })
    print(f"Scores saved: {csv_path}")

    # Cleanup
    for ext in extractors.values():
        ext.remove()
    cleanup(str(stage_dir))

    print(f"\nDone.")
    print(f"Candidate figures: {figs_dir}/")
    print(f"Scores CSV:        {csv_path}")
    print(f"\nNext: review candidate figures, pick 3 filenames,")
    print(f"      update FINAL_ROWS in fig1_teaser_final.py")

if __name__ == "__main__":
    main()