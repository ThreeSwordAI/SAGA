#!/usr/bin/env python3
"""
figures/fig1_review_grids_inet.py
===================================
Creates 10 review grid figures from ImageNet selection results.
Each figure has 3 rows × 4 columns.

Reads scores_imagenet.csv from fig1_select_imagenet.py.
Extracts only the needed images from the tar — no full extraction.

Groups:
    Figure 01: ranks  1-3
    Figure 02: ranks  4-6
    ...
    Figure 10: ranks 28-30

Visual design: identical to fig1_teaser_final.py.

Usage:
    python3 figures/fig1_review_grids_inet.py \
        --paths_e2      /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --imagenet_tar  /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
        --scores_csv    /home/woody/iwi5/iwi5359h/saga_figures/selection_inet/scores_imagenet.csv \
        --out_dir       /home/woody/iwi5/iwi5359h/saga_figures/selection_inet/grids \
        --top_k         30
"""

import argparse
import csv
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
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

VEIL_BLUE       = "#2E5E88"
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
            cls_attn = attn[:,:,0,self.num_prefix_tokens:].mean(1).squeeze(0).cpu()
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


# ── Stage only needed files from tar ──────────────────────────────────────────

def stage_selected_from_tar(tar_path, filenames, stage_dir):
    """
    Stream through tar once, extract only the requested filenames.
    Much faster than full extraction for a small number of files.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(filenames)
    found  = {}

    print(f"  Scanning tar for {len(wanted)} files...")
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            base = Path(m.name).name
            if base in wanted and base not in found:
                src = tf.extractfile(m)
                if src is None:
                    continue
                out = stage_dir / base
                out.write_bytes(src.read())
                found[base] = str(out)
                print(f"  Extracted: {base}", flush=True)
            if len(found) == len(wanted):
                break

    missing = wanted - set(found.keys())
    if missing:
        raise FileNotFoundError(f"Not found in tar: {missing}")
    return found


def cleanup(d):
    if Path(d).exists():
        shutil.rmtree(d)
        print(f"  Cleaned up: {d}")


# ── Visualisation helpers — identical to fig1_teaser_final.py ─────────────────

def make_blue_veil(h, w):
    veil       = np.zeros((h, w, 4), dtype=np.float32)
    r, g, b    = to_rgb(VEIL_BLUE)
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


# ── 3-row grid — identical layout to fig1_teaser_final.py ────────────────────

def save_grid_figure(rows_data, fig_idx, out_path):
    """
    rows_data: list of 3 dicts with keys:
        display, attn_b, attn_r, attn_s, rank, filename, synset
    """
    n_rows = len(rows_data)

    # ── Exact layout from fig1_teaser_final.py ────────────────────────────────
    fig, axes = plt.subplots(n_rows, 4, figsize=(7.15, 5.55))

    fig.subplots_adjust(
        left   = 0.045,
        right  = 0.915,
        top    = 0.92,
        bottom = 0.05,
        wspace = 0.045,
        hspace = 0.07,
    )

    for r_idx, row in enumerate(rows_data):
        disp = row["display"]
        h, w = disp.shape[:2]

        all_maps = np.stack([row["attn_b"], row["attn_r"], row["attn_s"]])
        vmin = np.percentile(all_maps, 10)
        vmax = np.percentile(all_maps, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-8

        raws = [None, row["attn_b"], row["attn_r"], row["attn_s"]]

        for c_idx in range(4):
            ax = axes[r_idx, c_idx]
            ax.imshow(disp)

            if c_idx > 0:
                raw     = raws[c_idx]
                norm    = np.clip((raw - vmin) / (vmax - vmin + 1e-8), 0, 1)
                attn_up = upsample_attn(norm, w, h)
                veil    = make_blue_veil(h, w)
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

        # Row label — rank + synset + filename
        axes[r_idx, 0].set_ylabel(
            f"#{row['rank']}  [{row['synset']}]\n{row['filename']}",
            fontsize=5.8, color=SLATE,
            rotation=0, ha="right", va="center", labelpad=4,
        )

    # ── Column headers — identical to fig1_teaser_final.py ────────────────────
    headers = ["Input", "ViT", "ViT+Registers", "SAGA (ours)"]
    for i, title in enumerate(headers):
        ax = axes[0, i]
        if i == 3:
            ax.set_title("")
            ax.text(0.48, 1.055, "SAGA",
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=11.5, fontweight="bold", color=TEXT)
            ax.text(0.50, 1.055, "(ours)",
                    transform=ax.transAxes,
                    ha="left", va="bottom",
                    fontsize=11.0, fontstyle="italic", color=TEAL)
        else:
            ax.set_title(title, fontsize=11.7, fontweight="bold",
                         color=TEXT, pad=10)

    # ── Figure title ───────────────────────────────────────────────────────────
    rank_start = (fig_idx - 1) * 3 + 1
    rank_end   = rank_start + 2
    fig.suptitle(
        f"Review Grid {fig_idx:02d}  —  Ranks {rank_start}–{rank_end}  "
        f"(ImageNet val)",
        fontsize=10, color=NAVY, fontweight="bold", y=0.975,
    )

    # ── Shared colorbar — identical to fig1_teaser_final.py ───────────────────
    sm  = ScalarMappable(norm=Normalize(0, 1), cmap=ATTN_CMAP)
    sm.set_array(np.linspace(0, 1, 256))
    cax = fig.add_axes([0.928, 0.08, 0.018, 0.81])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["0.0", "0.5", "1.0"])
    cb.set_label("Attention weight", fontsize=10.6, color=TEXT, labelpad=7)
    cb.ax.tick_params(labelsize=9.6, colors=TEXT, length=3)
    cb.outline.set_edgecolor(SLATE)

    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_e2",     required=True)
    parser.add_argument("--imagenet_tar", required=True)
    parser.add_argument("--scores_csv",   required=True,
                        help="scores_imagenet.csv from fig1_select_imagenet.py")
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--top_k",        type=int, default=30)
    parser.add_argument("--stage_base",   default="/tmp/fig1_inet_grids_stage")
    args = parser.parse_args()

    assert args.top_k % 3 == 0, "--top_k must be divisible by 3"
    n_figs = args.top_k // 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Read CSV ───────────────────────────────────────────────────────────────
    rows = []
    with open(args.scores_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) == args.top_k:
                break

    filenames = [r["filename"] for r in rows]
    print(f"Read {len(filenames)} entries from {args.scores_csv}\n")

    # ── Stage only the needed images from tar ──────────────────────────────────
    stage_dir = Path(args.stage_base)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    path_map = stage_selected_from_tar(args.imagenet_tar, filenames, stage_dir)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading models...")
    models     = {k: load_model(k, args.paths_e2, device)
                  for k in ["baseline", "registers", "saga"]}
    extractors = {k: AttentionExtractor(m, k) for k, m in models.items()}
    print()

    # ── Inference for all images ───────────────────────────────────────────────
    print("Running inference...")
    all_rows = []
    for i, row in enumerate(rows):
        fname   = row["filename"]
        synset  = row.get("synset", "unknown")
        rank    = int(row["rank"])

        img     = Image.open(path_map[fname]).convert("RGB")
        img     = TF.resize(img, 256)
        img     = TF.center_crop(img, [224, 224])
        display = np.array(img)
        tensor  = TF.normalize(TF.to_tensor(img), MEAN, STD).unsqueeze(0)

        attn_b = extractors["baseline"].extract(tensor, device)
        attn_r = extractors["registers"].extract(tensor, device)
        attn_s = extractors["saga"].extract(tensor, device)

        all_rows.append({
            "display":  display,
            "attn_b":   attn_b,
            "attn_r":   attn_r,
            "attn_s":   attn_s,
            "rank":     rank,
            "filename": fname,
            "synset":   synset,
        })
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    # ── Save 10 grid figures ───────────────────────────────────────────────────
    print(f"\nSaving {n_figs} grid figures...")
    for fig_idx in range(1, n_figs + 1):
        start      = (fig_idx - 1) * 3
        group      = all_rows[start:start + 3]
        rank_start = group[0]["rank"]
        rank_end   = group[-1]["rank"]
        out_path   = out_dir / \
            f"grid_{fig_idx:02d}_ranks{rank_start:02d}-{rank_end:02d}.png"
        save_grid_figure(group, fig_idx, out_path)

    # Cleanup
    for ext in extractors.values():
        ext.remove()
    cleanup(str(stage_dir))

    print(f"\nDone. Grids saved to: {out_dir}")
    print("\nFilename reference:")
    for row in rows:
        print(f"  Rank {row['rank']:>2}  [{row.get('synset','?')}]  {row['filename']}")


if __name__ == "__main__":
    main()