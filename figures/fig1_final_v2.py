#!/usr/bin/env python3
"""
figures/fig1_final_v2.py
=========================
Final NeurIPS teaser figure — SAGA.

Fixes vs previous version:
  - Tight row/column spacing matching fig1_teaser_final.py
  - Inferno colormap only (no viridis conflict with teal border)
  - Alpha floor = 0 (background shows cleanly, only high-attention gets colour)
  - Row 1 image changed to ImageNet bird-against-sky
  - Row 2 COCO tennis player (clear sink bottom-left, works well)
  - Row 3 CUB bird in tree (fine-grained)

Layout:
    3 rows x 4 columns
    Input | ViT | ViT + Registers | SAGA (ours)

Saves:
    <out_dir>/fig1_final_v2.png
    <out_dir>/fig1_final_v2.pdf

Usage:
    python3 figures/fig1_final_v2.py \
        --paths_e2      /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --imagenet_tar  /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
        --coco_zip      /home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip \
        --cub_tar       /home/woody/iwi5/iwi5359h/Data/CUB-200/CUB_200_2011.tgz \
        --out_dir       /home/woody/iwi5/iwi5359h/saga_figures/final
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
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
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga import build_saga_vit


# ── Colour constants ───────────────────────────────────────────────────────────
NAVY  = "#1B2A4A"
TEAL  = "#00A99D"
BLACK = "#000000"
MEAN  = [0.485, 0.456, 0.406]
STD   = [0.229, 0.224, 0.225]


# ── Row configuration ──────────────────────────────────────────────────────────
# Row 1: ImageNet bird against plain sky — large uniform background,
#         sinks maximally visible
# Row 2: COCO tennis player — clear sink bottom-left, confirmed working well
# Row 3: CUB fine-grained bird — connects to E6 story

FINAL_ROWS = [
    {
        "dataset":  "imagenet",
        "filename": "ILSVRC2012_val_00002425_n01530575.JPEG",
        "note":     "Bird against sky (ImageNet brambling)",
    },
    {
        "dataset":  "coco",
        "filename": "000000055950.jpg",
        "note":     "Tennis player — clear sink bottom-left (COCO)",
    },
    {
        "dataset":  "ade",
        "filename": "ADE_val_00000775.jpg",
        "note":     "Indoor scene (ADE20K)",
    },
]


# ── Staging helpers ────────────────────────────────────────────────────────────

def safe_cleanup(d):
    if Path(d).exists():
        shutil.rmtree(d)
        print(f"  Cleaned: {d}")

def extract_from_tar(archive, wanted_files, stage_dir):
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    wanted = set(wanted_files)
    found  = {}
    print(f"  Scanning tar: {Path(archive).name}")
    with tarfile.open(archive, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            base = Path(m.name).name
            if base in wanted and base not in found:
                src = tf.extractfile(m)
                if src is None:
                    continue
                out = Path(stage_dir) / base
                out.write_bytes(src.read())
                found[base] = str(out)
                print(f"  Extracted: {base}")
            if len(found) == len(wanted):
                break
    missing = wanted - set(found.keys())
    if missing:
        raise FileNotFoundError(
            f"Not found in {archive}: {missing}")
    return found

def extract_from_zip(archive, wanted_files, stage_dir):
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    wanted = set(wanted_files)
    found  = {}
    print(f"  Scanning zip: {Path(archive).name}")
    with zipfile.ZipFile(archive, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            base = Path(name).name
            if base in wanted and base not in found:
                out = Path(stage_dir) / base
                out.write_bytes(zf.read(name))
                found[base] = str(out)
                print(f"  Extracted: {base}")
            if len(found) == len(wanted):
                break
    missing = wanted - set(found.keys())
    if missing:
        raise FileNotFoundError(
            f"Not found in {archive}: {missing}")
    return found

def extract_from_tgz(archive, wanted_files, stage_dir):
    """Same as tar but for .tgz (CUB)."""
    return extract_from_tar(archive, wanted_files, stage_dir)


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(path, size=224):
    img     = Image.open(path).convert("RGB")
    img     = TF.resize(img, size + 32)
    img     = TF.center_crop(img, size)
    display = np.array(img)
    tensor  = TF.normalize(TF.to_tensor(img), MEAN, STD).unsqueeze(0)
    return display, tensor


# ── Attention extraction ───────────────────────────────────────────────────────

class AttentionExtractor:
    """
    Last-block CLS-to-patch attention, averaged over all heads.
    Handles CLS-only prefix (baseline/SAGA) and
    CLS + register prefix (registers model).
    """

    def __init__(self, model, name):
        self.model  = model
        self.name   = name
        self._attn  = None
        self._prefix = self._count_prefix(model)
        self._hook   = model.blocks[-1].attn.register_forward_hook(
            self._hook_fn)
        print(f"  Extractor [{name}]  prefix_tokens={self._prefix}")

    def _count_prefix(self, m):
        if hasattr(m, "num_prefix_tokens"):
            return int(m.num_prefix_tokens)
        if hasattr(m, "reg_token") and m.reg_token is not None:
            return 1 + int(m.reg_token.shape[1])
        return 1

    def _hook_fn(self, module, inp, out):
        x       = inp[0]
        B, N, C = x.shape
        H       = module.num_heads
        D       = getattr(module, "head_dim", C // H)
        with torch.no_grad():
            qkv    = module.qkv(x).reshape(B, N, 3, H, D).permute(2,0,3,1,4)
            q, k, _ = qkv.unbind(0)
            q = getattr(module, "q_norm", nn.Identity())(q)
            k = getattr(module, "k_norm", nn.Identity())(k)
            scale = getattr(module, "scale", D**-0.5)
            attn  = (q * scale) @ k.transpose(-2, -1)
            attn  = attn.softmax(dim=-1)          # [B, H, N, N]
        # CLS row → patch tokens (skip register tokens if any)
        self._attn = attn[:, :, 0, self._prefix:].mean(1).squeeze(0).cpu()

    def extract(self, tensor, device):
        self._attn = None
        with torch.no_grad():
            _ = self.model(tensor.to(device))
        a  = self._attn.numpy()
        gs = int(a.shape[0] ** 0.5)
        return a.reshape(gs, gs)   # [14, 14]

    def remove(self):
        self._hook.remove()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(key, ckpt_dir, device):
    name_map = {
        "baseline":  "ViT-B_baseline_nomix",
        "registers": "ViT-B_registers_nomix",
        "saga":      "ViT-B_SAGA_nomix",
    }
    ckpt_path = Path(ckpt_dir) / name_map[key] / "best.pth"
    print(f"  Loading {key}...")

    if key == "registers":
        import timm
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


# ── Visualisation helpers ──────────────────────────────────────────────────────

def row_normalise(maps, lo_pct=2.0, hi_pct=98.0):
    """
    Shared percentile normalisation across the 3 method maps in one row.
    Using 2-98 percentile avoids outlier sinks dominating the scale.
    """
    stacked = np.stack(maps)
    lo = float(np.percentile(stacked, lo_pct))
    hi = float(np.percentile(stacked, hi_pct))
    if hi - lo < 1e-10:
        return [np.zeros_like(m) for m in maps]
    return [np.clip((m - lo) / (hi - lo), 0, 1) for m in maps]

def upsample_attn(attn, h, w):
    pil = Image.fromarray((attn * 255).astype(np.uint8), "L")
    return np.array(pil.resize((w, h), Image.BILINEAR)) / 255.0

def make_alpha(attn_up, alpha_max=0.82, gamma=1.2):
    """
    Key fix: alpha floor = 0.
    Background (low attention) → fully transparent → image shows cleanly.
    High attention → alpha_max → inferno colour clearly visible.
    gamma > 1 sharpens the transition — peaks become crisper.
    """
    a = np.power(np.clip(attn_up, 0, 1), gamma) * alpha_max
    return np.clip(a, 0, 1)


# ── Figure rendering ───────────────────────────────────────────────────────────

def render_figure(rows_data, out_png, out_pdf,
                  lo_pct=2.0, hi_pct=98.0,
                  alpha_max=0.82, gamma=1.2):
    """
    Tight spacing matching fig1_teaser_final.py.
    Inferno colormap with alpha-floor-zero overlay.
    """
    n = len(rows_data)

    # ── Figure size — NeurIPS two-column span, compact height ─────────────────
    fig, axes = plt.subplots(
        n, 4,
        figsize=(6.25, n * 1.60),    # same width as original, tighter height
    )

    # ── Tight spacing — matches original code ─────────────────────────────────
    fig.subplots_adjust(
        left   = 0.02,
        right  = 0.985,
        top    = 0.91,
        bottom = 0.02,
        wspace = 0.035,    # very tight columns — same as original
        hspace = 0.055,    # very tight rows    — same as original
    )

    cmap = plt.get_cmap("inferno")

    for row_idx, row in enumerate(rows_data):
        display    = row["display"]
        h, w       = display.shape[:2]

        # Shared normalisation across all 3 method maps in this row
        norm_maps = row_normalise(
            [row["baseline"], row["registers"], row["saga"]],
            lo_pct=lo_pct, hi_pct=hi_pct)

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]

            # Col 0: input image, no overlay
            if col_idx == 0:
                ax.imshow(display)

            else:
                m_norm   = norm_maps[col_idx - 1]
                m_up     = upsample_attn(m_norm, h, w)
                alpha    = make_alpha(m_up, alpha_max=alpha_max, gamma=gamma)

                ax.imshow(display)
                ax.imshow(m_up, cmap=cmap, vmin=0, vmax=1,
                          alpha=alpha, interpolation="bilinear")

            ax.set_xticks([])
            ax.set_yticks([])

            # Remove spines on all except SAGA column
            if col_idx != 3:
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # ── Column headers ─────────────────────────────────────────────────────────
    col_labels = ["Input", "ViT", "ViT + Registers", "SAGA (ours)"]
    for col, label in enumerate(col_labels):
        ax = axes[0, col]
        if label == "SAGA (ours)":
            # "SAGA" — navy bold,  "(ours)" — teal italic
            # Two text objects side by side, centred over the column
            ax.text(0.50, 1.055, "SAGA",
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=10.2, fontweight="bold", color=NAVY)
            ax.text(0.52, 1.055, "(ours)",
                    transform=ax.transAxes,
                    ha="left", va="bottom",
                    fontsize=8.8, fontstyle="italic", color=TEAL)
        else:
            weight = "bold" if col > 0 else "regular"
            ax.set_title(label, fontsize=10.2, color=NAVY,
                         pad=5.0, fontweight=weight)

    # ── Teal border — full SAGA column ────────────────────────────────────────
    for row_idx in range(n):
        for spine in axes[row_idx, 3].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(TEAL)
            spine.set_linewidth(1.5)

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf,           bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {Path(out_png).name}")
    print(f"  Saved: {Path(out_pdf).name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_e2",     required=True)
    parser.add_argument("--imagenet_tar", required=True)
    parser.add_argument("--coco_zip",     required=True)
    parser.add_argument("--ade_zip",      required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--stage_base",
        default=str(Path(tempfile.gettempdir()) /
                    f"fig1_v2_{os.environ.get('USER','u')}"))
    parser.add_argument("--lo_pct",    type=float, default=2.0)
    parser.add_argument("--hi_pct",    type=float, default=98.0)
    parser.add_argument("--alpha_max", type=float, default=0.82)
    parser.add_argument("--gamma",     type=float, default=1.2)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stage = Path(args.stage_base)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Models ─────────────────────────────────────────────────────────────────
    print("Loading models...")
    models     = {k: load_model(k, args.paths_e2, device)
                  for k in ["baseline", "registers", "saga"]}
    extractors = {k: AttentionExtractor(m, k) for k, m in models.items()}
    print()

    # ── Stage only the 3 needed images ────────────────────────────────────────
    print("Staging images...")
    inet_files = extract_from_tar(
        args.imagenet_tar,
        ["ILSVRC2012_val_00002425_n01530575.JPEG"],
        str(stage / "inet"))

    coco_files = extract_from_zip(
        args.coco_zip,
        ["000000055950.jpg"],
        str(stage / "coco"))

    ade_files = extract_from_zip(
        args.ade_zip,
        ["ADE_val_00000775.jpg"],
        str(stage / "ade"))

    path_map = {
        "imagenet": inet_files,
        "coco":     coco_files,
        "ade":      ade_files,
    }
    print()

    # ── Inference ──────────────────────────────────────────────────────────────
    print("Running inference...")
    rows_data = []
    for row in FINAL_ROWS:
        img_path = path_map[row["dataset"]][row["filename"]]
        print(f"  {row['note']}")
        display, tensor = load_image(img_path)
        rows_data.append({
            "display":   display,
            "baseline":  extractors["baseline"].extract(tensor, device),
            "registers": extractors["registers"].extract(tensor, device),
            "saga":      extractors["saga"].extract(tensor, device),
        })
    print()

    # ── Render ─────────────────────────────────────────────────────────────────
    print("Rendering figure...")
    render_figure(
        rows_data,
        out_png   = str(Path(args.out_dir) / "fig1_final_v2.png"),
        out_pdf   = str(Path(args.out_dir) / "fig1_final_v2.pdf"),
        lo_pct    = args.lo_pct,
        hi_pct    = args.hi_pct,
        alpha_max = args.alpha_max,
        gamma     = args.gamma,
    )

    # Cleanup
    for ext in extractors.values():
        ext.remove()
    safe_cleanup(str(stage))

    print(f"\nDone. Outputs in: {args.out_dir}")
    print("  fig1_final_v2.png")
    print("  fig1_final_v2.pdf")

if __name__ == "__main__":
    main()