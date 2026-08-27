#!/usr/bin/env python3
"""
figures/fig1_variants.py
=========================
Generates 4 variants of the Figure 1 teaser for comparison:

  variant A: inferno colormap  + overlay on image  (current approach)
  variant B: viridis colormap  + overlay on image
  variant C: inferno colormap  + raw attention map  (like registers paper)
  variant D: viridis colormap  + raw attention map  (like registers paper)

Usage (TinyGPU interactive):
    python3 figures/fig1_variants.py \
        --paths_e2     /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --imagenet_tar /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
        --coco_zip     /home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip \
        --voc_tar      /home/woody/iwi5/iwi5359h/Data/VOC/VOCtest_06-Nov-2007.tar \
        --out_dir      /home/woody/iwi5/iwi5359h/saga_figures
"""

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga import build_saga_vit


# ── Colours ────────────────────────────────────────────────────────────────────
NAVY  = "#1B2A4A"
TEAL  = "#00A99D"
BLACK = "#000000"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Locked image selection (from candidate review) ─────────────────────────────
FINAL_ROWS = [
    {"dataset": "voc",      "filename": "003796.jpg"},
    {"dataset": "coco",     "filename": "000000055950.jpg"},
    {"dataset": "imagenet", "filename": "ILSVRC2012_val_00002425_n01530575.JPEG"},
]


# ── Staging ────────────────────────────────────────────────────────────────────

def safe_cleanup(d):
    if Path(d).exists():
        shutil.rmtree(d)

def extract_from_tar(archive, wanted_files, stage_dir):
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    wanted = set(wanted_files)
    found  = {}
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
        raise FileNotFoundError(f"Not found in {archive}: {missing}")
    return found

def extract_from_zip(archive, wanted_files, stage_dir):
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    wanted = set(wanted_files)
    found  = {}
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
        raise FileNotFoundError(f"Not found in {archive}: {missing}")
    return found


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
    def __init__(self, model, name):
        self.model = model
        self.name  = name
        self._attn = None
        # Count prefix tokens (CLS + optional registers)
        self._prefix = self._count_prefix(model)
        self._hook   = model.blocks[-1].attn.register_forward_hook(self._fn)
        print(f"  Extractor [{name}]  prefix_tokens={self._prefix}")

    def _count_prefix(self, m):
        if hasattr(m, "num_prefix_tokens"):
            return int(m.num_prefix_tokens)
        if hasattr(m, "reg_token") and m.reg_token is not None:
            return 1 + int(m.reg_token.shape[1])
        return 1

    def _fn(self, module, inp, out):
        x       = inp[0]
        B, N, C = x.shape
        H       = module.num_heads
        D       = getattr(module, "head_dim", C // H)
        with torch.no_grad():
            qkv     = module.qkv(x).reshape(B, N, 3, H, D).permute(2,0,3,1,4)
            q, k, _ = qkv.unbind(0)
            q_norm  = getattr(module, "q_norm", nn.Identity())
            k_norm  = getattr(module, "k_norm", nn.Identity())
            scale   = getattr(module, "scale", D**-0.5)
            attn    = (q_norm(q) * scale) @ k_norm(k).transpose(-2,-1)
            attn    = attn.softmax(dim=-1)
        # CLS → patch tokens only (skip register tokens)
        self._attn = attn[:,  :, 0, self._prefix:].mean(1).squeeze(0).cpu()

    def extract(self, tensor, device):
        self._attn = None
        with torch.no_grad():
            _ = self.model(tensor.to(device))
        a  = self._attn.numpy()
        gs = int(a.shape[0] ** 0.5)
        return a.reshape(gs, gs)

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
        model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                   num_classes=1000, reg_tokens=4,
                                   dynamic_img_size=True)
    else:
        model = build_saga_vit("vit_base_patch16_224", gate=(key == "saga"),
                               img_size=224, num_classes=1000, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = {k.replace("module.", ""): v
             for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(state, strict=False)
    print(f"  top-1={ckpt.get('top1','?')}%")
    return model.to(device).eval()


# ── Normalisation helpers ──────────────────────────────────────────────────────

def row_normalise(maps, lo_pct=2.0, hi_pct=98.0):
    """Shared percentile normalisation across 3 method maps in one row."""
    stacked = np.stack(maps)
    lo = float(np.percentile(stacked, lo_pct))
    hi = float(np.percentile(stacked, hi_pct))
    if hi - lo < 1e-10:
        return [np.zeros_like(m) for m in maps]
    return [np.clip((m - lo) / (hi - lo), 0, 1) for m in maps]

def upsample(attn, h, w):
    pil = Image.fromarray((attn * 255).astype(np.uint8), "L")
    return np.array(pil.resize((w, h), Image.BILINEAR)) / 255.0


# ── Figure rendering — 4 variants ─────────────────────────────────────────────

def render_variant(rows_data, cmap_name, raw_mode, out_png, out_pdf):
    """
    cmap_name : 'inferno' or 'viridis'
    raw_mode  : True  → show standalone heatmap  (like registers paper)
                False → overlay heatmap on image  (current approach)
    """
    n = len(rows_data)
    fig, axes = plt.subplots(n, 4, figsize=(6.75, n * 2.0),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    if n == 1:
        axes = axes[np.newaxis, :]

    cmap = plt.get_cmap(cmap_name)

    for row_idx, row in enumerate(rows_data):
        display    = row["display"]
        h, w       = display.shape[:2]
        norm_maps  = row_normalise(
            [row["baseline_raw"], row["registers_raw"], row["saga_raw"]])
        maps_up    = [upsample(m, h, w) for m in norm_maps]

        # Col 0 — always show input image
        axes[row_idx, 0].imshow(display)

        # Cols 1-3 — ViT / Registers / SAGA
        for col_idx, m_up in enumerate(maps_up, start=1):
            ax = axes[row_idx, col_idx]
            if raw_mode:
                # Standalone heatmap — no image underneath
                ax.imshow(m_up, cmap=cmap, vmin=0, vmax=1,
                          interpolation="nearest")
            else:
                # Overlay on image
                alpha = np.power(np.clip(m_up, 0, 1), 0.7) * 0.88
                ax.imshow(display)
                ax.imshow(m_up, cmap=cmap, vmin=0, vmax=1,
                          alpha=alpha, interpolation="bilinear")

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    # ── Column headers ─────────────────────────────────────────────────────────
    labels = ["Input", "ViT", "ViT+Registers", "SAGA (ours)"]
    for col, label in enumerate(labels):
        ax = axes[0, col]
        if label == "SAGA (ours)":
            ax.text(0.49, 1.05, "SAGA", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10,
                    fontweight="bold", color=BLACK)
            ax.text(0.51, 1.05, "(ours)", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8.5,
                    fontstyle="italic", color=TEAL)
        else:
            ax.set_title(label, fontsize=10, color=BLACK,
                         pad=5, fontweight="bold" if col > 0 else "regular")

    # ── Teal border on SAGA column ─────────────────────────────────────────────
    for row_idx in range(n):
        for spine in axes[row_idx, 3].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(TEAL)
            spine.set_linewidth(1.6)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight",           facecolor="white")
    plt.close(fig)
    print(f"  Saved: {Path(out_png).name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_e2",     required=True)
    parser.add_argument("--imagenet_tar", required=True)
    parser.add_argument("--coco_zip",     required=True)
    parser.add_argument("--voc_tar",      required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--stage_base",   default="/tmp/fig1_variants_stage")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading models...")
    models     = {k: load_model(k, args.paths_e2, device)
                  for k in ["baseline", "registers", "saga"]}
    extractors = {k: AttentionExtractor(m, k) for k, m in models.items()}
    print()

    # ── Stage images ───────────────────────────────────────────────────────────
    stage = Path(args.stage_base)
    print("Staging images...")
    voc_files = extract_from_tar(
        args.voc_tar, ["003796.jpg"],
        str(stage / "voc"))
    coco_files = extract_from_zip(
        args.coco_zip, ["000000055950.jpg"],
        str(stage / "coco"))
    inet_files = extract_from_tar(
        args.imagenet_tar,
        ["ILSVRC2012_val_00002425_n01530575.JPEG"],
        str(stage / "inet"))

    path_map = {
        "voc":      voc_files,
        "coco":     coco_files,
        "imagenet": inet_files,
    }

    # ── Run inference once — reuse for all variants ────────────────────────────
    print("\nRunning inference...")
    rows_data = []
    for row in FINAL_ROWS:
        img_path = path_map[row["dataset"]][row["filename"]]
        display, tensor = load_image(img_path)
        print(f"  {row['filename']}")
        rows_data.append({
            "display":       display,
            "baseline_raw":  extractors["baseline"].extract(tensor, device),
            "registers_raw": extractors["registers"].extract(tensor, device),
            "saga_raw":      extractors["saga"].extract(tensor, device),
        })

    # ── Render 4 variants ──────────────────────────────────────────────────────
    print("\nRendering variants...")
    od = args.out_dir
    variants = [
        ("inferno", False, "variantA_inferno_overlay"),
        ("viridis", False, "variantB_viridis_overlay"),
        ("inferno", True,  "variantC_inferno_rawmap"),
        ("viridis", True,  "variantD_viridis_rawmap"),
    ]
    for cmap, raw, name in variants:
        render_variant(
            rows_data, cmap, raw,
            f"{od}/{name}.png",
            f"{od}/{name}.pdf",
        )

    # Cleanup
    for ext in extractors.values():
        ext.remove()
    safe_cleanup(str(stage))

    print(f"\nAll 4 variants saved to: {args.out_dir}")
    print("\nVariant guide:")
    print("  A — inferno + overlay  (current approach)")
    print("  B — viridis + overlay")
    print("  C — inferno + raw map  (like registers paper)")
    print("  D — viridis + raw map  (like registers paper + NeurIPS 2025)")

if __name__ == "__main__":
    main()