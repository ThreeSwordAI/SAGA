#!/usr/bin/env python3
"""
figures/fig1_select_imagenet.py
================================
Same as fig1_select_coco.py but reads from ImageNet val_images.tar.gz.

ImageNet val images are single-object, clean background — better for
showing the "attention on right place" argument clearly.

Samples N images from the tar, scores each by:
    sink_contrast  = max(attn_baseline) / mean(attn_baseline)
    saga_focus     = mean of top-20% SAGA attention patches
    reg_saga_diff  = mean_abs_diff(attn_registers, attn_saga)
    combined       = product of all three

Saves top-K ranked 1-row figures + scores.csv.

Usage:
    python3 figures/fig1_select_imagenet.py \
        --paths_e2      /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --imagenet_tar  /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
        --out_dir       /home/woody/iwi5/iwi5359h/saga_figures/selection_inet \
        --n_sample      500 \
        --top_k         30

    # Optional: filter to specific synset classes (e.g. birds, dogs, cars)
    # --synsets n01530575 n01531178 n02084071
"""

import argparse
import csv
import random
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


# ── Staging from tar ───────────────────────────────────────────────────────────

def stage_imagenet_sample(tar_path, stage_dir, n_sample,
                           synset_filter=None, seed=42):
    """
    Stream through ImageNet val tar, collect member list,
    sample n_sample entries, extract only those.

    synset_filter: list of synset IDs to restrict to (e.g. ['n01530575'])
                   None = no filter, all classes included
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Scanning ImageNet tar for member list...")
    with tarfile.open(tar_path, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]

    # Filter by synset if requested
    if synset_filter:
        synset_filter = set(synset_filter)
        members = [m for m in members
                   if any(s in Path(m.name).stem for s in synset_filter)]
        print(f"  After synset filter: {len(members)} images")
    else:
        print(f"  Total images in tar: {len(members)}")

    random.seed(seed)
    selected = random.sample(members, min(n_sample, len(members)))
    print(f"  Sampling {len(selected)} images...")

    paths = []
    with tarfile.open(tar_path, "r:*") as tf:
        # Build a name→member dict for fast lookup
        member_dict = {m.name: m for m in tf.getmembers() if m.isfile()}
        for m in selected:
            src = tf.extractfile(member_dict[m.name])
            if src is None:
                continue
            base = Path(m.name).name
            out  = stage_dir / base
            out.write_bytes(src.read())
            paths.append(str(out))

    print(f"  Staged {len(paths)} images to {stage_dir}")
    return paths


def cleanup(d):
    if Path(d).exists():
        shutil.rmtree(d)
        print(f"  Cleaned up: {d}")


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_image(attn_b, attn_r, attn_s):
    b = attn_b.flatten()
    r = attn_r.flatten()
    s = attn_s.flatten()

    sink_contrast  = float(b.max() / (b.mean() + 1e-8))
    thresh         = np.percentile(s, 80)
    saga_focus     = float(s[s >= thresh].mean() / (s.mean() + 1e-8))
    reg_saga_diff  = float(np.abs(r - s).mean())
    combined       = sink_contrast * saga_focus * reg_saga_diff

    return {
        "sink_contrast":  round(sink_contrast,  4),
        "saga_focus":     round(saga_focus,      4),
        "reg_saga_diff":  round(reg_saga_diff,   6),
        "combined":       round(combined,         6),
    }


# ── Visualisation — identical to fig1_teaser_final.py ─────────────────────────

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


def save_candidate_figure(display, attn_b, attn_r, attn_s,
                           rank, score, filename, synset, out_path):
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

    fig.suptitle(
        f"Rank {rank:02d}  |  combined={score['combined']:.4f}  "
        f"sink={score['sink_contrast']:.2f}  "
        f"focus={score['saga_focus']:.2f}  "
        f"diff={score['reg_saga_diff']:.4f}\n"
        f"{filename}  [{synset}]",
        fontsize=7.2, color=NAVY, y=0.98,
    )

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
    parser.add_argument("--paths_e2",     required=True)
    parser.add_argument("--imagenet_tar", required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--n_sample",     type=int, default=500)
    parser.add_argument("--top_k",        type=int, default=30)
    parser.add_argument("--stage_base",   default="/tmp/fig1_inet_stage")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--synsets",      nargs="*", default=None,
        help="Optional synset IDs to filter (e.g. n01530575 n02084071). "
             "Default: all classes.")
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

    # ── Stage images ───────────────────────────────────────────────────────────
    stage_dir = Path(args.stage_base)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    img_paths = stage_imagenet_sample(
        args.imagenet_tar, stage_dir,
        args.n_sample, args.synsets, args.seed)
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
            # Extract synset from filename: ILSVRC2012_val_XXXXXXXX_nYYYYYYYY.JPEG
            stem   = Path(img_path).stem
            synset = stem.split("_")[-1] if "_" in stem else "unknown"
            sc["filename"] = Path(img_path).name
            sc["synset"]   = synset
            sc["path"]     = img_path
            records.append((sc, display, attn_b, attn_r, attn_s))

        except Exception as e:
            print(f"  WARNING: {Path(img_path).name}: {e}")

    # ── Sort and print ─────────────────────────────────────────────────────────
    records.sort(key=lambda x: x[0]["combined"], reverse=True)

    print(f"\n\nTop {args.top_k} candidates:")
    print(f"{'Rank':<6} {'combined':>10} {'sink':>8} {'focus':>8} "
          f"{'diff':>10}  synset       filename")
    print("-" * 85)
    for rank, (sc, *_) in enumerate(records[:args.top_k], start=1):
        print(f"{rank:<6} {sc['combined']:>10.4f} "
              f"{sc['sink_contrast']:>8.2f} "
              f"{sc['saga_focus']:>8.2f} "
              f"{sc['reg_saga_diff']:>10.4f}  "
              f"{sc['synset']:<14} {sc['filename']}")

    # ── Save top-K figures ─────────────────────────────────────────────────────
    figs_dir = out_dir / "candidates"
    figs_dir.mkdir(exist_ok=True)
    print(f"\nSaving top-{args.top_k} figures...")

    for rank, (sc, display, attn_b, attn_r, attn_s) in \
            enumerate(records[:args.top_k], start=1):
        out_path = figs_dir / \
            f"rank{rank:02d}_{sc['synset']}_{sc['filename'].replace('.JPEG','').replace('.jpg','')}.png"
        save_candidate_figure(
            display, attn_b, attn_r, attn_s,
            rank=rank, score=sc,
            filename=sc["filename"], synset=sc["synset"],
            out_path=out_path)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "scores_imagenet.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "filename", "synset", "combined",
            "sink_contrast", "saga_focus", "reg_saga_diff"])
        writer.writeheader()
        for rank, (sc, *_) in enumerate(records, start=1):
            writer.writerow({
                "rank":          rank,
                "filename":      sc["filename"],
                "synset":        sc["synset"],
                "combined":      sc["combined"],
                "sink_contrast": sc["sink_contrast"],
                "saga_focus":    sc["saga_focus"],
                "reg_saga_diff": sc["reg_saga_diff"],
            })
    print(f"Scores saved: {csv_path}")

    for ext in extractors.values():
        ext.remove()
    cleanup(str(stage_dir))

    print(f"\nDone. Candidate figures: {figs_dir}/")


if __name__ == "__main__":
    main()