"""
SAGA Attention Map Visualization
=================================
Reads images DIRECTLY from the val_images.tar.gz — no extraction needed.
Zero files written to scratch.

Usage (on TinyGPU via salloc):
  salloc --gres=gpu:1 --cpus-per-task=4 --time=02:00:00
  module load python/pytorch2.6py3.12
  module load cuda/12.4.1
  export PYTHONPATH=/home/hpc/iwi5/iwi5359h/my_repos/SAGA:$PYTHONPATH
  pip install --user matplotlib --quiet

  python tools/visualize_attention.py \
      --baseline_ckpt /home/vault/iwi5/iwi5359h/SAGA/checkpoints/e1_variants/V00_baseline/best.pth \
      --sagab_ckpt    /home/vault/iwi5/iwi5359h/SAGA/checkpoints/e1_variants/V02_B/best.pth \
      --val_tar       /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
      --out_dir       /home/woody/iwi5/iwi5359h/saga_figures \
      --n_figures     10 \
      --n_images      5
"""

import argparse
import io
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, '/home/hpc/iwi5/iwi5359h/my_repos/SAGA')
from saga_old import build_saga_vit


# ── Colormap ───────────────────────────────────────────────────────────────────
ATTN_CMAP = LinearSegmentedColormap.from_list(
    'attn', ['#0a0030', '#1a0060', '#0060c0', '#00b0c0', '#60e060', '#ffff00']
)


# ── Attention extractor ────────────────────────────────────────────────────────

class AttentionExtractor:
    def __init__(self, model):
        self.model     = model
        self.attn_maps = []
        self._hooks    = []
        attn_module    = model.blocks[-1].attn

        def attn_hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            H, D = module.num_heads, module.head_dim
            with torch.no_grad():
                qkv = module.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                q = module.q_norm(q)
                k = module.k_norm(k)
                attn = (q * module.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)          # [B, H, N, N]
                cls_attn = attn[:, :, 0, 1:]         # [B, H, n_patches]
                cls_attn = cls_attn.mean(dim=1)      # [B, n_patches]
            self.attn_maps.append(cls_attn.detach().cpu())

        self._hooks.append(attn_module.register_forward_hook(attn_hook))

    def clear(self):
        self.attn_maps = []

    def remove(self):
        for h in self._hooks:
            h.remove()

    def get_map(self):
        if not self.attn_maps:
            return np.zeros((14, 14))
        return self.attn_maps[-1][0].reshape(14, 14).numpy()


# ── Read images directly from tar.gz ──────────────────────────────────────────

def load_images_from_tar(val_tar_path, n=5, seed=42):
    """
    Reads n random JPEG images directly from val_images.tar.gz.
    No extraction — reads into memory only.
    Returns list of (PIL.Image, synset_name).
    HuggingFace naming: {image_id}_{synset}.JPEG — synset is after last underscore.
    """
    print(f"  Scanning tar index from {val_tar_path} ...")
    with tarfile.open(val_tar_path, 'r:gz') as tar:
        members = [m for m in tar.getmembers()
                   if m.name.endswith('.JPEG') and m.isfile()]

    random.seed(seed)
    chosen = random.sample(members, min(n, len(members)))

    print(f"  Loading {len(chosen)} images from tar (no extraction)...")
    images = []
    display_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    with tarfile.open(val_tar_path, 'r:gz') as tar:
        for member in chosen:
            f      = tar.extractfile(member)
            data   = f.read()
            img    = Image.open(io.BytesIO(data)).convert('RGB')
            img_d  = display_transform(img)

            # synset is the part after the last underscore, before .JPEG
            basename = Path(member.name).stem          # e.g. n01440764_10026
            synset   = basename.rsplit('_', 1)[-1]     # e.g. n01440764

            images.append((img_d, synset))

    return images


# ── Preprocessing ──────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def img_to_tensor(pil_img):
    return TRANSFORM(pil_img).unsqueeze(0)


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_attn(model, extractor, pil_img, device):
    extractor.clear()
    t = img_to_tensor(pil_img).to(device)
    _ = model(t)
    return extractor.get_map()


def norm(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def upsample(a, size=224):
    t = torch.tensor(a).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, (size, size), mode='bilinear', align_corners=False)
    return t[0, 0].numpy()


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(images_data, baseline_maps, sagab_maps, fig_idx, out_dir):
    n = len(images_data)
    fig = plt.figure(figsize=(n * 3.2, 10), facecolor='#0a0a0a')

    gs = gridspec.GridSpec(3, n, figure=fig,
                           hspace=0.08, wspace=0.04,
                           left=0.10, right=0.98,
                           top=0.92, bottom=0.02)

    row_labels = ['Original', 'Baseline ViT', 'Gate B (SAGA)']
    row_colors = ['#ffffff',  '#7ec8e3',       '#f0c060']

    for col, (img_pil, synset) in enumerate(images_data):
        img_np = np.array(img_pil)

        # Row 0 — original
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(img_np)
        ax0.axis('off')
        ax0.set_title(synset, color='#aaaaaa', fontsize=8, pad=3)

        # Row 1 — baseline
        ax1 = fig.add_subplot(gs[1, col])
        ab   = norm(baseline_maps[col])
        ax1.imshow(img_np)
        ax1.imshow(upsample(ab), cmap=ATTN_CMAP, alpha=0.65)
        ax1.axis('off')
        sink_b = float((ab > 0.85).mean() * 100)
        ax1.text(0.97, 0.03, f'sink {sink_b:.1f}%',
                 transform=ax1.transAxes, color='#ff6666', fontsize=8,
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#000000aa'))

        # Row 2 — gate B
        ax2 = fig.add_subplot(gs[2, col])
        ag   = norm(sagab_maps[col])
        ax2.imshow(img_np)
        ax2.imshow(upsample(ag), cmap=ATTN_CMAP, alpha=0.65)
        ax2.axis('off')
        sink_g = float((ag > 0.85).mean() * 100)
        ax2.text(0.97, 0.03, f'sink {sink_g:.1f}%',
                 transform=ax2.transAxes, color='#66ff66', fontsize=8,
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#000000aa'))

    # Row labels
    for i, (label, color) in enumerate(zip(row_labels, row_colors)):
        fig.text(0.005, 0.78 - i * 0.30, label,
                 ha='left', va='center', color=color,
                 fontsize=12, fontweight='bold', rotation=90)

    # Title
    fig.text(0.5, 0.96,
             'CLS Attention Maps — Baseline ViT vs SAGA Gate B',
             ha='center', color='white', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.935,
             'Yellow = high attention   |   Dark = low attention   |'
             '   sink % = fraction of high-attention patches',
             ha='center', color='#888888', fontsize=9)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f'attention_comparison_{fig_idx:02d}.png'
    pdf = out_dir / f'attention_comparison_{fig_idx:02d}.pdf'
    fig.savefig(png, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    fig.savefig(pdf,           bbox_inches='tight', facecolor='#0a0a0a')
    plt.close(fig)
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', required=True)
    parser.add_argument('--sagab_ckpt',    required=True)
    parser.add_argument('--val_tar',       required=True,
                        help='Path to val_images.tar.gz on janus')
    parser.add_argument('--out_dir',       required=True)
    parser.add_argument('--n_figures',     type=int, default=10)
    parser.add_argument('--n_images',      type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    print("\nLoading baseline (V00)...")
    baseline = build_saga_vit('vit_base_patch16_224',
                               gate_terms=[], num_classes=1000).to(device)
    ck = torch.load(args.baseline_ckpt, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in ck['model'].items()}
    baseline.load_state_dict(sd, strict=False)
    baseline.eval()
    print(f"  top-1 = {ck.get('top1', '?')}")

    print("\nLoading Gate B (V02_B)...")
    sagab = build_saga_vit('vit_base_patch16_224',
                            gate_terms=['B'], num_classes=1000).to(device)
    ck2 = torch.load(args.sagab_ckpt, map_location=device)
    sd2 = {k.replace('module.', ''): v for k, v in ck2['model'].items()}
    sagab.load_state_dict(sd2, strict=False)
    sagab.eval()
    print(f"  top-1 = {ck2.get('top1', '?')}")

    ext_b = AttentionExtractor(baseline)
    ext_g = AttentionExtractor(sagab)

    print(f"\nGenerating {args.n_figures} figures...")

    for fig_idx in range(1, args.n_figures + 1):
        print(f"\nFigure {fig_idx}/{args.n_figures}")
        images = load_images_from_tar(
            args.val_tar,
            n=args.n_images,
            seed=fig_idx * 42
        )

        baseline_maps, sagab_maps = [], []
        for img_pil, synset in images:
            ab = get_attn(baseline, ext_b, img_pil, device)
            ag = get_attn(sagab,    ext_g, img_pil, device)
            baseline_maps.append(ab)
            sagab_maps.append(ag)
            nb = float((norm(ab) > 0.85).mean() * 100)
            ng = float((norm(ag) > 0.85).mean() * 100)
            print(f"    {synset:12s}  baseline_sink={nb:.1f}%  gate_b_sink={ng:.1f}%")

        make_figure(images, baseline_maps, sagab_maps, fig_idx, args.out_dir)

    ext_b.remove()
    ext_g.remove()
    print(f"\nAll done. Figures in: {args.out_dir}")


if __name__ == '__main__':
    main()