"""
SAGA Paper-Style Attention Visualization
=========================================
Produces figures matching the "Vision Transformers Need Registers" style:
  - White background
  - Raw blocky 14×14 attention heatmap (no image overlay)
  - No sink score annotations
  - 5 columns × 4 rows

Rows:
  1. Original image
  2. Pretrained ViT   (timm pretrained=True, DeiT-III style — fully trained)
  3. Baseline ViT     (your V00, trained 100 epochs from scratch)
  4. SAGA             (your V02_B, trained 100 epochs from scratch)

Generates 4 figures (one per set of 5 images).

Usage (on TinyGPU via salloc):
  module load python/pytorch2.6py3.12
  module load cuda/12.4.1
  export PYTHONPATH=/home/hpc/iwi5/iwi5359h/my_repos/SAGA:$PYTHONPATH
  pip install --user matplotlib timm --quiet

  python tools/visualize_paper_style.py \
      --baseline_ckpt /home/vault/iwi5/iwi5359h/SAGA/checkpoints/e1_variants/V00_baseline/best.pth \
      --sagab_ckpt    /home/vault/iwi5/iwi5359h/SAGA/checkpoints/e1_variants/V02_B/best.pth \
      --val_tar       /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
      --out_dir       /home/woody/iwi5/iwi5359h/saga_paper_figures \
      --n_figures     4
"""

import argparse
import io
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
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


# ── Colormap — matches register paper (dark purple → yellow) ──────────────────
ATTN_CMAP = LinearSegmentedColormap.from_list(
    'attn', ['#08005a', '#1a0080', '#0050c8', '#00a0b0', '#40d060', '#ffff00']
)


# ── Attention extractor ────────────────────────────────────────────────────────
class AttentionExtractor:
    """
    Hooks last transformer block attention.
    Returns the CLS-to-patch attention averaged over all heads → 14×14 map.
    """
    def __init__(self, model):
        self.attn_maps = []
        self._hooks    = []
        attn_module    = model.blocks[-1].attn

        def hook(module, inp, out):
            x = inp[0]
            B, N, C = x.shape
            H, D = module.num_heads, module.head_dim
            with torch.no_grad():
                qkv = module.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                q = module.q_norm(q)
                k = module.k_norm(k)
                attn     = (q * module.scale) @ k.transpose(-2, -1)
                attn     = attn.softmax(dim=-1)          # [B, H, N, N]
                cls_attn = attn[:, :, 0, 1:]             # [B, H, n_patches]
                cls_attn = cls_attn.mean(1)              # [B, n_patches]  avg over heads
            self.attn_maps.append(cls_attn.detach().cpu())

        self._hooks.append(attn_module.register_forward_hook(hook))

    def clear(self):  self.attn_maps = []
    def remove(self): [h.remove() for h in self._hooks]

    def get_map(self):
        """Returns raw 14×14 numpy array — NOT upsampled, NOT normalised."""
        if not self.attn_maps:
            return np.zeros((14, 14))
        return self.attn_maps[-1][0].reshape(14, 14).numpy()


# ── Image loading ──────────────────────────────────────────────────────────────
def load_images_from_tar(val_tar, n=5, seed=42):
    """Read n random images directly from tar.gz into memory."""
    with tarfile.open(val_tar, 'r:gz') as tar:
        members = [m for m in tar.getmembers()
                   if m.name.endswith('.JPEG') and m.isfile()]
    random.seed(seed)
    chosen = random.sample(members, min(n, len(members)))

    display_tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])
    images = []
    with tarfile.open(val_tar, 'r:gz') as tar:
        for m in chosen:
            data   = tar.extractfile(m).read()
            img    = Image.open(io.BytesIO(data)).convert('RGB')
            synset = Path(m.name).stem.rsplit('_', 1)[-1]
            images.append((display_tfm(img), synset))
    return images


PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def get_attn_map(model, extractor, pil_img, device):
    extractor.clear()
    _ = model(PREPROCESS(pil_img).unsqueeze(0).to(device))
    return extractor.get_map()   # raw 14×14


# ── Figure drawing — paper style ───────────────────────────────────────────────
def make_paper_figure(images_data,
                      pretrained_maps,
                      baseline_maps,
                      saga_maps,
                      fig_idx,
                      out_dir):
    """
    White background, blocky 14×14 heatmap, no overlay, no sink score.
    Layout matches register paper style.
    """
    n_cols = len(images_data)
    n_rows = 4   # original / pretrained / baseline / SAGA

    # Tight layout — small square cells
    cell   = 1.6
    left_w = 1.5   # width reserved for row labels

    fig_w = left_w + n_cols * cell
    fig_h = n_rows * cell + 0.5   # small top margin for title

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.06,
        wspace=0.06,
        left=left_w / fig_w,
        right=0.99,
        top=1.0 - 0.3 / fig_h,
        bottom=0.02,
    )

    row_labels = ['Input', 'Pretrained ViT', 'Baseline ViT', 'SAGA']

    for col, (img_pil, synset) in enumerate(images_data):
        img_np = np.array(img_pil)

        # ── Row 0: original image ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_np)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── Row 1: pretrained ViT attention ───────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(pretrained_maps[col], cmap=ATTN_CMAP,
                  interpolation='nearest', aspect='equal',
                  vmin=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── Row 2: baseline ViT (100 epochs) ──────────────────────────────────
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(baseline_maps[col], cmap=ATTN_CMAP,
                  interpolation='nearest', aspect='equal',
                  vmin=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── Row 3: SAGA Gate B (100 epochs) ────────────────────────────────────
        ax = fig.add_subplot(gs[3, col])
        ax.imshow(saga_maps[col], cmap=ATTN_CMAP,
                  interpolation='nearest', aspect='equal',
                  vmin=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── Row labels on the left ─────────────────────────────────────────────────
    row_positions = [
        (3 * cell + 0.4 * cell) / fig_h,   # row 0 center (from bottom)
        (2 * cell + 0.4 * cell) / fig_h,
        (1 * cell + 0.4 * cell) / fig_h,
        (0 * cell + 0.4 * cell) / fig_h,
    ]
    # Simpler: use axes coordinates
    for row_idx, label in enumerate(row_labels):
        # Get the axes for this row, col 0
        ax_ref = fig.axes[row_idx * n_cols]
        # Place label to the left of the axes in figure coordinates
        bbox = ax_ref.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2

        fontsize  = 9
        fontstyle = 'italic' if label in ('Pretrained ViT', 'Baseline ViT') else 'normal'
        fontweight = 'bold' if label == 'SAGA' else 'normal'
        color     = '#000080' if label == 'SAGA' else 'black'

        fig.text(
            bbox.x0 - 0.015,
            y_center,
            label,
            ha='right', va='center',
            fontsize=fontsize,
            fontstyle=fontstyle,
            fontweight=fontweight,
            color=color,
        )

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png = out_dir / f'saga_attention_{fig_idx:02d}.png'
    pdf = out_dir / f'saga_attention_{fig_idx:02d}.pdf'

    fig.savefig(png, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf,           bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_ckpt', required=True,
                        help='V00_baseline best.pth')
    parser.add_argument('--sagab_ckpt',    required=True,
                        help='V02_B best.pth')
    parser.add_argument('--val_tar',       required=True,
                        help='val_images.tar.gz')
    parser.add_argument('--out_dir',       required=True)
    parser.add_argument('--n_figures',     type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Pretrained ViT (timm, pretrained on ImageNet-21k + fine-tuned) ─────
    # Using vit_base_patch16_224.augreg2_in21k_ft_in1k — same arch as your
    # baseline but fully pretrained. This is the "fully trained ViT" reference.
    import timm
    print("\nLoading pretrained ViT (timm)...")
    pretrained_vit = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
    ).to(device)
    pretrained_vit.eval()

    # Wrap in build_saga_vit style so AttentionExtractor works the same way
    # (timm's Attention module already has qkv, q_norm etc.)
    ext_pretrained = AttentionExtractor(pretrained_vit)
    print("  Pretrained ViT loaded.")

    # ── 2. Baseline ViT (your V00, 100 epochs from scratch) ───────────────────
    print("\nLoading baseline ViT (V00, 100 epochs)...")
    baseline = build_saga_vit(
        'vit_base_patch16_224', gate_terms=[], num_classes=1000
    ).to(device)
    ck = torch.load(args.baseline_ckpt, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in ck['model'].items()}
    baseline.load_state_dict(sd, strict=False)
    baseline.eval()
    ext_baseline = AttentionExtractor(baseline)
    print(f"  Loaded. top-1 = {ck.get('top1', '?')}")

    # ── 3. SAGA Gate B (your V02_B, 100 epochs) ───────────────────────────────
    print("\nLoading SAGA Gate B (V02_B, 100 epochs)...")
    saga = build_saga_vit(
        'vit_base_patch16_224', gate_terms=['B'], num_classes=1000
    ).to(device)
    ck2 = torch.load(args.sagab_ckpt, map_location=device)
    sd2 = {k.replace('module.', ''): v for k, v in ck2['model'].items()}
    saga.load_state_dict(sd2, strict=False)
    saga.eval()
    ext_saga = AttentionExtractor(saga)
    print(f"  Loaded. top-1 = {ck2.get('top1', '?')}")

    # ── Generate figures ───────────────────────────────────────────────────────
    print(f"\nGenerating {args.n_figures} figures...")

    for fig_idx in range(1, args.n_figures + 1):
        seed = fig_idx * 77
        print(f"\nFigure {fig_idx}/{args.n_figures}  (seed={seed})")

        images = load_images_from_tar(args.val_tar, n=5, seed=seed)
        print(f"  Classes: {[s for _, s in images]}")

        pretrained_maps = []
        baseline_maps   = []
        saga_maps       = []

        for img_pil, synset in images:
            mp = get_attn_map(pretrained_vit, ext_pretrained, img_pil, device)
            mb = get_attn_map(baseline,       ext_baseline,   img_pil, device)
            ms = get_attn_map(saga,           ext_saga,       img_pil, device)
            pretrained_maps.append(mp)
            baseline_maps.append(mb)
            saga_maps.append(ms)
            print(f"    {synset:15s}  pretrained_max={mp.max():.3f}  "
                  f"baseline_max={mb.max():.3f}  saga_max={ms.max():.3f}")

        make_paper_figure(
            images, pretrained_maps, baseline_maps, saga_maps,
            fig_idx, args.out_dir
        )

    ext_pretrained.remove()
    ext_baseline.remove()
    ext_saga.remove()

    print(f"\nDone. {args.n_figures} figures in: {args.out_dir}")


if __name__ == '__main__':
    main()