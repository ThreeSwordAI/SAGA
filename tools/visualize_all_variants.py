"""
SAGA All-Variants Attention Map Visualization
=============================================
Generates 12 figures total:
  - 3 sets of 5 images × 4 figures per set = 12 figures
  Each figure has:
    Row 1: 5 original images
    Row 2: Baseline ViT
    Row 3-6: Gate variants (grouping depends on figure number)

  Figure layout per set of 5 images:
    Fig A: [baseline, A, B, C, D]
    Fig B: [baseline, AB, AC, AD, BC]
    Fig C: [baseline, BD, CD, ABC, ABD]
    Fig D: [baseline, ACD, BCD, ABCD]  ← only 5 rows (4 variants + baseline)

Row label color:
  GREEN  if variant best_top1 > V00 baseline (77.47%)
  RED    if variant best_top1 < V00 baseline
  WHITE  for baseline row and original images row

Reads images directly from val_images.tar.gz — no extraction.

Usage:
  python tools/visualize_all_variants.py \
      --ckpt_dir   /home/vault/iwi5/iwi5359h/SAGA/checkpoints/e1_variants \
      --val_tar    /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
      --out_dir    /home/woody/iwi5/iwi5359h/saga_figures \
      --n_sets     3
"""

import argparse
import io
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

# ── E1 wave 1 top-1 results ───────────────────────────────────────────────────
# Used to colour row labels green/red vs baseline
BASELINE_TOP1 = 77.47

VARIANT_TOP1 = {
    'V00_baseline':  77.47,
    'V01_A':         76.62,
    'V02_B':         77.94,
    'V03_C':         77.30,
    'V04_D':         77.56,
    'V05_AB':        77.03,
    'V06_AC':        76.76,
    'V07_AD':        76.77,
    'V08_BC':        77.20,
    'V09_BD':        77.20,
    'V10_CD':        77.56,
    'V11_ABC':       77.72,
    'V12_ABD':       76.89,
    'V13_ACD':       76.54,
    'V14_BCD':       77.39,
    'V15_ABCD_full': 76.84,
}

# ── 4 figure groupings — each list = [baseline + 4 or 5 variants] ─────────────
FIGURE_GROUPS = [
    # Fig 1: singles
    ['V00_baseline', 'V01_A', 'V02_B', 'V03_C', 'V04_D'],
    # Fig 2: pairs (first 4)
    ['V00_baseline', 'V05_AB', 'V06_AC', 'V07_AD', 'V08_BC'],
    # Fig 3: pairs (last 2) + triples (first 2)
    ['V00_baseline', 'V09_BD', 'V10_CD', 'V11_ABC', 'V12_ABD'],
    # Fig 4: triples (last 2) + full
    ['V00_baseline', 'V13_ACD', 'V14_BCD', 'V15_ABCD_full'],
]

# Readable row labels
VARIANT_LABEL = {
    'V00_baseline':  'Baseline ViT',
    'V01_A':         'Gate A',
    'V02_B':         'Gate B',
    'V03_C':         'Gate C',
    'V04_D':         'Gate D',
    'V05_AB':        'Gate A+B',
    'V06_AC':        'Gate A+C',
    'V07_AD':        'Gate A+D',
    'V08_BC':        'Gate B+C',
    'V09_BD':        'Gate B+D',
    'V10_CD':        'Gate C+D',
    'V11_ABC':       'Gate A+B+C',
    'V12_ABD':       'Gate A+B+D',
    'V13_ACD':       'Gate A+C+D',
    'V14_BCD':       'Gate B+C+D',
    'V15_ABCD_full': 'Gate A+B+C+D',
}

GATE_TERMS = {
    'V00_baseline':  [],
    'V01_A':         ['A'],
    'V02_B':         ['B'],
    'V03_C':         ['C'],
    'V04_D':         ['D'],
    'V05_AB':        ['A', 'B'],
    'V06_AC':        ['A', 'C'],
    'V07_AD':        ['A', 'D'],
    'V08_BC':        ['B', 'C'],
    'V09_BD':        ['B', 'D'],
    'V10_CD':        ['C', 'D'],
    'V11_ABC':       ['A', 'B', 'C'],
    'V12_ABD':       ['A', 'B', 'D'],
    'V13_ACD':       ['A', 'C', 'D'],
    'V14_BCD':       ['B', 'C', 'D'],
    'V15_ABCD_full': ['A', 'B', 'C', 'D'],
}


# ── Attention extractor ────────────────────────────────────────────────────────
class AttentionExtractor:
    def __init__(self, model):
        self.attn_maps = []
        self._hooks    = []
        attn_module    = model.blocks[-1].attn

        def hook(module, inp, out):
            x = inp[0]
            B, N, C = x.shape
            H, D = module.num_heads, module.head_dim
            with torch.no_grad():
                qkv  = module.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                q = module.q_norm(q)
                k = module.k_norm(k)
                attn = (q * module.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                cls_attn = attn[:, :, 0, 1:].mean(1)   # [B, n_patches]
            self.attn_maps.append(cls_attn.detach().cpu())

        self._hooks.append(attn_module.register_forward_hook(hook))

    def clear(self):  self.attn_maps = []
    def remove(self): [h.remove() for h in self._hooks]

    def get_map(self):
        if not self.attn_maps: return np.zeros((14, 14))
        return self.attn_maps[-1][0].reshape(14, 14).numpy()


# ── Image loading from tar.gz ──────────────────────────────────────────────────
def load_images_from_tar(val_tar, n=5, seed=42):
    with tarfile.open(val_tar, 'r:gz') as tar:
        members = [m for m in tar.getmembers()
                   if m.name.endswith('.JPEG') and m.isfile()]
    random.seed(seed)
    chosen = random.sample(members, min(n, len(members)))

    display_tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224)
    ])
    images = []
    with tarfile.open(val_tar, 'r:gz') as tar:
        for m in chosen:
            data   = tar.extractfile(m).read()
            img    = Image.open(io.BytesIO(data)).convert('RGB')
            synset = Path(m.name).stem.rsplit('_', 1)[-1]
            images.append((display_tfm(img), synset))
    return images


TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


@torch.no_grad()
def get_attn(model, extractor, pil_img, device):
    extractor.clear()
    _ = model(TRANSFORM(pil_img).unsqueeze(0).to(device))
    return extractor.get_map()


def norm(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def upsample(a):
    t = torch.tensor(a).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (224, 224), mode='bilinear',
                         align_corners=False)[0, 0].numpy()


# ── Label colour ───────────────────────────────────────────────────────────────
def row_color(variant_name):
    """Green if better than baseline, red if worse, white for baseline."""
    if variant_name == 'V00_baseline':
        return '#dddddd'
    top1 = VARIANT_TOP1.get(variant_name, BASELINE_TOP1)
    return '#55ee55' if top1 > BASELINE_TOP1 else '#ee5555'


# ── Figure drawing ─────────────────────────────────────────────────────────────
def make_figure(images_data, attn_maps_by_variant,
                variant_names, fig_label, out_dir):
    """
    images_data        : list of (PIL, synset)   length n_images
    attn_maps_by_variant: dict variant_name → list of 14×14 arrays
    variant_names      : ordered list of variant names (first = baseline)
    fig_label          : string used in filename
    """
    n_cols = len(images_data)
    n_rows = 1 + len(variant_names)   # original + variants

    fig_h  = n_rows * 2.4 + 1.0
    fig    = plt.figure(figsize=(n_cols * 3.0, fig_h), facecolor='#0d0d0d')

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.06, wspace=0.04,
        left=0.14, right=0.98,
        top=1.0 - 0.5/fig_h,
        bottom=0.5/fig_h,
    )

    for col, (img_pil, synset) in enumerate(images_data):
        img_np = np.array(img_pil)

        # ── Row 0: original ────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_np)
        ax.axis('off')
        if col == 0:
            ax.text(-0.18, 0.5, 'Original',
                    transform=ax.transAxes,
                    color='#dddddd', fontsize=10.5, fontweight='bold',
                    va='center', ha='center', rotation=90)
        ax.set_title(synset, color='#888888', fontsize=7.5, pad=2)

        # ── Rows 1+: variants ──────────────────────────────────────────────────
        for row_idx, vname in enumerate(variant_names):
            row = row_idx + 1
            ax  = fig.add_subplot(gs[row, col])
            am  = norm(attn_maps_by_variant[vname][col])
            ax.imshow(img_np)
            ax.imshow(upsample(am), cmap=ATTN_CMAP, alpha=0.60)
            ax.axis('off')

            if col == 0:
                top1  = VARIANT_TOP1.get(vname, BASELINE_TOP1)
                label = VARIANT_LABEL[vname]
                color = row_color(vname)
                # Show top-1 in parentheses
                full_label = f"{label}\n({top1:.2f}%)"
                ax.text(-0.18, 0.5, full_label,
                        transform=ax.transAxes,
                        color=color, fontsize=9.5, fontweight='bold',
                        va='center', ha='center', rotation=90,
                        linespacing=1.4)

    # Title
    fig.text(0.56, 0.995,
             f'CLS Attention Maps — {fig_label}',
             ha='center', va='top', color='white',
             fontsize=12, fontweight='bold')
    fig.text(0.56, 0.975,
             'Green label = better than baseline   |   Red label = worse',
             ha='center', va='top', color='#888888', fontsize=8.5)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f'variants_{fig_label}'
    fig.savefig(f'{fname}.png', dpi=140, bbox_inches='tight',
                facecolor='#0d0d0d')
    fig.savefig(f'{fname}.pdf', bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close(fig)
    print(f"  Saved: {fname}.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True,
                        help='e.g. /home/vault/.../checkpoints/e1_variants')
    parser.add_argument('--val_tar',  required=True,
                        help='val_images.tar.gz on janus')
    parser.add_argument('--out_dir',  required=True)
    parser.add_argument('--n_sets',   type=int, default=3,
                        help='Number of image sets (3 → 12 figures)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load all 16 models ─────────────────────────────────────────────────────
    print("\nLoading 16 models...")
    models     = {}
    extractors = {}

    for vname, terms in GATE_TERMS.items():
        ckpt_path = Path(args.ckpt_dir) / vname / 'best.pth'
        if not ckpt_path.exists():
            print(f"  SKIP {vname} — checkpoint not found at {ckpt_path}")
            continue

        m = build_saga_vit('vit_base_patch16_224',
                           gate_terms=terms, num_classes=1000).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        sd = {k.replace('module.', ''): v for k, v in ck['model'].items()}
        m.load_state_dict(sd, strict=False)
        m.eval()
        models[vname]     = m
        extractors[vname] = AttentionExtractor(m)
        print(f"  Loaded {vname:20s}  top-1={ck.get('top1','?')}")

    available = list(models.keys())
    print(f"\n{len(available)} models loaded.")

    # ── Generate figures ───────────────────────────────────────────────────────
    for set_idx in range(1, args.n_sets + 1):
        seed = set_idx * 100
        print(f"\n=== Image set {set_idx}/{args.n_sets}  (seed={seed}) ===")
        images = load_images_from_tar(args.val_tar, n=5, seed=seed)
        print(f"  Images: {[s for _, s in images]}")

        for fig_num, group in enumerate(FIGURE_GROUPS, start=1):
            # Filter to only variants we have checkpoints for
            group_avail = [v for v in group if v in models]
            if len(group_avail) < 2:
                print(f"  Fig {fig_num}: skipping — not enough variants loaded")
                continue

            print(f"  Fig {fig_num}: {[VARIANT_LABEL[v] for v in group_avail]}")

            # Compute attention maps for this image set
            attn_maps = {}
            for vname in group_avail:
                attn_maps[vname] = []
                for img_pil, _ in images:
                    am = get_attn(models[vname], extractors[vname],
                                  img_pil, device)
                    attn_maps[vname].append(am)

            fig_label = f'set{set_idx:02d}_fig{fig_num}'
            make_figure(images, attn_maps, group_avail, fig_label, args.out_dir)

    # Cleanup hooks
    for ext in extractors.values():
        ext.remove()

    print(f"\nAll done. Figures saved to: {args.out_dir}")
    print(f"Total figures: {args.n_sets * len(FIGURE_GROUPS)} "
          f"({args.n_sets} sets × {len(FIGURE_GROUPS)} figures)")


if __name__ == '__main__':
    main()