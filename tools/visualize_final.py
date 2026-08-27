"""
SAGA Final Visualization — Two Versions
=========================================
Version 1: Paper style  (white bg, blocky 14×14, no overlay)
Version 2: Overlay style (dark bg, smooth heatmap over image)

4 rows × 5 columns:
  Row 1: Original image
  Row 2: Pretrained ViT  (timm pretrained=True)
  Row 3: Baseline ViT    (V00, 100 epochs)
  Row 4: SAGA            (V02_B, 100 epochs)

Images chosen from specific synsets:
  n02666196, n03697007, n02488291, n02104365, n03920288, n02441942
  (falls back to random if not found)

No sink score. Labels fixed in left column only.

Usage:
  python tools/visualize_final.py \
      --baseline_ckpt /home/vault/iwi5/iwi5359h/SAGA/checkpoints/e1_variants/V00_baseline/best.pth \
      --sagab_ckpt    /home/vault/iwi5/iwi5359h/SAGA/checkpoints/e1_variants/V02_B/best.pth \
      --val_tar       /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
      --out_dir       /home/woody/iwi5/iwi5359h/saga_paper_figures
"""

import argparse, io, random, tarfile, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torchvision import transforms
import timm

sys.path.insert(0, '/home/hpc/iwi5/iwi5359h/my_repos/SAGA')
from saga_old import build_saga_vit

# ── Colourmaps ─────────────────────────────────────────────────────────────────
CMAP_PAPER   = LinearSegmentedColormap.from_list(
    'paper', ['#08005a','#1a0080','#0050c8','#00a0b0','#40d060','#ffff00'])
CMAP_OVERLAY = LinearSegmentedColormap.from_list(
    'overlay', ['#0a0030','#1a0060','#0060c0','#00b0c0','#60e060','#ffff00'])

# ── Target synsets ──────────────────────────────────────────────────────────────
TARGET_SYNSETS = ['n02666196','n03697007','n02488291',
                  'n02104365','n03920288','n02441942']

# ── Attention extractor ─────────────────────────────────────────────────────────
class AttentionExtractor:
    def __init__(self, model):
        self.maps  = []
        self._hooks = []
        attn = model.blocks[-1].attn
        def hook(mod, inp, out):
            x = inp[0]; B,N,C = x.shape
            H,D = mod.num_heads, mod.head_dim
            with torch.no_grad():
                qkv = mod.qkv(x).reshape(B,N,3,H,D).permute(2,0,3,1,4)
                q,k,_ = qkv.unbind(0)
                q = mod.q_norm(q); k = mod.k_norm(k)
                a = (q*mod.scale)@k.transpose(-2,-1)
                a = a.softmax(-1)
                self.maps.append(a[:,:,0,1:].mean(1).detach().cpu())
        self._hooks.append(attn.register_forward_hook(hook))
    def clear(self):  self.maps = []
    def remove(self): [h.remove() for h in self._hooks]
    def get(self):
        if not self.maps: return np.zeros((14,14))
        return self.maps[-1][0].reshape(14,14).numpy()

# ── Load images from tar ────────────────────────────────────────────────────────
def load_target_images(val_tar, synsets, n=5):
    """
    Try to load one image per synset from the list.
    Falls back to random if synset not found.
    """
    print(f"  Scanning tar for synsets: {synsets}")
    found = {}   # synset → member
    with tarfile.open(val_tar, 'r:gz') as tar:
        members = [m for m in tar.getmembers()
                   if m.name.endswith('.JPEG') and m.isfile()]
        # Group by synset (filename format: ILSVRC2012_val_XXXXXXXX_nXXXXXXXX.JPEG
        # OR nXXXXXXXX_XXXXX.JPEG)
        for m in members:
            stem = Path(m.name).stem
            # Try both naming conventions
            parts = stem.split('_')
            for p in parts:
                if p in synsets and p not in found:
                    found[p] = m
                    break
        # Fill remaining with random
        missing = [s for s in synsets if s not in found]
        if missing:
            print(f"  Synsets not found: {missing} — using random")
            pool = [m for m in members
                    if Path(m.name).stem.split('_')[-1] not in found
                    and m not in found.values()]
            random.seed(99)
            extra = random.sample(pool, min(len(missing), len(pool)))
            for m in extra[:len(missing)]:
                stem = Path(m.name).stem
                syn  = stem.rsplit('_',1)[-1]
                found[syn] = m

    # Load in synset order (then random extras at end)
    order = [s for s in synsets if s in found]
    order += [s for s in found if s not in synsets]
    order = order[:n]

    display_tfm = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224)])
    images = []
    with tarfile.open(val_tar, 'r:gz') as tar:
        for syn in order:
            m   = found[syn]
            img = Image.open(io.BytesIO(tar.extractfile(m).read())).convert('RGB')
            images.append((display_tfm(img), syn))
            print(f"    Loaded {syn}")
    return images

PREPROCESS = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@torch.no_grad()
def get_map(model, ext, pil_img, device):
    ext.clear()
    model(PREPROCESS(pil_img).unsqueeze(0).to(device))
    return ext.get()

def norm(a):
    lo,hi = a.min(), a.max()
    return (a-lo)/(hi-lo+1e-8)

def upsample(a, size=224):
    t = torch.tensor(a).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t,(size,size),mode='bilinear',
                         align_corners=False)[0,0].numpy()

# ── Figure builder ──────────────────────────────────────────────────────────────
ROW_LABELS  = ['Input', 'Pretrained\nViT', 'Baseline\nViT', 'SAGA']
ROW_COLORS  = ['black', 'black',           'black',          '#000099']
ROW_WEIGHTS = ['normal','normal',          'normal',          'bold']

def _draw_labels(fig, axes_grid, n_rows, n_cols):
    """
    Draw row labels using fig.text() positioned at the vertical centre
    of each row — measured from axes bboxes AFTER layout is finalised.
    This avoids the diagonal label bug.
    """
    fig.canvas.draw()   # force layout so get_position() is accurate
    for row in range(n_rows):
        # collect all axes in this row
        row_axes = [axes_grid[row][col] for col in range(n_cols)]
        bboxes   = [ax.get_position() for ax in row_axes]
        y_center = (bboxes[0].y0 + bboxes[0].y1) / 2
        x_left   = bboxes[0].x0 - 0.01   # just left of first column

        fig.text(x_left, y_center,
                 ROW_LABELS[row],
                 ha='right', va='center',
                 fontsize=10,
                 color=ROW_COLORS[row],
                 fontweight=ROW_WEIGHTS[row],
                 transform=fig.transFigure)


def make_paper_figure(images, pre_maps, base_maps, saga_maps, out_dir, suffix):
    """
    White background, blocky 14×14, no overlay.
    """
    n      = len(images)
    cell   = 1.55
    lpad   = 1.4     # left padding for labels

    fig_w  = lpad + n * cell + 0.1
    fig_h  = 4 * cell + 0.15

    fig, axes = plt.subplots(4, n,
        figsize=(fig_w, fig_h),
        facecolor='white',
        gridspec_kw=dict(
            hspace=0.06, wspace=0.06,
            left=lpad/fig_w, right=0.99,
            top=0.99, bottom=0.01))

    all_maps = [None, pre_maps, base_maps, saga_maps]

    for col, (img_pil, synset) in enumerate(images):
        img_np = np.array(img_pil)

        # Row 0 — original
        ax = axes[0, col]
        ax.imshow(img_np)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

        # Rows 1-3 — attention maps
        for row, maps in enumerate(all_maps[1:], start=1):
            ax = axes[row, col]
            ax.imshow(maps[col], cmap=CMAP_PAPER,
                      interpolation='nearest', aspect='equal', vmin=0)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

    _draw_labels(fig, axes, 4, n)

    _save(fig, out_dir, f'paper_style_{suffix}')


def make_overlay_figure(images, pre_maps, base_maps, saga_maps, out_dir, suffix):
    """
    Dark background, smooth upsampled heatmap over original image.
    """
    n      = len(images)
    cell   = 2.0
    lpad   = 1.6

    fig_w  = lpad + n * cell + 0.1
    fig_h  = 4 * cell + 0.15

    fig, axes = plt.subplots(4, n,
        figsize=(fig_w, fig_h),
        facecolor='#0d0d0d',
        gridspec_kw=dict(
            hspace=0.05, wspace=0.04,
            left=lpad/fig_w, right=0.99,
            top=0.99, bottom=0.01))

    overlay_colors = ['white','#7ec8e3','#aaaaaa','#f5c842']
    all_maps = [None, pre_maps, base_maps, saga_maps]

    for col, (img_pil, synset) in enumerate(images):
        img_np = np.array(img_pil)

        # Row 0 — original
        ax = axes[0, col]
        ax.imshow(img_np)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

        # Rows 1-3 — overlay
        for row, maps in enumerate(all_maps[1:], start=1):
            ax = axes[row, col]
            am = upsample(norm(maps[col]))
            ax.imshow(img_np)
            ax.imshow(am, cmap=CMAP_OVERLAY, alpha=0.62)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

    _draw_labels_dark(fig, axes, 4, n, overlay_colors)
    _save(fig, out_dir, f'overlay_style_{suffix}')


def _draw_labels_dark(fig, axes, n_rows, n_cols, colors):
    fig.canvas.draw()
    for row in range(n_rows):
        row_axes = [axes[row][col] for col in range(n_cols)]
        bboxes   = [ax.get_position() for ax in row_axes]
        y_center = (bboxes[0].y0 + bboxes[0].y1) / 2
        x_left   = bboxes[0].x0 - 0.01
        fig.text(x_left, y_center,
                 ROW_LABELS[row],
                 ha='right', va='center',
                 fontsize=10.5,
                 color=colors[row],
                 fontweight=ROW_WEIGHTS[row],
                 transform=fig.transFigure)

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

def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fc = fig.get_facecolor()
    fig.savefig(out_dir/f'{name}.png', dpi=200,
                bbox_inches='tight', facecolor=fc)
    fig.savefig(out_dir/f'{name}.pdf',
                bbox_inches='tight', facecolor=fc)
    plt.close(fig)
    print(f"  Saved: {out_dir/name}.png")
    print(f"  Saved: {out_dir/name}.pdf")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline_ckpt', required=True)
    p.add_argument('--sagab_ckpt',    required=True)
    p.add_argument('--val_tar',       required=True)
    p.add_argument('--out_dir',       required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading Pretrained ViT (timm)...")
    pre_vit = timm.create_model('vit_base_patch16_224',
                                pretrained=True, num_classes=1000).to(device)
    pre_vit.eval()
    ext_pre = AttentionExtractor(pre_vit)

    print("Loading Baseline ViT (V00)...")
    base = build_saga_vit('vit_base_patch16_224',
                          gate_terms=[], num_classes=1000).to(device)
    ck = torch.load(args.baseline_ckpt, map_location=device)
    base.load_state_dict(
        {k.replace('module.',''):v for k,v in ck['model'].items()}, strict=False)
    base.eval()
    ext_base = AttentionExtractor(base)
    print(f"  top-1 = {ck.get('top1','?')}")

    print("Loading SAGA Gate B (V02_B)...")
    saga = build_saga_vit('vit_base_patch16_224',
                          gate_terms=['B'], num_classes=1000).to(device)
    ck2 = torch.load(args.sagab_ckpt, map_location=device)
    saga.load_state_dict(
        {k.replace('module.',''):v for k,v in ck2['model'].items()}, strict=False)
    saga.eval()
    ext_saga = AttentionExtractor(saga)
    print(f"  top-1 = {ck2.get('top1','?')}")

    # ── Load images ────────────────────────────────────────────────────────────
    print("\nLoading images...")
    images = load_images_from_tar(val_tar_path=args.val_tar, n=5, seed=42)

    # ── Compute attention maps ─────────────────────────────────────────────────
    print("\nComputing attention maps...")
    pre_maps, base_maps, saga_maps = [], [], []
    for img_pil, syn in images:
        pre_maps.append(get_map(pre_vit, ext_pre,  img_pil, device))
        base_maps.append(get_map(base,   ext_base, img_pil, device))
        saga_maps.append(get_map(saga,   ext_saga, img_pil, device))
        print(f"  {syn} done")

    # ── Generate both figures ──────────────────────────────────────────────────
    print("\nGenerating paper-style figure...")
    make_paper_figure(images, pre_maps, base_maps, saga_maps,
                      args.out_dir, 'final')

    print("\nGenerating overlay-style figure...")
    make_overlay_figure(images, pre_maps, base_maps, saga_maps,
                        args.out_dir, 'final')

    ext_pre.remove(); ext_base.remove(); ext_saga.remove()
    print(f"\nDone. Both figures saved to: {args.out_dir}")


if __name__ == '__main__':
    main()