#!/usr/bin/env python3
"""
figures/fig1_teaser_final.py
============================

Final Figure 1 teaser:
    3 rows × 4 columns
        Input | ViT | ViT+Registers | SAGA (ours)

Current locked rows:
    1) VOC       : 003796.jpg
    2) COCO      : 000000055950.jpg
    3) ImageNet  : ILSVRC2012_val_00002425_n01530575.JPEG

Visual design:
    - Shared row-wise normalization (scientifically fair)
    - Subtle slate-blue veil under attention overlays
    - Teal attention heatmap
    - Amber border for SAGA column
    - SAGA title: black bold "SAGA" + teal italic "(ours)"
    - Saves PNG and PDF

Usage:
    python3 figures/fig1_teaser_final.py \
        --paths_e2      /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --imagenet_tar  /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
        --coco_zip      /home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip \
        --voc_tar       /home/woody/iwi5/iwi5359h/Data/VOC/VOCtest_06-Nov-2007.tar \
        --out_dir       /home/woody/iwi5/iwi5359h/saga_figures/final \
        --basename      fig1_teaser_attention \
        --stage_base    /tmp/fig1_teaser_stage_${USER}

Notes:
    - If 003796.jpg is not found inside the provided VOC tar, the script
      automatically tries the sibling file VOCtrainval_06-Nov-2007.tar.
    - If you want to change the blue veil strength:
          BLUE_VEIL_ALPHA
    - If you want to change the veil color:
          VEIL_BLUE
"""

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from PIL import Image
import torchvision.transforms.functional as TF
import timm

# Add repo root for `from saga import build_saga_vit`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga import build_saga_vit


# ──────────────────────────────────────────────────────────────────────────────
# Locked image selection (current visible 3-row version)
# ──────────────────────────────────────────────────────────────────────────────
FINAL_ROWS = [
    {"dataset": "voc", "filename": "003796.jpg"},
    {"dataset": "coco", "filename": "000000055950.jpg"},
    {"dataset": "imagenet", "filename": "ILSVRC2012_val_00002425_n01530575.JPEG"},
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Visual style
# ──────────────────────────────────────────────────────────────────────────────
NAVY = "#1B2A4A"
SLATE = "#6B7E9B"
TEXT = "#111111"
TEAL = "#00A99D"
AMBER = "#F5A623"

# subtle blue veil under the attention map
#VEIL_BLUE = "#5E789C"
#VEIL_BLUE = "#183A66"
#BLUE_VEIL_ALPHA = 0.58

#ATTN_ALPHA_MAX = 0.95
#ATTN_POWER = 0.72

#VEIL_BLUE = "#173B74"
#VEIL_BLUE = "#3E6F99"
VEIL_BLUE = "#2E5E88"
#BLUE_VEIL_ALPHA = 0.58
#ATTN_ALPHA_MAX = 0.95
#ATTN_POWER = 0.72

#BLUE_VEIL_ALPHA = 0.42 
#ATTN_ALPHA_MAX = 0.95 
#ATTN_POWER = 0.75

BLUE_VEIL_ALPHA = 0.42
ATTN_ALPHA_MAX = 0.95
ATTN_POWER = 1.20

ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attn_teal",
    [
        #"#EAF5F4",
        #"#54A6C7",
        #"#3E6F99",
        
        #"#8FD4CD",
        #"#2AA7B2",
        #"#1FA3A6",
        
        #"#00A99D"
        #"#00C2B8" 
        "#3E6F99",  # low attention
        "#22B8C8",  # low-mid
        "#FFC933",  # high-mid
        "#FFF2A6",  # peak
    ],
    N=256,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model / attention extraction
# ──────────────────────────────────────────────────────────────────────────────
class AttentionExtractor:
    """
    Extract final-layer CLS-to-patch attention.

    For registers model:
        skips CLS + register tokens when extracting patch attention.
    """

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self._attn = None
        self.num_prefix_tokens = self._infer_num_prefix_tokens(model)
        self._hook = model.blocks[-1].attn.register_forward_hook(self._hook_fn)

    def _infer_num_prefix_tokens(self, model):
        if hasattr(model, "num_prefix_tokens"):
            return int(model.num_prefix_tokens)

        if hasattr(model, "num_reg_tokens"):
            return 1 + int(model.num_reg_tokens)

        if hasattr(model, "reg_token") and model.reg_token is not None:
            if model.reg_token.ndim == 3:
                return 1 + int(model.reg_token.shape[1])

        return 1  # CLS only

    def _hook_fn(self, module, inputs, output):
        x = inputs[0]  # [B, N, C]
        B, N, C = x.shape
        H = module.num_heads
        D = C // H

        with torch.no_grad():
            qkv = module.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)

            q_norm = getattr(module, "q_norm", None)
            k_norm = getattr(module, "k_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            if k_norm is not None:
                k = k_norm(k)

            if hasattr(module, "scale") and module.scale is not None:
                scale = module.scale
            else:
                scale = D ** -0.5

            attn = ((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1)
            cls_to_patches = attn[:, :, 0, self.num_prefix_tokens:]   # [B, H, n_patches]
            cls_to_patches = cls_to_patches.mean(dim=1).squeeze(0).cpu()

        self._attn = cls_to_patches

    def extract_raw(self, tensor, device):
        self._attn = None
        with torch.no_grad():
            _ = self.model(tensor.to(device))

        if self._attn is None:
            raise RuntimeError(f"Attention hook did not capture output for model: {self.name}")

        a = self._attn.numpy()
        gs = int(np.sqrt(a.shape[0]))
        if gs * gs != a.shape[0]:
            raise RuntimeError(
                f"Patch attention length is not square: {a.shape[0]} for model {self.name}"
            )
        return a.reshape(gs, gs)


def load_model(model_key, paths_e2, device):
    name_map = {
        "baseline": "ViT-B_baseline_nomix",
        "registers": "ViT-B_registers_nomix",
        "saga": "ViT-B_SAGA_nomix",
    }

    ckpt_path = Path(paths_e2) / name_map[model_key] / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_key == "registers":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=1000,
            reg_tokens=4,
            dynamic_img_size=True,
        )
    else:
        model = build_saga_vit(
            "vit_base_patch16_224",
            gate=(model_key == "saga"),
            img_size=224,
            num_classes=1000,
            pretrained=False,
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    msg = model.load_state_dict(state, strict=False)
    top1 = ckpt.get("top1", None)

    print(f"  Loading {model_key} from {ckpt_path}...")
    if top1 is not None:
        try:
            print(f"  top-1={float(top1):.3f}%")
        except Exception:
            pass
    print(f"  missing keys: {len(msg.missing_keys)}")
    print(f"  unexpected keys: {len(msg.unexpected_keys)}")

    model = model.to(device).eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Archive extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_named_from_zip(zip_path, wanted_names, stage_dir):
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    found = {}
    wanted = set(wanted_names)

    print(f"  Opening zip archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"  Sample entries: {names[:5]}")

        for name in names:
            base = Path(name).name
            if base in wanted and base not in found:
                out_path = stage_dir / base
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                found[base] = str(out_path)
                print(f"  Extracted: {base}")

    missing = [x for x in wanted_names if x not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not find these files in zip archive {zip_path}: {missing}"
        )

    return found


def _extract_from_tar_once(tar_path, wanted_names, stage_dir, found):
    wanted = set(wanted_names)

    print(f"  Opening tar archive: {tar_path}")
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        sample = [m.name for m in members[:5]]
        print(f"  Sample entries: {sample}")

        for member in members:
            base = Path(member.name).name
            if base in wanted and base not in found and member.isfile():
                out_path = Path(stage_dir) / base
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                found[base] = str(out_path)
                print(f"  Extracted: {base}")


def extract_named_from_tar_with_voc_fallback(primary_tar, wanted_names, stage_dir):
    """
    Extract from a tar archive.
    If files are missing and the provided tar is a VOC tar, also try the sibling VOC tar.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    primary_tar = Path(primary_tar)
    found = {}

    searched = []
    if primary_tar.exists():
        searched.append(str(primary_tar))
        _extract_from_tar_once(primary_tar, wanted_names, stage_dir, found)

    missing = [x for x in wanted_names if x not in found]
    if not missing:
        return found

    # VOC fallback:
    # If user passes VOCtest_06-Nov-2007.tar and image is not there,
    # automatically try VOCtrainval_06-Nov-2007.tar in the same folder.
    fallback_candidates = []
    if "VOCtest_06-Nov-2007.tar" in primary_tar.name:
        fallback_candidates.append(primary_tar.with_name("VOCtrainval_06-Nov-2007.tar"))
    elif "VOCtrainval_06-Nov-2007.tar" in primary_tar.name:
        fallback_candidates.append(primary_tar.with_name("VOCtest_06-Nov-2007.tar"))

    for fb in fallback_candidates:
        if fb.exists():
            print(f"  Not all files found in primary VOC tar. Trying fallback: {fb}")
            searched.append(str(fb))
            _extract_from_tar_once(fb, wanted_names, stage_dir, found)

    missing = [x for x in wanted_names if x not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not find these files in tar archives {searched}: {missing}"
        )

    return found


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────
def make_blue_veil(h, w, hex_color=VEIL_BLUE, alpha=BLUE_VEIL_ALPHA):
    veil = np.zeros((h, w, 4), dtype=np.float32)
    r, g, b = to_rgb(hex_color)
    veil[..., 0] = r
    veil[..., 1] = g
    veil[..., 2] = b
    veil[..., 3] = alpha
    return veil


def upsample_attention_map(attn_map, out_w, out_h):
    arr = np.clip(attn_map, 0.0, 1.0)
    up = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (out_w, out_h), Image.BILINEAR
    )
    return np.asarray(up).astype(np.float32) / 255.0


def plot_teaser(rows_data, out_base):
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 4, figsize=(7.15, 5.55))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.subplots_adjust(
        left=0.045,
        right=0.915,
        top=0.92,
        bottom=0.05,
        wspace=0.045,
        hspace=0.07,
    )

    for r_idx, row in enumerate(rows_data):
        disp = row["display"]
        h, w = disp.shape[:2]

        all_maps = np.stack([row["b_raw"], row["r_raw"], row["s_raw"]], axis=0)

        vmin = np.percentile(all_maps, 10)
        vmax = np.percentile(all_maps, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-8

        for c_idx in range(4):
            ax = axes[r_idx, c_idx]
            ax.imshow(disp)

            if c_idx > 0:
                raw = [row["b_raw"], row["r_raw"], row["s_raw"]][c_idx - 1]

                # shared row normalization
                norm = np.clip((raw - vmin) / (vmax - vmin + 1e-8), 0, 1)
                attn_up = upsample_attention_map(norm, w, h)

                # subtle blue veil
                veil = make_blue_veil(h, w, hex_color=VEIL_BLUE, alpha=BLUE_VEIL_ALPHA)
                ax.imshow(veil, interpolation="nearest")

                # teal attention heatmap
                alpha_map = np.power(attn_up, ATTN_POWER)
                ax.imshow(
                    attn_up,
                    cmap=ATTN_CMAP,
                    alpha=ATTN_ALPHA_MAX * alpha_map,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="bilinear",
                )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            # SAGA border in amber
            if c_idx == 3:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(AMBER)
                    spine.set_linewidth(2.3)

    # column headers
    headers = ["Input", "ViT", "ViT+Registers", "SAGA (ours)"]

    for i, title in enumerate(headers):
        ax = axes[0, i]
        if i == 3:
            ax.set_title("")
            ax.text(
                0.48, 1.055, "SAGA",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color=TEXT,
            )
            ax.text(
                0.50, 1.055, "(ours)",
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=11.0,
                fontstyle="italic",
                color=TEAL,
            )
        else:
            ax.set_title(
                title,
                fontsize=11.7,
                fontweight="bold",
                color=TEXT,
                pad=10,
            )

    # shared colorbar
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=ATTN_CMAP)
    sm.set_array(np.linspace(0.0, 1.0, 256))
    cax = fig.add_axes([0.928, 0.08, 0.018, 0.81])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["0.0", "0.5", "1.0"])
    cb.set_label("Attention weight", fontsize=10.6, color=TEXT, labelpad=7)
    cb.ax.tick_params(labelsize=9.6, colors=TEXT, length=3)
    cb.outline.set_edgecolor(SLATE)

    out_png = str(out_base) + ".png"
    out_pdf = str(out_base) + ".pdf"

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths_e2", required=True, help="Checkpoint root containing ViT-B_* folders")
    parser.add_argument("--imagenet_tar", required=True, help="Path to ImageNet val tar")
    parser.add_argument("--coco_zip", required=True, help="Path to COCO val2017 zip")
    parser.add_argument("--voc_tar", required=True, help="Path to either VOCtest_06-Nov-2007.tar or VOCtrainval_06-Nov-2007.tar")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--basename", default="fig1_teaser_attention", help="Output basename")
    parser.add_argument("--stage_base", required=True, help="Temporary staging directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_base = Path(args.stage_base)

    if stage_base.exists():
        shutil.rmtree(stage_base)
    stage_base.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Temporary staging base: {stage_base}")
    print()

    try:
        # Load models
        print("Loading models...")
        models = {
            k: load_model(k, args.paths_e2, device)
            for k in ["baseline", "registers", "saga"]
        }
        extractors = {k: AttentionExtractor(models[k], k) for k in models.keys()}
        print()
        for k, ex in extractors.items():
            print(f"  AttentionExtractor [{k}] prefix tokens: {ex.num_prefix_tokens}")
        print()

        # Stage selected files
        print("Preparing selected images...\n")

        selected_imagenet = [x["filename"] for x in FINAL_ROWS if x["dataset"] == "imagenet"]
        selected_coco = [x["filename"] for x in FINAL_ROWS if x["dataset"] == "coco"]
        selected_voc = [x["filename"] for x in FINAL_ROWS if x["dataset"] == "voc"]

        extracted = {}

        if selected_voc:
            print("Stage VOC selected files:")
            extracted["voc"] = extract_named_from_tar_with_voc_fallback(
                args.voc_tar,
                selected_voc,
                stage_base / "voc",
            )
            print()

        if selected_coco:
            print("Stage COCO selected files:")
            extracted["coco"] = extract_named_from_zip(
                args.coco_zip,
                selected_coco,
                stage_base / "coco",
            )
            print()

        if selected_imagenet:
            print("Stage ImageNet selected files:")
            extracted["imagenet"] = extract_named_from_tar_with_voc_fallback(
                args.imagenet_tar,
                selected_imagenet,
                stage_base / "imagenet",
            )
            print()

        # Build rows
        rows_data = []
        print("Running inference on selected images...\n")

        for row in FINAL_ROWS:
            dataset = row["dataset"]
            filename = row["filename"]

            img_path = extracted[dataset][filename]
            img = Image.open(img_path).convert("RGB")

            # Standard 224 eval crop
            img = TF.resize(img, 256)
            img = TF.center_crop(img, [224, 224])

            disp = np.array(img)
            tensor = TF.normalize(TF.to_tensor(img), MEAN, STD).unsqueeze(0)

            print(f"  Processing [{dataset}] {filename}")

            row_dict = {
                "display": disp,
                "b_raw": extractors["baseline"].extract_raw(tensor, device),
                "r_raw": extractors["registers"].extract_raw(tensor, device),
                "s_raw": extractors["saga"].extract_raw(tensor, device),
            }
            rows_data.append(row_dict)

        # Plot
        out_base = out_dir / args.basename
        print("\nRendering final teaser figure...")
        plot_teaser(rows_data, out_base)

        print("\nDone.")
        print(f"Outputs saved in: {out_dir}")

    finally:
        if stage_base.exists():
            shutil.rmtree(stage_base)
            print(f"Cleaned up: {stage_base}")


if __name__ == "__main__":
    main()