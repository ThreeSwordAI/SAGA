#!/usr/bin/env python3
"""
figures/fig1_candidates.py
===========================

Generates candidate attention map grids for Figure 1 selection.

Output:
    5 slots × 5 PNGs per slot = 25 PNGs total

Each PNG contains:
    5 candidate images × 4 columns

Columns:
    Input | ViT | ViT + Registers | SAGA (ours)

Important:
    - Baseline and SAGA use build_saga_vit().
    - ViT + Registers is loaded as a pure timm model.
    - saga/vit.py and saga/gate.py are NOT modified.
    - SAGA design remains CLS + patch tokens only.
    - Register tokens are handled only inside this script.
"""

import argparse
import json
import os
import random
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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga import build_saga_vit


# ── Paper colour palette ───────────────────────────────────────────────────────

NAVY = "#1B2A4A"
TEAL = "#00A99D"
SLATE = "#6B7E9B"

SAGA_CMAP = LinearSegmentedColormap.from_list(
    "saga",
    ["#1B2A4A", "#2E4A7A", "#8A9BB0", "#C4A882", "#F5A623"],
    N=256,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ── Staging helpers ────────────────────────────────────────────────────────────

def reset_stage_dir(stage_dir: str):
    """Remove old staged data, then recreate directory."""
    p = Path(stage_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def stage_imagenet_val(val_tar: str, stage_dir: str) -> str:
    """
    Extract ImageNet val tar to stage_dir.
    Returns path to the val images directory.
    """
    reset_stage_dir(stage_dir)
    out = Path(stage_dir)

    print("  Staging ImageNet val (~6GB)...")

    with tarfile.open(val_tar, "r:gz") as tf:
        members = tf.getmembers()
        sample = [m.name for m in members[:5]]
        print(f"  Sample entries: {sample}")
        tf.extractall(out, filter="data")

    val_dir = out

    n = (
        sum(1 for _ in val_dir.rglob("*.JPEG"))
        + sum(1 for _ in val_dir.rglob("*.jpg"))
        + sum(1 for _ in val_dir.rglob("*.jpeg"))
    )

    print(f"  ImageNet val staged: {n} images")
    return str(val_dir)


def stage_coco_val(val_zip: str, ann_zip: str, stage_dir: str) -> tuple:
    """
    Extract COCO val images + annotations to stage_dir.
    Returns (val_images_dir, annotations_json_path).
    """
    reset_stage_dir(stage_dir)
    out = Path(stage_dir)

    print("  Staging COCO val (~1GB)...")
    with zipfile.ZipFile(val_zip, "r") as zf:
        zf.extractall(out)

    print("  Staging COCO annotations (~240MB)...")
    with zipfile.ZipFile(ann_zip, "r") as zf:
        members = [m for m in zf.namelist() if "instances_val2017" in m]
        zf.extractall(out, members=members)

    val_dir = out / "val2017"
    ann_file = out / "annotations" / "instances_val2017.json"

    n = len(list(val_dir.glob("*.jpg")))
    print(f"  COCO val staged: {n} images")

    return str(val_dir), str(ann_file)


def stage_cub(tar_path: str, stage_dir: str) -> str:
    """
    Extract CUB tar.
    Returns path to CUB_200_2011.
    """
    reset_stage_dir(stage_dir)
    out = Path(stage_dir)

    print("  Staging CUB-200-2011 (~1.1GB)...")

    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out, filter="data")

    cub_root = out / "CUB_200_2011"
    n = sum(1 for _ in cub_root.rglob("*.jpg"))

    print(f"  CUB staged: {n} images")
    return str(cub_root)


def stage_ade_val(zip_path: str, stage_dir: str) -> str:
    """
    Extract only ADE20K validation images.
    Returns path to images/validation.
    """
    reset_stage_dir(stage_dir)
    out = Path(stage_dir)

    print("  Staging ADE20K val images (~100MB)...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        val_members = [
            m
            for m in zf.namelist()
            if "images/validation/" in m and m.endswith(".jpg")
        ]

        for m in val_members:
            zf.extract(m, out)

    val_dir = out / "ADEChallengeData2016" / "images" / "validation"
    n = len(list(val_dir.glob("*.jpg")))

    print(f"  ADE20K val staged: {n} images")
    return str(val_dir)


def cleanup(stage_dir: str):
    """Remove staged data."""
    p = Path(stage_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"  Cleaned up: {stage_dir}")


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(path: str, size: int = 224) -> tuple:
    """
    Returns:
        display image: HWC uint8
        tensor: [1, 3, H, W] normalized
    """
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, size + 32)
    img = TF.center_crop(img, size)

    display = np.array(img)

    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, MEAN, STD)

    return display, tensor.unsqueeze(0)


# ── Attention extraction ───────────────────────────────────────────────────────

class AttentionExtractor:
    """
    Extracts last-block CLS → patch attention, averaged over heads.

    Handles:
        - normal ViT: CLS + patches
        - SAGA: CLS + patches
        - Registers: CLS + register tokens + patches

    For ViT + Registers:
        attention to register tokens is skipped.
        only CLS → patch attention is visualized.
    """

    def __init__(self, model: nn.Module, name: str):
        self.model = model
        self.name = name
        self._attn = None
        self.num_prefix_tokens = self._infer_num_prefix_tokens(model)

        print(
            f"  AttentionExtractor [{self.name}] "
            f"prefix tokens: {self.num_prefix_tokens}"
        )

        self._hook = model.blocks[-1].attn.register_forward_hook(self._hook_fn)

    def _infer_num_prefix_tokens(self, model: nn.Module) -> int:
        """
        Infer how many non-patch tokens appear before patch tokens.

        normal ViT / SAGA:
            CLS = 1

        ViT + Registers:
            CLS + register tokens = usually 5
        """
        if hasattr(model, "num_prefix_tokens"):
            return int(model.num_prefix_tokens)

        if hasattr(model, "num_reg_tokens"):
            return 1 + int(model.num_reg_tokens)

        if hasattr(model, "reg_token") and model.reg_token is not None:
            return 1 + int(model.reg_token.shape[1])

        if hasattr(model, "pos_embed") and hasattr(model, "patch_embed"):
            if hasattr(model.patch_embed, "num_patches"):
                return int(model.pos_embed.shape[1] - model.patch_embed.num_patches)

        return 1

    def _hook_fn(self, module, input, output):
        x = input[0]
        B, N, C = x.shape

        H = module.num_heads

        if hasattr(module, "head_dim"):
            D = module.head_dim
        else:
            D = C // H

        with torch.no_grad():
            qkv = module.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)

            q_norm = getattr(module, "q_norm", nn.Identity())
            k_norm = getattr(module, "k_norm", nn.Identity())

            q = q_norm(q)
            k = k_norm(k)

            scale = getattr(module, "scale", D ** -0.5)

            attn = (q * scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

        patch_attn = attn[:, :, 0, self.num_prefix_tokens:]
        self._attn = patch_attn.mean(1).squeeze(0).cpu()

    def extract(self, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        self._attn = None

        with torch.no_grad():
            _ = self.model(tensor.to(device))

        if self._attn is None:
            raise RuntimeError(f"Attention hook did not capture attention for {self.name}.")

        a = self._attn.numpy()

        gs = int(a.shape[0] ** 0.5)

        if gs * gs != a.shape[0]:
            raise RuntimeError(
                f"Patch attention is not square for {self.name}. "
                f"Got {a.shape[0]} tokens. "
                f"Prefix tokens used: {self.num_prefix_tokens}"
            )

        a = a.reshape(gs, gs)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)

        return a

    def remove(self):
        self._hook.remove()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(key: str, ckpt_dir: str, device: torch.device) -> nn.Module:
    name_map = {
        "baseline": "ViT-B_baseline_nomix",
        "registers": "ViT-B_registers_nomix",
        "saga": "ViT-B_SAGA_nomix",
    }

    ckpt_path = Path(ckpt_dir) / name_map[key] / "best.pth"
    print(f"  Loading {key} from {ckpt_path}...")

    if key == "registers":
        import timm

        # Important:
        # Register model is pure timm.
        # Do NOT wrap it with SAGAViT.
        m = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=1000,
            reg_tokens=4,
            dynamic_img_size=True,
        )

    else:
        # Baseline and SAGA use your custom wrapper.
        # gate=False gives baseline ViT.
        # gate=True gives SAGA.
        m = build_saga_vit(
            "vit_base_patch16_224",
            gate=(key == "saga"),
            img_size=224,
            num_classes=1000,
            pretrained=False,
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    raw_state = ckpt.get("model", ckpt)
    state = {
        k.replace("module.", ""): v
        for k, v in raw_state.items()
    }

    missing, unexpected = m.load_state_dict(state, strict=False)

    print(f"  top-1={ckpt.get('top1', '?')}%")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print(f"  first missing keys: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"  first unexpected keys: {unexpected[:5]}")

    return m.to(device).eval()


# ── Candidate selection ────────────────────────────────────────────────────────

def get_imagenet_candidates(imagenet_val_dir: str, synsets: list, n: int) -> list:
    val = Path(imagenet_val_dir)
    paths = []

    # Try synset subfolders first.
    for s in synsets:
        for d in [val / s, val / "val" / s]:
            if d.exists():
                paths.extend(d.glob("*.JPEG"))
                paths.extend(d.glob("*.jpg"))
                paths.extend(d.glob("*.jpeg"))

    # Fallback: flat directory, filter by synset name in filename.
    if not paths:
        all_imgs = (
            list(val.rglob("*.JPEG"))
            + list(val.rglob("*.jpg"))
            + list(val.rglob("*.jpeg"))
        )

        matched = [
            p
            for p in all_imgs
            if any(s in p.stem or s in str(p) for s in synsets)
        ]

        paths = matched if matched else all_imgs

        if not matched:
            print("  NOTE: no synset matches — using random images for slot 1")

    random.shuffle(paths)
    return paths[:n]


def get_coco_small(ann_file: str, val_dir: str, n: int) -> list:
    data = json.load(open(ann_file))

    small_ids = {
        a["image_id"]
        for a in data["annotations"]
        if not a.get("iscrowd") and a.get("area", 0) < 1024
    }

    imgs = [i for i in data["images"] if i["id"] in small_ids]
    random.shuffle(imgs)

    paths = []

    for img in imgs:
        p = Path(val_dir) / img["file_name"]

        if p.exists():
            paths.append(p)

        if len(paths) == n:
            break

    return paths


def get_coco_multi(ann_file: str, val_dir: str, n: int, min_obj: int = 8) -> list:
    from collections import Counter

    data = json.load(open(ann_file))

    cnt = Counter(
        a["image_id"]
        for a in data["annotations"]
        if not a.get("iscrowd")
    )

    ids = {k for k, v in cnt.items() if v >= min_obj}
    imgs = [i for i in data["images"] if i["id"] in ids]

    random.shuffle(imgs)

    paths = []

    for img in imgs:
        p = Path(val_dir) / img["file_name"]

        if p.exists():
            paths.append(p)

        if len(paths) == n:
            break

    return paths


def get_cub_candidates(cub_root: str, n: int) -> list:
    sky_classes = [
        "001.Black_footed_Albatross",
        "002.Laysan_Albatross",
        "003.Sooty_Albatross",
        "059.California_Gull",
        "060.Glaucous_winged_Gull",
        "061.Heermann_Gull",
        "062.Herring_Gull",
        "063.Ivory_Gull",
    ]

    paths = []

    for cls in sky_classes:
        d = Path(cub_root) / "images" / cls

        if d.exists():
            paths.extend(d.glob("*.jpg"))

    random.shuffle(paths)
    return paths[:n]


def get_ade_candidates(ade_val: str, n: int) -> list:
    paths = list(Path(ade_val).glob("*.jpg"))
    random.shuffle(paths)
    return paths[:n]


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_candidates(
    image_paths,
    extractors,
    device,
    slot_name,
    out_path,
    alpha=0.55,
):
    """
    Save one candidate grid PNG.

    Each PNG contains:
        rows = candidate images
        columns = Input | ViT | ViT + Registers | SAGA
    """
    n = len(image_paths)

    if n == 0:
        print(f"  WARNING: no images found for {slot_name} — skipping")
        return

    fig, axes = plt.subplots(
        n,
        4,
        figsize=(12, n * 2.8),
        gridspec_kw={"wspace": 0.04, "hspace": 0.08},
    )

    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Input", "ViT", "ViT + Registers", "SAGA (ours)"]
    model_keys = ["baseline", "registers", "saga"]

    for row, img_path in enumerate(image_paths):
        try:
            display, tensor = load_image(str(img_path))
        except Exception as e:
            print(f"  WARNING: {img_path}: {e}")
            continue

        # Column 0: input image.
        axes[row, 0].imshow(display)
        axes[row, 0].set_ylabel(
            Path(img_path).name,
            fontsize=5,
            color=SLATE,
            rotation=0,
            ha="right",
            va="center",
            labelpad=3,
        )

        # Columns 1-3: attention overlays.
        for col, key in enumerate(model_keys, start=1):
            attn = extractors[key].extract(tensor, device)

            attn_up = np.array(
                Image.fromarray((attn * 255).astype(np.uint8), "L").resize(
                    (display.shape[1], display.shape[0]),
                    Image.BILINEAR,
                )
            ) / 255.0

            axes[row, col].imshow(display)
            axes[row, col].imshow(
                attn_up,
                cmap=SAGA_CMAP,
                alpha=alpha,
                vmin=0,
                vmax=1,
            )

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Column headers.
    for col, label in enumerate(col_labels):
        ax = axes[0, col]

        if label == "SAGA (ours)":
            ax.text(
                0.5,
                1.07,
                "SAGA",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=NAVY,
            )
            ax.text(
                0.5,
                1.01,
                "(ours)",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                fontstyle="italic",
                color=TEAL,
            )
        else:
            ax.set_title(
                label,
                fontsize=10,
                color=NAVY,
                pad=6,
                fontweight="regular",
            )

    # Teal border on SAGA column.
    for row in range(n):
        for spine in axes[row, 3].spines.values():
            spine.set_edgecolor(TEAL)
            spine.set_linewidth(1.8)
            spine.set_visible(True)

    fig.suptitle(
        f"Candidates — {slot_name}",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
        y=1.01,
    )

    # Colourbar.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=SAGA_CMAP)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention weight", fontsize=7, color=SLATE)
    cbar.ax.tick_params(labelsize=6, colors=SLATE)
    cbar.outline.set_edgecolor(SLATE)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Saved: {out_path}")


def plot_candidate_sets(
    image_paths,
    extractors,
    device,
    slot_name,
    file_stem,
    out_dir,
    n_per_png=5,
    sets_per_slot=5,
    alpha=0.55,
):
    """
    Save multiple candidate PNGs for one slot.

    Example:
        sets_per_slot = 5
        n_per_png = 5

    Output:
        candidates_slot1_uniform_background_set01.png
        candidates_slot1_uniform_background_set02.png
        ...
        candidates_slot1_uniform_background_set05.png
    """
    needed = n_per_png * sets_per_slot

    if len(image_paths) == 0:
        print(f"  WARNING: no images found for {slot_name} — skipping")
        return

    if len(image_paths) < needed:
        print(
            f"  WARNING: requested {needed} images for {slot_name}, "
            f"but only found {len(image_paths)}. Will save fewer sets."
        )

    for set_idx in range(sets_per_slot):
        start = set_idx * n_per_png
        end = start + n_per_png

        batch_paths = image_paths[start:end]

        if len(batch_paths) == 0:
            break

        out_path = Path(out_dir) / f"{file_stem}_set{set_idx + 1:02d}.png"

        print(
            f"  Saving {slot_name}, set {set_idx + 1}/{sets_per_slot}: "
            f"{[Path(p).name for p in batch_paths]}"
        )

        plot_candidates(
            batch_paths,
            extractors,
            device,
            f"{slot_name} — set {set_idx + 1}/{sets_per_slot}",
            str(out_path),
            alpha,
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    default_stage = str(
        Path(tempfile.gettempdir()) / f"fig1_stage_{os.environ.get('USER', 'user')}"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--paths_e2", required=True)

    parser.add_argument(
        "--imagenet_tar",
        required=True,
        help="ImageNet val_images.tar.gz",
    )
    parser.add_argument(
        "--coco_zip",
        required=True,
        help="COCO val2017.zip",
    )
    parser.add_argument(
        "--coco_ann_zip",
        required=True,
        help="COCO annotations_trainval2017.zip",
    )
    parser.add_argument(
        "--cub_tar",
        required=True,
        help="CUB_200_2011.tgz",
    )
    parser.add_argument(
        "--ade_zip",
        required=True,
        help="ADEChallengeData2016.zip",
    )

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stage_base", default=default_stage)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of candidate images per PNG.",
    )

    parser.add_argument(
        "--sets_per_slot",
        type=int,
        default=5,
        help="Number of PNG candidate grids to generate per slot.",
    )

    parser.add_argument("--alpha", type=float, default=0.55)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Temporary staging base: {args.stage_base}")
    print(f"Images per PNG: {args.n}")
    print(f"Sets per slot: {args.sets_per_slot}")
    print(f"Expected PNGs: {5 * args.sets_per_slot}\n")

    # ── Load models once ──────────────────────────────────────────────────────
    print("Loading models...")

    models = {
        "baseline": load_model("baseline", args.paths_e2, device),
        "registers": load_model("registers", args.paths_e2, device),
        "saga": load_model("saga", args.paths_e2, device),
    }

    extractors = {
        k: AttentionExtractor(m, name=k)
        for k, m in models.items()
    }

    print()

    bird_synsets = [
        "n01530575",
        "n01531178",
        "n01532829",
        "n01534433",
        "n01537544",
        "n01558993",
        "n01560419",
        "n01580077",
    ]

    try:
        # ── Slot 1: ImageNet birds ────────────────────────────────────────────
        print("=" * 55)
        print("  Slot 1 — Uniform background (ImageNet birds)")
        print("=" * 55)

        imagenet_stage = str(Path(args.stage_base) / "imagenet_val")

        try:
            imagenet_val_dir = stage_imagenet_val(
                args.imagenet_tar,
                imagenet_stage,
            )

            paths1 = get_imagenet_candidates(
                imagenet_val_dir,
                bird_synsets,
                args.n * args.sets_per_slot,
            )

            print(f"  Total candidates: {len(paths1)}")

            plot_candidate_sets(
                paths1,
                extractors,
                device,
                "Slot 1 — Uniform background",
                "candidates_slot1_uniform_background",
                args.out_dir,
                n_per_png=args.n,
                sets_per_slot=args.sets_per_slot,
                alpha=args.alpha,
            )

        finally:
            cleanup(imagenet_stage)

        # ── Slots 2 + 3: COCO ────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("  Staging COCO for Slots 2 + 3")
        print("=" * 55)

        coco_stage = str(Path(args.stage_base) / "coco")

        try:
            coco_val_dir, coco_ann_file = stage_coco_val(
                args.coco_zip,
                args.coco_ann_zip,
                coco_stage,
            )

            print("\n  Slot 2 — Small object in cluttered background")

            paths2 = get_coco_small(
                coco_ann_file,
                coco_val_dir,
                args.n * args.sets_per_slot,
            )

            print(f"  Total candidates: {len(paths2)}")

            plot_candidate_sets(
                paths2,
                extractors,
                device,
                "Slot 2 — Small object (COCO)",
                "candidates_slot2_small_object",
                args.out_dir,
                n_per_png=args.n,
                sets_per_slot=args.sets_per_slot,
                alpha=args.alpha,
            )

            print("\n  Slot 3 — Multiple objects, complex scene")

            paths3 = get_coco_multi(
                coco_ann_file,
                coco_val_dir,
                args.n * args.sets_per_slot,
            )

            print(f"  Total candidates: {len(paths3)}")

            plot_candidate_sets(
                paths3,
                extractors,
                device,
                "Slot 3 — Multiple objects (COCO)",
                "candidates_slot3_multiple_objects",
                args.out_dir,
                n_per_png=args.n,
                sets_per_slot=args.sets_per_slot,
                alpha=args.alpha,
            )

        finally:
            cleanup(coco_stage)

        # ── Slot 4: CUB ──────────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("  Slot 4 — Fine-grained bird against sky (CUB-200)")
        print("=" * 55)

        cub_stage = str(Path(args.stage_base) / "cub")

        try:
            cub_root = stage_cub(
                args.cub_tar,
                cub_stage,
            )

            paths4 = get_cub_candidates(
                cub_root,
                args.n * args.sets_per_slot,
            )

            print(f"  Total candidates: {len(paths4)}")

            plot_candidate_sets(
                paths4,
                extractors,
                device,
                "Slot 4 — Fine-grained bird (CUB-200)",
                "candidates_slot4_finegrained",
                args.out_dir,
                n_per_png=args.n,
                sets_per_slot=args.sets_per_slot,
                alpha=args.alpha,
            )

        finally:
            cleanup(cub_stage)

        # ── Slot 5: ADE20K ───────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("  Slot 5 — Indoor scene (ADE20K val)")
        print("=" * 55)

        ade_stage = str(Path(args.stage_base) / "ade")

        try:
            ade_val = stage_ade_val(
                args.ade_zip,
                ade_stage,
            )

            paths5 = get_ade_candidates(
                ade_val,
                args.n * args.sets_per_slot,
            )

            print(f"  Total candidates: {len(paths5)}")

            plot_candidate_sets(
                paths5,
                extractors,
                device,
                "Slot 5 — Indoor scene (ADE20K)",
                "candidates_slot5_indoor",
                args.out_dir,
                n_per_png=args.n,
                sets_per_slot=args.sets_per_slot,
                alpha=args.alpha,
            )

        finally:
            cleanup(ade_stage)

    finally:
        for ext in extractors.values():
            ext.remove()

    print(f"\n{'=' * 55}")
    print("  Candidate PNGs saved to:")
    print(f"  {args.out_dir}")
    print(f"  Expected total: {5 * args.sets_per_slot} PNGs")
    print()
    print("  Output pattern:")
    print("  candidates_slot1_uniform_background_set01.png")
    print("  candidates_slot1_uniform_background_set02.png")
    print("  ...")
    print("  candidates_slot5_indoor_set05.png")
    print()
    print("  Open each PNG and pick one final candidate per slot.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()