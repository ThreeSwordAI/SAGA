#!/usr/bin/env python3
"""
figures/fig1_aircraft_candidates.py
===================================

Generates aircraft candidate attention map grids for replacing Figure 1 Slot 4.

Output:
    5 PNGs total for aircraft replacement slot

Each PNG contains:
    5 candidate images × 4 columns

Columns:
    Input | ViT | ViT + Registers | SAGA (ours)

Important:
    - Baseline and SAGA use build_saga_vit().
    - ViT + Registers is loaded as pure timm.
    - saga/vit.py and saga/gate.py are NOT modified.
    - SAGA design remains CLS + patch tokens only.
    - Register tokens are handled only inside this script.
"""

import argparse
import os
import random
import shutil
import sys
import tarfile
import tempfile
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
    p = Path(stage_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def cleanup(stage_dir: str):
    p = Path(stage_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"  Cleaned up: {stage_dir}")


def stage_aircraft(aircraft_tar: str, stage_dir: str) -> str:
    """
    Extract FGVC-Aircraft tar temporarily.
    Returns dataset root.
    Expected structure usually:
        fgvc-aircraft-2013b/data/images/*.jpg
    """
    reset_stage_dir(stage_dir)
    out = Path(stage_dir)

    print("  Staging FGVC-Aircraft dataset...")

    with tarfile.open(aircraft_tar, "r:gz") as tf:
        members = tf.getmembers()
        sample = [m.name for m in members[:5]]
        print(f"  Sample entries: {sample}")
        tf.extractall(out, filter="data")

    possible_roots = [
        out / "fgvc-aircraft-2013b",
        out / "fgvc-aircraft-2013b" / "data",
        out,
    ]

    for root in possible_roots:
        if root.exists():
            n = len(list(root.rglob("*.jpg"))) + len(list(root.rglob("*.jpeg"))) + len(list(root.rglob("*.png")))
            if n > 0:
                print(f"  Aircraft staged: {n} images")
                return str(root)

    raise RuntimeError("Could not find aircraft images after extraction.")


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

        m = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=1000,
            reg_tokens=4,
            dynamic_img_size=True,
        )

    else:
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

def get_aircraft_candidates(aircraft_root: str, n: int) -> list:
    """
    Select random aircraft images.
    """
    root = Path(aircraft_root)

    preferred_dirs = [
        root / "data" / "images",
        root / "images",
        root,
    ]

    paths = []

    for d in preferred_dirs:
        if d.exists():
            paths = (
                list(d.rglob("*.jpg"))
                + list(d.rglob("*.jpeg"))
                + list(d.rglob("*.png"))
            )
            if len(paths) > 0:
                break

    if len(paths) == 0:
        raise RuntimeError(f"No aircraft images found under {aircraft_root}")

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
        Path(tempfile.gettempdir()) / f"fig1_aircraft_stage_{os.environ.get('USER', 'user')}"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--paths_e2", required=True)
    parser.add_argument("--aircraft_tar", required=True)
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
        help="Number of PNG candidate grids to generate.",
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
    print(f"Sets: {args.sets_per_slot}")
    print(f"Expected PNGs: {args.sets_per_slot}\n")

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

    aircraft_stage = str(Path(args.stage_base) / "aircraft")

    try:
        print("=" * 55)
        print("  Replacement Slot 4 — Aircraft")
        print("=" * 55)

        try:
            aircraft_root = stage_aircraft(
                args.aircraft_tar,
                aircraft_stage,
            )

            paths = get_aircraft_candidates(
                aircraft_root,
                args.n * args.sets_per_slot,
            )

            print(f"  Total candidates: {len(paths)}")

            plot_candidate_sets(
                paths,
                extractors,
                device,
                "Slot 4 — Aircraft",
                "candidates_slot4_aircraft",
                args.out_dir,
                n_per_png=args.n,
                sets_per_slot=args.sets_per_slot,
                alpha=args.alpha,
            )

        finally:
            cleanup(aircraft_stage)

    finally:
        for ext in extractors.values():
            ext.remove()

    print(f"\n{'=' * 55}")
    print("  Aircraft candidate PNGs saved to:")
    print(f"  {args.out_dir}")
    print()
    print("  Output pattern:")
    print("  candidates_slot4_aircraft_set01.png")
    print("  candidates_slot4_aircraft_set02.png")
    print("  candidates_slot4_aircraft_set03.png")
    print("  candidates_slot4_aircraft_set04.png")
    print("  candidates_slot4_aircraft_set05.png")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()