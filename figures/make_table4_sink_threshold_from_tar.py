#!/usr/bin/env python3
"""
Compute Table 4: Sink-score threshold robustness from ImageNet val tar.

This version is designed for the HPC setup where ImageNet val is stored as:

    /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz

It does NOT require ImageFolder format.
It streams images directly from the tar file.

Usage debug:
    python3 make_table4_sink_threshold_from_tar.py \
        --paths_e2 /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --imagenet_tar /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
        --output /home/woody/iwi5/iwi5359h/saga_figures/table4_sink_threshold_debug.csv \
        --batch-size 32 \
        --max-images 512

Usage full:
    python3 make_table4_sink_threshold_from_tar.py \
        --paths_e2 /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints \
        --imagenet_tar /home/janus/iwi5-datasets/imagenet/imagenet-1k/data/val_images.tar.gz \
        --output /home/woody/iwi5/iwi5359h/saga_figures/table4_sink_threshold_robustness.csv \
        --batch-size 128
"""

import argparse
import csv
import sys
import tarfile
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga import build_saga_vit


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(key, paths_e2, device):
    name_map = {
        "baseline":  "ViT-B_baseline_nomix",
        "registers": "ViT-B_registers_nomix",
        "saga":      "ViT-B_SAGA_nomix",
    }

    ckpt_path = Path(paths_e2) / name_map[key] / "best.pth"

    print(f"\nLoading {key}")
    print(f"  checkpoint: {ckpt_path}")

    if key == "registers":
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
            gate=(key == "saga"),
            img_size=224,
            num_classes=1000,
            pretrained=False,
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"  top-1 from ckpt: {ckpt.get('top1', '?')}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction
# ─────────────────────────────────────────────────────────────────────────────

def infer_num_prefix_tokens(model):
    """
    Standard ViT/SAGA:
        [CLS] [PATCH] -> prefix = 1

    Register ViT in timm:
        [CLS] [REG] [PATCH] -> prefix = 1 + num_reg_tokens
    """
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)

    if hasattr(model, "num_reg_tokens"):
        return 1 + int(model.num_reg_tokens)

    if hasattr(model, "reg_token") and model.reg_token is not None:
        if model.reg_token.ndim == 3:
            return 1 + int(model.reg_token.shape[1])

    return 1


def extract_tokens(model, images):
    """
    Returns final token embeddings [B, T, C].
    """
    out = model.forward_features(images)

    if isinstance(out, dict):
        for key in ["x", "tokens", "last_hidden_state", "features"]:
            if key in out:
                out = out[key]
                break
        else:
            raise ValueError(f"Unknown dict output keys: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        found = None
        for item in out:
            if torch.is_tensor(item) and item.ndim == 3:
                found = item
                break
        if found is None:
            raise ValueError("Could not find [B,T,C] tensor in forward_features output.")
        out = found

    if not torch.is_tensor(out):
        raise TypeError(f"Unsupported forward_features output type: {type(out)}")

    if out.ndim != 3:
        raise ValueError(
            f"Expected token tensor [B,T,C], got shape {tuple(out.shape)}. "
            "Your model may be returning pooled features."
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet tar streaming
# ─────────────────────────────────────────────────────────────────────────────

def is_image_file(name):
    name = name.lower()
    return (
        name.endswith(".jpg")
        or name.endswith(".jpeg")
        or name.endswith(".png")
        or name.endswith(".JPEG".lower())
    )


def preprocess_pil(img):
    img = img.convert("RGB")
    img = TF.resize(img, 256, interpolation=TF.InterpolationMode.BICUBIC)
    img = TF.center_crop(img, [224, 224])
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, MEAN, STD)
    return tensor


def iter_image_batches_from_tar(tar_path, batch_size, max_images=None):
    """
    Streams images from tar and yields batches:
        images: [B,3,224,224]
        names: list[str]
    """
    tar_path = Path(tar_path)

    images = []
    names = []
    count = 0

    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()

        for m in members:
            if not m.isfile():
                continue

            if not is_image_file(m.name):
                continue

            f = tf.extractfile(m)
            if f is None:
                continue

            try:
                img = Image.open(f).convert("RGB")
                tensor = preprocess_pil(img)
            except Exception as e:
                print(f"Warning: failed to read {m.name}: {e}")
                continue

            images.append(tensor)
            names.append(Path(m.name).name)
            count += 1

            if len(images) == batch_size:
                yield torch.stack(images, dim=0), names
                images = []
                names = []

            if max_images is not None and count >= max_images:
                break

    if len(images) > 0:
        yield torch.stack(images, dim=0), names


# ─────────────────────────────────────────────────────────────────────────────
# Sink score computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_sink_scores(
    model,
    model_name,
    imagenet_tar,
    batch_size,
    thresholds,
    device,
    max_images=None,
):
    model.eval()

    num_prefix_tokens = infer_num_prefix_tokens(model)

    print(f"\nComputing sink scores for {model_name}")
    print(f"  num_prefix_tokens: {num_prefix_tokens}")
    print(f"  thresholds: {thresholds}")

    totals = {tau: 0.0 for tau in thresholds}
    total_images = 0

    batch_iter = iter_image_batches_from_tar(
        tar_path=imagenet_tar,
        batch_size=batch_size,
        max_images=max_images,
    )

    pbar = tqdm(batch_iter, desc=model_name)

    for images, names in pbar:
        images = images.to(device, non_blocking=True)

        tokens = extract_tokens(model, images)

        # Token layout expected:
        # baseline/SAGA: [CLS] [PATCH]
        # registers:     [CLS] [REG] [PATCH]
        patch_tokens = tokens[:, num_prefix_tokens:num_prefix_tokens + 196, :]

        if patch_tokens.shape[1] != 196:
            raise ValueError(
                f"{model_name}: expected 196 patch tokens, got "
                f"{patch_tokens.shape[1]}. Full token shape: {tuple(tokens.shape)}. "
                "Check register token layout."
            )

        # [B,196]
        patch_norms = patch_tokens.norm(dim=-1)

        # Per-image statistics
        mu = patch_norms.mean(dim=1, keepdim=True)
        sigma = patch_norms.std(dim=1, keepdim=True, unbiased=False)

        for tau in thresholds:
            sink_mask = patch_norms > (mu + tau * sigma)
            sink_fraction = sink_mask.float().mean(dim=1)
            totals[tau] += sink_fraction.sum().item()

        total_images += images.shape[0]
        pbar.set_postfix({"images": total_images})

    scores = {
        tau: 100.0 * totals[tau] / total_images
        for tau in thresholds
    }

    return scores, total_images


def save_results_csv(rows, output, thresholds):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "num_images"]
    for tau in thresholds:
        fieldnames.append(f"sink_{tau:.1f}sigma_percent")

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved CSV:")
    print(f"  {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths_e2", required=True)
    parser.add_argument("--imagenet_tar", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[2.5, 3.0, 3.5])
    parser.add_argument("--max-images", type=int, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []

    for key, display_name in [
        ("baseline", "ViT-B"),
        ("registers", "ViT-B + Registers"),
        ("saga", "SAGA"),
    ]:
        model = load_model(key, args.paths_e2, device)

        scores, num_images = compute_sink_scores(
            model=model,
            model_name=display_name,
            imagenet_tar=args.imagenet_tar,
            batch_size=args.batch_size,
            thresholds=args.thresholds,
            device=device,
            max_images=args.max_images,
        )

        row = {
            "model": display_name,
            "num_images": num_images,
        }

        for tau in args.thresholds:
            row[f"sink_{tau:.1f}sigma_percent"] = f"{scores[tau]:.4f}"

        results.append(row)

        print(f"\nResults: {display_name}")
        for tau in args.thresholds:
            print(f"  SinkScore {tau:.1f}σ: {scores[tau]:.4f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results_csv(results, args.output, args.thresholds)

    print("\nFinal Table 4:")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()