#!/usr/bin/env python3
"""
evaluation/e5_lost/tools/extract_features.py
=============================================
Extract patch features from ViT models on VOC 2007 test split.
Saves features to disk for LOST evaluation.

Features extracted: last-block patch token outputs [N_images, N_patches, dim]
Also saves CLS attention maps for visualisation.

Usage (on TinyGPU):
    python3 evaluation/e5_lost/tools/extract_features.py \
        --paths  evaluation/e5_lost/configs/paths.yaml \
        --models baseline registers saga pretrained \
        --split  test \
        --batch  32

One run extracts all requested models sequentially.
~5 min per model on TinyGPU RTX 2080 Ti.
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.e5_lost.data.voc_dataset import VOC2007Dataset, stage_voc
from saga import build_saga_vit


# ── Feature extractor ──────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Wraps a SAGAViT model to extract:
      1. Last-block patch features [B, N_patches, dim]
      2. CLS-to-patch attention map from last block [B, H, N_patches]

    These are the two feature types used by LOST.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._patch_features  = None
        self._cls_attention   = None
        self._hooks           = []
        self._register_hooks()

    def _register_hooks(self):
        last_block = self.model.blocks[-1]

        # Hook 1: patch tokens after last block (before norm + head)
        def feat_hook(module, input, output):
            # output: [B, N, C] where N = 1 (CLS) + n_patches
            self._patch_features = output[:, 1:, :].detach().cpu()

        # Hook 2: CLS attention from last block attention module
        def attn_hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            H = module.num_heads
            D = C // H
            with torch.no_grad():
                qkv = module.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                q = module.q_norm(q) if hasattr(module, 'q_norm') else q
                k = module.k_norm(k) if hasattr(module, 'k_norm') else k
                attn = (q * module.scale) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                # CLS-to-patch: [B, H, N_patches]
                self._cls_attention = attn[:, :, 0, 1:].detach().cpu()

        self._hooks.append(last_block.register_forward_hook(feat_hook))
        self._hooks.append(last_block.attn.register_forward_hook(attn_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        _ = self.model(x)
        return self._patch_features, self._cls_attention


# ── Model builders ─────────────────────────────────────────────────────────────

def load_model(model_key: str, paths: dict, device) -> nn.Module:
    """Load a ViT model from an E2 checkpoint or timm pretrained weights."""

    if model_key == 'pretrained':
        # Fully pretrained ViT-B from timm (ImageNet-21k + finetune)
        print("  Loading pretrained ViT-B (timm imagenet21k)...")
        import timm
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=1000,
        )
        # Wrap in SAGAViT for consistent interface
        from saga.vit import SAGAViT
        model = SAGAViT(model)
        return model.to(device).eval()

    # E2 checkpoint
    ckpt_path = paths['checkpoints'][model_key]
    is_saga   = (model_key == 'saga')
    is_reg    = (model_key == 'registers')

    print(f"  Loading {model_key} from {ckpt_path}...")

    if is_reg:
        import timm
        from saga.vit import SAGAViT
        timm_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=1000,
            reg_tokens=4,
            dynamic_img_size=True,
        )
        model = SAGAViT(timm_model)
    else:
        model = build_saga_vit(
            'vit_base_patch16_224',
            gate=is_saga,
            img_size=224,
            num_classes=1000,
            pretrained=False,
        )

    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = {k.replace('module.', ''): v
             for k, v in ckpt.get('model', ckpt).items()}
    model.load_state_dict(state, strict=False)
    print(f"  Loaded. top-1={ckpt.get('top1', '?')}")

    return model.to(device).eval()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths',   required=True,
                        help='evaluation/e5_lost/configs/paths.yaml')
    parser.add_argument('--models',  nargs='+',
                        default=['baseline', 'registers', 'saga', 'pretrained'],
                        choices=['baseline', 'registers', 'saga', 'pretrained'],
                        help='Which models to extract features for')
    parser.add_argument('--split',   default='test',
                        help='VOC split (test = 4952 images, standard for LOST)')
    parser.add_argument('--batch',   type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip models whose features already exist on disk')
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Stage VOC
    stage_dir = str(Path(paths['data']['stage_base']) / 'voc_e5')
    voc_root  = stage_voc(
        paths['data']['voc_trainval_tar'],
        paths['data']['voc_test_tar'],
        stage_dir,
    )

    # Output directory
    feat_dir = Path(paths['outputs']['features'])
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = VOC2007Dataset(voc_root, split=args.split, img_size=224)
    loader = DataLoader(ds, batch_size=args.batch,
                        num_workers=args.workers,
                        shuffle=False, pin_memory=True,
                        collate_fn=lambda b: (
                            torch.stack([x[0] for x in b]),
                            [x[1] for x in b],
                            [x[2] for x in b],
                        ))

    print(f"Extracting features for: {args.models}\n")

    for model_key in args.models:
        out_file = feat_dir / f'{model_key}_{args.split}_features.npz'

        if args.skip_existing and out_file.exists():
            print(f"Skipping {model_key} — features already exist: {out_file}")
            continue

        print(f"\n{'='*50}")
        print(f"  Model: {model_key}")
        print(f"{'='*50}")

        model     = load_model(model_key, paths, device)
        extractor = FeatureExtractor(model)

        all_patch_feats = []   # [N_images, N_patches, dim]
        all_cls_attns   = []   # [N_images, H, N_patches]
        all_img_ids     = []
        all_annotations = []

        for i, (images, img_ids, anns) in enumerate(loader):
            if i % 20 == 0:
                print(f"  {i * args.batch}/{len(ds)}", flush=True)

            images = images.to(device)
            patch_feats, cls_attn = extractor(images)

            all_patch_feats.append(patch_feats.numpy())
            all_cls_attns.append(cls_attn.numpy())
            all_img_ids.extend(img_ids)
            all_annotations.extend(anns)

        all_patch_feats = np.concatenate(all_patch_feats, axis=0)
        all_cls_attns   = np.concatenate(all_cls_attns,   axis=0)

        print(f"  Patch features: {all_patch_feats.shape}")
        print(f"  CLS attentions: {all_cls_attns.shape}")

        # Save features
        np.savez_compressed(
            out_file,
            patch_feats  = all_patch_feats,   # [N, 196, 768]
            cls_attns    = all_cls_attns,      # [N, H, 196]
            img_ids      = np.array(all_img_ids),
        )

        # Save annotations separately (Python objects — use pickle via np)
        import pickle
        ann_file = feat_dir / f'{model_key}_{args.split}_annotations.pkl'
        with open(ann_file, 'wb') as f:
            pickle.dump(all_annotations, f)

        print(f"  Saved: {out_file}")
        extractor.remove_hooks()
        del model, extractor
        torch.cuda.empty_cache()

    print(f"\nAll features saved to: {feat_dir}")
    print("Next: run evaluation/e5_lost/tools/run_lost.py")


if __name__ == '__main__':
    main()