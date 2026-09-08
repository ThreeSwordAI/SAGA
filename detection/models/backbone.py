"""
detection/models/backbone.py
=============================
Loads an ImageNet classification checkpoint and wraps it for use as a dense
prediction backbone (detection AND segmentation — segmentation/models/
backbone.py re-exports this module, so there is ONE implementation).

The backbone:
  1. builds the model EXACTLY as the classifier was trained
     (tools/model_factory mirrors classification/tools/train.py::build_model)
     but with dynamic_img_size=True so non-224 dense inputs work;
  2. strict-loads the checkpoint (zero missing / zero unexpected keys) and
     records its sha256 for provenance;
  3. returns patch-token sequences from `fpn_indices` blocks for the neck;
  4. supports freezing for the first N epochs of dense training.

TASK-09 bug B6 — the registers variant:
    The previous version wrapped the timm `reg_tokens=4` model in SAGAViT.
    That wrapper copies only `cls_token`/`pos_embed` and assumes one prefix
    token, so the model's `reg_token` parameter was DROPPED (it survived
    only because the load was strict=False) and the pos-embed layout
    (`[1, 5+HW, C]` for a register model) was misread. Every registers
    detection number produced that way is void.
    The fix: for `registers > 0` the timm model is used DIRECTLY and its own
    `forward_intermediates()` (timm >= 1.0) provides the intermediates — it
    strips `num_prefix_tokens` (cls + registers) itself and resamples the
    pos-embed for the actual input grid. SAGAViT now REFUSES register-token
    models outright (saga/vit.py).

Checkpoint compatibility note (verified): adding `dynamic_img_size=True`
changes no parameter name or shape and is value-identical at 224, so a
checkpoint trained without it strict-loads here.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from saga import build_saga_vit
from saga.run_registry import file_sha256
from tools.model_factory import extract_state_dict


class DetectionBackbone(nn.Module):
    """
    ViT backbone for dense prediction.

    Args:
        arch          timm arch name (e.g. 'vit_base_patch16_224')
        gate          True = SAGA (SpatialGate at G1), False = plain ViT
        registers     Number of register tokens (0 for baseline/SAGA)
        ckpt_path     Classification checkpoint to load. None = random init;
                      TESTS/SMOKES ONLY (production always resolves a real
                      checkpoint from configs/dense_matrix.yaml).
        fpn_indices   Which block outputs to return, e.g. [3, 6, 9, 11]
        img_size      Backbone training image size (224) — sets the pos-embed
                      and SAGA gate grids that get interpolated at run time
        expected_sha256  If given, the checkpoint's sha256 must match or the
                      constructor raises (never fine-tune from the wrong file)
        model_kwargs  Extra timm.create_model kwargs. Production passes none;
                      tests shrink the model (depth/embed_dim/num_heads) to
                      exercise this exact path at real dense resolutions.

    Attributes set after construction:
        embed_dim, grid_size (training grid), ckpt_sha256
    Attributes set after forward():
        last_patch_grid  (gh, gw) of the input just seen
    """

    def __init__(
        self,
        arch:        str,
        gate:        bool,
        registers:   int,
        ckpt_path:   Optional[str],
        fpn_indices: List[int],
        img_size:    int = 224,
        expected_sha256: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if registers > 0 and gate:
            raise ValueError(
                "registers and the SAGA gate are mutually exclusive variants")
        if not fpn_indices:
            raise ValueError("fpn_indices must be a non-empty list")
        self.fpn_indices = list(fpn_indices)
        self.registers = int(registers)
        self.is_registers = self.registers > 0
        mk = dict(model_kwargs or {})

        if self.is_registers:
            # B6: the timm model is used DIRECTLY — no SAGAViT wrapper.
            import timm
            self.vit = timm.create_model(
                arch,
                pretrained       = False,
                num_classes      = 1000,
                img_size         = img_size,
                reg_tokens       = self.registers,
                dynamic_img_size = True,   # allows dense-size inputs
                **mk,
            )
            self.num_prefix_tokens = self.vit.num_prefix_tokens
            self.grid_size = tuple(self.vit.patch_embed.grid_size)
        else:
            self.vit = build_saga_vit(
                arch        = arch,
                gate        = gate,
                img_size    = img_size,
                patch_size  = None,        # read back from the built model
                num_classes = 1000,
                pretrained  = False,
                **mk,
            )
            self.num_prefix_tokens = 1
            self.grid_size = tuple(self.vit.grid_size)

        n_blocks = len(self.vit.blocks)
        bad = [i for i in self.fpn_indices if not 0 <= i < n_blocks]
        if bad:
            raise ValueError(
                f"fpn_indices {bad} out of range for a {n_blocks}-block model")

        self.embed_dim = self.vit.embed_dim
        self.last_patch_grid = None
        self.ckpt_sha256 = None
        self.ckpt_meta = {}

        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path, expected_sha256)
        else:
            print("  DetectionBackbone: ckpt_path=None — RANDOM INIT "
                  "(tests/smokes only)", flush=True)

    # ── checkpoint ────────────────────────────────────────────────────────
    def _load_checkpoint(self, ckpt_path, expected_sha256):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"backbone checkpoint {ckpt_path} not found")

        sha = file_sha256(ckpt_path)
        if expected_sha256 and sha != expected_sha256:
            raise RuntimeError(
                f"backbone sha256 mismatch for {ckpt_path}:\n"
                f"  expected {expected_sha256}\n  found    {sha}\n"
                f"This is not the checkpoint the matrix refers to — refusing "
                f"to train a dense head on it.")
        self.ckpt_sha256 = sha

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            # legacy pickles (older torch saves); trusted project ckpts only
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state, wrapped = extract_state_dict(ckpt)

        # strict=True: a silent strict=False load is exactly how B6 hid the
        # dropped reg_token for months
        self.vit.load_state_dict(state, strict=True)
        if wrapped and isinstance(ckpt, dict):
            self.ckpt_meta = {k: v for k, v in ckpt.items()
                              if isinstance(v, (int, float, str, bool))}
        print(f"  backbone loaded (strict) from {ckpt_path}\n"
              f"    sha256 {sha[:12]}...  ckpt top1="
              f"{self.ckpt_meta.get('top1', '?')}", flush=True)

    # ── freezing ──────────────────────────────────────────────────────────
    def freeze(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.vit.parameters():
            p.requires_grad = True

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return one [B, n_patches, embed_dim] tensor per fpn index, with
        every prefix token (CLS and, for the registers variant, the register
        tokens) removed. Sets `last_patch_grid` to the input's patch grid."""
        H, W = int(x.shape[-2]), int(x.shape[-1])

        if self.is_registers:
            # timm's own path: strips num_prefix_tokens, resamples pos-embed.
            # timm appends ONE intermediate per matching block in block
            # order, so a repeated or unsorted fpn_indices list would come
            # back with the wrong length/order (SAGAViT's own
            # forward_intermediates honours the request literally). Ask for
            # the unique indices and re-expand, so both variants answer the
            # same fpn_indices identically.
            uniq = sorted(set(self.fpn_indices))
            outs = self.vit.forward_intermediates(
                x, indices=uniq, norm=False,
                output_fmt="NLC", intermediates_only=True)
            if len(outs) != len(uniq):
                raise RuntimeError(
                    f"timm returned {len(outs)} intermediates for "
                    f"{len(uniq)} requested block indices {uniq}")
            by_index = dict(zip(uniq, outs))
            intermediates = [by_index[i] for i in self.fpn_indices]
            self.last_patch_grid = tuple(
                int(v) for v in self.vit.patch_embed.dynamic_feat_size((H, W)))
        else:
            intermediates, _ = self.vit.forward_intermediates(
                x, indices=self.fpn_indices)
            self.last_patch_grid = tuple(int(v)
                                         for v in self.vit.last_patch_grid)

        if len(intermediates) != len(self.fpn_indices):
            raise RuntimeError(
                f"backbone returned {len(intermediates)} feature maps for "
                f"fpn_indices {self.fpn_indices} — the neck expects one per "
                f"index")

        gh, gw = self.last_patch_grid
        for i, feat in zip(self.fpn_indices, intermediates):
            if feat.shape[1] != gh * gw:
                raise RuntimeError(
                    f"block {i} returned {feat.shape[1]} tokens but the patch "
                    f"grid is {gh}x{gw}={gh * gw} — prefix tokens leaked into "
                    f"the feature sequence (bug B6 class)")
        return list(intermediates)

    def get_gate_maps(self):
        """SAGA gate maps, or {} for baseline/registers (no gates)."""
        if hasattr(self.vit, "get_gate_maps"):
            return self.vit.get_gate_maps()
        return {}
