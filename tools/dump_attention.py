#!/usr/bin/env python3
"""
tools/dump_attention.py
=======================
Dump the attention a checkpoint pays to the frozen probe set.

    python tools/dump_attention.py \
        --run-id e2r_vits_mixup_saga_s1 --arch vit_small --variant saga \
        --ckpt results/runs/e2r_vits_mixup_saga_s1/ckpt/last.pth \
        --data $STAGE_DIR --groups curated,random200

One npz per (run, group, image) at
`results/runs/<run_id>/attn/probe_<group>_<image_id>.npz`. These are LARGE and
git-ignored (`.gitignore: results/**/attn/`) — they stay on the HPC and must
not be deleted: `analysis/collect_F1.py` and every later figure read them
instead of re-running inference.

Per image the npz carries:
  cls_attn_mean     [L, P]      CLS -> patch, mean over heads, all blocks, fp16
  cls_attn_heads    [L, H, P]   CLS -> patch, per head, all blocks, fp16
  cls_prefix_mass   [L]         CLS -> prefix tokens (so patch+prefix == 1)
  full_attn_last4   [4, H, T, T]  full maps for the last 4 blocks, fp16
  last4_blocks      [4]         which block indices those are
  token_norms       [L, T]      per-block L2 token norms, fp32
  gate_sigmoid      [L, P]      SAGA only: sigmoid(phi), mean over heads
  provenance        JSON string: ckpt sha256, git sha, arch, variant, probe sha

PREFIX TOKENS ARE NEVER ASSUMED. The count comes from the model object via
`saga.metrics.infer_num_prefix_tokens` (1 for baseline/SAGA, 5 for timm
registers), so baseline / registers / SAGA / a TASK-10 TTR-patched model all
go through this code unchanged.

TTR (TASK-10) models are built through `--model-builder module:function`; the
hook receives the parsed args and returns an `nn.Module`. Nothing about
TASK-10's interface is assumed here.

Idempotent: an npz whose recorded ckpt sha256, probe-group sha256 and schema
all match is skipped. Writes are tmp+rename, so a killed job never leaves a
file that looks complete.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.gate_structure import head_mean_gate            # noqa: E402
from saga.attn_extract import capture_attention               # noqa: E402
from saga.metrics import infer_num_prefix_tokens              # noqa: E402
from saga.run_registry import file_sha256, git_sha            # noqa: E402
from tools.dense_runtime import atomic_npz_save               # noqa: E402
from tools.model_factory import build_model, load_checkpoint  # noqa: E402

SCHEMA_VERSION = 1
N_FULL_BLOCKS = 4          # how many trailing blocks keep their full map


def sanitize(image_id: str) -> str:
    """`n02123045/ILSVRC2012_val_00003014` -> a legal single path component."""
    return image_id.replace("/", "_").replace("\\", "_")


def load_probe_group(probe_json, group: str):
    with open(probe_json, encoding="utf-8") as f:
        doc = json.load(f)
    g = doc.get("groups", {}).get(group)
    if g is None:
        raise KeyError(f"probe set has no group {group!r}")
    if g.get("status") != "frozen":
        raise ValueError(
            f"probe group {group!r} is {g.get('status')!r}, not frozen — "
            f"build it before dumping attention against it")
    return g


def resolve_root(group: dict, imagenet_root, coco_root) -> Path:
    ds = group.get("dataset")
    if ds == "imagenet-1k":
        if not imagenet_root:
            raise ValueError("--data is required for the ImageNet groups")
        return Path(imagenet_root)
    if ds == "coco":
        if not coco_root:
            raise ValueError("--coco-root is required for the boxes20 group")
        return Path(coco_root)
    raise ValueError(f"unknown probe dataset {ds!r}")


def build_the_model(args):
    """model_factory by default; a TASK-10 hook when --model-builder is given."""
    if args.model_builder:
        mod_name, _, fn_name = args.model_builder.partition(":")
        if not fn_name:
            raise ValueError("--model-builder must be 'module:function'")
        import importlib
        fn = getattr(importlib.import_module(mod_name), fn_name)
        model = fn(args)
        if not hasattr(model, "blocks"):
            raise TypeError(
                f"{args.model_builder} returned {type(model).__name__}, which "
                f"exposes no .blocks — capture_attention needs it")
        return model, f"{args.model_builder}"

    model = build_model(args.arch, args.variant, num_classes=args.num_classes)
    load_checkpoint(model, args.ckpt)     # strict=True: head shape included
    return model, "tools.model_factory.build_model"


def gate_maps(model, n_patches: int):
    """sigmoid(phi) per layer, head-mean — via the TASK-06B convention.
    Returns [L, P] or None when the model has no gate."""
    phis = []
    for blk in model.blocks:
        gate = getattr(getattr(blk, "attn", None), "gate", None)
        phi = getattr(gate, "phi", None) if gate is not None else None
        if phi is None:
            return None
        phis.append(phi.detach().float().cpu().numpy())
    arr = np.stack(phis, axis=0)                 # [L, H, N]
    if arr.ndim != 3 or arr.shape[-1] != n_patches:
        raise ValueError(
            f"gate phi has shape {arr.shape}, expected [L, H, {n_patches}]")
    return head_mean_gate(arr).astype(np.float32)   # sigmoid FIRST, then mean


def dump_one(store, model, n_prefix, n_blocks):
    """Turn one captured forward into the per-image arrays (batch dim kept)."""
    L = n_blocks
    first = store[0]
    B, H, T, _ = first.shape
    P = T - n_prefix

    cls_heads = np.empty((B, L, H, P), dtype=np.float16)
    cls_mean = np.empty((B, L, P), dtype=np.float16)
    prefix_mass = np.empty((B, L), dtype=np.float32)
    for li in range(L):
        a = store[li]                              # [B, H, T, T]
        cls_row = a[:, :, 0, :].float()            # CLS query -> everything
        patch = cls_row[..., n_prefix:]            # [B, H, P]
        cls_heads[:, li] = patch.cpu().numpy().astype(np.float16)
        cls_mean[:, li] = patch.mean(1).cpu().numpy().astype(np.float16)
        prefix_mass[:, li] = cls_row[..., :n_prefix].sum(-1).mean(1).cpu().numpy()

    last4 = sorted(range(L))[-N_FULL_BLOCKS:]
    full = np.stack(
        [store[li].float().cpu().numpy().astype(np.float16) for li in last4],
        axis=1)                                    # [B, 4, H, T, T]
    return cls_mean, cls_heads, prefix_mass, full, np.array(last4), P


def token_norms_for(feats):
    """feats: list of [B, T, C] per block -> [B, L, T] fp32."""
    return np.stack([f.float().norm(dim=-1).cpu().numpy() for f in feats],
                    axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Dump probe-set attention maps.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--arch", default=None,
                   help="vit_small/vit_base/... (not needed with "
                        "--model-builder)")
    p.add_argument("--variant", default=None, choices=(None, "baseline",
                                                       "registers", "saga"))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--probe", default="results/probe/probe_set.json")
    p.add_argument("--groups", default="curated,random200")
    p.add_argument("--data", metavar="ROOT", help="ImageNet root with val/")
    p.add_argument("--coco-root", metavar="ROOT",
                   help="COCO root containing val2017/")
    p.add_argument("--out-root", default="results/runs")
    p.add_argument("--attn-root", default=None,
                   help="where the attn/ trees go (default: --out-root). The "
                        "dumps are ~3 GB for the planned five models and must "
                        "never be deleted, so point this at a bulk filesystem "
                        "if the repo one is tight (TASK-09 --ckpt_root "
                        "precedent).")
    p.add_argument("--model-builder", default=None,
                   help="TASK-10 hook, 'module:function'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000,
                   help="classifier head width; must match the checkpoint "
                        "(load is strict=True)")
    p.add_argument("--force", action="store_true",
                   help="re-dump even when the skip guard matches")
    args = p.parse_args()

    if not args.model_builder and not (args.arch and args.variant and args.ckpt):
        p.error("--arch, --variant and --ckpt are required without "
                "--model-builder")

    from PIL import Image
    from tools.eval import build_val_transform
    transform = build_val_transform(args.img_size)

    model, builder = build_the_model(args)
    model.eval().to(args.device)
    n_prefix = infer_num_prefix_tokens(model)
    n_blocks = len(model.blocks)
    ckpt_sha = file_sha256(args.ckpt) if args.ckpt else None
    repo_sha = git_sha()        # one subprocess, not one per image

    attn_dir = Path(args.attn_root or args.out_root) / args.run_id / "attn"
    attn_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for group_name in [g.strip() for g in args.groups.split(",") if g.strip()]:
        group = load_probe_group(args.probe, group_name)
        root = resolve_root(group, args.data, args.coco_root)
        probe_sha = group["sha256"]

        todo = []
        for item in group["items"]:
            out = attn_dir / f"probe_{group_name}_{sanitize(item['image_id'])}.npz"
            if out.exists() and not args.force:
                try:
                    with np.load(out, allow_pickle=False) as z:
                        prov = json.loads(str(z["provenance"]))
                    if (prov.get("ckpt_sha256") == ckpt_sha
                            and prov.get("probe_sha256") == probe_sha
                            and prov.get("schema_version") == SCHEMA_VERSION):
                        skipped += 1
                        continue
                except Exception:
                    pass    # unreadable/partial -> redo it
            todo.append((item, out))

        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            batch = torch.stack([
                transform(Image.open(root / it["path"]).convert("RGB"))
                for it, _ in chunk]).to(args.device)

            feats = []
            hooks = [blk.register_forward_hook(
                lambda m, i, o, s=feats: s.append(
                    o[0] if isinstance(o, tuple) else o))
                for blk in model.blocks]
            try:
                with torch.no_grad(), capture_attention(model) as store:
                    model(batch)
                    cls_mean, cls_heads, prefix_mass, full, last4, P = dump_one(
                        store, model, n_prefix, n_blocks)
                    norms = token_norms_for(feats)
            finally:
                for h in hooks:
                    h.remove()

            grid = int(round(P ** 0.5))
            if grid * grid != P:
                raise ValueError(f"{P} patch tokens is not a square grid")
            # the gate is a parameter, not a function of the input: compute it
            # once per batch shape, not once per image
            gates = gate_maps(model, P)

            for bi, (item, out) in enumerate(chunk):
                prov = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": args.run_id,
                    "group": group_name,
                    "image_id": item["image_id"],
                    "path": item["path"],
                    "dataset": group["dataset"],
                    "arch": args.arch,
                    "variant": args.variant,
                    "builder": builder,
                    "ckpt": str(args.ckpt) if args.ckpt else None,
                    "ckpt_sha256": ckpt_sha,
                    "probe_sha256": probe_sha,
                    "git_sha": repo_sha,
                    "num_prefix_tokens": int(n_prefix),
                    "grid_side": int(grid),
                    "img_size": int(args.img_size),
                    "written": datetime.now(timezone.utc).isoformat(),
                }
                arrays = {
                    "cls_attn_mean": cls_mean[bi],
                    "cls_attn_heads": cls_heads[bi],
                    "cls_prefix_mass": prefix_mass[bi],
                    "full_attn_last4": full[bi],
                    "last4_blocks": last4,
                    "token_norms": norms[bi],
                    "provenance": np.array(json.dumps(prov, sort_keys=True)),
                }
                if gates is not None:
                    arrays["gate_sigmoid"] = gates

                atomic_npz_save(out, **arrays)
                written += 1

        print(f"{group_name:10s} {len(group['items']):4d} images "
              f"-> {attn_dir}")

    print(f"done: {written} written, {skipped} skipped (guard matched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
