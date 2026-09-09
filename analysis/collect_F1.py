#!/usr/bin/env python3
"""
analysis/collect_F1.py
======================
Extract ONLY what Figure 1 draws from the (large, git-ignored, HPC-resident)
attention dumps into one small committed archive:
`results/figures_data/F1_teaser.npz`.

    python analysis/collect_F1.py --runs e2r_vits_mixup_baseline_s1,... \
        --data $STAGE_DIR

This runs on the HPC, because the `attn/*.npz` dumps never leave it. Phase C
plots from the archive alone — no inference, no dataset, no checkpoints.

Contents (all 12 curated images, so Phase C can choose its 3 without another
HPC round trip):
  thumb/<image_id>            [S, S, 3] uint8   the model's actual input crop
  attn/<run_id>/<image_id>    [grid, grid] f32  last-block CLS->patch, head-mean,
                                                renormalized over patches
  gate/<run_id>/<layer>       [grid, grid] f32  sigmoid(phi) head-mean, SAGA only
  sink/<run_id>               [grid, grid] f32  freq_canon from TASK-07's
                                                <stem>_addr.json, if present
  ring1_mask                  [grid, grid] bool the TASK-07 ring-1 positions
  index                       JSON: run/image order, provenance, what is ABSENT

A model that was never dumped is recorded in `index.absent` and simply has no
arrays — never a zero-filled panel that would render as a real result.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saga.run_registry import git_sha                         # noqa: E402
from tools.dense_runtime import atomic_npz_save               # noqa: E402
from tools.dump_attention import sanitize                     # noqa: E402
from tools.localization_score import (patch_distribution,     # noqa: E402
                                      resized_size, crop_origin, ring_mask)

GATE_LAYERS = (7, 8)      # TASK-07 Q5: where the ViT-S/mixup gate tracks the address
DEFAULT_CROP_PCT = 0.875


def load_thumbnail(root: Path, rel: str, img_size: int) -> np.ndarray:
    """The exact crop the model sees, WITHOUT normalization (uint8 RGB)."""
    from PIL import Image
    img = Image.open(root / rel).convert("RGB")
    short_side = int(img_size / DEFAULT_CROP_PCT)
    new_w, new_h = resized_size(img.width, img.height, short_side)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left, top = crop_origin(new_w, new_h, img_size)
    img = img.crop((left, top, left + img_size, top + img_size))
    return np.asarray(img, dtype=np.uint8)


def find_addr_json(run_id: str, out_root: Path):
    """TASK-07 sink-address file for a run's LAST checkpoint, if it exists."""
    diag = out_root / run_id / "diag"
    if not diag.is_dir():
        return None
    cands = sorted(diag.glob("*_addr.json"))
    last = [c for c in cands if "last" in c.name]
    return (last or cands or [None])[0]


def main():
    p = argparse.ArgumentParser(description="Build F1_teaser.npz for Figure 1.")
    p.add_argument("--runs", required=True, help="comma-separated run_ids")
    p.add_argument("--probe", default="results/probe/probe_set.json")
    p.add_argument("--data", metavar="ROOT", required=True,
                   help="ImageNet root containing val/")
    p.add_argument("--out-root", default="results/runs")
    p.add_argument("--out", default="results/figures_data/F1_teaser.npz")
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    with open(args.probe, encoding="utf-8") as f:
        probe = json.load(f)
    curated = probe.get("groups", {}).get("curated")
    if not curated or curated.get("status") != "frozen":
        raise SystemExit("curated group is not frozen — nothing to collect")

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    out_root = Path(args.out_root)
    arrays, absent, prov = {}, [], {}
    grid = None

    for item in curated["items"]:
        arrays[f"thumb/{item['image_id']}"] = load_thumbnail(
            Path(args.data), item["path"], args.img_size)

    for run_id in runs:
        attn_dir = out_root / run_id / "attn"
        got = 0
        for item in curated["items"]:
            npz = attn_dir / f"probe_curated_{sanitize(item['image_id'])}.npz"
            if not npz.exists():
                absent.append(f"attn/{run_id}/{item['image_id']}")
                continue
            with np.load(npz, allow_pickle=False) as z:
                dist = patch_distribution(z["cls_attn_mean"])
                meta = json.loads(str(z["provenance"]))
                gates = z["gate_sigmoid"] if "gate_sigmoid" in z else None
            g = int(meta.get("grid_side") or round(dist.size ** 0.5))
            grid = grid or g
            if g != grid:
                raise ValueError(
                    f"{run_id}: grid {g} != {grid} — models on different grids "
                    f"cannot share one figure")
            arrays[f"attn/{run_id}/{item['image_id']}"] = (
                dist.reshape(g, g).astype(np.float32))
            prov.setdefault(run_id, {
                "ckpt_sha256": meta.get("ckpt_sha256"),
                "arch": meta.get("arch"), "variant": meta.get("variant"),
                "builder": meta.get("builder"),
                "num_prefix_tokens": meta.get("num_prefix_tokens"),
                "dump_git_sha": meta.get("git_sha"),
            })
            got += 1
            if gates is not None:
                for layer in GATE_LAYERS:
                    if layer < gates.shape[0]:
                        arrays[f"gate/{run_id}/{layer}"] = (
                            np.asarray(gates[layer], dtype=np.float32)
                            .reshape(g, g))
        if got == 0:
            absent.append(f"run/{run_id}: no curated dumps at all")

        addr = find_addr_json(run_id, out_root)
        if addr is not None:
            with open(addr, encoding="utf-8") as f:
                a = json.load(f)
            freq = a.get("freq_canon")
            if freq:
                side = int(round(len(freq) ** 0.5))
                arrays[f"sink/{run_id}"] = (
                    np.asarray(freq, dtype=np.float32).reshape(side, side))
        else:
            absent.append(f"sink/{run_id}: no *_addr.json")

    if grid is None:
        raise SystemExit("no attention dumps found for any run — nothing to do")

    arrays["ring1_mask"] = ring_mask(grid * grid, 1).reshape(grid, grid)
    arrays["index"] = np.array(json.dumps({
        "runs": runs, "grid": grid, "img_size": args.img_size,
        # criterion/basis are figure CAPTIONS, not numbers: a probe set that
        # omits them still yields a usable archive
        "images": [{"image_id": i["image_id"],
                    "criterion": i.get("criterion", ""),
                    "criterion_basis": i.get("criterion_basis", ""),
                    "synset": i.get("synset", ""), "path": i["path"]}
                   for i in curated["items"]],
        "curated_sha256": curated.get("sha256"),
        "gate_layers": list(GATE_LAYERS),
        "provenance": prov, "absent": absent,
        "collect_git_sha": git_sha(),
    }, indent=2, sort_keys=True))

    out = Path(args.out)
    atomic_npz_save(out, **arrays)

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}: {len(arrays)} arrays, {size_mb:.2f} MB, "
          f"grid {grid}x{grid}, {len(absent)} absent")
    for a in absent[:10]:
        print(f"  ABSENT {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
