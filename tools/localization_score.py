#!/usr/bin/env python3
"""
tools/localization_score.py
===========================
Turn the probe-set attention dumps into the small, committed table that
Figure 1 and every later localization claim read:
`results/figures_data/F1_localization.csv`, one row per
(run_id, group, image, metric, value).

    python tools/localization_score.py --runs e2r_vits_mixup_baseline_s1,... \
        --groups curated,random200,boxes20

METRICS
  boxes20 (needs GT boxes)
    inbox_mass       CLS->patch attention mass (LAST block, mean over heads,
                     renormalized over patches) falling inside the GT boxes,
                     weighted by each patch's fractional coverage.
    pointing_hit     1.0 if the argmax patch's CENTRE lies inside a box.
    box_area_frac    the coverage fraction itself — i.e. what `inbox_mass`
                     would be for UNIFORM attention. inbox_mass is
                     meaningless without it (TASK-07's lesson: a statistic
                     needs its null), so it is written as its own row.

  random200 (no boxes needed)
    ring1_mass          attention mass on the ring-1 address positions.
    uniform_ring1_mass  ring 1's area fraction — the same null, per image.
    attn_entropy_bits   Shannon entropy of the patch distribution, in bits
                        (log2(P) = maximum = perfectly diffuse attention).

THE RING IS NOT REDEFINED HERE. Ring membership is derived from TASK-07's
`analysis.address_analysis.border_rings` by probing it with one-hot maps, so
the two can never drift apart; `tests/test_task11_probe.py` pins the
agreement. Ring 0 = the outermost row/col, ring 1 = one patch inside — the
ring TASK-07 found the sink address sits on.

BOX FRAME. TASK-09 lost ~3 GPU-days to boxes scored in the wrong coordinate
frame, so nothing here is assumed: `map_boxes_to_input_frame()` reproduces
`tools.eval.build_val_transform`'s Resize(shorter side -> 256) + CenterCrop(224)
arithmetic exactly (torchvision's own truncation and rounding), and a test
compares its predicted geometry against the REAL transform rather than
against a re-derivation of it.
"""

import argparse
import csv
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.address_analysis import border_rings            # noqa: E402
from saga.run_registry import git_sha                         # noqa: E402
from tools.dense_runtime import atomic_write_text             # noqa: E402
from tools.dump_attention import sanitize                     # noqa: E402

CSV_FIELDS = ("run_id", "group", "image_id", "metric", "value")
DEFAULT_CROP_PCT = 0.875     # timm DEFAULT_CROP_PCT, as build_val_transform uses
SUPERSAMPLE = 4              # sub-pixel resolution for fractional coverage


# ── ring membership, derived from TASK-07's border_rings ─────────────────────

@lru_cache(maxsize=8)
def ring_index_map(n_positions: int) -> tuple:
    """Ring index of every grid position, obtained by probing
    `border_rings` with one-hot maps. Imported definition, not a copy:
    if TASK-07's ring convention ever changed, this follows it."""
    idx = []
    for p in range(n_positions):
        onehot = np.zeros(n_positions, dtype=np.float64)
        onehot[p] = 1.0
        idx.append(int(np.argmax(border_rings(onehot))))
    return tuple(idx)


def ring_mask(n_positions: int, ring: int) -> np.ndarray:
    return np.asarray(ring_index_map(n_positions)) == ring


# ── the val transform's geometry, reproduced exactly ─────────────────────────

def resized_size(width: int, height: int, short_side: int) -> tuple:
    """torchvision Resize(size=int): the SHORTER side becomes `short_side`,
    the longer is scaled and TRUNCATED (int(), not round())."""
    if width <= height:
        return short_side, int(short_side * height / width)
    return int(short_side * width / height), short_side


def crop_origin(new_w: int, new_h: int, crop: int) -> tuple:
    """torchvision CenterCrop: int(round((size - crop) / 2.0))."""
    return (int(round((new_w - crop) / 2.0)), int(round((new_h - crop) / 2.0)))


def map_boxes_to_input_frame(boxes_xywh, width, height, img_size=224,
                             crop_pct=DEFAULT_CROP_PCT):
    """COCO xywh in ORIGINAL pixels -> xyxy in the model's `img_size` frame.
    Boxes are clipped to the crop; ones that fall entirely outside vanish."""
    short_side = int(img_size / crop_pct)
    new_w, new_h = resized_size(width, height, short_side)
    left, top = crop_origin(new_w, new_h, img_size)
    sx, sy = new_w / width, new_h / height

    out = []
    for x, y, w, h in boxes_xywh:
        x1 = x * sx - left
        y1 = y * sy - top
        x2 = (x + w) * sx - left
        y2 = (y + h) * sy - top
        x1, x2 = max(0.0, min(x1, img_size)), max(0.0, min(x2, img_size))
        y1, y2 = max(0.0, min(y1, img_size)), max(0.0, min(y2, img_size))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def patch_coverage(boxes_xyxy, grid_side: int, img_size: int = 224,
                   supersample: int = SUPERSAMPLE) -> np.ndarray:
    """Fraction of each patch covered by the UNION of the boxes, [P].
    The union is rasterized at `supersample`x resolution, so overlapping
    boxes are never double-counted."""
    n = img_size * supersample
    mask = np.zeros((n, n), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        c0, c1 = int(np.floor(x1 * supersample)), int(np.ceil(x2 * supersample))
        r0, r1 = int(np.floor(y1 * supersample)), int(np.ceil(y2 * supersample))
        mask[max(0, r0):min(n, r1), max(0, c0):min(n, c1)] = True
    cell = n // grid_side
    if cell * grid_side != n:
        raise ValueError(f"{n} px does not split into {grid_side} patches")
    blocks = mask.reshape(grid_side, cell, grid_side, cell)
    return blocks.mean(axis=(1, 3)).reshape(-1).astype(np.float64)


def union_mask_contains(boxes_xyxy, x: float, y: float) -> bool:
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in boxes_xyxy)


# ── metrics ──────────────────────────────────────────────────────────────────

def patch_distribution(cls_attn_mean: np.ndarray) -> np.ndarray:
    """Last block, already head-meaned; renormalized over patches so the
    prefix mass (which differs between baseline and registers) cannot make
    the models incomparable."""
    row = np.asarray(cls_attn_mean, dtype=np.float64)[-1]
    total = row.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("last-block CLS->patch row has non-positive mass")
    return row / total


def entropy_bits(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def metrics_for_image(attn_npz, item, group_name, img_size=224):
    with np.load(attn_npz, allow_pickle=False) as z:
        cls_mean = z["cls_attn_mean"]
        prov = json.loads(str(z["provenance"]))
    p = patch_distribution(cls_mean)
    n_positions = p.size
    grid = int(prov.get("grid_side") or round(n_positions ** 0.5))
    out = {}

    if group_name == "boxes20":
        boxes = map_boxes_to_input_frame(
            item["boxes_xywh"], item["width"], item["height"], img_size)
        cov = patch_coverage(boxes, grid, img_size)
        out["inbox_mass"] = float((p * cov).sum())
        out["box_area_frac"] = float(cov.mean())
        arg = int(np.argmax(p))
        cx = (arg % grid + 0.5) * (img_size / grid)
        cy = (arg // grid + 0.5) * (img_size / grid)
        out["pointing_hit"] = float(union_mask_contains(boxes, cx, cy))
    else:
        m = ring_mask(n_positions, 1)
        out["ring1_mass"] = float(p[m].sum())
        out["uniform_ring1_mass"] = float(m.mean())
        out["attn_entropy_bits"] = entropy_bits(p)
    return out, prov


def main():
    p = argparse.ArgumentParser(
        description="Localization statistics from the probe attention dumps.")
    p.add_argument("--runs", required=True,
                   help="comma-separated run_ids under --out-root")
    p.add_argument("--groups", default="random200,boxes20")
    p.add_argument("--probe", default="results/probe/probe_set.json")
    p.add_argument("--out-root", default="results/runs")
    p.add_argument("--attn-root", default=None,
                   help="where the attn/ trees live (default: --out-root); "
                        "must match tools/dump_attention.py")
    p.add_argument("--out", default="results/figures_data/F1_localization.csv")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--allow-missing", action="store_true",
                   help="report absent runs/dumps instead of failing (a model "
                        "that was never dumped is ABSENT, never a zero)")
    args = p.parse_args()

    with open(args.probe, encoding="utf-8") as f:
        probe = json.load(f)

    rows, absent, provenance = [], [], {}
    for run_id in [r.strip() for r in args.runs.split(",") if r.strip()]:
        attn_dir = Path(args.attn_root or args.out_root) / run_id / "attn"
        for group_name in [g.strip() for g in args.groups.split(",") if g.strip()]:
            group = probe.get("groups", {}).get(group_name)
            if not group or group.get("status") != "frozen":
                absent.append(f"{run_id}/{group_name}: group not frozen")
                continue
            for item in group["items"]:
                npz = attn_dir / (f"probe_{group_name}_"
                                  f"{sanitize(item['image_id'])}.npz")
                if not npz.exists():
                    absent.append(f"{run_id}/{group_name}/{item['image_id']}")
                    continue
                vals, prov = metrics_for_image(
                    npz, item, group_name, args.img_size)
                provenance.setdefault(run_id, {
                    "ckpt_sha256": prov.get("ckpt_sha256"),
                    "arch": prov.get("arch"),
                    "variant": prov.get("variant"),
                    "builder": prov.get("builder"),
                    "dump_git_sha": prov.get("git_sha"),
                    "num_prefix_tokens": prov.get("num_prefix_tokens"),
                })
                for metric, value in sorted(vals.items()):
                    rows.append({"run_id": run_id, "group": group_name,
                                 "image_id": item["image_id"],
                                 "metric": metric, "value": f"{value:.10g}"})

    if absent and not args.allow_missing:
        head = "\n  ".join(absent[:10])
        raise SystemExit(
            f"{len(absent)} missing dump(s); pass --allow-missing to record "
            f"them as ABSENT instead:\n  {head}")

    out = Path(args.out)
    rows.sort(key=lambda r: (r["run_id"], r["group"], r["image_id"], r["metric"]))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    atomic_write_text(buf.getvalue(), out)

    meta = out.with_name(out.stem + "_meta.json")
    atomic_write_text(
        json.dumps({"git_sha": git_sha(), "img_size": args.img_size,
                    "probe": str(args.probe), "runs": provenance,
                    "absent": absent, "n_rows": len(rows)},
                   indent=2, sort_keys=True) + "\n", meta)

    print(f"wrote {out}: {len(rows)} rows, {len(provenance)} runs, "
          f"{len(absent)} absent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
