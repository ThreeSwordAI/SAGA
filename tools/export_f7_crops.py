#!/usr/bin/env python3
"""
tools/export_f7_crops.py
========================
TASK-09 Phase C.3, export half: fetch the two COCO crops F7 needs.

COCO val2017 lives only on the HPC, so the crops frozen by
`analysis/collect_F7.py` are exported here and travel back through git as
two small JPEGs plus the ground-truth boxes for those images.

This reads the two members it needs STRAIGHT OUT OF THE ZIPS — no staging,
no GPU, seconds on a login node. The zip paths are the ones the committed
detection/scripts/env_alex.sh declares (COCO_VAL_ZIP / COCO_ANN_ZIP); they
are read from the environment when set, so nothing here is invented.

    # on the HPC login node, env activated
    python tools/export_f7_crops.py

Writes results/figures_data/f7_coco/:
    <image_id>_crop.jpg     the padded crop, original pixels
    crops.json              crop geometry + the GT small boxes, in crop
                            coordinates, so the plot script needs no dataset
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def show(path: Path):
    """Repo-relative when the path is inside the repo, absolute otherwise —
    a --out outside the tree (tests, scratch) must not crash the tool."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path

DEFAULT_VAL_ZIP = "/home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip"
DEFAULT_ANN_ZIP = ("/home/woody/iwi5/iwi5359h/Data/COCO/"
                   "annotations_trainval2017.zip")
SMALL_AREA = 32.0 * 32.0     # must match analysis/collect_F7.py


def clip_crop(crop, width, height):
    x, y, w, h = crop
    x0 = max(0.0, min(float(x), width - 1.0))
    y0 = max(0.0, min(float(y), height - 1.0))
    x1 = max(x0 + 1.0, min(float(x) + float(w), float(width)))
    y1 = max(y0 + 1.0, min(float(y) + float(h), float(height)))
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    p = argparse.ArgumentParser("export the F7 COCO crops (HPC)")
    p.add_argument("--selection",
                   default="results/figures_data/F7_coco_selection.json")
    p.add_argument("--val-zip", default=os.environ.get("COCO_VAL_ZIP",
                                                       DEFAULT_VAL_ZIP))
    p.add_argument("--ann-zip", default=os.environ.get("COCO_ANN_ZIP",
                                                       DEFAULT_ANN_ZIP))
    p.add_argument("--out-dir", default="results/figures_data/f7_coco")
    p.add_argument("--quality", type=int, default=92)
    args = p.parse_args()

    from PIL import Image

    def rel(x):
        q = Path(x)
        return q if q.is_absolute() else REPO / q

    sel = json.loads(rel(args.selection).read_text(encoding="utf-8"))
    out_dir = rel(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = {im["image_id"]: im for im in sel["images"]}
    print(f"exporting {len(wanted)} crops for {sorted(wanted)}")

    # ── ground truth for just these images ────────────────────────────────
    with zipfile.ZipFile(args.ann_zip) as z:
        name = next(n for n in z.namelist()
                    if n.endswith("instances_val2017.json"))
        print(f"  reading {name} from {args.ann_zip}")
        ann = json.loads(z.read(name).decode("utf-8"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    sizes = {im["id"]: (im["width"], im["height"]) for im in ann["images"]
             if im["id"] in wanted}
    gt = {i: [] for i in wanted}
    for a in ann["annotations"]:
        if a["image_id"] in wanted and not a.get("iscrowd", 0):
            gt[a["image_id"]].append({
                "bbox": [round(float(v), 2) for v in a["bbox"]],
                "category_id": a["category_id"],
                "name": cats.get(a["category_id"], "MISSING"),
                "area": round(float(a["area"]), 2),
                "is_small": float(a["area"]) < SMALL_AREA,
            })

    # ── the crops ─────────────────────────────────────────────────────────
    records = []
    with zipfile.ZipFile(args.val_zip) as z:
        members = {Path(n).name: n for n in z.namelist()
                   if n.lower().endswith(".jpg")}
        for img_id, spec in sorted(wanted.items()):
            fn = spec["file_name"]
            if fn not in members:
                raise SystemExit(f"{fn} not in {args.val_zip}")
            img = Image.open(io.BytesIO(z.read(members[fn]))).convert("RGB")
            w, h = img.size
            if img_id in sizes and sizes[img_id] != (w, h):
                raise SystemExit(
                    f"image {img_id}: annotation says {sizes[img_id]}, JPEG "
                    f"is {(w, h)} — refusing to export mismatched geometry")
            cx, cy, cw, ch = clip_crop(spec["crop_xywh"], w, h)
            crop = img.crop((int(cx), int(cy),
                             int(cx + cw), int(cy + ch)))
            path = out_dir / f"{img_id}_crop.jpg"
            crop.save(path, format="JPEG", quality=args.quality)

            def to_crop(b):
                return [round(b[0] - cx, 2), round(b[1] - cy, 2),
                        round(b[2], 2), round(b[3], 2)]

            records.append({
                "image_id": img_id,
                "file_name": fn,
                "image_size_wh": [w, h],
                "crop_xywh_original": [round(v, 2) for v in (cx, cy, cw, ch)],
                "crop_size_wh": list(crop.size),
                "crop_file": path.name,
                "n_small": spec["n_small"],
                "detections_in_crop": {
                    v: [{**d, "bbox_crop": to_crop(d["bbox"])}
                        for d in spec["detections"][v]]
                    for v in spec["detections"]},
                "gt_in_crop": [{**g, "bbox_crop": to_crop(g["bbox"])}
                               for g in gt[img_id]],
            })
            print(f"  {img_id}: image {w}x{h} -> crop {crop.size} "
                  f"({path.stat().st_size / 1024:.0f} KiB), "
                  f"{len(gt[img_id])} GT boxes "
                  f"({sum(1 for g in gt[img_id] if g['is_small'])} small)")

    payload = {
        "figure": "F7",
        "half": "coco",
        "selection_rule": sel["rule"],
        "source_runs": sel["source_runs"],
        "val_zip": args.val_zip,
        "ann_zip": args.ann_zip,
        "images": records,
    }
    out = out_dir / "crops.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8",
                   newline="\n")
    tmp.replace(out)
    print(f"wrote {show(out)}")
    print("\ncommit these and push:")
    print("  git add -f results/figures_data/f7_coco/")
    print("  git commit -m '[TASK-09] F7 COCO crops (phase C export)'")


if __name__ == "__main__":
    main()
