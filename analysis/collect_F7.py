#!/usr/bin/env python3
"""
analysis/collect_F7.py
======================
TASK-09 Phase C.3, selection half: freeze WHICH images F7 shows.

F7 has two halves:
  * the ADE20K half is already local — every segmentation run committed
    `preds_fixed20/<stem>_{pred,gt}.png` and `<stem>_img.jpg` for the 20
    frozen probe images, so nothing has to be selected or fetched;
  * the COCO half needs the val2017 JPEGs, which live ONLY on the HPC.
    `detections_val.json` made re-inference unnecessary (TASK-09 A3) but it
    carries boxes, not pixels.

So this script picks the two COCO images DETERMINISTICALLY from committed
files alone and writes the choice to results/figures_data/F7_coco_selection.json.
`tools/export_f7_crops.py` then exports just those two crops on the HPC, and
`plotting/plot_F7.py` renders them. Choosing here rather than there means
the selection is committed and reviewable BEFORE anyone sees the crops.

SELECTION RULE (frozen; recorded in the output file):
  small detection := box area < 32^2 px in ORIGINAL image coordinates,
                     which is COCO's own "small" threshold and the frame
                     detections_val.json is written in;
  eligible image  := every variant has >= MIN_SMALL small detections above
                     SCORE_MIN, so no panel can come out empty;
  ranking         := |n_small(saga) - n_small(baseline)|, descending — the
                     images where the two disagree most about small objects,
                     which is what the panel is for;
  ties            := broken by image_id ascending, so the choice is a pure
                     function of the committed inputs.

    python analysis/collect_F7.py [--det-root results/detection]
        [--out results/figures_data/F7_coco_selection.json]
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def show(path: Path):
    """Repo-relative when the path is inside the repo, absolute otherwise —
    a --out outside the tree (tests, scratch) must not crash the tool."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path

SMALL_AREA = 32.0 * 32.0     # COCO's small-object threshold
SCORE_MIN = 0.30             # displayable confidence
MIN_SMALL = 2                # per variant, so no panel is empty
N_IMAGES = 2                 # TASK-09 C.3: "2 COCO small-object crops"
PAD_FRAC = 0.35              # crop padding around the small-box envelope
VARIANTS = ["baseline", "registers", "saga"]


def load_dets(det_root: Path, run_id: str):
    path = det_root / run_id / "detections_val.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path) as f:
        return json.load(f)


def small_by_image(dets):
    """{image_id: [det, ...]} keeping only small, confident detections."""
    out = {}
    for d in dets:
        w, h = float(d["bbox"][2]), float(d["bbox"][3])
        if w * h >= SMALL_AREA or float(d["score"]) < SCORE_MIN:
            continue
        out.setdefault(int(d["image_id"]), []).append(d)
    return out


def envelope(boxes):
    """[x, y, w, h] covering every box, padded by PAD_FRAC (unclipped —
    the exporter clips to the real image size, which it alone knows)."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    pw, ph = (x1 - x0) * PAD_FRAC, (y1 - y0) * PAD_FRAC
    return [round(x0 - pw, 2), round(y0 - ph, 2),
            round((x1 - x0) + 2 * pw, 2), round((y1 - y0) + 2 * ph, 2)]


def main():
    p = argparse.ArgumentParser("TASK-09 Phase C: freeze the F7 COCO crops")
    p.add_argument("--det-root", default="results/detection")
    p.add_argument("--matrix", default="configs/dense_matrix.yaml")
    p.add_argument("--out", default="results/figures_data/F7_coco_selection.json")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing selection (it is write-once: "
                        "the committed choice is what F7 shows)")
    args = p.parse_args()

    def rel(x):
        q = Path(x)
        return q if q.is_absolute() else REPO / q

    out = rel(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists — the F7 selection is "
                         f"write-once (pass --force to re-pick deliberately)")

    det_root = rel(args.det_root)
    runs = {v: f"det_vitb_{v}_s1" for v in VARIANTS}
    small = {v: small_by_image(load_dets(det_root, r))
             for v, r in runs.items()}

    eligible = [i for i in small["baseline"]
                if all(len(small[v].get(i, [])) >= MIN_SMALL
                       for v in VARIANTS)]
    ranked = sorted(
        eligible,
        key=lambda i: (-abs(len(small["saga"][i]) - len(small["baseline"][i])),
                       i))
    chosen = ranked[:N_IMAGES]
    if len(chosen) < N_IMAGES:
        raise SystemExit(f"only {len(chosen)} eligible images — loosen "
                         f"MIN_SMALL/SCORE_MIN deliberately, do not pick "
                         f"by hand")

    images = []
    for img_id in chosen:
        boxes_all = [d["bbox"] for v in VARIANTS for d in small[v][img_id]]
        images.append({
            "image_id": img_id,
            "file_name": f"{img_id:012d}.jpg",
            "crop_xywh": envelope(boxes_all),
            "n_small": {v: len(small[v][img_id]) for v in VARIANTS},
            "detections": {v: sorted(small[v][img_id],
                                     key=lambda d: -float(d["score"]))
                           for v in VARIANTS},
        })

    spec = {
        "figure": "F7",
        "half": "coco",
        "rule": {
            "small_area_max_px2": SMALL_AREA,
            "score_min": SCORE_MIN,
            "min_small_per_variant": MIN_SMALL,
            "n_images": N_IMAGES,
            "pad_frac": PAD_FRAC,
            "ranking": "abs(n_small[saga] - n_small[baseline]) desc, "
                       "then image_id asc",
            "coordinate_frame": "original val2017 pixels (detections_val.json "
                                "is written in that frame)",
        },
        "source_runs": runs,
        "n_eligible": len(eligible),
        "images": images,
        "note": "Selected from committed detections alone; the JPEGs are "
                "exported on the HPC by tools/export_f7_crops.py.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8",
                   newline="\n")
    tmp.replace(out)
    print(f"wrote {show(out)}")
    for im in images:
        print(f"  image {im['image_id']:>12d}  n_small="
              f"{ {k: v for k, v in im['n_small'].items()} }  "
              f"crop={im['crop_xywh']}")
    print(f"  ({len(eligible)} images were eligible)")


if __name__ == "__main__":
    main()
