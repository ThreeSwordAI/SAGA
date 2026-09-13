#!/usr/bin/env python3
"""
analysis/build_dense_tables.py
==============================
TASK-09 Phase C.1/C.2: the two Gate-2 tables, plus their appendix
per-category / per-class tables.

    results/tables/T4_coco.csv              AP,AP50,AP75,AP_S,AP_M,AP_L (+ARs)
    results/tables/T4_coco_per_category.csv 80 COCO categories, with deltas
    results/tables/T5_ade20k.csv            mIoU ss/ms, pixel/mean acc
    results/tables/T5_ade20k_per_class.csv  150 ADE20K classes, with deltas
                                            and the background rows flagged

Every value is read from a committed run artifact — coco_eval_best.json,
miou_ss.json, miou_ms.json, per_class_iou.csv — and nothing is recomputed
from a checkpoint. A run whose recorded backbone sha256 does not match the
candidate configs/dense_matrix.yaml says it used yields MISSING rather than
a number from an unverified checkpoint (the build_ft_tables rule).

UNITS. The detection numbers are AP percentages as COCOeval reports them.
The segmentation JSONs report mIoU/pixel_acc/mean_acc as PERCENT and
iou_per_class as FRACTIONS; per_class_iou.csv's column is named `iou_frac`
for that reason. This script keeps per-class values as fractions and ALSO
emits `*_iou_pts` columns (fraction x 100) so a delta can be read in IoU
points without the reader having to know which convention a column follows.

MISSING is never averaged, interpolated or filled. A run the matrix marks
`submitted: false` is EXCLUDED from these tables rather than rendered as
MISSING: it was prepared and never started, which is a different fact from
"expected but absent", and a MISSING row would read as a failure. A run that
IS expected and whose artifact is absent still yields MISSING.

    python analysis/build_dense_tables.py [--matrix configs/dense_matrix.yaml]
        [--det-root results/detection] [--seg-root results/segmentation]
        [--out-dir results/tables]
"""

import argparse
import csv
import json
from pathlib import Path

import yaml

MISSING = "MISSING"
REPO = Path(__file__).resolve().parents[1]


def show(path: Path):
    """Repo-relative when the path is inside the repo, absolute otherwise —
    a --out outside the tree (tests, scratch) must not crash the tool."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def rel_note(path: Path) -> str:
    """Paths that land in a COMMITTED table must be repo-relative — an
    absolute local path is machine-specific noise in a results file."""
    try:
        return str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return path.name

AP_KEYS = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]
AR_KEYS = ["AR_1", "AR_10", "AR_100", "AR_S", "AR_M", "AR_L"]
SEG_KEYS = ["mIoU_ss", "mIoU_ms", "pixel_acc", "mean_acc"]

# The Gate-2 "background classes". TASK-09 Phase C.2 names exactly these
# three ("the sky/wall/floor rows explicitly surfaced"); they are matched by
# the FIRST comma-separated token of the ADE20K name (the official names are
# "wall", "sky", "floor, flooring"), and their canonical indices are
# asserted as a cross-check so a reordered names file cannot silently move
# them.
BACKGROUND = {"wall": 0, "sky": 2, "floor": 3}

# baseline first: every delta in these tables is <variant> minus baseline
VARIANT_ORDER = ["baseline", "registers", "saga"]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    tmp.replace(path)
    return path


def fmt(v, nd=4):
    if v is None or v == MISSING:
        return MISSING
    return f"{float(v):.{nd}f}"


def delta(a, b, nd=4):
    """a - b, MISSING-propagating."""
    if a in (None, MISSING) or b in (None, MISSING):
        return MISSING
    return f"{float(a) - float(b):+.{nd}f}"


# ── provenance ────────────────────────────────────────────────────────────

def backbone_is_verified(meta_or_result: dict, matrix: dict):
    """The run's recorded backbone sha256 must equal the sha of the matrix
    candidate it says it used. Returns (ok, detail)."""
    key = None
    for run_id, run in matrix.get("runs", {}).items():
        if run_id == meta_or_result.get("run_id"):
            key = run["backbone"]
            break
    if key is None:
        return False, f"run_id not in the matrix ({meta_or_result.get('run_id')})"
    spec = matrix["backbones"][key]
    source = meta_or_result.get("backbone_source")
    cand = spec if source == "primary" else spec.get("fallback", {})
    expected = cand.get("sha256")
    observed = meta_or_result.get("backbone_sha256")
    if not expected:
        return False, f"matrix pins no sha256 for the {source} candidate"
    if expected != observed:
        return False, (f"sha mismatch: matrix {str(expected)[:12]}... vs run "
                       f"{str(observed)[:12]}...")
    return True, f"{source}/{str(observed)[:12]}..."


# ── detection ─────────────────────────────────────────────────────────────

def load_detection(det_root: Path, matrix: dict):
    """{variant: (run_id, coco_eval_best dict or None, note)}"""
    out = {}
    for run_id, run in sorted(matrix.get("runs", {}).items()):
        if run["task"] != "detection" or not run.get("submitted", True):
            continue
        path = det_root / run_id / "coco_eval_best.json"
        if not path.exists():
            out.setdefault(run["variant"], []).append(
                (run_id, None, f"{rel_note(path)} not found"))
            continue
        j = load_json(path)
        ok, detail = backbone_is_verified(j, matrix)
        out.setdefault(run["variant"], []).append(
            (run_id, j if ok else None,
             detail if ok else f"UNVERIFIED BACKBONE: {detail}"))
    return out


def det_rows(runs, matrix):
    """One row per run, then one delta row per non-baseline variant."""
    rows = []
    base = None
    for variant in VARIANT_ORDER:
        for run_id, j, note in runs.get(variant, []):
            row = {"table": "T4_coco", "kind": "value", "variant": variant,
                   "run_id": run_id, "note": note}
            if j is None:
                row.update({k: MISSING for k in AP_KEYS + AR_KEYS})
                row.update({"epoch": MISSING, "n_val_images": MISSING,
                            "seed": MISSING, "backbone_run": MISSING,
                            "backbone_source": MISSING,
                            "backbone_sha256": MISSING, "git_sha": MISSING})
            else:
                row.update({k: fmt(j.get(k), 3) for k in AP_KEYS + AR_KEYS})
                row.update({
                    "epoch": j.get("epoch"),
                    "n_val_images": j.get("n_val_images"),
                    "seed": j.get("seed"),
                    "backbone_run": j.get("backbone_run"),
                    "backbone_source": j.get("backbone_source"),
                    "backbone_sha256": j.get("backbone_sha256"),
                    "git_sha": j.get("git_sha"),
                })
                if variant == "baseline":
                    base = j
            rows.append(row)

    for variant in VARIANT_ORDER:
        if variant == "baseline":
            continue
        for run_id, j, _ in runs.get(variant, []):
            row = {"table": "T4_coco", "kind": f"delta_{variant}_minus_baseline",
                   "variant": variant, "run_id": run_id,
                   "note": ("baseline MISSING" if base is None else "")}
            for k in AP_KEYS + AR_KEYS:
                row[k] = (MISSING if (j is None or base is None)
                          else delta(j.get(k), base.get(k), 3))
            rows.append(row)
    return rows


def det_per_category(runs, out_path):
    """80 rows x {AP, AP50, AP_S} per variant + deltas vs baseline."""
    by_variant = {}
    for variant, entries in runs.items():
        for _, j, _ in entries:
            if j is None:
                continue
            by_variant[variant] = {int(c["category_id"]): c
                                   for c in j.get("per_category", [])}
    names = {}
    for cats in by_variant.values():
        for cid, c in cats.items():
            names[cid] = c.get("name", MISSING)

    metrics = ["AP", "AP50", "AP_S"]
    fields = ["category_id", "name"]
    for m in metrics:
        fields += [f"{v}_{m}" for v in VARIANT_ORDER]
        fields += [f"{v}_minus_baseline_{m}" for v in VARIANT_ORDER
                   if v != "baseline"]

    rows = []
    for cid in sorted(names):
        row = {"category_id": cid, "name": names[cid]}
        for m in metrics:
            vals = {}
            for v in VARIANT_ORDER:
                c = by_variant.get(v, {}).get(cid)
                val = None if c is None else c.get(m)
                vals[v] = MISSING if val is None else val
                row[f"{v}_{m}"] = fmt(vals[v], 3)
            for v in VARIANT_ORDER:
                if v == "baseline":
                    continue
                row[f"{v}_minus_baseline_{m}"] = delta(vals[v],
                                                       vals["baseline"], 3)
        rows.append(row)
    write_csv(out_path, fields, rows)
    return rows


# ── segmentation ──────────────────────────────────────────────────────────

def load_segmentation(seg_root: Path, matrix: dict):
    out = {}
    for run_id, run in sorted(matrix.get("runs", {}).items()):
        if run["task"] != "segmentation" or not run.get("submitted", True):
            continue
        d = seg_root / run_id
        ss_p, ms_p, pc_p = (d / "miou_ss.json", d / "miou_ms.json",
                            d / "per_class_iou.csv")
        if not ss_p.exists():
            out.setdefault(run["variant"], []).append(
                (run_id, None, None, None, f"{rel_note(ss_p)} not found"))
            continue
        ss = load_json(ss_p)
        ms = load_json(ms_p) if ms_p.exists() else None
        pc = read_csv(pc_p) if pc_p.exists() else None
        ok, detail = backbone_is_verified(ss, matrix)
        if not ok:
            out.setdefault(run["variant"], []).append(
                (run_id, None, None, None, f"UNVERIFIED BACKBONE: {detail}"))
            continue
        out.setdefault(run["variant"], []).append((run_id, ss, ms, pc, detail))
    return out


def seg_rows(runs):
    rows, base = [], None
    for variant in VARIANT_ORDER:
        for run_id, ss, ms, _pc, note in runs.get(variant, []):
            row = {"table": "T5_ade20k", "kind": "value", "variant": variant,
                   "run_id": run_id, "note": note}
            if ss is None:
                row.update({k: MISSING for k in SEG_KEYS})
                row.update({k: MISSING for k in
                            ("epoch_ss", "epoch_ms", "n_images", "seed",
                             "n_classes_scored", "n_labelled_pixels",
                             "ms_scales", "backbone_run", "backbone_source",
                             "backbone_sha256", "git_sha")})
            else:
                vals = {"mIoU_ss": ss.get("mIoU"),
                        "mIoU_ms": None if ms is None else ms.get("mIoU"),
                        "pixel_acc": ss.get("pixel_acc"),
                        "mean_acc": ss.get("mean_acc")}
                row.update({k: fmt(v, 4) if v is not None else MISSING
                            for k, v in vals.items()})
                row.update({
                    "epoch_ss": ss.get("epoch"),
                    "epoch_ms": MISSING if ms is None else ms.get("epoch"),
                    "n_images": ss.get("n_images"),
                    "seed": ss.get("seed"),
                    "n_classes_scored": ss.get("n_classes_scored"),
                    "n_labelled_pixels": ss.get("n_labelled_pixels"),
                    "ms_scales": ("" if ms is None
                                  else json.dumps(ms.get("ms_scales"))),
                    "backbone_run": ss.get("backbone_run"),
                    "backbone_source": ss.get("backbone_source"),
                    "backbone_sha256": ss.get("backbone_sha256"),
                    "git_sha": ss.get("git_sha"),
                })
                if variant == "baseline":
                    base = dict(vals)
            rows.append(row)

    for variant in VARIANT_ORDER:
        if variant == "baseline":
            continue
        for run_id, ss, ms, _pc, _n in runs.get(variant, []):
            row = {"table": "T5_ade20k",
                   "kind": f"delta_{variant}_minus_baseline",
                   "variant": variant, "run_id": run_id,
                   "note": ("baseline MISSING" if base is None else "")}
            cur = ({} if ss is None else
                   {"mIoU_ss": ss.get("mIoU"),
                    "mIoU_ms": None if ms is None else ms.get("mIoU"),
                    "pixel_acc": ss.get("pixel_acc"),
                    "mean_acc": ss.get("mean_acc")})
            for k in SEG_KEYS:
                a = cur.get(k)
                b = None if base is None else base.get(k)
                row[k] = (MISSING if a is None or b is None
                          else delta(a, b, 4))
            rows.append(row)
    return rows


def seg_per_class(runs, out_path):
    """150 rows; fractions as written, plus IoU points, plus deltas."""
    by_variant, names = {}, {}
    for variant, entries in runs.items():
        for _run_id, _ss, _ms, pc, _n in entries:
            if pc is None:
                continue
            by_variant[variant] = {int(r["class_index"]): r for r in pc}
            for r in pc:
                names[int(r["class_index"])] = r.get("name", MISSING)

    fields = ["class_index", "name", "is_background_class"]
    for v in VARIANT_ORDER:
        fields += [f"{v}_iou_frac", f"{v}_iou_pts"]
    for v in VARIANT_ORDER:
        if v != "baseline":
            fields += [f"{v}_minus_baseline_iou_frac",
                       f"{v}_minus_baseline_iou_pts"]

    bg_index = {}
    rows = []
    for idx in sorted(names):
        name = names[idx]
        head = name.split(",")[0].strip().lower()
        is_bg = head in BACKGROUND
        if is_bg:
            bg_index[head] = idx
        row = {"class_index": idx, "name": name,
               "is_background_class": "yes" if is_bg else "no"}
        vals = {}
        for v in VARIANT_ORDER:
            r = by_variant.get(v, {}).get(idx)
            raw = None if r is None else r.get("iou_frac")
            val = MISSING if raw in (None, "", MISSING) else float(raw)
            vals[v] = val
            row[f"{v}_iou_frac"] = fmt(val, 6)
            row[f"{v}_iou_pts"] = (MISSING if val == MISSING
                                   else f"{val * 100:.4f}")
        for v in VARIANT_ORDER:
            if v == "baseline":
                continue
            row[f"{v}_minus_baseline_iou_frac"] = delta(vals[v],
                                                        vals["baseline"], 6)
            row[f"{v}_minus_baseline_iou_pts"] = (
                MISSING if MISSING in (vals[v], vals["baseline"])
                else f"{(vals[v] - vals['baseline']) * 100:+.4f}")
        rows.append(row)

    # cross-check: the three background rows must be where ADE20K puts them
    for head, expected in BACKGROUND.items():
        got = bg_index.get(head)
        if got is None:
            raise ValueError(
                f"background class {head!r} not found in per_class_iou.csv — "
                f"the class names did not resolve, so the Gate-2 background "
                f"criterion cannot be evaluated")
        if got != expected:
            raise ValueError(
                f"background class {head!r} is at index {got}, expected "
                f"{expected} (ADE20K canonical order) — refusing to build a "
                f"table whose class identities are uncertain")
    write_csv(out_path, fields, rows)
    return rows


def main():
    p = argparse.ArgumentParser("TASK-09 Phase C dense tables")
    p.add_argument("--matrix", default="configs/dense_matrix.yaml")
    p.add_argument("--det-root", default="results/detection")
    p.add_argument("--seg-root", default="results/segmentation")
    p.add_argument("--out-dir", default="results/tables")
    args = p.parse_args()

    def rel(x):
        q = Path(x)
        return q if q.is_absolute() else REPO / q

    matrix = load_yaml(rel(args.matrix))
    out_dir = rel(args.out_dir)

    det = load_detection(rel(args.det_root), matrix)
    seg = load_segmentation(rel(args.seg_root), matrix)

    det_fields = (["table", "kind", "variant", "run_id"] + AP_KEYS + AR_KEYS
                  + ["epoch", "n_val_images", "seed", "backbone_run",
                     "backbone_source", "backbone_sha256", "git_sha", "note"])
    t4 = write_csv(out_dir / "T4_coco.csv", det_fields, det_rows(det, matrix))
    t4c = out_dir / "T4_coco_per_category.csv"
    det_per_category(det, t4c)

    seg_fields = (["table", "kind", "variant", "run_id"] + SEG_KEYS
                  + ["epoch_ss", "epoch_ms", "n_images", "seed",
                     "n_classes_scored", "n_labelled_pixels", "ms_scales",
                     "backbone_run", "backbone_source", "backbone_sha256",
                     "git_sha", "note"])
    t5 = write_csv(out_dir / "T5_ade20k.csv", seg_fields, seg_rows(seg))
    t5c = out_dir / "T5_ade20k_per_class.csv"
    seg_per_class(seg, t5c)

    for path in (t4, t4c, t5, t5c):
        print(f"wrote {show(path)}")


if __name__ == "__main__":
    main()
