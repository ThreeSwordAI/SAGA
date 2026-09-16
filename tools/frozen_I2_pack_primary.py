#!/usr/bin/env python3
"""
tools/frozen_I2_pack_primary.py
===============================
TASK I2 — the compact per-image record of the primary diagnostics, so that
every reported evaluation number regenerates from the repository even though
the raw `diag.parquet` files stay on the cluster.

    python tools/frozen_I2_pack_primary.py \
        --results results/frozen/I2_terminal/evaluation \
        --out figures_data/frozen/I2_evaluation_primary.npz

The evaluation sweep is 19 runs x 10,000 images x up to 11 (condition, stage)
pairs, which is ~90 MB of parquet. Almost all of that is the §2.7 provenance
columns repeated on every row and the nine diagnostics the bootstrap does not
resample. What C2 actually needs is the PER-IMAGE values of the five primary
diagnostics — the image-level bootstrap resamples them, and Figure 5A plots
their means against the terminal constant — so that is what is packed, in
float32, with the image order recorded once.

The raw files are not deleted and are not a secret: each run's committed
`run_meta.json` records their sha256 and byte size, so a number in a table
still traces to an identified file on the filesystem it lives on.

Determinism: arrays are written in sorted key order, values come straight out
of the diag file with no recomputation, and nothing here averages, fits or
selects. No training, no optimizer.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.build_I2_tables import read_rows  # noqa: E402
from analysis.i2_decision import PRIMARY_DIAGNOSTICS  # noqa: E402
from saga.frozen.records import records_path  # noqa: E402
from saga.frozen.runner import eligible_run_ids  # noqa: E402

MISSING = "MISSING"


def pack(results_root, manifest) -> dict:
    """`{"<run_id>|<condition>|<stage>|<diagnostic>": float32[n_images]}`
    plus the shared image order and a small provenance block."""
    results_root = Path(results_root)
    arrays, image_ids, meta = {}, None, {}
    for run_id in eligible_run_ids(manifest):
        run_dir = results_root / run_id
        diag_path = records_path(run_dir, "diag")
        if not diag_path.exists():
            continue
        grouped = {}
        for row in read_rows(diag_path):
            key = (row["condition_id"], row["stage"])
            g = grouped.setdefault(key, {"ids": [], "vals": {
                d: [] for d in PRIMARY_DIAGNOSTICS}})
            g["ids"].append(row["image_id"])
            for d in PRIMARY_DIAGNOSTICS:
                v = row.get(d)
                g["vals"][d].append(np.nan if v in (None, "", MISSING)
                                    else float(v))
        for (cond, stage), g in sorted(grouped.items()):
            order = np.argsort(np.asarray(g["ids"]))
            ids = np.asarray(g["ids"])[order]
            if image_ids is None:
                image_ids = ids
            elif not np.array_equal(ids, image_ids):
                raise SystemExit(
                    f"{run_id}/{cond}/{stage} carries a different image set "
                    f"from the first run packed — one pack, one split")
            for d in PRIMARY_DIAGNOSTICS:
                arrays[f"{run_id}|{cond}|{stage}|{d}"] = np.asarray(
                    g["vals"][d], dtype=np.float64)[order].astype(np.float32)
        rm = run_dir / "run_meta.json"
        if rm.exists():
            doc = json.loads(rm.read_text(encoding="utf-8"))
            meta[run_id] = {k: doc.get(k) for k in
                            ("ckpt_sha256", "split_sha256", "git_sha",
                             "outputs")}
    if not arrays:
        raise SystemExit(f"no diag files under {results_root}")
    return arrays, image_ids, meta


def main():
    p = argparse.ArgumentParser(
        description="Pack the primary diagnostics per image (float32).")
    p.add_argument("--results", required=True,
                   help="ONE split's sweep directory, e.g. "
                        "results/frozen/I2_terminal/evaluation")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    arrays, image_ids, meta = pack(args.results, args.manifest)
    payload = dict(sorted(arrays.items()))
    payload["__image_ids"] = np.asarray(image_ids)
    payload["__diagnostics"] = np.asarray(list(PRIMARY_DIAGNOSTICS))
    payload["__meta_json"] = np.asarray(
        json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    n_series = len(arrays)
    print(f"wrote {out}: {n_series} series x {image_ids.size} images "
          f"({out.stat().st_size / 1e6:.1f} MB, {len(meta)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
