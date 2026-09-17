#!/usr/bin/env python3
"""
tools/build_I3_primary_pack.py
==============================
TASK B B9 — `figures_data/frozen/I3_primary.npz`, the committed per-image
pack every reported I3 number is resampled from.

    python tools/build_I3_primary_pack.py
        [--results results/frozen/I3_gate_edits/evaluation]
        [--out figures_data/frozen/I3_primary.npz]

WHY IT EXISTS EVEN THOUGH THE PARQUET WAS COMMITTED BY ACCIDENT
----------------------------------------------------------------
On 2026-09-17 a broad `git add` on the HPC took I3's 16 raw
`records.parquet` / `diag.parquet` files (113.1 MiB) to the remote, against
the I2 storage policy. Those blobs are in history and stay there, but the
files are now UNTRACKED and future sweeps are git-ignored, so the repository
must carry the per-image values some other way — which is what this pack is
and always was for (task file §9).

It is not a copy of the parquet. It holds the three per-image quantities the
tables actually resample, as the DIFFERENCES the contrasts are defined on:

    dnll[run][cond][image]     nll(cond) - nll(original), float32
    dcorrect[run][cond][image] correct(cond) - correct(original), int8
    dun[run][cond][image]      delta_update_norm, float32
    native_nll[run][image]     nll(original), float32, so the absolute
                               scale is recoverable and a Delta can be
                               reported beside the loss it moved

plus the image ids, the condition ids in file order, and the provenance of
every run (checkpoint sha, split sha, permutations sha, git sha).

`max_abs_logit_diff_vs_native` is stored PER CONDITION as its maximum over
images, not per image: it is a bit-identity check (`dihedral0` must be
exactly 0), and a maximum answers that question exactly while a per-image
column would add 19 MB to answer it 10,000 times over.

Deterministic, no sampling, no model: this reads committed records and
writes float32. The image order is the SPLIT's own order, taken from the
`original` condition and reused for every other, so a row index means the
same image in every array.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen.records import records_path  # noqa: E402

MISSING = "MISSING"
REFERENCE = "original"

#: One builder, two work packages. The reference condition and the extra
#: per-image columns are all that differ: I3 measures how big its edit was,
#: I4 measures how much energy it injected and whether it hit its target.
#: A second tool would be a second place for the (image, condition) ordering
#: to drift, and the ordering is what makes a pack comparable to a parquet.
PACKS = {
    "I3": {"reference": "original",
           "extra": ("delta_update_norm",)},
    "I4": {"reference": "native",
           "extra": ("measured_perturbation_norm", "energy_rel_error")},
}

#: Provenance carried per run, from the records themselves.
PROVENANCE = ("run_id", "arch", "recipe_actual", "variant", "ckpt_kind",
              "ckpt_sha256", "split_name", "split_sha256", "stage",
              "precision", "git_sha", "permutations_sha256")


class PackError(ValueError):
    """A pack this tool refuses to build."""


def _num(v):
    if v is None or v == "" or v == MISSING:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def read_rows(path):
    import csv
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_run(rows, *, reference=REFERENCE, extra=("delta_update_norm",)) -> dict:
    """One checkpoint's arrays, keyed by condition, in the split's order."""
    REFERENCE = reference
    by_cond = {}
    for r in rows:
        by_cond.setdefault(str(r["condition_id"]), {})[str(r["image_id"])] = r
    if REFERENCE not in by_cond:
        raise PackError(f"no {REFERENCE!r} condition — every Delta is "
                        f"measured against it")
    ref = by_cond[REFERENCE]
    # THE image order: the reference condition's, sorted once and reused, so
    # a row index means the same image in every array of this run.
    ids = sorted(ref)
    conds = [c for c in sorted(by_cond) if c != REFERENCE]

    n_i, n_c = len(ids), len(conds)
    dnll = np.zeros((n_c, n_i), dtype=np.float32)
    dcorrect = np.zeros((n_c, n_i), dtype=np.int8)
    extras = {name: np.zeros((n_c, n_i), dtype=np.float32) for name in extra}
    max_logit = np.zeros(n_c, dtype=np.float32)

    ref_nll = np.asarray([_num(ref[i]["nll"]) for i in ids], dtype=np.float64)
    ref_cor = np.asarray([_num(ref[i]["correct"]) for i in ids],
                         dtype=np.float64)

    for k, cond in enumerate(conds):
        got = by_cond[cond]
        missing = [i for i in ids if i not in got]
        if missing:
            raise PackError(
                f"condition {cond!r} is missing {len(missing)} image(s) that "
                f"{REFERENCE!r} has — a partial condition is not packed")
        dnll[k] = (np.asarray([_num(got[i]["nll"]) for i in ids],
                              dtype=np.float64) - ref_nll).astype(np.float32)
        dcorrect[k] = (np.asarray([_num(got[i]["correct"]) for i in ids],
                                  dtype=np.float64) - ref_cor).astype(np.int8)
        for name, arr in extras.items():
            arr[k] = np.asarray([_num(got[i].get(name)) for i in ids],
                                dtype=np.float64).astype(np.float32)
        mal = np.asarray([_num(got[i].get("max_abs_logit_diff_vs_native"))
                          for i in ids], dtype=np.float64)
        max_logit[k] = np.nanmax(mal) if mal.size else np.nan

    out = {"image_ids": np.asarray(ids), "conditions": np.asarray(conds),
           "dnll": dnll, "dcorrect": dcorrect}
    out.update(extras)
    out.update({"max_abs_logit_diff": max_logit,
                "native_nll": ref_nll.astype(np.float32),
                "provenance": {k: str(rows[0].get(k, MISSING))
                               for k in PROVENANCE}})
    return out


def main():
    p = argparse.ArgumentParser(
        description="Build the committed per-image I3 primary pack.")
    p.add_argument("--results",
                   default="results/frozen/I3_gate_edits/evaluation")
    p.add_argument("--out", default="figures_data/frozen/I3_primary.npz")
    p.add_argument("--work-package", default="I3", choices=sorted(PACKS))
    args = p.parse_args()
    spec = PACKS[args.work_package]

    root = Path(args.results)
    runs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() \
        else []
    if not runs:
        print(f"no run directories under {root}", file=sys.stderr)
        return 1

    payload, meta = {}, {"runs": [], "work_package": args.work_package,
                         "reference": spec["reference"],
                         "extra": list(spec["extra"]),
                         "generated_by": __file__.replace("\\", "/"),
                         "generated_at": datetime.now(timezone.utc).isoformat(
                             timespec="seconds"),
                         "results": str(root)}
    for d in runs:
        path = records_path(d, "records")
        if not path.exists():
            print(f"  SKIP {d.name}: no records file", file=sys.stderr)
            continue
        built = build_run(read_rows(path), reference=spec["reference"],
                          extra=spec["extra"])
        run = d.name
        for key in (("image_ids", "conditions", "dnll", "dcorrect",
                     "max_abs_logit_diff", "native_nll") + spec["extra"]):
            payload[f"{run}__{key}"] = built[key]
        meta["runs"].append(dict(built["provenance"], run_dir=run,
                                 n_images=int(built["dnll"].shape[1]),
                                 n_conditions=int(built["dnll"].shape[0])))
        print(f"  {run}: {built['dnll'].shape[0]} conditions x "
              f"{built['dnll'].shape[1]} images")

    if not meta["runs"]:
        print("nothing packed", file=sys.stderr)
        return 1

    payload["meta_json"] = np.array(json.dumps(meta, indent=2, sort_keys=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temp file and renamed over the target, as every other
    # artifact in this project is. `np.savez_compressed` APPENDS `.npz` to a
    # name that does not already end in it, so the temp file is handed an
    # open file object rather than a path — otherwise it writes
    # `<name>.npz.tmp.npz` and the rename fails on a file that is not there.
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(out)
    print(f"\nwrote {out} ({out.stat().st_size / 1048576:.1f} MiB, "
          f"{len(meta['runs'])} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
