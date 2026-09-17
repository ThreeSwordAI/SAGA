#!/usr/bin/env python3
"""
tools/verify_primary_pack.py
============================
THE REPRODUCTION CHECK, under the standing rule of 2026-09-17.

    python tools/verify_primary_pack.py --work-package I4

> A reproduction check compares **per-condition MEANS, never contrasts**, and
> only for conditions selected by a **fixed rule stated in the log BEFORE the
> check runs**.

WHY THE RULE EXISTS. Verifying `figures_data/frozen/I3_primary.npz` against
its parquet, I compared C1 and C2 — contrasts — on three real checkpoints,
and so saw two primary contrast values before the §4 interpretation guide had
been amended. The disclosure is in `docs/TASK_B_I3_I4.md` §4 and in the log.
A per-condition mean would have verified the pack exactly as well: a pack
that reproduces every condition's mean reproduces every contrast built from
those conditions, because a contrast is a linear combination of them. The
weaker check is the sufficient one, so it is the one that is now allowed.

THE CONDITIONS ARE FIXED HERE, IN CODE, NOT CHOSEN AT THE PROMPT. They were
written into `docs/TASK_LOG.md` before this file ran for the first time. A
`--conditions` flag would make "stated in the log beforehand" a promise
rather than a property.

What it compares, per (run, condition):

    mean over images of `nll(condition) - nll(reference)`

from the PACK, against the same quantity recomputed from the PARQUET. Both
are means of the same per-image differences, so they must agree to float32
storage precision. Nothing else is printed: not a contrast, not a CI, not a
verdict.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen.records import records_path  # noqa: E402
from tools.build_I3_primary_pack import PACKS, read_rows, _num  # noqa: E402

#: The fixed rule, per work package: which conditions, and how many runs in
#: MANIFEST ORDER. Stated in `docs/TASK_LOG.md` before this ran.
RULE = {
    "I3": {"conditions": ("mean_L7", "perm0_L7", "dihedral0_L7"), "n_runs": 3},
    "I4": {"conditions": ("prim_e10_L7", "ctrl0_e10m_L7"), "n_runs": 3},
}

#: float32 storage round-off on a mean of 10,000 values.
TOL = 1e-6


def manifest_order(manifest, results_root):
    """The first N run_ids of the cohort, in the manifest's own order."""
    from saga.frozen.runner import eligible_cohort
    root = Path(results_root)
    return [r["run_id"] for r in eligible_cohort(manifest)
            if (root / r["run_id"]).is_dir()]


def check(work_package, pack_path, results_root, manifest, tol=TOL) -> int:
    spec = PACKS[work_package]
    rule = RULE[work_package]
    reference = spec["reference"]
    z = np.load(pack_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if meta.get("reference") not in (None, reference):
        print(f"pack reference is {meta['reference']!r}, expected "
              f"{reference!r}", file=sys.stderr)
        return 2

    runs = manifest_order(manifest, results_root)[:rule["n_runs"]]
    print(f"{work_package}: per-condition MEANS only, "
          f"{list(rule['conditions'])}, first {rule['n_runs']} run_ids in "
          f"manifest order, reference {reference!r}\n")

    worst, n = 0.0, 0
    for run in runs:
        rows = read_rows(records_path(Path(results_root) / run, "records"))
        by = {}
        for r in rows:
            by.setdefault(str(r["condition_id"]), {})[str(r["image_id"])] = r
        ref = by[reference]
        conds = list(z[f"{run}__conditions"])
        dnll = z[f"{run}__dnll"]
        ids = list(z[f"{run}__image_ids"])
        for cond in rule["conditions"]:
            if cond not in by:
                print(f"  SKIP {run} / {cond}: not in these records")
                continue
            pq = float(np.mean([_num(by[cond][i]["nll"]) - _num(ref[i]["nll"])
                                for i in ids]))
            pk = float(dnll[conds.index(cond)].mean())
            d = abs(pq - pk)
            worst = max(worst, d)
            n += 1
            print(f"  {'OK ' if d <= tol else 'BAD'} {run[:34]:36s} "
                  f"{cond:16s} parquet {pq:+.8f}  pack {pk:+.8f}  "
                  f"|diff| {d:.2e}")
    print(f"\n{n} comparison(s), largest |diff| {worst:.3e}, tolerance {tol:g}")
    if worst > tol:
        print("PACK DOES NOT REPRODUCE THE PARQUET — do not build tables",
              file=sys.stderr)
        return 1
    print("PACK REPRODUCES THE PARQUET. No contrast was computed.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--work-package", default="I4", choices=sorted(RULE))
    p.add_argument("--pack", default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    args = p.parse_args()
    wp = args.work_package
    root = args.results or (
        f"results/frozen/{'I3_gate_edits' if wp == 'I3' else 'I4_perturbation'}"
        f"/evaluation")
    pack = args.pack or f"figures_data/frozen/{wp}_primary.npz"
    return check(wp, pack, root, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
