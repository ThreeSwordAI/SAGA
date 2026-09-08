#!/usr/bin/env python3
"""
segmentation/tools/build_fixed20.py
====================================
TASK-09: build the frozen 20-image ADE20K validation probe list
(results/probe/ade20k_fixed20.json).

The acceptance criterion is that this list is COMMITTED BEFORE the first
segmentation eval, so it cannot be chosen after seeing any result. ADE20K's
validation split is canonical and its file names are fully determined
(`ADE_val_%08d`, 1..2000), so the list is derivable without the dataset —
which is exactly why it can be committed now, while ADE20K itself is only
on the HPC. The trainer then re-resolves every stem against the staged split
and HARD-ERRORS if any one is missing, so a wrong assumption about the split
cannot silently degrade into a smaller probe set.

Selection rule (recorded in the file): sorted(random.Random(seed).sample(
range(1, n_val + 1), n)).

    python segmentation/tools/build_fixed20.py            # writes once
    python segmentation/tools/build_fixed20.py --force    # deliberate rebuild
"""

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from saga.run_registry import git_sha
from tools.dense_runtime import atomic_json_dump

DEFAULT_OUT = "results/probe/ade20k_fixed20.json"


def build_spec(n: int = 20, n_val: int = 2000, seed: int = 0) -> dict:
    if not 0 < n <= n_val:
        raise ValueError(f"need 0 < n <= n_val, got n={n}, n_val={n_val}")
    ids = sorted(random.Random(seed).sample(range(1, n_val + 1), n))
    return {
        "dataset": "ade20k",
        "split": "validation",
        "n": n,
        "n_val_source": n_val,
        "seed": seed,
        "selector": ("sorted(random.Random(seed).sample("
                     "range(1, n_val + 1), n))"),
        "stem_format": "ADE_val_%08d",
        "ids": ids,
        "stems": [f"ADE_val_{i:08d}" for i in ids],
        "note": ("Derived from the canonical ADEChallengeData2016 validation "
                 "naming, not from a directory listing — segmentation/tools/"
                 "train.py re-resolves every stem against the staged split "
                 "and raises if one is missing."),
        "git_sha": git_sha(),
    }


def main():
    p = argparse.ArgumentParser("build the frozen ADE20K 20-image probe list")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing list (it is write-once: the "
                        "committed list is what every seg run is scored on)")
    args = p.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists — the probe list is write-once "
                 f"(pass --force to rebuild deliberately)")

    spec = build_spec(n=args.n, n_val=args.n_val, seed=args.seed)
    atomic_json_dump(spec, out)
    print(f"wrote {out}\n  {spec['n']} stems, seed {spec['seed']}: "
          f"{spec['stems'][:3]} ... {spec['stems'][-1]}")


if __name__ == "__main__":
    main()
