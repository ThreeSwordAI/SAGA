#!/usr/bin/env python3
"""
evaluation/e6_finegrained/tools/build_ft_split.py
=================================================
TASK-08 (B7 fix, part 1): freeze a stratified VAL split carved from the
OFFICIAL TRAIN split of CUB-200-2011 / FGVC-Aircraft. The official test
split is NEVER part of this file and is evaluated exactly once per run,
at the end (tools/train_ft.py).

Per class, round(val_frac * n_class) images (round-half-up, minimum 1) are
drawn with random.Random(seed) from the class's relpath-sorted train items.
Both resulting lists are stored IN FULL (relative paths + labels, like the
diag split), so the trainer never re-derives membership:

    {"dataset": "cub", "seed": 0, "val_frac": 0.1,
     "n_official_train": 5994, "n_train": 5394, "n_val": 600,
     "n_classes": 200,
     "items_train": [["images/001.../....jpg", 0], ...],
     "items_val":   [...]}

Runs on the HPC (CPU, seconds — reads only the small metadata text members,
straight from the tarball; no image extraction):

    python evaluation/e6_finegrained/tools/build_ft_split.py \
        --dataset cub \
        --tar /home/woody/iwi5/iwi5359h/Data/CUB-200/CUB_200_2011.tgz \
        --out results/ftsplit/cub_val_split.json

The JSON gets committed; determinism and disjointness are test-pinned
(tests/test_task08_ft.py) and re-asserted here at build time.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluation.e6_finegrained.data.ft_meta import DATASETS, official_splits
from saga.run_registry import file_sha256, git_sha


def build_split(dataset: str, root=None, tar=None,
                val_frac: float = 0.1, seed: int = 0) -> dict:
    """Deterministically carve the stratified val split from the official
    train split. Returns the full split dict (see module docstring)."""
    train_items, test_items, n_classes = official_splits(
        dataset, root=root, tar=tar)

    by_class = defaultdict(list)
    for rel, label in train_items:
        by_class[label].append((rel, label))

    rng = random.Random(seed)
    val = []
    for label in sorted(by_class):
        cls_items = sorted(by_class[label])  # relpath-sorted, deterministic
        n_val = max(1, int(len(cls_items) * val_frac + 0.5))  # round-half-up
        if n_val >= len(cls_items):
            raise ValueError(
                f"class {label} has only {len(cls_items)} train images — "
                f"carving {n_val} for val would leave no train items")
        val.extend(rng.sample(cls_items, n_val))

    val_set = set(v[0] for v in val)
    train = sorted(it for it in train_items if it[0] not in val_set)
    val = sorted(val)

    # disjointness/consistency — enforced at build time (explicit raises so
    # PYTHONOPTIMIZE can never strip them), test-pinned too
    train_set = set(t[0] for t in train)
    test_set = set(t[0] for t in test_items)
    official_set = set(t[0] for t in train_items)

    def _refuse(cond, msg):
        if cond:
            raise ValueError(f"split invariant violated: {msg}")

    _refuse(len(official_set) != len(train_items),
            "duplicate paths in the official train metadata")
    _refuse(bool(official_set & test_set),
            "the OFFICIAL train and test splits overlap — corrupt dataset "
            "metadata")
    _refuse(not val_set <= official_set,
            "val must be a subset of official train")
    _refuse(bool(val_set & test_set), "val and official test overlap")
    _refuse(bool(train_set & val_set), "carved train and val overlap")
    _refuse(train_set | val_set != official_set,
            "carved train + val must reassemble the official train split")
    _refuse(len(train) + len(val) != len(train_items),
            "item counts do not add up")

    return {
        "dataset": dataset,
        "seed": seed,
        "val_frac": val_frac,
        "n_official_train": len(train_items),
        "n_train": len(train),
        "n_val": len(val),
        "n_classes": n_classes,
        # provenance (informational; determinism is over the item lists)
        "source": str(root if root is not None else tar),
        "source_sha256": file_sha256(tar) if tar is not None else None,
        "git_sha": git_sha(),
        "items_train": [list(it) for it in train],
        "items_val": [list(it) for it in val],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Freeze the seeded stratified fine-tune val split "
                    "(carved from the OFFICIAL train split; test untouched).")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="extracted dataset root "
                                    "(CUB_200_2011/ or fgvc-aircraft-2013b/)")
    src.add_argument("--tar", help="distribution tarball (only metadata "
                                   "members are read)")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True,
                        help="e.g. results/ftsplit/cub_val_split.json")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(
            f"{out} already exists — frozen splits are WRITE-ONCE (the "
            f"committed carve is the protocol). If you truly intend to "
            f"change the carve, delete the file explicitly first and expect "
            f"every downstream run to be redone.")

    split = build_split(args.dataset, root=args.root, tar=args.tar,
                        val_frac=args.val_frac, seed=args.seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(split, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    print(f"wrote {out}: dataset={split['dataset']} seed={split['seed']} "
          f"official_train={split['n_official_train']} -> "
          f"train={split['n_train']} + val={split['n_val']} "
          f"({split['n_classes']} classes)")


if __name__ == "__main__":
    main()
