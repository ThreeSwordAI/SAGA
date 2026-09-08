#!/usr/bin/env python3
"""
tools/dense_done.py
===================
TASK-09: is this dense run already finished?

    python tools/dense_done.py --matrix configs/dense_matrix.yaml \
        --run det_vitb_saga_s1
    exit 0 -> complete (the caller may skip everything)
    exit 1 -> not complete (the caller must proceed)
    exit 2 -> could not tell (bad arguments / unreadable matrix)
    exit 3 -> --require_backbone was given and no backbone resolves

The generated sbatch files call this BEFORE staging the dataset. Without it
every surplus job in a `--dependency=singleton,afterany` chain would unzip
~18 GB of COCO (10-15 min) onto node-local scratch on four A100s just to
have the trainer's in-process fast path tell it there is nothing to do. The
chains are deliberately over-long (a kill can cost a whole job), so surplus
jobs are the NORMAL case, not the exception.

`--require_backbone` additionally resolves the run's backbone and prints
which candidate won. That matters most for the registers runs: theirs is the
only entry with a fallback and an absolute vault path, and neither smoke
covers it, so without this a moved or unreadable registers checkpoint would
only surface AFTER the dataset stage, minutes into a 4x24h chain.

Reads nothing but the matrix, the run directory and the backbone's
existence; touches no GPU and loads no weights.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.dense_runtime import (repo_path, resolve_backbone,
                                 resolve_dense_config, run_is_complete)


def main() -> int:
    p = argparse.ArgumentParser("dense run completion check (TASK-09)")
    p.add_argument("--matrix", default="configs/dense_matrix.yaml")
    p.add_argument("--run", required=True)
    p.add_argument("--out_root", default=None)
    p.add_argument("--require_backbone", action="store_true",
                   help="also resolve the backbone and exit 3 if none is "
                        "usable (run this BEFORE staging a dataset)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    try:
        cfg = resolve_dense_config(repo_path(args.matrix), args.run)
    except Exception as exc:                     # bad matrix / unknown run
        print(f"dense_done: cannot resolve {args.run!r}: {exc}", flush=True)
        return 2

    out_root = repo_path(args.out_root or cfg["out_root"])
    run_dir = out_root / args.run
    epochs = int(cfg["train"]["epochs"])
    done = run_is_complete(run_dir, epochs)
    if not args.quiet:
        print(f"dense_done: {args.run} "
              f"{'COMPLETE' if done else 'not complete'} "
              f"({run_dir}, {epochs} epochs expected)", flush=True)
    if done:
        return 0

    if args.require_backbone:
        try:
            bb = resolve_backbone(cfg["backbone_spec"])
        except Exception as exc:
            print(f"dense_done: NO USABLE BACKBONE for {args.run}: {exc}",
                  flush=True)
            return 3
        size = Path(bb["ckpt"]).stat().st_size
        print(f"dense_done: backbone ok [{bb['source']}] {bb['ckpt']} "
              f"({size / 2 ** 20:.0f} MiB, run={bb['run']})", flush=True)
        if size == 0:
            print("dense_done: backbone file is EMPTY", flush=True)
            return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
