#!/usr/bin/env python3
"""
tools/ttr_prepare_run.py
========================
Create the TTR run directory `results/runs/ttr_<base_run_id>/` that the
patched model's eval/diag artifacts land in (TASK-10 A5.3).

    python tools/ttr_prepare_run.py \
        --base-run-id e2r_vits_mixup_baseline_s1 \
        --arch vit_small --recipe mixup --variant baseline \
        --neurons-file results/ttr/e2r_vits_mixup_baseline_s1/neurons.json \
        --n-neurons 16

Why this exists rather than a `mkdir` in the sbatch: the directory must carry
a `config.resolved.yaml` whose top-level `recipe:` is the BASE cell's, because
that is exactly how `tools/apply_fixed_thr.py --version canon` resolves which
tau a diag file belongs to (`recipe_from_run_dir` reads that key). Without it
the TTR diag files would be refused with "cannot resolve recipe_actual", there
would be no `sink_fixed_canon` field, and `tools/sink_address.py`'s hard
cross-check would have nothing to compare against.

The cell identity written here is the BASE cell's, deliberately: TTR is a
modification of that cell's baseline, not a new cell, so it must never be
given a tau of its own (TASK-10 acceptance).

Idempotent: re-running rewrites the config/meta pair and leaves any eval/diag
artifacts alone, so a requeued job is safe.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.run_registry import create_run, file_sha256
from saga.ttr import load_neurons_json, neurons_to_dict, top_k


def main():
    p = argparse.ArgumentParser(
        description="Create results/runs/ttr_<base_run_id>/ for TASK-10.")
    p.add_argument("--base-run-id", required=True,
                   help="the UNPATCHED baseline run this is derived from")
    p.add_argument("--arch", required=True, choices=["vit_small", "vit_base"])
    p.add_argument("--recipe", required=True, choices=["mixup", "nomix"],
                   help="recipe_actual of the BASE cell — sets the canon tau")
    p.add_argument("--variant", default="baseline",
                   choices=["baseline", "registers", "saga"])
    p.add_argument("--neurons-file", required=True)
    p.add_argument("--n-neurons", type=int, required=True,
                   help="the sweep value chosen by the A3 validation gate")
    p.add_argument("--n-extra-tokens", type=int, default=1)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--normal-values", default="zero",
                   choices=["zero", "mean", "same"])
    p.add_argument("--runs-root", default="results/runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    neurons_file = Path(args.neurons_file)
    payload = load_neurons_json(neurons_file)
    if payload["arch"] != args.arch:
        raise SystemExit(
            f"--arch {args.arch} disagrees with {neurons_file}'s "
            f"{payload['arch']!r} — refusing to mislabel the cell")
    if args.n_neurons > len(payload["neurons"]):
        raise SystemExit(
            f"--n-neurons {args.n_neurons} exceeds the {len(payload['neurons'])} "
            f"entries stored in {neurons_file} (top_n_stored="
            f"{payload['top_n_stored']}) — rerun the scan with a larger "
            f"--top-n-stored rather than silently selecting fewer")

    selection = top_k(payload["neurons"], args.n_neurons)
    grouped = neurons_to_dict(selection)

    run_id = f"ttr_{args.base_run_id}"
    config = {
        # the three keys the existing tools read; the values are the BASE
        # cell's, which is what makes the canon tau resolve correctly
        "arch": args.arch,
        "variant": args.variant,
        "recipe": args.recipe,
        "ttr": {
            "base_run_id": args.base_run_id,
            "method": "test-time registers (Jiang et al., NeurIPS 2025)",
            "implementation": "saga/ttr.py (reimplementation; see "
                              "third_party/ttr/PROVENANCE.md)",
            "neurons_file": neurons_file.as_posix(),
            "neurons_file_sha256": file_sha256(neurons_file),
            "criterion": payload["criterion"],
            "outlier_tau": payload["outlier_tau"],
            "tau_key": payload["tau_key"],
            "tau_source": payload["tau_source"],
            "tau_recalibrated_on_patched_model": False,
            "n_neurons": args.n_neurons,
            "n_extra_tokens": args.n_extra_tokens,
            "scale": args.scale,
            "normal_values": args.normal_values,
            "layers_touched": sorted(grouped),
            "selection": [[l, n] for l, n, _ in selection],
        },
    }

    run_dir = create_run(args.runs_root, run_id, config, args.seed)
    print(f"prepared {run_dir}")
    print(f"  base cell     {args.arch}|{args.recipe} (variant {args.variant})")
    print(f"  canon tau     {payload['outlier_tau']} "
          f"(key {payload['tau_key']}, recalibrated=False)")
    print(f"  n_neurons     {args.n_neurons} over layers {sorted(grouped)}")
    print(f"  neurons sha   {config['ttr']['neurons_file_sha256'][:16]}")
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "diag").mkdir(exist_ok=True)
    print(f"  wrote {run_dir / 'config.resolved.yaml'}")
    print(f"  wrote {run_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
