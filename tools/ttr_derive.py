#!/usr/bin/env python3
"""
tools/ttr_derive.py
===================
TASK-10 A5.3 — full-val eval + canonical diagnostics for ONE checkpoint with
test-time registers applied, written into the TTR run dir with the SAME
schemas the unpatched tools produce, so every existing collector reads them.

    python tools/ttr_derive.py \
        --ckpt results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth \
        --arch vit_small --recipe mixup --variant baseline \
        --neurons-file results/ttr/e2r_vits_mixup_baseline_s1/neurons.json \
        --n-neurons 16 --data $STAGE_DIR \
        --run-dir results/runs/ttr_e2r_vits_mixup_baseline_s1

Why a separate tool instead of a `--ttr` flag on `tools/eval.py` /
`tools/diagnose.py`: those two produce every paper number in the project, and
wrapping their eval loops in a context manager means re-indenting them. This
tool instead IMPORTS their exact pieces —

  * `tools.eval.shard_counts` / `counts_to_metrics` — the float64 counting
    arithmetic, already unit-tested in `tests/test_eval_reduction.py`
  * `tools.eval.build_val_transform` — the trainers' val transform code path
  * `tools.eval.IMAGENET_VAL_SIZE` and the same `n == len(dataset)` assert
  * `saga.metrics.compute_diagnostics` — the one canonical diagnostics pass

— so no arithmetic is re-implemented and `eval.py`/`diagnose.py` keep a zero
diff. `tests/test_task10_ttr.py` pins that this tool's UNPATCHED path
reproduces `tools/eval.py`'s numbers rather than assuming it.

Single-GPU by design (Phase B is one A40, eval-only), so there is no
sharding and no all_reduce: one process sees all 50 000 images.

tau: diagnostics are run WITHOUT `--fixed-thr`. The canon field is backfilled
afterwards by `tools/apply_fixed_thr.py --version canon`, which resolves the
BASE cell from the run dir's `config.resolved.yaml` (written by
`tools/ttr_prepare_run.py`). That keeps exactly one code path writing
`sink_fixed_canon` for TTR and non-TTR runs alike, and guarantees the tau is
the base cell's rather than anything recalibrated on the patched model.

Idempotent: each of the two steps is skipped when its output JSON already
exists AND records this checkpoint's sha256, so a requeued job redoes only
what is missing. The npz is written before the JSON, because the JSON is the
completion marker (the TASK-02 ordering rule).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.metrics import compute_diagnostics
from saga.run_registry import file_sha256, git_sha
from saga.ttr import apply_ttr, load_neurons_json, neurons_to_dict, top_k
from tools.build_diag_split import DiagSplitDataset
from tools.eval import (IMAGENET_VAL_SIZE, build_val_transform,
                        counts_to_metrics, shard_counts)
from tools.model_factory import build_model, load_checkpoint


def already_done(path: Path, sha: str) -> bool:
    """True iff `path` exists and was produced from this exact checkpoint."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            return json.load(f).get("ckpt_sha256") == sha
    except (json.JSONDecodeError, OSError):
        print(f"  {path.name}: unreadable/torn — redoing it")
        return False


def ttr_block(payload: dict, selection, args) -> dict:
    """The provenance block stamped into both output JSONs."""
    return {
        "method": "test-time registers (Jiang et al., NeurIPS 2025)",
        "implementation": "saga/ttr.py (reimplementation; see "
                          "third_party/ttr/PROVENANCE.md)",
        "neurons_file": str(args.neurons_file),
        "criterion": payload["criterion"],
        "outlier_tau": payload["outlier_tau"],
        "tau_key": payload["tau_key"],
        "tau_source": payload["tau_source"],
        "tau_recalibrated_on_patched_model": False,
        "n_neurons": args.n_neurons,
        "n_extra_tokens": args.n_extra_tokens,
        "scale": args.scale,
        "normal_values": args.normal_values,
        "layers_touched": sorted(neurons_to_dict(selection)),
        "base_run_id": args.base_run_id,
    }


@torch.no_grad()
def run_eval(model, args, device, sha, extra: dict, out: Path) -> None:
    """Exact full-val pass; same schema as tools/eval.py plus a `ttr` block."""
    from timm.data import create_dataset
    dataset = create_dataset("imagefolder", root=args.data, split="val",
                             is_training=False)
    dataset.transform = build_val_transform(224)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    counts = torch.zeros(4, dtype=torch.float64, device=device)
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        counts += shard_counts(model(images), targets)
        if step % 20 == 0:
            print(f"  [{step + 1}/{len(loader)}] top1 so far: "
                  f"{100.0 * counts[0] / counts[3]:.3f}%", flush=True)

    n = int(counts[3].item())
    assert n == len(dataset) == IMAGENET_VAL_SIZE, (
        f"evaluated {n} images but dataset has {len(dataset)} "
        f"(expected {IMAGENET_VAL_SIZE}) -- refusing to write results")

    result = {
        **counts_to_metrics(counts.cpu()),
        "ckpt": Path(args.ckpt).resolve().as_posix(),
        "ckpt_sha256": sha,
        "git_sha": git_sha(),
        "arch": args.arch,
        "variant": args.variant,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "world_size": 1,
        "amp": "off",
        "ttr": extra,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  top1={result['top1']:.3f}  top5={result['top5']:.3f}  n={n}")
    print(f"  wrote {out}")


def run_diagnose(model, args, device, sha, extra: dict, out: Path) -> None:
    """Canonical diagnostics on the frozen split; same schema as diagnose.py.

    No --fixed-thr: apply_fixed_thr --version canon backfills the canon field
    from the npz, resolving the BASE cell via config.resolved.yaml.
    """
    dataset = DiagSplitDataset(args.data, args.split_file,
                               transform=build_val_transform(224))
    loader = DataLoader(dataset, batch_size=args.diag_batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    diag = compute_diagnostics(model, loader, device, with_attn=args.attn,
                               fixed_thr=None, n_effrank=args.n_effrank,
                               collect_norms=True)
    arrays = diag.pop("_norms_arrays")

    result = {
        "ckpt": Path(args.ckpt).resolve().as_posix(),
        "ckpt_sha256": sha,
        "git_sha": git_sha(),
        "arch": args.arch,
        "variant": args.variant,
        **diag,
        "ttr": extra,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    # npz FIRST, JSON LAST — the JSON is the completion marker
    npz_path = out.with_name(out.stem + "_norms.npz")
    np.savez(npz_path,
             last_block_patch_norms=arrays["last_block_patch_norms"],
             cls_norms=arrays["cls_norms"],
             median_patch_norms=arrays["median_patch_norms"])
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  sink_mad_k5={result['sink_mad_k5']:.4f}  "
          f"oversmooth_pairwise={result['oversmooth_pairwise']:.4f}  "
          f"eff_rank={result['eff_rank']:.2f}  "
          f"num_prefix_tokens={result['num_prefix_tokens']}")
    print(f"  wrote {npz_path} (git-ignored)")
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser(
        description="Full-val eval + diagnostics for a TTR-patched checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", required=True, choices=["vit_small", "vit_base"])
    p.add_argument("--recipe", required=True, choices=["mixup", "nomix"],
                   help="recipe_actual of the BASE cell (provenance only "
                        "here; the canon tau is applied by apply_fixed_thr)")
    p.add_argument("--variant", default="baseline",
                   choices=["baseline", "registers", "saga"])
    p.add_argument("--neurons-file", required=True)
    p.add_argument("--n-neurons", type=int, required=True)
    p.add_argument("--base-run-id", default=None)
    p.add_argument("--data", required=True, metavar="ROOT",
                   help="ImageNet root containing val/ (ImageFolder layout)")
    p.add_argument("--split-file",
                   default="results/diagsplit/val_diag_split.json")
    p.add_argument("--run-dir", required=True,
                   help="results/runs/ttr_<base_run_id> (ttr_prepare_run.py)")
    p.add_argument("--eval-name", default="eval_last")
    p.add_argument("--diag-name", default="diag_last")
    p.add_argument("--n-extra-tokens", type=int, default=1)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--normal-values", default="zero",
                   choices=["zero", "mean", "same"])
    p.add_argument("--attn", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--n-effrank", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--diag-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-diagnose", action="store_true")
    args = p.parse_args()

    # A diag filename whose stem has exactly 7 underscore-separated parts is
    # parsed as a LEGACY name by apply_fixed_thr.recipe_from_stem, which would
    # then resolve the cell from the filename instead of this run's config.
    if len(Path(args.diag_name).stem.split("_")) == 7:
        raise SystemExit(
            f"--diag-name {args.diag_name!r} has 7 underscore-separated "
            f"parts, which apply_fixed_thr.recipe_from_stem reads as a legacy "
            f"e2 filename — the canon tau would be resolved from the name "
            f"rather than from config.resolved.yaml. Pick another name.")

    run_dir = Path(args.run_dir)
    if not (run_dir / "config.resolved.yaml").exists():
        raise SystemExit(
            f"{run_dir}/config.resolved.yaml is missing — run "
            f"tools/ttr_prepare_run.py first, or apply_fixed_thr --version "
            f"canon will not be able to resolve this run's base cell")

    payload = load_neurons_json(args.neurons_file)
    if payload["arch"] != args.arch:
        raise SystemExit(f"--arch {args.arch} disagrees with the neurons "
                         f"file's {payload['arch']!r}")
    if args.n_neurons > len(payload["neurons"]):
        raise SystemExit(
            f"--n-neurons {args.n_neurons} exceeds the "
            f"{len(payload['neurons'])} entries in {args.neurons_file}")
    selection = top_k(payload["neurons"], args.n_neurons)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sha = file_sha256(args.ckpt)

    eval_out = run_dir / "eval" / f"{args.eval_name}.json"
    diag_out = run_dir / "diag" / f"{args.diag_name}.json"
    do_eval = not args.skip_eval and not already_done(eval_out, sha)
    do_diag = not args.skip_diagnose and not already_done(diag_out, sha)

    print(f"ttr_derive: {args.arch}/{args.variant} base cell "
          f"{args.arch}|{args.recipe}")
    print(f"  ckpt      {args.ckpt}")
    print(f"  sha256    {sha}")
    print(f"  neurons   {args.n_neurons} from {args.neurons_file} "
          f"(criterion {payload['criterion']})")
    print(f"  layers    {sorted(neurons_to_dict(selection))}")
    print(f"  eval      {'RUN' if do_eval else 'skip (already done)'}")
    print(f"  diagnose  {'RUN' if do_diag else 'skip (already done)'}")
    if not (do_eval or do_diag):
        print("nothing to do")
        return 0

    model = build_model(args.arch, args.variant)
    load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    extra = ttr_block(payload, selection, args)
    with apply_ttr(model, selection, n_extra_tokens=args.n_extra_tokens,
                   scale=args.scale,
                   normal_values=args.normal_values) as patched:
        from saga.metrics import infer_num_prefix_tokens
        print(f"  patched   num_prefix_tokens="
              f"{infer_num_prefix_tokens(patched)}")
        if do_eval:
            print("== eval (full 50k) ==")
            run_eval(patched, args, device, sha, extra, eval_out)
        if do_diag:
            print("== diagnose (frozen 10k split) ==")
            run_diagnose(patched, args, device, sha, extra, diag_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
