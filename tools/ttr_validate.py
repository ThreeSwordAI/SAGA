#!/usr/bin/env python3
"""
tools/ttr_validate.py
=====================
TASK-10 A3 — the validation gate for test-time registers (saga/ttr.py).

Runs ONE baseline checkpoint through an `--n-neurons` sweep and reports, per
sweep value, the sink count under the cell's CANON tau and the top-1 on a
fixed 5 000-image subset of the frozen diagnostic split. Prints one
PASS/FAIL line.

    python tools/ttr_validate.py \
        --ckpt results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth \
        --arch vit_small --recipe mixup \
        --n-neurons 4,8,16,32,64 \
        --data $STAGE_DIR

The gate exists because TTR may simply not transfer to supervised ViT-S/B
(the paper used CLIP / DINOv2 / DeiT-III). **On FAIL for every n, the full
matrix must NOT be run** — the validated negative is the deliverable. The
exit code says so: 0 = PASS, 3 = FAIL (a real failure, honestly measured),
anything else = the tool itself broke.

tau NOTE: the threshold is read from the committed
`results/diagsplit/fixed_thresholds_canon.json` for the base cell
`<arch>|<recipe>` and is NEVER recalibrated on a patched model — TTR is a
modification of that cell's baseline, not a new cell (TASK-10 acceptance).
"""

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.metrics import (infer_num_prefix_tokens, oversmoothing_pairwise,
                          sink_counts_fixed, sink_counts_mad, token_norms)
from saga.run_registry import file_sha256, git_sha
from saga.ttr import (CRITERION_MEAN_ABS, CRITERIA, apply_ttr,
                      find_register_neurons, top_k, write_neurons_json)
from tools.build_diag_split import DiagSplitDataset
from tools.eval import build_val_transform
from tools.model_factory import build_model, load_checkpoint

CANON_THR_FILE = "results/diagsplit/fixed_thresholds_canon.json"

#: How the two disjoint image subsets are carved out of the frozen split.
#: Deterministic, class-balanced, no RNG — recorded in the output JSON so the
#: selection can never become folklore.
EVAL_RULE = ("first --eval-per-class items of each class in split order "
             "(class-balanced, disjoint from the find pool)")
FIND_RULE = ("evenly strided over the remaining items (per-class index >= "
             "--eval-per-class), so the scan set spans many classes rather "
             "than the first few")


def infer_base_run_id(ckpt: Path) -> str:
    """`results/runs/<id>/ckpt/last.pth` -> `<id>`.

    Legacy checkpoints do not live under results/runs, so they must pass
    --base-run-id explicitly rather than get a guessed identity.
    """
    parts = ckpt.resolve().as_posix().split("/")
    for i in range(len(parts) - 1):
        if parts[i] == "runs" and i + 1 < len(parts):
            return parts[i + 1]
    raise ValueError(
        f"cannot infer a base run id from {ckpt} — pass --base-run-id "
        f"(expected a .../runs/<run_id>/ckpt/... path)")


def resolve_canon_tau(thr_file: Path, arch: str, recipe: str):
    """(tau, key) for the base cell, or a hard error naming what is there."""
    with open(thr_file) as f:
        thresholds = json.load(f)
    key = f"{arch}|{recipe}"
    if key not in thresholds:
        available = sorted(k for k in thresholds
                           if "|" in k and not k.startswith("source"))
        raise KeyError(
            f"{thr_file} has no canon tau for {key!r}; it holds {available}. "
            f"TTR must reuse the base cell's tau — refusing to invent one.")
    return float(thresholds[key]), key


def carve_subsets(dataset, eval_per_class: int, n_find: int):
    """Two disjoint, deterministic index lists: (eval_idx, find_idx)."""
    seen: dict = {}
    eval_idx, pool = [], []
    for i, (_, label) in enumerate(dataset.items):
        n = seen.get(label, 0)
        seen[label] = n + 1
        (eval_idx if n < eval_per_class else pool).append(i)
    if not pool:
        raise ValueError(
            f"--eval-per-class {eval_per_class} consumes the whole split; no "
            f"images are left to scan for register neurons")
    stride = max(1, len(pool) // max(1, n_find))
    find_idx = pool[::stride][:n_find]
    return eval_idx, find_idx


@torch.no_grad()
def measure(model, loader, tau: float, device) -> dict:
    """One fp32 pass: top-1/top-5 and last-block sink / oversmoothing stats.

    Reads ``infer_num_prefix_tokens`` off the model it is HANDED, so the
    patched proxy's widened prefix is honoured and the patch slice is always
    the same 196 tokens.
    """
    model.eval()
    P = infer_num_prefix_tokens(model)
    blocks = model.blocks
    store: dict = {}
    handle = blocks[len(blocks) - 1].register_forward_hook(
        lambda mod, inp, out: store.__setitem__("x", out.detach()))

    c1 = c5 = n = 0
    sink_canon = sink_mad = over = 0.0
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)

            top5 = logits.float().topk(5, dim=1).indices
            correct = top5 == labels[:, None]
            c1 += int(correct[:, 0].sum())
            c5 += int(correct.any(dim=1).sum())

            patch = store["x"][:, P:, :].float()
            norms = token_norms(patch)
            sink_canon += float(sink_counts_fixed(norms, tau).sum())
            sink_mad += float(sink_counts_mad(norms, k=5.0).sum())
            over += float(oversmoothing_pairwise(patch).sum())

            n += images.shape[0]
            store.clear()
    finally:
        handle.remove()

    if n == 0:
        raise RuntimeError("measure(): the loader yielded no images")
    return {
        "n_images": n,
        "num_prefix_tokens": P,
        "top1": 100.0 * c1 / n,
        "top5": 100.0 * c5 / n,
        "sink_fixed_canon": sink_canon / n,
        "sink_mad_k5": sink_mad / n,
        "oversmooth_pairwise": over / n,
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def verdict(unpatched: dict, rows: list, min_reduction: float,
            max_top1_drop: float) -> dict:
    """Apply the PASS criterion and return the full reasoning, not just a bool.

    TASK-10 A3 states the criterion in words ("reduces the sink count
    materially ... without destroying accuracy") and leaves the numbers to
    the implementation, so both thresholds are CLI flags and both are printed
    and recorded with the result.
    """
    base_sink = unpatched["sink_fixed_canon"]
    base_top1 = unpatched["top1"]
    passing = []
    for r in rows:
        red = (None if base_sink <= 0.0
               else (base_sink - r["sink_fixed_canon"]) / base_sink)
        drop = base_top1 - r["top1"]
        r["sink_reduction_frac"] = red
        r["top1_drop"] = drop
        r["meets_sink"] = bool(red is not None and red >= min_reduction)
        r["meets_top1"] = bool(drop <= max_top1_drop)
        r["passes"] = bool(r["meets_sink"] and r["meets_top1"])
        if r["passes"]:
            passing.append(r)

    best = max(passing, key=lambda r: r["sink_reduction_frac"], default=None)
    return {
        "passed": bool(passing),
        "criterion": (
            f"PASS iff some n gives sink_fixed_canon reduced by >= "
            f"{min_reduction:.0%} vs unpatched AND top1 drop <= "
            f"{max_top1_drop:.2f} points"),
        "min_sink_reduction": min_reduction,
        "max_top1_drop": max_top1_drop,
        "undefined_reason": (None if base_sink > 0.0 else
                             "unpatched sink_fixed_canon is 0 — a relative "
                             "reduction is undefined, so no n can pass"),
        "best_n_neurons": (None if best is None else best["n_neurons"]),
        "n_passing": len(passing),
    }


def main():
    p = argparse.ArgumentParser(
        description="TASK-10 A3 validation gate for test-time registers.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", required=True, choices=["vit_small", "vit_base"])
    p.add_argument("--recipe", required=True, choices=["mixup", "nomix"],
                   help="recipe_actual of the BASE cell — selects the canon tau")
    p.add_argument("--variant", default="baseline",
                   choices=["baseline", "registers", "saga"])
    p.add_argument("--n-neurons", default="4,8,16,32,64",
                   help="comma-separated sweep values")
    p.add_argument("--data", required=True, metavar="ROOT",
                   help="ImageNet root containing val/ (ImageFolder layout)")
    p.add_argument("--split-file",
                   default="results/diagsplit/val_diag_split.json")
    p.add_argument("--thr-file", default=CANON_THR_FILE)
    p.add_argument("--base-run-id", default=None,
                   help="override the run id inferred from --ckpt")
    p.add_argument("--out-root", default="results/ttr")
    p.add_argument("--eval-per-class", type=int, default=5,
                   help="images per class in the eval subset (5 -> 5000)")
    p.add_argument("--n-find-images", type=int, default=500,
                   help="images scanned for register neurons (paper: 500)")
    p.add_argument("--layer-range", default=None, metavar="LO,HI",
                   help="half-open block range to scan (default: all blocks)")
    p.add_argument("--criterion", default=CRITERION_MEAN_ABS,
                   choices=sorted(CRITERIA))
    p.add_argument("--n-extra-tokens", type=int, default=1)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--normal-values", default="zero",
                   choices=["zero", "mean", "same"])
    p.add_argument("--min-sink-reduction", type=float, default=0.50)
    p.add_argument("--max-top1-drop", type=float, default=1.00)
    p.add_argument("--top-n-stored", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--find-batch-size", type=int, default=32,
                   help="smaller: the scan holds every layer's MLP hidden "
                        "activations for the batch at once")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sweep = [int(s) for s in args.n_neurons.split(",") if s.strip()]
    if not sweep or any(n < 1 for n in sweep):
        raise SystemExit(f"--n-neurons must be positive integers, "
                         f"got {args.n_neurons!r}")
    layer_range = None
    if args.layer_range:
        lo, hi = (int(v) for v in args.layer_range.split(","))
        layer_range = (lo, hi)

    ckpt = Path(args.ckpt)
    run_id = args.base_run_id or infer_base_run_id(ckpt)
    # TASK-10 A2.1: results/ttr/<base_run_id>/. The patched model's eval/diag
    # artifacts go somewhere else on purpose — results/runs/ttr_<base_run_id>/
    # (A5.3) — so that the existing collectors, which glob results/runs/*/,
    # find them without a change.
    out_dir = Path(args.out_root) / run_id
    tau, tau_key = resolve_canon_tau(Path(args.thr_file), args.arch,
                                     args.recipe)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = build_model(args.arch, args.variant)
    load_checkpoint(model, ckpt)
    model = model.to(device).eval()
    sha = file_sha256(ckpt)

    dataset = DiagSplitDataset(args.data, args.split_file,
                               transform=build_val_transform(224))
    eval_idx, find_idx = carve_subsets(dataset, args.eval_per_class,
                                       args.n_find_images)
    common = dict(shuffle=False, num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"))
    eval_loader = DataLoader(Subset(dataset, eval_idx),
                             batch_size=args.batch_size, **common)
    find_loader = DataLoader(Subset(dataset, find_idx),
                             batch_size=args.find_batch_size, **common)

    print(f"TTR validate: {args.arch}/{args.variant} recipe={args.recipe}")
    print(f"  ckpt      {ckpt}")
    print(f"  sha256    {sha}")
    print(f"  canon tau {tau}  (key {tau_key} from {args.thr_file})")
    print(f"  eval      {len(eval_idx)} images  [{EVAL_RULE}]")
    print(f"  find      {len(find_idx)} images  [{FIND_RULE}]")
    print(f"  sweep     {sweep}   criterion={args.criterion}")
    print(f"  layers    {layer_range or 'all'}   "
          f"n_extra_tokens={args.n_extra_tokens} "
          f"normal_values={args.normal_values} scale={args.scale}")
    print()

    unpatched = measure(model, eval_loader, tau, device)
    print(f"  unpatched: top1={unpatched['top1']:.3f}  "
          f"sink_canon={unpatched['sink_fixed_canon']:.4f}  "
          f"sink_mad={unpatched['sink_mad_k5']:.4f}  "
          f"over={unpatched['oversmooth_pairwise']:.4f}")

    ranking = find_register_neurons(
        model, find_loader, None, layer_range,
        outlier_tau=tau, device=device, seed=args.seed,
        criterion=args.criterion, max_images=args.n_find_images)
    scan_stats = find_register_neurons.last_scan_stats
    neurons_path = out_dir / "neurons.json"
    write_neurons_json(
        neurons_path, ranking, criterion=args.criterion, outlier_tau=tau,
        tau_key=tau_key, tau_source=args.thr_file, seed=args.seed,
        arch=args.arch, variant=args.variant, ckpt=ckpt.resolve().as_posix(),
        ckpt_sha256=sha, scan_stats=scan_stats,
        top_n_stored=args.top_n_stored)
    print(f"  scanned {scan_stats['n_images_seen']} images, "
          f"{scan_stats['n_images_scored']} had an outlier at tau={tau}")
    print(f"  wrote {neurons_path}")
    print(f"  top-5 neurons (layer, neuron, score): "
          f"{[(l, n, round(s, 4)) for l, n, s in ranking[:5]]}")
    print()

    rows = []
    for n_sel in sweep:
        sel = top_k(ranking, n_sel)
        with apply_ttr(model, sel, n_extra_tokens=args.n_extra_tokens,
                       scale=args.scale,
                       normal_values=args.normal_values) as patched:
            m = measure(patched, eval_loader, tau, device)
        layers_hit = sorted({l for l, _, _ in sel})
        m.update(n_neurons=n_sel, n_layers_touched=len(layers_hit),
                 layers_touched=",".join(str(l) for l in layers_hit))
        rows.append(m)
        print(f"  n={n_sel:<4} top1={m['top1']:.3f}  "
              f"sink_canon={m['sink_fixed_canon']:.4f}  "
              f"sink_mad={m['sink_mad_k5']:.4f}  "
              f"over={m['oversmooth_pairwise']:.4f}  "
              f"layers={m['layers_touched']}")

    v = verdict(unpatched, rows, args.min_sink_reduction, args.max_top1_drop)

    fields = ["n_neurons", "n_images", "num_prefix_tokens", "top1", "top5",
              "sink_fixed_canon", "sink_mad_k5", "oversmooth_pairwise",
              "sink_reduction_frac", "top1_drop", "meets_sink", "meets_top1",
              "passes", "n_layers_touched", "layers_touched"]
    # n_neurons=0 is the unpatched reference row, so the CSV is self-contained
    base_row = dict(unpatched, n_neurons=0, sink_reduction_frac=None,
                    top1_drop=0.0, meets_sink="", meets_top1="", passes="",
                    n_layers_touched=0, layers_touched="")
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=fields, extrasaction="ignore",
                       lineterminator="\n")
    w.writeheader()
    for r in [base_row] + rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k, ""))
                    for k in fields})
    out_csv = out_dir / "sweep.csv"
    atomic_write_text(out_csv, sio.getvalue())

    summary = {
        "schema": "saga.ttr.validate/v1",
        "base_run_id": run_id,
        "arch": args.arch,
        "variant": args.variant,
        "recipe": args.recipe,
        "ckpt": ckpt.resolve().as_posix(),
        "ckpt_sha256": sha,
        "git_sha": git_sha(),
        "canon_tau": tau,
        "tau_key": tau_key,
        "tau_source": args.thr_file,
        "tau_recalibrated_on_patched_model": False,
        "criterion": args.criterion,
        "criterion_description": CRITERIA[args.criterion],
        "n_extra_tokens": args.n_extra_tokens,
        "scale": args.scale,
        "normal_values": args.normal_values,
        "layer_range": list(layer_range) if layer_range else None,
        "eval_subset_rule": EVAL_RULE,
        "eval_per_class": args.eval_per_class,
        "n_eval_images": len(eval_idx),
        "find_subset_rule": FIND_RULE,
        "n_find_images": len(find_idx),
        "split_file": args.split_file,
        "seed": args.seed,
        "amp": False,
        "scan_stats": scan_stats,
        "unpatched": unpatched,
        "sweep": rows,
        "verdict": v,
        "neurons_json": neurons_path.as_posix(),
        "sweep_csv": out_csv.as_posix(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(out_dir / "validate.json",
                      json.dumps(summary, indent=2))

    print()
    print(f"  {v['criterion']}")
    if v["undefined_reason"]:
        print(f"  NOTE: {v['undefined_reason']}")
    label = "PASS" if v["passed"] else "FAIL"
    detail = (f"best n={v['best_n_neurons']} "
              f"({v['n_passing']} of {len(rows)} sweep values qualify)"
              if v["passed"] else
              "no sweep value met both conditions — DO NOT run the full "
              "matrix; the validated negative is the deliverable (A3)")
    print(f"TTR VALIDATION {label}: {detail}")
    print(f"  wrote {out_csv}")
    print(f"  wrote {out_dir / 'validate.json'}")
    return 0 if v["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
