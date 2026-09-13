#!/usr/bin/env python3
"""
tools/ttr_paired_ci.py
======================
TASK-10 PHASE C, decision 1 — paired uncertainty on a TTR top-1 drop.

The A3 threshold is HELD at 1.00 points and is not revisited. ViT-B/mixup
misses it by 0.02 (drop 1.02 at n=24), and a bare FAIL under-reports that:
whether 1.02 is distinguishable from 1.00 at n=50 000 is an empirical
question, and it needs the PAIRED comparison, not the two marginals. The
marginal top-1 values in the eval JSONs cannot answer it — from them you can
only recover b - c, never b and c separately.

    python tools/ttr_paired_ci.py \
        --ckpt results/runs/e2r_vitb_mixup_baseline_s1/ckpt/last.pth \
        --arch vit_base --recipe mixup \
        --neurons-file results/ttr_midlayer/e2r_vitb_mixup_baseline_s1/neurons.json \
        --n-neurons 24 --data $STAGE_DIR \
        --out results/ttr_midlayer/e2r_vitb_mixup_baseline_s1/paired_ci.json

Evaluates the SAME images twice in the SAME order — unpatched, then TTR —
recording per-image top-1 correctness, and reports the 2x2 discordant table

        b = unpatched correct, patched wrong     (TTR broke it)
        c = unpatched wrong,   patched correct   (TTR fixed it)

with three statements about drop = top1_unpatched - top1_patched:

* a **paired bootstrap** percentile CI. Because each image contributes
  d_i = correct_patched - correct_unpatched in {-1, 0, +1}, and only the
  three counts matter, a bootstrap resample is exactly a multinomial draw
  over (b, c, concordant) — so this is the exact paired bootstrap, computed
  in O(B) rather than by resampling 50 000 rows B times.
* **McNemar's exact test** on (b, c), the standard paired test for two
  classifiers on one sample, plus its mid-p variant (less conservative).
* the comparison against the frozen 1.00 threshold, as a fact about the CI
  rather than a re-decision: `threshold_inside_ci` says whether 1.00 lies
  within the interval.

None of this changes the verdict. The verdict is whatever
tools/ttr_validate.py recorded; this only says how precisely the drop that
produced it was measured.
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
from saga.run_registry import file_sha256, git_sha
from saga.ttr import apply_ttr, load_neurons_json, neurons_to_dict, top_k
from tools.eval import IMAGENET_VAL_SIZE, build_val_transform
from tools.model_factory import build_model, load_checkpoint

SCHEMA = "saga.ttr.paired_ci/v1"


@torch.no_grad()
def per_image_correct(model, loader, device) -> np.ndarray:
    """Top-1 correctness per image, in loader order. -> bool[n]"""
    model.eval()
    out = []
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pred = model(images).float().argmax(dim=1)
        out.append((pred == targets).cpu().numpy())
        if step % 20 == 0:
            print(f"    [{step + 1}/{len(loader)}]", flush=True)
    return np.concatenate(out)


def paired_stats(unp: np.ndarray, pat: np.ndarray, n_boot: int, seed: int,
                 alpha: float = 0.05) -> dict:
    """Discordant counts, exact paired bootstrap CI, and McNemar.

    `drop` is positive when TTR COSTS accuracy, matching the gate's
    `top1_drop` convention.
    """
    if unp.shape != pat.shape:
        raise ValueError(f"shape mismatch {unp.shape} vs {pat.shape}")
    n = int(unp.size)
    b = int(np.sum(unp & ~pat))           # TTR broke it
    c = int(np.sum(~unp & pat))           # TTR fixed it
    n11 = int(np.sum(unp & pat))
    n00 = int(np.sum(~unp & ~pat))
    assert b + c + n11 + n00 == n

    top1_unp = 100.0 * (n11 + b) / n
    top1_pat = 100.0 * (n11 + c) / n
    drop = top1_unp - top1_pat

    # Exact paired bootstrap: only (b, c, concordant) matter, so a resample
    # is a multinomial draw over those three categories.
    rng = np.random.default_rng(seed)
    probs = [b / n, c / n, (n - b - c) / n]
    draws = rng.multinomial(n, probs, size=n_boot)
    boot_drop = 100.0 * (draws[:, 0] - draws[:, 1]) / n
    lo, hi = np.percentile(boot_drop, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    stats = {
        "n_images": n,
        "n_both_correct": n11,
        "n_both_wrong": n00,
        "b_unpatched_correct_patched_wrong": b,
        "c_unpatched_wrong_patched_correct": c,
        "n_discordant": b + c,
        "top1_unpatched": top1_unp,
        "top1_patched": top1_pat,
        "top1_drop": drop,
        "bootstrap": {
            "method": "exact paired bootstrap via multinomial over "
                      "(b, c, concordant); equivalent to resampling the "
                      "paired per-image outcomes",
            "n_boot": n_boot,
            "seed": seed,
            "alpha": alpha,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "se": float(np.std(boot_drop, ddof=1)),
        },
    }

    try:
        from scipy.stats import binomtest
        res = binomtest(b, b + c, 0.5) if (b + c) else None
        stats["mcnemar"] = {
            "test": "exact binomial on the discordant pairs (McNemar)",
            "p_value": float(res.pvalue) if res is not None else None,
            "p_value_midp": (_midp(b, c) if (b + c) else None),
            "backend": "scipy.stats.binomtest",
        }
    except ImportError:
        stats["mcnemar"] = {"test": "unavailable", "p_value": None,
                            "p_value_midp": None, "backend": "scipy missing"}
    return stats


def _midp(b: int, c: int) -> float:
    """Mid-p McNemar: exact p minus half the point mass at the observed
    value — the conventional correction for the exact test's conservatism."""
    from scipy.stats import binom, binomtest
    m = b + c
    p_exact = float(binomtest(b, m, 0.5).pvalue)
    return max(0.0, min(1.0, p_exact - float(binom.pmf(b, m, 0.5))))


def main():
    p = argparse.ArgumentParser(
        description="Paired bootstrap CI + McNemar for one TTR operating point.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", required=True, choices=["vit_small", "vit_base"])
    p.add_argument("--recipe", required=True, choices=["mixup", "nomix"])
    p.add_argument("--variant", default="baseline",
                   choices=["baseline", "registers", "saga"])
    p.add_argument("--neurons-file", required=True)
    p.add_argument("--n-neurons", type=int, required=True)
    p.add_argument("--data", required=True, metavar="ROOT")
    p.add_argument("--out", required=True)
    p.add_argument("--n-extra-tokens", type=int, default=1)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--normal-values", default="zero",
                   choices=["zero", "mean", "same"])
    p.add_argument("--n-boot", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=1.00,
                   help="the FROZEN A3 max-top1-drop, reported against but "
                        "never used to re-decide anything")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    payload = load_neurons_json(args.neurons_file)
    if payload["arch"] != args.arch:
        raise SystemExit(f"--arch {args.arch} disagrees with the neurons "
                         f"file's {payload['arch']!r}")
    selection = top_k(payload["neurons"], args.n_neurons)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sha = file_sha256(args.ckpt)

    model = build_model(args.arch, args.variant)
    load_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    from timm.data import create_dataset
    dataset = create_dataset("imagefolder", root=args.data, split="val",
                             is_training=False)
    dataset.transform = build_val_transform(224)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    print(f"paired CI: {args.arch}|{args.recipe} n_neurons={args.n_neurons} "
          f"layers {payload['layer_range']}")
    print(f"  {len(dataset)} images, fp32, same order for both passes")
    print("  pass 1/2: UNPATCHED")
    unp = per_image_correct(model, loader, device)
    print("  pass 2/2: TTR")
    with apply_ttr(model, selection, n_extra_tokens=args.n_extra_tokens,
                   scale=args.scale,
                   normal_values=args.normal_values) as patched:
        pat = per_image_correct(patched, loader, device)

    n = int(unp.size)
    assert n == len(dataset) == IMAGENET_VAL_SIZE, (
        f"scored {n} images but the dataset has {len(dataset)} "
        f"(expected {IMAGENET_VAL_SIZE}) — refusing to write results")

    stats = paired_stats(unp, pat, args.n_boot, args.seed, args.alpha)
    lo, hi = stats["bootstrap"]["ci_low"], stats["bootstrap"]["ci_high"]
    result = {
        "schema": SCHEMA,
        "arch": args.arch,
        "recipe": args.recipe,
        "variant": args.variant,
        "ckpt": Path(args.ckpt).resolve().as_posix(),
        "ckpt_sha256": sha,
        "git_sha": git_sha(),
        "neurons_file": str(args.neurons_file),
        "criterion": payload["criterion"],
        "layer_range": payload["layer_range"],
        "n_neurons": args.n_neurons,
        "layers_touched": sorted(neurons_to_dict(selection)),
        "n_extra_tokens": args.n_extra_tokens,
        "scale": args.scale,
        "normal_values": args.normal_values,
        "amp": "off",
        "seed": args.seed,
        **stats,
        "frozen_threshold": args.threshold,
        "threshold_inside_ci": bool(lo <= args.threshold <= hi),
        "note": ("The A3 verdict is NOT recomputed here and is not affected "
                 "by this file. The threshold is frozen at "
                 f"{args.threshold} and the cell's verdict stands as "
                 "tools/ttr_validate.py recorded it; these numbers only "
                 "quantify how precisely the drop was measured."),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # per-image outcomes first (packed bools, ~6 KB each) so the CI can be
    # recomputed by any method without another GPU pass; JSON last, as the
    # completion marker.
    np.savez(out.with_suffix(".npz"),
             unpatched_correct=np.packbits(unp),
             patched_correct=np.packbits(pat),
             n_images=np.array([n]))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"  b (TTR broke) = {stats['b_unpatched_correct_patched_wrong']}   "
          f"c (TTR fixed) = {stats['c_unpatched_wrong_patched_correct']}   "
          f"discordant = {stats['n_discordant']}")
    print(f"  top1 {stats['top1_unpatched']:.3f} -> {stats['top1_patched']:.3f}"
          f"   drop = {stats['top1_drop']:.4f}")
    print(f"  {100 * (1 - args.alpha):.0f}% paired-bootstrap CI on the drop: "
          f"[{lo:.4f}, {hi:.4f}]")
    print(f"  McNemar exact p = {stats['mcnemar']['p_value']}")
    print(f"  frozen threshold {args.threshold} inside the CI: "
          f"{result['threshold_inside_ci']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
