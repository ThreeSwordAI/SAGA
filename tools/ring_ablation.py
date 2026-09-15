#!/usr/bin/env python3
"""
tools/ring_ablation.py — TASK-13 A2: the background-informativeness test
========================================================================
EVAL-ONLY. Re-evaluates an already fine-tuned TASK-08 checkpoint on the
official test split under four input conditions, all applied in PIXEL space
to the patch regions of the 14x14 grid — no model surgery, no retraining,
no change to the TASK-08 protocol.

  full            unmodified — must reproduce eval/test_final.json
  mask_ring1      the 44 ring-1 patch regions (one patch inside the border)
  mask_center44   the 44 most central patch regions (area-matched control)
  mask_random44   44 seeded random patch regions (position-agnostic control)

WHY RING 1: TASK-07 located the sink address — and SAGA's strongest gate
suppression — on ring 1, not on the outermost ring. The ring definition is
IMPORTED from analysis/address_analysis.py and never re-derived here.

THE HYPOTHESIS (to be tested, not assumed): positional suppression helps
when the border region is uninformative for the class (planes against sky)
and hurts when it carries class evidence (birds in habitat). It predicts a
LARGER accuracy drop under mask_ring1 for CUB than for Aircraft, relative to
each dataset's own mask_random44 control. This tool only measures; it takes
no position on which way the numbers fall.

TEST-SPLIT DISCIPLINE: TASK-08's rule is that the test split is touched
exactly once per *training* run, so that no checkpoint is ever selected on
it. This tool selects nothing — the checkpoint is frozen, already chosen on
val, and its sha256 is asserted against the one recorded in test_final.json.
Re-evaluating frozen weights cannot leak test information back into model
selection. The `full` condition re-measures the committed number as the
control that proves the path is the same one.

Output: results/finegrained/ring_ablation/<run_id>.json (atomic, written
LAST; a rerun whose checkpoint/seed/conditions match is a no-op).

Launch (one GPU, minutes per run):
    python tools/ring_ablation.py --matrix configs/ft_matrix.yaml \\
        --run ft_cub_vits_saga_bs1_f0 --stage_base $FT_STAGE
    python tools/ring_ablation.py --matrix configs/ft_matrix.yaml \\
        --all --stage_base $FT_STAGE
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.address_analysis import border_distance_map, ring_indices
from evaluation.e6_finegrained.data.ft_meta import (
    FTItemsDataset, build_eval_transform)
from evaluation.e6_finegrained.tools.train_ft import (
    evaluate, load_items, resolve_run_config, stage_dataset)
from saga.run_registry import file_sha256, git_sha
from tools.model_factory import build_model, load_checkpoint

PATCH = 16                  # vit_*_patch16_224
N_MASK = 44                 # ring 1 of the 14x14 grid; the controls match it
FULL_TOL = 0.05             # task-mandated reproduction tolerance, in points
CONDITIONS = ("full", "mask_ring1", "mask_center44", "mask_random44")
OUT_SUBDIR = Path("results") / "finegrained" / "ring_ablation"


# ── The four patch-index sets ────────────────────────────────────────────────

def center_indices(side: int, n: int):
    """The n most central positions of a side x side grid.

    Ordered by Chebyshev distance from the border DESCENDING (most interior
    first), ties broken by Euclidean distance from the grid centre ASCENDING,
    then by flat index — a total order, so the set is deterministic.

    A tie-break is unavoidable: the concentric rings of a 14x14 grid hold
    4, 12, 20, 28, ... positions counting inwards, so no union of whole rings
    equals 44 (4+12+20 = 36, the next ring would overshoot to 64). The exact
    indices are recorded in the output JSON rather than left implicit.
    """
    dist = border_distance_map(side).reshape(-1)
    centre = (side - 1) / 2.0
    rows, cols = np.divmod(np.arange(side * side), side)
    radius = np.hypot(rows - centre, cols - centre)
    order = sorted(range(side * side),
                   key=lambda i: (-int(dist[i]), float(radius[i]), i))
    return np.array(sorted(order[:n]), dtype=int)


def random_indices(side: int, n: int, seed: int):
    """n distinct positions drawn with a seeded generator — the
    position-agnostic control. Deterministic given (side, n, seed)."""
    rng = np.random.default_rng(seed)
    return np.array(sorted(rng.choice(side * side, size=n, replace=False)),
                    dtype=int)


def condition_indices(side: int, seed: int) -> dict:
    """The masked-patch index set of every condition. All three mask
    conditions are AREA-MATCHED by construction (N_MASK patches each)."""
    sets = {
        "full": np.array([], dtype=int),
        "mask_ring1": ring_indices(side, 1),
        "mask_center44": center_indices(side, N_MASK),
        "mask_random44": random_indices(side, N_MASK, seed),
    }
    for name, idx in sets.items():
        if name == "full":
            continue
        if len(idx) != N_MASK:
            raise RuntimeError(
                f"condition {name} masks {len(idx)} patches, expected "
                f"{N_MASK} — the three mask conditions must be area-matched")
        if len(set(idx.tolist())) != len(idx):
            raise RuntimeError(f"condition {name} has duplicate indices")
        if idx.min() < 0 or idx.max() >= side * side:
            raise RuntimeError(f"condition {name} has out-of-grid indices")
    return sets


# ── Masking: pixel space, post-resize, pre-normalize ─────────────────────────

class PatchMask:
    """Replace whole patch regions of a CHW float tensor with a per-channel
    fill. Applied AFTER ToTensor and BEFORE Normalize, so the fill is stated
    in the same [0,1] pixel space the images live in ("post-resize,
    pre-normalize")."""

    def __init__(self, indices, fill, img_size: int, side: int):
        self.indices = tuple(int(i) for i in indices)
        self.fill = tuple(float(v) for v in fill)
        self.img_size = int(img_size)
        self.side = int(side)
        self.patch = self.img_size // self.side

    def __call__(self, t):
        fill = torch.tensor(self.fill, dtype=t.dtype).view(-1, 1, 1)
        p = self.patch
        for i in self.indices:
            r, c = divmod(i, self.side)
            t[:, r * p:(r + 1) * p, c * p:(c + 1) * p] = fill
        return t


def _assert_legacy_tail(tfms):
    """The TASK-08 eval pipeline ends ToTensor() -> Normalize(). The mask
    goes between them, so if that tail ever changes this tool must be
    revisited rather than silently masking in the wrong space."""
    if not isinstance(tfms[-1], T.Normalize):
        raise RuntimeError(
            "build_eval_transform no longer ends with Normalize — "
            "ring_ablation inserts its mask before that Normalize and must "
            "be revisited")
    if not isinstance(tfms[-2], T.ToTensor):
        raise RuntimeError(
            "build_eval_transform no longer has ToTensor as its penultimate "
            "step — the mask's [0,1] pixel space is no longer guaranteed")


def masked_eval_transform(img_size: int, side: int, indices, fill):
    """The TASK-08 eval transform, with the mask inserted before Normalize.

    For the `full` condition (no indices) the ORIGINAL object is returned
    unchanged, so that condition runs the TASK-08 eval path itself rather
    than a reconstruction of it."""
    base = build_eval_transform(img_size)
    tfms = list(base.transforms)
    _assert_legacy_tail(tfms)
    if indices is None or len(indices) == 0:
        return base
    tfms.insert(len(tfms) - 1, PatchMask(indices, fill, img_size, side))
    return T.Compose(tfms)


def pixel_mean_transform(img_size: int):
    """The eval transform MINUS its Normalize — the pipeline whose output
    space the fill value is expressed in."""
    tfms = list(build_eval_transform(img_size).transforms)
    _assert_legacy_tail(tfms)
    return T.Compose(tfms[:-1])


def dataset_channel_mean(items, data_root, img_size, batch_size, workers):
    """Per-channel mean of the dataset, post-resize/crop and pre-normalize.

    Computed over the TRAIN items only — never the test split — so this tool
    cannot be said to have looked at test pixels for anything but the four
    scored evaluations. Exact and order-independent (a sum over pixels).
    """
    ds = FTItemsDataset(data_root, items, transform=pixel_mean_transform(img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    total = torch.zeros(3, dtype=torch.float64)
    n_pix = 0
    for images, _ in loader:
        total += images.to(torch.float64).sum(dim=(0, 2, 3))
        n_pix += images.shape[0] * images.shape[2] * images.shape[3]
    if n_pix == 0:
        raise RuntimeError("no images to compute the dataset mean from")
    return [float(v) for v in (total / n_pix)], len(ds)


# ── Model ────────────────────────────────────────────────────────────────────

def grid_side(model, img_size: int) -> int:
    """Patch-grid side, taken from the model where it states it."""
    side = img_size // PATCH
    pe = getattr(model, "patch_embed", None)
    stated = getattr(pe, "grid_size", None) if pe is not None else None
    if stated is not None:
        sr, sc = (stated, stated) if isinstance(stated, int) else stated
        if (sr, sc) != (side, side):
            raise RuntimeError(
                f"model patch grid {sr}x{sc} != {side}x{side} derived from "
                f"img_size {img_size} / patch {PATCH}")
    if side * side != 196:
        raise RuntimeError(
            f"patch grid is {side}x{side} = {side * side}, not the 196 the "
            f"TASK-07 ring definition and SAGA's phi=[H,196] are keyed to")
    return side


def load_finetuned(cfg, run_dir: Path, device, n_classes: int, final: dict):
    """Rebuild the fine-tuned model and load ckpt/best.pth — the exact
    weights test_final.json was produced from, enforced by sha256."""
    ckpt_path = run_dir / "ckpt" / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found — the fine-tuned weights live on the HPC "
            f"(results/**/ckpt/ is git-ignored); run this there")
    sha = file_sha256(ckpt_path)
    expected = final.get("finetuned_sha256")
    if expected and sha != expected:
        raise RuntimeError(
            f"fine-tuned sha256 mismatch for {ckpt_path}:\n  expected "
            f"{expected} (from eval/test_final.json)\n  found    {sha}\n"
            f"These are not the weights the committed test number came "
            f"from — refusing to ablate them.")

    model = build_model(cfg["arch"], cfg["variant"],
                        img_size=cfg["img_size"], num_classes=1000)
    model.head = nn.Linear(model.embed_dim, n_classes)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # strict=True: the head is already the fine-tuned n_classes shape, so a
    # zero-missing/zero-unexpected load is the only acceptable outcome
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), sha


# ── Driver ───────────────────────────────────────────────────────────────────

def atomic_json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def already_done(out_path: Path, ckpt_sha: str, seed: int, sets: dict) -> bool:
    """Idempotency: a rerun is a no-op only if the checkpoint, the seed AND
    every masked index set match what is on disk."""
    if not out_path.exists():
        return False
    try:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"  {out_path.name} unreadable — redoing", flush=True)
        return False
    if prev.get("finetuned_sha256") != ckpt_sha or prev.get("seed") != seed:
        return False
    prev_idx = prev.get("masked_indices", {})
    for name, idx in sets.items():
        if prev_idx.get(name) != [int(i) for i in idx]:
            return False
    return set(prev.get("top1", {})) == set(CONDITIONS)


def run_one(cfg, run_id, data_root, out_root: Path, seed: int, device,
            batch_size=None, workers=None, runs_root: Path = None):
    runs_root = runs_root or (REPO_ROOT / "results" / "runs")
    run_dir = runs_root / run_id
    final_path = run_dir / "eval" / "test_final.json"
    if not final_path.exists():
        raise FileNotFoundError(
            f"{final_path} not found — {run_id} has no committed test number "
            f"to reproduce; fine-tune it first")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final.get("smoke"):
        raise RuntimeError(f"{run_id} is a smoke artifact — refusing")

    batch_size = batch_size or cfg["batch_size"]
    workers = cfg["workers"] if workers is None else workers

    (train_items, _val_items, test_items, n_classes,
     split_sha) = load_items(cfg, data_root)

    model, ckpt_sha = load_finetuned(cfg, run_dir, device, n_classes, final)
    side = grid_side(model, cfg["img_size"])
    sets = condition_indices(side, seed)

    out_path = out_root / f"{run_id}.json"
    if already_done(out_path, ckpt_sha, seed, sets):
        print(f"SKIP {run_id} — {out_path.name} matches this checkpoint, "
              f"seed and index sets", flush=True)
        return "skipped"

    fill, n_fill_images = dataset_channel_mean(
        train_items, data_root, cfg["img_size"], batch_size, workers)
    print(f"  fill = dataset train per-channel mean "
          f"{[round(v, 5) for v in fill]} over {n_fill_images} images",
          flush=True)

    top1, top5, n_images = {}, {}, None
    t0 = time.time()
    for name in CONDITIONS:
        tfm = masked_eval_transform(cfg["img_size"], side, sets[name], fill)
        ds = FTItemsDataset(data_root, test_items, transform=tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers,
                            pin_memory=device.type == "cuda")
        with torch.no_grad():
            a1, a5, n = evaluate(model, loader, device)
        if n != len(ds):
            raise RuntimeError(f"{name}: evaluated {n} of {len(ds)} images")
        if n_images is None:
            n_images = n
        elif n != n_images:
            raise RuntimeError("conditions disagree on the image count")
        top1[name], top5[name] = round(a1, 3), round(a5, 3)
        print(f"  {name:<14} top1 {a1:7.3f}  top5 {a5:7.3f}", flush=True)

    # the mandated control: `full` must reproduce the committed number
    committed = float(final["top1"])
    delta = top1["full"] - committed
    ok = abs(delta) <= FULL_TOL
    print(f"  full-reproduction: {top1['full']:.3f} vs committed "
          f"{committed:.3f}  (delta {delta:+.3f}, tol {FULL_TOL}) -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

    result = {
        "run_id": run_id,
        "dataset": cfg["dataset"],
        "arch": cfg["arch"],
        "variant": cfg["variant"],
        "ft_seed": cfg["ft_seed"],
        "seed": seed,
        "conditions": list(CONDITIONS),
        "top1": top1,
        "top5": top5,
        "drop_vs_full": {c: round(top1["full"] - top1[c], 3)
                         for c in CONDITIONS},
        "n_images": n_images,
        "n_classes": n_classes,
        "grid_side": side,
        "patch_px": PATCH,
        "n_masked_patches": {c: int(len(sets[c])) for c in CONDITIONS},
        "masked_indices": {c: [int(i) for i in sets[c]] for c in CONDITIONS},
        "fill_value": [round(v, 6) for v in fill],
        "fill_source": "dataset train split, per-channel mean, "
                       "post-resize/crop, pre-normalize",
        "fill_n_images": n_fill_images,
        "full_reproduces_test_final": bool(ok),
        "full_minus_committed": round(delta, 3),
        "full_tolerance": FULL_TOL,
        "committed_test_top1": committed,
        "finetuned_ckpt": "ckpt/best.pth",
        "finetuned_sha256": ckpt_sha,
        "backbone_run": final.get("backbone_run"),
        "backbone_sha256": final.get("backbone_sha256"),
        "val_split": cfg["val_split"],
        "val_split_sha256": split_sha,
        "eval_seconds": round(time.time() - t0, 1),
        "git_sha": git_sha(),
    }
    atomic_json_dump(result, out_path)
    print(f"  -> {out_path}", flush=True)
    if not ok:
        print(f"WARNING {run_id}: full condition did NOT reproduce "
              f"test_final.json within {FULL_TOL}", flush=True)
    return "ok" if ok else "full_mismatch"


def main():
    ap = argparse.ArgumentParser("TASK-13 ring ablation (eval-only)")
    ap.add_argument("--matrix", default="configs/ft_matrix.yaml")
    ap.add_argument("--run", help="one run_id")
    ap.add_argument("--all", action="store_true",
                    help="every run in the matrix that has a test_final.json")
    ap.add_argument("--data_root", help="staged dataset root (skips staging)")
    ap.add_argument("--stage_base", help="node-local dir to stage into")
    ap.add_argument("--runs_root", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds mask_random44; recorded in every output")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    if bool(args.run) == bool(args.all):
        ap.error("give exactly one of --run or --all")
    if not args.data_root and not args.stage_base:
        ap.error("give --data_root or --stage_base")

    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = REPO_ROOT / matrix_path
    with open(matrix_path) as f:
        matrix = yaml.safe_load(f)

    runs_root = Path(args.runs_root) if args.runs_root else \
        REPO_ROOT / "results" / "runs"
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / OUT_SUBDIR
    device = torch.device(args.device)

    if args.run:
        run_ids = [args.run]
    else:
        run_ids = [r for r in matrix["runs"]
                   if (runs_root / r / "eval" / "test_final.json").exists()]
        print(f"{len(run_ids)} of {len(matrix['runs'])} matrix runs have a "
              f"test_final.json", flush=True)

    # stage each dataset once, not once per run
    staged, results, failures = {}, {}, []
    for run_id in run_ids:
        print(f"\n=== {run_id} ===", flush=True)
        try:
            cfg = resolve_run_config(matrix, run_id)
            ds = cfg["dataset"]
            if args.data_root:
                data_root = args.data_root
            else:
                if ds not in staged:
                    staged[ds] = stage_dataset(cfg, args.stage_base)
                data_root = staged[ds]
            results[run_id] = run_one(
                cfg, run_id, data_root, out_root, args.seed, device,
                args.batch_size, args.workers, runs_root)
        except Exception as exc:                       # noqa: BLE001
            print(f"FAILED {run_id}: {type(exc).__name__}: {exc}", flush=True)
            failures.append((run_id, f"{type(exc).__name__}: {exc}"))

    print("\n" + "=" * 62)
    for state in ("ok", "skipped", "full_mismatch"):
        hits = [r for r, s in results.items() if s == state]
        print(f"  {state:<14} {len(hits)}")
    print(f"  failed         {len(failures)}")
    for run_id, msg in failures:
        print(f"    {run_id}: {msg}")
    mismatches = [r for r, s in results.items() if s == "full_mismatch"]
    if mismatches:
        print(f"  FULL-CONDITION MISMATCHES (report these): {mismatches}")
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
