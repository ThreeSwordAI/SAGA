#!/usr/bin/env python3
"""
tools/extract_gate.py
=====================
TASK-02C A2 — CPU-only checkpoint inspection. Two modes.

Gate extraction (default): for every SAGA checkpoint in the manifest, dump
the raw gate logits phi and per-layer statistics.

    python tools/extract_gate.py --manifest results/legacy/checkpoint_manifest.csv \
        --variant saga --out-dir results/legacy/gates

phi layout (from saga/gate.py): each block owns SpatialGate.phi, an
nn.Parameter of shape [num_heads, n_patches]; state-dict key
`blocks.{i}.attn.gate.phi`. Stacked across blocks -> [L, H, N]. Outputs per
checkpoint (small, committable):
- <stem>_phi.npz          key "phi", float32 [L, H, N]
- <stem>_phi_stats.json   per-layer stats of sigmoid(phi): mean/std/min/max,
                          frac<0.4, frac<0.25, frac>0.75, NaN/Inf flags —
                          written LAST (completion marker; requeue-safe).

Forensics mode: dump whatever training-state metadata every e2 checkpoint
carries (trainers save {'epoch','model','optimizer','scaler','best_top1',
'top1'}; older saves may lack some) -> one CSV row per file, MISSING for
absent keys.

    python tools/extract_gate.py --forensics \
        --manifest results/legacy/checkpoint_manifest.csv \
        --out results/legacy/ckpt_forensics.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.model_factory import extract_state_dict

MISSING = "MISSING"

FORENSICS_COLUMNS = [
    "stem", "path", "filename", "exp", "arch", "recipe", "variant", "seed",
    "sha256_manifest", "top_level_keys", "epoch", "top1", "best_top1",
    "last_lr", "optimizer_step_count", "n_optimizer_state_entries",
    "has_scaler", "has_ema", "n_model_tensors", "model_params_m",
]


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def manifest_rows(manifest, exp="e2", variant=None):
    rows = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if exp and r["exp"] != exp:
                continue
            if variant and r["variant"] != variant:
                continue
            rows.append(r)
    return rows


def stem_of(row):
    tag = row["filename"].removesuffix(".pth")
    return "_".join([row["exp"], row["arch"], row["recipe"], row["variant"],
                     row["seed"], tag])


# ─────────────────────────────────────────────────────────────────────────────
# Gate extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_phi(state_dict) -> np.ndarray:
    """Stack blocks.{i}.attn.gate.phi -> float32 [L, H, N] (raw logits)."""
    keys = sorted((k for k in state_dict if k.endswith(".attn.gate.phi")),
                  key=lambda k: int(k.split(".")[1]))
    if not keys:
        raise KeyError("no '*.attn.gate.phi' keys — not a SAGA checkpoint?")
    return np.stack([state_dict[k].float().numpy() for k in keys])


def phi_stats(phi: np.ndarray) -> dict:
    """Per-layer stats of the gate values sigmoid(phi)."""
    gate = 1.0 / (1.0 + np.exp(-phi.astype(np.float64)))
    layers = []
    for li in range(phi.shape[0]):
        g = gate[li]
        finite = np.isfinite(phi[li])
        layers.append({
            "layer": li,
            "mean_gate": float(np.nanmean(g)),
            "std_gate": float(np.nanstd(g)),
            "min_gate": float(np.nanmin(g)),
            "max_gate": float(np.nanmax(g)),
            "frac_below_0.4": float(np.mean(g < 0.4)),
            "frac_below_0.25": float(np.mean(g < 0.25)),
            "frac_above_0.75": float(np.mean(g > 0.75)),
            "has_nan": bool(np.isnan(phi[li]).any()),
            "has_inf": bool((~finite & ~np.isnan(phi[li])).any()),
        })
    return {"shape_LHN": list(phi.shape), "layers": layers}


def run_gate_extraction(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for r in rows:
        stem = stem_of(r)
        npz_path = out_dir / f"{stem}_phi.npz"
        stats_path = out_dir / f"{stem}_phi_stats.json"

        if stats_path.exists():
            try:
                old = json.load(open(stats_path))
            except ValueError:
                old = {}
            if old.get("ckpt_sha256") == r["sha256"] and npz_path.exists():
                skipped += 1
                print(f"  SKIP {stem} (up to date)")
                continue

        ckpt = load_ckpt(r["path"])
        state, _ = extract_state_dict(ckpt)
        phi = extract_phi(state)

        # npz first, stats JSON last: the JSON is the completion marker —
        # and a stale marker is invalidated BEFORE the npz is rewritten
        if stats_path.exists():
            stats_path.unlink()
        np.savez(npz_path, phi=phi)
        stats = phi_stats(phi)
        stats.update({
            "ckpt": r["path"],
            "ckpt_sha256": r["sha256"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        written += 1
        print(f"  wrote {stem}_phi.npz + _phi_stats.json "
              f"(shape {stats['shape_LHN']})")
    print(f"gate extraction done: {written} written, {skipped} skipped")


# ─────────────────────────────────────────────────────────────────────────────
# Forensics
# ─────────────────────────────────────────────────────────────────────────────

def forensics_row(row, ckpt) -> dict:
    out = {"stem": stem_of(row), "path": row["path"],
           "filename": row["filename"], "exp": row["exp"],
           "arch": row["arch"], "recipe": row["recipe"],
           "variant": row["variant"], "seed": row["seed"],
           "sha256_manifest": row["sha256"]}

    is_raw_state_dict = (isinstance(ckpt, dict) and ckpt
                         and all(torch.is_tensor(v) for v in ckpt.values()))
    if not isinstance(ckpt, dict) or is_raw_state_dict:
        out.update({k: MISSING for k in FORENSICS_COLUMNS if k not in out})
        out["top_level_keys"] = (
            f"<raw state_dict, {len(ckpt)} tensors>" if is_raw_state_dict
            else f"<raw {type(ckpt).__name__}>")
        return out

    out["top_level_keys"] = ";".join(sorted(ckpt.keys()))
    for key in ("epoch", "top1", "best_top1"):
        out[key] = ckpt.get(key, MISSING)
    out["has_scaler"] = "scaler" in ckpt
    out["has_ema"] = any("ema" in k.lower() for k in ckpt.keys())

    opt = ckpt.get("optimizer")
    if isinstance(opt, dict):
        groups = opt.get("param_groups", [])
        out["last_lr"] = groups[0].get("lr", MISSING) if groups else MISSING
        state = opt.get("state", {})
        out["n_optimizer_state_entries"] = len(state)
        steps = []
        for entry in state.values():
            s = entry.get("step")
            if s is not None:
                steps.append(int(s.item() if torch.is_tensor(s) else s))
        out["optimizer_step_count"] = max(steps) if steps else MISSING
    else:
        out["last_lr"] = MISSING
        out["optimizer_step_count"] = MISSING
        out["n_optimizer_state_entries"] = MISSING

    model = ckpt.get("model")
    if isinstance(model, dict):
        out["n_model_tensors"] = len(model)
        out["model_params_m"] = round(
            sum(v.numel() for v in model.values()
                if torch.is_tensor(v)) / 1e6, 3)
    else:
        out["n_model_tensors"] = MISSING
        out["model_params_m"] = MISSING
    return out


def run_forensics(rows, out_path: Path):
    out_rows = []
    for r in rows:
        print(f"  reading {stem_of(r)}")
        out_rows.append(forensics_row(r, load_ckpt(r["path"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FORENSICS_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path}: {len(out_rows)} rows")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract SAGA gate logits / checkpoint forensics (CPU).")
    parser.add_argument("--manifest",
                        default="results/legacy/checkpoint_manifest.csv")
    parser.add_argument("--exp", default="e2",
                        help="manifest experiment filter (default e2)")
    parser.add_argument("--variant", default=None,
                        help="gate mode: manifest variant filter (use saga)")
    parser.add_argument("--out-dir", default="results/legacy/gates",
                        help="gate mode: output directory")
    parser.add_argument("--forensics", action="store_true",
                        help="dump training-state metadata for every "
                             "matching checkpoint instead")
    parser.add_argument("--out", default="results/legacy/ckpt_forensics.csv",
                        help="forensics mode: output CSV")
    args = parser.parse_args()

    if args.forensics:
        rows = manifest_rows(args.manifest, exp=args.exp,
                             variant=args.variant)
        if not rows:
            sys.exit("no matching manifest rows")
        run_forensics(rows, Path(args.out))
    else:
        variant = args.variant or "saga"
        rows = manifest_rows(args.manifest, exp=args.exp, variant=variant)
        if not rows:
            sys.exit(f"no manifest rows for variant {variant!r}")
        run_gate_extraction(rows, Path(args.out_dir))


if __name__ == "__main__":
    main()
