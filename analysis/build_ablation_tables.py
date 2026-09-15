#!/usr/bin/env python3
"""
analysis/build_ablation_tables.py
=================================
TASK-12 Phase C.1: the matched-init ablation table ->
results/tables/T2_ablation.csv.

One row per arm of configs/abl_matrix.yaml, in the matrix's own order.
The question the table answers: does SAGA's gain come from the LEARNED
SPATIAL STRUCTURE, or would any per-layer/per-head rescaling at the same
insertion point do as well? So the columns that matter are read side by
side: top-1 against arm A, the trainable gate parameter count, and the
final within-layer spatial std of the gate — which is ZERO BY CONSTRUCTION
for the frozen and head-scalar arms and is reported as the structural
sanity check that each arm is what it claims to be.

Provenance, per column:
- `top1_last`, `top5_last`, `ckpt_sha256` come from
  results/runs/<id>/eval/imagenet_val_last.json — the exact fp32 full-50k
  evaluation (tools/eval.py), which asserts n == 50000 before writing.
  These are MISSING until the Phase-C derivation has run
  (scripts/jobs/abl_derive.sbatch).
- `top1_last_bf16_log` comes from the last row of the run's own log.csv.
  It is the trainer's per-epoch full-val under bf16 autocast, kept in a
  SEPARATE column because it is not the same quantity: TASK-06B measured a
  bf16-vs-fp32 gap of the same order as the deltas here (a +0.52 log value
  against +0.456 fp32). It is never mixed into the fp32 columns and never
  used for a delta that is reported as canonical.
- `sink_fixed_canon` (the PRIMARY sink metric) and `sink_mad_k5`,
  `oversmooth_pairwise`, `oversmooth_pairwise_nosink`, `eff_rank` come from
  results/runs/<id>/diag/diag_final_last.json. `sink_fixed_canon` exists
  only after tools/apply_fixed_thr.py --version canon has backfilled it; the
  tau is the committed ViT-S/mixup value and is never recalibrated here.
- `gate_params_*` are counted from a model BUILT at the run's own recorded
  gate_mode, not typed in.
- `gate_spatial_std_final`, `gate_mean_final` come from the run's last
  gates/phi_e###.npz.

MISSING is written wherever the evidence is absent, never a substitute
value, and a delta against a MISSING side is itself MISSING.

    python analysis/build_ablation_tables.py \
        [--runs-root results/runs] [--matrix configs/abl_matrix.yaml] \
        [--out results/tables/T2_ablation.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MISSING = "MISSING"
N_PATCHES = 196          # ViT-S/16 at 224: the saturation denominator

# arm letters are TASK-12's own design table, keyed by gate_mode + init so a
# renamed run cannot silently change what a row means
ARM_LETTER = {
    ("none", 0.0): "A",
    ("const", 0.0): "B",
    ("headscalar", 0.0): "C",
    ("layerscale", 0.0): "D",
    ("spatial", 0.0): "E",
    ("spatial", 4.0): "F",
}

FIELDS = [
    "arm", "run_id", "gate_mode", "gate_init_logit",
    "gate_params_registered", "gate_params_trainable",
    "top1_last", "delta_top1_vs_A", "top5_last",
    "top1_last_bf16_log", "delta_top1_bf16_vs_A",
    "sink_fixed_canon", "canon_thr_value", "delta_sink_canon_vs_A",
    "sink_canon_saturated",
    "sink_mad_k5", "oversmooth_pairwise", "oversmooth_pairwise_nosink",
    "eff_rank",
    # the trainer's OWN periodic diagnostics at its last diag epoch. Same
    # metric code and same frozen split as the canonical columns above, but
    # a different file and a norms-only pass (no attention), so they are
    # kept separate and never merged into the canonical ones.
    "intrain_diag_epoch", "sink_mad_k5_intrain", "oversmooth_pairwise_intrain",
    "oversmooth_pairwise_nosink_intrain", "eff_rank_intrain",
    "gate_mean_final", "gate_spatial_std_final", "spatial_std_expected",
    "n_epochs", "seed", "ckpt_sha256", "note",
]


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def gate_param_counts(arch: str, gate_mode: str):
    """(registered, trainable) for this arm, from a model built at the run's
    own gate_mode. Built rather than tabulated so the number can never
    disagree with the code that trained it."""
    from tools.model_factory import build_model
    from tools.model_factory import ARCH_MAP
    short = next((k for k, v in ARCH_MAP.items() if v == arch), arch)
    model = build_model(short, "saga" if gate_mode != "none" else "baseline",
                        gate_mode=gate_mode)
    total = trainable = 0
    for name, p in model.named_parameters():
        if ".attn.gate." in name:
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
    return total, trainable


def gate_stats(run_dir: Path, gate_mode: str):
    """(mean gate value, mean within-layer spatial std) at the last dumped
    epoch. The spatial std is the structural check: 0 for const (frozen
    uniform) and for headscalar (one value per head, broadcast), > 0 only
    where the gate can vary across positions."""
    dumps = sorted((run_dir / "gates").glob("phi_e*.npz"))
    if not dumps:
        return MISSING, MISSING
    phi = np.load(dumps[-1])["phi"].astype(np.float64)   # [L, H, N or 1]
    gate = 1.0 / (1.0 + np.exp(-phi))
    if gate.shape[-1] == 1:
        # headscalar: the applied map is this value at every position, so
        # its spatial std is exactly 0 — stated, not inferred from a
        # 1-element std (which numpy would also give as 0, for the wrong
        # reason)
        return float(gate.mean()), 0.0
    return float(gate.mean()), float(gate.std(axis=2).mean())


def build_rows(runs_root: Path, matrix: dict):
    rows, notes = [], []
    for run_id, run in matrix["runs"].items():
        gate_mode = run.get(
            "gate_mode", "spatial" if run["variant"] == "saga" else "none")
        init = float(run.get("gate_init_logit", 0.0))
        arm = ARM_LETTER.get((gate_mode, init), "?")
        d = runs_root / run_id
        row = {k: MISSING for k in FIELDS}
        row.update(arm=arm, run_id=run_id, gate_mode=gate_mode,
                   gate_init_logit=init)
        why = []

        if not d.is_dir():
            row["note"] = "run dir absent"
            rows.append(row)
            continue

        meta = read_json(d / "meta.json") or {}
        cfg = yaml.safe_load((d / "config.resolved.yaml").read_text())
        row["seed"] = cfg.get("seed", MISSING)
        row["n_epochs"] = cfg["train"]["epochs"]
        if cfg["model"].get("gate_mode") != gate_mode:
            why.append(f"resolved gate_mode {cfg['model'].get('gate_mode')!r} "
                       f"!= matrix {gate_mode!r}")

        reg, tr = gate_param_counts(cfg["model"]["arch"], gate_mode)
        row["gate_params_registered"], row["gate_params_trainable"] = reg, tr

        # bf16 per-epoch value from the run's own log
        log_rows = list(csv.DictReader(open(d / "log.csv", newline="")))
        if log_rows:
            row["top1_last_bf16_log"] = float(log_rows[-1]["val_top1_full"])

        # canonical fp32 eval
        ev = read_json(d / "eval" / "imagenet_val_last.json")
        if ev is None:
            why.append("no eval/imagenet_val_last.json "
                       "(run scripts/jobs/abl_derive.sbatch)")
        else:
            if ev.get("n_images") != 50000:
                why.append(f"eval n_images {ev.get('n_images')} != 50000")
            else:
                row["top1_last"] = ev["top1"]
                row["top5_last"] = ev["top5"]
                row["ckpt_sha256"] = ev.get("ckpt_sha256", MISSING)

        # canonical diagnostics
        dg = read_json(d / "diag" / "diag_final_last.json")
        if dg is None:
            why.append("no diag/diag_final_last.json "
                       "(run scripts/jobs/abl_derive.sbatch)")
        else:
            for key in ("sink_mad_k5", "oversmooth_pairwise",
                        "oversmooth_pairwise_nosink", "eff_rank",
                        "sink_fixed_canon", "canon_thr_value"):
                if dg.get(key) is not None:
                    row[key] = dg[key]
            if dg.get("sink_fixed_canon") is None:
                why.append("sink_fixed_canon not backfilled "
                           "(apply_fixed_thr --version canon)")
            else:
                # TASK-02B's own flag: a fixed threshold that counts >=95% of
                # the 196 patch tokens as sinks is SATURATED and orders
                # nothing. Recorded, never silently compared.
                sat = float(dg["sink_fixed_canon"]) >= 0.95 * N_PATCHES
                row["sink_canon_saturated"] = sat
                if sat:
                    why.append(
                        f"canon tau SATURATED: "
                        f"{dg['sink_fixed_canon']:.4f} of {N_PATCHES} patch "
                        f"tokens counted as sinks (>=95%) — the primary sink "
                        f"metric orders nothing in this cell")
            if (ev is not None and dg.get("ckpt_sha256")
                    and ev.get("ckpt_sha256")
                    and dg["ckpt_sha256"] != ev["ckpt_sha256"]):
                why.append("eval and diag describe DIFFERENT checkpoints")
                row["top1_last"] = row["sink_fixed_canon"] = MISSING

        # the trainer's own last periodic diag — present from the moment the
        # run finished, and superseded (never replaced) by the canonical one
        intrain = sorted((d / "diag").glob("diag_e*.json"))
        if intrain:
            last_it = read_json(intrain[-1]) or {}
            row["intrain_diag_epoch"] = last_it.get("epoch", MISSING)
            for key in ("sink_mad_k5", "oversmooth_pairwise",
                        "oversmooth_pairwise_nosink", "eff_rank"):
                if last_it.get(key) is not None:
                    row[f"{key}_intrain"] = last_it[key]

        mean_g, std_g = gate_stats(d, gate_mode)
        row["gate_mean_final"], row["gate_spatial_std_final"] = mean_g, std_g
        row["spatial_std_expected"] = {
            "none": "n/a (no gate)", "layerscale": "n/a (per-channel)",
            "const": "0 (frozen uniform)",
            "headscalar": "0 (one value per head)",
            "spatial": ">0 (learned per position)"}[gate_mode]

        row["note"] = "; ".join(why) if why else ""
        rows.append(row)
        notes.extend(f"{run_id}: {w}" for w in why)

    # deltas against arm A, computed only where BOTH sides are present
    ref = next((r for r in rows if r["arm"] == "A"), None)
    for r in rows:
        for col, ref_col, out in (
                ("top1_last", "top1_last", "delta_top1_vs_A"),
                ("top1_last_bf16_log", "top1_last_bf16_log",
                 "delta_top1_bf16_vs_A"),
                ("sink_fixed_canon", "sink_fixed_canon",
                 "delta_sink_canon_vs_A")):
            if (ref is not None and r[col] != MISSING
                    and ref[ref_col] != MISSING):
                r[out] = round(float(r[col]) - float(ref[ref_col]), 4)
    return rows, notes


def main():
    ap = argparse.ArgumentParser(description="TASK-12 ablation table T2.")
    ap.add_argument("--runs-root", default="results/runs")
    ap.add_argument("--matrix", default="configs/abl_matrix.yaml")
    ap.add_argument("--out", default="results/tables/T2_ablation.csv")
    args = ap.parse_args()

    matrix = yaml.safe_load(Path(args.matrix).read_text())
    rows, notes = build_rows(Path(args.runs_root), matrix)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}  ({len(rows)} arms)")
    for r in rows:
        print(f"  {r['arm']}  {r['run_id']:<30} "
              f"top1={r['top1_last']}  d_vs_A={r['delta_top1_vs_A']}  "
              f"sink_canon={r['sink_fixed_canon']}  "
              f"gate_params={r['gate_params_trainable']}")
    if notes:
        print("\nincomplete evidence (MISSING is written, never a "
              "substitute):")
        for n in notes:
            print(f"  {n}")


if __name__ == "__main__":
    main()
