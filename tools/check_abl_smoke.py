#!/usr/bin/env python3
"""
tools/check_abl_smoke.py
========================
TASK-12 A4 — read the six ablation smoke run dirs and print one PASS/FAIL
line per acceptance item. Missing evidence is a FAIL, never an assumption
(TASK-09's rule: a checker that cannot see a thing must not claim it).

    python tools/check_abl_smoke.py [--runs-root results/runs_smoke] \
        [--matrix configs/abl_matrix.yaml]

Exit 0 = every check passed, 1 = at least one FAIL. CPU only, stdlib +
numpy/yaml — it runs on a login node as happily as inside the smoke job.

Checked per arm (all of it from FILES the run wrote, never from this
process re-deriving what the run should have done):
  - the contract: meta.json (with end_time), config.resolved.yaml, log.csv
    with the exact e2r schema, ckpt/last.pth
  - the resolved config differs from the other arms ONLY in run_id /
    variant / model.gate / model.gate_mode / knobs.gate_init_logit
  - gates/phi_e###.npz present for const/headscalar/spatial, ABSENT for
    layerscale and baseline, with the per-arm shape
  - arm B (const): phi is exactly 0 at every epoch and BIT-IDENTICAL across
    epochs — the frozen-ness claim, measured on the production path
  - arm C (headscalar): one value per (layer, head)
  - arm F: phi starts at +4.0 (its epoch-0 dump), i.e. the init knob really
    reached the model
  - grads/grad_phi.csv for the two arms that request it, and only those
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOG_FIELDS = ["epoch", "lr", "train_loss", "val_top1_full", "val_top5_full",
              "val_loss", "img_per_sec", "wall_time"]

# what each gate_mode must produce; phi_slots is the trailing phi dimension
# (None = no phi dump at all). Trainable/registered counts are ViT-S/16:
# 12 layers x 6 heads x {196 | 1} and 12 x 384 for LayerScale.
ARM_SPEC = {
    "none":       {"phi_slots": None, "registered": 0,     "trainable": 0},
    "const":      {"phi_slots": 196,  "registered": 14112, "trainable": 0},
    "headscalar": {"phi_slots": 1,    "registered": 72,    "trainable": 72},
    "layerscale": {"phi_slots": None, "registered": 4608,  "trainable": 4608},
    "spatial":    {"phi_slots": 196,  "registered": 14112, "trainable": 14112},
}

# keys that are ALLOWED to differ between arms; anything else differing is
# a FAIL, because then the arms are not a controlled comparison
ALLOWED_DIFFS = {
    ("run_id",), ("variant",),
    ("model", "gate"), ("model", "gate_mode"),
    ("knobs", "gate_init_logit"),
    ("instrumentation", "log_grad_phi"),
}


class Report:
    def __init__(self):
        self.n_pass = self.n_fail = 0

    def check(self, ok: bool, label: str, detail: str = ""):
        self.n_pass += bool(ok)
        self.n_fail += (not ok)
        tail = f"  ({detail})" if detail else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{tail}")
        return ok


def flatten(d, prefix=()):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def check_arm(run_dir: Path, run_id: str, spec: dict, rep: Report):
    print(f"\n{run_id}  [gate_mode={spec['gate_mode']}]")
    if not run_dir.is_dir():
        rep.check(False, "run dir exists", str(run_dir))
        return None

    ok_meta = (run_dir / "meta.json").exists()
    rep.check(ok_meta, "meta.json present")
    meta = json.loads((run_dir / "meta.json").read_text()) if ok_meta else {}
    rep.check(bool(meta.get("end_time")), "run finished (meta.end_time set)")

    cfg_path = run_dir / "config.resolved.yaml"
    rep.check(cfg_path.exists(), "config.resolved.yaml present")
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    rep.check(cfg.get("model", {}).get("gate_mode") == spec["gate_mode"],
              "resolved gate_mode matches the matrix",
              str(cfg.get("model", {}).get("gate_mode")))

    log_path = run_dir / "log.csv"
    rows = []
    if log_path.exists():
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            rep.check(reader.fieldnames == LOG_FIELDS, "log.csv schema exact",
                      str(reader.fieldnames))
            rows = list(reader)
    else:
        rep.check(False, "log.csv present")
    rep.check(len(rows) == 2 and [int(r["epoch"]) for r in rows] == [0, 1],
              "2 epochs logged, contiguous from 0",
              f"{len(rows)} rows")
    rep.check(all(r["val_top1_full"] not in ("", None) for r in rows),
              "full-val top-1 recorded every epoch")

    rep.check((run_dir / "ckpt" / "last.pth").exists(),
              "ckpt/last.pth present")

    # ── gate artifacts ────────────────────────────────────────────────────
    gates = sorted((run_dir / "gates").glob("phi_e*.npz"))
    want_slots = ARM_SPEC[spec["gate_mode"]]["phi_slots"]
    if want_slots is None:
        rep.check(not gates, "no phi dump (arm has no phi)",
                  f"{len(gates)} files")
    else:
        rep.check(len(gates) == 2, "phi dumped every epoch",
                  f"{len(gates)} files")
        if gates:
            phis = [np.load(p)["phi"] for p in gates]
            shape_ok = all(p.shape == (12, 6, want_slots) for p in phis)
            rep.check(shape_ok, f"phi shape [12, 6, {want_slots}]",
                      str(phis[0].shape))
            if spec["gate_mode"] == "const":
                rep.check(bool(np.all(phis[0] == 0.0)),
                          "arm B: phi is exactly 0 (gate = 0.5)")
                rep.check(len(phis) > 1
                          and bool(np.array_equal(phis[0], phis[-1])),
                          "arm B: phi BIT-IDENTICAL across a trained epoch")
            if spec["gate_mode"] == "headscalar":
                rep.check(phis[0].shape[-1] == 1,
                          "arm C: one value per (layer, head), no spatial axis")
            init = float(spec.get("gate_init_logit", 0.0))
            if init:
                rep.check(bool(np.allclose(phis[0], init, atol=1e-6)),
                          f"epoch-0 phi == gate_init_logit ({init})",
                          f"mean {float(phis[0].mean()):.4f}")
            if spec["gate_mode"] != "const":
                rep.check(not np.array_equal(phis[0], phis[-1]),
                          "learnable phi actually moved in 2 epochs")

    grads = run_dir / "grads" / "grad_phi.csv"
    if spec.get("log_grad_phi"):
        n = sum(1 for _ in open(grads)) - 1 if grads.exists() else 0
        rep.check(grads.exists() and n > 0, "grad_phi.csv written",
                  f"{n} rows")
    else:
        rep.check(not grads.exists(), "no grad_phi.csv (not requested)")

    return cfg


def main():
    ap = argparse.ArgumentParser(
        description="Verify the TASK-12 six-arm smoke.")
    ap.add_argument("--runs-root", default="results/runs_smoke")
    ap.add_argument("--matrix", default="configs/abl_matrix.yaml")
    args = ap.parse_args()

    matrix = yaml.safe_load(open(args.matrix))
    rep = Report()
    cfgs = {}

    print(f"TASK-12 ablation smoke check — {args.runs_root}")
    # ARM_SPEC's counts and the phi shapes below are ViT-S/16 geometry
    # (12 layers x 6 heads x 196 positions); refuse to "check" anything else
    # rather than print shapes that do not mean what they say.
    arches = {r["arch"] for r in matrix["runs"].values()}
    if arches != {"vit_small_patch16_224"}:
        sys.exit(f"this checker pins ViT-S/16 geometry; matrix has {arches}")
    for run_id, run in matrix["runs"].items():
        spec = {"gate_mode": run.get(
                    "gate_mode",
                    "spatial" if run["variant"] == "saga" else "none"),
                "gate_init_logit": run.get("gate_init_logit", 0.0),
                "log_grad_phi": run.get("log_grad_phi", False)}
        cfg = check_arm(Path(args.runs_root) / run_id, run_id, spec, rep)
        if cfg:
            cfgs[run_id] = cfg

    # ── the arms are a controlled comparison ──────────────────────────────
    print("\narms differ only where they are allowed to")
    if len(cfgs) < 2:
        rep.check(False, "at least two resolved configs to compare",
                  f"{len(cfgs)} found")
    else:
        ref_id, ref = sorted(cfgs.items())[0]
        flat_ref = flatten(ref)
        for run_id, cfg in sorted(cfgs.items())[1:]:
            flat = flatten(cfg)
            keys = set(flat_ref) | set(flat)
            bad = sorted(k for k in keys
                         if k not in ALLOWED_DIFFS
                         and flat_ref.get(k) != flat.get(k))
            rep.check(not bad, f"{run_id} vs {ref_id}",
                      "unexpected: " + ", ".join(".".join(k) for k in bad)
                      if bad else "")

    print(f"\n{rep.n_pass} passed, {rep.n_fail} failed")
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
