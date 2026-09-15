#!/usr/bin/env python3
"""
analysis/ablation_address.py
============================
TASK-12 Phase C.2: does the learned gate of the spatial arms track the sink
address of THIS cell's own baseline (arm A)?

    python analysis/ablation_address.py \
        [--runs-root results/runs] [--matrix configs/abl_matrix.yaml] \
        [--out results/tables/T2_ablation_address.csv]

The statistical machinery is IMPORTED from TASK-07, never re-derived:
`analysis.address_analysis.rho_spatial` (Spearman plus the EXACT permutation
null over the 8 dihedral x 196 torus-roll transforms) and `head_mean_gate`
(sigmoid first, then the head mean — the ordering TASK-06B had to fix once).
The iid Spearman reference is far too generous for maps this smooth; TASK-07
measured the spatial null's sd at 0.157-0.342 against the iid 0.0716, so a
correlation judged against the wrong null can be off by a factor of 4.

Reported per arm and per basis (canon = the primary metric's tau, mad =
the secondary), per layer, plus the extremal layer with a Bonferroni
correction over the 12 layers it was chosen from — an argmax over 12 needs
it, as TASK-07 Q5 records.

**Arms whose gate cannot vary across positions get no number.** The frozen
arm (B) is uniform by construction and the head-scalar arm (C) holds one
value per (layer, head) broadcast over all 196 positions, so a rank
correlation against any map is UNDEFINED, not zero: `zrank` refuses a
constant vector. Those rows say so in words. Printing 0.0 there would
invite the reading "the non-spatial gate ignores the address", which is a
statement about arithmetic, not about the model.
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

from analysis.address_analysis import head_mean_gate, rho_spatial  # noqa: E402

MISSING = "MISSING"
BASES = ("canon", "mad")
FIELDS = ["arm", "run_id", "gate_mode", "map_basis", "layer", "rho",
          "p_spatial", "p_bonferroni", "null_sd", "n_transforms",
          "gate_mean_layer", "gate_spatial_std_layer", "comparator", "note"]

ARM_LETTER = {("none", 0.0): "A", ("const", 0.0): "B",
              ("headscalar", 0.0): "C", ("layerscale", 0.0): "D",
              ("spatial", 0.0): "E", ("spatial", 4.0): "F"}


def arm_of(run):
    mode = run.get("gate_mode",
                   "spatial" if run["variant"] == "saga" else "none")
    return ARM_LETTER.get((mode, float(run.get("gate_init_logit", 0.0))),
                          "?"), mode


def rel(p: Path) -> str:
    """Repo-relative if possible, else the tail only. A committed results
    file must never carry an absolute path — TASK-09 leaked one into T4's
    note column, and a test now greps every table for that shape."""
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(*p.parts[-4:]).as_posix()


def baseline_address(runs_root: Path, run_id: str):
    """{basis: freq map} for arm A, or (None, reason)."""
    p = runs_root / run_id / "diag" / "diag_final_last_addr.json"
    if not p.exists():
        return None, (f"no {rel(p)} — run scripts/jobs/abl_derive.sbatch")
    addr = json.loads(p.read_text())
    maps = {}
    for basis in BASES:
        key = f"freq_{basis}"
        if key not in addr:
            return None, f"address file has no {key}"
        maps[basis] = np.asarray(addr[key], dtype=np.float64)
    return maps, ""


def last_gate(run_dir: Path):
    dumps = sorted((run_dir / "gates").glob("phi_e*.npz"))
    if not dumps:
        return None, "no gate dump (arm has no phi)"
    gate = head_mean_gate(np.load(dumps[-1])["phi"])      # [L, N or 1]
    return gate, dumps[-1].name


def main():
    ap = argparse.ArgumentParser(
        description="TASK-12 gate-vs-address correlation, TASK-07's null.")
    ap.add_argument("--runs-root", default="results/runs")
    ap.add_argument("--matrix", default="configs/abl_matrix.yaml")
    ap.add_argument("--out", default="results/tables/T2_ablation_address.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    runs = matrix["runs"]

    base_id = next(r for r, run in runs.items() if arm_of(run)[0] == "A")
    addr_maps, why = baseline_address(runs_root, base_id)

    rows = []

    def add(**kw):
        row = {k: MISSING for k in FIELDS}
        row.update(kw)
        rows.append(row)

    for run_id, run in runs.items():
        arm, mode = arm_of(run)
        if arm == "A":
            continue
        run_dir = runs_root / run_id
        gate, gate_note = last_gate(run_dir)
        if gate is None:
            add(arm=arm, run_id=run_id, gate_mode=mode,
                comparator=f"baseline={base_id}",
                note=f"{gate_note}: no gate map exists to correlate")
            continue

        constant = gate.shape[1] == 1
        for basis in BASES:
            if addr_maps is None:
                add(arm=arm, run_id=run_id, gate_mode=mode, map_basis=basis,
                    comparator=f"baseline={base_id}", note=why)
                continue
            target = addr_maps[basis]
            if constant or gate.shape[1] != target.size:
                reason = ("gate is CONSTANT across positions by construction "
                          f"(gate_mode={mode}) — a rank correlation against "
                          "any map is undefined, not zero"
                          if constant else
                          f"gate has {gate.shape[1]} positions, address map "
                          f"has {target.size}")
                for li in range(gate.shape[0]):
                    add(arm=arm, run_id=run_id, gate_mode=mode,
                        map_basis=basis, layer=li,
                        gate_mean_layer=round(float(gate[li].mean()), 6),
                        gate_spatial_std_layer=0.0 if constant else MISSING,
                        comparator=f"baseline={base_id}", note=reason)
                continue

            per_layer = []
            for li in range(gate.shape[0]):
                res = rho_spatial(gate[li], target)
                if res is None:
                    add(arm=arm, run_id=run_id, gate_mode=mode,
                        map_basis=basis, layer=li,
                        gate_mean_layer=round(float(gate[li].mean()), 6),
                        gate_spatial_std_layer=round(
                            float(gate[li].std()), 8),
                        comparator=f"baseline={base_id}",
                        note=f"layer is spatially constant "
                             f"(gate_mode={mode}) — rho is undefined, not "
                             f"zero")
                    per_layer.append(np.nan)
                    continue
                r, p, sd, ntr = res
                per_layer.append(r)
                add(arm=arm, run_id=run_id, gate_mode=mode, map_basis=basis,
                    layer=li, rho=round(r, 6),
                    p_spatial=(round(p, 6) if p is not None else MISSING),
                    null_sd=(round(sd, 6) if sd is not None else MISSING),
                    n_transforms=ntr,
                    gate_mean_layer=round(float(gate[li].mean()), 6),
                    gate_spatial_std_layer=round(float(gate[li].std()), 8),
                    comparator=f"baseline={base_id}",
                    note="exact permutation null (TASK-07), not iid")

            prof = np.asarray(per_layer, dtype=np.float64)
            if np.all(np.isnan(prof)):
                continue
            # the extremal layer is an ARGMAX over 12 — Bonferroni, as
            # TASK-07 Q5 does, and report the opposite-sign extreme too
            n_def = int((~np.isnan(prof)).sum())
            for label, idx in (("layer_absmax", int(np.nanargmax(
                    np.abs(prof)))),
                    ("layer_min", int(np.nanargmin(prof))),
                    ("layer_max", int(np.nanargmax(prof)))):
                res = rho_spatial(gate[idx], target)
                r, p, sd, ntr = res
                add(arm=arm, run_id=run_id, gate_mode=mode, map_basis=basis,
                    layer=f"{label}={idx}", rho=round(r, 6),
                    p_spatial=(round(p, 6) if p is not None else MISSING),
                    p_bonferroni=(round(min(1.0, p * n_def), 6)
                                  if p is not None else MISSING),
                    null_sd=(round(sd, 6) if sd is not None else MISSING),
                    n_transforms=ntr,
                    gate_mean_layer=round(float(gate[idx].mean()), 6),
                    gate_spatial_std_layer=round(float(gate[idx].std()), 8),
                    comparator=f"baseline={base_id}",
                    note=f"Bonferroni over the {n_def} layers it was "
                         f"chosen from")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    if addr_maps is None:
        print(f"  NO ADDRESS MAP: {why}")
        print("  every correlation is MISSING until the derivation runs")
    else:
        for r in rows:
            if str(r["layer"]).startswith("layer_absmax"):
                print(f"  {r['arm']} {r['map_basis']:<6} {r['layer']:<16} "
                      f"rho={r['rho']}  p={r['p_spatial']}  "
                      f"bonf={r['p_bonferroni']}")
        for r in rows:
            if r["rho"] == MISSING and r["layer"] == 0:
                print(f"  {r['arm']} {r['map_basis']:<6} undefined: "
                      f"{r['note']}")


if __name__ == "__main__":
    main()
