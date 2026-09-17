#!/usr/bin/env python3
"""
analysis/frozen_I1_scale_null.py
================================
T_I1b — the scale-only null. TASK A / I1, Phase C.

    python analysis/frozen_I1_scale_null.py --split evaluation

**RUN THIS WHERE THE NORMS ARE.** It reads `norms_<stage>.npz`, which is
git-ignored and stays on the cluster (~8 MB per stage per run, ~700 MB per
split), so it runs on the HPC — on a LOGIN NODE, since it is numpy only and
touches no GPU. It writes one small CSV, which is committed.

That is a gap in the task file's phase plan, not a choice: §7 gives T_I1b's
data as "I1 norms (after B)" and §10 puts T_I1b in Phase C, which is local,
while §4 puts the norms permanently on the cluster. The three cannot all
hold. Everything else in Phase C is computed from the committed maps.

THE ALTERNATIVE EXPLANATION THIS TESTS
--------------------------------------
SAGA's maps carry fewer exceedances than the baseline's. If the only thing
SAGA did was shrink the whole norm field by a constant, thresholding at a
FIXED tau would find fewer positions above it and the map would look sparser
while the underlying spatial arrangement was untouched. So: rescale the
BASELINE's norm field by the scalar `c` that reproduces SAGA's count, and ask
whether the rescaled baseline map now looks like SAGA's map.

  rho(rescaled baseline, SAGA)   vs   rho(baseline, SAGA)
  rho(baseline, other baseline seed)  — the ceiling any pair can reach

If rescaling closes the gap, "less of the same structure" survives. If it
does not, the difference is not a scale effect.

WHY THE MAD BASIS IS REPORTED BESIDE IT. A per-image median + k*MAD threshold
scales with the field, so its count is EXACTLY invariant under any positive
c — `median(cv) + k*MAD(cv) = c*(median(v) + k*MAD(v))`. The invariance is
asserted exactly (not `allclose`) in `tests/test_I1_spatial.py`, and a
difference that survives on the MAD basis is not a scale effect at all.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.address_analysis import rho_spatial  # noqa: E402
from saga.frozen import norms as fnorms  # noqa: E402
from saga.frozen import prevalence as P  # noqa: E402
from saga.frozen.runner import eligible_cohort  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"
I1_ROOT = Path("results/frozen/I1_spatial")
#: The stages the scale null is reported at (task file §7): D1's stage and
#: the historical control.
STAGES = ("s11_out", "hist")

FIELDS = ["work_package", "split_name", "split_sha256", "git_sha",
          "generated_by", "cell", "arch", "recipe_actual", "provenance_tag",
          "stage", "saga_run_id", "baseline_run_id", "tau_cal", "mad_k",
          "n_images", "n_positions",
          "count_baseline", "count_saga", "c_matched", "count_rescaled",
          "mad_count_baseline", "mad_count_baseline_rescaled",
          "mad_count_invariant",
          "rho_baseline_vs_saga", "p_baseline_vs_saga",
          "rho_rescaled_vs_saga", "p_rescaled_vs_saga",
          "rho_baseline_vs_baseline", "p_baseline_vs_baseline",
          "ring1_share_baseline", "ring1_share_rescaled", "ring1_share_saga",
          "ring_profile_rescaled", "reference", "note"]


def norms_path(split, run_id, stage) -> Path:
    return I1_ROOT / split / run_id / f"norms_{stage}.npz"


def load_norms(path):
    with np.load(Path(path), allow_pickle=False) as z:
        return z["norms"], json.loads(str(z["meta_json"]))


def _num(v, nd=6):
    if v is None:
        return MISSING
    if isinstance(v, str):
        return v
    v = float(v)
    return MISSING if not np.isfinite(v) else round(v, nd)


def build(split_name, *, manifest="results/frozen/I0_manifest/manifest.json",
          out_dir=None, git_sha_value=None, root=I1_ROOT) -> dict:
    cohort = eligible_cohort(manifest)
    rows_by_run = {r["run_id"]: r for r in cohort}
    by_cell_tag = {}
    for r in cohort:
        key = (r["arch"], r["recipe_actual"], r["provenance_tag"])
        by_cell_tag.setdefault(key, {})[r["variant"]] = r["run_id"]

    head = {"work_package": "I1_spatial", "split_name": split_name,
            "git_sha": git_sha_value if git_sha_value else git_sha(),
            "generated_by": "analysis/frozen_I1_scale_null.py"}
    rows, missing = [], []

    for (arch, recipe, tag), variants in sorted(by_cell_tag.items()):
        if "saga" not in variants or "baseline" not in variants:
            continue
        saga_id, base_id = variants["saga"], variants["baseline"]
        # a same-cell, different-provenance baseline: the ceiling a pair of
        # checkpoints that differ only by seed can reach
        others = sorted(
            v["baseline"] for (a, rc, t), v in by_cell_tag.items()
            if (a, rc) == (arch, recipe) and t != tag and "baseline" in v)

        for stage in STAGES:
            bp, sp = (norms_path(split_name, base_id, stage),
                      norms_path(split_name, saga_id, stage))
            if not bp.exists() or not sp.exists():
                missing.append(f"{arch}|{recipe}|{tag}|{stage}")
                continue
            bn, bmeta = load_norms(bp)
            sn, _ = load_norms(sp)
            tau = float((bmeta.get("tau_cal") if isinstance(
                bmeta.get("tau_cal"), (int, float)) else
                (bmeta.get("tau_cal") or {}).get(stage)))
            k = float(bmeta.get("mad_k", 5.0))
            n_pos = int(bn.shape[1])

            count_b = fnorms.mean_fixed_count(bn, tau, 1.0)
            count_s = fnorms.mean_fixed_count(sn, tau, 1.0)
            matched = fnorms.match_scale(bn, tau, count_s)
            c = matched["c"]

            mad_b = float(fnorms.mad_counts(bn, k).mean())
            mad_bc = float(fnorms.mad_counts(bn, k, c=c).mean())

            base_map = fnorms.scaled_frequency_map(bn, tau, 1.0)
            resc_map = fnorms.scaled_frequency_map(bn, tau, c)
            saga_map = fnorms.scaled_frequency_map(sn, tau, 1.0)

            r_bs = rho_spatial(base_map, saga_map)
            r_rs = rho_spatial(resc_map, saga_map)
            r_bb = None
            if others:
                op = norms_path(split_name, others[0], stage)
                if op.exists():
                    on, _ = load_norms(op)
                    r_bb = rho_spatial(
                        base_map, fnorms.scaled_frequency_map(on, tau, 1.0))

            rows.append(dict(
                head, split_sha256=bmeta["split_sha256"],
                cell=f"{arch}|{recipe}", arch=arch, recipe_actual=recipe,
                provenance_tag=tag, stage=stage, saga_run_id=saga_id,
                baseline_run_id=base_id, tau_cal=_num(tau), mad_k=k,
                n_images=int(bn.shape[0]), n_positions=n_pos,
                count_baseline=_num(count_b), count_saga=_num(count_s),
                c_matched=_num(c), count_rescaled=_num(matched["achieved"]),
                mad_count_baseline=_num(mad_b),
                mad_count_baseline_rescaled=_num(mad_bc),
                mad_count_invariant=int(np.array_equal(
                    fnorms.mad_counts(bn, k), fnorms.mad_counts(bn, k, c=c))),
                rho_baseline_vs_saga=_num(r_bs[0]) if r_bs else MISSING,
                p_baseline_vs_saga=_num(r_bs[1]) if r_bs else MISSING,
                rho_rescaled_vs_saga=_num(r_rs[0]) if r_rs else MISSING,
                p_rescaled_vs_saga=_num(r_rs[1]) if r_rs else MISSING,
                rho_baseline_vs_baseline=_num(r_bb[0]) if r_bb else MISSING,
                p_baseline_vs_baseline=_num(r_bb[1]) if r_bb else MISSING,
                ring1_share_baseline=_num(P.ring_share(base_map, 1)),
                ring1_share_rescaled=_num(P.ring_share(resc_map, 1)),
                ring1_share_saga=_num(P.ring_share(saga_map, 1)),
                ring_profile_rescaled=" ".join(
                    f"{v:.6f}" for v in P.ring_profile(resc_map)),
                reference="rho_baseline_vs_baseline is the ceiling two "
                          "checkpoints differing only by seed reach",
                note="c is the SMALLEST scale reaching SAGA's mean fixed "
                     "count; the count is a step function of c so exact "
                     "equality is generally unattainable and the achieved "
                     "count is reported beside it. mad_count_invariant=1 "
                     "confirms the MAD count did not move under c, exactly"))
            del bn, sn

    out_dir = Path(out_dir) if out_dir else (root / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "T_I1b_scale_null.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    return {"table": path, "rows": len(rows), "missing": missing}


def main():
    p = argparse.ArgumentParser(
        description="T_I1b, the scale-only null. Run where norms_*.npz are.")
    p.add_argument("--split", default="evaluation")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    r = build(args.split, manifest=args.manifest, out_dir=args.out_dir)
    print(f"wrote {r['table']}  ({r['rows']} row(s))")
    if r["missing"]:
        print(f"no norms for {len(r['missing'])} pair-stage(s): "
              f"{r['missing'][:6]}{' ...' if len(r['missing']) > 6 else ''}")
        print("  norms_*.npz are git-ignored and live on the cluster — run "
              "this there, on a login node (numpy only, no GPU).")
    return 0 if r["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
