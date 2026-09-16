#!/usr/bin/env python3
"""
tools/frozen_I1_norms.py
========================
TASK A / I1 A3 — the extraction I2 did not do.

    python tools/frozen_I1_norms.py \
        --run-id e2r_vits_mixup_baseline_s1 \
        --conditions configs/frozen/I1_spatial.yaml \
        --split results/frozen/splits/evaluation.json \
        --data $PROBE_IN_DIR --device cuda --seed 0 --skip-if-done

Per run and per stage it writes two files into
`results/frozen/I1_spatial/<split_name>/<run_id>/`:

    norms_<stage>.npz   float32 [n_images, N_patches] per-token L2 norms,
                        the per-image MAD thresholds, the image ids.
                        GIT-IGNORED — ~8 MB per stage per run. It stays on
                        the cluster; every number the paper prints is
                        derived from it into the file below.

    maps_<stage>.npz    the frequency map per basis, the per-image
                        exceedance indicators as np.packbits, the counts,
                        the thresholds used, the image ids and the full
                        provenance. COMMITTED — Phase C reads only this.

WHAT THIS TOOL DOES NOT DO. It does not recompute the maps TASK I2 already
produced at `s11_out` and `hist`: those are loaded from
`results/frozen/I2_terminal/<split>/<run_id>/maps.npz` by
`saga.frozen.prevalence.from_i2_maps_npz`. It captures those two stages
anyway because the per-token norms and the per-image indicators are NOT
recoverable from I2's summed counts, and because capturing them costs
nothing once the hooks for `in_b07` / `in_b08` are registered — one forward
pass per batch feeds all four stages. The counts it writes at `s11_out` and
`hist` must therefore REPRODUCE I2's, and a Phase-C check compares them.

Append-safe and resubmission-safe (I0 handoff §8.5): `--skip-if-done` exits
0 when the completion marker already names this exact checkpoint, every npz
is written to a temp file and renamed, and the marker is written last.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import diag as fdiag  # noqa: E402
from saga.frozen import norms as fnorms  # noqa: E402
from saga.frozen import records as rec  # noqa: E402
from saga.frozen.edits import state_hash, terminal_gate_override  # noqa: E402
from saga.frozen.runner import (build_from_row, condition_stages,  # noqa: E402
                                conditions_for, image_id_for,
                                load_conditions, load_manifest_row,
                                load_split, verify_checkpoint)
from saga.frozen.stages import resolve_stage  # noqa: E402
from saga.metrics import infer_num_prefix_tokens  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"

#: The completion marker's kind. `norms` and not `records`: this tool writes
#: no records/diag parquet, and a marker that shared I2's name would let
#: `--skip-if-done` confuse the two work packages in one directory.
DONE_KIND = "maps"


def edit_factory(condition, model):
    """A zero-argument context manager for one condition, or None.

    Only the two edit types I1 declares are reachable: `native` (no edit)
    and `terminal_gate_override` (the `term_1.00` bypass control). Anything
    else is a configuration error and is refused here rather than silently
    measured as native.
    """
    kind = condition["edit_type"]
    if kind == "native":
        return None
    if kind == "terminal_gate_override":
        value = float((condition.get("params") or {})["value"])
        return lambda: terminal_gate_override(model, value)
    raise SystemExit(
        f"condition {condition['id']!r} declares edit_type {kind!r}, which "
        f"I1 does not use. configs/frozen/I1_spatial.yaml declares exactly "
        f"two conditions and this tool implements exactly those two.")


def main():
    p = argparse.ArgumentParser(
        description="TASK A / I1: per-token norms, per-image indicators and "
                    "frequency maps at the I1 stages.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ckpt-kind", default="last")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--conditions", required=True,
                   help="configs/frozen/I1_spatial.yaml — the stages, the "
                        "conditions and the threshold paths come from it")
    p.add_argument("--split", required=True,
                   help="discovery (D5 selection) or evaluation (reporting)")
    p.add_argument("--data", required=True, metavar="ROOT")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out-root", default="results/frozen")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-images", type=int, default=None,
                   help="smoke only: the first N images of the split")
    p.add_argument("--keep-norms", action="store_true", default=True,
                   help="write norms_<stage>.npz (default; git-ignored)")
    p.add_argument("--no-keep-norms", dest="keep_norms", action="store_false",
                   help="skip the norm dumps — the scale null (T_I1b) then "
                        "cannot be computed for this run")
    p.add_argument("--skip-if-done", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (torch.cuda.is_available()
                        or not args.device.startswith("cuda")) else "cpu")

    row = load_manifest_row(args.manifest, args.run_id, args.ckpt_kind)
    sha = verify_checkpoint(row, args.ckpt)
    conditions = load_conditions(args.conditions)
    split_doc, items, split_sha = load_split(args.split)
    split_name = split_doc.get("name", Path(args.split).stem)
    if args.max_images is not None:
        items = items[:args.max_images]
    image_ids = [image_id_for(rel) for rel, _ in items]

    out_dir = (Path(args.out_root) / conditions["work_package"] / split_name
               / args.run_id)
    print(f"frozen_I1_norms: {args.run_id}/{args.ckpt_kind} "
          f"{row['arch']}/{row['variant']}")
    print(f"                 ckpt_sha256={sha[:16]}…  split={split_name} "
          f"sha={split_sha[:16]}… n={len(items)}")
    print(f"                 -> {out_dir}")

    if args.skip_if_done and rec.is_done(out_dir, DONE_KIND, sha):
        print("                 already complete for this checkpoint — "
              "nothing rewritten")
        return 0

    cell = fdiag.cell_key(row["arch"], row["recipe_actual"])
    tau_path = fdiag.thresholds_cal_path(conditions["thresholds_cal"],
                                         split_name)
    tau_doc = fdiag.load_thresholds_cal(tau_path, split_sha256=split_sha)
    declared = conditions_for(conditions, row["variant"])
    needed = sorted({s for c in declared for s in condition_stages(conditions, c)})
    tau_cal = {s: fdiag.tau_cal_for(tau_doc, cell, s) for s in needed}
    uncalibrated = [s for s, v in tau_cal.items() if not isinstance(v, float)]
    if uncalibrated:
        raise SystemExit(
            f"cell {cell!r} has no calibrated tau at stage(s) {uncalibrated} "
            f"on split {split_name}. Run tools/frozen_I2_thresholds.py "
            f"--conditions {args.conditions} --split {args.split} first — a "
            f"stage with no threshold has no fixed_cal map.")
    print(f"                 tau_cal[{cell}] = " + "  ".join(
        f"{s}={tau_cal[s]:.6f}" for s in needed))

    from tools.eval import build_val_transform
    from tools.frozen_eval import FrozenSplitDataset, output_digests

    model = build_from_row(row, ckpt_path=args.ckpt, device=device)
    n_prefix = infer_num_prefix_tokens(model)
    hash_before = state_hash(model)
    dataset = FrozenSplitDataset(args.data, items,
                                 transform=build_val_transform(224))

    by_stage, summary = {}, {}
    for cond in declared:
        stages = condition_stages(conditions, cond)
        print(f"                 condition {cond['id']}: "
              f"{', '.join(stages)}")
        acc = fnorms.extract_stage_norms(
            model, dataset, stages, device=device,
            batch_size=args.batch_size, image_ids=image_ids, tau_cal=tau_cal,
            k=float(tau_doc.get("k", fdiag.MAD_K)),
            max_images=args.max_images, condition_id=cond["id"],
            edit=edit_factory(cond, model))
        for stage, a in acc.items():
            by_stage.setdefault(stage, []).append(a)
        summary[cond["id"]] = {"stages": list(stages),
                               "n_images": acc[stages[0]].n_images}

    hash_after = state_hash(model)
    if hash_before != hash_after:
        raise SystemExit(
            f"{args.run_id}: model state changed across the condition sweep "
            f"({hash_before[:12]} -> {hash_after[:12]})")

    meta = {
        "work_package": conditions["work_package"], "run_id": row["run_id"],
        "arch": row["arch"], "recipe_actual": row["recipe_actual"],
        "variant": row["variant"], "cell": cell,
        "ckpt_kind": row["ckpt_kind"], "ckpt_sha256": sha,
        "split_name": split_name, "split_sha256": split_sha,
        "n_prefix": int(n_prefix),
        "precision": conditions.get("precision", "fp32"),
        "tau_cal": tau_cal, "mad_k": float(tau_doc.get("k", fdiag.MAD_K)),
        "thresholds_cal_file": str(tau_path),
        "git_sha": git_sha(), "seed": args.seed,
        "generated_by": "tools/frozen_I1_norms.py",
    }

    written = []
    for stage in sorted(by_stage):
        accs = by_stage[stage]
        written.append(fnorms.write_stage_maps_npz(out_dir, accs, meta))
        if args.keep_norms:
            for a in accs:
                written.append(fnorms.write_norms_npz(out_dir, a, meta))
    for path in written:
        print(f"wrote {path}")

    marker = dict(meta, n_conditions=len(declared),
                  stages=sorted(by_stage),
                  conditions=summary,
                  state_hash_before=hash_before, state_hash_after=hash_after,
                  state_restored=True,
                  files=[p.name for p in written])
    (out_dir / "run_meta.json").write_text(
        json.dumps(dict(marker, outputs=output_digests(out_dir)),
                   indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n")
    # LAST, always: its presence is what `--skip-if-done` trusts.
    rec.mark_done(out_dir, DONE_KIND, marker)
    print(f"\nstate restored: True\n{len(written)} file(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
