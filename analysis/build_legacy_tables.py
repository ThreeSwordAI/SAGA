#!/usr/bin/env python3
"""
analysis/build_legacy_tables.py
===============================
TASK-02 Phase 2 (2.1 + 2.2 + 2.3): completeness check over the e2 manifest,
then the corrected legacy table.

    python analysis/build_legacy_tables.py \
        [--manifest results/legacy/checkpoint_manifest.csv]
        [--eval-dir results/legacy/eval] [--diag-dir results/legacy/diag]
        [--out results/tables/legacy_e2_corrected.csv]

One row per (arch, recipe, variant). All values verbatim from the JSONs
under results/ — nothing is estimated or interpolated; a missing file puts
the literal string MISSING in every cell it would have fed.

Column sources:
- top1_best / top1_last            eval JSON of best.pth / last.pth
- top1_best_minus_last             derived from the two (2.3 memo column)
- delta_top1_last_vs_baseline      top1_last minus the same (arch, recipe)
                                   baseline's top1_last (blank on baseline)
- all diagnostic columns           diag JSON of LAST.pth (the paper reports
                                   last.pth; old best-selection used a buggy
                                   12.5k shard)
- *_lastblock                      last element of the per-block list
"""

import argparse
import csv
import json
import sys
from pathlib import Path

MISSING = "MISSING"

ID_COLS = ["arch", "recipe", "variant"]
COLUMNS = ID_COLS + [
    "top1_best", "top1_last", "delta_top1_last_vs_baseline",
    "sink_mad_k5", "sink_mu3s", "sink_fixed_thr",
    "oversmooth_pairwise", "oversmooth_pairwise_nosink",
    "nosink_excluded_mean", "eff_rank",
    "cls_attn_share_lastblock", "cls_norm_ratio_lastblock",
    "reg_norm_mean", "ckpt_sha256_last",
    "top1_best_minus_last",       # 2.3 best-vs-last memo column
]

VARIANT_ORDER = {"baseline": 0, "registers": 1, "saga": 2}


def e2_runs(manifest_path):
    """(arch, recipe, variant, seed) -> {tag: sha} from the e2 manifest rows."""
    runs = {}
    with open(manifest_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["exp"] != "e2":
                continue
            key = (r["arch"], r["recipe"], r["variant"], r["seed"])
            tag = r["filename"].removesuffix(".pth")
            runs.setdefault(key, {})[tag] = r["sha256"]
    return runs


def load_json(path: Path, expected_sha: str, gaps: list):
    """Load a result JSON; record a gap and return None if absent or if it
    was computed from a different checkpoint than the manifest lists."""
    if not path.exists():
        gaps.append(f"missing file: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("ckpt_sha256") != expected_sha:
        gaps.append(f"ckpt_sha256 mismatch vs manifest: {path}")
        return None
    return data


def build_rows(runs, eval_dir: Path, diag_dir: Path):
    rows, gaps = [], []
    for (arch, recipe, variant, seed), shas in sorted(runs.items()):
        stem = f"e2_{arch}_{recipe}_{variant}_{seed}"
        ev_best = load_json(eval_dir / f"{stem}_best.json",
                            shas.get("best"), gaps)
        ev_last = load_json(eval_dir / f"{stem}_last.json",
                            shas.get("last"), gaps)
        dg_last = load_json(diag_dir / f"{stem}_last.json",
                            shas.get("last"), gaps)
        # diag(best) is required by the completeness check even though the
        # table reads no column from it
        load_json(diag_dir / f"{stem}_best.json", shas.get("best"), gaps)

        row = {"arch": arch, "recipe": recipe, "variant": variant}
        row["top1_best"] = ev_best["top1"] if ev_best else MISSING
        row["top1_last"] = ev_last["top1"] if ev_last else MISSING
        if ev_best and ev_last:
            row["top1_best_minus_last"] = round(
                ev_best["top1"] - ev_last["top1"], 6)
        else:
            row["top1_best_minus_last"] = MISSING

        if dg_last:
            row["sink_mad_k5"] = dg_last["sink_mad_k5"]
            row["sink_mu3s"] = dg_last["sink_mu3s"]
            row["sink_fixed_thr"] = dg_last["sink_fixed_thr"]
            row["oversmooth_pairwise"] = dg_last["oversmooth_pairwise"]
            row["oversmooth_pairwise_nosink"] = \
                dg_last["oversmooth_pairwise_nosink"]
            row["nosink_excluded_mean"] = dg_last["nosink_excluded_mean"]
            row["eff_rank"] = dg_last["eff_rank"]
            attn = dg_last.get("cls_attn_share")
            row["cls_attn_share_lastblock"] = attn[-1] if attn else MISSING
            row["cls_norm_ratio_lastblock"] = dg_last["cls_norm_ratio"][-1]
            # structurally null for models without registers -> blank
            reg = dg_last["reg_norm_mean"]
            row["reg_norm_mean"] = "" if reg is None else reg
        else:
            for col in ["sink_mad_k5", "sink_mu3s", "sink_fixed_thr",
                        "oversmooth_pairwise", "oversmooth_pairwise_nosink",
                        "nosink_excluded_mean", "eff_rank",
                        "cls_attn_share_lastblock", "cls_norm_ratio_lastblock",
                        "reg_norm_mean"]:
                row[col] = MISSING
        row["ckpt_sha256_last"] = shas.get("last", MISSING)
        rows.append(row)

    # delta vs the same (arch, recipe) baseline's top1_last
    base_top1 = {(r["arch"], r["recipe"]): r["top1_last"]
                 for r in rows if r["variant"] == "baseline"}
    for r in rows:
        if r["variant"] == "baseline":
            r["delta_top1_last_vs_baseline"] = ""
            continue
        base = base_top1.get((r["arch"], r["recipe"]))
        if base in (None, MISSING) or r["top1_last"] == MISSING:
            r["delta_top1_last_vs_baseline"] = MISSING
        else:
            r["delta_top1_last_vs_baseline"] = round(r["top1_last"] - base, 6)

    rows.sort(key=lambda r: (r["arch"], r["recipe"],
                             VARIANT_ORDER.get(r["variant"], 9)))
    return rows, gaps


def main():
    parser = argparse.ArgumentParser(
        description="Build the corrected legacy e2 table from result JSONs.")
    parser.add_argument("--manifest",
                        default="results/legacy/checkpoint_manifest.csv")
    parser.add_argument("--eval-dir", default="results/legacy/eval")
    parser.add_argument("--diag-dir", default="results/legacy/diag")
    parser.add_argument("--out",
                        default="results/tables/legacy_e2_corrected.csv")
    args = parser.parse_args()

    runs = e2_runs(args.manifest)
    rows, gaps = build_rows(runs, Path(args.eval_dir), Path(args.diag_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}: {len(rows)} runs "
          f"({len(runs)} manifest run dirs, 4 JSONs each expected)")
    if gaps:
        print(f"\nCOMPLETENESS GAPS ({len(gaps)}) — cells written as MISSING:")
        for g in gaps:
            print("  -", g)
        sys.exit(1)
    print("completeness check: every e2 run has eval(best), eval(last), "
          "diag(best), diag(last) with matching ckpt_sha256")


if __name__ == "__main__":
    main()
