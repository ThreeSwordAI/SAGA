#!/usr/bin/env python3
"""
analysis/build_C_figures.py
===========================
TASK C / C11 — the data behind Track C's figure panels.

    python analysis/build_C_figures.py
        -> figures_data/frozen/F5B_readout.npz
        -> figures_data/frozen/F1C_readout_draft.npz
        -> figures_data/frozen/F7_attention.npz

(`F_ttr_curve.npz` is written by `analysis/frozen_I7_ttr_curve.py --npz`,
which owns the TTR grid and its MISSING bookkeeping.)

EVERYTHING IS READ FROM THE COMMITTED TABLES, never recomputed from a
records file — the same rule `analysis/build_F5A_terminal.py` follows, and
for the same reason: a figure and a table that were computed twice can
disagree, and then neither is trustworthy. Every number in these files
appears in a `T_I5*` or `T_I7*` CSV.

THE PANELS

F5B_readout — what the correspondence readout measured, against chance.
  acc|<run_id>|<stage>|<transform>|<descriptor>        mean acc_exact
  ci_lo|… / ci_hi|…                                    its bootstrap interval
  over_chance|…                                        acc / chance
  chance|<transform>                                   the chance line
  delta|<contrast>|<pair_label>|<stage>|<transform>|<descriptor>
                                                       paired method delta
  delta_ci_lo|… / delta_ci_hi|…                        and its interval

F1C_readout_draft — the compact panel: the PRIMARY cut only (D1 stage
  `s11_out`, primary descriptor `l2`, `native`), one value per checkpoint per
  transform, with chance. A draft, and named one.

F7_attention — incoming attention at the exceedance positions.
  ratio|<run_id>|<block>|<query_group>|<basis>         mass_exc / mass_other
  ratio_ci_lo|… / ratio_ci_hi|…
  mass_exc|… / mass_other|…                            the two levels
  auc|…                                                AUC for the flag
  share_ratio|…                                        mass share / position share
  value_norm_ratio|<run_id>|<block>|<basis>            EXPLORATORY
  threshold                                            D10's conventional 3x

MISSING stays MISSING: a cell that has none becomes NaN in the numeric
array and `True` in the parallel `<key>__is_missing` array, so a reader can
tell an absent value from a measured one and never averages the first.

No training, no optimizer, no probe fitting.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MISSING = "MISSING"
OUT_DIR = REPO / "figures_data" / "frozen"

I5_TABLES = REPO / "results" / "frozen" / "I5_readout" / "sub2k" / "tables"
I7_TABLES = REPO / "results" / "frozen" / "I7_attention" / "sub1k" / "tables"

#: D1, and D9's primary descriptor. The compact panel shows this cut only.
PRIMARY_STAGE = "s11_out"
PRIMARY_DESCRIPTOR = "l2"
NATIVE = "native"


def read(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist — build the tables first "
            f"(analysis/frozen_I5_analysis.py, analysis/frozen_I7_analysis.py)")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    """A float, or MISSING. Never a silent NaN."""
    if v is None or v == "" or v == MISSING:
        return MISSING
    try:
        f = float(v)
    except (TypeError, ValueError):
        return MISSING
    return f if np.isfinite(f) else MISSING


class Pack:
    """A flat {key: scalar} store that writes MISSING beside every number."""

    def __init__(self):
        self.values = {}

    def put(self, key, value):
        self.values[key] = value

    def arrays(self, extra=None):
        out = {}
        for k, v in self.values.items():
            out[k] = np.asarray(
                [np.nan if v is MISSING else float(v)], dtype=np.float64)
            out[f"{k}__is_missing"] = np.asarray([v is MISSING], dtype=bool)
        out.update(extra or {})
        return out


def write_npz(path, arrays, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(arrays)
    arrays["meta_json"] = np.asarray(
        json.dumps(meta, sort_keys=True, default=str))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# F5B — the readout panel
# ─────────────────────────────────────────────────────────────────────────────

def build_f5b(out_dir):
    a = read(I5_TABLES / "T_I5a_readout.csv")
    b = read(I5_TABLES / "T_I5b_methods.csv")
    pack = Pack()
    chance = {}
    for r in a:
        if r["condition_id"] != NATIVE:
            continue
        k = f"{r['run_id']}|{r['stage']}|{r['transform']}|{r['descriptor']}"
        pack.put(f"acc|{k}", num(r["acc_exact_mean"]))
        pack.put(f"ci_lo|{k}", num(r["acc_exact_ci_lo"]))
        pack.put(f"ci_hi|{k}", num(r["acc_exact_ci_hi"]))
        pack.put(f"over_chance|{k}", num(r["acc_exact_over_chance"]))
        chance[r["transform"]] = num(r["chance_exact"])
    for t, c in chance.items():
        pack.put(f"chance|{t}", c)
    for r in b:
        k = (f"{r['contrast']}|{r['pair_label']}|{r['stage']}|"
             f"{r['transform']}|{r['descriptor']}")
        pack.put(f"delta|{k}", num(r["delta_acc_exact"]))
        pack.put(f"delta_ci_lo|{k}", num(r["ci_lo"]))
        pack.put(f"delta_ci_hi|{k}", num(r["ci_hi"]))

    meta = {
        "panel": "F5B — correspondence readout, per checkpoint and per "
                 "method contrast, against chance",
        "source_tables": ["T_I5a_readout.csv", "T_I5b_methods.csv"],
        "condition": NATIVE,
        "endpoint_class": "secondary",
        "note": "Every I5 comparison is a SECONDARY endpoint under D7. "
                "acc_exact is the primary readout; chance is 1/196.",
    }
    return write_npz(out_dir / "F5B_readout.npz", pack.arrays(), meta)


# ─────────────────────────────────────────────────────────────────────────────
# F1C — the compact draft panel
# ─────────────────────────────────────────────────────────────────────────────

def build_f1c(out_dir):
    a = read(I5_TABLES / "T_I5a_readout.csv")
    pack = Pack()
    variants, transforms = {}, set()
    for r in a:
        if (r["condition_id"] != NATIVE or r["stage"] != PRIMARY_STAGE
                or r["descriptor"] != PRIMARY_DESCRIPTOR):
            continue
        k = f"{r['run_id']}|{r['transform']}"
        pack.put(f"acc|{k}", num(r["acc_exact_mean"]))
        pack.put(f"ci_lo|{k}", num(r["acc_exact_ci_lo"]))
        pack.put(f"ci_hi|{k}", num(r["acc_exact_ci_hi"]))
        pack.put(f"chance|{r['transform']}", num(r["chance_exact"]))
        variants[r["run_id"]] = r["variant"]
        transforms.add(r["transform"])

    extra = {
        "run_ids": np.asarray(sorted(variants)),
        "variants": np.asarray([variants[k] for k in sorted(variants)]),
        "transforms": np.asarray(sorted(transforms)),
    }
    meta = {
        "panel": "F1C (DRAFT) — the primary cut only: D1 stage s11_out, "
                 "primary descriptor l2, native condition",
        "source_tables": ["T_I5a_readout.csv"],
        "stage": PRIMARY_STAGE, "descriptor": PRIMARY_DESCRIPTOR,
        "endpoint_class": "secondary",
        "note": "DRAFT. One value per checkpoint per transform; the full "
                "stage/descriptor grid is in F5B_readout.npz.",
    }
    return write_npz(out_dir / "F1C_readout_draft.npz",
                     pack.arrays(extra), meta)


# ─────────────────────────────────────────────────────────────────────────────
# F7 — incoming attention
# ─────────────────────────────────────────────────────────────────────────────

def build_f7(out_dir):
    a = read(I7_TABLES / "T_I7a_incoming.csv")
    pack = Pack()
    for r in a:
        k = f"{r['run_id']}|{r['block']}|{r['query_group']}|{r['basis']}"
        pack.put(f"ratio|{k}", num(r["ratio"]))
        pack.put(f"ratio_ci_lo|{k}", num(r["ratio_ci_lo"]))
        pack.put(f"ratio_ci_hi|{k}", num(r["ratio_ci_hi"]))
        pack.put(f"mass_exc|{k}", num(r["mass_per_exceedance_token"]))
        pack.put(f"mass_other|{k}", num(r["mass_per_other_token"]))
        pack.put(f"auc|{k}", num(r["auc"]))
        pack.put(f"share_ratio|{k}", num(r["share_over_position_share"]))

    vn_path = I7_TABLES / "T_I7c_value_norm.csv"
    if vn_path.exists():
        for r in read(vn_path):
            k = f"{r['run_id']}|{r['block']}|{r['basis']}"
            pack.put(f"value_norm_ratio|{k}", num(r["ratio"]))

    verdict_path = (REPO / "results" / "frozen" / "I7_attention" / "sub1k"
                    / "i7_wording_verdict.json")
    verdict = (json.loads(verdict_path.read_text(encoding="utf-8"))
               if verdict_path.exists() else {})
    pack.put("threshold", float(verdict.get("threshold", 3.0)))

    meta = {
        "panel": "F7 — incoming attention mass per exceedance token vs per "
                 "other token, with D10's conventional 3x line",
        "source_tables": ["T_I7a_incoming.csv", "T_I7c_value_norm.csv"],
        "endpoint_class": "secondary",
        "value_norm_note": "value_norm_ratio is EXPLORATORY (§5)",
        "wording_verdict": verdict.get("verdict", MISSING),
        "wording_sentence": verdict.get("sentence", MISSING),
        "note": "The verdict and sentence are produced ONCE by "
                "analysis/i7_wording.py and copied, never re-decided here.",
    }
    return write_npz(out_dir / "F7_attention.npz", pack.arrays(), meta)


def main():
    p = argparse.ArgumentParser(description="TASK C / C11 — figure data.")
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()
    out = Path(args.out)
    written = [build_f5b(out), build_f1c(out), build_f7(out)]
    for w in written:
        print(f"wrote {w}  ({w.stat().st_size / 1e3:.1f} kB)")
    print("\nF_ttr_curve.npz is written by analysis/frozen_I7_ttr_curve.py "
          "--npz (it owns the TTR grid and its MISSING bookkeeping).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
