#!/usr/bin/env python3
"""
tools/frozen_C_verify.py
========================
TASK C / Phase C §12 — verify what Phase B landed, BEFORE any table is built.

    python tools/frozen_C_verify.py

Checks, for every eligible run and against the manifest and the frozen
splits:

    completion marker present, and naming THIS checkpoint's sha256
    split sha256 == the frozen split's own recorded sha
    row / image counts exactly as the declared conditions imply
    T0 reproduces the plain forward (the §7 control) — ON THE REAL RECORDS
    term_1.00 left `s11_out` unchanged      (the other §7 control)
    the D9 / D10 gate: are they signed into docs/LOCKED_ANALYSIS.md?

Nothing here computes a result. It answers one question — is the evidence
complete and is it what it claims to be — and it answers it from the
committed files alone. A table built on unverified records is not a table
this project reports.

Exit status is 0 when every check passes and 1 otherwise, so a Phase C
session can refuse to proceed on a partial sweep.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MISSING = "MISSING"

#: The declared axes, from the committed YAMLs (read, not restated).
I5_ROOT = REPO / "results" / "frozen" / "I5_readout" / "sub2k"
I7_ROOT = REPO / "results" / "frozen" / "I7_attention" / "sub1k"
SEG_ROOT = REPO / "results" / "frozen" / "I5_readout" / "seg"
MANIFEST = REPO / "results" / "frozen" / "I0_manifest" / "manifest.json"
LOCKED = REPO / "docs" / "LOCKED_ANALYSIS.md"


def split_sha(name):
    doc = json.loads((REPO / "results" / "frozen" / "splits"
                      / f"{name}.json").read_text(encoding="utf-8"))
    return doc["sha256"], int(doc["n"])


def read_parquet(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns)


def check_i5a(cohort, problems, notes):
    """Row counts, shas, and the two §7 controls on the REAL records."""
    from saga.frozen.runner import load_conditions

    conditions = load_conditions(REPO / "configs" / "frozen"
                                 / "I5_readout.yaml")
    n_t = len(conditions["transforms"])
    n_d = len(conditions["descriptors"])
    n_s = len(conditions["conditions"][0]["stages"])
    sha, n_images = split_sha("sub2k")

    t0_worst, s11_worst, n_rows_total = 0.0, 0.0, 0
    for r in cohort:
        rid, var = r["run_id"], r["variant"]
        d = I5_ROOT / rid
        marker = d / "corr_records.done.json"
        if not marker.exists():
            problems.append(f"I5a {rid}: no completion marker")
            continue
        M = json.loads(marker.read_text(encoding="utf-8"))
        T = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        if M["ckpt_sha256"] != r["ckpt_sha256"]:
            problems.append(f"I5a {rid}: ckpt sha mismatch")
        if M["split_sha256"] != sha:
            problems.append(f"I5a {rid}: split sha mismatch")

        n_cond = 2 if var == "saga" else 1
        want = n_cond * n_t * n_s * n_d * n_images
        tbl = read_parquet(d / "corr_records.parquet",
                           columns=["condition_id", "stage",
                                    "max_abs_s11_diff_vs_native"])
        got = tbl.num_rows
        n_rows_total += got
        if got != want:
            problems.append(f"I5a {rid}: {got} rows, expected {want}")

        # §7 control 1: T0 reproduces the plain forward, on the real model.
        t0 = T.get("t0_matches_plain_forward", {})
        for stage, v in t0.items():
            # `isinstance(True, int)` is True in Python, and this dict
            # carries `pair_members_identical` (a bool) and `n_images` (a
            # count) beside the per-stage differences. Both must be excluded
            # or a healthy run reads as a failure.
            if isinstance(v, bool) or stage == "n_images":
                continue
            if isinstance(v, (int, float)):
                t0_worst = max(t0_worst, abs(float(v)))
                if float(v) != 0.0:
                    problems.append(f"I5a {rid}: T0 differs at {stage} = {v}")
        if t0.get("pair_members_identical") is not True:
            problems.append(f"I5a {rid}: T0 pair members are not identical")
        if T.get("state_restored") is not True:
            problems.append(f"I5a {rid}: state not restored")

        # §7 control 2: term_1.00 cannot change s11_out.
        if var == "saga":
            cols = tbl.to_pydict()
            vals = {v for c, s, v in zip(cols["condition_id"], cols["stage"],
                                         cols["max_abs_s11_diff_vs_native"])
                    if c == "term_1.00"}
            if vals != {"0.0"}:
                problems.append(
                    f"I5a {rid}: term_1.00 moved s11_out: {sorted(vals)[:4]}")
            else:
                s11_worst = max(s11_worst, 0.0)
    notes["I5a_runs"] = len(cohort)
    notes["I5a_rows"] = n_rows_total
    notes["I5a_T0_max_abs_diff"] = t0_worst
    notes["I5a_term_s11_max_abs_diff"] = s11_worst
    notes["I5a_axes"] = {"transforms": n_t, "stages": n_s, "descriptors": n_d,
                         "images": n_images}


def check_i7(cohort, problems, notes):
    import numpy as np

    sha, n_images = split_sha("sub1k")
    cell = [r for r in cohort if r["arch"] == "vit_small"
            and r["recipe_actual"] == "mixup"]
    shapes = {}
    for r in cell:
        rid = r["run_id"]
        d = I7_ROOT / rid
        marker = d / "incoming_mass.done.json"
        if not marker.exists():
            problems.append(f"I7 {rid}: no completion marker")
            continue
        M = json.loads(marker.read_text(encoding="utf-8"))
        if M["ckpt_sha256"] != r["ckpt_sha256"]:
            problems.append(f"I7 {rid}: ckpt sha mismatch")
        if M["split_sha256"] != sha:
            problems.append(f"I7 {rid}: split sha mismatch")
        with np.load(d / "incoming_mass.npz", allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            for q in ("in_mass_patchq", "in_mass_clsq", "value_norm"):
                a = z[q]
                shapes.setdefault(int(meta["n_prefix"]), {})[q] = list(a.shape)
                if a.shape[0] != n_images:
                    problems.append(
                        f"I7 {rid}: {q} has {a.shape[0]} images, "
                        f"expected {n_images}")
                if a.ndim >= 4:
                    problems.append(f"I7 {rid}: {q} looks like a stored map")
            # post-softmax rows sum to 1 -> the per-key mean does too
            tot = z["in_mass_patchq"].sum(axis=-1)
            if not np.allclose(tot, 1.0, atol=1e-4):
                problems.append(
                    f"I7 {rid}: in_mass_patchq does not sum to 1 over keys "
                    f"(max dev {float(np.abs(tot - 1).max()):.2e})")
            if meta.get("state_restored") is not True:
                problems.append(f"I7 {rid}: state not restored")
            if meta.get("alignment") != {"10": "in_b10", "11": "s11_out"}:
                problems.append(f"I7 {rid}: alignment is {meta.get('alignment')}")
    notes["I7_runs"] = len(cell)
    notes["I7_shapes_by_prefix"] = shapes


def check_i5b(problems, notes):
    if not SEG_ROOT.exists():
        notes["I5b"] = "DROPPED (no output directory)"
        return
    summary = SEG_ROOT / "I5b_summary.json"
    if not summary.exists():
        problems.append("I5b: no I5b_summary.json — the evaluator did not "
                        "finish")
        notes["I5b"] = "INCOMPLETE"
        return
    doc = json.loads(summary.read_text(encoding="utf-8"))
    for key, run in doc["runs"].items():
        rid = key.split("/")[0]
        if not (SEG_ROOT / rid / "seg_records.done.json").exists():
            problems.append(f"I5b {rid}: no completion marker")
    notes["I5b"] = "RUN"
    notes["I5b_gate"] = doc["gate_reason"]
    notes["I5b_miou"] = {k: v["mIoU"] for k, v in doc["runs"].items()}


def check_locked(notes):
    """The D9 / D10 gate. Reported, never decided here."""
    text = LOCKED.read_text(encoding="utf-8")
    notes["locked_status_frozen"] = "STATUS: **FROZEN**" in text
    notes["D9_present"] = "D9" in text
    notes["D10_present"] = "D10" in text
    notes["locked_signature_pending"] = "Date frozen: `PENDING`" in text


def main():
    p = argparse.ArgumentParser(description="TASK C / Phase C verification.")
    p.add_argument("--json", default=None, help="also write the report here")
    args = p.parse_args()

    from saga.frozen.runner import eligible_cohort

    cohort = eligible_cohort(MANIFEST)
    problems, notes = [], {}
    check_i5a(cohort, problems, notes)
    check_i7(cohort, problems, notes)
    check_i5b(problems, notes)
    check_locked(notes)

    report = {"problems": problems, "notes": notes, "ok": not problems}
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8", newline="\n")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
