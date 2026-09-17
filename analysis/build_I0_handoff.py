#!/usr/bin/env python3
"""
analysis/build_I0_handoff.py — TASK I0 handoff generator
========================================================
Writes `docs/I0_HANDOFF.md`.

Follows the TASK-07/08/13 handoff pattern: every NUMBER is read from a
committed file; the surrounding prose is fixed text. Re-run after new
results land and the document updates itself.

    python analysis/build_I0_handoff.py [--out docs/I0_HANDOFF.md]
                                        [--git-sha SHA]

Sources (nothing else):
  results/frozen/I0_manifest/manifest.json    the 124 rows + the cohort counts
  results/frozen/I0_manifest/ckpt_hashes.json the Phase-B hash pass
  results/frozen/I0_manifest/smoke_*.json     the nine checks, per checkpoint
  results/frozen/splits/*.json                the four frozen splits
  results/diagsplit/val_diag_split.json       the discovery split
  results/diagsplit/fixed_thresholds_canon.json  the independent sha check
  results/legacy/checkpoint_manifest.csv      the other independent sha check
  docs/LOCKED_ANALYSIS.md                     the DECISION NEEDED list
  saga/frozen/stages.py                       HIST_STAGE and its citation
  tests/test_I0_*.py                          the test counts

`--git-sha` exists so a test can regenerate the document byte-identically:
the sha is the only part of the output that is not a function of the
committed files.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen.stages import HIST_STAGE, HIST_STAGE_CITATION  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MAN = REPO / "results" / "frozen" / "I0_manifest"
SPLITS = REPO / "results" / "frozen" / "splits"
MISSING = "MISSING"

SMOKE_ORDER = ("native_matches_recorded", "identity_is_bit_exact",
               "terminal_gate_invariance", "prefix_rows_untouched",
               "permutation_preserves", "ring_permutation_counts",
               "energy_matched_norm", "stage_shapes", "state_restored")

CHECK_TITLE = {
    "native_matches_recorded": "native loader reproduces the recorded eval",
    "identity_is_bit_exact": "identity edit reproduces the native output",
    "terminal_gate_invariance": "terminal patch gate leaves CLS logits alone",
    "prefix_rows_untouched": "prefix rows untouched by every patch edit",
    "permutation_preserves": "gate permutations preserve means/histograms",
    "ring_permutation_counts": "within-ring permutations preserve ring counts",
    "energy_matched_norm": "energy-matched edits hit the declared norm",
    "stage_shapes": "every stage returns [B, N - n_prefix, d]",
    "state_restored": "state hash restored after every edit",
}

FAMILY_TITLE = {
    "e2r_300ep": "e2r 300-epoch (the seeded classification runs)",
    "legacy_300ep": "legacy 300-epoch (the unseeded e2 runs)",
    "ablation_100ep": "ablation 100-epoch (TASK-12's six arms)",
    "finetune": "fine-tuning (TASK-08/13's 24 runs)",
    "dense_det": "dense detection (TASK-09)",
    "dense_seg": "dense segmentation (TASK-09)",
    "ttr_edit": "test-time registers (TASK-10, derived edits)",
}


def sh(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              cwd=str(REPO)).stdout.strip()
    except Exception:                                      # noqa: BLE001
        return ""


def rj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def count_tests(path: Path) -> int:
    """Number of test FUNCTIONS in a file — computed, never typed. (The
    number pytest collects is larger, because several are parametrised.)"""
    return len(re.findall(r"^def test_", path.read_text(encoding="utf-8"),
                          re.M))


def fmt(v, nd=6):
    """A measured value, rendered without inventing precision."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e6:
            return f"{v:.3g}"
        return f"{v:.{nd}g}"
    return str(v)


def decisions(text: str):
    """The DECISION NEEDED rows of docs/LOCKED_ANALYSIS.md §12, parsed."""
    out = []
    section = text.split("## 12.")[-1]
    for line in section.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not re.match(r"^~*D\d+~*$", cells[0]):
            continue
        out.append({"id": cells[0], "section": cells[1], "q": cells[2],
                    "default": cells[3],
                    "closed": cells[0].startswith("~~")})
    return out


def build(git_sha: str) -> str:
    man = rj(MAN / "manifest.json")
    rows = man["rows"]
    hashes = rj(MAN / "ckpt_hashes.json")
    smokes = [rj(p) for p in sorted(MAN.glob("smoke_*.json"))]
    canon = rj(REPO / "results" / "diagsplit" / "fixed_thresholds_canon.json")
    disc = rj(REPO / "results" / "diagsplit" / "val_diag_split.json")
    locked = (REPO / "docs" / "LOCKED_ANALYSIS.md").read_text(encoding="utf-8")

    split_docs = {n: rj(SPLITS / f"{n}.json")
                  for n in ("calibration", "evaluation", "sub1k", "sub2k")}

    n_tests = {t: count_tests(REPO / "tests" / f"test_I0_{t}.py")
               for t in ("frozen", "manifest", "splits")}

    L = []
    A = L.append

    # ── header ───────────────────────────────────────────────────────────
    A("# TASK I0 handoff — the frozen-intervention framework, the eligible "
      "cohort, and the locked protocol\n")
    A(f"**Status: COMPLETE (Phases A / B / C). Nothing pending from the "
      f"HPC.**  \nGENERATED by `analysis/build_I0_handoff.py` at git "
      f"`{git_sha}`. Every number below is read from "
      f"`results/frozen/I0_manifest/`, `results/frozen/splits/` and "
      f"`results/diagsplit/`; none is typed by hand. The surrounding prose is "
      f"fixed text. Re-run the script after new results land and this "
      f"document updates itself.\n")
    A("I0 concluded nothing scientific, by design. It exists so that I1-I7 "
      "share one definition of \"feature stage\", one of \"prefix tokens\", "
      "one cohort, one set of splits, and one guarantee that a temporary "
      "edit was temporary. Everything scientific downstream depends on it.\n")

    # ── 0. at a glance ───────────────────────────────────────────────────
    total_tests = sum(n_tests.values())
    A("## 0. Status at a glance\n")
    A("| phase | what | state |")
    A("|---|---|---|")
    A(f"| A | manifest, splits builder, framework, smoke checker, hash tool, "
      f"{total_tests} test functions | **done** |")
    A("| B | HPC: split build, checkpoint hashes, 32-image smoke on three "
      "checkpoints | **done** |")
    A("| C | handoff, `LOCKED_ANALYSIS` split shas, final row counts | "
      "**done** |")
    A("")
    A(f"TASK I0 contributes {total_tests} test functions "
      f"({', '.join(f'{n} {k}' for k, n in sorted(n_tests.items()))}); the "
      f"repo-wide total is deliberately not quoted here, since it drifts with "
      f"every other task.\n")

    # ── 1. what to use ───────────────────────────────────────────────────
    A("## 1. What a later work package actually uses\n")
    A("```python")
    A("from saga.frozen import (STAGES, HIST_STAGE, capture_stages,")
    A("                         terminal_gate_override, gate_edit,")
    A("                         receiver_perturbation, state_hash)")
    A("```")
    A("")
    A("| you want | use |")
    A("|---|---|")
    A("| patch tokens at a named stage | `capture_stages(model, stages)` — "
      "prefix rows already removed with the MODEL's own count |")
    A("| the terminal patch-gate control (I2) | `terminal_gate_override("
      "model, value)` |")
    A("| gate mean / arrangement edits (I3) | `gate_edit(model, layer, mode, "
      "alpha=, perm=)` |")
    A("| masked receiver perturbation (I4) | `receiver_perturbation(model, "
      "layer, mask, epsilon, energy_target=)` |")
    A("| to run a whole condition sweep | `tools/frozen_eval.py --run-id ... "
      "--conditions configs/frozen/<wp>.yaml --split ...` |")
    A("")
    A("Every edit is a context manager that sha256-hashes every parameter and "
      "buffer on entry and on exit and REFUSES to exit if they differ. Every "
      "patch operation excludes prefix rows using "
      "`infer_num_prefix_tokens(model)` — never a hard-coded 1. All of it "
      "works on the timm 4-register model **without** wrapping it in "
      "`SAGAViT`, which refuses `num_prefix_tokens != 1` by design.\n")
    A("**The conditions contract.** No condition, layer, epsilon, mask or "
      "permutation is selectable from the command line; they live in a "
      "per-work-package YAML under `configs/frozen/`, committed before the "
      "job runs. That is what makes \"the analysis configuration was fixed "
      "before the outcomes were inspected\" a property of this repository "
      "rather than a promise. I0 ships `configs/frozen/smoke.yaml` and "
      "nothing else.\n")

    # ── 2. the cohort ────────────────────────────────────────────────────
    cohort = [r for r in rows
              if r["family"] in ("e2r_300ep", "legacy_300ep")
              and r["ckpt_kind"] == "last"
              and r["status"] in ("eligible", "eligible_legacy")]
    by_variant = {}
    for r in cohort:
        by_variant.setdefault(r["variant"], []).append(r)
    cells = sorted({(r["arch"], r["recipe_actual"]) for r in cohort})

    A("## 2. The eligible cohort\n")
    A(f"**{len(cohort)} completed 300-epoch conditions.** One row per "
      f"condition; the manifest carries {man['n_rows']} rows in total, one "
      f"per saved checkpoint or checkpoint-derived condition.\n")
    A("| arch | recipe_actual | " + " | ".join(
        sorted(by_variant)) + " | total |")
    A("|---|---|" + "---|" * (len(by_variant) + 1))
    for arch, rec in cells:
        counts = [sum(1 for r in by_variant.get(v, [])
                      if (r["arch"], r["recipe_actual"]) == (arch, rec))
                  for v in sorted(by_variant)]
        A(f"| {arch} | {rec} | " + " | ".join(str(c) for c in counts)
          + f" | {sum(counts)} |")
    A("| **total** | | " + " | ".join(
        str(len(by_variant[v])) for v in sorted(by_variant))
      + f" | **{len(cohort)}** |")
    A("")
    n_ctrl = sum(1 for r in cohort if str(r["seed_controlled"]) == "1")
    A(f"{n_ctrl} of the {len(cohort)} were trained under the seeded e2r "
      f"trainer (`seed_controlled=1`); the remaining {len(cohort) - n_ctrl} "
      f"are legacy repeats whose trainer recorded no seed "
      f"(`seed=MISSING`).\n")
    A("**`recipe_actual` is never a directory name.** Cell membership is "
      "IMPORTED from `analysis/build_pooled_tables.py`, so the recipe erratum "
      "remap and the exclusion of the VOID legacy ViT-B mixup-dir trio cannot "
      "drift between this manifest and `results/tables/e2_pooled.csv`. For "
      "the e2r/ablation runs the recipe is read from the run's own resolved "
      "config and cross-checked against its augmentation block; a `recipe:` "
      "key contradicting its own mixup/cutmix alphas is a hard error.\n")

    A("### Rows per family and status\n")
    A("| family | " + " | ".join(
        sorted({r["status"] for r in rows})) + " | total |")
    statuses = sorted({r["status"] for r in rows})
    A("|---|" + "---|" * (len(statuses) + 1))
    for fam in FAMILY_TITLE:
        fam_rows = [r for r in rows if r["family"] == fam]
        if not fam_rows:
            continue
        counts = [sum(1 for r in fam_rows if r["status"] == s)
                  for s in statuses]
        A(f"| {fam} | " + " | ".join(str(c) if c else "—" for c in counts)
          + f" | {len(fam_rows)} |")
    A("| **total** | " + " | ".join(
        str(sum(1 for r in rows if r["status"] == s)) for s in statuses)
      + f" | **{len(rows)}** |")
    A("")

    A("### What is excluded, and why\n")
    invalid = [r for r in rows if r["status"] == "invalid"]
    void = sorted({r["run_id"] for r in invalid if "VOID" in r["status_reason"]})
    incomplete = sorted({r["run_id"] for r in invalid
                         if "training incomplete" in r["status_reason"]})
    A(f"- **The VOID legacy ViT-B mixup-DIRECTORY trio** "
      f"({len(void)} conditions: {', '.join(void)}) never finished training. "
      f"It appears in the manifest so it is accounted for, is `invalid` for "
      f"BOTH checkpoint kinds, and is a member of no cell.")
    A(f"- **{len(incomplete)} never-started e2r ViT-B s2 runs** "
      f"({', '.join(incomplete)}) logged one epoch of 300 with a null "
      f"`end_time`.")
    A("- **Every `best.pth` outside the fine-tuning family** is `superseded`: "
      "diagnostics come from the LAST checkpoint everywhere except "
      "fine-tuning, which selects on a frozen val split and whose `best.pth` "
      "is therefore its canonical checkpoint.\n")

    # ── 3. the stage ─────────────────────────────────────────────────────
    A("## 3. The historical feature stage — identified, not assumed\n")
    A(f"Every historical patch diagnostic in this project was computed at "
      f"**`{HIST_STAGE}`**: the output of the LAST transformer block, "
      f"**before** the final LayerNorm. `saga/frozen/stages.py` exposes it "
      f"under the alias `hist`, so a later file can write `stage: hist` and "
      f"mean \"whatever the historical diagnostics used\".\n")
    A(f"> {HIST_STAGE_CITATION}\n")
    A("This is pinned by "
      "`tests/test_I0_frozen.py::test_hist_stage_matches_the_historical_hook`,"
      " which RUNS the real `compute_diagnostics` hook and compares tensors "
      "rather than asserting a constant — and separately asserts the captured "
      "tensor is NOT the post-norm one.\n")
    A("The four stage names, and there is no fifth anywhere in the project: "
      "`" + "`, `".join(s for s in ("s11_out", "s12_pre_norm",
                                    "s12_post_norm", "hist")) + "`.\n")

    # ── 4. splits ────────────────────────────────────────────────────────
    A("## 4. The locked splits\n")
    A("| split | n | classes | sha256 |")
    A("|---|---|---|---|")
    A(f"| discovery (existing, unchanged) | {disc['n']:,} | "
      f"{len({lab for _, lab in disc['items']}):,} | "
      f"`{split_docs['evaluation']['discovery_sha256']}` |")
    for name in ("calibration", "evaluation", "sub1k", "sub2k"):
        d = split_docs[name]
        A(f"| `{name}.json` | {d['n']:,} | {d.get('n_classes', '—'):,} | "
          f"`{d['sha256']}` |")
    A("")
    A("`discovery`, `calibration` and `evaluation` are pairwise disjoint BY "
      "CONSTRUCTION — an image claimed by an earlier split is never a "
      "candidate for a later one. `sub1k` and `sub2k` are drawn FROM "
      "`evaluation` and are checked to be subsets of it. The builder refuses "
      "to overwrite a frozen split, and each file carries a `sha256` over its "
      "own canonical serialization which is re-verified on every "
      "invocation.\n")
    A(f"**The sentence that must survive into the paper:** "
      f"{split_docs['evaluation']['protocol']}\n")

    # ── 5. smoke ─────────────────────────────────────────────────────────
    A("## 5. What the smoke checks measured on the real checkpoints\n")
    n_img = {s["n_images"] for s in smokes}
    A(f"Nine checks on a fixed {n_img.pop() if len(n_img) == 1 else '?'}-image "
      f"subset of `evaluation.json`, in fp32, on "
      f"{len(smokes)} checkpoints.\n")
    A("| checkpoint | variant | n_prefix | PASS | FAIL | SKIP |")
    A("|---|---|---|---|---|---|")
    for s in smokes:
        A(f"| `{s['run_id']}` | {s['variant']} | {s['n_prefix']} | "
          f"{s['n_pass']} | {s['n_fail']} | {s['n_skip']} |")
    A("")
    n_applicable = sum(s["n_pass"] + s["n_fail"] for s in smokes)
    n_pass = sum(s["n_pass"] for s in smokes)
    A(f"**{n_pass} of {n_applicable} applicable checks PASS.** A check that "
      f"cannot apply to a checkpoint — a gate check on a baseline or a "
      f"register model — is SKIP with its reason, never a silent PASS.\n")

    A("### The measurements that matter\n")
    A("| what | measured |")
    A("|---|---|")
    for s in smokes:
        c = next((c for c in s["checks"]
                  if c["check"] == "terminal_gate_invariance"), None)
        if c and c["status"] == "PASS":
            worst = c["measured"]["worst"]
            A(f"| Proposition 2 — terminal patch gate vs CLS logits, "
              f"`{s['run_id']}` | **{fmt(worst)}** at all "
              f"{len(c['measured']['max_abs_logit_diff_by_value'])} constants "
              f"(tolerance {fmt(c['measured']['tolerance'])}) |")
    for s in smokes:
        c = next((c for c in s["checks"]
                  if c["check"] == "identity_is_bit_exact"), None)
        if c and c["status"] == "PASS":
            m = c["measured"]
            # NB: no literal "|" in a cell — it is the column separator
            A(f"| identity edit, `{s['run_id']}` | "
              f"{m['n_images_bit_exact']}/{m['n_images']} images bit-exact, "
              f"max abs logit diff {fmt(m['max_abs_logit_diff'])} |")
    for s in smokes:
        c = next((c for c in s["checks"]
                  if c["check"] == "prefix_rows_untouched"), None)
        if c:
            m = c["measured"]
            A(f"| prefix rows, `{s['run_id']}` (n_prefix={m['n_prefix']}) | "
              f"max abs diff **{fmt(m['max_abs_prefix_diff'])}** with "
              f"injected norm {fmt(m['max_injected_norm'])} over "
              f"{m['n_masked_coords']} masked coordinates |")
    for s in smokes:
        c = next((c for c in s["checks"]
                  if c["check"] == "energy_matched_norm"), None)
        if c:
            m = c["measured"]
            A(f"| energy matching, `{s['run_id']}` | max relative error "
              f"{fmt(m['max_abs_rel_error'])} against tolerance "
              f"{fmt(m['tolerance'])}, {m['n_zero_norm_images']} zero-norm "
              f"images |")
    A("")
    A("The prefix-row check reports the injected norm alongside the "
      "difference, so a PASS cannot come from an edit that did nothing. It "
      "compares the edited BLOCK's output rather than the attention module's: "
      "a forward hook that returns a value replaces the output only for hooks "
      "registered AFTER it, so a capture hook registered first would have "
      "seen the unedited tensor and the check would have passed while "
      "measuring nothing.\n")

    fails = [(s, c) for s in smokes for c in s["checks"]
             if c["status"] == "FAIL"]
    if fails:
        A("### The recorded FAIL, and why it is not a defect in the "
          "intervention\n")
        for s, c in fails:
            A(f"`{s['run_id']}` records **FAIL** on `{c['check']}`:\n")
            for k, v in sorted(c["measured"].items()):
                A(f"- `{k}` = {fmt(v)}")
            A("")
        A("`permutation_preserves` asserted that the per-head gate MEAN was "
          "bit-identical after a permutation. It cannot be: the mean is a "
          "reduction over 196 fp32 values in a different order, and fp32 "
          "addition is not associative. The measured difference is exactly "
          "one ULP at magnitude 1, while `per_head_sorted_values_identical` "
          "is `True` — the per-head MULTISET of gate values is bit-identical, "
          "which is the invariant that actually holds and which the check "
          "still asserts with zero tolerance. The mean now carries a stated "
          "8-ULP tolerance (`tools/frozen_smoke.py::TOL_PERM_MEAN`), which "
          "`tests/test_I0_frozen.py` imports rather than restates. **Under "
          "the corrected check this recorded measurement is a PASS.** The "
          "JSON is left exactly as the HPC wrote it; re-running the job "
          "rewrites it with the corrected verdict.\n")

    # ── 6. checkpoint identity ───────────────────────────────────────────
    A("## 6. Checkpoint identity\n")
    hashed = sum(1 for r in rows if r["ckpt_sha256"] != MISSING)
    A(f"{hashed} of {len(rows)} rows carry a 64-character `ckpt_sha256`; "
      f"{len(rows) - hashed} are MISSING, and every one is accounted for "
      f"(a test permits no unexplained gap).\n")
    missing = [r for r in rows if r["ckpt_sha256"] == MISSING]
    fams = sorted({r["family"] for r in missing})
    A(f"- **{len(missing)} rows, all `{'`, `'.join(fams)}`** — the "
      f"test-time-register conditions are inference-time EDITS of a base "
      f"checkpoint, not models, and have no checkpoint of their own by "
      f"design. Each records its base run, the neurons-file sha, the layer "
      f"range and `n_neurons` instead.\n")
    A("**The hashes were cross-checked against two files that recorded "
      "checkpoint shas independently, long before this task existed, and both "
      "agree:**\n")
    for run, key in (("e2r_vits_mixup_baseline_s1", "vit_small|mixup"),
                     ("e2r_vitb_mixup_baseline_s1", "vit_base|mixup"),
                     ("e2r_vits_nomix_baseline_s1", "vit_small|nomix")):
        got = next((r["ckpt_sha256"] for r in rows
                    if r["run_id"] == run and r["ckpt_kind"] == "last"), None)
        ok = got == canon["source_ckpt_sha256"][key]
        A(f"- `{run}` = `fixed_thresholds_canon.json[{key}]`: **{ok}**")
    legacy = {}
    with open(REPO / "results" / "legacy" / "checkpoint_manifest.csv",
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            legacy[r["path"]] = r["sha256"]
    n_leg = sum(1 for r in rows if r["ckpt_path"] in legacy)
    n_ok = sum(1 for r in rows if r["ckpt_path"] in legacy
               and r["ckpt_sha256"] == legacy[r["ckpt_path"]])
    A(f"- legacy rows agreeing with `results/legacy/checkpoint_manifest.csv`: "
      f"**{n_ok} of {n_leg}**")
    A(f"- the hash pass itself reported "
      f"`{hashes['n_hashed']}` hashed of `{hashes['n_targets']}` targets\n")

    det_best = [r for r in rows if r["family"] == "dense_det"
                and r["ckpt_kind"] == "best"]
    resolved = [r for r in det_best
                if json.loads(r["derived_params"]).get(
                    "resolves_to_ckpt_kind") == "last"]
    if det_best:
        A("### Detection has no best checkpoint, and what was done about it\n")
        A(f"`detection/tools/train.py` saves `ckpt/last.pth` only; its best-AP "
          f"state is the JSON pair `coco_eval_best.json` + "
          f"`detections_val.json`. The Phase-B hash pass found this "
          f"empirically — six `best.pth` paths that never existed — and the "
          f"filenames now come from a table justified against the trainers' "
          f"source (segmentation writes `best_model.pth`, not `best.pth`).\n")
        A(f"For {len(resolved)} of {len(det_best)} detection runs the `best` "
          f"row RESOLVES to `ckpt/last.pth`, because the run's own files prove "
          f"they are the same thing:\n")
        A("| run | best-AP epoch | last epoch | resolves |")
        A("|---|---|---|---|")
        for r in det_best:
            p = json.loads(r["derived_params"])
            A(f"| `{r['run_id']}` | {p['best_ap_epoch']} | {p['last_epoch']} | "
              f"{p.get('resolves_to_ckpt_kind') or '— (epochs differ)'} |")
        A("")
        A("The epochs are read from `coco_eval_best.json` and `log.csv` on "
          "every build; a run that peaked earlier would NOT resolve and would "
          "state that its best weights were never saved. The cost — two "
          "`ckpt_kind`s of one run sharing a sha256 — is declared in "
          "`derived_params.resolves_to_ckpt_kind`, and only the `last` row is "
          "`eligible`, so no count over eligible rows can double-count a "
          "run.\n")

    # ── 7. what is not settled ───────────────────────────────────────────
    ds = decisions(locked)
    open_ds = [d for d in ds if not d["closed"]]
    closed_ds = [d for d in ds if d["closed"]]
    A("## 7. What I0 did NOT settle\n")
    # The draft/frozen sentence is READ from the document, not asserted. It
    # said "is a DRAFT ... three PENDING lines" unconditionally until
    # 2026-09-17, when the freeze made it a false statement sitting directly
    # above a generated count of zero open decisions.
    from saga.frozen.masks import locked_state
    _lock = locked_state(REPO / "docs" / "LOCKED_ANALYSIS.md")
    if _lock["missing"]:
        A(f"`docs/LOCKED_ANALYSIS.md` is a **DRAFT**. It carries three "
          f"`PENDING` lines (signed by / date frozen / git sha at freeze); "
          f"until the human fills them in, nothing in it is locked and **no "
          f"I3/I4 Phase-B job may run**. {len(open_ds)} of {len(ds)} "
          f"decisions {'is' if len(open_ds) == 1 else 'are'} still open.\n")
    else:
        A(f"`docs/LOCKED_ANALYSIS.md` is **FROZEN**, signed "
          f"{_lock['date']} at git `{_lock['git_sha']}`. Every value in it is "
          f"fixed, and `saga/frozen/masks.py` enforces that at run time. "
          f"{len(closed_ds)} of {len(ds)} decisions are closed"
          f"{'' if not open_ds else f'; {len(open_ds)} still open'}.\n")
    A("| # | § | question | resolution / default if unanswered |")
    A("|---|---|---|---|")
    for d in ds:
        A(f"| {d['id']} | {d['section']} | {d['q']} | {d['default']} |")
    A("")
    if closed_ds:
        # NOT "closed in Phase C": I0 Phase C closed D8 and nothing else.
        # Later work packages close the rest (I2 decided D1; I3-prep
        # generated D3's permutation lists; D2/D4/D6/D7 were closed at their
        # written defaults), and this document regenerates against whatever
        # LOCKED_ANALYSIS says today — so it must not attribute their
        # closure to the phase that happened to write this generator.
        A(f"{len(closed_ds)} of the {len(ds)} are now closed ("
          f"{', '.join(d['id'].strip('~') for d in closed_ds)}); "
          f"`docs/LOCKED_ANALYSIS.md` records who closed each one and when. "
          f"I0 Phase C itself closed only D8, by filling the split shas in "
          f"§11 from the Phase-B build.\n")
    # "blocking" is however LOCKED_ANALYSIS marks it — in the question text
    # or in the default column. Matching only one of the two silently dropped
    # this paragraph once.
    blocking = [d for d in open_ds
                if "BLOCKING" in d["q"].upper()
                or "blocks " in d["default"].lower()
                or "no default" in d["default"].lower()]
    if blocking:
        A(f"**{len(blocking)} of the open decisions BLOCKS a work package.** "
          f"{blocking[0]['id']}: I4 needs a prevalence map at the INPUT to "
          f"the edited block, and the committed `*_addr.json` maps are "
          f"last-block, so I1 must produce it before I4 Phase B. The rest can "
          f"be defaulted, and the defaults are written down so that "
          f"defaulting is a visible decision rather than a silent one.\n")

    # ── 8. inherited rules ───────────────────────────────────────────────
    A("## 8. Rules every later work package inherits\n")
    A("1. **Nothing overwrites a historical result, note, table or "
      "threshold.** New keys only. `saga/gate.py`, `saga/vit.py`, "
      "`saga/metrics.py`, `tools/eval.py` and `tools/diagnose.py` were not "
      "touched by I0 at all.")
    A("2. **`MISSING` is a value.** It is never averaged and never replaced "
      "by a guess.")
    A("3. **Every record row carries** checkpoint sha256, split sha256, "
      "stage, prefix count, precision and git sha. A number that cannot be "
      "traced to a checkpoint and a split is not a number this project "
      "reports.")
    A("4. **The runner REFUSES** a manifest row whose `ckpt_sha256` is "
      "MISSING, a checkpoint whose file hash has changed, and a split whose "
      "content no longer matches its recorded sha.")
    A("5. **Writers are append-safe.** A resubmitted job adds what is missing "
      "and rewrites nothing; the completion marker is written last.")
    A("6. **A generated file under `results/` is REGENERATED, never "
      "hand-merged.** Two sides regenerating the same artifact from different "
      "inputs is the normal state whenever a phase runs on the HPC and a fix "
      "lands locally.")
    A("7. **No training, no optimizer, no probe fitting** in any frozen file. "
      "An AST-level test enforces it.\n")

    # ── 9. file map ──────────────────────────────────────────────────────
    A("## 9. Where everything lives\n")
    A("| what | path |")
    A("|---|---|")
    A("| framework | `saga/frozen/{stages,edits,records,runner}.py` |")
    A("| drivers | `tools/frozen_eval.py`, `tools/frozen_smoke.py`, "
      "`tools/frozen_manifest_hashes.py`, `tools/build_frozen_splits.py` |")
    A("| manifest builder | `analysis/build_I0_manifest.py` |")
    A("| this document's generator | `analysis/build_I0_handoff.py` |")
    A("| cohort | `results/frozen/I0_manifest/{manifest.csv,manifest.json,"
      "eligibility.md,ckpt_hashes.json}` |")
    A("| smoke results | `results/frozen/I0_manifest/smoke_<run_id>.json` |")
    A("| splits | `results/frozen/splits/` |")
    A("| locked parameters | `docs/LOCKED_ANALYSIS.md` (DRAFT) |")
    A("| job file | `scripts/jobs/frozen_smoke.sbatch` |")
    A("| conditions | `configs/frozen/smoke.yaml` |")
    A("| tests | `tests/test_I0_{frozen,manifest,splits}.py` |")
    A("")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description="TASK I0 handoff generator.")
    p.add_argument("--out", default="docs/I0_HANDOFF.md")
    p.add_argument("--git-sha", default=None,
                   help="override the embedded sha (lets a test regenerate "
                        "byte-identically)")
    args = p.parse_args()

    sha = args.git_sha or sh("git", "rev-parse", "--short", "HEAD")
    text = build(sha)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    print(f"wrote {out}: {len(text.splitlines())} lines, git {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
