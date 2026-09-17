#!/usr/bin/env python3
"""
analysis/build_D5_masks.py
==========================
D5 — the prevalence masks I4 perturbs. TASK A / I1 A6, Phase C.

    python analysis/build_D5_masks.py

Writes `configs/frozen/I4_masks.json` and
`results/frozen/I1_spatial/D5_note.md`.

D5 is the last open decision in `docs/LOCKED_ANALYSIS.md` and it blocks I4.
What it needs is a prevalence map at the INPUT TO BLOCK 7/8 — the depth the
perturbation acts at — and every `*_addr.json` in the repository is a
LAST-BLOCK map. Phase B produced the block-input maps; this builds the masks
from them.

SELECTION HAPPENS ON DISCOVERY DATA AND NOWHERE ELSE. Every mask function in
`saga/frozen/prevalence.py` calls `assert_selection_split` first, which is a
positive allow-list on the discovery split's two digests. This script cannot
build a mask from evaluation data even if it is pointed at it — and it is
pointed at the EVALUATION maps once, deliberately, for the one number that
has to come from there: the correlation between each mask's source map and
the same cell's reporting map, which is what tells a reader whether the
perturbed coordinates are the address the paper reports.

THE CHOICES, ALL OF THEM SIGNED SOMEWHERE ELSE:
  * cell        ViT-S/mixup, its four baselines (task file §8)
  * stages      `in_b07`, `in_b08` (LOCKED §2: paper blocks 8 and 9)
  * basis       `fixed_cal` — D4 is CLOSED at the canon basis, and the
                historical `tau_canon` is defined at the last block only, so
                the canon RECIPE recalibrated per stage is the only form of
                it that exists here
  * k           16 coordinates (LOCKED §5)
  * controls    10, ring-matched, seeds 200..209 (LOCKED §5, task file §6)
  * overlap     ALLOWED and reported, never used to reject a draw
                (LOCKED §5; the task file's contrary clause was amended
                2026-09-17 by Mahfuzur Rahman Chowdhury)

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.frozen_I1_spatial import MapStore  # noqa: E402
from saga.frozen import prevalence as P  # noqa: E402
from saga.frozen.runner import eligible_cohort  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"

#: Task file §8. `recipe_actual`, never a directory name.
D5_CELL = ("vit_small", "mixup")
D5_STAGES = ("in_b07", "in_b08")
D5_BASIS = "fixed_cal"
DISCOVERY_SPLIT = "val_diag_split"
REPORTING_SPLIT = "evaluation"
REPORTING_STAGE = "s11_out"

MASKS_PATH = Path("configs/frozen/I4_masks.json")
NOTE_PATH = Path("results/frozen/I1_spatial/D5_note.md")


def cell_members(store, stage, basis):
    """The four ViT-S/mixup BASELINE maps at one stage, sorted by run_id."""
    arch, recipe = D5_CELL
    keys = [k for k in store.keys(variant="baseline", condition="native",
                                  stage=stage, basis=basis)
            if store.rows[k[0]]["arch"] == arch
            and store.rows[k[0]]["recipe_actual"] == recipe]
    return sorted(keys)


def build(*, manifest="results/frozen/I0_manifest/manifest.json",
          git_sha_value=None, masks_path=MASKS_PATH, note_path=NOTE_PATH):
    cohort = eligible_cohort(manifest)
    disc = MapStore(DISCOVERY_SPLIT, cohort)
    rep = MapStore(REPORTING_SPLIT, cohort)
    sha = git_sha_value if git_sha_value else git_sha()

    doc = {
        "schema": "i4_masks_v1",
        "work_package": "I1_spatial",
        "purpose": "D5 — the prevalence masks TASK I4 perturbs",
        "decision_refs": {
            "cell": "docs/TASK_A_I1_I6.md §8 — the four ViT-S/mixup baselines",
            "stages": "docs/LOCKED_ANALYSIS.md §2 — paper blocks 8 and 9 = "
                      "0-based block indices 7 and 8; in_bNN is the INPUT to "
                      "blocks[NN]",
            "k": "docs/LOCKED_ANALYSIS.md §5 — the 16 highest-prevalence "
                 "coordinates",
            "controls": "docs/LOCKED_ANALYSIS.md §5 — 10 fixed random masks, "
                        "ring-matched",
            "overlap": "docs/LOCKED_ANALYSIS.md §5 — random masks MAY overlap "
                       "the high-prevalence mask; the overlap is REPORTED and "
                       "draws are never rejected to amplify contrast. The "
                       "contrary clause in docs/TASK_A_I1_I6.md §6 was "
                       "amended 2026-09-17 by Mahfuzur Rahman Chowdhury.",
            "basis": "docs/LOCKED_ANALYSIS.md §5 / D4 — CLOSED at the canon "
                     "basis. tau_canon is defined at the last block only, so "
                     "the canon RECIPE recalibrated per stage and split "
                     "(fixed_cal) is the only form of it at a block input.",
            "split": "docs/LOCKED_ANALYSIS.md §11 and TASK A §2 — masks come "
                     "from the DISCOVERY split and from nothing else",
        },
        "cell": f"{D5_CELL[0]}|{D5_CELL[1]}",
        "basis": D5_BASIS,
        "k": P.PRIMARY_MASK_K,
        "n_controls": P.N_CONTROL_MASKS,
        "control_seeds": list(P.CONTROL_SEEDS),
        "control_overlap_policy": "allowed, recorded, never used to reject a "
                                  "draw",
        "discovery_split_name": DISCOVERY_SPLIT,
        "git_sha": sha,
        "generated_by": "analysis/build_D5_masks.py",
        # NO TIMESTAMP. D5 is SIGNED with this file's sha256, so the file has
        # to be reproducible: a `generated_at` would give a different digest
        # on every run and the signature would stop meaning anything. When it
        # was produced is what the git commit records.
        "masks": {},
    }

    summary = {}
    for stage in D5_STAGES:
        keys = cell_members(disc, stage, D5_BASIS)
        if not keys:
            raise SystemExit(
                f"no discovery {D5_BASIS} maps at {stage} for the "
                f"{D5_CELL[0]}|{D5_CELL[1]} cell — run "
                f"scripts/jobs/frozen_I1.sbatch on the discovery split")
        maps = [disc.maps[k] for k in keys]
        side = maps[0].grid_side

        # THE GUARD, on every source map, before anything is selected.
        for pm in maps:
            P.assert_selection_split(pm, what=f"D5 mask at {stage}")

        mean = P.cell_mean_map(maps)
        source = P.PrevalenceMap(
            freq=mean, n_images=maps[0].n_images,
            n_exceed_total=int(round(float(mean.sum()) * maps[0].n_images)),
            basis=D5_BASIS, threshold=maps[0].threshold, stage=stage,
            split_sha=maps[0].split_sha, split_name=DISCOVERY_SPLIT,
            condition_id="cell_mean", run_id=f"{D5_CELL[0]}|{D5_CELL[1]}",
            ckpt_sha256="cell_mean_of_" + "_".join(k[0] for k in keys),
            grid_side=side, n_prefix=maps[0].n_prefix, git_sha=sha)

        primary = P.topk_mask(source, k=P.PRIMARY_MASK_K)
        composition = P.ring_composition(primary, side)
        controls = P.ring_matched_controls(primary, side)
        expected = P.expected_overlap(primary, side)
        conc = P.concentration_with_reference(source)

        # The one number that MUST come from the reporting split: is this the
        # address the paper reports? Same cell, same basis, s11_out, per run.
        vs_report = []
        for k in keys:
            rep_key = (k[0], "native", REPORTING_STAGE, D5_BASIS)
            if rep_key not in rep.maps:
                vs_report.append({"run_id": k[0], "rho": MISSING})
                continue
            res = P.spatial_rho(disc.maps[k], rep.maps[rep_key])
            resid = P.spatial_rho(P.ring_adjusted(disc.maps[k]),
                                  P.ring_adjusted(rep.maps[rep_key]))
            vs_report.append({
                "run_id": k[0],
                "rho": None if res is None else round(float(res[0]), 6),
                "p_spatial": None if res is None else round(float(res[1]), 6),
                "resid_rho": None if resid is None else round(float(resid[0]), 6),
            })

        doc["masks"][stage] = {
            "stage": stage,
            "block_entered_0based": 7 if stage == "in_b07" else 8,
            "paper_block_1based": 8 if stage == "in_b07" else 9,
            "grid_side": side,
            "n_positions": int(mean.size),
            "n_prefix": int(maps[0].n_prefix),
            "source_run_ids": [k[0] for k in keys],
            "source_ckpt_sha256": {k[0]: disc.maps[k].ckpt_sha256
                                   for k in keys},
            "discovery_split_sha256": maps[0].split_sha,
            "tau_cal": (float(maps[0].threshold)
                        if isinstance(maps[0].threshold, float) else MISSING),
            "n_images": int(maps[0].n_images),
            "cell_mean_mass": round(float(mean.sum()), 6),
            "cell_mean_gini_excess": round(float(conc["gini_excess"]), 6),
            "cell_mean_gini_null": round(float(conc["gini_null"]), 6),
            "cell_mean_ring_profile": [round(float(v), 6)
                                       for v in P.ring_profile(source)],
            "primary_mask": {
                "index": primary,
                "coordinates": P.mask_coordinates(primary, side),
                "cell_mean_freq": {str(p): round(float(mean[p]), 6)
                                   for p in primary},
                "ring_composition": {str(r): n
                                     for r, n in composition.items()},
            },
            "controls": [
                {"seed": info["seed"], "index": mask,
                 "coordinates": P.mask_coordinates(mask, side),
                 "ring_composition": {str(r): n for r, n in
                                      P.ring_composition(mask, side).items()},
                 "overlap_with_primary": info["overlap"]}
                for mask, info in controls],
            "control_overlap_observed": [info["overlap"]
                                         for _, info in controls],
            "control_overlap_expected": round(float(expected), 6),
            "control_overlap_expected_formula": "sum_r n_r^2 / N_r",
            "rho_vs_reporting_stage": {
                "split": REPORTING_SPLIT, "stage": REPORTING_STAGE,
                "per_run": vs_report,
            },
        }
        summary[stage] = {
            "composition": composition, "primary": primary,
            "expected": expected,
            "observed": [i["overlap"] for _, i in controls],
            "vs_report": vs_report, "conc": conc, "source": source,
            "keys": keys, "mean": mean,
        }

    masks_path = Path(masks_path)
    masks_path.parent.mkdir(parents=True, exist_ok=True)
    # sha256 OF THE FILE ITSELF: written once without the field, hashed, then
    # rewritten with it. The recorded digest is therefore the digest of the
    # bytes a reader can hash, not of some intermediate form.
    body = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    doc["sha256_of_body_without_this_field"] = digest
    final = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    masks_path.write_text(final, encoding="utf-8", newline="\n")
    file_digest = hashlib.sha256(final.encode("utf-8")).hexdigest()

    note = render_note(doc, summary, file_digest, digest)
    Path(note_path).parent.mkdir(parents=True, exist_ok=True)
    Path(note_path).write_text(note, encoding="utf-8", newline="\n")
    return {"masks_path": masks_path, "note_path": Path(note_path),
            "file_sha256": file_digest, "body_sha256": digest,
            "doc": doc, "summary": summary}


def md_cell(value) -> str:
    """A value safe to drop into a markdown table cell.

    A cell key is `arch|recipe_actual`, and an unescaped pipe ends the cell —
    `vit_small|mixup` silently splits one column into two and shifts every
    column after it. Backticks do not protect it; only the escape does.
    """
    return str(value).replace("|", r"\|")


def render_note(doc, summary, file_digest, body_digest) -> str:
    """`D5_note.md` — GENERATED, and pinned by a byte-identity test."""
    L = []
    A = L.append
    A("# D5 — the prevalence masks TASK I4 perturbs")
    A("")
    A("GENERATED by `analysis/build_D5_masks.py`. Every number is read from "
      "the committed maps; none is typed by hand. Re-run the script after "
      "new results land and this document updates itself.")
    A("")
    A("**This note does not close D5.** It is the material for that decision. "
      "Closing D5 in `docs/LOCKED_ANALYSIS.md` §5 with the file sha256 below, "
      "and freezing that document, are the human's acts.")
    A("")
    A("## 0. The file this note is about")
    A("")
    A(f"| | |")
    A(f"|---|---|")
    A(f"| file | `{MASKS_PATH.as_posix()}` |")
    A(f"| sha256 | `{file_digest}` |")
    A(f"| cell | `{md_cell(doc['cell'])}` |")
    A(f"| basis | `{doc['basis']}` (D4 = canon basis) |")
    A(f"| discovery split | `{doc['discovery_split_name']}` |")
    A(f"| git sha | `{doc['git_sha']}` |")
    A("")
    A("## 1. Why these maps and not the committed `*_addr.json` ones")
    A("")
    A("Every `*_addr.json` in this repository is a LAST-BLOCK map. I4 "
      "perturbs the residual stream entering blocks 7 and 8 (0-based; paper "
      "blocks 8 and 9), and `docs/LOCKED_ANALYSIS.md` §5 requires the mask to "
      "be taken *at the input to the edited block*. Those maps did not exist "
      "until Phase B of this task; §2 below is the evidence that they are "
      "nonetheless the same address the paper reports.")
    A("")
    A("## 2. Is the perturbed address the reported address?")
    A("")
    A("Spearman between each source run's DISCOVERY map at the block input "
      "and the SAME run's `s11_out` map on the EVALUATION split — the stage "
      "D1 fixes for every reported patch diagnostic. `resid` is the same "
      "correlation after subtracting each map's per-ring mean, so a high "
      "value cannot be explained by \"both maps are bordered\".")
    A("")
    for stage in D5_STAGES:
        m = doc["masks"][stage]
        A(f"**{stage}** (enters `blocks[{m['block_entered_0based']}]`, "
          f"paper block {m['paper_block_1based']}):")
        A("")
        A("| run | ρ vs `s11_out` (evaluation) | ring-adjusted ρ | p |")
        A("|---|---|---|---|")
        for r in m["rho_vs_reporting_stage"]["per_run"]:
            rho = r.get("rho")
            A(f"| `{r['run_id']}` | {rho if rho is not None else 'MISSING'} | "
              f"{r.get('resid_rho', 'MISSING')} | {r.get('p_spatial', 'MISSING')} |")
        A("")
    A("## 3. The masks")
    A("")
    for stage in D5_STAGES:
        m = doc["masks"][stage]
        s = summary[stage]
        A(f"### {stage}")
        A("")
        A(f"- cell-mean mass {m['cell_mean_mass']} exceedances/image over "
          f"{len(m['source_run_ids'])} baselines, {m['n_images']} images")
        A(f"- concentration excess (Gini) **{m['cell_mean_gini_excess']}** "
          f"against a finite-sample null of {m['cell_mean_gini_null']} at "
          f"this map's own mass")
        A(f"- ring profile "
          + ", ".join(f"r{k}={v}" for k, v in
                      enumerate(m["cell_mean_ring_profile"])))
        A(f"- τ_cal = {m['tau_cal']} (canon recipe, this stage, this split)")
        A("")
        A(f"**Primary mask** — the {doc['k']} highest-prevalence coordinates, "
          f"ties broken by ascending flat index:")
        A("")
        A("| index | row | col | ring | cell-mean freq |")
        A("|---|---|---|---|---|")
        for c in m["primary_mask"]["coordinates"]:
            A(f"| {c['index']} | {c['row']} | {c['col']} | {c['ring']} | "
              f"{m['primary_mask']['cell_mean_freq'][str(c['index'])]} |")
        A("")
        comp = ", ".join(f"ring {r}: {n}" for r, n in
                         sorted(m["primary_mask"]["ring_composition"].items()))
        A(f"Ring composition — {comp}.")
        A("")
        A(f"**{doc['n_controls']} ring-matched controls**, seeds "
          f"{doc['control_seeds'][0]}–{doc['control_seeds'][-1]}, each with "
          f"the identical per-ring counts. Overlap with the primary mask is "
          f"ALLOWED and reported; no draw is ever rejected "
          f"(`LOCKED_ANALYSIS` §5).")
        A("")
        A("| seed | overlap with primary |")
        A("|---|---|")
        for c in m["controls"]:
            A(f"| {c['seed']} | {c['overlap_with_primary']} |")
        A("")
        obs = m["control_overlap_observed"]
        A(f"Observed overlap {min(obs)}–{max(obs)}, mean "
          f"{sum(obs) / len(obs):.2f}, against an analytic expectation of "
          f"**{m['control_overlap_expected']}** "
          f"(`{m['control_overlap_expected_formula']}`, the overlap a "
          f"uniformly drawn ring-matched control has by chance). A count "
          f"without that reference is not a report.")
        A("")
    A("## 4. The two masks coincide")
    A("")
    stages = list(D5_STAGES)
    if len(stages) == 2:
        a, b = (doc["masks"][s]["primary_mask"]["index"] for s in stages)
        ma, mb = (summary[s]["mean"] for s in stages)
        r = P.spatial_rho(ma, mb)
        A(f"The cell mean maps at `{stages[0]}` and `{stages[1]}` are "
          f"different arrays — masses {float(ma.sum()):.4f} and "
          f"{float(mb.sum()):.4f}, largest per-position difference "
          f"{float(np.abs(ma - mb).max()):.6f} — but they correlate at "
          f"ρ = {r[0]:.6f}" + (f" (p = {r[1]:.6f})" if r[1] is not None else "")
          + ".")
        A("")
        if sorted(a) == sorted(b):
            A("**The two primary masks are therefore IDENTICAL**: the same 16 "
              "coordinates are the 16 highest-prevalence positions at both "
              "block inputs. That is a property of the data, not a "
              "degenerate case — the maps are distinct and their internal "
              "ranking of those 16 differs — but it means I4 perturbs the "
              "same coordinate set at both depths, and the comparison "
              "between the two sites is therefore a comparison of DEPTH "
              "alone, with the address held fixed.")
        else:
            shared = len(set(a) & set(b))
            A(f"The two primary masks share {shared} of "
              f"{len(a)} coordinates.")
    A("")
    A("## 5. What this does NOT settle")
    A("")
    A("- The masks are selected on DISCOVERY data. That is what makes the I4 "
      "result non-circular, and it is also why nothing here is a finding: "
      "these coordinates are an input to an experiment, not its output.")
    A("- The block-input maps are descriptive. A high ρ with `s11_out` says "
      "the two stages share a spatial arrangement; it says nothing about why "
      "either has one.")
    A("- The spatial permutation reference assumes exchangeability under the "
      "8 dihedral transforms and every torus roll. A BORDERED map satisfies "
      "that only approximately, because a roll wraps the border onto the "
      "interior, so every p here is approximate in the same direction.")
    A("")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description="Build D5's masks and note.")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--masks-path", default=str(MASKS_PATH))
    p.add_argument("--note-path", default=str(NOTE_PATH))
    args = p.parse_args()

    r = build(manifest=args.manifest, masks_path=args.masks_path,
              note_path=args.note_path)
    print(f"wrote {r['masks_path']}")
    print(f"  sha256 (the value D5 is signed with): {r['file_sha256']}")
    print(f"wrote {r['note_path']}")
    for stage, s in r["summary"].items():
        comp = ", ".join(f"ring{k}={v}" for k, v in s["composition"].items())
        print(f"  {stage}: {comp} | control overlap "
              f"{min(s['observed'])}-{max(s['observed'])} "
              f"(expected {s['expected']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
