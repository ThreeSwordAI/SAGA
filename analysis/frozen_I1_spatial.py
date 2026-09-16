#!/usr/bin/env python3
"""
analysis/frozen_I1_spatial.py
=============================
TASK A / I1 A5 — the spatial-structure tables.

    python analysis/frozen_I1_spatial.py --split evaluation

CONSUMES I2, DOES NOT RECOMPUTE IT. Every map at `s11_out`, `hist` and
`hist/term_1.00` is loaded from `results/frozen/I2_terminal/<split>/<run_id>/
maps.npz` through `saga.frozen.prevalence.from_i2_maps_npz`. The maps at
`in_b07` / `in_b08`, and the per-image indicators at every stage, come from
`results/frozen/I1_spatial/<split>/<run_id>/maps_<stage>.npz` once Phase B
has run; the same table code emits more rows and nothing else changes.

WHAT IS AND IS NOT AVAILABLE BEFORE PHASE B, and why the tables say so
rather than omitting a column:

  available   every statistic derivable from a FREQUENCY MAP — the ring
              profile, the concentration excess against its finite-sample
              reference, the spatial Spearman against the 1568-transform
              permutation reference, eta^2_pos, and every difference of
              those between two maps.

  PENDING B   every statistic that needs a per-IMAGE indicator matrix:
              the image-level bootstrap CIs (LOCKED_ANALYSIS 8, D6 seed 0),
              the split-half reliability, and the whole scale-only null
              (T_I1b), which rescales a NORM FIELD and re-thresholds it.
              I2 stored per-position COUNTS summed over images, and a sum
              cannot be un-summed. Those cells carry the literal string
              `PENDING B`, which is never averaged and never guessed at.

THREE NORMALISATIONS OF ONE MAP, and the arithmetic relation between them,
because it is easy to read the second as new information:

    absolute     f(p)            = P(position p exceeds)
    conditional  f(p) / p_any    = P(p exceeds | the image has at least one)
    share        f(p) / sum_q f(q)

`p exceeds` implies `the image has at least one`, so the conditional map is
the absolute map divided by the scalar `p_any`, and the share map is it
divided by the scalar mass. ALL THREE ARE PROPORTIONAL: they have identical
rankings, identical Spearman correlations and identical ring PROFILE SHAPES.
What differs is the scale a reader sees, so the tables report the scalars
(`p_any`, `mass`) and the ring-1 value under each normalisation, and the
figure data carries the maps themselves.

DESCRIPTIVE WORDING ONLY (TASK A 12). A correlation between two runs' maps
is a "residual map correlation" and nothing else — not "relocation", not
"emptying", not a "sink function". What the number licenses is a statement
about two maps.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.address_analysis import (NULL_SEED, NULL_SIMS,  # noqa: E402
                                       concentration_null,
                                       concentration_recompute,
                                       position_rel_se)
from saga.frozen import prevalence as P  # noqa: E402
from saga.frozen.runner import eligible_cohort  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"
PENDING = "PENDING B"

#: Where I2's committed maps live, and where I1's land.
I2_ROOT = Path("results/frozen/I2_terminal")
I1_ROOT = Path("results/frozen/I1_spatial")
TABLE_DIR = I1_ROOT / "tables"

#: The stages I2 already measured, and the ones I1 adds.
I2_STAGES = ("s11_out", "hist")
I1_ONLY_STAGES = ("in_b07", "in_b08")
ALL_STAGES = I2_STAGES + I1_ONLY_STAGES

BASES = ("fixed_cal", "mad")

#: The I0 provenance columns every table carries (I0 handoff 8.3).
PROVENANCE = ["work_package", "split_name", "split_sha256", "git_sha",
              "generated_by"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_maps(split_name: str, cohort, *, manifest_rows=None) -> dict:
    """`{(run_id, condition, stage, basis): PrevalenceMap}` for one split.

    I2's `maps.npz` first, then I1's `maps_<stage>.npz` where it exists. A
    stage present in both is loaded from BOTH and the two are cross-checked
    by `agreement_rows` rather than one silently winning.
    """
    out, sources = {}, {}
    for row in cohort:
        run_id = row["run_id"]
        n_prefix = int(row.get("n_prefix", 1))
        i2 = I2_ROOT / split_name / run_id / "maps.npz"
        if i2.exists():
            with np.load(i2, allow_pickle=False) as z:
                triples = P.i2_map_keys(z)
            for cond, stage, basis in triples:
                if stage not in I2_STAGES or basis not in BASES:
                    continue                       # s12_post_norm: zero mass
                if cond != "native" and not (cond == "term_1.00"
                                             and stage == "hist"):
                    continue
                key = (run_id, cond, stage, basis)
                out[key] = P.from_i2_maps_npz(
                    i2, condition_id=cond, stage=stage, basis=basis,
                    n_prefix=n_prefix)
                sources[key] = str(i2)
        for stage in ALL_STAGES:
            i1 = I1_ROOT / split_name / run_id / f"maps_{stage}.npz"
            if not i1.exists():
                continue
            for cond, basis, pm in _read_i1_maps(i1, n_prefix):
                key = (run_id, cond, stage, basis)
                if key in out:
                    key = (run_id, cond, stage, basis, "I1")
                out[key] = pm
                sources[key] = str(i1)
    return out, sources


def _read_i1_maps(path, n_prefix):
    """Every `(condition, basis, PrevalenceMap)` in one `maps_<stage>.npz`."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        keys = sorted(k[len("counts__"):] for k in z.files
                      if k.startswith("counts__"))
        for key in keys:
            cond, basis = key.split("|")
            tau = (meta.get("tau_cal", {}) or {}).get(meta["stage"], MISSING)
            yield cond, basis, P.from_counts(
                z[f"counts__{key}"], n_images=int(z[f"n_images__{key}"]),
                basis=basis,
                threshold=(float(tau) if basis == "fixed_cal"
                           and isinstance(tau, (int, float)) else MISSING),
                stage=meta["stage"], split_sha=meta["split_sha256"],
                split_name=meta.get("split_name", MISSING), condition_id=cond,
                run_id=meta["run_id"], ckpt_sha256=meta["ckpt_sha256"],
                n_prefix=n_prefix, git_sha=meta["git_sha"])


def load_p_any(split_name: str) -> dict:
    """`{(run_id, condition, stage, basis): fraction of images with >= 1
    exceedance}` from I2's committed per-image primary pack.

    The conditional map needs this scalar and nothing else (see the module
    docstring). It is the ONLY image-level quantity recoverable before Phase
    B: the pack carries each image's exceedance COUNT but not WHICH
    positions, so it gives `p_any` and no indicator matrix.
    """
    pack = Path("figures_data/frozen") / f"I2_{split_name}_primary.npz"
    if not pack.exists():
        return {}
    out = {}
    with np.load(pack, allow_pickle=False) as z:
        for key in z.files:
            parts = key.split("|")
            if len(parts) != 4:
                continue
            run_id, cond, stage, metric = parts
            basis = {"count_fixed_cal": "fixed_cal",
                     "count_mad": "mad"}.get(metric)
            if basis is None:
                continue
            counts = np.asarray(z[key], dtype=np.float64)
            out[(run_id, cond, stage, basis)] = float((counts > 0).mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Row helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num(v, nd=6):
    """A float rounded for the CSV, or the literal MISSING / PENDING B."""
    if v is None:
        return MISSING
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return MISSING if not np.isfinite(v) else round(v, nd)


def base_columns(head: dict, pm, row) -> dict:
    """The provenance every row of every table carries."""
    return dict(
        head,
        run_id=pm.run_id, arch=row["arch"], recipe_actual=row["recipe_actual"],
        cell=f"{row['arch']}|{row['recipe_actual']}", variant=row["variant"],
        provenance_tag=row["provenance_tag"], ckpt_sha256=pm.ckpt_sha256,
        condition_id=pm.condition_id, stage=pm.stage, basis=pm.basis,
        n_images=pm.n_images, n_positions=pm.n_positions,
        grid_side=pm.grid_side, n_prefix=pm.n_prefix,
        tau=_num(pm.threshold) if isinstance(pm.threshold, float) else MISSING)


def write_table(path: Path, rows, fields) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    return path


def correlation_cells(a, b, prefix="") -> dict:
    """rho, its exact permutation p, the null sd and the transform count.

    The reference and its assumption travel with the number: exchangeability
    under the 8 dihedral transforms x every torus roll, which a BORDERED map
    satisfies only approximately because a roll wraps the border onto the
    interior. Stated here, in the note column, and in the handoff.
    """
    res = P.spatial_rho(a, b)
    if res is None:
        return {f"{prefix}rho": MISSING, f"{prefix}p_spatial": MISSING,
                f"{prefix}null_sd": MISSING, f"{prefix}n_transforms": 0}
    rho, p, sd, n = res
    return {f"{prefix}rho": _num(rho), f"{prefix}p_spatial": _num(p),
            f"{prefix}null_sd": _num(sd), f"{prefix}n_transforms": int(n)}


def concentration_cells(pm) -> dict:
    conc = P.concentration_with_reference(pm, sims=NULL_SIMS, seed=NULL_SEED)
    return {
        "mass": _num(conc["total_mass"]),
        "entropy_normalized": _num(conc["entropy_normalized"]),
        "entropy_normalized_excess": _num(conc["entropy_normalized_excess"]),
        "gini": _num(conc["gini"]),
        "gini_null": _num(conc["gini_null"]),
        "gini_excess": _num(conc["gini_excess"]),
        "top5_share": _num(conc["top5_share"]),
        "top5_share_uniform": _num(5.0 / pm.n_positions),
        "top5_share_excess": _num(conc["top5_share_excess"]),
    }


def ring_cells(pm) -> dict:
    """The 7-ring profile of a 14x14 grid, and the summary numbers over it.

    The column set is fixed at 7 rings so a 14x14 map (7 rings) and a 16x16
    map (8) share one header; a grid with fewer rings leaves the extra
    columns MISSING rather than shifting the others along. `centre_freq` is
    the INNERMOST ring of whatever grid this is.
    """
    prof = P.ring_profile(pm)
    out = {f"ring{k}_freq": (_num(prof[k]) if k < len(prof) else MISSING)
           for k in range(7)}
    out.update(
        ring0_share=_num(P.ring_share(pm, 0)),
        ring1_share=_num(P.ring_share(pm, 1)),
        peak_ring=int(np.argmax(prof)),
        centre_freq=_num(prof[-1]),
        ring1_minus_ring0=(_num(prof[1] - prof[0]) if len(prof) > 1
                           else MISSING),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# T_I1a — prevalence, per run x stage x basis
# ─────────────────────────────────────────────────────────────────────────────

T_I1A_FIELDS = (PROVENANCE + [
    "run_id", "arch", "recipe_actual", "cell", "variant", "provenance_tag",
    "ckpt_sha256", "condition_id", "stage", "basis", "n_images",
    "n_positions", "grid_side", "n_prefix", "tau", "source",
    "mass", "count_per_image", "p_any",
    "entropy_normalized", "entropy_normalized_excess",
    "gini", "gini_null", "gini_excess",
    "top5_share", "top5_share_uniform", "top5_share_excess",
    "eta2_pos", "position_rel_se"]
    + [f"ring{k}_freq" for k in range(7)]
    + ["ring0_share", "ring1_share", "peak_ring", "centre_freq",
       "ring1_minus_ring0",
       "ring1_freq_conditional", "ring1_freq_share",
       "split_half_rho", "split_half_p", "note"])


def table_I1a(maps, sources, rows_by_run, head, p_any) -> list:
    rows = []
    for key in sorted(maps, key=lambda k: (k[0], k[2], k[3], k[1])):
        pm = maps[key]
        run_id, cond, stage, basis = key[:4]
        row = rows_by_run[run_id]
        pa = p_any.get((run_id, cond, stage, basis))
        prof = P.ring_profile(pm)
        rec = base_columns(head, pm, row)
        rec.update(source=sources[key])
        rec.update(concentration_cells(pm))
        rec.update(ring_cells(pm))
        rec.update(
            count_per_image=_num(pm.mass),
            p_any=_num(pa) if pa is not None else PENDING,
            eta2_pos=_num(P.eta2_positional(pm)),
            position_rel_se=_num(position_rel_se(pm.mass, pm.n_images,
                                                 pm.n_positions)),
            ring1_freq_conditional=(_num(prof[1] / pa)
                                    if pa not in (None, 0.0) and len(prof) > 1
                                    else PENDING),
            ring1_freq_share=_num(prof[1] / pm.mass) if pm.mass > 0 else MISSING,
            split_half_rho=PENDING, split_half_p=PENDING,
            note="conditional = absolute / p_any and share = absolute / mass; "
                 "all three are proportional, so rankings and Spearman "
                 "correlations are identical and only the scale differs")
        rows.append(rec)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1d — seed stability, all baseline pairs per cell
# ─────────────────────────────────────────────────────────────────────────────

PAIR_FIELDS = (PROVENANCE + [
    "cell", "arch", "recipe_actual", "variant", "stage", "basis",
    "condition_id", "subject", "comparator", "n_positions", "n_images",
    "rho", "p_spatial", "null_sd", "n_transforms",
    "resid_rho", "resid_p_spatial", "resid_null_sd", "resid_n_transforms",
    "reference", "note"])


def _cell_of(row) -> str:
    return f"{row['arch']}|{row['recipe_actual']}"


def _by_cell(maps, rows_by_run, variant, condition="native"):
    """`{(cell, stage, basis): [(run_id, PrevalenceMap)]}` for one variant."""
    out = {}
    for key, pm in maps.items():
        run_id, cond, stage, basis = key[:4]
        if cond != condition:
            continue
        row = rows_by_run[run_id]
        if row["variant"] != variant:
            continue
        out.setdefault((_cell_of(row), stage, basis), []).append((run_id, pm))
    return {k: sorted(v) for k, v in out.items()}


def table_I1d(maps, rows_by_run, head) -> list:
    """Every baseline pair in a cell, plus the RING-ADJUSTED residual.

    A shared ring profile alone can carry a large rho between two maps that
    agree on nothing else, so each map's per-ring mean is subtracted and the
    residuals are correlated under the same permutation reference. The pair
    of numbers is the answer: rho says how much structure is shared, the
    residual rho says how much of it is not just "both are bordered".
    """
    rows = []
    groups = _by_cell(maps, rows_by_run, "baseline")
    for (cell, stage, basis), members in sorted(groups.items()):
        vals, resid_vals = [], []
        for (ra, pa), (rb, pb) in combinations(members, 2):
            rec = dict(head, cell=cell, arch=cell.split("|")[0],
                       recipe_actual=cell.split("|")[1], variant="baseline",
                       stage=stage, basis=basis, condition_id="native",
                       subject=ra, comparator=rb, n_positions=pa.n_positions,
                       n_images=pa.n_images,
                       reference="zero = no shared spatial structure",
                       note="residual map correlation over positions; "
                            "permutation reference assumes exchangeability "
                            "under 8 dihedral x all torus rolls, which a "
                            "bordered map satisfies only approximately")
            rec.update(correlation_cells(pa, pb))
            rec.update(correlation_cells(P.ring_adjusted(pa),
                                         P.ring_adjusted(pb), prefix="resid_"))
            rows.append(rec)
            if isinstance(rec["rho"], float):
                vals.append(rec["rho"])
            if isinstance(rec["resid_rho"], float):
                resid_vals.append(rec["resid_rho"])
        rows.append(dict(
            head, cell=cell, arch=cell.split("|")[0],
            recipe_actual=cell.split("|")[1], variant="baseline", stage=stage,
            basis=basis, condition_id="native", subject="MEAN",
            comparator="all-pairs", n_positions=members[0][1].n_positions,
            n_images=members[0][1].n_images,
            rho=_num(float(np.mean(vals))) if vals else MISSING,
            p_spatial=MISSING, null_sd=MISSING, n_transforms=len(vals),
            resid_rho=(_num(float(np.mean(resid_vals)))
                       if resid_vals else MISSING),
            resid_p_spatial=MISSING, resid_null_sd=MISSING,
            resid_n_transforms=len(resid_vals),
            reference="zero = no shared spatial structure",
            note=f"mean over {len(vals)} pair(s); n_transforms column holds "
                 f"the pair count on a MEAN row"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1c — the recipe contrast
# ─────────────────────────────────────────────────────────────────────────────

RECIPE_FIELDS = (PROVENANCE + [
    "question", "arch", "recipe_group", "stage", "basis", "subject",
    "comparator", "statistic", "value", "ci_lo", "ci_hi", "n",
    "reference", "note"])


def _recipe_row(head, **kw):
    base = dict(head, question="", arch="", recipe_group="", stage="",
                basis="", subject="", comparator="", statistic="", value="",
                ci_lo=PENDING, ci_hi=PENDING, n=0, reference="", note="")
    base.update(kw)
    return base


def table_I1c(maps, rows_by_run, head) -> list:
    """mixup vs true-nomix baselines, on identical images.

    The "object-centric framing" explanation says the border structure comes
    from what the training recipe does to the images. MixUp and true-nomix
    baselines of the same architecture, evaluated on the SAME split, are the
    direct test: if the ring profile and the map agree across the two
    recipes, the recipe is not what puts the mass where it is.

    `recipe_actual` and never a directory name — `legacy_e2_vit_small_
    nomixdir_baseline` is a MIXUP run (results/notes/recipe_erratum.md).
    """
    rows = []
    groups = _by_cell(maps, rows_by_run, "baseline")
    for arch in sorted({r["arch"] for r in rows_by_run.values()}):
        for stage in sorted({k[1] for k in groups}):
            for basis in BASES:
                mix = groups.get((f"{arch}|mixup", stage, basis), [])
                nomix = groups.get((f"{arch}|nomix", stage, basis), [])
                if not mix or not nomix:
                    continue
                within = [P.spatial_rho(a, b)
                          for (_, a), (_, b) in combinations(mix, 2)]
                cross = [(ra, rb, P.spatial_rho(a, b))
                         for ra, a in mix for rb, b in nomix]
                for ra, rb, res in cross:
                    rho, p, sd, n = res
                    rows.append(_recipe_row(
                        head, question="cross_recipe_map_correlation",
                        arch=arch, recipe_group="mixup|nomix", stage=stage,
                        basis=basis, subject=ra, comparator=rb,
                        statistic="rho", value=_num(rho), ci_lo=MISSING,
                        ci_hi=MISSING, n=n,
                        reference=f"within-recipe mean rho="
                                  f"{np.mean([r[0] for r in within]):.4f}",
                        note=f"residual map correlation; exact permutation "
                             f"p={p:.4f}, null sd={sd:.4f}"))
                rows.append(_recipe_row(
                    head, question="cross_recipe_map_correlation", arch=arch,
                    recipe_group="mixup|nomix", stage=stage, basis=basis,
                    subject="MEAN", comparator="all-cross-pairs",
                    statistic="rho_mean",
                    value=_num(float(np.mean([r[2][0] for r in cross]))),
                    ci_lo=MISSING, ci_hi=MISSING, n=len(cross),
                    reference="within-recipe mean is the comparator",
                    note="mean over all cross-recipe baseline pairs"))
                rows.append(_recipe_row(
                    head, question="within_recipe_map_correlation", arch=arch,
                    recipe_group="mixup", stage=stage, basis=basis,
                    subject="MEAN", comparator="all-within-pairs",
                    statistic="rho_mean",
                    value=(_num(float(np.mean([r[0] for r in within])))
                           if within else MISSING),
                    ci_lo=MISSING, ci_hi=MISSING, n=len(within),
                    reference="zero = no shared spatial structure",
                    note="mean over all within-recipe mixup baseline pairs"))

                # ring profiles per recipe group, with the finite-sample band
                for group, members in (("mixup", mix), ("nomix", nomix)):
                    for run_id, pm in members:
                        prof = P.ring_profile(pm)
                        for k, v in enumerate(prof):
                            rows.append(_recipe_row(
                                head, question="ring_profile", arch=arch,
                                recipe_group=group, stage=stage, basis=basis,
                                subject=run_id, comparator="border-ring",
                                statistic=f"ring{k}_freq", value=_num(v),
                                ci_lo=PENDING, ci_hi=PENDING,
                                n=pm.n_images,
                                reference=f"uniform={pm.mass / pm.n_positions:.6f} "
                                          f"at this map's own mass",
                                note="image-level bootstrap CI needs the "
                                     "per-image indicators — PENDING B"))
                        rows.append(_recipe_row(
                            head, question="ring_ordering", arch=arch,
                            recipe_group=group, stage=stage, basis=basis,
                            subject=run_id, comparator="ring0_vs_ring1",
                            statistic="ring1_minus_ring0",
                            value=_num(prof[1] - prof[0]), n=pm.n_images,
                            reference=">0 = ring 1 carries more mass per "
                                      "position than ring 0",
                            note="point estimate; image-level bootstrap CI "
                                 "PENDING B"))
                        conc = P.concentration_with_reference(pm)
                        rows.append(_recipe_row(
                            head, question="concentration_excess", arch=arch,
                            recipe_group=group, stage=stage, basis=basis,
                            subject=run_id, comparator="finite-sample-null",
                            statistic="gini_excess",
                            value=_num(conc["gini_excess"]),
                            ci_lo=MISSING, ci_hi=MISSING, n=pm.n_positions,
                            reference=f"null gini={conc['gini_null']:.6f} at "
                                      f"mass={pm.mass:.4f}, "
                                      f"{NULL_SIMS} sims, seed {NULL_SEED}",
                            note="observed minus the finite-sample null — the "
                                 "only cross-member comparable form"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1g — variant maps against the same-provenance baseline
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_FIELDS = (PROVENANCE + [
    "cell", "arch", "recipe_actual", "variant", "provenance_tag", "stage",
    "basis", "condition_id", "subject", "comparator", "n_images",
    "n_positions", "rho", "p_spatial", "null_sd", "n_transforms",
    "resid_rho", "resid_p_spatial", "resid_null_sd", "resid_n_transforms",
    "delta_gini_excess", "delta_entropy_normalized_excess",
    "delta_ring1_share", "delta_mass", "baseline_run_id", "reference", "note"])


def table_I1g(maps, rows_by_run, head) -> list:
    """SAGA and registers against the BASELINE OF THE SAME PROVENANCE.

    Paired by provenance tag for both the correlation and the concentration
    deltas, exactly as `analysis/address_analysis.py` Q4 does, and the
    concentration deltas are differences of EXCESS values: the raw
    statistics sit on a mass-dependent floor and the variants change the
    mass by up to two orders of magnitude, so differencing raw values
    inverts the sign of the registers result.

    Descriptive wording only: this is a residual map correlation between two
    checkpoints, and the table says nothing about what either one does.
    """
    rows = []
    baselines = {}
    for key, pm in maps.items():
        run_id, cond, stage, basis = key[:4]
        row = rows_by_run[run_id]
        if row["variant"] == "baseline" and cond == "native":
            baselines[(_cell_of(row), row["provenance_tag"], stage, basis)] = \
                (run_id, pm)

    for key in sorted(maps, key=lambda k: (k[2], k[3], k[0], k[1])):
        pm = maps[key]
        run_id, cond, stage, basis = key[:4]
        row = rows_by_run[run_id]
        if row["variant"] == "baseline":
            continue
        cell, tag = _cell_of(row), row["provenance_tag"]
        pair = baselines.get((cell, tag, stage, basis))
        rec = dict(head, cell=cell, arch=row["arch"],
                   recipe_actual=row["recipe_actual"], variant=row["variant"],
                   provenance_tag=tag, stage=stage, basis=basis,
                   condition_id=cond, subject=run_id,
                   comparator="baseline-matched", n_images=pm.n_images,
                   n_positions=pm.n_positions,
                   reference="1.0 = identical arrangement, 0 = none shared",
                   note="residual map correlation against the "
                        "same-provenance baseline; concentration deltas are "
                        "differences of EXCESS over the finite-sample null")
        if pair is None:
            rec.update({k: MISSING for k in (
                "rho", "p_spatial", "null_sd", "resid_rho", "resid_p_spatial",
                "resid_null_sd", "delta_gini_excess",
                "delta_entropy_normalized_excess", "delta_ring1_share",
                "delta_mass")},
                n_transforms=0, resid_n_transforms=0,
                baseline_run_id=MISSING,
                note="no same-provenance baseline map — EXCLUDED, never "
                     "folded into a cell mean")
            rows.append(rec)
            continue
        base_run, bpm = pair
        rec.update(correlation_cells(pm, bpm))
        rec.update(correlation_cells(P.ring_adjusted(pm), P.ring_adjusted(bpm),
                                     prefix="resid_"))
        vc = P.concentration_with_reference(pm)
        bc = P.concentration_with_reference(bpm)
        rec.update(
            baseline_run_id=base_run,
            delta_gini_excess=_num(vc["gini_excess"] - bc["gini_excess"]),
            delta_entropy_normalized_excess=_num(
                vc["entropy_normalized_excess"]
                - bc["entropy_normalized_excess"]),
            delta_ring1_share=_num(P.ring_share(pm, 1) - P.ring_share(bpm, 1)),
            delta_mass=_num(pm.mass - bpm.mass))
        rows.append(rec)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1e — eta^2, and the two tables Phase B owns
# ─────────────────────────────────────────────────────────────────────────────

ETA2_FIELDS = (PROVENANCE + [
    "run_id", "arch", "recipe_actual", "cell", "variant", "provenance_tag",
    "ckpt_sha256", "condition_id", "stage", "basis", "n_images",
    "n_positions", "mass", "f_bar", "eta2_pos", "ci_lo", "ci_hi",
    "bootstrap_seed", "bootstrap_resamples", "status", "note"])


def table_I1e(maps, rows_by_run, head, *, have_indicators: bool) -> list:
    """eta^2_pos = Var_P f(P) / (f_bar (1 - f_bar)).

    The POINT ESTIMATE is a function of the frequency map alone, so it is
    computed here from I2's committed maps. The CI is an image-level
    bootstrap (LOCKED_ANALYSIS 8; D6 = seed 0) and needs the per-image
    indicator matrix, which I2 did not store — a per-position count summed
    over images cannot be resampled by image. Those two columns are
    `PENDING B` until `maps_<stage>.npz` lands.
    """
    rows = []
    for key in sorted(maps, key=lambda k: (k[0], k[2], k[3], k[1])):
        pm = maps[key]
        run_id, cond, stage, basis = key[:4]
        row = rows_by_run[run_id]
        rows.append(dict(
            head, run_id=run_id, arch=row["arch"],
            recipe_actual=row["recipe_actual"], cell=_cell_of(row),
            variant=row["variant"], provenance_tag=row["provenance_tag"],
            ckpt_sha256=pm.ckpt_sha256, condition_id=cond, stage=stage,
            basis=basis, n_images=pm.n_images, n_positions=pm.n_positions,
            mass=_num(pm.mass), f_bar=_num(float(pm.freq.mean())),
            eta2_pos=_num(P.eta2_positional(pm)),
            ci_lo=PENDING, ci_hi=PENDING, bootstrap_seed=0,
            bootstrap_resamples=10000,
            status="point estimate only" if not have_indicators else "complete",
            note="Var over the 196 positions (population, ddof=0) divided by "
                 "f_bar(1-f_bar); the image-level bootstrap CI needs the "
                 "per-image indicators"))
    return rows


PENDING_FIELDS = (PROVENANCE
                  + ["table", "status", "needs", "produced_by", "note"])


def pending_table(head, table: str, needs: str, note: str) -> list:
    return [dict(head, table=table, status=PENDING, needs=needs,
                 produced_by="Phase C, after scripts/jobs/frozen_I1.sbatch",
                 note=note)]


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def build(split_name: str, *, manifest="results/frozen/I0_manifest/manifest.json",
          out_dir=None) -> dict:
    cohort = eligible_cohort(manifest)
    rows_by_run = {r["run_id"]: r for r in cohort}
    maps, sources = load_maps(split_name, cohort)
    if not maps:
        raise SystemExit(
            f"no maps found for split {split_name!r} under {I2_ROOT} or "
            f"{I1_ROOT} — nothing to tabulate")
    p_any = load_p_any(split_name)
    split_sha = sorted({pm.split_sha for pm in maps.values()})
    if len(split_sha) != 1:
        raise SystemExit(
            f"maps from {len(split_sha)} different splits in one table: "
            f"{split_sha}")
    head = {"work_package": "I1_spatial", "split_name": split_name,
            "split_sha256": split_sha[0], "git_sha": git_sha(),
            "generated_by": "analysis/frozen_I1_spatial.py"}
    have_indicators = any(
        (I1_ROOT / split_name).glob("*/maps_*.npz"))
    out_dir = Path(out_dir) if out_dir else TABLE_DIR

    written = {}
    written["T_I1a_prevalence"] = write_table(
        out_dir / "T_I1a_prevalence.csv",
        table_I1a(maps, sources, rows_by_run, head, p_any), T_I1A_FIELDS)
    written["T_I1b_scale_null"] = write_table(
        out_dir / "T_I1b_scale_null.csv",
        pending_table(head, "T_I1b_scale_null",
                      "results/frozen/I1_spatial/<split>/<run>/norms_*.npz",
                      "the scale-only null rescales a NORM FIELD by a scalar "
                      "and re-thresholds it; I2 stored per-position counts, "
                      "which cannot be rescaled"), PENDING_FIELDS)
    written["T_I1c_recipe"] = write_table(
        out_dir / "T_I1c_recipe.csv",
        table_I1c(maps, rows_by_run, head), RECIPE_FIELDS)
    written["T_I1d_seed_stability"] = write_table(
        out_dir / "T_I1d_seed_stability.csv",
        table_I1d(maps, rows_by_run, head), PAIR_FIELDS)
    written["T_I1e_eta2"] = write_table(
        out_dir / "T_I1e_eta2.csv",
        table_I1e(maps, rows_by_run, head, have_indicators=have_indicators),
        ETA2_FIELDS)
    written["T_I1f_blockinput"] = write_table(
        out_dir / "T_I1f_blockinput.csv",
        pending_table(head, "T_I1f_blockinput",
                      "results/frozen/I1_spatial/<split>/<run>/"
                      "maps_in_b07.npz and maps_in_b08.npz",
                      "no committed artifact contains a block-input map; "
                      "every *_addr.json in the repository is a last-block "
                      "map"), PENDING_FIELDS)
    written["T_I1g_variant_maps"] = write_table(
        out_dir / "T_I1g_variant_maps.csv",
        table_I1g(maps, rows_by_run, head), VARIANT_FIELDS)
    return {"tables": written, "n_maps": len(maps), "head": head,
            "stages": sorted({k[2] for k in maps}),
            "have_indicators": bool(have_indicators)}


def main():
    p = argparse.ArgumentParser(description="TASK A / I1 spatial tables.")
    p.add_argument("--split", default="evaluation",
                   help="split NAME (evaluation for reporting; discovery is "
                        "selection material and is tabulated only as the "
                        "source of a mask)")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    result = build(args.split, manifest=args.manifest, out_dir=args.out_dir)
    print(f"split={args.split}  maps={result['n_maps']}  "
          f"stages={result['stages']}  "
          f"indicators={'yes' if result['have_indicators'] else 'PENDING B'}")
    for name, path in result["tables"].items():
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print(f"  wrote {path}  ({n} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
