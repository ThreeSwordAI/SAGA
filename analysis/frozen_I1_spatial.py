#!/usr/bin/env python3
"""
analysis/frozen_I1_spatial.py
=============================
TASK A / I1 A5 — the spatial-structure tables.

    python analysis/frozen_I1_spatial.py --split evaluation

CONSUMES I2, DOES NOT RECOMPUTE IT. The maps at `s11_out`, `hist` and
`hist/term_1.00` are loaded from `results/frozen/I2_terminal/<split>/
<run_id>/maps.npz`. The maps at `in_b07` / `in_b08`, and the per-image
INDICATORS at every stage, come from `results/frozen/I1_spatial/<split>/
<run_id>/maps_<stage>.npz`, which Phase B wrote.

I1 recaptured `s11_out` and `hist` in the same forward passes that produced
the block-input stages, so both sources hold those two stages. They are
loaded from BOTH and cross-checked (`T_I1h_agreement`) rather than one
silently winning: two pipelines that measured the same thing must agree, and
if they ever stop agreeing that is a result in itself.

WHAT THE SPLITS ARE FOR. `evaluation` is the reporting split and every table
here comes from it. `val_diag_split` — the discovery split, whose directory
is named from the file stem because that file predates
`tools/build_frozen_splits.py` and records no `name` — is SELECTION material
and is tabulated only as the source of a D5 mask, clearly labelled.

THREE NORMALISATIONS OF ONE MAP. `p exceeds` implies `the image has at least
one`, so the CONDITIONAL map is the absolute map divided by the scalar
`p_any` and the SHARE map is it divided by the scalar mass. All three are
proportional: identical rankings, identical Spearman correlations, identical
ring profile SHAPES. The tables report the two scalars rather than three
copies of one ranking. (On the mixup cells `p_any` is 1.000 — every image
has at least one exceedance — so there the conditional map IS the absolute
map.)

DESCRIPTIVE WORDING ONLY (TASK A §12). A correlation between two runs' maps
is a "residual map correlation" and nothing else.

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
from analysis import frozen_I1_stats as S  # noqa: E402
from analysis.address_analysis import (NULL_SEED, NULL_SIMS,  # noqa: E402
                                       concentration_null,
                                       concentration_recompute,
                                       position_rel_se)
from saga.frozen import norms as fnorms  # noqa: E402
from saga.frozen import prevalence as P  # noqa: E402
from saga.frozen.runner import eligible_cohort  # noqa: E402
from saga.run_registry import git_sha  # noqa: E402

MISSING = "MISSING"
PENDING = "PENDING B"

I2_ROOT = Path("results/frozen/I2_terminal")
I1_ROOT = Path("results/frozen/I1_spatial")
TABLE_DIR = I1_ROOT / "tables"

#: Stages I2 already measured, and the ones only I1 has.
I2_STAGES = ("s11_out", "hist")
BLOCK_INPUT_STAGES = ("in_b07", "in_b08")
ALL_STAGES = I2_STAGES + BLOCK_INPUT_STAGES
#: Reporting order: D1's stage first, then the two block inputs, then the
#: historical control.
STAGE_ORDER = {"s11_out": 0, "in_b07": 1, "in_b08": 2, "hist": 3}

BASES = ("fixed_cal", "mad")

#: The I0 provenance columns every table carries (I0 handoff §8.3).
PROVENANCE = ["work_package", "split_name", "split_sha256", "git_sha",
              "generated_by"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

class MapStore:
    """Every map for one split, plus lazy access to its indicators.

    The maps themselves are 196 floats each and are held; the indicator
    matrices are ~2 MB each and 336 of them will not fit comfortably, so
    they are read on demand and only their small summaries are cached.
    """

    def __init__(self, split_name, cohort, *, i1_root=I1_ROOT, i2_root=I2_ROOT,
                 seed=S.BOOTSTRAP_SEED, resamples=S.BOOTSTRAP_RESAMPLES):
        self.split_name = split_name
        self.rows = {r["run_id"]: r for r in cohort}
        self.maps, self.source, self.ind_source, self.i2_maps = {}, {}, {}, {}
        self._stats = {}
        self.seed, self.resamples = seed, resamples

        for row in cohort:
            run_id, n_prefix = row["run_id"], int(row.get("n_prefix", 1))
            i2 = Path(i2_root) / split_name / run_id / "maps.npz"
            if i2.exists():
                with np.load(i2, allow_pickle=False) as z:
                    triples = P.i2_map_keys(z)
                for cond, stage, basis in triples:
                    if stage not in I2_STAGES or basis not in BASES:
                        continue                # s12_post_norm: zero mass
                    if cond != "native" and not (cond == "term_1.00"
                                                 and stage == "hist"):
                        continue
                    self.i2_maps[(run_id, cond, stage, basis)] = \
                        P.from_i2_maps_npz(i2, condition_id=cond, stage=stage,
                                           basis=basis, n_prefix=n_prefix)
            for stage in ALL_STAGES:
                i1 = Path(i1_root) / split_name / run_id / f"maps_{stage}.npz"
                if not i1.exists():
                    continue
                for cond, basis, pm in _read_i1_maps(i1, n_prefix):
                    key = (run_id, cond, stage, basis)
                    self.maps[key] = pm
                    self.source[key] = str(i1)
                    self.ind_source[key] = (i1, cond, basis)
        # I1 is the primary source; fall back to I2 where I1 did not run
        for key, pm in self.i2_maps.items():
            self.maps.setdefault(key, pm)
            self.source.setdefault(key, str(
                Path(i2_root) / split_name / key[0] / "maps.npz"))

    def keys(self, *, variant=None, condition=None, stage=None, basis=None):
        out = []
        for key in self.maps:
            run_id, cond, stg, bas = key
            if condition is not None and cond != condition:
                continue
            if stage is not None and stg != stage:
                continue
            if basis is not None and bas != basis:
                continue
            if variant is not None and self.rows[run_id]["variant"] != variant:
                continue
            out.append(key)
        return sorted(out, key=lambda k: (STAGE_ORDER.get(k[2], 9), k[3],
                                          k[0], k[1]))

    def stats(self, key, *, with_rings=False, with_eta2=False):
        """Cached image-level statistics, or None when no indicators exist."""
        want = (bool(with_rings), bool(with_eta2))
        have = self._stats.get(key)
        if have is not None and all(a or not b
                                    for a, b in zip(have["_have"], want)):
            return have
        src = self.ind_source.get(key)
        if src is None:
            return None
        path, cond, basis = src
        ind = fnorms.read_indicators(path, cond, basis)
        st = S.image_stats(ind, self.maps[key].grid_side,
                           with_rings=with_rings, with_eta2=with_eta2,
                           seed=self.seed, resamples=self.resamples)
        st["_have"] = want
        self._stats[key] = st
        return st


def _read_i1_maps(path, n_prefix):
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        for key in sorted(k[len("counts__"):] for k in z.files
                          if k.startswith("counts__")):
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


# ─────────────────────────────────────────────────────────────────────────────
# Row helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num(v, nd=6):
    if v is None:
        return MISSING
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return MISSING if not np.isfinite(v) else round(v, nd)


def _ci(pair, which):
    if not pair or pair[0] is None:
        return MISSING
    return _num(pair[0] if which == "lo" else pair[1])


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
    interior.
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
        "gini": _num(conc["gini"]), "gini_null": _num(conc["gini_null"]),
        "gini_excess": _num(conc["gini_excess"]),
        "top5_share": _num(conc["top5_share"]),
        "top5_share_uniform": _num(5.0 / pm.n_positions),
        "top5_share_excess": _num(conc["top5_share_excess"]),
    }


def ring_cells(pm) -> dict:
    """The 7-ring profile of a 14x14 grid and the summaries over it.

    The column set is fixed at 7 so a 14x14 map and a 16x16 map share one
    header; a grid with fewer rings leaves the extra columns MISSING rather
    than shifting the others along. `centre_freq` is the INNERMOST ring.
    """
    prof = P.ring_profile(pm)
    out = {f"ring{k}_freq": (_num(prof[k]) if k < len(prof) else MISSING)
           for k in range(7)}
    out.update(ring0_share=_num(P.ring_share(pm, 0)),
               ring1_share=_num(P.ring_share(pm, 1)),
               peak_ring=int(np.argmax(prof)), centre_freq=_num(prof[-1]),
               ring1_minus_ring0=(_num(prof[1] - prof[0]) if len(prof) > 1
                                  else MISSING))
    return out


def base_columns(head, store, key) -> dict:
    pm = store.maps[key]
    row = store.rows[key[0]]
    return dict(head, run_id=pm.run_id, arch=row["arch"],
                recipe_actual=row["recipe_actual"],
                cell=f"{row['arch']}|{row['recipe_actual']}",
                variant=row["variant"], provenance_tag=row["provenance_tag"],
                ckpt_sha256=pm.ckpt_sha256, condition_id=pm.condition_id,
                stage=pm.stage, basis=pm.basis, n_images=pm.n_images,
                n_positions=pm.n_positions, grid_side=pm.grid_side,
                n_prefix=pm.n_prefix,
                tau=_num(pm.threshold) if isinstance(pm.threshold, float)
                else MISSING)


def _cell_of(row) -> str:
    return f"{row['arch']}|{row['recipe_actual']}"


# ─────────────────────────────────────────────────────────────────────────────
# T_I1a — prevalence
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
       "ring1_minus_ring0", "ring1_freq_conditional", "ring1_freq_share",
       "split_half_rho", "split_half_p", "split_half_n_transforms",
       "bootstrap_seed", "note"])


def table_I1a(store, head) -> list:
    rows = []
    for key in store.keys():
        pm = store.maps[key]
        st = store.stats(key)
        prof = P.ring_profile(pm)
        pa = st["p_any"] if st else None
        rec = base_columns(head, store, key)
        rec.update(source=store.source[key])
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
            ring1_freq_share=(_num(prof[1] / pm.mass) if pm.mass > 0
                              else MISSING),
            split_half_rho=_num(st["split_half_rho"]) if st else PENDING,
            split_half_p=_num(st["split_half_p"]) if st else PENDING,
            split_half_n_transforms=(st["split_half_n_transforms"] if st
                                     else 0),
            bootstrap_seed=S.BOOTSTRAP_SEED,
            note="conditional = absolute / p_any and share = absolute / mass; "
                 "all three are proportional, so rankings and Spearman "
                 "correlations are identical and only the scale differs")
        rows.append(rec)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1d — seed stability
# ─────────────────────────────────────────────────────────────────────────────

PAIR_FIELDS = (PROVENANCE + [
    "cell", "arch", "recipe_actual", "variant", "stage", "basis",
    "condition_id", "subject", "comparator", "n_positions", "n_images",
    "rho", "p_spatial", "null_sd", "n_transforms",
    "resid_rho", "resid_p_spatial", "resid_null_sd", "resid_n_transforms",
    "reference", "note"])


def table_I1d(store, head) -> list:
    """Every baseline pair in a cell, plus the RING-ADJUSTED residual.

    A shared ring profile alone can carry a large rho between two maps that
    agree on nothing else, so each map's per-ring mean is subtracted and the
    residuals correlated under the same reference. The pair of numbers is
    the answer: rho says how much structure is shared, the residual rho says
    how much of it is not just "both are bordered".
    """
    rows = []
    groups = {}
    for key in store.keys(variant="baseline", condition="native"):
        run_id, _, stage, basis = key
        groups.setdefault((_cell_of(store.rows[run_id]), stage, basis),
                          []).append(key)
    for (cell, stage, basis), members in sorted(
            groups.items(), key=lambda kv: (STAGE_ORDER.get(kv[0][1], 9),
                                            kv[0][2], kv[0][0])):
        vals, resid = [], []
        for ka, kb in combinations(sorted(members), 2):
            pa, pb = store.maps[ka], store.maps[kb]
            rec = dict(head, cell=cell, arch=cell.split("|")[0],
                       recipe_actual=cell.split("|")[1], variant="baseline",
                       stage=stage, basis=basis, condition_id="native",
                       subject=ka[0], comparator=kb[0],
                       n_positions=pa.n_positions, n_images=pa.n_images,
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
                resid.append(rec["resid_rho"])
        first = store.maps[sorted(members)[0]]
        rows.append(dict(
            head, cell=cell, arch=cell.split("|")[0],
            recipe_actual=cell.split("|")[1], variant="baseline", stage=stage,
            basis=basis, condition_id="native", subject="MEAN",
            comparator="all-pairs", n_positions=first.n_positions,
            n_images=first.n_images,
            rho=_num(float(np.mean(vals))) if vals else MISSING,
            p_spatial=MISSING, null_sd=MISSING, n_transforms=len(vals),
            resid_rho=_num(float(np.mean(resid))) if resid else MISSING,
            resid_p_spatial=MISSING, resid_null_sd=MISSING,
            resid_n_transforms=len(resid),
            reference="zero = no shared spatial structure",
            note=f"mean over {len(vals)} pair(s); n_transforms holds the pair "
                 f"count on a MEAN row"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1c — the recipe contrast
# ─────────────────────────────────────────────────────────────────────────────

RECIPE_FIELDS = (PROVENANCE + [
    "question", "arch", "recipe_group", "stage", "basis", "subject",
    "comparator", "statistic", "value", "ci_lo", "ci_hi", "n", "reference",
    "note"])


def _recipe_row(head, **kw):
    base = dict(head, question="", arch="", recipe_group="", stage="",
                basis="", subject="", comparator="", statistic="", value="",
                ci_lo=MISSING, ci_hi=MISSING, n=0, reference="", note="")
    base.update(kw)
    return base


def table_I1c(store, head) -> list:
    """mixup vs true-nomix baselines, on identical images.

    The "object-centric framing" explanation says the border structure comes
    from what the training recipe does to the images. MixUp and true-nomix
    baselines of the same architecture, evaluated on the SAME split, are the
    direct test: if the ring profile and the map agree across recipes, the
    recipe is not what puts the mass where it is.

    `recipe_actual`, never a directory name — `legacy_e2_vit_small_nomixdir_
    baseline` is a MIXUP run (results/notes/recipe_erratum.md).
    """
    rows = []
    by_cell = {}
    for key in store.keys(variant="baseline", condition="native"):
        by_cell.setdefault((_cell_of(store.rows[key[0]]), key[2], key[3]),
                           []).append(key)
    archs = sorted({r["arch"] for r in store.rows.values()})
    for stage in sorted({k[2] for k in store.maps},
                        key=lambda s: STAGE_ORDER.get(s, 9)):
        for basis in BASES:
            for arch in archs:
                mix = sorted(by_cell.get((f"{arch}|mixup", stage, basis), []))
                nomix = sorted(by_cell.get((f"{arch}|nomix", stage, basis), []))
                if not mix or not nomix:
                    continue
                within = [P.spatial_rho(store.maps[a], store.maps[b])
                          for a, b in combinations(mix, 2)]
                cross = [(a, b, P.spatial_rho(store.maps[a], store.maps[b]))
                         for a in mix for b in nomix]
                wmean = float(np.mean([r[0] for r in within])) if within else None
                for a, b, res in cross:
                    rho, p, sd, n = res
                    rows.append(_recipe_row(
                        head, question="cross_recipe_map_correlation",
                        arch=arch, recipe_group="mixup|nomix", stage=stage,
                        basis=basis, subject=a[0], comparator=b[0],
                        statistic="rho", value=_num(rho), n=n,
                        reference=(f"within-recipe mean rho={wmean:.4f}"
                                   if wmean is not None else "no within pair"),
                        note=f"residual map correlation; exact permutation "
                             f"p={p:.4f}, null sd={sd:.4f}"))
                rows.append(_recipe_row(
                    head, question="cross_recipe_map_correlation", arch=arch,
                    recipe_group="mixup|nomix", stage=stage, basis=basis,
                    subject="MEAN", comparator="all-cross-pairs",
                    statistic="rho_mean",
                    value=_num(float(np.mean([r[2][0] for r in cross]))),
                    n=len(cross),
                    reference="within-recipe mean is the comparator",
                    note="mean over all cross-recipe baseline pairs"))
                rows.append(_recipe_row(
                    head, question="within_recipe_map_correlation", arch=arch,
                    recipe_group="mixup", stage=stage, basis=basis,
                    subject="MEAN", comparator="all-within-pairs",
                    statistic="rho_mean",
                    value=_num(wmean) if wmean is not None else MISSING,
                    n=len(within),
                    reference="zero = no shared spatial structure",
                    note="mean over all within-recipe mixup baseline pairs"))

                for group, members in (("mixup", mix), ("nomix", nomix)):
                    for key in members:
                        pm = store.maps[key]
                        st = store.stats(key, with_rings=True)
                        prof = P.ring_profile(pm)
                        for k, v in enumerate(prof):
                            ci = (st or {}).get("ring_ci", {}).get(k)
                            rows.append(_recipe_row(
                                head, question="ring_profile", arch=arch,
                                recipe_group=group, stage=stage, basis=basis,
                                subject=key[0], comparator="border-ring",
                                statistic=f"ring{k}_freq", value=_num(v),
                                ci_lo=_ci(ci, "lo") if st else PENDING,
                                ci_hi=_ci(ci, "hi") if st else PENDING,
                                n=pm.n_images,
                                reference=f"uniform="
                                          f"{pm.mass / pm.n_positions:.6f} at "
                                          f"this map's own mass",
                                note="image-level bootstrap, 10,000 "
                                     "resamples, seed 0 (D6)"))
                        dci = (st or {}).get("ring1_minus_ring0_ci")
                        rows.append(_recipe_row(
                            head, question="ring_ordering", arch=arch,
                            recipe_group=group, stage=stage, basis=basis,
                            subject=key[0], comparator="ring0_vs_ring1",
                            statistic="ring1_minus_ring0",
                            value=_num(prof[1] - prof[0]),
                            ci_lo=_ci(dci, "lo") if st else PENDING,
                            ci_hi=_ci(dci, "hi") if st else PENDING,
                            n=pm.n_images,
                            reference=">0 = ring 1 carries more mass per "
                                      "position than ring 0",
                            note="the ring CIs and this contrast come from "
                                 "ONE resample of images, so they are "
                                 "mutually consistent"))
                        conc = P.concentration_with_reference(pm)
                        rows.append(_recipe_row(
                            head, question="concentration_excess", arch=arch,
                            recipe_group=group, stage=stage, basis=basis,
                            subject=key[0], comparator="finite-sample-null",
                            statistic="gini_excess",
                            value=_num(conc["gini_excess"]), n=pm.n_positions,
                            reference=f"null gini={conc['gini_null']:.6f} at "
                                      f"mass={pm.mass:.4f}, {NULL_SIMS} sims, "
                                      f"seed {NULL_SEED}",
                            note="observed minus the finite-sample null — the "
                                 "only cross-member comparable form"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1e — eta^2
# ─────────────────────────────────────────────────────────────────────────────

ETA2_FIELDS = (PROVENANCE + [
    "run_id", "arch", "recipe_actual", "cell", "variant", "provenance_tag",
    "ckpt_sha256", "condition_id", "stage", "basis", "n_images",
    "n_positions", "mass", "f_bar", "eta2_pos", "ci_lo", "ci_hi",
    "bootstrap_seed", "bootstrap_resamples", "status", "note"])


def table_I1e(store, head) -> list:
    """η²_pos = Var_P f(P) / (f̄(1 − f̄)), with an image-level bootstrap CI.

    The CI is computed for BASELINE maps, which is the scope the task file
    gives this table; other variants carry the point estimate and say so.
    Each bootstrap resample needs a whole frequency MAP, so it is the most
    expensive statistic here — about 3 s per map.
    """
    rows = []
    for key in store.keys():
        pm = store.maps[key]
        row = store.rows[key[0]]
        is_baseline = row["variant"] == "baseline"
        st = store.stats(key, with_eta2=is_baseline)
        ci = (st or {}).get("eta2_ci")
        rows.append(dict(
            head, run_id=key[0], arch=row["arch"],
            recipe_actual=row["recipe_actual"], cell=_cell_of(row),
            variant=row["variant"], provenance_tag=row["provenance_tag"],
            ckpt_sha256=pm.ckpt_sha256, condition_id=key[1], stage=key[2],
            basis=key[3], n_images=pm.n_images, n_positions=pm.n_positions,
            mass=_num(pm.mass), f_bar=_num(float(pm.freq.mean())),
            eta2_pos=_num(P.eta2_positional(pm)),
            ci_lo=_ci(ci, "lo"), ci_hi=_ci(ci, "hi"),
            bootstrap_seed=S.BOOTSTRAP_SEED,
            bootstrap_resamples=S.BOOTSTRAP_RESAMPLES,
            status=("complete" if ci and ci[0] is not None
                    else "point estimate only"),
            note="Var over the positions (population, ddof=0) divided by "
                 "f_bar(1-f_bar); CI resamples IMAGES. Computed for baseline "
                 "maps, the scope the task file gives this table"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1f — the block-input stages (D5's stages)
# ─────────────────────────────────────────────────────────────────────────────

BLOCKINPUT_FIELDS = (PROVENANCE + [
    "role", "cell", "arch", "recipe_actual", "variant", "provenance_tag",
    "run_id", "stage", "basis", "n_images", "n_positions", "mass",
    "gini_excess", "ring0_freq", "ring1_freq", "ring1_share", "peak_ring",
    "eta2_pos", "rho_vs_s11_out", "p_vs_s11_out",
    "resid_rho_vs_s11_out", "baseline_run_id", "rho_vs_baseline",
    "p_vs_baseline", "resid_rho_vs_baseline", "delta_gini_excess",
    "delta_ring1_share", "delta_mass", "reference", "note"])


def table_I1f(store, head, *, role) -> list:
    """`in_b07` / `in_b08`: the stages D5's masks are defined at.

    Two questions per map. First, is the block-input map the same address as
    the reporting stage — `rho_vs_s11_out` within the SAME run, which is what
    says whether the coordinates I4 perturbs are the ones the paper reports
    on. Second, the variant comparison at the block input, paired by
    provenance exactly as at `s11_out`.

    `role` labels the split: `reporting` for evaluation, `discovery/selection`
    for the mask source, so a reader can never mistake one for the other.
    """
    rows = []
    baselines = {}
    for key in store.keys(variant="baseline", condition="native"):
        baselines[(_cell_of(store.rows[key[0]]),
                   store.rows[key[0]]["provenance_tag"], key[2], key[3])] = key

    for stage in BLOCK_INPUT_STAGES:
        for key in store.keys(stage=stage, condition="native"):
            run_id, _, _, basis = key
            pm = store.maps[key]
            row = store.rows[run_id]
            cell, tag = _cell_of(row), row["provenance_tag"]
            prof = P.ring_profile(pm)
            conc = P.concentration_with_reference(pm)
            rec = dict(head, role=role, cell=cell, arch=row["arch"],
                       recipe_actual=row["recipe_actual"],
                       variant=row["variant"], provenance_tag=tag,
                       run_id=run_id, stage=stage, basis=basis,
                       n_images=pm.n_images, n_positions=pm.n_positions,
                       mass=_num(pm.mass),
                       gini_excess=_num(conc["gini_excess"]),
                       ring0_freq=_num(prof[0]), ring1_freq=_num(prof[1]),
                       ring1_share=_num(P.ring_share(pm, 1)),
                       peak_ring=int(np.argmax(prof)),
                       eta2_pos=_num(P.eta2_positional(pm)),
                       reference="1.0 = identical arrangement",
                       note="residual map correlation; rho_vs_s11_out is the "
                            "SAME run at the reporting stage, which is what "
                            "says whether the perturbed coordinates are the "
                            "address the paper reports")
            own = (run_id, "native", "s11_out", basis)
            if own in store.maps:
                c = correlation_cells(pm, store.maps[own])
                rec.update(rho_vs_s11_out=c["rho"], p_vs_s11_out=c["p_spatial"])
                rec["resid_rho_vs_s11_out"] = correlation_cells(
                    P.ring_adjusted(pm),
                    P.ring_adjusted(store.maps[own]))["rho"]
            else:
                rec.update(rho_vs_s11_out=MISSING, p_vs_s11_out=MISSING,
                           resid_rho_vs_s11_out=MISSING)

            bkey = baselines.get((cell, tag, stage, basis))
            if bkey is None or bkey == key:
                rec.update(baseline_run_id=MISSING, rho_vs_baseline=MISSING,
                           p_vs_baseline=MISSING,
                           resid_rho_vs_baseline=MISSING,
                           delta_gini_excess=MISSING,
                           delta_ring1_share=MISSING, delta_mass=MISSING)
            else:
                bpm = store.maps[bkey]
                c = correlation_cells(pm, bpm)
                bconc = P.concentration_with_reference(bpm)
                rec.update(
                    baseline_run_id=bkey[0], rho_vs_baseline=c["rho"],
                    p_vs_baseline=c["p_spatial"],
                    resid_rho_vs_baseline=correlation_cells(
                        P.ring_adjusted(pm), P.ring_adjusted(bpm))["rho"],
                    delta_gini_excess=_num(conc["gini_excess"]
                                           - bconc["gini_excess"]),
                    delta_ring1_share=_num(P.ring_share(pm, 1)
                                           - P.ring_share(bpm, 1)),
                    delta_mass=_num(pm.mass - bpm.mass))
            rows.append(rec)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1g — variant maps
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_FIELDS = (PROVENANCE + [
    "cell", "arch", "recipe_actual", "variant", "provenance_tag", "stage",
    "basis", "condition_id", "subject", "comparator", "n_images",
    "n_positions", "rho", "p_spatial", "null_sd", "n_transforms",
    "resid_rho", "resid_p_spatial", "resid_null_sd", "resid_n_transforms",
    "delta_gini_excess", "delta_entropy_normalized_excess",
    "delta_ring1_share", "delta_mass", "baseline_run_id", "reference", "note"])


def table_I1g(store, head) -> list:
    """SAGA and registers against the BASELINE OF THE SAME PROVENANCE.

    Paired by provenance tag for both the correlation and the concentration
    deltas, and the deltas are differences of EXCESS values: the raw
    statistics sit on a mass-dependent floor and the variants change the mass
    by up to two orders of magnitude, so differencing raw values inverts the
    sign of the registers result.

    Descriptive wording only: a residual map correlation between two
    checkpoints, saying nothing about what either one does.
    """
    rows = []
    baselines = {}
    for key in store.keys(variant="baseline", condition="native"):
        baselines[(_cell_of(store.rows[key[0]]),
                   store.rows[key[0]]["provenance_tag"], key[2], key[3])] = key

    for key in store.keys():
        run_id, cond, stage, basis = key
        row = store.rows[run_id]
        if row["variant"] == "baseline":
            continue
        pm = store.maps[key]
        cell, tag = _cell_of(row), row["provenance_tag"]
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
        bkey = baselines.get((cell, tag, stage, basis))
        if bkey is None:
            rec.update({k: MISSING for k in (
                "rho", "p_spatial", "null_sd", "resid_rho", "resid_p_spatial",
                "resid_null_sd", "delta_gini_excess",
                "delta_entropy_normalized_excess", "delta_ring1_share",
                "delta_mass")}, n_transforms=0, resid_n_transforms=0,
                baseline_run_id=MISSING,
                note="no same-provenance baseline map — EXCLUDED, never "
                     "folded into a cell mean")
            rows.append(rec)
            continue
        bpm = store.maps[bkey]
        rec.update(correlation_cells(pm, bpm))
        rec.update(correlation_cells(P.ring_adjusted(pm), P.ring_adjusted(bpm),
                                     prefix="resid_"))
        vc = P.concentration_with_reference(pm)
        bc = P.concentration_with_reference(bpm)
        rec.update(
            baseline_run_id=bkey[0],
            delta_gini_excess=_num(vc["gini_excess"] - bc["gini_excess"]),
            delta_entropy_normalized_excess=_num(
                vc["entropy_normalized_excess"]
                - bc["entropy_normalized_excess"]),
            delta_ring1_share=_num(P.ring_share(pm, 1) - P.ring_share(bpm, 1)),
            delta_mass=_num(pm.mass - bpm.mass))
        rows.append(rec)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I1h — do I1 and I2 agree where they measured the same thing?
# ─────────────────────────────────────────────────────────────────────────────

AGREE_FIELDS = (PROVENANCE + [
    "run_id", "condition_id", "stage", "basis", "n_images_i1", "n_images_i2",
    "max_abs_count_diff", "n_positions_differing", "mass_i1", "mass_i2",
    "rho", "identical", "note"])


def table_I1h(store, head) -> list:
    """I1 recaptured `s11_out` and `hist` in the passes that produced the
    block-input stages, so both work packages measured them independently.

    They must agree. This table is the check, and it is a table rather than
    an assertion because a disagreement is a RESULT — it would mean the two
    sweeps did not see the same model or the same images — and a number the
    reader can see beats an exception nobody kept.
    """
    rows = []
    for key in sorted(store.i2_maps):
        if key not in store.ind_source:
            continue                       # I1 did not measure this one
        a, b = store.maps[key], store.i2_maps[key]
        diff = np.abs(a.counts - b.counts) if (a.counts is not None
                                               and b.counts is not None) else None
        res = P.spatial_rho(a, b)
        rows.append(dict(
            head, run_id=key[0], condition_id=key[1], stage=key[2],
            basis=key[3], n_images_i1=a.n_images, n_images_i2=b.n_images,
            max_abs_count_diff=(int(diff.max()) if diff is not None
                                else MISSING),
            n_positions_differing=(int((diff > 0).sum()) if diff is not None
                                   else MISSING),
            mass_i1=_num(a.mass), mass_i2=_num(b.mass),
            rho=_num(res[0]) if res else MISSING,
            identical=int(diff is not None and int(diff.max()) == 0),
            note="I1 and I2 measured this stage independently; identical=1 "
                 "means the per-position counts agree exactly"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def build(split_name: str, *, manifest="results/frozen/I0_manifest/manifest.json",
          out_dir=None, git_sha_value=None, i2_root=I2_ROOT, i1_root=I1_ROOT,
          resamples=S.BOOTSTRAP_RESAMPLES, role=None) -> dict:
    """Build every table for one split.

    `git_sha_value`, the roots and `resamples` are ARGUMENTS so two builds
    from the same inputs produce the same bytes — a byte-identity test would
    otherwise fail on the next commit, which is a property of the repository
    and not of the table builder.
    """
    cohort = eligible_cohort(manifest)
    store = MapStore(split_name, cohort, i1_root=Path(i1_root),
                     i2_root=Path(i2_root), resamples=resamples)
    if not store.maps:
        raise SystemExit(
            f"no maps for split {split_name!r} under {i2_root} or {i1_root}")
    shas = sorted({pm.split_sha for pm in store.maps.values()})
    if len(shas) != 1:
        raise SystemExit(f"maps from {len(shas)} splits in one table: {shas}")
    head = {"work_package": "I1_spatial", "split_name": split_name,
            "split_sha256": shas[0],
            "git_sha": git_sha_value if git_sha_value else git_sha(),
            "generated_by": "analysis/frozen_I1_spatial.py"}
    if role is None:
        role = ("reporting" if split_name == "evaluation"
                else "discovery/selection")
    out_dir = Path(out_dir) if out_dir else TABLE_DIR

    written = {}
    written["T_I1a_prevalence"] = write_table(
        out_dir / "T_I1a_prevalence.csv", table_I1a(store, head), T_I1A_FIELDS)
    written["T_I1c_recipe"] = write_table(
        out_dir / "T_I1c_recipe.csv", table_I1c(store, head), RECIPE_FIELDS)
    written["T_I1d_seed_stability"] = write_table(
        out_dir / "T_I1d_seed_stability.csv", table_I1d(store, head),
        PAIR_FIELDS)
    written["T_I1e_eta2"] = write_table(
        out_dir / "T_I1e_eta2.csv", table_I1e(store, head), ETA2_FIELDS)
    written["T_I1f_blockinput"] = write_table(
        out_dir / "T_I1f_blockinput.csv", table_I1f(store, head, role=role),
        BLOCKINPUT_FIELDS)
    written["T_I1g_variant_maps"] = write_table(
        out_dir / "T_I1g_variant_maps.csv", table_I1g(store, head),
        VARIANT_FIELDS)
    written["T_I1h_agreement"] = write_table(
        out_dir / "T_I1h_agreement.csv", table_I1h(store, head), AGREE_FIELDS)
    return {"tables": written, "n_maps": len(store.maps), "head": head,
            "stages": sorted({k[2] for k in store.maps},
                             key=lambda s: STAGE_ORDER.get(s, 9)),
            "store": store, "role": role}


def main():
    p = argparse.ArgumentParser(description="TASK A / I1 spatial tables.")
    p.add_argument("--split", default="evaluation",
                   help="split NAME. `evaluation` is the reporting split; "
                        "`val_diag_split` is the discovery split and is "
                        "tabulated only as the source of a D5 mask")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resamples", type=int, default=S.BOOTSTRAP_RESAMPLES)
    args = p.parse_args()

    result = build(args.split, manifest=args.manifest, out_dir=args.out_dir,
                   resamples=args.resamples)
    print(f"split={args.split} ({result['role']})  maps={result['n_maps']}  "
          f"stages={result['stages']}")
    for name, path in result["tables"].items():
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print(f"  wrote {path}  ({n} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
