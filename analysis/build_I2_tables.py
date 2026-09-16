#!/usr/bin/env python3
"""
analysis/build_I2_tables.py
===========================
TASK I2 D4 — the five tables of §6, from the records the HPC sweep wrote.

    python analysis/build_I2_tables.py \
        [--results results/frozen/I2_terminal/<split_name>] \
        [--out    results/frozen/I2_terminal/<split_name>/tables]

`--results` names ONE split's sweep directory. A sweep writes under
`results/frozen/I2_terminal/<split_name>/<run_id>/`, so calibration and
evaluation can never share a records file, a completion marker or a
threshold; `build_all` additionally refuses a directory whose runs do not
all carry the same split sha.

| table | content |
|---|---|
| `T_I2a_invariance.csv` | per SAGA checkpoint x constant: max abs logit diff vs native, top-1 agreement, mean abs delta-NLL |
| `T_I2b_sweep.csv` | per checkpoint x constant x stage: every diagnostic, mean over images |
| `T_I2c_gaps.csv` | per cell x stage x diagnostic: the paired SAGA-baseline gap at three settings, with an image-level bootstrap CI, plus survival |
| `T_I2d_maps.csv` | per pair x stage x condition x map basis: spatial Spearman with the exact permutation reference, ring-1 share, excess concentration over the finite-sample null |
| `T_I2e_registers.csv` | registers vs baseline at `s11_out` and `hist`, native only, n stated |

NOTHING IS RE-DERIVED HERE.

  pairing of repeats      analysis/build_pooled_tables.py (CELLS,
                          LEGACY_MEMBERS, E2R_MEMBERS) — the recipe erratum
                          remap and the VOID legacy ViT-B trio exclusion
                          cannot drift between this table and e2_pooled.csv
  spatial permutation     analysis/address_analysis.rho_spatial — the exact
                          8 x side^2 dihedral/torus-roll reference TASK-07's
                          answers were written against
  rings                   analysis/address_analysis.ring_indices
  concentration + null    analysis/address_analysis.concentration_recompute,
                          concentration_null, position_rel_se
  survival                analysis/i2_decision.survival_of
  diagnostics             already computed per image by saga/frozen/diag.py

T_I2b covers ALL 19 eligible runs, not only the SAGA ones: a gap in T_I2c is
uninterpretable without the level it is a gap from, and the baseline rows cost
nothing to print.

Determinism and byte-identity: every row set is sorted explicitly, every float
is formatted through one function, and NO TIMESTAMP is written into a CSV —
the build timestamp and the input file shas live in `tables/build_meta.json`,
so `tests/test_I2_terminal.py` can pin the tables byte for byte.

No training, no optimizer, no probe fitting.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.address_analysis import (NOISY_REL_SE, NULLABLE,  # noqa: E402
                                       concentration_null,
                                       concentration_recompute,
                                       position_rel_se, rho_spatial,
                                       ring_indices)
from analysis.build_pooled_tables import (CELLS, E2R_MEMBERS,  # noqa: E402
                                          LEGACY_MEMBERS)
from analysis.i2_decision import survival_of  # noqa: E402
from saga.frozen import diag as fdiag  # noqa: E402
from saga.frozen.records import records_path  # noqa: E402
from saga.frozen.runner import eligible_cohort  # noqa: E402

MISSING = "MISSING"
WORK_PACKAGE = "I2_terminal"

#: Every per-image diagnostic the sweep records, in table order.
DIAGNOSTICS = ("norm_p50", "norm_p90", "norm_p99", "norm_p999", "norm_max",
               "mad_thr", "count_fixed_canon", "count_fixed_cal", "count_mad",
               "cos_all", "cos_nosink_mad", "eff_rank")

#: The three settings §6 asks the paired gap for: (name, stage, condition).
#: The SAGA side is taken under `condition`, the baseline side ALWAYS under
#: `native` — a baseline has no terminal gate to override, so `term_1.00` on
#: the SAGA side against `native` on the baseline side is the comparison the
#: bypass question actually asks.
GAP_SETTINGS = (("hist_native", "hist", "native"),
                ("hist_term_1.00", "hist", "term_1.00"),
                ("s11_out_native", "s11_out", "native"))

#: LOCKED_ANALYSIS §8: image-level bootstrap, 10,000 resamples. D6 (the seed)
#: is still DECISION NEEDED; 0 is its stated default and is recorded in every
#: output. The chunk size is part of the seed contract: RandomState draws a
#: single stream, so the resample matrix must be drawn in the same block size
#: every time for the CI to be reproducible.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CHUNK = 500
CI_PERCENTILES = (2.5, 97.5)

PROV_FIELDS = ("work_package", "split_name", "split_sha256", "git_sha")


class TableError(RuntimeError):
    """Results that cannot be tabulated as asked."""


# ─────────────────────────────────────────────────────────────────────────────
# Reading what the sweep wrote
# ─────────────────────────────────────────────────────────────────────────────

def read_rows(path: Path) -> list:
    """A records/diag file as a list of dicts, parquet or CSV."""
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:                       # pragma: no cover
            raise TableError(
                f"{path} is parquet and pyarrow is not installed: {exc}. "
                f"`pip install pyarrow` (it is in requirements.txt).") from exc
        return pq.read_table(path).to_pylist()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v):
    """A cell as a float, or the literal MISSING. Never a guess."""
    if v is None or v == "" or v == MISSING:
        return MISSING
    try:
        return float(v)
    except (TypeError, ValueError):
        return MISSING


def load_primary_pack(path):
    """`figures_data/frozen/I2_*_primary.npz` as
    `{run_id: {(condition, stage): {diagnostic: float32[n]}}}` + image ids.

    The evaluation sweep's raw `diag.parquet` stays on the cluster (19 runs x
    10,000 images is ~90 MB that nothing reads twice), so the repository
    commits this pack instead: the five PRIMARY diagnostics per image, which
    is what the image-level bootstrap resamples. Reading it here means the
    paired gaps are re-derived by the SAME `_paired_deltas` /
    `paired_bootstrap` code that built them from the parquet, not by a second
    implementation that could disagree.

    It carries the primary five and nothing else, so every other diagnostic
    is the literal MISSING in a table built from it — which is the honest
    statement, not a gap silently filled.
    """
    path = Path(path)
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as z:
        ids = np.asarray(z["__image_ids"])
        primary = [str(d) for d in np.asarray(z["__diagnostics"])]
        out = {}
        for key in z.files:
            if key.startswith("__"):
                continue
            run_id, cond, stage, diag = key.split("|")
            out.setdefault(run_id, {}).setdefault((cond, stage), {})[diag] = \
                np.asarray(z[key], dtype=np.float64)
    return {"runs": out, "primary": primary}, ids


def _grouped_from_pack(pack, ids, run_id):
    """One run's `grouped` structure, primary diagnostics only."""
    by_key = pack["runs"].get(run_id)
    if not by_key:
        return None
    order = np.argsort(ids)
    grouped = {}
    for key, vals in by_key.items():
        grouped[key] = {
            "image_ids": ids[order],
            "values": {d: (vals[d][order] if d in vals else MISSING)
                       for d in DIAGNOSTICS},
        }
    return grouped


def load_run(results_root: Path, run_id: str, pack=None, pack_ids=None):
    """(records rows, {(condition, stage): {"image_ids": [...], diag: array}}).

    Returns (None, None) for a run whose sweep has not landed yet — a run
    that has not been computed is an ABSENCE, reported as such, never a zero.

    When the raw diag file is absent but `pack` carries this run, the
    diagnostics come from the committed primary pack and every non-primary
    column is MISSING. The records file has no such fallback: `T_I2a` is
    about logits and NLL, which the pack does not hold.
    """
    run_dir = Path(results_root) / run_id
    rec_path = records_path(run_dir, "records")
    diag_path = records_path(run_dir, "diag")
    if not diag_path.exists() and pack is not None:
        grouped = _grouped_from_pack(pack, pack_ids, run_id)
        if grouped:
            records = read_rows(rec_path) if rec_path.exists() else None
            return records, grouped
    if not run_dir.exists():
        return None, None
    if not rec_path.exists() or not diag_path.exists():
        return None, None

    records = read_rows(rec_path)
    grouped = {}
    for r in read_rows(diag_path):
        key = (r["condition_id"], r["stage"])
        g = grouped.setdefault(key, {"image_ids": [], "values": {
            d: [] for d in DIAGNOSTICS}})
        g["image_ids"].append(r["image_id"])
        for d in DIAGNOSTICS:
            g["values"][d].append(_num(r.get(d)))
    for g in grouped.values():
        order = np.argsort(np.asarray(g["image_ids"]))
        g["image_ids"] = np.asarray(g["image_ids"])[order]
        g["values"] = {
            d: _as_array(np.asarray(v, dtype=object)[order])
            for d, v in g["values"].items()}
    return records, grouped


def _as_array(values):
    """float64 array, or MISSING when ANY entry is MISSING.

    A half-MISSING column is not a column to average: `count_fixed_canon` is
    defined at `hist` and nowhere else, and this keeps that fact intact
    instead of averaging over the stages where it exists.
    """
    if any(v == MISSING for v in values):
        return MISSING
    return np.asarray(values, dtype=np.float64)


def load_maps(results_root: Path, run_id: str):
    """{(condition, stage, basis): (counts int64[N], n_images)} or None."""
    path = Path(results_root) / run_id / "maps.npz"
    if not path.exists():
        return None
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            if key.startswith("n_images__") or key == "meta_json":
                continue
            cond, stage, basis = key.split("|")
            out[(cond, stage, basis)] = (np.asarray(z[key], dtype=np.int64),
                                         int(z[f"n_images__{key}"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cells and pairing — imported, never re-derived
# ─────────────────────────────────────────────────────────────────────────────

def cell_tags(arch: str, recipe_actual: str) -> list:
    """The provenance tags of one cell, in `build_pooled_tables`' own order.

    That module IS the cell definition: which legacy directory belongs to
    which `recipe_actual`, and which e2r seeds exist, are recorded there
    (recipe erratum; the VOID legacy ViT-B mixup-dir trio appears nowhere).
    Reading the tags from it is what keeps a gap in this table and a delta in
    `results/tables/e2_pooled.csv` about the same repeats.
    """
    return ([tag for tag, _ in LEGACY_MEMBERS.get((arch, recipe_actual), [])]
            + [tag for tag, _ in E2R_MEMBERS.get((arch, recipe_actual), [])])


def pairs_for(cohort, arch, recipe_actual, variant):
    """[(tag, baseline_run_id, variant_run_id)] for one cell, tag-paired."""
    by = {(r["variant"], r.get("provenance_tag")): r["run_id"]
          for r in cohort
          if r["arch"] == arch and r["recipe_actual"] == recipe_actual}
    out = []
    for tag in cell_tags(arch, recipe_actual):
        base, other = by.get(("baseline", tag)), by.get((variant, tag))
        if base and other:
            out.append((tag, base, other))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Formatting and writing
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v) -> str:
    """One float format for every table cell; MISSING passes through."""
    if v == MISSING or v is None:
        return MISSING
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, bool):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return MISSING
    return f"{f:.10g}"


def write_csv(path: Path, fields, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, MISSING)) for k in fields})
    return path


def _mean(values):
    """Mean over images, or MISSING for a column that has none."""
    if values is MISSING or values is None:
        return MISSING
    arr = np.asarray(values, dtype=np.float64)
    return MISSING if arr.size == 0 else float(arr.mean())


# ─────────────────────────────────────────────────────────────────────────────
# The image-level bootstrap (LOCKED_ANALYSIS §8)
# ─────────────────────────────────────────────────────────────────────────────

def paired_bootstrap(per_pair_deltas, *, resamples=BOOTSTRAP_RESAMPLES,
                     seed=BOOTSTRAP_SEED, chunk=BOOTSTRAP_CHUNK):
    """(mean, ci_lo, ci_hi) for the mean over pairs of paired image means.

    The resampling unit is the IMAGE (LOCKED_ANALYSIS §8) and the same
    resampled image indices are used for every pair, which is what keeps the
    pairing intact. Because every pair covers the same image set, the mean
    over pairs of per-pair image means equals the image mean of the
    across-pair average delta, so one [N] vector is resampled rather than a
    [P, N] matrix — the same statistic, exactly, at a fraction of the cost.
    """
    if not per_pair_deltas:
        return MISSING, MISSING, MISSING
    stack = np.stack([np.asarray(d, dtype=np.float64)
                      for d in per_pair_deltas])
    if stack.shape[1] == 0:
        return MISSING, MISSING, MISSING
    per_image = stack.mean(axis=0)                       # [N]
    n = per_image.size
    rng = np.random.RandomState(seed)
    stats = np.empty(resamples, dtype=np.float64)
    done = 0
    while done < resamples:
        take = min(chunk, resamples - done)
        idx = rng.randint(0, n, size=(take, n))
        stats[done:done + take] = per_image[idx].mean(axis=1)
        done += take
    lo, hi = np.percentile(stats, CI_PERCENTILES)
    return float(per_image.mean()), float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# T_I2a — the invariance table
# ─────────────────────────────────────────────────────────────────────────────

def build_a(cohort, loaded, prov) -> list:
    rows = []
    for r in cohort:
        if r["variant"] != "saga":
            continue
        records, grouped = loaded[r["run_id"]]
        if records is None:
            continue
        by_cond = {}
        for rec_row in records:
            by_cond.setdefault(rec_row["condition_id"], {})[
                rec_row["image_id"]] = rec_row
        native = by_cond.get("native", {})
        for cond in sorted(by_cond):
            if cond == "native":
                continue
            ids = sorted(set(by_cond[cond]) & set(native))
            if not ids:
                continue
            diffs = np.array([_num(by_cond[cond][i][
                "max_abs_logit_diff_vs_native"]) for i in ids],
                dtype=np.float64)
            agree = np.mean([by_cond[cond][i]["top1"] == native[i]["top1"]
                             for i in ids])
            dnll = np.abs(np.array([_num(by_cond[cond][i]["nll"]) for i in ids])
                          - np.array([_num(native[i]["nll"]) for i in ids]))
            cos_diff = _cos_shift(grouped, cond, "native", "hist")
            rows.append(dict(
                prov, run_id=r["run_id"], arch=r["arch"],
                recipe_actual=r["recipe_actual"], variant=r["variant"],
                provenance_tag=r.get("provenance_tag", MISSING),
                ckpt_sha256=r.get("ckpt_sha256", MISSING),
                condition_id=cond, n_images=len(ids),
                max_abs_logit_diff_vs_native=float(diffs.max()),
                mean_abs_logit_diff_vs_native=float(diffs.mean()),
                top1_agreement=float(agree),
                mean_abs_delta_nll=float(dnll.mean()),
                max_abs_delta_nll=float(dnll.max()),
                max_abs_cos_all_diff_hist=cos_diff,
                # phi_L = 0 is a weight-decay fixed point, so sigma(phi_L)
                # = 0.50 and term_0.50 must reproduce native exactly. The
                # flag is MEASURED here and is MISSING for the constants
                # that are not expected to be identical.
                phi_L_zero_bit_identical=(
                    int(diffs.max() == 0.0 and cos_diff == 0.0)
                    if cond == "term_0.50" else MISSING)))
    return sorted(rows, key=lambda r: (r["arch"], r["recipe_actual"],
                                       r["run_id"], r["condition_id"]))


def _cos_shift(grouped, cond, ref_cond, stage):
    """max |cos_all(cond) - cos_all(native)| over images at `stage`."""
    a = grouped.get((cond, stage))
    b = grouped.get((ref_cond, stage))
    if not a or not b:
        return MISSING
    if not np.array_equal(a["image_ids"], b["image_ids"]):
        return MISSING
    va, vb = a["values"]["cos_all"], b["values"]["cos_all"]
    if va is MISSING or vb is MISSING:
        return MISSING
    return float(np.abs(va - vb).max())


# ─────────────────────────────────────────────────────────────────────────────
# T_I2b — every diagnostic, mean over images
# ─────────────────────────────────────────────────────────────────────────────

def build_b(cohort, loaded, prov) -> list:
    rows = []
    for r in cohort:
        _records, grouped = loaded[r["run_id"]]
        if grouped is None:
            continue
        for (cond, stage) in sorted(grouped):
            g = grouped[(cond, stage)]
            rows.append(dict(
                prov, run_id=r["run_id"], arch=r["arch"],
                recipe_actual=r["recipe_actual"], variant=r["variant"],
                provenance_tag=r.get("provenance_tag", MISSING),
                ckpt_sha256=r.get("ckpt_sha256", MISSING),
                condition_id=cond, stage=stage,
                n_images=len(g["image_ids"]),
                **{d: _mean(g["values"][d]) for d in DIAGNOSTICS}))
    return sorted(rows, key=lambda r: (r["arch"], r["recipe_actual"],
                                       r["variant"], r["run_id"],
                                       r["condition_id"], r["stage"]))


# ─────────────────────────────────────────────────────────────────────────────
# T_I2c — the paired gaps and the survival fractions
# ─────────────────────────────────────────────────────────────────────────────

def _paired_deltas(loaded, pairs, stage, condition, diagnostic):
    """[per-pair delta arrays] for one (stage, condition, diagnostic).

    A pair whose two runs do not carry the SAME image ids in the same order
    is dropped with a reason rather than aligned by position: two splits that
    happen to be the same length are not the same split.
    """
    out, dropped = [], []
    for tag, base_id, other_id in pairs:
        gb = (loaded[base_id][1] or {}).get(("native", stage))
        go = (loaded[other_id][1] or {}).get((condition, stage))
        if not gb or not go:
            dropped.append(f"{tag}:absent")
            continue
        if not np.array_equal(gb["image_ids"], go["image_ids"]):
            dropped.append(f"{tag}:image_ids_differ")
            continue
        vb, vo = gb["values"][diagnostic], go["values"][diagnostic]
        if vb is MISSING or vo is MISSING:
            dropped.append(f"{tag}:{diagnostic}_MISSING")
            continue
        out.append((tag, vo - vb))
    return out, dropped


def build_c(cohort, loaded, prov, *, variant="saga", seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES) -> list:
    rows = []
    for arch, rec in CELLS:
        cell = fdiag.cell_key(arch, rec)
        pairs = pairs_for(cohort, arch, rec, variant)
        if not pairs:
            continue
        gap_by = {}
        collected = {}
        for setting, stage, cond in GAP_SETTINGS:
            for d in DIAGNOSTICS:
                deltas, dropped = _paired_deltas(loaded, pairs, stage, cond, d)
                if deltas:
                    mean, lo, hi = paired_bootstrap(
                        [v for _, v in deltas], resamples=resamples, seed=seed)
                    n_images = int(len(deltas[0][1]))
                else:
                    mean = lo = hi = MISSING
                    n_images = 0
                gap_by[(setting, d)] = mean
                collected[(setting, d)] = (deltas, dropped, mean, lo, hi,
                                           n_images)
        for setting, stage, cond in GAP_SETTINGS:
            for d in DIAGNOSTICS:
                deltas, dropped, mean, lo, hi, n_images = collected[
                    (setting, d)]
                base = gap_by[("hist_native", d)]
                rows.append(dict(
                    prov, cell=cell, arch=arch, recipe_actual=rec,
                    variant=variant, setting=setting, stage=stage,
                    condition_id=cond, diagnostic=d,
                    n_pairs=len(deltas), n_images=n_images,
                    gap_mean=mean, ci_lo=lo, ci_hi=hi,
                    bootstrap_resamples=resamples, bootstrap_seed=seed,
                    survival=(survival_of(base, mean)
                              if setting == "hist_term_1.00" else MISSING),
                    survival_s11=(survival_of(base, mean)
                                  if setting == "s11_out_native" else MISSING),
                    per_pair=";".join(f"{tag}={fmt(float(v.mean()))}"
                                      for tag, v in deltas) or MISSING,
                    dropped_pairs=";".join(dropped) or ""))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# T_I2d — the exceedance maps
# ─────────────────────────────────────────────────────────────────────────────

def _map_stats(counts, n_images):
    """Ring-1 share, concentration, and the excess over the finite-sample
    null, for one exceedance map. MISSING for a map with no mass."""
    freq = counts.astype(np.float64) / max(n_images, 1)
    n_positions = freq.size
    side = int(round(n_positions ** 0.5))
    out = {"total_mass": float(freq.sum()), "n_positions": n_positions}
    if side * side != n_positions:
        out["ring1_share"] = MISSING
    elif freq.sum() <= 0:
        out["ring1_share"] = MISSING
    else:
        out["ring1_share"] = float(freq[ring_indices(side, 1)].sum()
                                   / freq.sum())
    if freq.sum() <= 0:
        out.update({f"excess_{s}": MISSING for s in NULLABLE})
        out.update({s: MISSING for s in NULLABLE})
        out["rel_se"] = MISSING
        out["noisy"] = MISSING
        return out
    conc = concentration_recompute(freq)
    null = concentration_null(conc["total_mass"], n_images, n_positions)
    for s in NULLABLE:
        out[s] = float(conc[s])
        out[f"excess_{s}"] = (float(conc[s]) - null[s][0]
                              if null[s][0] is not None else MISSING)
    rel_se = position_rel_se(conc["total_mass"], n_images, n_positions)
    out["rel_se"] = MISSING if rel_se is None else float(rel_se)
    out["noisy"] = (MISSING if rel_se is None
                    else int(rel_se > NOISY_REL_SE))
    return out


def build_d(cohort, results_root, prov, *, variant="saga") -> list:
    maps_by_run = {r["run_id"]: load_maps(results_root, r["run_id"])
                   for r in cohort}
    rows = []
    for arch, rec in CELLS:
        cell = fdiag.cell_key(arch, rec)
        for tag, base_id, other_id in pairs_for(cohort, arch, rec, variant):
            mb, mo = maps_by_run.get(base_id), maps_by_run.get(other_id)
            if not mb or not mo:
                continue
            for (cond, stage, basis) in sorted(mo):
                ref = mb.get(("native", stage, basis))
                if ref is None:
                    continue
                o_counts, o_n = mo[(cond, stage, basis)]
                b_counts, b_n = ref
                if o_counts.shape != b_counts.shape:
                    raise TableError(
                        f"{other_id} and {base_id} disagree on the patch grid "
                        f"at {stage}/{basis}: {o_counts.shape} vs "
                        f"{b_counts.shape}")
                o_stats = _map_stats(o_counts, o_n)
                b_stats = _map_stats(b_counts, b_n)
                spatial = rho_spatial(o_counts.astype(np.float64),
                                      b_counts.astype(np.float64))
                if spatial is None:
                    rho = p_spatial = null_sd = n_transforms = MISSING
                else:
                    rho, p_spatial, null_sd, n_transforms = spatial
                    p_spatial = MISSING if p_spatial is None else p_spatial
                    null_sd = MISSING if null_sd is None else null_sd
                rows.append(dict(
                    prov, cell=cell, arch=arch, recipe_actual=rec,
                    provenance_tag=tag, variant=variant,
                    baseline_run_id=base_id, variant_run_id=other_id,
                    condition_id=cond, stage=stage, map_basis=basis,
                    n_images=o_n, n_positions=o_stats["n_positions"],
                    rho_spatial=rho, p_spatial=p_spatial,
                    null_sd=null_sd, n_transforms=n_transforms,
                    mass_variant=o_stats["total_mass"],
                    mass_baseline=b_stats["total_mass"],
                    ring1_share_variant=o_stats["ring1_share"],
                    ring1_share_baseline=b_stats["ring1_share"],
                    rel_se_variant=o_stats["rel_se"],
                    noisy_variant=o_stats["noisy"],
                    **{f"excess_{s}_variant": o_stats[f"excess_{s}"]
                       for s in NULLABLE},
                    **{f"excess_{s}_baseline": b_stats[f"excess_{s}"]
                       for s in NULLABLE}))
    return sorted(rows, key=lambda r: (r["cell"], r["provenance_tag"],
                                       r["stage"], r["condition_id"],
                                       r["map_basis"]))


# ─────────────────────────────────────────────────────────────────────────────
# T_I2e — registers
# ─────────────────────────────────────────────────────────────────────────────

REGISTER_STAGES = ("s11_out", "hist")


def build_e(cohort, loaded, prov, *, seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES) -> list:
    rows = []
    for arch, rec in CELLS:
        cell = fdiag.cell_key(arch, rec)
        pairs = pairs_for(cohort, arch, rec, "registers")
        if not pairs:
            continue
        for stage in REGISTER_STAGES:
            for d in DIAGNOSTICS:
                deltas, dropped = _paired_deltas(loaded, pairs, stage,
                                                 "native", d)
                if deltas:
                    mean, lo, hi = paired_bootstrap(
                        [v for _, v in deltas], resamples=resamples, seed=seed)
                    n_images = int(len(deltas[0][1]))
                else:
                    mean = lo = hi = MISSING
                    n_images = 0
                rows.append(dict(
                    prov, cell=cell, arch=arch, recipe_actual=rec,
                    variant="registers", stage=stage, condition_id="native",
                    diagnostic=d, n_pairs=len(deltas), n_images=n_images,
                    gap_mean=mean, ci_lo=lo, ci_hi=hi,
                    bootstrap_resamples=resamples, bootstrap_seed=seed,
                    per_pair=";".join(f"{tag}={fmt(float(v.mean()))}"
                                      for tag, v in deltas) or MISSING,
                    dropped_pairs=";".join(dropped) or ""))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

FIELDS_A = PROV_FIELDS + (
    "run_id", "arch", "recipe_actual", "variant", "provenance_tag",
    "ckpt_sha256", "condition_id", "n_images",
    "max_abs_logit_diff_vs_native", "mean_abs_logit_diff_vs_native",
    "top1_agreement", "mean_abs_delta_nll", "max_abs_delta_nll",
    "max_abs_cos_all_diff_hist", "phi_L_zero_bit_identical")

FIELDS_B = PROV_FIELDS + (
    "run_id", "arch", "recipe_actual", "variant", "provenance_tag",
    "ckpt_sha256", "condition_id", "stage", "n_images") + DIAGNOSTICS

FIELDS_C = PROV_FIELDS + (
    "cell", "arch", "recipe_actual", "variant", "setting", "stage",
    "condition_id", "diagnostic", "n_pairs", "n_images", "gap_mean",
    "ci_lo", "ci_hi", "survival", "survival_s11", "bootstrap_resamples",
    "bootstrap_seed", "per_pair", "dropped_pairs")

FIELDS_D = PROV_FIELDS + (
    "cell", "arch", "recipe_actual", "provenance_tag", "variant",
    "baseline_run_id", "variant_run_id", "condition_id", "stage",
    "map_basis", "n_images", "n_positions", "rho_spatial", "p_spatial",
    "null_sd", "n_transforms", "mass_variant", "mass_baseline",
    "ring1_share_variant", "ring1_share_baseline", "rel_se_variant",
    "noisy_variant") + tuple(
        f"excess_{s}_{side}" for side in ("variant", "baseline")
        for s in NULLABLE)

FIELDS_E = PROV_FIELDS + (
    "cell", "arch", "recipe_actual", "variant", "stage", "condition_id",
    "diagnostic", "n_pairs", "n_images", "gap_mean", "ci_lo", "ci_hi",
    "bootstrap_resamples", "bootstrap_seed", "per_pair", "dropped_pairs")


def build_all(results_root, out_dir, *, manifest, git_sha_value,
              seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES,
              primary_pack=None) -> dict:
    results_root, out_dir = Path(results_root), Path(out_dir)
    cohort = eligible_cohort(manifest)
    pack, pack_ids = load_primary_pack(primary_pack) if primary_pack \
        else (None, None)
    loaded = {r["run_id"]: load_run(results_root, r["run_id"], pack, pack_ids)
              for r in cohort}
    # a run counts as present if EITHER file landed: the diagnostics can come
    # from the committed primary pack while the records stay on the cluster
    present = [r["run_id"] for r in cohort
               if loaded[r["run_id"]][0] or loaded[r["run_id"]][1]]
    with_records = [rid for rid in present if loaded[rid][0]]
    if not present:
        raise TableError(
            f"no sweep results under {results_root} — run the HPC array "
            f"first; nothing here computes a diagnostic")

    if with_records:
        split_names = {r.get("split_name") for rid in with_records
                       for r in loaded[rid][0][:1]}
        split_shas = {r.get("split_sha256") for rid in with_records
                      for r in loaded[rid][0][:1]}
    else:
        # no records file anywhere: take the split from the completion
        # markers, which are committed for every run
        split_names, split_shas = set(), set()
        for rid in present:
            marker = results_root / rid / "diag.done.json"
            if marker.exists():
                doc = json.loads(marker.read_text(encoding="utf-8"))
                split_names.add(doc.get("split_name"))
                split_shas.add(doc.get("split_sha256"))
    if len(split_shas) != 1:
        raise TableError(
            f"the runs under {results_root} carry {len(split_shas)} different "
            f"split shas {sorted(split_shas)} — one table, one split")
    prov = {"work_package": WORK_PACKAGE, "split_name": sorted(split_names)[0],
            "split_sha256": sorted(split_shas)[0], "git_sha": git_sha_value}

    tables = {
        "T_I2a_invariance.csv": (FIELDS_A, build_a(cohort, loaded, prov)),
        "T_I2b_sweep.csv": (FIELDS_B, build_b(cohort, loaded, prov)),
        "T_I2c_gaps.csv": (FIELDS_C, build_c(cohort, loaded, prov, seed=seed,
                                             resamples=resamples)),
        "T_I2d_maps.csv": (FIELDS_D, build_d(cohort, results_root, prov)),
        "T_I2e_registers.csv": (FIELDS_E, build_e(cohort, loaded, prov,
                                                  seed=seed,
                                                  resamples=resamples)),
    }
    written = {}
    for name, (fields, rows) in tables.items():
        write_csv(out_dir / name, fields, rows)
        written[name] = len(rows)
    return {"prov": prov, "written": written, "present": present,
            "with_records": with_records,
            "diagnostics_from_pack": sorted(set(present) - set(with_records)),
            "absent": [r["run_id"] for r in cohort
                       if r["run_id"] not in present]}


def main():
    from saga.run_registry import git_sha

    p = argparse.ArgumentParser(description="Build TASK I2's five tables.")
    p.add_argument("--results",
                   default=f"results/frozen/{WORK_PACKAGE}/calibration",
                   help="the sweep directory for ONE split; B2/C2 pass "
                        f"results/frozen/{WORK_PACKAGE}/evaluation")
    p.add_argument("--out", default=None,
                   help="default: <results>/tables")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED,
                   help="bootstrap seed (LOCKED_ANALYSIS D6; default 0)")
    p.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    p.add_argument("--primary-pack", default=None,
                   help="figures_data/frozen/I2_<split>_primary.npz — the "
                        "committed per-image primary diagnostics, used for "
                        "any run whose raw diag file stayed on the cluster")
    args = p.parse_args()

    out_dir = Path(args.out or (Path(args.results) / "tables"))
    summary = build_all(args.results, out_dir, manifest=args.manifest,
                        git_sha_value=git_sha(), seed=args.seed,
                        resamples=args.resamples,
                        primary_pack=args.primary_pack)
    for name, n in summary["written"].items():
        print(f"wrote {out_dir / name}: {n} rows")
    print(f"split: {summary['prov']['split_name']} "
          f"({summary['prov']['split_sha256'][:16]}…)")
    print(f"runs present: {len(summary['present'])} "
          f"({len(summary['with_records'])} with a records file)")
    if summary["diagnostics_from_pack"]:
        print(f"diagnostics from the primary pack (raw diag on the cluster): "
              f"{len(summary['diagnostics_from_pack'])} run(s); every "
              f"non-primary diagnostic is MISSING in these tables")
    if summary["absent"]:
        print(f"runs ABSENT (not computed, never a zero): "
              f"{summary['absent']}")

    (out_dir / "build_meta.json").write_text(
        json.dumps({"generated_by": "analysis/build_I2_tables.py",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "bootstrap_seed": args.seed,
                    "bootstrap_resamples": args.resamples,
                    "bootstrap_chunk": BOOTSTRAP_CHUNK,
                    "primary_pack": args.primary_pack,
                    "runs_with_records": summary["with_records"],
                    "runs_from_primary_pack": summary["diagnostics_from_pack"],
                    **summary["prov"],
                    "rows": summary["written"],
                    "runs_present": summary["present"],
                    "runs_absent": summary["absent"]},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
