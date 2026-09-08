#!/usr/bin/env python3
"""
analysis/address_analysis.py
============================
TASK-07 C1: does the sink live at an "address"?

Consumes the committed `*_addr.json` maps (tools/sink_address.py, HPC A1)
and the committed learned gates, and answers the five TASK-07 questions
with numbers, per cell (arch x recipe_actual):

  Q1 does an address exist?    concentration of the BASELINE freq maps
                               (entropy, Gini, top-5/top-20 share) against
                               the explicit uniform reference.
  Q2 is it seed-stable?        Spearman of baseline freq maps across all
                               repeat pairs (the make-or-break number).
  Q3 is it recipe-stable?      Spearman of S/mixup vs S/true-nomix
                               baseline maps (all cross pairs).
  Q4 does SAGA relocate it?    variant-vs-baseline map Spearman (paired by
                               provenance) + the change in concentration.
                               Registers likewise.
  Q5 does the gate know it?    Spearman of a baseline freq map against
                               sigmoid(phi) per layer (mean over heads) for
                               every SAGA run in the cell, plus the run's
                               OWN freq map as the second comparator.

Outputs:
- results/tables/sink_address.csv   — long format; every correlation row
  carries `n` and an explicit `reference` (uniform / zero).
- results/figures_data/Faddr.npz    — the freq maps, gate maps and
  per-layer gate-vs-address profiles that plotting/plot_address.py draws.

TWO REFERENCES THAT ARE NOT ZERO (both added after an adversarial review
found the naive ones invert a result):

1. Concentration has a FINITE-SAMPLE FLOOR. A spatially uniform ground
   truth does NOT give Gini 0 when estimated from 10000 images: at mass
   0.02 sinks/image almost every position is 0 or 1, and the null Gini is
   ~0.49, while at mass 20 it is ~0.017. Since mass ranges over three
   orders of magnitude across members, raw Gini/entropy/top-k are NOT
   comparable between them. Every concentration statistic is therefore
   reported with its own null (`_null`, mean over `--null-sims` binomial
   draws at that map's own mass and n_images) and as EXCESS over that null
   (`_excess`), and Q4's deltas are differences of EXCESS values.

2. The 196 positions are NOT independent - the maps are spatially smooth,
   so the iid Spearman reference (sd = 1/sqrt(195) = 0.072) is far too
   generous; the true spatial null sd is 0.16-0.33, i.e. n_eff ~ 20. Every
   correlation is therefore reported with an EXACT permutation p-value
   under the 8 dihedral x 196 torus-roll transforms of one map (1568
   transforms, deterministic - no seed), which preserves both maps'
   autocorrelation, plus that null's sd.

Discipline:
- Cell membership is IMPORTED from analysis/build_pooled_tables.py, so the
  erratum remap (legacy dirnames -> recipe_actual) and the exclusion of the
  VOID legacy ViT-B mixup-dir trio can never drift between the two tables.
- Diagnostics come from the LAST checkpoint only; `*_best_addr.json` files
  exist but are never read (no best/last contamination).
- Every legacy map is sha-matched to `checkpoint_manifest.csv` and every
  run-dir map to its sibling diag JSON; a mismatch is a GAP (excluded and
  reported), never a silently plotted number.
- The concentration block stored by the tool is RE-DERIVED here through an
  independent implementation (pairwise-difference Gini, explicit entropy)
  and disagreement is a hard error.
- Spatially constant maps have no defined rank correlation; such layers /
  pairs are excluded and COUNTED, never averaged in as NaN.
- Q4 pairs variant against baseline BY PROVENANCE for both the
  correlations and the concentration deltas; an unpaired repeat is an
  explicit MISSING row, never silently dropped into a cell mean.
- Q5 reports each run's extremal layer as an argmax over 12 layers, so it
  carries a Bonferroni-corrected p beside the raw one, and the opposite-
  sign extreme of the same profile.

    python analysis/address_analysis.py
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.build_pooled_tables import (CELLS, E2R_MEMBERS, LEGACY_MEMBERS,
                                          VARIANTS)

MISSING = "MISSING"
BASES = ("canon", "mad")
# concentration statistics that have a finite-sample null (total_mass does
# not - it IS the mass the null is conditioned on)
NULLABLE = ("entropy_bits", "entropy_normalized", "gini", "top5_share",
            "top20_share")
CONC_STATS = NULLABLE + ("total_mass",)
NULL_SIMS = 400
NULL_SEED = 0
# a per-position frequency estimated from n_images has relative standard
# error sqrt((1-p)/(n*p)); above this the map is mostly Monte-Carlo noise
# and its concentration/correlation numbers are not comparable
NOISY_REL_SE = 0.5

FIELDS = ["question", "arch", "recipe_actual", "map_basis", "variant",
          "subject", "comparator", "statistic", "value", "n", "reference",
          "note"]


# ── independent concentration re-derivation (cross-check of the A1 tool) ─────

def gini_pairwise(x: np.ndarray) -> float:
    """Gini via the mean-absolute-difference definition — deliberately NOT
    the sorted-cumulative formula used by tools/sink_address.py."""
    x = np.asarray(x, dtype=np.float64)
    if x.sum() == 0:
        return float("nan")
    return float(np.abs(x[:, None] - x[None, :]).mean() / (2.0 * x.mean()))


def concentration_recompute(freq) -> dict:
    f = np.asarray(freq, dtype=np.float64)
    n = f.size
    mass = f.sum()
    if mass <= 0:
        raise ValueError("concentration is undefined for a zero-mass map")
    q = f / mass
    nz = q[q > 0]
    desc = np.sort(q)[::-1]
    return {
        "total_mass": float(mass),
        "entropy_bits": float(-(nz * np.log2(nz)).sum()),
        "entropy_uniform_bits": float(np.log2(n)),
        "entropy_normalized": float(-(nz * np.log2(nz)).sum() / np.log2(n)),
        "gini": gini_pairwise(f),
        "top5_share": float(desc[:5].sum()),
        "top20_share": float(desc[:20].sum()),
    }


def verify_stored_concentration(addr: dict, basis: str, label: str,
                                tol=1e-9):
    """The tool's stored block must match an independent recompute.

    A one-sided NaN is a FAILURE (an `abs(nan - x) > tol` comparison is
    False, so a naive check would wave it through), and the stored
    top-10 positions are verified too since they reach the CSV.
    """
    stored = addr[f"concentration_{basis}"]
    freq = np.asarray(addr[f"freq_{basis}"], dtype=np.float64)
    got = concentration_recompute(freq)
    for key, value in got.items():
        ref = stored.get(key)
        if ref is None:
            raise SystemExit(f"{label} [{basis}]: stored concentration is "
                             f"missing {key!r}")
        finite_got, finite_ref = np.isfinite(value), np.isfinite(ref)
        if finite_got != finite_ref:
            raise SystemExit(
                f"{label} [{basis}]: stored {key}={ref!r} and recomputed "
                f"{value!r} disagree on finiteness")
        if not finite_got:
            continue
        if abs(value - ref) > tol:
            raise SystemExit(
                f"{label} [{basis}]: stored {key}={ref!r} disagrees with "
                f"independent recompute {value!r}")
    top = stored.get("top10_positions")
    if top is None:
        raise SystemExit(f"{label} [{basis}]: stored block has no "
                         f"top10_positions")
    expect = np.argsort(-freq, kind="stable")[:len(top)]
    if [int(p) for p in top] != [int(p) for p in expect]:
        raise SystemExit(
            f"{label} [{basis}]: stored top10_positions {list(top)} "
            f"disagrees with independent recompute {list(expect)}")


# ── references ───────────────────────────────────────────────────────────────

def uniform_reference(statistic: str, n_positions: int) -> str:
    u = {
        "entropy_bits": f"uniform={np.log2(n_positions):.4f}",
        "entropy_normalized": "uniform=1.0",
        "gini": f"uniform=0.0, one-hot={(n_positions - 1) / n_positions:.4f}",
        "top5_share": f"uniform={5 / n_positions:.5f}",
        "top20_share": f"uniform={20 / n_positions:.5f}",
        "total_mass": "mean sinks/image (no uniform ref)",
    }
    return u.get(statistic, "")


# ── member registry ──────────────────────────────────────────────────────────

def cell_members(arch, rec, variant):
    """[(provenance_tag, addr_path, gate_path|None, manifest_key|None)]."""
    out = []
    for tag, dirname in LEGACY_MEMBERS.get((arch, rec), []):
        stem = f"e2_{arch}_{dirname}_{variant}_rlast_last"
        addr = Path("results/legacy/diag") / f"{stem}_addr.json"
        gate = (Path("results/legacy/gates")
                / f"e2_{arch}_{dirname}_saga_rlast_last_phi.npz"
                if variant == "saga" else None)
        out.append((tag, addr, gate,
                    (arch, dirname, variant, "rlast", "last.pth")))
    for tag, pattern in E2R_MEMBERS.get((arch, rec), []):
        run = Path("results/runs") / pattern.format(v=variant)
        # a run that was never trained is an expected ABSENCE, not a gap
        # (the e2r registers chains are TASK-07 A2 optional runs) — same
        # convention as analysis/build_pooled_tables.py
        if not run.exists():
            continue
        gate = run / "gates" / "phi_e299.npz" if variant == "saga" else None
        out.append((tag, run / "diag" / "diag_final_last_addr.json",
                    gate, None))
    return out


def absent_members():
    """(cell, variant, tag) entries whose run dir does not exist at all —
    reported separately from gaps so a never-submitted optional run is
    never mistaken for a missing result."""
    out = []
    for arch, rec in CELLS:
        for variant in VARIANTS:
            for tag, pattern in E2R_MEMBERS.get((arch, rec), []):
                run = Path("results/runs") / pattern.format(v=variant)
                if not run.exists():
                    out.append(f"{arch}|{rec}|{variant}|{tag} "
                               f"({run.name}: not trained)")
    return out


def load_manifest(path: Path) -> dict:
    """(arch, recipe_dirname, variant, seed, filename) -> sha256."""
    man = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            man[(r["arch"], r["recipe"], r["variant"], r["seed"],
                 r["filename"])] = r["sha256"]
    return man


def load_member(addr_path: Path, manifest_key, manifest, label):
    """(addr dict, None) or (None, gap reason) — sha-validated."""
    if not addr_path.exists():
        return None, f"no {addr_path.name}"
    addr = json.load(open(addr_path))
    if manifest_key is not None:
        expect = manifest.get(manifest_key)
        if expect is None:
            return None, f"no manifest row for {manifest_key}"
        if addr["ckpt_sha256"] != expect:
            return None, "ckpt_sha256 mismatch vs manifest"
    else:
        sibling = addr_path.with_name("diag_final_last.json")
        if not sibling.exists():
            return None, "no sibling diag_final_last.json"
        if json.load(open(sibling)).get("ckpt_sha256") != addr["ckpt_sha256"]:
            return None, "ckpt_sha256 mismatch vs sibling diag"
    for basis in BASES:
        if addr.get(f"freq_{basis}") is None:
            return None, f"freq_{basis} is null ({addr.get('canon_skip_reason')})"
        verify_stored_concentration(addr, basis, label)
    return addr, None


def collect(manifest) -> tuple:
    """(members, gaps): members[(arch, rec, variant)] = [(tag, addr, gate)]."""
    members, gaps = {}, []
    for arch, rec in CELLS:
        for variant in VARIANTS:
            found = []
            for tag, addr_path, gate_path, mkey in cell_members(
                    arch, rec, variant):
                label = f"{arch}|{rec}|{variant}|{tag}"
                addr, reason = load_member(addr_path, mkey, manifest, label)
                if addr is None:
                    gaps.append(f"{label}: {reason}")
                    continue
                found.append((tag, addr, gate_path))
            if found:
                members[(arch, rec, variant)] = found
    return members, gaps


# ── correlation helpers ──────────────────────────────────────────────────────

def zrank(v):
    """Mid-ranks, centred and unit-normalised, so that a dot product of two
    of these IS the Spearman correlation (tie-corrected). None when the map
    is spatially constant (no defined ranking)."""
    r = rankdata(np.asarray(v, dtype=np.float64))   # 'average' = mid-ranks
    r = r - r.mean()
    n = float(np.linalg.norm(r))
    return None if n == 0.0 else r / n


def build_perm(side: int) -> np.ndarray:
    """Index permutations of the 8 dihedral x side^2 torus-roll transforms
    of a side x side grid, as (8*side^2, side^2) indices.

    Rolling and reflecting on the torus leaves a map's spatial
    autocorrelation intact while destroying its alignment with any other
    map, which is exactly the null needed for "do these two maps share
    spatial structure?". The set is fixed and complete, so the resulting
    p-value is exact and needs no random seed.
    """
    base = np.arange(side * side).reshape(side, side)
    perms = []
    for k in range(4):
        rot = np.rot90(base, k)
        for grid in (rot, np.fliplr(rot)):
            for dy in range(side):
                for dx in range(side):
                    perms.append(np.roll(np.roll(grid, dy, 0), dx, 1).ravel())
    return np.asarray(perms)


_PERM_CACHE = {}


def perm_for(n_positions: int):
    """Cached permutation index set, or None for a non-square map."""
    side = int(round(n_positions ** 0.5))
    if side * side != n_positions:
        return None
    if side not in _PERM_CACHE:
        _PERM_CACHE[side] = build_perm(side)
    return _PERM_CACHE[side]


def rho(a, b):
    """Spearman over positions; None when either map is constant."""
    za, zb = zrank(a), zrank(b)
    if za is None or zb is None:
        return None
    return float(za @ zb)


def rho_spatial(a, b):
    """(rho, p_spatial, null_sd, n_transforms) or None.

    p is the two-sided exact permutation p-value: the fraction of the
    transform set whose |rho| reaches the observed |rho|. The identity is
    in the set, so p >= 1/n_transforms and can never be reported as 0.
    """
    za, zb = zrank(a), zrank(b)
    if za is None or zb is None:
        return None
    observed = float(za @ zb)
    perm = perm_for(za.size)
    if perm is None:
        return observed, None, None, 0
    nulls = zb[perm] @ za
    p = float(np.mean(np.abs(nulls) >= abs(observed) - 1e-12))
    return observed, p, float(nulls.std()), int(perm.shape[0])


# ── finite-sample null for the concentration statistics ──────────────────────

_NULL_CACHE = {}


def concentration_null(mass: float, n_images: int, n_positions: int,
                       sims: int = NULL_SIMS, seed: int = NULL_SEED) -> dict:
    """{stat: (null_mean, null_sd)} under a SPATIALLY UNIFORM ground truth
    with the same mass and n_images: every position's count is an
    independent Binomial(n_images, mass/n_positions).

    This is the floor that finite sampling alone produces; the raw
    statistics are meaningless across members without it, because the
    floor moves with mass (Gini ~0.017 at mass 20, ~0.49 at mass 0.02).
    """
    key = (round(float(mass), 9), int(n_images), int(n_positions), sims, seed)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    rng = np.random.RandomState(seed)
    p = float(mass) / n_positions
    draws = defaultdict(list)
    for _ in range(sims):
        counts = rng.binomial(n_images, p, size=n_positions)
        if counts.sum() == 0:
            continue
        conc = concentration_recompute(counts / float(n_images))
        for stat in NULLABLE:
            draws[stat].append(conc[stat])
    out = {stat: ((float(np.mean(v)), float(np.std(v))) if v else (None, None))
           for stat, v in draws.items()}
    for stat in NULLABLE:
        out.setdefault(stat, (None, None))
    _NULL_CACHE[key] = out
    return out


def position_rel_se(mass: float, n_images: int, n_positions: int):
    """Relative standard error of ONE position's frequency estimate under
    the uniform null — the honest measure of how noisy a map is (it is
    Monte-Carlo noise, not ties, that destroys a sparse map's ranking)."""
    p = float(mass) / n_positions
    if p <= 0:
        return None
    return float(((1.0 - p) / (n_images * p)) ** 0.5)


def head_mean_gate(phi: np.ndarray) -> np.ndarray:
    """sigmoid(phi) [L, H, N] -> mean over heads -> [L, N].

    sigmoid FIRST, then the arithmetic head mean — the ordering fixed in
    TASK-06B (analysis/gate_structure.py); the reverse pooling silently
    produced wrong gate numbers once already.
    """
    sig = 1.0 / (1.0 + np.exp(-phi.astype(np.float64)))
    return sig.mean(axis=1)


# ── row builders ─────────────────────────────────────────────────────────────

class Rows:
    def __init__(self):
        self.rows = []

    def add(self, question, arch, rec, basis, variant, subject, comparator,
            statistic, value, n, reference="", note=""):
        self.rows.append({
            "question": question, "arch": arch, "recipe_actual": rec,
            "map_basis": basis, "variant": variant, "subject": subject,
            "comparator": comparator, "statistic": statistic,
            "value": (round(value, 6) if isinstance(value, float)
                      else (MISSING if value is None else value)),
            "n": n, "reference": reference, "note": note})

    def add_correlation(self, question, arch, rec, basis, variant, subject,
                        comparator, a, b, reference, note="",
                        statistic="spearman", n_tests=1):
        """One correlation row plus its exact spatial-permutation p (and a
        Bonferroni-corrected p when the value was selected out of n_tests).
        Returns the rho, or None when a map is constant."""
        res = rho_spatial(a, b)
        if res is None:
            self.add(question, arch, rec, basis, variant, subject,
                     comparator, statistic, None, 0, reference,
                     (note + "; " if note else "") + "map spatially constant")
            return None
        value, p, null_sd, n_perm = res
        detail = (f"spatial-permutation p={p:.4f} over {n_perm} transforms, "
                  f"null sd={null_sd:.4f}" if p is not None
                  else "no permutation null (non-square map)")
        if n_tests > 1 and p is not None:
            detail += (f"; Bonferroni p={min(1.0, p * n_tests):.4f} "
                       f"for {n_tests} layers")
        self.add(question, arch, rec, basis, variant, subject, comparator,
                 statistic, value, len(a), reference,
                 (note + "; " if note else "") + detail)
        if p is not None:
            self.add(question, arch, rec, basis, variant, subject,
                     comparator, f"{statistic}_p_spatial", p, n_perm,
                     "exact permutation, dihedral x torus rolls", note)
            self.add(question, arch, rec, basis, variant, subject,
                     comparator, f"{statistic}_null_sd", null_sd, n_perm,
                     f"iid reference sd={1 / (len(a) - 1) ** 0.5:.4f} "
                     f"(too generous for smooth maps)", note)
        return value

    def summarize(self, question, arch, rec, basis, variant, subject,
                  comparator, values, reference, note=""):
        """mean/min/max over a set of pair correlations, n = #defined."""
        defined = [v for v in values if v is not None]
        if not defined:
            self.add(question, arch, rec, basis, variant, subject, comparator,
                     "spearman_mean", None, 0, reference,
                     note or "no defined pairs")
            return
        for stat, val in (("spearman_mean", float(np.mean(defined))),
                          ("spearman_min", float(np.min(defined))),
                          ("spearman_max", float(np.max(defined)))):
            self.add(question, arch, rec, basis, variant, subject, comparator,
                     stat, val, len(defined), reference, note)


def excess_table(members) -> dict:
    """(cell, variant, tag, basis) -> {stat: excess over the finite-sample
    null}. Computed once so Q1 and Q4 report the same numbers."""
    out = {}
    for (arch, rec, variant), found in sorted(members.items()):
        for basis in BASES:
            for tag, addr, _ in found:
                conc = addr[f"concentration_{basis}"]
                null = concentration_null(conc["total_mass"],
                                          addr["n_images"],
                                          addr["n_positions"])
                out[(arch, rec, variant, tag, basis)] = {
                    stat: (float(conc[stat]) - null[stat][0]
                           if null[stat][0] is not None else None)
                    for stat in NULLABLE}
    return out


def q1_concentration(rows: Rows, members, excess):
    """Every repeat's concentration: observed, its finite-sample null, and
    the excess over that null (the only cross-member comparable form)."""
    for (arch, rec, variant), found in sorted(members.items()):
        for basis in BASES:
            per_stat = defaultdict(list)
            for tag, addr, _ in found:
                conc = addr[f"concentration_{basis}"]
                n_pos, n_img = addr["n_positions"], addr["n_images"]
                mass = float(conc["total_mass"])
                null = concentration_null(mass, n_img, n_pos)
                exc = excess[(arch, rec, variant, tag, basis)]

                # noise level: it is per-position Monte-Carlo error, NOT
                # ties, that makes a sparse map's ranking meaningless
                rel_se = position_rel_se(mass, n_img, n_pos)
                rows.add("Q1_concentration", arch, rec, basis, variant, tag,
                         "noise", "position_rel_se", rel_se, n_img,
                         f"flagged above {NOISY_REL_SE}",
                         "relative standard error of ONE position's "
                         "frequency under the uniform null")
                f = np.asarray(addr[f"freq_{basis}"], dtype=np.float64)
                rows.add("Q1_concentration", arch, rec, basis, variant, tag,
                         "noise", "n_distinct_values", int(np.unique(f).size),
                         n_pos, f"{n_pos} positions",
                         "context only - mid-rank spread stays near maximal "
                         "even with large tie groups")

                for s in CONC_STATS:
                    rows.add("Q1_concentration", arch, rec, basis, variant,
                             tag, "uniform", s, float(conc[s]), n_pos,
                             uniform_reference(s, n_pos))
                    per_stat[s].append(float(conc[s]))
                    if s not in NULLABLE:
                        continue
                    nm, nsd = null[s]
                    rows.add("Q1_concentration", arch, rec, basis, variant,
                             tag, "finite-sample-null", f"{s}_null", nm,
                             NULL_SIMS,
                             f"uniform ground truth at mass={mass:.4f}",
                             f"binomial null, sd={nsd:.6f}, "
                             f"sims={NULL_SIMS}, seed={NULL_SEED}")
                    rows.add("Q1_concentration", arch, rec, basis, variant,
                             tag, "finite-sample-null", f"{s}_excess",
                             exc[s], n_pos, "zero = indistinguishable from "
                             "a uniform map at this mass",
                             "observed minus null - the cross-member "
                             "comparable form")
                    per_stat[f"{s}_excess"].append(exc[s])
                rows.add("Q1_concentration", arch, rec, basis, variant, tag,
                         "uniform", "top10_positions",
                         " ".join(str(p) for p in conc["top10_positions"]),
                         n_pos, "", "highest-frequency positions")
            for s, vals in per_stat.items():
                clean = [v for v in vals if v is not None]
                rows.add("Q1_concentration", arch, rec, basis, variant,
                         "MEAN", "uniform", s,
                         float(np.mean(clean)) if clean else None,
                         len(clean),
                         uniform_reference(s, found[0][1]["n_positions"]),
                         "mean over repeats")


def border_rings(freq) -> list:
    """Mean frequency per ring of constant Chebyshev distance from the grid
    border: ring 0 = the outermost row/col, ring 1 = one patch inside, ...
    Characterizes the GEOMETRY of whatever concentration Q1 measures."""
    f = np.asarray(freq, dtype=np.float64)
    side = int(round(f.size ** 0.5))
    g = f.reshape(side, side)
    idx = np.arange(side)
    dist = np.minimum(
        np.minimum(idx[:, None], side - 1 - idx[:, None]),
        np.minimum(idx[None, :], side - 1 - idx[None, :]))
    return [float(g[dist == k].mean()) for k in range(side // 2)]


def q1b_geometry(rows: Rows, members):
    """Where the sink mass sits relative to the image border."""
    for (arch, rec, variant), found in sorted(members.items()):
        for basis in BASES:
            for tag, addr, _ in found:
                prof = border_rings(addr[f"freq_{basis}"])
                for k, v in enumerate(prof):
                    rows.add("Q1b_geometry", arch, rec, basis, variant, tag,
                             "border-ring", f"ring{k}_mean_freq", v,
                             addr["n_positions"],
                             "flat profile = no border structure",
                             "ring 0 = outermost row/col")
                rows.add("Q1b_geometry", arch, rec, basis, variant, tag,
                         "border-ring", "peak_ring",
                         int(np.argmax(prof)), addr["n_positions"],
                         "flat profile = no border structure",
                         f"peak_mean_freq={max(prof):.6f}")


def q2_seed_stability(rows: Rows, members):
    for (arch, rec, variant), found in sorted(members.items()):
        if len(found) < 2:
            for basis in BASES:
                rows.add("Q2_seed_stability", arch, rec, basis, variant,
                         "all-pairs", "zero", "spearman_mean", None, 0,
                         "zero = no shared structure",
                         f"only {len(found)} repeat(s) — no pair exists")
            continue
        for basis in BASES:
            vals = []
            for (ta, aa, _), (tb, ab, _) in combinations(found, 2):
                vals.append(rows.add_correlation(
                    "Q2_seed_stability", arch, rec, basis, variant,
                    f"{ta} vs {tb}", "spatial-null",
                    aa[f"freq_{basis}"], ab[f"freq_{basis}"],
                    "zero = no shared structure"))
            rows.summarize("Q2_seed_stability", arch, rec, basis, variant,
                           "all-pairs", "spatial-null", vals,
                           "zero = no shared structure")


def q3_recipe_stability(rows: Rows, members):
    """S/mixup baseline maps vs S/true-nomix baseline maps (all cross pairs)."""
    a = members.get(("vit_small", "mixup", "baseline"), [])
    b = members.get(("vit_small", "nomix", "baseline"), [])
    if not a or not b:
        for basis in BASES:
            rows.add("Q3_recipe_stability", "vit_small", "mixup|nomix", basis,
                     "baseline", "all-cross-pairs", "zero", "spearman_mean",
                     None, 0, "zero = no shared structure",
                     "one side has no members")
        return
    for basis in BASES:
        vals = []
        for ta, aa, _ in a:
            for tb, ab, _ in b:
                vals.append(rows.add_correlation(
                    "Q3_recipe_stability", "vit_small", "mixup|nomix", basis,
                    "baseline", f"mixup:{ta} vs nomix:{tb}", "spatial-null",
                    aa[f"freq_{basis}"], ab[f"freq_{basis}"],
                    "zero = no shared structure",
                    note="the two cells use DIFFERENT canon taus"
                    if basis == "canon" else ""))
        rows.summarize("Q3_recipe_stability", "vit_small", "mixup|nomix",
                       basis, "baseline", "all-cross-pairs", "spatial-null",
                       vals, "zero = no shared structure")


def q4_relocation(rows: Rows, members, excess):
    """variant-vs-baseline map correlation + concentration change.

    BOTH are paired by provenance tag. The concentration deltas are
    differences of EXCESS-over-null values, because the raw statistics sit
    on a mass-dependent floor and the variants change the mass by up to
    two orders of magnitude — differencing raw values inverts the sign of
    the registers result.
    """
    for arch, rec in CELLS:
        base = members.get((arch, rec, "baseline"), [])
        if not base:
            continue
        base_by_tag = {tag: addr for tag, addr, _ in base}
        for variant in ("registers", "saga"):
            found = members.get((arch, rec, variant), [])
            if not found:
                continue
            for basis in BASES:
                vals, paired_tags = [], []
                for tag, addr, _ in found:
                    if tag not in base_by_tag:
                        rows.add("Q4_relocation", arch, rec, basis, variant,
                                 tag, "baseline-matched", "spearman", None, 0,
                                 "1.0 = identical address",
                                 "no same-provenance baseline - EXCLUDED "
                                 "from the paired summary")
                        continue
                    paired_tags.append(tag)
                    vals.append(rows.add_correlation(
                        "Q4_relocation", arch, rec, basis, variant, tag,
                        "baseline-matched", addr[f"freq_{basis}"],
                        base_by_tag[tag][f"freq_{basis}"],
                        "1.0 = identical address, zero = relocated"))
                rows.summarize("Q4_relocation", arch, rec, basis, variant,
                               "paired", "baseline-matched", vals,
                               "1.0 = identical address, zero = relocated",
                               note=f"paired by provenance: "
                                    f"{', '.join(paired_tags)}")

                # paired concentration deltas, on EXCESS values
                for s in NULLABLE:
                    diffs = []
                    for tag in paired_tags:
                        ve = excess[(arch, rec, variant, tag, basis)][s]
                        be = excess[(arch, rec, "baseline", tag, basis)][s]
                        if ve is None or be is None:
                            continue
                        diffs.append(ve - be)
                        rows.add("Q4_relocation", arch, rec, basis, variant,
                                 tag, "baseline-matched",
                                 f"delta_{s}_excess", ve - be, 1,
                                 f"baseline excess={be:.6f}",
                                 f"{variant} excess={ve:.6f}")
                    direction = ("<0 = more concentrated"
                                 if s.startswith("entropy")
                                 else ">0 = more concentrated")
                    rows.add("Q4_relocation", arch, rec, basis, variant,
                             "MEAN", "baseline-matched",
                             f"delta_{s}_excess",
                             float(np.mean(diffs)) if diffs else None,
                             len(diffs),
                             "excess-over-null differences, paired",
                             direction)
                # mass has no null to subtract; it IS the conditioning value
                diffs = []
                for tag in paired_tags:
                    vm = found_by_tag(found, tag)[f"concentration_{basis}"][
                        "total_mass"]
                    bm = base_by_tag[tag][f"concentration_{basis}"][
                        "total_mass"]
                    diffs.append(float(vm) - float(bm))
                rows.add("Q4_relocation", arch, rec, basis, variant, "MEAN",
                         "baseline-matched", "delta_total_mass",
                         float(np.mean(diffs)) if diffs else None,
                         len(diffs), "paired difference of mean sinks/image",
                         "negative = fewer sinks")


def found_by_tag(found, tag):
    for t, addr, _ in found:
        if t == tag:
            return addr
    raise KeyError(tag)


def q5_gate_address(rows: Rows, members, profiles: dict):
    """Per-layer Spearman of sigmoid(phi) against the address maps."""
    for arch, rec in CELLS:
        base = members.get((arch, rec, "baseline"), [])
        saga = members.get((arch, rec, "saga"), [])
        if not saga:
            continue
        base_by_tag = {tag: addr for tag, addr, _ in base}
        for basis in BASES:
            cellmean = (np.mean([a[f"freq_{basis}"]
                                 for a in base_by_tag.values()], axis=0)
                        if base_by_tag else None)
            for tag, addr, gate_path in saga:
                label = f"{arch}|{rec}|saga|{tag}"
                if gate_path is None or not gate_path.exists():
                    rows.add("Q5_gate_address", arch, rec, basis, "saga", tag,
                             "baseline-matched", "spearman_layer_absmax",
                             None, 0, "zero = gate ignores the address",
                             "no gate dump for this run")
                    continue
                gate = head_mean_gate(np.load(gate_path)["phi"])
                # amplitude context: a rank correlation is scale-free, so
                # the reader needs the gate's spatial spread to judge how
                # much absolute modulation that rho corresponds to.
                # Basis-independent -> emitted once, under map_basis '-'.
                if basis == BASES[0]:
                    for li in range(gate.shape[0]):
                        rows.add("Q5_gate_address", arch, rec, "-", "saga",
                                 tag, "-", f"gate_spatial_std_layer{li:02d}",
                                 float(gate[li].std()), gate.shape[1],
                                 "0 = spatially constant layer",
                                 f"mean={gate[li].mean():.5f}, "
                                 f"range={np.ptp(gate[li]):.6f}")
                targets = {}
                if tag in base_by_tag:
                    targets["baseline-matched"] = base_by_tag[tag][f"freq_{basis}"]
                else:
                    # never let an unpaired run vanish from Q5
                    rows.add("Q5_gate_address", arch, rec, basis, "saga", tag,
                             "baseline-matched", "spearman_layer_absmax",
                             None, 0, "zero = gate ignores the address",
                             "no same-provenance baseline map")
                if cellmean is not None:
                    targets["baseline-cellmean"] = cellmean
                targets["own"] = addr[f"freq_{basis}"]
                for comparator, target in targets.items():
                    per_layer, n_const = [], 0
                    for li in range(gate.shape[0]):
                        v = rho(gate[li], target)
                        if v is None:
                            n_const += 1
                            per_layer.append(np.nan)
                            continue
                        per_layer.append(v)
                        if comparator != "baseline-cellmean":
                            rows.add_correlation(
                                "Q5_gate_address", arch, rec, basis, "saga",
                                tag, comparator, gate[li], target,
                                "zero = gate ignores the address",
                                statistic=f"spearman_layer{li:02d}")
                    prof = np.asarray(per_layer, dtype=np.float64)
                    defined = prof[~np.isnan(prof)]
                    profiles[f"{basis}__{comparator}__{label}"] = prof
                    if defined.size == 0:
                        rows.add("Q5_gate_address", arch, rec, basis, "saga",
                                 tag, comparator, "spearman_layer_absmax",
                                 None, 0, "zero = gate ignores the address",
                                 "every layer spatially constant")
                        continue
                    imax = int(np.nanargmax(prof))
                    imin = int(np.nanargmin(prof))
                    iabs = int(np.nanargmax(np.abs(prof)))
                    # each extremum is an argmax over the defined layers, so
                    # it carries a Bonferroni correction for that selection
                    for stat, li in (("spearman_layer_max", imax),
                                     ("spearman_layer_min", imin),
                                     ("spearman_layer_absmax", iabs)):
                        rows.add_correlation(
                            "Q5_gate_address", arch, rec, basis, "saga", tag,
                            comparator, gate[li], target,
                            "zero = gate ignores the address",
                            note=f"layer={li}", statistic=stat,
                            n_tests=int(defined.size))
                    rows.add("Q5_gate_address", arch, rec, basis, "saga", tag,
                             comparator, "spearman_layer_mean",
                             float(defined.mean()), int(defined.size),
                             "zero = gate ignores the address",
                             f"n_layers_defined={defined.size} of "
                             f"{gate.shape[0]} ({n_const} constant); signed "
                             f"mean over layers whose signs differ, so "
                             f"cancellation pulls it toward zero")
                    rows.add("Q5_gate_address", arch, rec, basis, "saga", tag,
                             comparator, "spearman_layer_absmean",
                             float(np.abs(defined).mean()),
                             int(defined.size),
                             "no sign cancellation",
                             "mean |rho| over defined layers")


# ── main ─────────────────────────────────────────────────────────────────────

def build(manifest_path: Path):
    manifest = load_manifest(manifest_path)
    members, gaps = collect(manifest)
    rows = Rows()
    profiles = {}
    excess = excess_table(members)
    q1_concentration(rows, members, excess)
    q1b_geometry(rows, members)
    q2_seed_stability(rows, members)
    q3_recipe_stability(rows, members)
    q4_relocation(rows, members, excess)
    q5_gate_address(rows, members, profiles)
    return rows.rows, members, profiles, gaps


def save_npz(out: Path, members, profiles):
    arrays, keys = {}, []
    for (arch, rec, variant), found in sorted(members.items()):
        for tag, addr, gate_path in found:
            key = f"{arch}|{rec}|{variant}|{tag}"
            keys.append(key)
            for basis in BASES:
                arrays[f"freq_{basis}__{key}"] = np.asarray(
                    addr[f"freq_{basis}"], dtype=np.float64)
            arrays[f"mean_norm__{key}"] = np.asarray(addr["mean_norm"],
                                                     dtype=np.float64)
            if gate_path is not None and gate_path.exists():
                arrays[f"gate__{key}"] = head_mean_gate(
                    np.load(gate_path)["phi"])
    for name, prof in profiles.items():
        arrays[f"profile__{name}"] = prof
    arrays["member_keys"] = np.array(keys)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return len(keys)


def main():
    parser = argparse.ArgumentParser(
        description="TASK-07 C1 sink-address analysis.")
    parser.add_argument("--manifest",
                        default="results/legacy/checkpoint_manifest.csv")
    parser.add_argument("--out", default="results/tables/sink_address.csv")
    parser.add_argument("--npz", default="results/figures_data/Faddr.npz")
    args = parser.parse_args()

    rows, members, profiles, gaps = build(Path(args.manifest))
    absent = absent_members()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_members = save_npz(Path(args.npz), members, profiles)

    print(f"wrote {out}: {len(rows)} rows")
    print(f"wrote {args.npz}: {n_members} members, "
          f"{len(profiles)} gate profiles")
    print(f"members: {sum(len(v) for v in members.values())} maps in "
          f"{len(members)} (cell, variant) groups")
    if gaps:
        print(f"GAPS ({len(gaps)}) — present but unusable:")
        for g in gaps:
            print(f"  {g}")
    else:
        print("no gaps: every present map is sha-matched and complete")
    if absent:
        print(f"not trained ({len(absent)}) - expected, not a gap:")
        for a in absent:
            print(f"  {a}")

    for arch, rec in CELLS:
        for basis in ("canon",):
            conc = [r for r in rows
                    if r["question"] == "Q1_concentration"
                    and r["arch"] == arch and r["recipe_actual"] == rec
                    and r["map_basis"] == basis and r["variant"] == "baseline"
                    and r["subject"] == "MEAN"
                    and r["statistic"] == "entropy_normalized"]
            seed = [r for r in rows
                    if r["question"] == "Q2_seed_stability"
                    and r["arch"] == arch and r["recipe_actual"] == rec
                    and r["map_basis"] == basis and r["variant"] == "baseline"
                    and r["subject"] == "all-pairs"
                    and r["statistic"] == "spearman_mean"]
            if conc:
                print(f"  {arch}|{rec} [canon] baseline H/H_uniform="
                      f"{conc[0]['value']}, seed-stability rho="
                      f"{seed[0]['value'] if seed else MISSING} "
                      f"(n={seed[0]['n'] if seed else 0} pairs)")


if __name__ == "__main__":
    main()
