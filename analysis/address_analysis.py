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

    python analysis/address_analysis.py
"""

import argparse
import csv
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.build_pooled_tables import (CELLS, E2R_MEMBERS, LEGACY_MEMBERS,
                                          VARIANTS)

MISSING = "MISSING"
BASES = ("canon", "mad")

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
    """The tool's stored block must match an independent recompute."""
    stored = addr[f"concentration_{basis}"]
    got = concentration_recompute(addr[f"freq_{basis}"])
    for key, value in got.items():
        ref = stored.get(key)
        if ref is None:
            raise SystemExit(f"{label} [{basis}]: stored concentration is "
                             f"missing {key!r}")
        if not np.isfinite(value) and not np.isfinite(ref):
            continue
        if abs(value - ref) > tol:
            raise SystemExit(
                f"{label} [{basis}]: stored {key}={ref!r} disagrees with "
                f"independent recompute {value!r}")


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

def rho(a, b):
    """Spearman over positions; None when either map is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = spearmanr(a, b).statistic
    return None if np.isnan(v) else float(v)


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


def q1_concentration(rows: Rows, members):
    """Concentration of every repeat's freq map (all variants; Q1 reads the
    baseline ones, Q4 reuses the rest for its deltas)."""
    stats = ["entropy_bits", "entropy_normalized", "gini", "top5_share",
             "top20_share", "total_mass"]
    for (arch, rec, variant), found in sorted(members.items()):
        for basis in BASES:
            per_stat = {s: [] for s in stats}
            for tag, addr, _ in found:
                conc = addr[f"concentration_{basis}"]
                n_pos = addr["n_positions"]
                # rank-correlation health: a map with few distinct values is
                # mostly ties, so any Spearman computed on it is degenerate.
                # Recorded as numbers so the caveat is never just prose.
                f = np.asarray(addr[f"freq_{basis}"], dtype=np.float64)
                rows.add("Q1_concentration", arch, rec, basis, variant, tag,
                         "ties", "n_distinct_values", int(np.unique(f).size),
                         n_pos, f"{n_pos} positions",
                         "few distinct values = ties dominate any Spearman")
                rows.add("Q1_concentration", arch, rec, basis, variant, tag,
                         "ties", "zero_fraction",
                         float(np.mean(f == 0.0)), n_pos, "0.0 = no empty "
                         "positions", "fraction of positions never a sink")
                for s in stats:
                    rows.add("Q1_concentration", arch, rec, basis, variant,
                             tag, "uniform", s, float(conc[s]), n_pos,
                             uniform_reference(s, n_pos))
                    per_stat[s].append(float(conc[s]))
                rows.add("Q1_concentration", arch, rec, basis, variant, tag,
                         "uniform", "top10_positions",
                         " ".join(str(p) for p in conc["top10_positions"]),
                         n_pos, "", "highest-frequency positions")
            for s in stats:
                rows.add("Q1_concentration", arch, rec, basis, variant,
                         "MEAN", "uniform", s,
                         float(np.mean(per_stat[s])), len(found),
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
                v = rho(aa[f"freq_{basis}"], ab[f"freq_{basis}"])
                vals.append(v)
                rows.add("Q2_seed_stability", arch, rec, basis, variant,
                         f"{ta} vs {tb}", "zero", "spearman", v,
                         aa["n_positions"], "zero = no shared structure")
            rows.summarize("Q2_seed_stability", arch, rec, basis, variant,
                           "all-pairs", "zero", vals,
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
                v = rho(aa[f"freq_{basis}"], ab[f"freq_{basis}"])
                vals.append(v)
                rows.add("Q3_recipe_stability", "vit_small", "mixup|nomix",
                         basis, "baseline", f"mixup:{ta} vs nomix:{tb}",
                         "zero", "spearman", v, aa["n_positions"],
                         "zero = no shared structure")
        rows.summarize("Q3_recipe_stability", "vit_small", "mixup|nomix",
                       basis, "baseline", "all-cross-pairs", "zero", vals,
                       "zero = no shared structure")


def q4_relocation(rows: Rows, members):
    """variant-vs-baseline map correlation + concentration change."""
    conc_stats = ["entropy_normalized", "gini", "top5_share", "top20_share",
                  "total_mass"]
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
                vals = []
                for tag, addr, _ in found:
                    if tag not in base_by_tag:
                        rows.add("Q4_relocation", arch, rec, basis, variant,
                                 tag, "baseline-matched", "spearman", None, 0,
                                 "1.0 = identical address",
                                 "no same-provenance baseline")
                        continue
                    v = rho(addr[f"freq_{basis}"],
                            base_by_tag[tag][f"freq_{basis}"])
                    vals.append(v)
                    rows.add("Q4_relocation", arch, rec, basis, variant, tag,
                             "baseline-matched", "spearman", v,
                             addr["n_positions"],
                             "1.0 = identical address, zero = relocated")
                rows.summarize("Q4_relocation", arch, rec, basis, variant,
                               "paired", "baseline-matched", vals,
                               "1.0 = identical address, zero = relocated")
                for s in conc_stats:
                    bm = float(np.mean([a[f"concentration_{basis}"][s]
                                        for a in base_by_tag.values()]))
                    vm = float(np.mean([a[f"concentration_{basis}"][s]
                                        for _, a, _ in found]))
                    rows.add("Q4_relocation", arch, rec, basis, variant,
                             "MEAN", "baseline-cellmean", f"delta_{s}",
                             vm - bm, len(found),
                             f"baseline mean={bm:.6f}",
                             f"{variant} mean={vm:.6f}; >0 = more concentrated"
                             if s != "entropy_normalized"
                             else f"{variant} mean={vm:.6f}; <0 = more "
                                  f"concentrated")


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
                            rows.add("Q5_gate_address", arch, rec, basis,
                                     "saga", tag, comparator,
                                     f"spearman_layer{li:02d}", v,
                                     len(target),
                                     "zero = gate ignores the address")
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
                    for stat, li, val in (
                            ("spearman_layer_max", imax, prof[imax]),
                            ("spearman_layer_min", imin, prof[imin]),
                            ("spearman_layer_absmax", iabs, prof[iabs])):
                        rows.add("Q5_gate_address", arch, rec, basis, "saga",
                                 tag, comparator, stat, float(val),
                                 len(target),
                                 "zero = gate ignores the address",
                                 f"layer={li}")
                    rows.add("Q5_gate_address", arch, rec, basis, "saga", tag,
                             comparator, "spearman_layer_mean",
                             float(defined.mean()), int(defined.size),
                             "zero = gate ignores the address",
                             f"n_layers_defined={defined.size} of "
                             f"{gate.shape[0]} ({n_const} constant)")


# ── main ─────────────────────────────────────────────────────────────────────

def build(manifest_path: Path):
    manifest = load_manifest(manifest_path)
    members, gaps = collect(manifest)
    rows = Rows()
    profiles = {}
    q1_concentration(rows, members)
    q1b_geometry(rows, members)
    q2_seed_stability(rows, members)
    q3_recipe_stability(rows, members)
    q4_relocation(rows, members)
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
