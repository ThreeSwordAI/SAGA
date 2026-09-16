"""
saga/frozen/prevalence.py
=========================
The prevalence-map contract — TASK A / I1 A1.

ONE representation of "how often does each patch position exceed its
threshold", shared by:

  * TASK I2's committed `maps.npz` (per-position exceedance COUNTS at
    `s11_out`, `hist`, `s12_post_norm`, on calibration and evaluation),
  * TASK I1's `maps_<stage>.npz` (the same, at the block-input stages, with
    per-image indicators beside them),
  * TASK-07's committed `*_addr.json` maps (the historical last-block
    address maps), and
  * TASK I6's external public checkpoints, whose grids are 16x16 rather
    than 14x14.

and readable by `analysis/address_analysis.py` WITHOUT changing that module.
Its ring profile, its finite-sample concentration reference and its
1568-transform spatial permutation reference are IMPORTED here and never
re-derived — the numbers TASK-07 published and the numbers I1 publishes have
to come out of the same code, or the comparison between them is not a
comparison.

  `address_analysis.border_distance_map(side)`, `ring_indices(side, k)`,
  `border_rings(freq)`, `perm_for(n)`, `rho_spatial(a, b)`,
  `concentration_recompute(freq)`, `concentration_null(mass, n_img, n_pos)`
  already take their grid size as an argument or infer it from the map, so
  the 16x16 grids of I6 need NO change to that module. The optional `side`
  argument TASK A §6 allows for is therefore not added: there is nothing
  hard-coded to parameterise. `tests/test_I1_spatial.py` pins both
  `ring_indices(14, k)` and `ring_indices(16, k)` against an independent
  derivation so that stays true.

SELECTION AND REPORTING NEVER MIX
---------------------------------
Every mask-building function in this module calls `assert_selection_split`
first, which accepts the DISCOVERY split and nothing else. It is a positive
allow-list, not a denial of the evaluation sha: a map from a split nobody
listed is refused too. D5's masks are built from discovery data and the
paper's numbers come from evaluation data, and the only way that stays true
under maintenance is if the code refuses the alternative.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.address_analysis import (border_distance_map, border_rings,
                                       concentration_null,
                                       concentration_recompute, ring_indices,
                                       rho_spatial)
from saga.frozen.diag import MAP_BASES

MISSING = "MISSING"

#: The `*_addr.json` schema string this contract round-trips.
ADDR_SCHEMA = "sink_address_v1"

#: sha256 of the DISCOVERY split — the only split any mask may be built
#: from (`docs/LOCKED_ANALYSIS.md` §11; the file's own digest, as recorded
#: there). `tests/test_I1_spatial.py` pins it against the file.
DISCOVERY_SPLIT_SHA256 = (
    "0a686340c00846818a857cf4cbf472cbdc035e0a88c20d79483f7d9f463b69b1")

#: The REPORTING splits, named only so the refusal can say what went wrong.
#: The guard is the allow-list above; this map never widens it.
REPORTING_SPLIT_SHA256 = {
    "7fdf5f9f2ace98ef03a6267455daa1104b92b6689c510ab8b1e5420340f27014":
        "evaluation",
    "6b707eb39f934274a5ea753d613af54dc9aad01510808f246b70506a96003003":
        "calibration",
    "6a38d1000b32f2ca0c5bcfd1187cde86258b2d45a4b1e4b47cfd5580fe088ef3":
        "sub1k",
    "d2fc8b5b4c5ab40928c1ea861f21a3b561c0654a93479048773802568d338117":
        "sub2k",
}

#: LOCKED_ANALYSIS §5: 16 coordinates, 10 ring-matched controls.
PRIMARY_MASK_K = 16
N_CONTROL_MASKS = 10
#: The control seeds, declared here and nowhere else (TASK A §6).
CONTROL_SEEDS = tuple(range(200, 200 + N_CONTROL_MASKS))

#: Provenance a map cannot be validated without (I0 handoff §8.3).
REQUIRED_PROVENANCE = ("stage", "split_sha", "run_id", "ckpt_sha256",
                       "git_sha")


class PrevalenceError(ValueError):
    """A prevalence map that cannot be built, validated or used as asked."""


# ─────────────────────────────────────────────────────────────────────────────
# The map
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrevalenceMap:
    """One run's per-position exceedance frequency at one stage and basis.

    `freq[p]` is the fraction of the split's images in which patch position
    `p` exceeded the threshold — `counts[p] / n_images` exactly, and
    `counts` is carried whenever it is known because an integer count is
    the thing that was measured and a float frequency is a view of it.

    `threshold` is the absolute tau for `fixed_cal` and the literal MISSING
    for `mad`, whose threshold is per-image by construction and therefore
    not a property of the map.
    """

    freq: np.ndarray
    n_images: int
    n_exceed_total: int
    basis: str
    threshold: object
    stage: str
    split_sha: str
    run_id: str
    ckpt_sha256: str
    grid_side: int
    n_prefix: int
    git_sha: str
    split_name: str = MISSING
    condition_id: str = "native"
    counts: object = None
    #: The `*_addr.json` document this map was parsed out of, key order
    #: intact, so `to_addr_json` can hand it back unchanged (see there).
    source_document: object = None

    # ── construction ────────────────────────────────────────────────────────

    def __post_init__(self):
        self.freq = np.asarray(self.freq, dtype=np.float64).reshape(-1)
        if self.counts is not None:
            self.counts = np.asarray(self.counts, dtype=np.int64).reshape(-1)
        self.n_images = int(self.n_images)
        self.n_exceed_total = int(self.n_exceed_total)
        self.grid_side = int(self.grid_side)
        self.n_prefix = int(self.n_prefix)

    @property
    def n_positions(self) -> int:
        return int(self.freq.size)

    @property
    def mass(self) -> float:
        """Total mass = mean exceedances per image. The value every
        concentration statistic is conditioned on."""
        return float(self.freq.sum())

    def validate(self) -> "PrevalenceMap":
        """Refuse a map that cannot carry a number the paper prints."""
        f = self.freq
        if f.ndim != 1 or f.size == 0:
            raise PrevalenceError(
                f"freq must be a non-empty 1-D array, got shape {f.shape}")
        if not np.isfinite(f).all():
            raise PrevalenceError(
                f"freq carries {int((~np.isfinite(f)).sum())} non-finite "
                f"value(s) — a NaN here would be averaged into a table")
        if f.min() < 0.0 or f.max() > 1.0:
            raise PrevalenceError(
                f"freq must lie in [0, 1]; got [{f.min()!r}, {f.max()!r}]. A "
                f"frequency outside it means counts and n_images disagree")
        if self.grid_side * self.grid_side != f.size:
            raise PrevalenceError(
                f"grid_side={self.grid_side} implies "
                f"{self.grid_side ** 2} positions but freq has {f.size} — the "
                f"ring geometry and the permutation null both need a square "
                f"grid that matches")
        if self.n_images <= 0:
            raise PrevalenceError(f"n_images={self.n_images}, expected > 0")
        if self.n_prefix < 1:
            raise PrevalenceError(
                f"n_prefix={self.n_prefix}: the prefix count is read from the "
                f"model and is never zero (I0 audit bug B2)")
        if self.basis not in MAP_BASES:
            raise PrevalenceError(
                f"basis {self.basis!r} is not one of {MAP_BASES}")
        if self.basis == "fixed_cal" and not isinstance(
                self.threshold, (int, float)):
            raise PrevalenceError(
                f"basis 'fixed_cal' needs an absolute threshold, got "
                f"{self.threshold!r}")
        missing = [k for k in REQUIRED_PROVENANCE
                   if not str(getattr(self, k) or "").strip()
                   or getattr(self, k) == MISSING]
        if missing:
            raise PrevalenceError(
                f"provenance {missing} is missing — a number that cannot be "
                f"traced to a checkpoint and a split is not a number this "
                f"project reports (I0 handoff §8.3)")
        if self.counts is not None:
            if self.counts.size != f.size:
                raise PrevalenceError(
                    f"counts has {self.counts.size} positions, freq has "
                    f"{f.size}")
            if not np.allclose(self.counts / float(self.n_images), f,
                               rtol=0, atol=1e-12):
                raise PrevalenceError(
                    "freq is not counts / n_images — two derivations of the "
                    "same quantity disagree")
            if int(self.counts.sum()) != self.n_exceed_total:
                raise PrevalenceError(
                    f"n_exceed_total={self.n_exceed_total} but the counts sum "
                    f"to {int(self.counts.sum())}")
        return self

    # ── npz round trip ──────────────────────────────────────────────────────

    def _payload(self) -> dict:
        meta = {
            "n_images": self.n_images, "n_exceed_total": self.n_exceed_total,
            "basis": self.basis, "threshold": self.threshold,
            "stage": self.stage, "split_sha": self.split_sha,
            "split_name": self.split_name, "condition_id": self.condition_id,
            "run_id": self.run_id, "ckpt_sha256": self.ckpt_sha256,
            "grid_side": self.grid_side, "n_prefix": self.n_prefix,
            "git_sha": self.git_sha, "schema": "prevalence_map_v1",
        }
        payload = {"freq": self.freq,
                   "meta_json": np.array(json.dumps(
                       meta, sort_keys=True, separators=(",", ":"),
                       default=str))}
        if self.counts is not None:
            payload["counts"] = self.counts
        return payload

    def save(self, path) -> Path:
        """Write atomically, the way every artifact in this package is."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        with open(tmp, "wb") as f:
            np.savez(f, **self._payload())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path) -> "PrevalenceMap":
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            counts = z["counts"] if "counts" in z.files else None
            freq = z["freq"]
        return cls(freq=freq, counts=counts,
                   **{k: meta[k] for k in (
                       "n_images", "n_exceed_total", "basis", "threshold",
                       "stage", "split_sha", "split_name", "condition_id",
                       "run_id", "ckpt_sha256", "grid_side", "n_prefix",
                       "git_sha")}).validate()


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def grid_side_of(n_positions: int) -> int:
    """The side of a square patch grid, refusing a non-square token count."""
    side = int(round(float(n_positions) ** 0.5))
    if side * side != int(n_positions):
        raise PrevalenceError(
            f"{n_positions} patch positions is not a square grid — the ring "
            f"geometry and the torus-roll permutation null are both defined "
            f"on a square grid only")
    return side


def from_counts(counts, *, n_images, basis, threshold, stage, split_sha,
                run_id, ckpt_sha256, n_prefix, git_sha, split_name=MISSING,
                condition_id="native") -> PrevalenceMap:
    """A map from the per-position exceedance COUNTS a sweep accumulates."""
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    n_images = int(n_images)
    if n_images <= 0:
        raise PrevalenceError(f"n_images={n_images}, expected > 0")
    return PrevalenceMap(
        freq=counts / float(n_images), counts=counts, n_images=n_images,
        n_exceed_total=int(counts.sum()), basis=basis, threshold=threshold,
        stage=stage, split_sha=split_sha, split_name=split_name,
        condition_id=condition_id, run_id=run_id, ckpt_sha256=ckpt_sha256,
        grid_side=grid_side_of(counts.size), n_prefix=n_prefix,
        git_sha=git_sha).validate()


def i2_map_keys(npz) -> list:
    """The `(condition, stage, basis)` triples a TASK I2 `maps.npz` holds."""
    return sorted(tuple(k.split("|")) for k in npz.files
                  if "|" in k and not k.startswith("n_images__"))


def from_i2_maps_npz(path, *, condition_id, stage, basis,
                     n_prefix=1) -> PrevalenceMap:
    """One map out of TASK I2's committed `maps.npz`.

    I2 stores COUNTS keyed `"<condition>|<stage>|<basis>"` with a sibling
    `"n_images__<same>"`, and its provenance in a single `meta_json` string
    (`saga/frozen/diag.py::MapAccumulator`). This is the only place that
    layout is decoded, so I1 consumes I2 rather than recomputing it.

    `n_prefix` is not recorded in that file; it is supplied by the caller
    from the run's `records`/`diag` provenance and defaults to 1, which is
    every cohort member except the register models.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        key = f"{condition_id}|{stage}|{basis}"
        if key not in z.files:
            raise PrevalenceError(
                f"{Path(path).name} has no map {key!r}; it holds "
                f"{sorted(k for k in z.files if '|' in k)}")
        counts = z[key]
        n_images = int(z[f"n_images__{key}"])
        meta = json.loads(str(z["meta_json"]))
    tau = (meta.get("tau_cal", {}) or {}).get(stage, MISSING)
    return from_counts(
        counts, n_images=n_images, basis=basis,
        threshold=(float(tau) if basis == "fixed_cal"
                   and isinstance(tau, (int, float)) else MISSING),
        stage=stage, split_sha=meta["split_sha256"],
        split_name=meta.get("split_name", MISSING), condition_id=condition_id,
        run_id=meta["run_id"], ckpt_sha256=meta["ckpt_sha256"],
        n_prefix=n_prefix, git_sha=meta["git_sha"])


def cell_mean_map(maps) -> np.ndarray:
    """Mean `freq` over a cell's repeats, EQUAL WEIGHT PER CHECKPOINT.

    Not a pooled frequency: a checkpoint evaluated on more images would
    otherwise count for more, and the unit of analysis is the checkpoint
    (`docs/LOCKED_ANALYSIS.md` §8).
    """
    maps = list(maps)
    if not maps:
        raise PrevalenceError("no maps to average")
    sizes = {m.n_positions for m in maps}
    if len(sizes) != 1:
        raise PrevalenceError(f"maps have different grids: {sorted(sizes)}")
    stages = {m.stage for m in maps}
    bases = {m.basis for m in maps}
    if len(stages) != 1 or len(bases) != 1:
        raise PrevalenceError(
            f"a cell mean mixes stage(s) {sorted(stages)} and bases "
            f"{sorted(bases)} — it is defined at one stage and one basis")
    return np.mean([m.freq for m in maps], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# The selection guard — written before the mask builders it protects
# ─────────────────────────────────────────────────────────────────────────────

def assert_selection_split(obj, *, what="mask") -> str:
    """Refuse to select anything from data that is not the DISCOVERY split.

    `obj` is a `PrevalenceMap`, a split sha string, or anything exposing
    `split_sha`. The check is an ALLOW-LIST: only the discovery sha passes,
    so a map from a split that nobody has named is refused as firmly as an
    evaluation map is. Returns the sha on success.

    TASK A §2/§12 and `docs/LOCKED_ANALYSIS.md` §11: masks come from
    discovery only. Every table the paper shows comes from evaluation, and
    a coordinate chosen by looking at evaluation data would make the
    perturbation result circular.
    """
    sha = getattr(obj, "split_sha", obj)
    if isinstance(sha, (bytes, bytearray)):
        sha = sha.decode()
    sha = str(sha or "").strip().lower()
    if sha == DISCOVERY_SPLIT_SHA256:
        return sha
    named = REPORTING_SPLIT_SHA256.get(sha)
    if named is not None:
        raise PrevalenceError(
            f"refusing to build a {what} from the {named.upper()} split "
            f"(sha {sha[:16]}…). Selection happens on the discovery split "
            f"and reporting on the evaluation split; a mask chosen on "
            f"reporting data would make the I4 result circular "
            f"(TASK A §2, LOCKED_ANALYSIS §11).")
    raise PrevalenceError(
        f"refusing to build a {what} from split sha {sha[:16] or '<empty>'}… "
        f"— it is not the discovery split "
        f"({DISCOVERY_SPLIT_SHA256[:16]}…). Masks are built from discovery "
        f"data and from nothing else.")


# ─────────────────────────────────────────────────────────────────────────────
# Masks
# ─────────────────────────────────────────────────────────────────────────────

def topk_mask(pm, k: int = PRIMARY_MASK_K) -> list:
    """The `k` highest-prevalence positions of a DISCOVERY map, sorted.

    Ties are broken by ascending flat index — `np.argsort(-freq,
    kind="stable")`, the same rule `tools/sink_address.py` used for its
    `top10_positions`, so a mask and a stored top-10 list agree. The
    returned list is sorted ascending so a mask is a canonical object: two
    runs that select the same positions produce the same JSON.
    """
    assert_selection_split(pm, what="top-k mask")
    freq = np.asarray(pm.freq if isinstance(pm, PrevalenceMap) else pm,
                      dtype=np.float64).reshape(-1)
    k = int(k)
    if not 0 < k <= freq.size:
        raise PrevalenceError(
            f"k={k} outside 1..{freq.size} for this grid")
    chosen = np.argsort(-freq, kind="stable")[:k]
    return sorted(int(p) for p in chosen)


def ring_composition(mask, side: int) -> dict:
    """{ring: how many of `mask`'s positions sit in it}."""
    rings = border_distance_map(int(side)).reshape(-1)
    out = {}
    for p in mask:
        if not 0 <= int(p) < rings.size:
            raise PrevalenceError(
                f"position {p} is outside a {side}x{side} grid")
        out[int(rings[int(p)])] = out.get(int(rings[int(p)]), 0) + 1
    return dict(sorted(out.items()))


def mask_coordinates(mask, side: int) -> list:
    """`[{index, row, col, ring}]` — the form `configs/frozen/I4_masks.json`
    records, so a reader never has to divide by the grid width themselves."""
    side = int(side)
    rings = border_distance_map(side).reshape(-1)
    return [{"index": int(p), "row": int(p) // side, "col": int(p) % side,
             "ring": int(rings[int(p)])} for p in mask]


def ring_matched_controls(mask, side: int, *, n: int = N_CONTROL_MASKS,
                          seeds=CONTROL_SEEDS, exclude_primary: bool = False):
    """`n` random masks with the SAME number of positions in every ring.

    A control that ignored the rings would differ from the primary mask in
    two ways at once — where the mass is, and how far from the border it is
    — and I4 could not tell which one moved the result.

    Deterministic: control `i` is drawn with `np.random.RandomState(seeds[i])`,
    ring by ring in ascending ring order, so the same call always returns the
    same masks.

    OVERLAP WITH THE PRIMARY MASK — the one open question in this function.
    `docs/LOCKED_ANALYSIS.md` §5 says: "random masks MAY overlap the
    high-prevalence mask. The overlap is REPORTED; draws are never rejected
    to amplify contrast." `docs/TASK_A_I1_I6.md` §6 says the controls are
    "drawn from positions outside the primary mask". Those are different
    rules and they cannot both be followed. The LOCKED document is the
    signed one, so its rule is the DEFAULT here (`exclude_primary=False`)
    and the overlap is returned beside every control; the task file's rule
    is available as `exclude_primary=True` and nothing calls it. D5 is
    closed in Phase C — the human decides before then, and the choice is
    recorded in `configs/frozen/I4_masks.json`.

    Returns `[(control_mask, {"seed": s, "overlap": m}), ...]`.
    """
    side = int(side)
    primary = sorted(int(p) for p in mask)
    if len(set(primary)) != len(primary):
        raise PrevalenceError("the primary mask has duplicate positions")
    seeds = tuple(seeds)[:int(n)]
    if len(seeds) != int(n):
        raise PrevalenceError(
            f"{len(seeds)} seed(s) for {n} control mask(s) — every control "
            f"names its own seed")
    wanted = ring_composition(primary, side)
    primary_set = set(primary)

    out = []
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        picked = []
        for ring in sorted(wanted):
            pool = ring_indices(side, ring)
            if exclude_primary:
                pool = np.asarray([p for p in pool if int(p) not in primary_set])
            if pool.size < wanted[ring]:
                raise PrevalenceError(
                    f"ring {ring} of a {side}x{side} grid offers {pool.size} "
                    f"position(s) but the mask needs {wanted[ring]}")
            picked.extend(int(p) for p in
                          rng.choice(pool, size=wanted[ring], replace=False))
        control = sorted(picked)
        if ring_composition(control, side) != wanted:
            raise PrevalenceError(            # pragma: no cover - invariant
                f"control for seed {seed} is not ring-matched")
        out.append((control, {"seed": int(seed),
                              "overlap": len(primary_set & set(control))}))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The `*_addr.json` round trip
# ─────────────────────────────────────────────────────────────────────────────

def addr_basis_key(basis: str) -> str:
    """The `_addr.json` suffix for a basis. TASK-07 wrote `canon` and `mad`;
    I1/I2's fixed-tau basis is `fixed_cal`, the same RECIPE recalibrated per
    stage and split, and it keeps its own name so the two are never confused
    inside one document."""
    return str(basis)


def dumps_addr_json(doc: dict) -> str:
    """Serialize exactly as `tools/sink_address.py` does — `json.dump` with
    its defaults, no indent, no trailing newline. Byte identity with a
    committed map depends on this and on nothing else."""
    return json.dumps(doc)


def from_addr_json(path, basis: str, *, stage=None, split_sha=None,
                   n_prefix=1, split_name=MISSING) -> PrevalenceMap:
    """Parse one basis of a committed `*_addr.json` into a `PrevalenceMap`.

    The WHOLE document is carried on `source_document` with its key order
    intact, so `to_addr_json` can hand back the bytes it was given. The
    historical maps carry no stage and no split sha — they were written
    before either concept existed — so the caller supplies them; `stage`
    defaults to `hist`, which is where TASK-07 computed them
    (`docs/I0_HANDOFF.md` §3), and `split_sha` to the discovery split, which
    is the split `tools/sink_address.py` ran on.
    """
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != ADDR_SCHEMA:
        raise PrevalenceError(
            f"{path.name}: schema {doc.get('schema')!r}, expected "
            f"{ADDR_SCHEMA!r}")
    key = f"freq_{addr_basis_key(basis)}"
    if doc.get(key) is None:
        raise PrevalenceError(
            f"{path.name}: {key} is null "
            f"({doc.get('canon_skip_reason')!r}) — an absent map, not a map "
            f"of zeros")
    freq = np.asarray(doc[key], dtype=np.float64)
    n_images = int(doc["n_images"])
    pm = PrevalenceMap(
        freq=freq, n_images=n_images,
        n_exceed_total=int(round(float(freq.sum()) * n_images)),
        basis=basis if basis in MAP_BASES else "mad",
        threshold=doc.get("tau_canon", MISSING) if basis != "mad" else MISSING,
        stage=stage or "hist", split_sha=split_sha or DISCOVERY_SPLIT_SHA256,
        split_name=split_name, run_id=doc.get("source_npz", path.stem),
        ckpt_sha256=doc["ckpt_sha256"],
        grid_side=grid_side_of(int(doc["n_positions"])), n_prefix=n_prefix,
        git_sha=doc["git_sha"])
    pm.source_document = doc
    return pm


def to_addr_json(pm) -> dict:
    """The `_addr.json` document for a map (or an iterable of maps that
    share one checkpoint and differ only in basis).

    A map PARSED from a committed file is handed back verbatim — key order,
    float digits and all — because this contract reads history and never
    rewrites it (I0 handoff §8.1). `verify_addr_document` is the check that
    the carried values are the ones the project's own verifier accepts; the
    two together are what makes the round trip meaningful rather than
    circular. A map built from a sweep gets a freshly derived document.
    """
    maps = [pm] if isinstance(pm, PrevalenceMap) else list(pm)
    if not maps:
        raise PrevalenceError("no maps to serialize")
    carried = [m.source_document for m in maps if m.source_document]
    if carried:
        if any(c is not carried[0] and c != carried[0] for c in carried):
            raise PrevalenceError(
                "these maps came from different `_addr.json` documents and "
                "cannot be written back into one")
        return carried[0]

    head = maps[0]
    for m in maps:
        if (m.ckpt_sha256, m.n_images, m.n_positions) != (
                head.ckpt_sha256, head.n_images, head.n_positions):
            raise PrevalenceError(
                "one `_addr.json` document describes one checkpoint on one "
                "split; these maps do not agree on checkpoint, n_images or "
                "grid")
    doc = {"schema": ADDR_SCHEMA, "source_npz": head.run_id,
           "ckpt_sha256": head.ckpt_sha256, "git_sha": head.git_sha,
           "n_images": head.n_images, "n_positions": head.n_positions,
           "stage": head.stage, "split_sha256": head.split_sha,
           "split_name": head.split_name, "n_prefix": head.n_prefix,
           "grid_side": head.grid_side, "condition_id": head.condition_id,
           "written_by": "saga/frozen/prevalence.py"}
    for m in maps:
        b = addr_basis_key(m.basis)
        conc = concentration_recompute(m.freq)
        doc[f"tau_{b}"] = m.threshold
        doc[f"freq_{b}"] = [float(v) for v in m.freq]
        doc[f"mass_{b}"] = float(m.mass)
        doc[f"concentration_{b}"] = dict(
            conc, top10_positions=[int(p) for p in
                                   np.argsort(-m.freq, kind="stable")[:10]])
    return doc


def verify_addr_document(doc: dict, basis: str, label="", tol=1e-9) -> None:
    """Run the PROJECT's own stored-concentration check on a document.

    `analysis.address_analysis.verify_stored_concentration` re-derives every
    concentration statistic through an independent implementation and
    raises on disagreement. Importing it here is the point: a document this
    module produces or carries has to pass the same gate the TASK-07 tables
    passed.
    """
    from analysis.address_analysis import verify_stored_concentration
    verify_stored_concentration(doc, addr_basis_key(basis),
                                label or doc.get("source_npz", "<map>"),
                                tol=tol)


# ─────────────────────────────────────────────────────────────────────────────
# Reused statistics — thin, named wrappers over address_analysis
# ─────────────────────────────────────────────────────────────────────────────

def ring_profile(pm) -> list:
    """Mean frequency per border ring, ring 0 = the outermost row/col."""
    return border_rings(pm.freq if isinstance(pm, PrevalenceMap) else pm)


def ring_share(pm, ring: int = 1) -> float:
    """The fraction of a map's TOTAL mass sitting in one ring."""
    freq = np.asarray(pm.freq if isinstance(pm, PrevalenceMap) else pm,
                      dtype=np.float64)
    side = grid_side_of(freq.size)
    total = float(freq.sum())
    if total <= 0:
        raise PrevalenceError("a zero-mass map has no ring share")
    return float(freq[ring_indices(side, int(ring))].sum() / total)


def concentration_with_reference(pm, *, sims=400, seed=0) -> dict:
    """Every concentration statistic, its finite-sample null at THIS map's
    own mass, and the excess — the only cross-member comparable form
    (`analysis/address_analysis.py`, reference 1)."""
    freq = np.asarray(pm.freq if isinstance(pm, PrevalenceMap) else pm,
                      dtype=np.float64)
    n_images = int(pm.n_images) if isinstance(pm, PrevalenceMap) else None
    if n_images is None:
        raise PrevalenceError(
            "the finite-sample reference needs n_images — pass a "
            "PrevalenceMap, not a bare array")
    observed = concentration_recompute(freq)
    null = concentration_null(observed["total_mass"], n_images, freq.size,
                              sims=sims, seed=seed)
    out = dict(observed)
    for stat, (mean, sd) in null.items():
        out[f"{stat}_null"] = mean
        out[f"{stat}_null_sd"] = sd
        out[f"{stat}_excess"] = (None if mean is None
                                 else float(observed[stat] - mean))
    return out


def spatial_rho(a, b):
    """`(rho, p, null_sd, n_transforms)` under the dihedral x torus-roll
    permutation reference, or None when either map is spatially constant.

    The assumption this reference makes, stated wherever it is reported: the
    two maps are exchangeable under the 8 dihedral transforms and all
    side^2 torus rolls of the grid. A BORDERED map — and every map here is
    bordered, since the ring profile is the whole point — satisfies that
    only approximately, because a roll wraps the border onto the interior.
    """
    fa = np.asarray(a.freq if isinstance(a, PrevalenceMap) else a,
                    dtype=np.float64)
    fb = np.asarray(b.freq if isinstance(b, PrevalenceMap) else b,
                    dtype=np.float64)
    if fa.size != fb.size:
        raise PrevalenceError(
            f"cannot correlate a {fa.size}-position map with a "
            f"{fb.size}-position one")
    return rho_spatial(fa, fb)


def ring_adjusted(pm) -> np.ndarray:
    """A map with its per-ring mean subtracted.

    Correlating two of these answers "do these maps agree on WHERE the mass
    is beyond the fact that both put it near the border?" — a ring profile
    alone can carry a large rho between two maps that share nothing else.
    A map that is constant within every ring becomes zero to floating-point
    accuracy, which is what the test pins.
    """
    freq = np.asarray(pm.freq if isinstance(pm, PrevalenceMap) else pm,
                      dtype=np.float64).copy()
    side = grid_side_of(freq.size)
    rings = border_distance_map(side).reshape(-1)
    out = freq.copy()
    for ring in np.unique(rings):
        sel = rings == ring
        out[sel] = freq[sel] - freq[sel].mean()
    return out


def eta2_positional(pm) -> float:
    """η²_pos = Var_P f(P) / (f̄(1 − f̄)).

    The share of the total Bernoulli variance of "does position P exceed in
    image i" that the POSITION explains. 0 when every position has the same
    frequency, whatever that frequency is; it is therefore comparable across
    maps of very different mass in a way raw variance is not.

    Population variance over positions (ddof = 0): the 196 positions are the
    whole grid, not a sample from one.
    """
    freq = np.asarray(pm.freq if isinstance(pm, PrevalenceMap) else pm,
                      dtype=np.float64)
    fbar = float(freq.mean())
    if not 0.0 < fbar < 1.0:
        raise PrevalenceError(
            f"eta2 is undefined at mean frequency {fbar} — every position "
            f"exceeds always, or none ever does")
    return float(freq.var(ddof=0) / (fbar * (1.0 - fbar)))


__all__ = [
    "PrevalenceMap", "PrevalenceError", "MISSING", "MAP_BASES",
    "DISCOVERY_SPLIT_SHA256", "REPORTING_SPLIT_SHA256", "PRIMARY_MASK_K",
    "N_CONTROL_MASKS", "CONTROL_SEEDS", "ADDR_SCHEMA",
    "grid_side_of", "from_counts", "from_i2_maps_npz", "i2_map_keys",
    "cell_mean_map", "assert_selection_split", "topk_mask",
    "ring_composition", "mask_coordinates", "ring_matched_controls",
    "from_addr_json", "to_addr_json", "dumps_addr_json",
    "verify_addr_document", "addr_basis_key",
    "ring_profile", "ring_share", "concentration_with_reference",
    "spatial_rho", "ring_adjusted", "eta2_positional",
]
