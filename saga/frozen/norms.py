"""
saga/frozen/norms.py
====================
Per-token norm extraction, per-image exceedance indicators, and the
scale-only null — TASK A / I1 A3.

TASK I2 already stored, for every run on both splits, the per-position
exceedance COUNTS at `s11_out`, `hist` and `s12_post_norm`. I1 consumes
those. What it cannot consume, because I2 never stored it, is:

  * the per-token NORM FIELD, `[n_images, N_patches]` — the scale-only null
    rescales it and re-thresholds, which a count cannot support; and
  * the per-image INDICATOR matrix, `[n_images, N_patches]` of "did this
    position exceed in this image" — every image-level bootstrap and the
    split-half reliability need it, and a summed count has thrown it away.

so those are what this module extracts, at the four stages I1 reports and
at the two block-input stages D5 needs.

WHERE THE TWO ARTIFACTS GO, AND WHY DIFFERENTLY
-----------------------------------------------
`norms_<stage>.npz`  float32 [n_images, N] — 10,000 x 196 x 4 B ~ 8 MB per
                     stage per run, ~1 GB over the cohort and both splits.
                     Written to the results tree ON THE CLUSTER and
                     GIT-IGNORED (`.gitignore`: results/frozen/**/norms_*.npz).
                     It is an intermediate: everything the paper prints is
                     derived from it into the file below.

`maps_<stage>.npz`   the frequency map per basis, the per-image indicators
                     `np.packbits`-ed (196 bits -> 25 B per image, ~245 kB
                     for 10,000), the counts, the thresholds used and the
                     provenance. COMMITTED — this is what Phase C reads.

Both are append-safe in the sense I0 §8.5 requires: they are written whole,
atomically, to a temp file and renamed, and the run's completion marker is
written after them.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from saga.frozen.diag import MAD_K, MAP_BASES, mad_threshold
from saga.frozen.stages import resolve_stage
from saga.metrics import token_norms

MISSING = "MISSING"

#: Stages I1 extracts. `hist` is the historical alias; `s11_out` is D1, the
#: reporting stage; the two block-input stages are D5's.
I1_STAGES = ("s11_out", "in_b07", "in_b08", "hist")


class NormsError(ValueError):
    """A norm field or indicator set that cannot be built as asked."""


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

class StageAccumulator:
    """Per-token norms and per-image exceedance indicators for ONE
    (condition, stage), accumulated batch by batch in the split's order.

    The norms are kept as float32 — the precision they were computed at
    would make the file twice the size for digits that a norm ratio cannot
    use — and the indicators as bool until they are packed at write time.
    """

    def __init__(self, condition_id: str, stage: str, *, tau_cal=None,
                 k: float = MAD_K):
        self.condition_id = condition_id
        self.stage = stage
        self.tau_cal = tau_cal
        self.k = float(k)
        self._norms = []
        self._ind = {b: [] for b in MAP_BASES}
        self._mad_thr = []
        self.image_ids = []
        self.n_positions = None

    def add(self, patches: torch.Tensor, image_ids) -> None:
        """One batch of `[B, N_patches, D]` patch rows — prefix rows ALREADY
        removed by `capture_stages` using the model's own count."""
        if patches.ndim != 3:
            raise NormsError(
                f"expected [B, N_patches, D] patch rows, got "
                f"{tuple(patches.shape)}")
        ids = list(image_ids)
        if len(ids) != patches.shape[0]:
            raise NormsError(
                f"{len(ids)} image id(s) for a batch of {patches.shape[0]}")
        norms = token_norms(patches)                            # [B, N]
        if self.n_positions is None:
            self.n_positions = int(norms.shape[1])
        elif int(norms.shape[1]) != self.n_positions:
            raise NormsError(
                f"{self.condition_id}/{self.stage}: patch count changed from "
                f"{self.n_positions} to {norms.shape[1]} inside one run")

        thr = mad_threshold(norms, self.k)                      # [B]
        self._norms.append(norms.detach().cpu().numpy().astype(np.float32))
        self._mad_thr.append(thr.detach().cpu().numpy().astype(np.float64))
        self._ind["mad"].append(
            (norms > thr.unsqueeze(1)).detach().cpu().numpy())
        if isinstance(self.tau_cal, (int, float)):
            self._ind["fixed_cal"].append(
                (norms > float(self.tau_cal)).detach().cpu().numpy())
        self.image_ids.extend(str(i) for i in ids)

    # ── assembled views ─────────────────────────────────────────────────────

    def norms(self) -> np.ndarray:
        if not self._norms:
            raise NormsError(
                f"{self.condition_id}/{self.stage}: nothing was accumulated")
        return np.concatenate(self._norms, axis=0)

    def mad_thresholds(self) -> np.ndarray:
        return np.concatenate(self._mad_thr, axis=0)

    def indicators(self, basis: str):
        """`[n_images, N]` bool, or None when the basis has no threshold."""
        if basis not in MAP_BASES:
            raise NormsError(f"unknown basis {basis!r}")
        chunks = self._ind[basis]
        return np.concatenate(chunks, axis=0) if chunks else None

    def counts(self, basis: str):
        ind = self.indicators(basis)
        return None if ind is None else ind.sum(axis=0).astype(np.int64)

    @property
    def n_images(self) -> int:
        return len(self.image_ids)


@torch.no_grad()
def extract_stage_norms(model, dataset, stages, *, device, batch_size,
                        image_ids, tau_cal, k: float = MAD_K,
                        max_images=None, condition_id="native",
                        edit=None) -> dict:
    """`{stage: StageAccumulator}` for one checkpoint over one split.

    ONE forward per batch serves every stage: `capture_stages` hooks them
    all, exactly as `run_work_package_stages` does for I2, so adding the two
    block-input stages costs no extra inference.

    `edit` is an optional zero-argument context manager factory (for the
    `term_1.00` bypass condition); the model is otherwise untouched.
    """
    from contextlib import nullcontext

    from torch.utils.data import DataLoader

    from saga.frozen.stages import capture_stages

    wanted = tuple(stages)
    for s in wanted:
        resolve_stage(s)
    acc = {s: StageAccumulator(condition_id, s, tau_cal=tau_cal.get(s),
                               k=k) for s in wanted}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    was_training = model.training
    model.eval()
    seen = 0
    try:
        with (edit() if edit is not None else nullcontext()):
            with capture_stages(model, wanted) as store:
                for images, _targets in loader:
                    if max_images is not None and seen >= max_images:
                        break
                    images = images.to(device)
                    if max_images is not None and \
                            seen + images.shape[0] > max_images:
                        images = images[:max_images - seen]
                    store.clear()
                    model(images)
                    n = images.shape[0]
                    ids = image_ids[seen:seen + n]
                    for s in wanted:
                        if s not in store:
                            raise NormsError(
                                f"stage {s!r} was not captured — the forward "
                                f"pass did not reach the hooked module")
                        acc[s].add(store[s], ids)
                    seen += n
    finally:
        if was_training:
            model.train()
    if not seen:
        raise NormsError("the split yielded no images")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_savez(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def _meta_array(meta: dict) -> np.ndarray:
    return np.array(json.dumps(meta, sort_keys=True, separators=(",", ":"),
                               default=str))


def write_norms_npz(out_dir, acc: StageAccumulator, meta: dict) -> Path:
    """`norms_<stage>.npz` — the per-token norm field. GIT-IGNORED.

    It stays on the cluster: ~8 MB per stage per run, ~1 GB over the cohort
    and both splits, and every number the paper prints is derived from it
    into `maps_<stage>.npz`, which IS committed.
    """
    payload = {
        "norms": acc.norms(),
        "mad_thresholds": acc.mad_thresholds(),
        "image_ids": np.array(acc.image_ids),
        "meta_json": _meta_array(dict(
            meta, kind="norms", stage=acc.stage,
            condition_id=acc.condition_id, n_images=acc.n_images,
            n_positions=acc.n_positions, mad_k=acc.k,
            tau_cal=acc.tau_cal if isinstance(acc.tau_cal, (int, float))
            else MISSING)),
    }
    return _atomic_savez(Path(out_dir) / f"norms_{acc.stage}.npz", payload)


def write_stage_maps_npz(out_dir, accs, meta: dict) -> Path:
    """`maps_<stage>.npz` — frequency map, packed per-image indicators,
    counts, thresholds, provenance. COMMITTED.

    `accs` is the list of `StageAccumulator`s for ONE stage, one per
    condition (`native`, and `term_1.00` for SAGA at `hist`). Keys are
    `"<condition>|<basis>"`, matching the `"<condition>|<stage>|<basis>"`
    convention of I2's `maps.npz` minus the stage, which names the file.

    Indicators are `np.packbits` over `[n_images, N]`: 196 bits is 25 bytes
    per image, ~245 kB for 10,000 images, against 2 MB unpacked. `n_positions`
    is stored because unpacking needs it — 25 bytes is 200 bits, and the
    last 4 are padding.
    """
    accs = list(accs)
    if not accs:
        raise NormsError("no accumulators to write")
    stage = accs[0].stage
    if any(a.stage != stage for a in accs):
        raise NormsError(
            f"one maps file is one stage; got {sorted({a.stage for a in accs})}")

    payload, per_condition = {}, {}
    for acc in accs:
        for basis in MAP_BASES:
            ind = acc.indicators(basis)
            if ind is None:
                continue
            key = f"{acc.condition_id}|{basis}"
            counts = ind.sum(axis=0).astype(np.int64)
            payload[f"counts__{key}"] = counts
            payload[f"freq__{key}"] = counts / float(acc.n_images)
            payload[f"ind__{key}"] = np.packbits(ind, axis=1)
            payload[f"n_images__{key}"] = np.int64(acc.n_images)
        per_condition[acc.condition_id] = {
            "n_images": acc.n_images, "n_positions": acc.n_positions,
            "mad_k": acc.k,
            "tau_cal": (acc.tau_cal if isinstance(acc.tau_cal, (int, float))
                        else MISSING),
            "bases": [b for b in MAP_BASES if acc.indicators(b) is not None],
        }
    if not payload:
        raise NormsError(f"no exceedance indicators at stage {stage!r}")

    payload["image_ids"] = np.array(accs[0].image_ids)
    payload["meta_json"] = _meta_array(dict(
        meta, kind="maps", stage=stage,
        stage_resolved=resolve_stage(stage),
        n_positions=accs[0].n_positions, conditions=per_condition))
    return _atomic_savez(Path(out_dir) / f"maps_{stage}.npz", payload)


def read_indicators(path, condition_id: str, basis: str) -> np.ndarray:
    """`[n_images, N]` bool out of a committed `maps_<stage>.npz`."""
    with np.load(Path(path), allow_pickle=False) as z:
        key = f"{condition_id}|{basis}"
        if f"ind__{key}" not in z.files:
            raise NormsError(
                f"{Path(path).name} has no indicators for {key!r}; it holds "
                f"{sorted(k[5:] for k in z.files if k.startswith('ind__'))}")
        packed = z[f"ind__{key}"]
        n_positions = int(json.loads(str(z["meta_json"]))["n_positions"])
    return np.unpackbits(packed, axis=1, count=n_positions).astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# The scale-only null (T_I1b)
# ─────────────────────────────────────────────────────────────────────────────
#
# THE ALTERNATIVE EXPLANATION IT TESTS. SAGA's maps carry fewer exceedances
# than the baseline's. If the only thing SAGA did was shrink the whole norm
# field by a constant, then thresholding at a FIXED tau would find fewer
# positions above it — and the map would look sparser while the underlying
# spatial arrangement was untouched. Rescaling the baseline's norm field by
# the scalar `c` that reproduces SAGA's count, and then asking whether the
# rescaled baseline map looks like SAGA's map, separates "less of the same
# structure" from "different structure".
#
# The MAD basis is the control that makes the argument legible: a per-image
# median + k*MAD threshold scales with the field, so its count is EXACTLY
# invariant under any positive c. A difference that survives on the MAD
# basis is not a scale effect. `tests/test_I1_spatial.py` asserts that
# invariance exactly, not approximately.

def fixed_counts(norms, tau: float, c: float = 1.0) -> np.ndarray:
    """Per-image count of positions whose norm, scaled by `c`, exceeds `tau`."""
    v = np.asarray(norms, dtype=np.float64)
    return (v * float(c) > float(tau)).sum(axis=1)


def mad_counts(norms, k: float = MAD_K, c: float = 1.0) -> np.ndarray:
    """Per-image count above the per-image median + k*MAD threshold.

    EXACTLY invariant in `c` for c > 0: median(cv) + k*MAD(cv) = c*(median(v)
    + k*MAD(v)), and the comparison `cv > c*t` has the same truth value as
    `v > t`. The identity is the reason this basis is reported beside the
    fixed one.
    """
    from tools.compute_fixed_thr import per_image_mad_thresholds
    v = np.asarray(norms, dtype=np.float64) * float(c)
    thr = per_image_mad_thresholds(v, float(k))
    return (v > thr[:, None]).sum(axis=1)


def mean_fixed_count(norms, tau: float, c: float = 1.0) -> float:
    return float(fixed_counts(norms, tau, c).mean())


def match_scale(norms, tau: float, target_count: float, *,
                lo: float = 1e-6, hi: float = 1e6, iters: int = 200) -> dict:
    """The scalar `c` that rescales a norm field to a target mean count.

    `mean_fixed_count(norms, tau, c)` is a NON-DECREASING step function of
    `c` (raising c can only push norms above a fixed tau, never below), so
    the smallest `c` reaching the target is well defined and is what
    bisection converges to. Exact equality is generally unattainable — the
    function jumps by 1/n_images at each of the n_images x N breakpoints —
    so the ACHIEVED count is returned beside `c` and the table reports both.

    "Unique" means: the smallest c with mean_count(c) >= target. Any c in
    the plateau below the next breakpoint gives the same count; the infimum
    is the canonical representative and is what two runs of this function
    agree on.
    """
    v = np.asarray(norms, dtype=np.float64)
    target = float(target_count)
    lo, hi = float(lo), float(hi)
    if mean_fixed_count(v, tau, hi) < target:
        raise NormsError(
            f"no scale in [{lo}, {hi}] reaches a mean count of {target}: even "
            f"c={hi} gives {mean_fixed_count(v, tau, hi)}")
    if mean_fixed_count(v, tau, lo) >= target:
        return {"c": lo, "achieved": mean_fixed_count(v, tau, lo),
                "target": target, "at_bound": "lo"}
    for _ in range(int(iters)):
        mid = (lo + hi) / 2.0
        if mean_fixed_count(v, tau, mid) >= target:
            hi = mid
        else:
            lo = mid
    return {"c": hi, "achieved": mean_fixed_count(v, tau, hi),
            "target": target, "at_bound": MISSING}


def scaled_frequency_map(norms, tau: float, c: float) -> np.ndarray:
    """The exceedance frequency map of a norm field scaled by `c`."""
    v = np.asarray(norms, dtype=np.float64)
    return (v * float(c) > float(tau)).mean(axis=0)


__all__ = [
    "I1_STAGES", "NormsError", "StageAccumulator", "extract_stage_norms",
    "write_norms_npz", "write_stage_maps_npz", "read_indicators",
    "fixed_counts", "mad_counts", "mean_fixed_count", "match_scale",
    "scaled_frequency_map",
]
