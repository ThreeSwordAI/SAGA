"""
saga/frozen/diag.py
===================
The patch diagnostics a stage sweep records, and the per-stage calibrated
thresholds it needs — added for TASK I2 and usable unchanged by I1/I3/I4/I5.

Additive by construction. Nothing here edits a model: `saga/frozen/edits.py`
is untouched by this module, `saga/metrics.py` is IMPORTED and never
re-implemented, and `results/diagsplit/fixed_thresholds_canon.json` is opened
READ-ONLY by `load_canon_thresholds` and by nothing else in the project that
this module can reach (`tests/test_I2_terminal.py` pins its sha256).

Two threshold families, and they are not interchangeable:

    tau_canon   the HISTORICAL absolute threshold, one per (arch,
                recipe_actual) cell, from `fixed_thresholds_canon.json`.
                Calibrated at the LAST BLOCK on the discovery split, so it
                is defined at stage `hist` and NOWHERE ELSE. Asking for it
                at another stage returns the literal MISSING, never a
                rescaled guess (TASK I2 §4).

    tau_cal     the per-stage recalibration of the SAME recipe on the split
                being run: median over images of the per-image
                median + 5*MAD threshold, from the cell's designated
                BASELINE checkpoint. New keys, written to
                `results/frozen/I2_terminal/thresholds_cal.json`. At `hist`
                it reproduces the canon recipe on a different split, so the
                two sit side by side in the tables and the split effect is
                visible rather than absorbed (TASK I2 §5).

The MAD convention — LOWER medians, matching `torch.median` as
`saga.metrics.sink_counts_mad` uses it — is the canon tool's own:
`tools/compute_fixed_thr.per_image_mad_thresholds` is imported for the
calibration, and the torch path in `mad_threshold` is pinned against BOTH
that function and `sink_counts_mad` by
`tests/test_I2_terminal.py::test_the_mad_threshold_is_the_existing_definition`.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from saga.frozen.stages import HIST_STAGE, resolve_stage
from saga.metrics import (_nosink_per_image, effective_rank,
                          oversmoothing_pairwise, sink_counts_fixed,
                          sink_counts_mad, token_norms)

MISSING = "MISSING"

#: The project's primary sink threshold constant (saga/metrics.py docstring,
#: `sink_mad_k5`). Not a tunable: every historical count uses k = 5.
MAD_K = 5.0

#: Norm quantiles recorded per image (TASK I2 §4).
NORM_QUANTILES = ((0.50, "norm_p50"), (0.90, "norm_p90"), (0.99, "norm_p99"),
                  (0.999, "norm_p999"))

#: The historical thresholds file. READ-ONLY from here, always.
CANON_THRESHOLDS = "results/diagsplit/fixed_thresholds_canon.json"

#: The one stage at which `count_fixed_canon` is defined (§4). It is the
#: DECLARED name `hist` as well as whatever `hist` resolves to, so a YAML
#: that spells the stage either way behaves identically.
CANON_STAGES = ("hist", HIST_STAGE)

#: The diagnostic keys of one (image, condition, stage) row.
DIAG_KEYS = tuple(name for _, name in NORM_QUANTILES) + (
    "norm_max", "mad_thr", "count_fixed_canon", "count_fixed_cal",
    "count_mad", "cos_all", "cos_nosink_mad", "eff_rank")

#: Exceedance-map bases aggregated into `maps.npz` (§4).
MAP_BASES = ("fixed_cal", "mad")


class DiagError(ValueError):
    """A diagnostic that cannot be computed as asked."""


# ─────────────────────────────────────────────────────────────────────────────
# The MAD threshold — one definition, two implementations, pinned to each other
# ─────────────────────────────────────────────────────────────────────────────

def mad_threshold(norms: torch.Tensor, k: float = MAD_K) -> torch.Tensor:
    """Per-image median + k*MAD, the threshold `sink_counts_mad` counts above.

    `saga/metrics.py` computes this quantity inside `sink_counts_mad` and
    `_nosink_per_image` but never returns it, and I2 needs the VALUE (it is
    what `tau_cal` is a median of, and it is recorded per image). This is
    the same arithmetic in the same order — lower medians, via
    `torch.median` — and a test asserts
    `(norms > mad_threshold(norms, k)).sum(1) == sink_counts_mad(norms, k)`
    exactly, so the two cannot drift.

    norms: [B, N] -> [B].
    """
    v = norms.float()
    m = v.median(dim=1, keepdim=True).values
    mad = (v - m).abs().median(dim=1, keepdim=True).values
    return (m + float(k) * mad).squeeze(1)


def canon_per_image_mad_thresholds(norms_np: np.ndarray, k: float = MAD_K):
    """THE canon recipe's per-image thresholds, imported not re-derived.

    `tools/compute_fixed_thr.py` produced every value in
    `fixed_thresholds_canon.json`; calling its function is what makes
    "tau_cal at `hist` is the canon recipe on another split" a fact about
    the code rather than a claim in a docstring.
    """
    try:
        from tools.compute_fixed_thr import per_image_mad_thresholds
    except ImportError as exc:                           # pragma: no cover
        raise DiagError(
            f"the calibrated thresholds use tools/compute_fixed_thr.py — THE "
            f"canon recipe — and it is not importable: {exc}. Run from the "
            f"repo root.") from exc
    return per_image_mad_thresholds(np.asarray(norms_np), float(k))


def canon_lower_median(values_np: np.ndarray) -> float:
    """The canon tool's outer median over images (lower median)."""
    try:
        from tools.compute_fixed_thr import _lower_median
    except ImportError as exc:                           # pragma: no cover
        raise DiagError(
            f"tools/compute_fixed_thr.py is not importable: {exc}") from exc
    arr = np.asarray(values_np, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise DiagError("no per-image thresholds to take a median of")
    return float(_lower_median(arr, axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# Cells and the historical thresholds (read-only)
# ─────────────────────────────────────────────────────────────────────────────

def cell_key(arch: str, recipe_actual: str) -> str:
    """`"<arch>|<recipe_actual>"` — the key both threshold files already use.

    `recipe_actual` is NEVER a directory name (results/notes/recipe_erratum.md).
    """
    return f"{arch}|{recipe_actual}"


def load_canon_thresholds(path=CANON_THRESHOLDS) -> dict:
    """The historical per-cell tau file, opened READ-ONLY.

    This module offers no writer for it and no caller can obtain one: the
    canon calibration is a historical value, and TASK I2 §10 forbids
    recalibrating it. `tests/test_I2_terminal.py` pins its sha256.
    """
    p = Path(path)
    if not p.exists():
        raise DiagError(
            f"{p} not found — the historical per-cell thresholds are a "
            f"committed artifact of TASK-02C/06B, not something I2 computes")
    return json.loads(p.read_text(encoding="utf-8"))


def canon_sha256(path=CANON_THRESHOLDS) -> str:
    """sha256 of the canon file with LINE ENDINGS NORMALISED to LF.

    The raw bytes are not portable: this repository is checked out on Windows
    locally and on Linux on the cluster, so the same committed JSON hashes to
    two different values depending on where it sits. Normalising makes the
    pin a statement about the CONTENT — which tau, for which cell, from which
    checkpoint — which is the thing that must never change.
    `tests/test_I2_terminal.py` pins this digest and the tau values beside it.
    """
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def canon_tau_for(cell: str, canon: dict):
    """tau_canon for a cell, or MISSING when the cell has none."""
    value = canon.get(cell)
    return float(value) if isinstance(value, (int, float)) else MISSING


def canon_defines_stage(stage: str) -> bool:
    """True only at `hist`. §4: `count_fixed_canon` is defined there and is
    the literal MISSING everywhere else — the historical tau was calibrated
    at the last block and means nothing at s11_out or after the final norm."""
    return stage in CANON_STAGES or resolve_stage(stage) == HIST_STAGE


# ─────────────────────────────────────────────────────────────────────────────
# Per-image diagnostics at one stage
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def patch_diagnostics(patches: torch.Tensor, *, tau_cal=None, tau_canon=None,
                      k: float = MAD_K):
    """(per-image values, {basis: [B, N] bool}) for one (condition, stage).

    `patches` is [B, N_patches, D] with the prefix rows ALREADY removed by
    `capture_stages` using the model's own count — this function never
    slices a prefix row itself and never sees one.

    `tau_cal` / `tau_canon` are absolute thresholds or None/MISSING; a
    threshold that is not a number yields the literal MISSING for its count
    and no exceedance map, which is the whole point of `count_fixed_canon`
    off `hist`.

    Every returned value is a list of length B, in the batch's order.
    """
    if patches.ndim != 3:
        raise DiagError(
            f"expected [B, N_patches, D] patch rows, got "
            f"{tuple(patches.shape)}")
    b, n, _ = patches.shape
    if n < 2:
        raise DiagError(
            f"{n} patch row(s) — the pairwise cosine and the MAD threshold "
            f"are undefined below 2")

    norms = token_norms(patches)                                  # [B, N]
    qs = torch.tensor([q for q, _ in NORM_QUANTILES], dtype=torch.float64)
    quantiles = torch.quantile(norms.double(), qs.to(norms.device), dim=1)
    thr_mad = mad_threshold(norms, k)                             # [B]

    cos_nosink, _ = _nosink_per_image(patches, norms, k)           # [B]
    if not bool(torch.isfinite(cos_nosink).all()):
        # Unreachable with N >= 4: at least half the norms lie at or below
        # the median and therefore at or below median + k*MAD. Refuse rather
        # than write a NaN that a mean would silently absorb.
        raise DiagError(
            "cos_nosink_mad is not finite — fewer than 2 tokens survived the "
            "MAD threshold, which cannot happen for a non-degenerate image")

    out = {name: quantiles[i].tolist()
           for i, (_, name) in enumerate(NORM_QUANTILES)}
    out["norm_max"] = norms.amax(dim=1).double().tolist()
    out["mad_thr"] = thr_mad.double().tolist()
    out["count_mad"] = sink_counts_mad(norms, k=k).tolist()
    out["cos_all"] = oversmoothing_pairwise(patches).double().tolist()
    out["cos_nosink_mad"] = cos_nosink.double().tolist()
    out["eff_rank"] = effective_rank(patches).double().tolist()

    maps = {"mad": (norms > thr_mad.unsqueeze(1)).cpu().numpy()}

    if isinstance(tau_cal, (int, float)):
        out["count_fixed_cal"] = sink_counts_fixed(norms, float(tau_cal)).tolist()
        maps["fixed_cal"] = (norms > float(tau_cal)).cpu().numpy()
    else:
        out["count_fixed_cal"] = [MISSING] * b
        maps["fixed_cal"] = None

    if isinstance(tau_canon, (int, float)):
        out["count_fixed_canon"] = [
            _int_str(v) for v in sink_counts_fixed(norms, float(tau_canon))]
    else:
        out["count_fixed_canon"] = [MISSING] * b

    missing = [key for key in DIAG_KEYS if key not in out]
    if missing:                                          # pragma: no cover
        raise DiagError(f"patch_diagnostics did not produce {missing}")
    return out, maps


def _int_str(v) -> str:
    """An exceedance count as a decimal string.

    `count_fixed_canon` is MISSING at every stage but `hist`, so the column
    holds both numbers and the literal MISSING. Parquet columns are typed:
    a float/str mixture is not writable, and a null would be averaged away
    by a reader who never noticed. A string column keeps `MISSING` a VALUE
    (I0 handoff §8.2) in both the parquet and the CSV form.
    """
    return str(int(round(float(v))))


# ─────────────────────────────────────────────────────────────────────────────
# Exceedance maps, aggregated per (condition, stage, basis)
# ─────────────────────────────────────────────────────────────────────────────

class MapAccumulator:
    """Per-position exceedance COUNTS over the images of one run.

    Counts, not frequencies: a frequency computed here and a frequency
    computed by the table builder would be two numbers that must agree, and
    `counts / n_images` reconstructs it exactly whenever it is wanted.
    """

    def __init__(self):
        self.counts: dict = {}
        self.n_images: dict = {}

    def add(self, condition_id: str, stage: str, maps: dict):
        for basis, m in maps.items():
            if m is None:
                continue
            arr = np.asarray(m, dtype=bool)
            if arr.ndim != 2:
                raise DiagError(
                    f"exceedance map for {condition_id}/{stage}/{basis} has "
                    f"shape {arr.shape}, expected [B, N_patches]")
            key = (condition_id, stage, basis)
            col = self.counts.get(key)
            if col is None:
                self.counts[key] = arr.sum(axis=0).astype(np.int64)
                self.n_images[key] = int(arr.shape[0])
            else:
                if col.shape[0] != arr.shape[1]:
                    raise DiagError(
                        f"{key}: patch count changed from {col.shape[0]} to "
                        f"{arr.shape[1]} inside one run")
                col += arr.sum(axis=0).astype(np.int64)
                self.n_images[key] += int(arr.shape[0])

    def key_name(self, condition_id: str, stage: str, basis: str) -> str:
        return f"{condition_id}|{stage}|{basis}"

    def arrays(self) -> dict:
        """`{"<cond>|<stage>|<basis>": int64[N], "n_images__<same>": int}`."""
        out = {}
        for (cond, stage, basis), col in sorted(self.counts.items()):
            name = self.key_name(cond, stage, basis)
            out[name] = col
            out[f"n_images__{name}"] = np.int64(self.n_images[(cond, stage,
                                                               basis)])
        return out


def write_maps_npz(out_dir, accumulator: MapAccumulator, meta: dict):
    """`maps.npz` — the aggregated exceedance maps plus their provenance.

    Written the way every other artifact in this package is: to a temp file,
    fsynced, then renamed over the target, so a job killed mid-write leaves
    the previous complete file or nothing extra.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "maps.npz"
    arrays = accumulator.arrays()
    if not arrays:
        raise DiagError("no exceedance maps were accumulated")
    payload = dict(arrays)
    payload["meta_json"] = np.array(
        json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str))
    tmp = path.with_name(path.name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage calibrated thresholds (D2)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_tau(per_image_thresholds) -> float:
    """tau_cal from one checkpoint's per-image MAD thresholds at one stage:
    the LOWER median over images, exactly as the canon tool takes it."""
    return canon_lower_median(np.asarray(per_image_thresholds,
                                         dtype=np.float64))


def thresholds_cal_path(template, split_name: str) -> Path:
    """The calibrated-thresholds path for one split.

    The conditions YAML declares a TEMPLATE containing `{split_name}` rather
    than a fixed path, because tau_cal is a property of the split it was
    calibrated on: `load_thresholds_cal` refuses a file whose split sha does
    not match, so one shared path would make the second split's sweep either
    fail or — worse, if the guard were ever relaxed — silently reuse the
    first split's thresholds.
    """
    text = str(template)
    if "{split_name}" not in text:
        raise DiagError(
            f"thresholds_cal path {text!r} does not contain '{{split_name}}'. "
            f"A calibrated threshold belongs to exactly one split; a fixed "
            f"path would let two splits share one file.")
    return Path(text.format(split_name=split_name))


def load_thresholds_cal(path, *, split_sha256=None) -> dict:
    """The calibrated thresholds, with the split they were calibrated on
    CHECKED against the split being run.

    A tau calibrated on `calibration.json` applied to `evaluation.json` rows
    would be an undeclared threshold: same key, different meaning, and
    nothing downstream could tell. Refuse it here.
    """
    p = Path(path)
    if not p.exists():
        raise DiagError(
            f"{p} not found — run tools/frozen_I2_thresholds.py for this "
            f"split before a sweep that records count_fixed_cal")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if split_sha256 is not None and doc.get("split_sha256") != split_sha256:
        raise DiagError(
            f"{p} was calibrated on split sha {doc.get('split_sha256')} but "
            f"this run's split hashes to {split_sha256} — a threshold from "
            f"another split is not this split's threshold")
    return doc


def tau_cal_for(doc: dict, cell: str, stage: str):
    """tau_cal[cell][stage], or MISSING when it was not calibrated."""
    value = (doc.get("tau_cal", {}).get(cell, {}) or {}).get(stage)
    return float(value) if isinstance(value, (int, float)) else MISSING


def write_thresholds_cal(path, doc: dict):
    """Write the calibrated thresholds atomically, refusing the canon file.

    The guard is on the RESOLVED path, so a relative path, a symlink or a
    `..` walk that lands on `fixed_thresholds_canon.json` is refused too.

    THE TEMP NAME IS PER-PROCESS. Every task of an array job writes its own
    run directory alone, but `thresholds_cal.json` is ONE file that all of
    them may add a key to. With a fixed `<name>.tmp`, two concurrent tasks
    write the same temp path and the first `os.replace` consumes it, so the
    second fails with

        FileNotFoundError: ... 'thresholds_cal.json.tmp' -> 'thresholds_cal.json'

    which is how task 0 of the I1 discovery array died (job 4263130,
    2026-09-17) while the other 18 succeeded. `os.replace` is atomic, so a
    unique source name is all that is needed for the write itself; use
    `update_thresholds_cal` for the read-modify-write.
    """
    p = Path(path)
    if p.name == Path(CANON_THRESHOLDS).name or \
            p.resolve() == Path(CANON_THRESHOLDS).resolve():
        raise DiagError(
            f"refusing to write {p}: {CANON_THRESHOLDS} is a historical "
            f"calibration and TASK I2 §10 forbids recalibrating it. The "
            f"per-stage thresholds are NEW KEYS in a NEW file.")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():                              # pragma: no cover
            tmp.unlink()
    return p


@contextmanager
def exclusive_lock(path, *, timeout=900.0, poll=1.0, stale=1800.0):
    """A crude cross-process lock for a SHARED output file.

    `os.open(O_CREAT | O_EXCL)` is atomic, which is all a lock needs. The
    holder writes its pid so a human reading the directory can see who has
    it, and a lock file older than `stale` is broken rather than deadlocking
    the array — a task killed at the wall clock must not block the next
    submission.

    Used for `thresholds_cal.json`, which every task of an array job may add
    a cell (I1/I2) or a model (I6) to. Without it, two tasks can both read
    the file, both add their own key and the later write silently drops the
    earlier one's.
    """
    lock = Path(f"{path}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + float(timeout)
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except OSError:
                continue                              # it vanished; retry
            if age > float(stale):
                lock.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise DiagError(
                    f"timed out after {timeout:.0f}s waiting for {lock}. If "
                    f"no job is running, delete it and resubmit.")
            time.sleep(float(poll))
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        fd = None
        yield lock
    finally:
        if fd is not None:                            # pragma: no cover
            os.close(fd)
        lock.unlink(missing_ok=True)


def update_thresholds_cal(path, fresh: dict):
    """Merge `fresh` into the thresholds file at `path`, race-safely.

    Read, merge and write all happen under one lock, so two array tasks
    calibrating different cells cannot lose each other's work. Returns the
    merged document.
    """
    p = Path(path)
    with exclusive_lock(p):
        existing = {}
        if p.exists():
            existing = json.loads(p.read_text(encoding="utf-8"))
        merged = merge_thresholds_cal(existing, fresh)
        # Derived from `sources`, so it is recomputed HERE rather than by the
        # caller: a caller that set it before the merge would be describing
        # the cells it knew about, not the cells the file ended up with.
        # I6's sources are models and carry no run_id, so the field is left
        # alone there.
        runs = sorted({s["run_id"] for s in (merged.get("sources") or {}).values()
                       if isinstance(s, dict) and s.get("run_id")})
        if runs:
            merged["baseline_run_ids"] = runs
        write_thresholds_cal(p, merged)
    return merged


def merge_thresholds_cal(existing: dict, fresh: dict) -> dict:
    """Append-safe merge: a cell already calibrated is NEVER recomputed away.

    A resubmitted job adds the cells that are missing and rewrites nothing
    (the standing rule after the legacy resume-overwrite incident). A cell
    whose recalibration disagrees with what is on disk is a hard error, not
    a silent overwrite.
    """
    if not existing:
        return fresh
    for key in ("split_sha256", "k", "definition"):
        if key in existing and existing[key] != fresh.get(key):
            raise DiagError(
                f"thresholds_cal.json records {key}={existing[key]!r} but "
                f"this pass computed {fresh.get(key)!r} — refusing to mix two "
                f"calibrations in one file")
    out = dict(existing)
    out["tau_cal"] = dict(existing.get("tau_cal", {}))
    out["sources"] = dict(existing.get("sources", {}))
    for cell, by_stage in fresh.get("tau_cal", {}).items():
        if cell in out["tau_cal"]:
            if out["tau_cal"][cell] != by_stage:
                raise DiagError(
                    f"cell {cell!r} is already calibrated with "
                    f"{out['tau_cal'][cell]} but this pass computed "
                    f"{by_stage} — refusing to overwrite a calibration")
            continue
        out["tau_cal"][cell] = by_stage
        out["sources"][cell] = fresh.get("sources", {}).get(cell)
    return out
