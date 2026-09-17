"""
saga/frozen/records.py
======================
The long-format record schema every work package writes, and the
append-safe atomic writers that put it on disk.

Two files per (work package, run):

    records.{parquet|csv}   one row per image x condition — the functional
                            response (nll, top-1, logit shift, the measured
                            perturbation norm)
    diag.{parquet|csv}      one row per image x condition — the patch
                            diagnostics (norm quantiles, fixed/MAD counts,
                            cosine both variants, effective rank)

Keyed the same way, so a join is `(run_id, condition_id, image_id)`.

Every row carries the TASK I0 §2.7 provenance columns: checkpoint sha256,
run_id, arch, recipe_actual, variant, n_prefix, stage, split sha, image_id,
condition_id, precision, git sha (+ dirty flag and patch file when dirty).
A number that cannot be traced back to a checkpoint and a split is not a
number this project reports.

Append-safe and atomic, in the sense the project settled on after the
legacy resume-overwrite incident: a writer appends complete rows to a
temp-then-renamed file and writes its completion marker LAST, so a job
killed mid-write leaves either the previous complete file or nothing extra
— never a half-row that a later reader averages.

Parquet is used when pyarrow is importable and CSV otherwise; the schema
and the file stem are identical either way, and `records_path()` tells a
reader which one is there.
"""

import csv
import json
import os
from pathlib import Path

try:                                             # optional, HPC-side
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
    HAVE_PARQUET = True
except ImportError:                              # pragma: no cover
    HAVE_PARQUET = False

MISSING = "MISSING"

#: §2.7 provenance — present on EVERY row of both files.
PROVENANCE_COLUMNS = (
    "run_id", "arch", "recipe_actual", "variant", "ckpt_kind", "ckpt_sha256",
    "n_prefix", "stage", "split_name", "split_sha256", "image_id",
    "condition_id", "precision", "git_sha", "git_dirty", "patch_file",
)

#: The functional response, one row per image x condition.
RECORD_COLUMNS = PROVENANCE_COLUMNS + (
    "edit_type", "edit_params", "layer", "epsilon",
    "measured_perturbation_norm", "nll", "correct", "top1", "label",
    "max_abs_logit_diff_vs_native",
)

#: The patch diagnostics, one row per image x condition, same key.
DIAG_COLUMNS = PROVENANCE_COLUMNS + (
    "edit_type", "layer",
    "patch_norm_q05", "patch_norm_q25", "patch_norm_median",
    "patch_norm_q75", "patch_norm_q95", "patch_norm_max", "patch_norm_mean",
    "count_fixed_thr", "fixed_thr_value", "count_mad_k5",
    "cos_pairwise", "cos_pairwise_nosink", "nosink_excluded",
    "eff_rank",
)

#: The per-STAGE diagnostics of a multi-stage sweep (TASK I2 §4), one row per
#: image x condition x stage. Written to the SAME `diag` file stem — a stage
#: sweep and a single-stage sweep are two column sets for one kind of file, so
#: `append_rows(..., "diag", rows, columns=DIAG_STAGE_COLUMNS)` selects this
#: one explicitly rather than a second file name drifting into the tree.
#:
#: `stage` (a provenance column) varies PER ROW here and joins the key;
#: `stage_resolved` carries what the alias resolved to, so a row that says
#: `hist` also says `s12_pre_norm` and a reader never has to know the alias.
#: `count_fixed_canon` and `tau_canon_value` are STRING columns: they are the
#: literal MISSING off `hist` (§4) and a parquet column cannot hold both a
#: float and a string, while a null would be silently skipped by a mean.
DIAG_STAGE_COLUMNS = PROVENANCE_COLUMNS + (
    "edit_type", "stage_resolved", "n_patches",
    "norm_p50", "norm_p90", "norm_p99", "norm_p999", "norm_max",
    "mad_thr", "tau_cal_value", "tau_canon_value",
    "count_fixed_canon", "count_fixed_cal", "count_mad",
    "cos_all", "cos_nosink_mad", "eff_rank",
)

#: The functional response of a sweep that also measures how big the edit
#: was (TASK B / I3 B2a). Two ADDITIVE columns on `RECORD_COLUMNS`, selected
#: explicitly by `append_rows(..., columns=RECORD_UPDATE_COLUMNS)` exactly as
#: `DIAG_STAGE_COLUMNS` is — a work package that measures the injected energy
#: and one that does not are two column sets for one kind of file, and
#: widening the shared schema would invalidate every `records.parquet` I2
#: already wrote.
#:
#: `delta_update_norm` is ||u_edit - u_native||_F over the patch rows of the
#: attention-branch residual update at the edited block (saga/frozen/reference.py).
#: `permutations_sha256` is the sha256 of the committed permutation file the
#: condition's `perm` was resolved from, on EVERY row — so a row whose
#: permutation came from an edited file can never be mistaken for one that
#: did not (TASK B §7, no sampling).
RECORD_UPDATE_COLUMNS = RECORD_COLUMNS + (
    "delta_update_norm", "permutations_sha256",
)

#: The functional response of a sweep that injects a MEASURED amount of
#: energy (TASK B / I4 B2b). Selected the same way, for the same reason.
#:
#: `measured_perturbation_norm` is ||dY_i||_F as the hook actually subtracted
#: it; `energy_target` is the per-image kappa_i a matched control was asked
#: for (the primary condition's measured norm on the SAME image) and the
#: literal MISSING for a fixed-epsilon condition; `energy_rel_error` is
#: |measured - target| / target, which TASK B §5 requires to be under 1%.
#: `mask_id` and `energy_match` name the D5 mask and the primary each row
#: belongs to, so a row is interpretable without re-reading the YAML, and
#: `masks_sha256` pins the file they were resolved from.
#:
#: All four numeric columns are STRING columns. They mix real values with the
#: literal MISSING — `native` has no epsilon, an unmatched control has no
#: target — and a parquet column cannot hold both a float and a string.
RECORD_PERTURBATION_COLUMNS = RECORD_COLUMNS + (
    "energy_target", "energy_rel_error", "mask_id", "energy_match",
    "n_masked_coords", "masks_sha256",
)

SCHEMAS = {"records": RECORD_COLUMNS, "diag": DIAG_COLUMNS}

#: Key for a per-stage diag file: one row per (condition, stage, image).
DIAG_STAGE_KEY = ("condition_id", "stage", "image_id")


def records_ext() -> str:
    return "parquet" if HAVE_PARQUET else "csv"


def records_path(out_dir, kind: str) -> Path:
    """Path of the `kind` file, whichever format is present or would be
    written. An existing file wins over the format preference, so a reader
    never misses a CSV written by a login node without pyarrow."""
    if kind not in SCHEMAS:
        raise ValueError(f"kind must be one of {sorted(SCHEMAS)}, got {kind!r}")
    out_dir = Path(out_dir)
    for ext in ("parquet", "csv"):
        p = out_dir / f"{kind}.{ext}"
        if p.exists():
            return p
    return out_dir / f"{kind}.{records_ext()}"


def done_marker(out_dir, kind: str) -> Path:
    return Path(out_dir) / f"{kind}.done.json"


def check_rows(rows, kind: str, columns=None):
    """Every row must carry exactly the schema's columns — no extra key that
    would be silently dropped, no missing key that would become an empty
    cell. Returns the column tuple.

    `columns` overrides the default schema for `kind` (a multi-stage sweep
    writes `DIAG_STAGE_COLUMNS` into the `diag` file); the check itself is
    unchanged, and a row set is still validated against exactly one schema.
    """
    cols = tuple(columns) if columns is not None else SCHEMAS[kind]
    want = set(cols)
    for i, row in enumerate(rows):
        got = set(row)
        if got != want:
            raise ValueError(
                f"{kind} row {i} does not match the schema:\n"
                f"  missing: {sorted(want - got)}\n"
                f"  unexpected: {sorted(got - want)}")
    return cols


def _atomic_replace(tmp: Path, path: Path):
    os.replace(tmp, path)


def _read_existing(path: Path, cols):
    """Rows already in the file, as dicts, or [] when it does not exist."""
    if not path.exists():
        return []
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.to_pylist()
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(out_dir, kind: str, rows, *, key=("condition_id", "image_id"),
                columns=None):
    """Append `rows` to the `kind` file, atomically and idempotently.

    Rows whose `key` tuple is already present are SKIPPED, so a resubmitted
    job adds what is missing and rewrites nothing — the standing rule after
    the legacy resume-overwrite incident. Returns (n_appended, n_skipped).

    The write is tmp + fsync + rename over the WHOLE file, so a reader never
    sees a partial row; the completion marker is written last.

    `key` widens for a file whose rows are keyed by more than (condition,
    image) — a per-stage sweep passes `DIAG_STAGE_KEY` — and `columns`
    selects a non-default schema for `kind`.
    """
    cols = check_rows(rows, kind, columns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = records_path(out_dir, kind)

    existing = _read_existing(path, cols)
    seen = {tuple(str(r.get(k, "")) for k in key) for r in existing}
    fresh, skipped = [], 0
    for r in rows:
        k = tuple(str(r.get(c, "")) for c in key)
        if k in seen:
            skipped += 1
            continue
        seen.add(k)
        fresh.append(r)
    if not fresh:
        return 0, skipped

    allrows = existing + fresh
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist([{c: r.get(c) for c in cols}
                                      for r in allrows],
                                     schema=None)
        pq.write_table(table, tmp)
    else:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cols))
            w.writeheader()
            for r in allrows:
                w.writerow({c: r.get(c, "") for c in cols})
            f.flush()
            os.fsync(f.fileno())
    _atomic_replace(tmp, path)
    return len(fresh), skipped


def mark_done(out_dir, kind: str, payload: dict):
    """Write the completion marker LAST (tools/check_done.py convention):
    its presence means every row of `kind` landed."""
    path = done_marker(out_dir, kind)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    _atomic_replace(tmp, path)
    return path


def is_done(out_dir, kind: str, ckpt_sha256: str) -> bool:
    """True iff `kind` completed for this exact checkpoint."""
    path = done_marker(out_dir, kind)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(
            "ckpt_sha256") == ckpt_sha256
    except (OSError, ValueError):
        return False


def provenance(*, run_id, arch, recipe_actual, variant, ckpt_kind,
               ckpt_sha256, n_prefix, stage, split_name, split_sha256,
               precision, git_sha, git_dirty, patch_file=MISSING) -> dict:
    """The §2.7 columns, filled once per (run, split, stage) and copied onto
    every row. `image_id` and `condition_id` are added per row."""
    return {
        "run_id": run_id, "arch": arch, "recipe_actual": recipe_actual,
        "variant": variant, "ckpt_kind": ckpt_kind,
        "ckpt_sha256": ckpt_sha256, "n_prefix": int(n_prefix), "stage": stage,
        "split_name": split_name, "split_sha256": split_sha256,
        "precision": precision, "git_sha": git_sha,
        "git_dirty": int(bool(git_dirty)),
        "patch_file": patch_file if patch_file else MISSING,
        "image_id": None, "condition_id": None,
    }
