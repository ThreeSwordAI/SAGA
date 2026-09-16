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

SCHEMAS = {"records": RECORD_COLUMNS, "diag": DIAG_COLUMNS}


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


def check_rows(rows, kind: str):
    """Every row must carry exactly the schema's columns — no extra key that
    would be silently dropped, no missing key that would become an empty
    cell. Returns the column tuple."""
    cols = SCHEMAS[kind]
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


def append_rows(out_dir, kind: str, rows, *, key=("condition_id", "image_id")):
    """Append `rows` to the `kind` file, atomically and idempotently.

    Rows whose `key` tuple is already present are SKIPPED, so a resubmitted
    job adds what is missing and rewrites nothing — the standing rule after
    the legacy resume-overwrite incident. Returns (n_appended, n_skipped).

    The write is tmp + fsync + rename over the WHOLE file, so a reader never
    sees a partial row; the completion marker is written last.
    """
    cols = check_rows(rows, kind)
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
