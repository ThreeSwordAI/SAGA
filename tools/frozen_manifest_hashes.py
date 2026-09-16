#!/usr/bin/env python3
"""
tools/frozen_manifest_hashes.py
===============================
TASK I0 D6 — sha256 every checkpoint the manifest names. Runs on the HPC
(CPU, login node or a small job), where the checkpoints actually live.

    python tools/frozen_manifest_hashes.py
        [--manifest results/frozen/I0_manifest/manifest.csv]
        [--out results/frozen/I0_manifest/ckpt_hashes.json]

Then fold the result back into the manifest, locally or on the HPC:

    python analysis/build_I0_manifest.py --hashes \
        results/frozen/I0_manifest/ckpt_hashes.json

Stdlib only — no torch, no timm, no conda env needed, so an unactivated
login node still runs it (TASK-09 lost a sync to the opposite assumption).

RESUME-SAFE AND APPEND-ONLY. An existing output file is READ FIRST and its
entries are kept: a path already hashed is not re-read (these are 0.26-1.2 GB
files and there are ~90 of them), and a path whose hash DISAGREES with what
is already recorded is a hard error, not an overwrite. The completion marker
`complete: true` is written LAST, so a job killed mid-pass simply re-runs and
continues (the standing idempotency rule after the legacy resume-overwrite
incident).

A checkpoint the manifest names but the filesystem does not have is recorded
as MISSING with the reason — never silently skipped, and never guessed at.

An entry for a path the manifest NO LONGER NAMES is a different thing and is
reported as such, under `no_longer_in_manifest`. It means a manifest
correction renamed or dropped that path, not that a checkpoint went missing;
reporting the two together once made a complete run look like six missing
files.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MISSING = "MISSING"
CHUNK = 1 << 20


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def wanted_paths(manifest_csv: Path):
    """[(ckpt_path, recorded_sha_or_MISSING)] — unique, in manifest order.

    `ckpt_kind == 'derived'` rows (test-time registers) have no checkpoint of
    their own and are skipped; their model IS the base run's checkpoint,
    which appears in the manifest under its own row.
    """
    out, seen = [], set()
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path = r["ckpt_path"]
            if not path or path == MISSING or r["ckpt_kind"] == "derived":
                continue
            if path in seen:
                continue
            seen.add(path)
            out.append((path, r.get("ckpt_sha256", MISSING)))
    return out


def load_existing(out_path: Path):
    if not out_path.exists():
        return {}, {}
    try:
        doc = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    return doc.get("sha256_by_path", {}), doc.get("absent", {})


def main():
    p = argparse.ArgumentParser(
        description="sha256 every checkpoint named by the I0 manifest.")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.csv")
    p.add_argument("--out",
                   default="results/frozen/I0_manifest/ckpt_hashes.json")
    p.add_argument("--limit", type=int, default=None,
                   help="hash at most N new files this pass (resume-safe: "
                        "re-run to continue)")
    args = p.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        sys.exit(f"{manifest} not found — run analysis/build_I0_manifest.py "
                 f"first (and `git pull` if it was built locally)")
    out_path = Path(args.out)
    sha_by_path, absent = load_existing(out_path)
    targets = wanted_paths(manifest)

    n_new = n_kept = n_absent = 0
    for path, recorded in targets:
        if path in sha_by_path:
            n_kept += 1
            continue
        f = Path(path)
        if not f.exists():
            absent[path] = "file not found on this filesystem"
            n_absent += 1
            continue
        if args.limit is not None and n_new >= args.limit:
            break
        try:
            got = file_sha256(f)
        except OSError as exc:
            absent[path] = f"unreadable: {exc}"
            n_absent += 1
            continue
        if recorded not in (MISSING, "", None) and recorded != got:
            sys.exit(
                f"HASH MISMATCH for {path}\n"
                f"  manifest records {recorded}\n"
                f"  file hashes to   {got}\n"
                f"The checkpoint changed since its provenance was recorded. "
                f"Refusing to overwrite a historical value — investigate "
                f"before anything else runs against it.")
        sha_by_path[path] = got
        absent.pop(path, None)
        n_new += 1
        print(f"  {got[:16]}…  {path}", flush=True)
        # write through after every file: a 24 h job killed at hour 23 keeps
        # everything it hashed
        atomic_write_json({
            "generated_by": "tools/frozen_manifest_hashes.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest),
            "complete": False,
            "n_targets": len(targets),
            "sha256_by_path": sha_by_path,
            "absent": absent,
        }, out_path)

    # An `absent` entry for a path the manifest NO LONGER NAMES is stale
    # bookkeeping, not a missing checkpoint: it means a manifest correction
    # renamed or dropped that path (the dense best.pth correction did exactly
    # this). Reporting it alongside the live ones reads as "6 checkpoints are
    # missing" when nothing is. Live entries stay; stale ones move aside with
    # the reason they went stale, so the history is kept without misstating
    # the present.
    target_paths = {p for p, _ in targets}
    stale = {p: w for p, w in absent.items() if p not in target_paths}
    absent = {p: w for p, w in absent.items() if p in target_paths}
    for p in stale:
        sha_by_path.pop(p, None)

    remaining = [p for p, _ in targets
                 if p not in sha_by_path and p not in absent]
    atomic_write_json({
        "generated_by": "tools/frozen_manifest_hashes.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "complete": not remaining,
        "n_targets": len(targets),
        "n_hashed": len(sha_by_path),
        "n_absent": len(absent),
        "n_remaining": len(remaining),
        "sha256_by_path": sha_by_path,
        "absent": absent,
        "no_longer_in_manifest": stale,
    }, out_path)

    print(f"\n{len(targets)} checkpoint paths in the manifest")
    print(f"  {n_new} hashed this pass, {n_kept} already recorded, "
          f"{len(absent)} absent, {len(remaining)} still to hash")
    if absent:
        print(f"ABSENT ({len(absent)}) — a path the manifest names that this "
              f"filesystem does not have; recorded as MISSING, never guessed:")
        for path, why in sorted(absent.items()):
            print(f"  {path}: {why}")
    if stale:
        print(f"\nnot in the manifest any more ({len(stale)}) — NOT missing "
              f"data; a manifest correction renamed or dropped these paths:")
        for path in sorted(stale):
            print(f"  {path}")
    if remaining:
        print(f"{len(remaining)} still to hash — re-run to continue")
    print(f"wrote {out_path}")
    print("\nNext: python analysis/build_I0_manifest.py --hashes "
          f"{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
