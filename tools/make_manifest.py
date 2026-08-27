#!/usr/bin/env python3
"""
tools/make_manifest.py
======================
Inventory every checkpoint (*.pth) under one or more roots into a CSV.

    python tools/make_manifest.py ROOT [ROOT ...] \
        -o results/legacy/checkpoint_manifest.csv [--no-hash]

Columns: path, filename, size_bytes, mtime_iso, sha256,
         exp, arch, recipe, variant, seed

The last five columns are best-effort regex guesses from the path and are
left EMPTY when unsure — the human fills gaps by hand. Never trust them
blindly; the manifest is the starting point for identifying the headline
runs, not the authority.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.run_registry import file_sha256


def guess_fields(path_str: str) -> dict:
    """Best-effort parse of exp/arch/recipe/variant/seed from a path.
    Empty string when unsure."""
    p = path_str.replace("\\", "/").lower()

    exp = ""
    m = re.search(r"(?:^|[/_\-])(e\d+)(?=[/_\-.]|$)", p)
    if m:
        exp = m.group(1)

    arch = ""
    if re.search(r"vit[-_]?small|vit[-_]s(?=[/_\-.]|$)", p):
        arch = "vit_small"
    elif re.search(r"vit[-_]?base|vit[-_]b(?=[/_\-.]|$)", p):
        arch = "vit_base"
    elif re.search(r"vit[-_]?large|vit[-_]l(?=[/_\-.]|$)", p):
        arch = "vit_large"
    elif re.search(r"vit[-_]?tiny|vit[-_]t(?=[/_\-.]|$)", p):
        arch = "vit_tiny"

    recipe = ""
    if "nomix" in p:
        recipe = "nomix"
    elif "mix" in p:
        recipe = "mixup"

    variant = ""
    if "saga" in p:
        variant = "saga"
    elif re.search(r"register|(?:^|[/_\-])reg\d*(?=[/_\-.]|$)", p):
        variant = "registers"
    elif "baseline" in p:
        variant = "baseline"

    seed = ""
    m = re.search(r"(?:^|[/_\-])s(?:eed)?[_\-]?(\d{1,2})(?=[/_\-.]|$)", p)
    if m:
        seed = m.group(1)

    return {"exp": exp, "arch": arch, "recipe": recipe,
            "variant": variant, "seed": seed}


def scan(roots, do_hash: bool):
    rows = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"WARNING: root does not exist, skipping: {root}", file=sys.stderr)
            continue
        found = sorted(root.rglob("*.pth"))
        print(f"{root}: {len(found)} .pth files")
        for f in found:
            stat = f.stat()
            row = {
                "path": f.resolve().as_posix(),
                "filename": f.name,
                "size_bytes": stat.st_size,
                "mtime_iso": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": file_sha256(f) if do_hash else "",
            }
            row.update(guess_fields(f.resolve().as_posix()))
            rows.append(row)
    return rows


def print_summary(rows):
    total_bytes = sum(r["size_bytes"] for r in rows)
    print(f"\n{'=' * 64}")
    print(f"  Manifest summary: {len(rows)} checkpoints, "
          f"{total_bytes / 1e9:.2f} GB total")
    print(f"{'=' * 64}")
    counts = Counter(
        (r["exp"] or "?", r["arch"] or "?", r["recipe"] or "?",
         r["variant"] or "?", r["seed"] or "?") for r in rows)
    header = f"  {'exp':<6}{'arch':<12}{'recipe':<8}{'variant':<11}{'seed':<6}{'n':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in sorted(counts):
        exp, arch, recipe, variant, seed = key
        print(f"  {exp:<6}{arch:<12}{recipe:<8}{variant:<11}{seed:<6}{counts[key]:>4}")
    n_gaps = sum(1 for r in rows if not all(
        (r["exp"], r["arch"], r["recipe"], r["variant"], r["seed"])))
    print(f"\n  rows with unfilled fields (human, please complete): {n_gaps}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively inventory *.pth checkpoints into a CSV manifest.")
    parser.add_argument("roots", nargs="+", metavar="ROOT",
                        help="directories to scan recursively")
    parser.add_argument("-o", "--out", required=True,
                        help="output CSV, e.g. results/legacy/checkpoint_manifest.csv")
    parser.add_argument("--no-hash", action="store_true",
                        help="skip sha256 (fast inventory pass)")
    args = parser.parse_args()

    rows = scan(args.roots, do_hash=not args.no_hash)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "filename", "size_bytes", "mtime_iso", "sha256",
                  "exp", "arch", "recipe", "variant", "seed"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out}")

    print_summary(rows)


if __name__ == "__main__":
    main()
