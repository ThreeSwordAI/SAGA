#!/usr/bin/env python3
"""
tools/build_frozen_splits.py
============================
TASK I0 D2 — freeze the calibration and locked-evaluation image splits.
Written ONCE, committed, never rebuilt.

    # on a login node (stdlib only — no torch, no timm, no conda env needed)
    python tools/build_frozen_splits.py --data $STAGE_DIR

Four files under `results/frozen/splits/`:

    calibration.json   2,000 val images, 2 per class   — thresholds, empirical
                                                         frequencies, setup
    evaluation.json   10,000 val images, 10 per class  — the LOCKED split for
                                                         every new intervention
    sub1k.json         1,000 images drawn from evaluation — SVD / attention
    sub2k.json         2,000 images drawn from evaluation — correspondence

DISJOINTNESS is by CONSTRUCTION, not by a check afterwards:

    discovery   = results/diagsplit/val_diag_split.json  (unchanged; its
                  sha256 recorded here)
    calibration = stratified draw from (all val) MINUS discovery
    evaluation  = stratified draw from (all val) MINUS discovery MINUS
                  calibration
    sub1k, sub2k = seeded draws from evaluation (nested inside it on purpose)

An image that already belongs to an earlier split is never a candidate for a
later one, so the pairwise-disjointness test can only ever confirm what the
construction guarantees.

WRITE-ONCE: an existing file is REFUSED, and each file carries a `sha256`
over its own canonical serialization so a later build re-verifies that the
earlier ones were not perturbed — the pattern
`tools/build_probe_set.py` established and this file follows.

DETERMINISM: every choice is `random.Random(f"{seed}:{name}:{synset}")` over
a SORTED candidate list, so two builds from the same val listing produce
byte-identical JSON. Paths are stored RELATIVE; a dataset index means
something different the moment a directory is re-listed.

WHAT THIS SPLIT IS, AND IS NOT
------------------------------
The 50k accuracies have already been seen. This is therefore a **locked
evaluation protocol for the new analyses**, not an untouched test set and
not a retrospectively preregistered study. That sentence is written into
`results/frozen/splits/README.md` by this tool and must survive into the
paper (plan §6.2).
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

IMG_EXTS = {".jpeg", ".jpg", ".png"}

SPLITS = ("calibration", "evaluation", "sub1k", "sub2k")

#: (per-class draw, total) for the two stratified splits.
N_PER_CLASS = {"calibration": 2, "evaluation": 10}
N_SUB = {"sub1k": 1000, "sub2k": 2000}

DEFAULT_SEED = 0
DISCOVERY = "results/diagsplit/val_diag_split.json"

PROTOCOL_SENTENCE = (
    "The original full-validation accuracies have already been seen, so this "
    "is a LOCKED EVALUATION PROTOCOL FOR THE NEW ANALYSES — not a completely "
    "untouched test set, and not a retrospectively preregistered study.")


# ── canonical hashing (same shape as tools/build_probe_set.py) ───────────────

def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def split_sha256(split: dict) -> str:
    """sha256 over a split's content, EXCLUDING its own sha field."""
    payload = {k: v for k, v in split.items() if k != "sha256"}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(obj, path) -> None:
    """tmp + fsync + rename. Deliberately stdlib-only: this tool runs on a
    LOGIN node where the conda env may not be active (TASK-09 lost a sync to
    exactly that assumption)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── the val listing ─────────────────────────────────────────────────────────

def list_val(data_root):
    """{synset: [filename, ...]} over the ImageNet val tree, all sorted.

    The same listing `tools/build_diag_split.py` walks, so class ids follow
    the ImageFolder convention (sorted synset directory names) and agree
    with the discovery split's labels.
    """
    val_dir = Path(data_root) / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"{val_dir} not found — --data must be an ImageNet root with a "
            f"val/<synset>/*.JPEG layout (e.g. a staged $STAGE_DIR)")
    classes = sorted(d.name for d in val_dir.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class directories under {val_dir}")
    return {c: sorted(p.name for p in (val_dir / c).iterdir()
                      if p.suffix.lower() in IMG_EXTS)
            for c in classes}


def load_claimed(discovery_json):
    """(relative paths already spoken for, discovery sha256, n)."""
    path = Path(discovery_json)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the DISCOVERY split is an input to this "
            f"build, not something it creates. It is committed; run from the "
            f"repo root.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return ({rel for rel, _ in doc["items"]}, file_sha256(path),
            int(doc.get("n", len(doc["items"]))))


# ── builders ────────────────────────────────────────────────────────────────

def image_id_for(rel: str) -> str:
    """`<synset>/<stem>` — the record key, stable under a re-listing."""
    p = Path(rel)
    return f"{p.parent.name}/{p.stem}"


def build_stratified(name, listing, claimed, seed):
    """`N_PER_CLASS[name]` images per class, drawn only from UNCLAIMED files.

    A per-class random stream keyed by the synset means adding a class, or
    building the two splits in either order, cannot reshuffle what the other
    classes drew.
    """
    per_class = N_PER_CLASS[name]
    classes = sorted(listing)
    class_index = {c: i for i, c in enumerate(classes)}
    items, short = [], []
    for synset in classes:
        pool = [f for f in listing[synset]
                if f"val/{synset}/{f}" not in claimed]
        take = min(per_class, len(pool))
        if take < per_class:
            short.append((synset, len(pool)))
        rng = random.Random(f"{seed}:{name}:{synset}")
        for fname in sorted(rng.sample(pool, take)):
            items.append([f"val/{synset}/{fname}", class_index[synset]])
    if short:
        raise ValueError(
            f"{name}: {len(short)} class(es) have fewer than {per_class} "
            f"unclaimed images, e.g. {short[:5]}. Refusing to build an "
            f"unbalanced split silently.")
    return {
        "name": name,
        "status": "frozen",
        "dataset": "imagenet-1k",
        "split": "val",
        "seed": seed,
        "n": len(items),
        "n_per_class": per_class,
        "n_classes": len(classes),
        "selector": (f"per-class Random(f'{{seed}}:{name}:{{synset}}'); "
                     f"sample(sorted(unclaimed files), {per_class})"),
        "disjoint_from": [],          # filled by the caller
        "protocol": PROTOCOL_SENTENCE,
        "items": items,
    }


def build_subset(name, parent, seed):
    """A fixed subset DRAWN FROM `parent` (nested on purpose).

    Class-spread: the draw walks the parent's classes in a seeded order and
    takes images round-robin, so a 1,000-image subset of a 10-per-class
    10,000-image split covers 1,000 distinct classes rather than 100 classes
    ten times over.
    """
    n = N_SUB[name]
    by_class = {}
    for rel, label in parent["items"]:
        by_class.setdefault(label, []).append([rel, label])
    labels = sorted(by_class)
    rng = random.Random(f"{seed}:{name}")
    order = labels[:]
    rng.shuffle(order)
    for lab in labels:
        by_class[lab] = sorted(by_class[lab])
        rng.shuffle(by_class[lab])

    chosen, cursor = [], 0
    while len(chosen) < n:
        progressed = False
        for lab in order:
            if cursor < len(by_class[lab]):
                chosen.append(by_class[lab][cursor])
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            raise ValueError(
                f"{name}: parent split has only {len(parent['items'])} "
                f"images, need {n}")
        cursor += 1
    chosen = sorted(chosen, key=lambda it: it[0])
    return {
        "name": name,
        "status": "frozen",
        "dataset": "imagenet-1k",
        "split": "val",
        "seed": seed,
        "n": len(chosen),
        "n_classes": len({lab for _, lab in chosen}),
        "parent": parent["name"],
        "parent_sha256": parent["sha256"],
        "selector": (f"Random(f'{{seed}}:{name}'); class-round-robin over the "
                     f"parent split's classes in a seeded order"),
        "disjoint_from": [],
        "protocol": PROTOCOL_SENTENCE,
        "items": chosen,
    }


# ── write-once guards ───────────────────────────────────────────────────────

def refuse_existing(out_dir: Path, names):
    """Refuse to touch a split that is already frozen."""
    existing = [n for n in names if (out_dir / f"{n}.json").exists()]
    if existing:
        raise SystemExit(
            f"REFUSING to overwrite frozen split(s) {existing} under "
            f"{out_dir}. These files are write-once: every number computed "
            f"on them is keyed by their sha256, so rebuilding one silently "
            f"invalidates every record that cites it. Delete them by hand "
            f"and say so in the TASK_LOG if a rebuild is genuinely intended.")


def verify_frozen(out_dir: Path):
    """Every already-frozen split must still hash to its recorded sha256."""
    for name in SPLITS:
        path = out_dir / f"{name}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        want, got = doc.get("sha256"), split_sha256(doc)
        if want != got:
            raise SystemExit(
                f"{path} does not match its recorded sha256 (recorded {want}, "
                f"computed {got}) — a frozen split has been edited. Refusing "
                f"to touch it.")


def check_disjoint(splits: dict, discovery_paths):
    """Pairwise disjointness, including against the discovery split.

    sub1k/sub2k are DRAWN FROM evaluation, so they are checked to be SUBSETS
    of it rather than disjoint from it — stating which relation is expected
    is the point of the check.
    """
    paths = {n: {rel for rel, _ in s["items"]} for n, s in splits.items()}
    paths["discovery"] = set(discovery_paths)
    must_be_disjoint = [("discovery", "calibration"),
                        ("discovery", "evaluation"),
                        ("calibration", "evaluation")]
    for a, b in must_be_disjoint:
        if a not in paths or b not in paths:
            continue
        overlap = paths[a] & paths[b]
        if overlap:
            raise SystemExit(
                f"{a} and {b} share {len(overlap)} image(s), e.g. "
                f"{sorted(overlap)[:3]} — they must be disjoint")
    for sub in ("sub1k", "sub2k"):
        if sub in paths and "evaluation" in paths:
            stray = paths[sub] - paths["evaluation"]
            if stray:
                raise SystemExit(
                    f"{sub} has {len(stray)} image(s) outside evaluation, "
                    f"e.g. {sorted(stray)[:3]} — it must be a SUBSET")


README = """# results/frozen/splits — the locked image splits

Built once by `tools/build_frozen_splits.py`, committed, never rebuilt. Each
file carries a `sha256` over its own canonical serialization; every record
written by `saga/frozen/runner.py` cites that sha, so a split that changed
would invalidate every number keyed to it — which is why the builder refuses
to overwrite one.

| file | n | drawn from | used for |
|---|---|---|---|
| (discovery) `results/diagsplit/val_diag_split.json` | 10,000 | val | the EXISTING diagnostic split; unchanged, its sha recorded here |
| `calibration.json` | 2,000 (2/class) | val minus discovery | thresholds, empirical frequencies, label-independent setup |
| `evaluation.json` | 10,000 (10/class) | val minus discovery minus calibration | every new frozen-model intervention |
| `sub1k.json` | 1,000 | evaluation | SVD / full attention extraction |
| `sub2k.json` | 2,000 | evaluation | correspondence |

`calibration` and `evaluation` are disjoint from the discovery split and from
each other BY CONSTRUCTION — an image claimed by an earlier split is never a
candidate for a later one. `sub1k` and `sub2k` are nested INSIDE `evaluation`
on purpose, and are checked to be subsets of it.

## What this protocol is, and is not

{protocol}

The discovery split and the current maps have already informed the
hypotheses, and the 50k accuracies have already been seen. What the lock
buys is that the split IDs and the analysis configuration
(`docs/LOCKED_ANALYSIS.md`) were fixed before the new interventions' outcomes
were inspected — not that the data is untouched.
"""


def main():
    p = argparse.ArgumentParser(
        description="Freeze the TASK I0 calibration/evaluation splits "
                    "(write-once).")
    p.add_argument("--data", required=True, metavar="ROOT",
                   help="ImageNet root containing val/ (e.g. $STAGE_DIR)")
    p.add_argument("--discovery", default=DISCOVERY,
                   help="the existing frozen 10k diagnostic split")
    p.add_argument("--out-dir", default="results/frozen/splits")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--if-missing", action="store_true",
                   help="skip (exit 0) splits that are already frozen instead "
                        "of failing — makes a job script resubmit-safe")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    verify_frozen(out_dir)

    want = [n for n in SPLITS
            if not (args.if_missing and (out_dir / f"{n}.json").exists())]
    if not want:
        print(f"{out_dir}: all four splits already frozen — nothing to do")
        return 0
    refuse_existing(out_dir, want)
    if set(want) != set(SPLITS):
        raise SystemExit(
            f"partial build requested ({want}) but the four splits are "
            f"constructed together — calibration claims images before "
            f"evaluation draws, and the subsets are drawn from evaluation. "
            f"Build all four or none.")

    claimed, disc_sha, disc_n = load_claimed(args.discovery)
    print(f"discovery: {disc_n} images, sha256={disc_sha[:16]}…")

    listing = list_val(args.data)
    n_val = sum(len(v) for v in listing.values())
    print(f"val listing: {len(listing)} classes, {n_val} images")

    splits = {}
    calib = build_stratified("calibration", listing, claimed, args.seed)
    calib["disjoint_from"] = ["discovery"]
    calib["discovery_sha256"] = disc_sha
    calib["sha256"] = split_sha256(calib)
    splits["calibration"] = calib
    claimed |= {rel for rel, _ in calib["items"]}

    ev = build_stratified("evaluation", listing, claimed, args.seed)
    ev["disjoint_from"] = ["discovery", "calibration"]
    ev["discovery_sha256"] = disc_sha
    ev["calibration_sha256"] = calib["sha256"]
    ev["sha256"] = split_sha256(ev)
    splits["evaluation"] = ev

    for name in ("sub1k", "sub2k"):
        sub = build_subset(name, ev, args.seed)
        sub["disjoint_from"] = ["discovery", "calibration"]
        sub["sha256"] = split_sha256(sub)
        splits[name] = sub

    check_disjoint(splits, load_claimed(args.discovery)[0])

    for name in SPLITS:
        atomic_write_json(splits[name], out_dir / f"{name}.json")
    readme = out_dir / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    with open(readme, "w", encoding="utf-8", newline="\n") as f:
        f.write(README.format(protocol=PROTOCOL_SENTENCE))

    verify_frozen(out_dir)
    for name in SPLITS:
        s = splits[name]
        print(f"wrote {name:12s} n={s['n']:6d} classes={s.get('n_classes'):5d} "
              f"sha256={s['sha256']}")
    print(f"-> {out_dir}")
    print("\nRecord these shas in docs/LOCKED_ANALYSIS.md before I3/I4 Phase B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
