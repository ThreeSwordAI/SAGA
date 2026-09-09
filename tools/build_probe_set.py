#!/usr/bin/env python3
"""
tools/build_probe_set.py
========================
Freeze the TASK-11 probe set: the images Figure 1 and every probe-set
statistic are computed on. Written ONCE, committed, never rebuilt.

    # ImageNet groups (needs only a val/<synset>/*.JPEG listing)
    python tools/build_probe_set.py --groups imagenet --data $STAGE_DIR

    # COCO group (needs instances_val2017.json), filled independently, later
    python tools/build_probe_set.py --groups boxes20 \
        --coco-ann /path/to/annotations/instances_val2017.json

Three groups, each frozen independently so the COCO group can land in a
later job WITHOUT touching (or being able to touch) the ImageNet groups:

  curated    12 ImageNet-val images, 4 per stated criterion — the Figure-1
             candidates. NEVER the basis of a statistic.
  random200  200 ImageNet-val images, one per class over 200 seeded classes
             (maximal class spread) — the population for every ImageNet
             probe statistic.
  boxes20    20 COCO-val images with GT boxes — the population for the
             in-box attention metric. `status: pending` until a job with
             COCO available fills it.

Determinism: every choice is `random.Random(seed)` over a SORTED candidate
list, so two builds of the same group from the same data produce byte-identical
JSON. Paths are stored RELATIVE (never dataset indices — a dataset index means
something different the moment a directory is re-listed).

Write-once: a group already marked `frozen` is refused, and each frozen group
carries a `sha256` over its own canonical serialization, so a later group's
build re-verifies that the earlier ones were not perturbed.

CURATED CRITERIA — stated, and honest about what they are:
each criterion is a CLASS-LEVEL prior (`criterion_basis: class_prior`), not a
per-image verification. The pools below say which synsets stand for which
criterion and why. A human who wants per-image control passes
`--curated-paths FILE` (one `val/<synset>/<file>.JPEG,<criterion>` per line),
which is recorded as `criterion_basis: explicit_paths`.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

IMG_EXTS = {".jpeg", ".jpg", ".png"}

GROUPS = ("curated", "random200", "boxes20")
IMAGENET_GROUPS = ("curated", "random200")

N_CURATED_PER_CRITERION = 4
N_RANDOM = 200
N_BOXES = 20

# ── Curated criterion pools ──────────────────────────────────────────────────
# Class-level priors. 8 candidate synsets per criterion; the build seeds-picks
# N_CURATED_PER_CRITERION of them, then one image from each chosen class.
CURATED_POOLS = {
    "single_dominant_centered": {
        "rationale": "ILSVRC classes whose val images are typically ONE large, "
                     "roughly centred object against a plain or shallow field.",
        "synsets": [
            "n02123045",  # tabby cat
            "n02110063",  # malamute
            "n03272010",  # electric guitar
            "n02504458",  # African elephant
            "n02951358",  # canoe
            "n03000684",  # chain saw
            "n03793489",  # mouse (computer)
            "n01608432",  # kite (bird)
        ],
    },
    "textured_cluttered_background": {
        "rationale": "Scene/landscape classes: no single figure-ground object, "
                     "high-frequency texture over most of the frame.",
        "synsets": [
            "n09332890",  # lakeside
            "n09428293",  # seashore
            "n03160309",  # dam
            "n09399592",  # promontory
            "n02793495",  # barn
            "n09246464",  # cliff
            "n04604644",  # worm fence
            "n03649909",  # lawn mower
        ],
    },
    "small_or_offcentre": {
        "rationale": "Small-animal/insect classes: the labelled object usually "
                     "occupies a small, often off-centre part of the frame.",
        "synsets": [
            "n01530575",  # brambling
            "n02165456",  # ladybug
            "n02219486",  # ant
            "n01770081",  # harvestman
            "n02268443",  # dragonfly
            "n01737021",  # water snake
            "n02233338",  # cockroach
            "n01755581",  # diamondback rattlesnake
        ],
    },
}

SCHEMA_VERSION = 1


# ── canonical hashing ────────────────────────────────────────────────────────

def atomic_write_json(obj, path) -> None:
    """tmp + fsync + rename. Deliberately local rather than
    `tools.dense_runtime.atomic_json_dump`: that module imports torch, and
    this tool is meant to run on a LOGIN node where the conda env may not be
    active (TASK-09 lost a sync to exactly that assumption)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def canonical_bytes(obj) -> bytes:
    """Stable serialization used for every group sha256."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def group_sha256(group: dict) -> str:
    """sha256 over a group's content, EXCLUDING its own sha field."""
    payload = {k: v for k, v in group.items() if k != "sha256"}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


# ── ImageNet groups ──────────────────────────────────────────────────────────

def list_val_classes(data_root) -> list:
    val_dir = Path(data_root) / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"{val_dir} not found — --data must be an ImageNet root with a "
            f"val/<synset>/*.JPEG layout (e.g. a staged $STAGE_DIR)")
    classes = sorted(d.name for d in val_dir.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class directories under {val_dir}")
    return classes


def list_class_files(data_root, synset) -> list:
    d = Path(data_root) / "val" / synset
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def build_curated(data_root, seed: int, classes: list) -> dict:
    """4 images per criterion, seeded, from the stated class pools."""
    class_index = {c: i for i, c in enumerate(classes)}
    items = []
    for criterion in sorted(CURATED_POOLS):
        pool = CURATED_POOLS[criterion]
        missing = [s for s in pool["synsets"] if s not in class_index]
        if missing:
            raise KeyError(
                f"criterion {criterion!r}: synsets absent from this ImageNet "
                f"val root: {missing}. Refusing to silently substitute — the "
                f"curated set must mean what it says.")
        # separate stream per criterion: adding a criterion cannot reshuffle
        # the images already chosen for the others
        rng = random.Random(f"{seed}:curated:{criterion}")
        chosen = sorted(rng.sample(sorted(pool["synsets"]),
                                   N_CURATED_PER_CRITERION))
        for synset in chosen:
            files = list_class_files(data_root, synset)
            if not files:
                raise FileNotFoundError(f"no images in val/{synset}")
            name = rng.choice(files)
            items.append({
                "image_id": f"{synset}/{Path(name).stem}",
                "path": f"val/{synset}/{name}",
                "label": class_index[synset],
                "synset": synset,
                "criterion": criterion,
                "criterion_basis": "class_prior",
                "criterion_rationale": pool["rationale"],
            })
    return {
        "status": "frozen",
        "dataset": "imagenet-1k",
        "split": "val",
        "seed": seed,
        "n": len(items),
        "selector": "per-criterion Random(f'{seed}:curated:{criterion}'); "
                    "sample(sorted(pool_synsets), 4) then choice(sorted(files))",
        "criteria": {k: v["rationale"] for k, v in sorted(CURATED_POOLS.items())},
        "items": items,
    }


def build_random200(data_root, seed: int, classes: list, exclude=()) -> dict:
    """One image from each of N_RANDOM seeded classes — maximal class spread.

    `exclude` holds relative paths already spoken for by another group. The
    population and the displayed panels must be disjoint, and enforcing that
    by CONSTRUCTION beats discovering it in check_disjoint: a curated image
    that also sat in random200 would make Figure 1's illustrations part of
    the statistic they are captioned with."""
    if len(classes) < N_RANDOM:
        raise ValueError(
            f"need >= {N_RANDOM} classes for class-spread sampling, "
            f"found {len(classes)}")
    class_index = {c: i for i, c in enumerate(classes)}
    exclude = set(exclude)
    rng = random.Random(f"{seed}:random200")
    chosen = sorted(rng.sample(classes, N_RANDOM))
    items = []
    for synset in chosen:
        files = [n for n in list_class_files(data_root, synset)
                 if f"val/{synset}/{n}" not in exclude]
        if not files:
            raise FileNotFoundError(
                f"no unclaimed images in val/{synset} (all are in another "
                f"probe group)")
        name = rng.choice(files)
        items.append({
            "image_id": f"{synset}/{Path(name).stem}",
            "path": f"val/{synset}/{name}",
            "label": class_index[synset],
            "synset": synset,
        })
    return {
        "status": "frozen",
        "dataset": "imagenet-1k",
        "split": "val",
        "seed": seed,
        "n": len(items),
        "n_classes": len({i["synset"] for i in items}),
        "selector": "Random(f'{seed}:random200'); sample(sorted(classes), 200) "
                    "then choice(sorted(files)) — one image per class",
        "items": items,
    }


# ── COCO group ───────────────────────────────────────────────────────────────

def build_boxes20(coco_ann, seed: int) -> dict:
    """20 COCO-val images carrying their GT boxes, in ORIGINAL image
    coordinates (xywh, the COCO convention). Mapping them into a model's
    input frame is the consumer's job — see tools/localization_score.py,
    and TASK-09's frame bug for why that mapping is never assumed."""
    with open(coco_ann, encoding="utf-8") as f:
        ann = json.load(f)

    by_image = {}
    for a in ann.get("annotations", []):
        if a.get("iscrowd", 0):
            continue
        by_image.setdefault(a["image_id"], []).append(a)

    # candidates: images with >=1 non-crowd box, sorted by id for determinism
    images = {im["id"]: im for im in ann.get("images", [])}
    candidates = sorted(i for i in by_image if i in images)
    if len(candidates) < N_BOXES:
        raise ValueError(
            f"only {len(candidates)} annotated COCO val images available, "
            f"need {N_BOXES}")

    rng = random.Random(f"{seed}:boxes20")
    chosen = sorted(rng.sample(candidates, N_BOXES))

    items = []
    for image_id in chosen:
        im = images[image_id]
        boxes = [[float(v) for v in a["bbox"]] for a in
                 sorted(by_image[image_id], key=lambda a: a["id"])]
        items.append({
            "image_id": im["file_name"].rsplit(".", 1)[0],
            "path": f"val2017/{im['file_name']}",
            "coco_image_id": int(image_id),
            "width": int(im["width"]),
            "height": int(im["height"]),
            "boxes_xywh": boxes,
            "n_boxes": len(boxes),
        })
    return {
        "status": "frozen",
        "dataset": "coco",
        "split": "val2017",
        "seed": seed,
        "n": len(items),
        "box_format": "xywh, ORIGINAL image pixel coordinates, iscrowd excluded",
        "selector": "Random(f'{seed}:boxes20'); "
                    "sample(sorted(annotated_image_ids), 20)",
        "items": items,
    }


def pending_boxes20() -> dict:
    return {
        "status": "pending",
        "dataset": "coco",
        "split": "val2017",
        "n": 0,
        "reason": "COCO val annotations not available at build time; fill with "
                  "--groups boxes20 --coco-ann .../instances_val2017.json",
        "items": [],
    }


# ── document assembly ────────────────────────────────────────────────────────

def load_existing(out: Path):
    if not out.exists():
        return None
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def verify_frozen_unchanged(doc: dict) -> None:
    """Every frozen group must still hash to its recorded sha256."""
    for name in GROUPS:
        g = doc.get("groups", {}).get(name)
        if not g or g.get("status") != "frozen":
            continue
        want = g.get("sha256")
        got = group_sha256(g)
        if want != got:
            raise ValueError(
                f"group {name!r} does not match its recorded sha256 "
                f"(recorded {want}, computed {got}) — the frozen probe set has "
                f"been edited. Refusing to touch it.")


def assemble(existing, new_groups: dict, seed: int) -> dict:
    doc = existing or {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "groups": {},
    }
    doc.setdefault("groups", {})

    for name, group in new_groups.items():
        prev = doc["groups"].get(name)
        if prev is not None and prev.get("status") == "frozen":
            raise ValueError(
                f"group {name!r} is already frozen ({prev.get('n')} items) — "
                f"the probe set is frozen forever. Refusing to rebuild it.")
        group["sha256"] = group_sha256(group)
        doc["groups"][name] = group

    for name in GROUPS:
        doc["groups"].setdefault(name, pending_boxes20() if name == "boxes20"
                                 else None)
    doc["groups"] = {k: doc["groups"][k] for k in GROUPS
                     if doc["groups"].get(k) is not None}
    return doc


def check_disjoint(doc: dict) -> None:
    """No image may appear in two groups (a curated image inside random200
    would make the population statistics and the displayed panels dependent)."""
    seen = {}
    for name in GROUPS:
        g = doc["groups"].get(name)
        if not g or g.get("status") != "frozen":
            continue
        for it in g["items"]:
            key = (g["dataset"], it["path"])
            if key in seen:
                raise ValueError(
                    f"{it['path']} appears in both {seen[key]!r} and {name!r} "
                    f"— probe groups must be disjoint")
            seen[key] = name


def parse_curated_paths(path, data_root, classes) -> dict:
    """`val/<synset>/<file>.JPEG,<criterion>` per line — human override."""
    class_index = {c: i for i, c in enumerate(classes)}
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<relpath>,<criterion>'")
            rel, criterion = parts
            if criterion not in CURATED_POOLS:
                raise ValueError(
                    f"{path}:{lineno}: unknown criterion {criterion!r}; "
                    f"expected one of {sorted(CURATED_POOLS)}")
            if not (Path(data_root) / rel).is_file():
                raise FileNotFoundError(f"{path}:{lineno}: {rel} does not exist")
            synset = Path(rel).parent.name
            if synset not in class_index:
                raise KeyError(f"{path}:{lineno}: {synset} is not a val class")
            items.append({
                "image_id": f"{synset}/{Path(rel).stem}",
                "path": rel,
                "label": class_index[synset],
                "synset": synset,
                "criterion": criterion,
                "criterion_basis": "explicit_paths",
                "criterion_rationale": f"chosen by hand ({Path(path).name})",
            })
    return {
        "status": "frozen",
        "dataset": "imagenet-1k",
        "split": "val",
        "seed": None,
        "n": len(items),
        "selector": f"explicit paths from {Path(path).name}",
        "criteria": {k: v["rationale"] for k, v in sorted(CURATED_POOLS.items())},
        "items": items,
    }


def main():
    p = argparse.ArgumentParser(
        description="Freeze the TASK-11 probe set (write-once, per group).")
    p.add_argument("--groups", default="imagenet",
                   choices=("imagenet", "boxes20", "all"),
                   help="imagenet = curated + random200 (default); boxes20 = "
                        "the COCO group only; all = both (needs both sources)")
    p.add_argument("--data", metavar="ROOT",
                   help="ImageNet root containing val/ (needed for --groups "
                        "imagenet/all)")
    p.add_argument("--coco-ann", metavar="JSON",
                   help="COCO instances_val2017.json (needed for boxes20/all)")
    p.add_argument("--curated-paths", metavar="FILE",
                   help="optional human override for the curated group")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/probe/probe_set.json")
    p.add_argument("--if-missing", action="store_true",
                   help="skip (exit 0) groups that are already frozen instead "
                        "of failing — makes a job script resubmit-safe")
    args = p.parse_args()

    out = Path(args.out)
    existing = load_existing(out)
    if existing:
        verify_frozen_unchanged(existing)

    want = (IMAGENET_GROUPS if args.groups == "imagenet"
            else ("boxes20",) if args.groups == "boxes20"
            else IMAGENET_GROUPS + ("boxes20",))

    if args.if_missing and existing:
        want = tuple(n for n in want
                     if existing.get("groups", {}).get(n, {}).get("status")
                     != "frozen")
        if not want:
            print(f"{out}: all requested groups already frozen — nothing to do")
            return 0

    new_groups = {}
    if set(want) & set(IMAGENET_GROUPS):
        if not args.data:
            p.error("--data is required to build the ImageNet groups")
        classes = list_val_classes(args.data)
        if "curated" in want:
            new_groups["curated"] = (
                parse_curated_paths(args.curated_paths, args.data, classes)
                if args.curated_paths
                else build_curated(args.data, args.seed, classes))
        if "random200" in want:
            # paths already claimed — by the curated group just built, or by
            # one frozen in an earlier invocation
            claimed = {it["path"] for it in
                       new_groups.get("curated", {}).get("items", [])}
            prev = (existing or {}).get("groups", {}).get("curated") or {}
            claimed |= {it["path"] for it in prev.get("items", [])}
            new_groups["random200"] = build_random200(
                args.data, args.seed, classes, exclude=claimed)
    if "boxes20" in want:
        if not args.coco_ann:
            p.error("--coco-ann is required to build boxes20")
        new_groups["boxes20"] = build_boxes20(args.coco_ann, args.seed)

    doc = assemble(existing, new_groups, args.seed)
    check_disjoint(doc)
    verify_frozen_unchanged(doc)

    atomic_write_json(doc, out)

    for name in GROUPS:
        g = doc["groups"].get(name)
        if not g:
            continue
        mark = "wrote" if name in new_groups else "kept "
        print(f"{mark} {name:10s} status={g['status']:7s} n={g['n']:4d} "
              f"sha256={g.get('sha256', '-')[:12]}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
