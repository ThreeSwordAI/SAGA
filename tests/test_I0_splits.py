"""
tests/test_I0_splits.py
=======================
TASK I0 D7 — the write-once split builder.

The real splits do not exist until Phase B (the builder needs an extracted
ImageNet val tree, which on this cluster lives only inside a job), so every
test here runs against a SYNTHETIC val listing in tmp_path: 40 classes x 50
images, with a 10-per-class discovery split carved out of it exactly as
`results/diagsplit/val_diag_split.json` was.

What is pinned: disjointness across all pairs, the subset relation for
sub1k/sub2k, stratification, determinism from the seed, the write-once
refusal, and sha stability under a rebuild. Also the reader side — the
runner must refuse a split whose content no longer matches its recorded sha.
"""

import importlib.util
import json
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

N_CLASSES = 40
N_PER_CLASS = 50
N_DISCOVERY_PER_CLASS = 10


def load_builder(n_sub1k=100, n_sub2k=200):
    """Import the builder fresh and scale the sub-subset sizes to the
    synthetic listing. The stratified counts (2/class, 10/class) are the
    REAL ones and are not scaled — they are what the test is checking."""
    spec = importlib.util.spec_from_file_location(
        "bfs_under_test", REPO / "tools" / "build_frozen_splits.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.N_SUB = {"sub1k": n_sub1k, "sub2k": n_sub2k}
    return m


@pytest.fixture
def fake_val(tmp_path):
    """(data_root, discovery_json). 40 classes x 50 images, 10/class claimed
    by a discovery split built the way tools/build_diag_split.py builds it."""
    val = tmp_path / "data" / "val"
    names = {}
    for c in range(N_CLASSES):
        syn = f"n{c:08d}"
        (val / syn).mkdir(parents=True)
        names[syn] = []
        for i in range(N_PER_CLASS):
            fn = f"IMG_{c:04d}_{i:03d}.JPEG"
            (val / syn / fn).write_text("x")
            names[syn].append(fn)

    items = []
    for c, syn in enumerate(sorted(names)):
        rng = random.Random(c)
        for fn in sorted(rng.sample(names[syn], N_DISCOVERY_PER_CLASS)):
            items.append([f"val/{syn}/{fn}", c])
    disc = tmp_path / "val_diag_split.json"
    disc.write_text(json.dumps(
        {"seed": 0, "n": len(items), "n_per_class": N_DISCOVERY_PER_CLASS,
         "items": items}), encoding="utf-8")
    return tmp_path / "data", disc


def build(m, fake_val, out_dir, extra=()):
    data, disc = fake_val
    import sys
    argv = sys.argv
    sys.argv = ["build_frozen_splits.py", "--data", str(data),
                "--discovery", str(disc), "--out-dir", str(out_dir), *extra]
    try:
        return m.main()
    finally:
        sys.argv = argv


def read(out_dir):
    return {n: json.loads((out_dir / f"{n}.json").read_text(encoding="utf-8"))
            for n in ("calibration", "evaluation", "sub1k", "sub2k")}


def paths(doc):
    return {rel for rel, _ in doc["items"]}


# ── construction ────────────────────────────────────────────────────────────

def test_all_four_splits_are_built_with_the_declared_sizes(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    assert build(m, fake_val, out) == 0
    docs = read(out)
    assert docs["calibration"]["n"] == 2 * N_CLASSES
    assert docs["evaluation"]["n"] == 10 * N_CLASSES
    assert docs["sub1k"]["n"] == 100
    assert docs["sub2k"]["n"] == 200
    for name, doc in docs.items():
        assert doc["status"] == "frozen"
        assert doc["name"] == name
        assert doc["seed"] == 0


def test_every_required_pair_is_disjoint(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    docs = read(out)
    discovery = {rel for rel, _ in
                 json.loads(fake_val[1].read_text())["items"]}

    assert not (discovery & paths(docs["calibration"]))
    assert not (discovery & paths(docs["evaluation"]))
    assert not (paths(docs["calibration"]) & paths(docs["evaluation"]))


def test_sub_subsets_are_subsets_of_evaluation_not_disjoint_from_it(fake_val,
                                                                    tmp_path):
    """sub1k/sub2k are DRAWN FROM evaluation on purpose. Stating which
    relation is expected is the point of the check."""
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    docs = read(out)
    ev = paths(docs["evaluation"])
    assert paths(docs["sub1k"]) <= ev
    assert paths(docs["sub2k"]) <= ev
    discovery = {rel for rel, _ in
                 json.loads(fake_val[1].read_text())["items"]}
    for sub in ("sub1k", "sub2k"):
        assert not (paths(docs[sub]) & discovery)
        assert not (paths(docs[sub]) & paths(docs["calibration"]))


def test_stratification_is_exact(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    docs = read(out)
    for name, per_class in (("calibration", 2), ("evaluation", 10)):
        counts = {}
        for _, label in docs[name]["items"]:
            counts[label] = counts.get(label, 0) + 1
        assert set(counts) == set(range(N_CLASSES)), name
        assert set(counts.values()) == {per_class}, name


def test_labels_follow_the_imagefolder_convention(fake_val, tmp_path):
    """Class ids are indices into the SORTED synset list, the same convention
    tools/build_diag_split.py uses — so a label means the same thing in a
    frozen split and in the discovery split."""
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    docs = read(out)
    classes = sorted(p.name for p in (fake_val[0] / "val").iterdir())
    index = {c: i for i, c in enumerate(classes)}
    for rel, label in docs["evaluation"]["items"]:
        assert index[Path(rel).parent.name] == label


def test_sub_subsets_spread_over_classes(fake_val, tmp_path):
    """A 100-image subset of a 10-per-class split must cover many classes,
    not ten classes ten times over."""
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    docs = read(out)
    assert docs["sub1k"]["n_classes"] == N_CLASSES
    counts = {}
    for _, label in docs["sub1k"]["items"]:
        counts[label] = counts.get(label, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


# ── determinism and integrity ───────────────────────────────────────────────

def test_a_rebuild_is_byte_identical(fake_val, tmp_path):
    m = load_builder()
    a, b = tmp_path / "a", tmp_path / "b"
    build(m, fake_val, a)
    build(m, fake_val, b)
    for name in ("calibration", "evaluation", "sub1k", "sub2k"):
        assert (a / f"{name}.json").read_bytes() == \
            (b / f"{name}.json").read_bytes(), name


def test_a_different_seed_gives_a_different_split(fake_val, tmp_path):
    m = load_builder()
    a, b = tmp_path / "a", tmp_path / "b"
    build(m, fake_val, a)
    build(m, fake_val, b, extra=("--seed", "1"))
    assert paths(read(a)["evaluation"]) != paths(read(b)["evaluation"])
    assert read(a)["evaluation"]["sha256"] != read(b)["evaluation"]["sha256"]


def test_each_split_carries_a_sha_over_its_own_content(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    for name, doc in read(out).items():
        assert doc["sha256"] == m.split_sha256(doc), name
        assert len(doc["sha256"]) == 64


def test_evaluation_records_the_shas_it_was_carved_against(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    docs = read(out)
    assert docs["evaluation"]["disjoint_from"] == ["discovery", "calibration"]
    assert docs["evaluation"]["calibration_sha256"] == \
        docs["calibration"]["sha256"]
    assert docs["evaluation"]["discovery_sha256"] == \
        m.file_sha256(fake_val[1])
    for sub in ("sub1k", "sub2k"):
        assert docs[sub]["parent"] == "evaluation"
        assert docs[sub]["parent_sha256"] == docs["evaluation"]["sha256"]


def test_an_edited_split_is_detected(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    doc = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    doc["items"] = doc["items"][:-1]
    (out / "evaluation.json").write_text(json.dumps(doc, indent=2,
                                                    sort_keys=True))
    with pytest.raises(SystemExit, match="does not match its recorded sha256"):
        m.verify_frozen(out)


# ── write-once ──────────────────────────────────────────────────────────────

def test_a_second_build_is_refused(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    before = {n: (out / f"{n}.json").read_bytes()
              for n in ("calibration", "evaluation", "sub1k", "sub2k")}
    with pytest.raises(SystemExit, match="REFUSING to overwrite"):
        build(m, fake_val, out)
    after = {n: (out / f"{n}.json").read_bytes() for n in before}
    assert after == before, "a refused build must not have touched anything"


def test_if_missing_makes_a_resubmit_a_no_op(fake_val, tmp_path):
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    before = (out / "evaluation.json").read_bytes()
    assert build(m, fake_val, out, extra=("--if-missing",)) == 0
    assert (out / "evaluation.json").read_bytes() == before


def test_a_partial_build_is_refused(fake_val, tmp_path):
    """The four splits are constructed TOGETHER — calibration claims images
    before evaluation draws, and the subsets are drawn from evaluation."""
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    (out / "sub2k.json").unlink()
    with pytest.raises(SystemExit, match="Build all four or none"):
        build(m, fake_val, out, extra=("--if-missing",))


def test_a_missing_discovery_split_is_refused(fake_val, tmp_path):
    m = load_builder()
    data, _ = fake_val
    import sys
    argv = sys.argv
    sys.argv = ["x", "--data", str(data), "--discovery",
                str(tmp_path / "nope.json"), "--out-dir",
                str(tmp_path / "splits")]
    try:
        with pytest.raises(FileNotFoundError, match="DISCOVERY split"):
            m.main()
    finally:
        sys.argv = argv


def test_too_few_unclaimed_images_is_refused_not_silently_unbalanced(tmp_path):
    """A class that cannot supply its quota must fail loudly. An unbalanced
    split that still says `n_per_class: 10` would be a lie."""
    m = load_builder(n_sub1k=1, n_sub2k=1)
    val = tmp_path / "data" / "val"
    for c in range(3):
        syn = f"n{c:08d}"
        (val / syn).mkdir(parents=True)
        for i in range(3 if c else 30):        # class 0 has 30, others 3
            (val / syn / f"IMG_{c}_{i}.JPEG").write_text("x")
    disc = tmp_path / "d.json"
    disc.write_text(json.dumps({"seed": 0, "n": 0, "items": []}))
    listing = m.list_val(tmp_path / "data")
    with pytest.raises(ValueError, match="fewer than 10 unclaimed"):
        m.build_stratified("evaluation", listing, set(), 0)


# ── the protocol sentence ───────────────────────────────────────────────────

def test_the_protocol_sentence_is_in_every_split_and_the_readme(fake_val,
                                                                tmp_path):
    """Plan §6.2: the 50k accuracies have already been seen, so this is a
    LOCKED PROTOCOL, not an untouched test set. That sentence must travel
    with the files."""
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    for name, doc in read(out).items():
        assert doc["protocol"] == m.PROTOCOL_SENTENCE, name
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert m.PROTOCOL_SENTENCE in readme
    assert "locked evaluation protocol" in m.PROTOCOL_SENTENCE.lower()
    assert "not" in m.PROTOCOL_SENTENCE.lower()


def test_the_builder_is_stdlib_only():
    """It runs where the conda env may not be active. torch/timm/numpy/yaml
    must not appear in its imports (TASK-09 lost a sync to that assumption)."""
    src = (REPO / "tools" / "build_frozen_splits.py").read_text(
        encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            for banned in ("torch", "timm", "numpy", "yaml", "scipy", "PIL"):
                assert not s.startswith(f"import {banned}"), s
                assert not s.startswith(f"from {banned}"), s


# ── the reader side ─────────────────────────────────────────────────────────

def test_the_runner_refuses_a_split_whose_sha_no_longer_matches(fake_val,
                                                                tmp_path):
    from saga.frozen.runner import RunnerError, load_split, split_sha256
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)

    doc, items, sha = load_split(out / "evaluation.json")
    assert sha == doc["sha256"]
    assert len(items) == doc["n"]

    doc["items"] = doc["items"][:-1]
    with pytest.raises(RunnerError, match="has been edited since it was "
                                          "frozen"):
        split_sha256(doc)


def test_image_ids_are_paths_not_indices(fake_val, tmp_path):
    from saga.frozen.runner import image_id_for
    m = load_builder()
    out = tmp_path / "splits"
    build(m, fake_val, out)
    docs = read(out)
    ids = [image_id_for(rel) for rel, _ in docs["evaluation"]["items"]]
    assert len(set(ids)) == len(ids)
    assert all("/" in i and not i.endswith(".JPEG") for i in ids)
    assert image_id_for("val/n01440764/ILSVRC2012_val_00003014.JPEG") == \
        "n01440764/ILSVRC2012_val_00003014"
