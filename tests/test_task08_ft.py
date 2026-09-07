"""
tests/test_task08_ft.py — TASK-08 fine-grained clean protocol.

CPU-only, fake data. Covers:
  - split builder: determinism across two builds, stratification,
    val subset-of-train, val/test disjointness, tar == root, tiny-class refusal
  - label-mapping conventions (CUB cls-1, Aircraft variants.txt order)
  - strict backbone loading (missing/unexpected key -> raise), sha256 guard
  - img_size != 224 refusal
  - the real configs/ft_matrix.yaml contract: 16 runs, naming, seeds,
    backbone sha256 pinned to the committed e2r eval JSONs
  - end-to-end trainer run on a fake CUB: results contract, exact log.csv
    schema, val-based best selection, test touched exactly once,
    marker-written-last idempotency (requeue guard), fresh-restart
    determinism
  - generated SLURM job files in sync with the matrix
"""

import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from evaluation.e6_finegrained.data import ft_meta
from evaluation.e6_finegrained.tools.build_ft_split import build_split
from evaluation.e6_finegrained.tools import train_ft
from saga.run_registry import file_sha256
from tools.model_factory import build_model

REPO = Path(__file__).resolve().parents[1]
TINY = "vit_tiny_patch16_224"


# ── fake datasets ─────────────────────────────────────────────────────────────

def _write_img(path: Path, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                    ).save(path)


@pytest.fixture(scope="module")
def fake_cub(tmp_path_factory):
    """3 classes x (10 train + 4 test) in the official CUB layout."""
    root = tmp_path_factory.mktemp("cubdata") / "CUB_200_2011"
    classes = ["001.Alpha_Bird", "002.Beta_Bird", "003.Gamma_Bird"]
    images, labels, split = [], [], []
    img_id = 0
    for ci, cls in enumerate(classes):
        for j in range(14):
            img_id += 1
            rel = f"{cls}/{cls.split('.')[1]}_{j:04d}.jpg"
            images.append(f"{img_id} {rel}")
            labels.append(f"{img_id} {ci + 1}")
            split.append(f"{img_id} {1 if j < 10 else 0}")
            _write_img(root / "images" / rel, img_id)
    (root / "images.txt").write_text("\n".join(images) + "\n")
    (root / "image_class_labels.txt").write_text("\n".join(labels) + "\n")
    (root / "train_test_split.txt").write_text("\n".join(split) + "\n")
    return root


@pytest.fixture(scope="module")
def fake_cub_tar(fake_cub, tmp_path_factory):
    tar_path = tmp_path_factory.mktemp("cubtar") / "CUB_200_2011.tgz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(fake_cub, arcname="CUB_200_2011")
    return tar_path


@pytest.fixture(scope="module")
def fake_aircraft(tmp_path_factory):
    """3 variants x (10 trainval + 4 test), variant names with spaces and
    hyphens (the parsing hazard)."""
    root = tmp_path_factory.mktemp("acdata") / "fgvc-aircraft-2013b"
    data = root / "data"
    variants = ["Boeing 707-320", "A320", "DR-400"]  # NOT sorted: file order
    (data / "images").mkdir(parents=True)
    (data / "variants.txt").write_text("\n".join(variants) + "\n")
    trainval, test = [], []
    n = 0
    for variant in variants:
        for j in range(14):
            n += 1
            img_id = f"{n:07d}"
            (trainval if j < 10 else test).append(f"{img_id} {variant}")
            _write_img(data / "images" / f"{img_id}.jpg", n)
    (data / "images_variant_trainval.txt").write_text(
        "\n".join(trainval) + "\n")
    (data / "images_variant_test.txt").write_text("\n".join(test) + "\n")
    return root


# ── split builder ─────────────────────────────────────────────────────────────

def _check_split_invariants(split, official_train, official_test):
    train = {tuple(it) for it in split["items_train"]}
    val = {tuple(it) for it in split["items_val"]}
    test = set(official_test)
    assert val and train
    assert not (set(official_train) & test), "official train/test overlap"
    assert not (train & test), "train intersects official test"
    assert not (val & test), "val intersects official test"
    assert not (train & val)
    assert val <= set(official_train), "val not a subset of official train"
    assert train | val == set(official_train)
    assert split["n_train"] == len(train)
    assert split["n_val"] == len(val)
    assert split["n_official_train"] == len(official_train)


def test_cub_split_deterministic_and_stratified(fake_cub):
    s1 = build_split("cub", root=fake_cub, val_frac=0.1, seed=0)
    s2 = build_split("cub", root=fake_cub, val_frac=0.1, seed=0)
    assert s1 == s2, "two builds with the same seed must be identical"

    train_items, test_items, n_classes = ft_meta.official_splits(
        "cub", root=fake_cub)
    assert n_classes == 3
    assert len(train_items) == 30 and len(test_items) == 12
    _check_split_invariants(s1, train_items, test_items)

    # stratified: 10 train per class, 10% round-half-up -> exactly 1 val each
    per_class = {}
    for _, label in s1["items_val"]:
        per_class[label] = per_class.get(label, 0) + 1
    assert per_class == {0: 1, 1: 1, 2: 1}

    # a different seed picks a different (valid) val set
    s3 = build_split("cub", root=fake_cub, val_frac=0.1, seed=1)
    _check_split_invariants(s3, train_items, test_items)
    assert s3["items_val"] != s1["items_val"]


def test_aircraft_split_and_label_mapping(fake_aircraft):
    train_items, test_items, n_classes = ft_meta.official_splits(
        "aircraft", root=fake_aircraft)
    assert n_classes == 3
    # labels follow variants.txt FILE ORDER (legacy convention), not sorted
    by_label = {}
    for rel, label in train_items + test_items:
        assert rel.startswith("data/images/") and rel.endswith(".jpg")
        by_label.setdefault(label, rel)
    # ids were written in variants.txt order: 0000001.. -> Boeing 707-320 = 0
    assert by_label[0] == "data/images/0000001.jpg"

    split = build_split("aircraft", root=fake_aircraft, val_frac=0.1, seed=0)
    _check_split_invariants(split, train_items, test_items)


def test_split_from_tar_equals_root(fake_cub, fake_cub_tar):
    from_root = build_split("cub", root=fake_cub, val_frac=0.1, seed=0)
    from_tar = build_split("cub", tar=fake_cub_tar, val_frac=0.1, seed=0)
    # identical carve; only the provenance fields describe their sources
    strip = ("source", "source_sha256")
    assert {k: v for k, v in from_root.items() if k not in strip} \
        == {k: v for k, v in from_tar.items() if k not in strip}
    assert from_root["source_sha256"] is None
    assert from_tar["source_sha256"] == file_sha256(fake_cub_tar)
    assert from_tar["git_sha"]


def test_split_refuses_class_with_no_train_leftover(tmp_path):
    root = tmp_path / "CUB_200_2011"
    _write_img(root / "images" / "001.A/x.jpg", 0)
    (root / "images.txt").write_text("1 001.A/x.jpg\n")
    (root / "image_class_labels.txt").write_text("1 1\n")
    (root / "train_test_split.txt").write_text("1 1\n")
    with pytest.raises(ValueError, match="leave no train items"):
        build_split("cub", root=root, val_frac=0.1, seed=0)


# ── strict loading + sha guard ────────────────────────────────────────────────

def _tiny_backbone_ckpt(tmp_path, variant="saga"):
    torch.manual_seed(0)
    model = build_model(TINY, variant, num_classes=1000)
    path = tmp_path / "tiny_last.pth"
    torch.save({"model": model.state_dict(), "epoch": 3, "top1": 12.3}, path)
    return path


def _cfg_for_backbone(ckpt_path, sha=None, variant="saga"):
    return {"arch": TINY, "variant": variant, "img_size": 224,
            "backbone": {"ckpt": str(ckpt_path), "sha256": sha,
                         "run": "fake"}}


def test_strict_load_replaces_head_and_records_sha(tmp_path):
    ckpt = _tiny_backbone_ckpt(tmp_path)
    cfg = _cfg_for_backbone(ckpt, sha=file_sha256(ckpt))
    model, sha, meta = train_ft.load_backbone(cfg, torch.device("cpu"),
                                              n_classes=3)
    assert sha == cfg["backbone"]["sha256"]
    assert meta["epoch"] == 3 and meta["top1"] == 12.3
    assert model.head.out_features == 3


def test_strict_load_raises_on_missing_and_unexpected_keys(tmp_path):
    ckpt_path = _tiny_backbone_ckpt(tmp_path)
    wrapped = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    missing = dict(wrapped["model"])
    missing.pop(sorted(missing)[0])
    p_missing = tmp_path / "missing.pth"
    torch.save({"model": missing}, p_missing)
    with pytest.raises(RuntimeError):
        train_ft.load_backbone(_cfg_for_backbone(p_missing),
                               torch.device("cpu"), n_classes=3)

    extra = dict(wrapped["model"])
    extra["totally.unexpected.key"] = torch.zeros(1)
    p_extra = tmp_path / "extra.pth"
    torch.save({"model": extra}, p_extra)
    with pytest.raises(RuntimeError):
        train_ft.load_backbone(_cfg_for_backbone(p_extra),
                               torch.device("cpu"), n_classes=3)


def test_backbone_sha_mismatch_refused(tmp_path):
    ckpt = _tiny_backbone_ckpt(tmp_path)
    cfg = _cfg_for_backbone(ckpt, sha="0" * 64)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        train_ft.load_backbone(cfg, torch.device("cpu"), n_classes=3)


# ── matrix + config resolution ────────────────────────────────────────────────

def _test_matrix(fake_root, split_path, backbone_ckpt, epochs=2):
    return {
        "defaults": {
            "epochs": epochs, "batch_size": 2, "workers": 0,
            "backbone_lr": 1.0e-5, "head_lr": 1.0e-3, "weight_decay": 0.05,
            "eta_min": 1.0e-7, "label_smoothing": 0.1, "grad_clip": 1.0,
            "img_size": 224, "split_seed": 0, "val_frac": 0.1,
        },
        "splits": {"cub": str(split_path)},
        "expected_counts": {
            "cub": {"official_train": 30, "test": 12, "n_classes": 3}},
        "backbones": {
            "tiny_saga": {"run": "fake_backbone_run",
                          "ckpt": str(backbone_ckpt),
                          "sha256": file_sha256(backbone_ckpt)}},
        "runs": {
            "ft_cub_tiny_saga_bs1_f0": {
                "dataset": "cub", "arch": TINY, "variant": "saga",
                "backbone": "tiny_saga", "ft_seed": 0}},
    }


def test_resolve_refuses_non_224(tmp_path):
    ckpt = _tiny_backbone_ckpt(tmp_path)
    matrix = _test_matrix(tmp_path, tmp_path / "s.json", ckpt)
    matrix["runs"]["ft_cub_tiny_saga_bs1_f0"]["overrides"] = {"img_size": 240}
    with pytest.raises(ValueError, match="224"):
        train_ft.resolve_run_config(matrix, "ft_cub_tiny_saga_bs1_f0")


def test_real_matrix_contract():
    with open(REPO / "configs" / "ft_matrix.yaml") as f:
        matrix = yaml.safe_load(f)
    runs = matrix["runs"]
    assert len(runs) == 16

    # the frozen carve is pinned matrix-side
    assert matrix["defaults"]["split_seed"] == 0
    assert matrix["defaults"]["val_frac"] == 0.1

    pat = re.compile(
        r"^ft_(cub|aircraft)_(vits|vitb)_(baseline|saga)_bs1_f(\d)$")
    cells = {}
    for run_id, run in runs.items():
        m = pat.match(run_id)
        assert m, f"run id {run_id} violates the naming convention"
        assert "overrides" not in run, \
            f"{run_id}: production runs must not carry overrides"
        dataset, archtok, variant, seed = m.groups()
        assert run["dataset"] == dataset
        assert run["variant"] == variant
        assert run["arch"] == {"vits": "vit_small_patch16_224",
                               "vitb": "vit_base_patch16_224"}[archtok]
        assert run["backbone"] == f"{archtok}_{variant}"
        assert run["ft_seed"] == int(seed)
        cells.setdefault((dataset, archtok, variant), set()).add(int(seed))

    for dataset in ("cub", "aircraft"):
        for variant in ("baseline", "saga"):
            assert cells[(dataset, "vits", variant)] == {0, 1, 2}
            assert cells[(dataset, "vitb", variant)] == {0}

    # backbone shas must equal the committed e2r eval JSONs (numbers are
    # sacred: the matrix cannot drift from the provenance files)
    for key, bb in matrix["backbones"].items():
        eval_json = (REPO / "results" / "runs" / bb["run"] / "eval" /
                     "imagenet_val_last.json")
        recorded = json.load(open(eval_json))
        assert bb["sha256"] == recorded["ckpt_sha256"], key
        assert recorded["ckpt"].endswith(bb["ckpt"]), key

    exp = matrix["expected_counts"]
    assert exp["cub"] == {"official_train": 5994, "test": 5794,
                          "n_classes": 200}
    assert exp["aircraft"] == {"official_train": 6667, "test": 3333,
                               "n_classes": 100}


def test_generated_job_scripts_in_sync_with_matrix():
    with open(REPO / "configs" / "ft_matrix.yaml") as f:
        run_ids = list(yaml.safe_load(f)["runs"])
    array = (REPO / "scripts" / "jobs" / "ft_finegrained_array.sbatch"
             ).read_text()
    assert f"#SBATCH --array=0-{len(run_ids) - 1}" in array
    # the index -> run mapping is POSITIONAL: order must match the matrix
    assert re.findall(r'"(ft_[a-z0-9_]+)"', array) == run_ids
    assert "set -u" in array
    assert "--stage_base" in array and "gpu:a100:1" in array

    smoke = (REPO / "scripts" / "jobs" / "ft_smoke.sbatch").read_text()
    assert "ft_cub_vits_saga_bs1_f0" in smoke
    assert "--max_epochs 2" in smoke
    assert "--out_root results/smoke" in smoke, \
        "smoke must never write into results/runs/"


# ── end-to-end trainer ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e2e(fake_cub, tmp_path_factory):
    """One full 2-epoch run on the fake CUB; shared by the checks below."""
    work = tmp_path_factory.mktemp("ft_e2e")
    split_path = work / "cub_val_split.json"
    with open(split_path, "w") as f:
        json.dump(build_split("cub", root=fake_cub, val_frac=0.1, seed=0), f)
    ckpt = _tiny_backbone_ckpt(work)
    matrix = _test_matrix(fake_cub, split_path, ckpt)
    out_root = work / "runs"
    status = train_ft.run_training(matrix, "ft_cub_tiny_saga_bs1_f0",
                                   data_root=fake_cub, out_root=str(out_root),
                                   device_str="cpu")
    return {"matrix": matrix, "out_root": out_root, "fake_cub": fake_cub,
            "status": status,
            "run_dir": out_root / "ft_cub_tiny_saga_bs1_f0",
            "backbone_ckpt": ckpt}


def test_e2e_contract_files(e2e):
    assert e2e["status"] == "completed"
    rd = e2e["run_dir"]
    for rel in ("meta.json", "config.resolved.yaml", "log.csv",
                "ckpt/best.pth", "ckpt/last.pth", "eval/test_final.json"):
        assert (rd / rel).exists(), rel
    assert not list(rd.rglob("*.tmp")), "atomic writes left tmp files"
    assert not (rd / "run.lock").exists(), "lock must be released"

    meta = json.load(open(rd / "meta.json"))
    assert meta["ft_seed"] == 0
    assert meta["backbone_sha256"] == file_sha256(e2e["backbone_ckpt"])
    assert meta["val_split_sha256"]
    assert meta["end_time"] is not None
    assert meta["n_train"] == 27 and meta["n_val"] == 3 and \
        meta["n_test"] == 12
    assert "no resume" in meta["resume"] or "restart" in meta["resume"]


def test_e2e_log_schema_exact(e2e):
    lines = (e2e["run_dir"] / "log.csv").read_text().strip().splitlines()
    assert lines[0] == "epoch,lr,train_loss,val_top1", \
        "log.csv schema is task-mandated: epoch, lr, train_loss, val_top1"
    assert len(lines) == 3  # header + 2 epochs
    assert lines[1].split(",")[0] == "0" and lines[2].split(",")[0] == "1"
    # the logged lr is the one the epoch TRAINED with (pre-scheduler.step):
    # epoch 0 must show the configured backbone lr exactly
    assert float(lines[1].split(",")[1]) == pytest.approx(1.0e-5)


def test_e2e_test_final_fields_and_selection(e2e):
    rd = e2e["run_dir"]
    res = json.load(open(rd / "eval" / "test_final.json"))
    for key in ("run_id", "dataset", "arch", "variant", "ft_seed", "top1",
                "top5", "n_images", "n_classes", "best_epoch",
                "val_top1_at_best", "backbone_sha256", "finetuned_sha256",
                "val_split_sha256", "git_sha"):
        assert key in res, key
    assert res["n_images"] == 12  # official fake test, counted exactly
    assert 0.0 <= res["top1"] <= 100.0
    assert res["ft_seed"] == 0 and res["seed"] == 0
    assert res["smoke"] is False  # no epochs override -> a real result
    assert res["backbone_sha256"] == file_sha256(e2e["backbone_ckpt"])
    # finetuned sha is the sha of the val-selected best.pth on disk
    assert res["finetuned_sha256"] == file_sha256(rd / "ckpt" / "best.pth")

    # best.pth is the VAL-selected checkpoint (B7 fix)
    best = torch.load(rd / "ckpt" / "best.pth", map_location="cpu",
                      weights_only=True)
    assert best["epoch"] == res["best_epoch"]
    assert round(best["val_top1"], 3) == res["val_top1_at_best"]


def test_e2e_requeue_guard_skips_completed_run(e2e):
    marker = e2e["run_dir"] / "eval" / "test_final.json"
    before = marker.read_bytes()
    status = train_ft.run_training(e2e["matrix"], "ft_cub_tiny_saga_bs1_f0",
                                   data_root=e2e["fake_cub"],
                                   out_root=str(e2e["out_root"]),
                                   device_str="cpu")
    assert status == "skipped"
    assert marker.read_bytes() == before, "completed run must not be touched"


def test_e2e_test_split_touched_exactly_once(e2e):
    model = build_model(TINY, "saga", num_classes=1000)
    cfg = train_ft.resolve_run_config(e2e["matrix"], "ft_cub_tiny_saga_bs1_f0")
    with pytest.raises(RuntimeError, match="touched once|already exists"):
        train_ft.final_test_eval(model, cfg, e2e["run_dir"], [],
                                 e2e["fake_cub"], torch.device("cpu"),
                                 extra={"n_classes": 3})


def test_load_items_refuses_wrong_carve_and_duplicates(fake_cub, tmp_path):
    good = build_split("cub", root=fake_cub, val_frac=0.1, seed=0)

    def _cfg(split_path):
        return {"dataset": "cub", "val_split": str(split_path),
                "split_seed": 0, "val_frac": 0.1, "expected_counts": None}

    # a rebuilt carve (different seed) must be refused even though it is a
    # perfectly valid partition
    other = build_split("cub", root=fake_cub, val_frac=0.1, seed=7)
    p = tmp_path / "other_seed.json"
    p.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="not the frozen carve"):
        train_ft.load_items(_cfg(p), fake_cub)

    # a duplicated train row must be refused
    dup = json.loads(json.dumps(good))
    dup["items_train"].append(dup["items_train"][0])
    dup["n_train"] += 1
    p2 = tmp_path / "dup.json"
    p2.write_text(json.dumps(dup))
    with pytest.raises(ValueError, match="duplicate"):
        train_ft.load_items(_cfg(p2), fake_cub)

    # the untouched frozen carve passes
    p3 = tmp_path / "good.json"
    p3.write_text(json.dumps(good))
    train_items, val_items, test_items, n_classes, _ = train_ft.load_items(
        _cfg(p3), fake_cub)
    assert (len(train_items), len(val_items), len(test_items),
            n_classes) == (27, 3, 12, 3)


def test_fresh_lock_refuses_concurrent_start(fake_cub, tmp_path):
    """A fresh run.lock means another process is training this run_dir —
    the legacy double-submission overwrite must be refused, fast."""
    ckpt = _tiny_backbone_ckpt(tmp_path)
    split_path = tmp_path / "cub_val_split.json"
    split_path.write_text(json.dumps(
        build_split("cub", root=fake_cub, val_frac=0.1, seed=0)))
    matrix = _test_matrix(fake_cub, split_path, ckpt)
    run_dir = tmp_path / "runs" / "ft_cub_tiny_saga_bs1_f0"
    run_dir.mkdir(parents=True)
    (run_dir / "run.lock").write_text("someone else")
    with pytest.raises(RuntimeError, match="RIGHT NOW"):
        train_ft.run_training(matrix, "ft_cub_tiny_saga_bs1_f0",
                              data_root=fake_cub,
                              out_root=str(tmp_path / "runs"),
                              device_str="cpu")


def test_stale_lock_is_replaced_and_smoke_override_flagged(fake_cub,
                                                           tmp_path):
    """A stale lock (dead job) must not block the fresh restart, and an
    epochs override must flag the result as smoke."""
    import os as _os
    ckpt = _tiny_backbone_ckpt(tmp_path)
    split_path = tmp_path / "cub_val_split.json"
    split_path.write_text(json.dumps(
        build_split("cub", root=fake_cub, val_frac=0.1, seed=0)))
    matrix = _test_matrix(fake_cub, split_path, ckpt)  # epochs: 2
    run_dir = tmp_path / "runs" / "ft_cub_tiny_saga_bs1_f0"
    run_dir.mkdir(parents=True)
    lock = run_dir / "run.lock"
    lock.write_text("dead job")
    old = lock.stat().st_mtime - (train_ft.LOCK_STALE_SECONDS + 60)
    _os.utime(lock, (old, old))

    status = train_ft.run_training(matrix, "ft_cub_tiny_saga_bs1_f0",
                                   data_root=fake_cub,
                                   out_root=str(tmp_path / "runs"),
                                   max_epochs=1, device_str="cpu")
    assert status == "completed"
    assert not lock.exists(), "lock must be released on completion"
    res = json.load(open(run_dir / "eval" / "test_final.json"))
    assert res["smoke"] is True
    assert res["epochs_trained"] == 1 and res["epochs_configured"] == 2

    with pytest.raises(ValueError, match="epochs must be >= 1"):
        train_ft.run_training(matrix, "ft_cub_tiny_saga_bs1_f0",
                              data_root=fake_cub,
                              out_root=str(tmp_path / "runs2"),
                              max_epochs=0, device_str="cpu")

    # a TORN marker (crashed node mid-write) must not pin the run as done:
    # it is quarantined and the run is redone
    marker = run_dir / "eval" / "test_final.json"
    marker.write_text('{"run_id": "ft_cub_tiny_saga_bs1_f0", "top')
    status = train_ft.run_training(matrix, "ft_cub_tiny_saga_bs1_f0",
                                   data_root=fake_cub,
                                   out_root=str(tmp_path / "runs"),
                                   max_epochs=1, device_str="cpu")
    assert status == "completed"
    assert (run_dir / "eval" / "test_final.json.corrupt").exists()
    assert train_ft.marker_is_valid(marker, "ft_cub_tiny_saga_bs1_f0")


def test_release_lock_respects_ownership(tmp_path):
    lock = tmp_path / "run.lock"
    lock.write_text("successor-process token")
    train_ft.release_lock(lock, "my-old-token")
    assert lock.exists(), "a zombie must never remove a successor's lock"
    train_ft.release_lock(lock, "successor-process token")
    assert not lock.exists()


def test_split_builder_is_write_once(fake_cub, tmp_path, monkeypatch):
    from evaluation.e6_finegrained.tools import build_ft_split as bfs
    out = tmp_path / "cub_val_split.json"
    argv = ["build_ft_split.py", "--dataset", "cub", "--root", str(fake_cub),
            "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    bfs.main()
    assert out.exists()
    with pytest.raises(SystemExit, match="WRITE-ONCE"):
        bfs.main()


def test_e2e_fresh_restart_is_clean_and_deterministic(e2e):
    """Deleting only the completion marker forces a fresh restart: stale
    log/ckpts must not leak into the new attempt, and the same ft_seed must
    reproduce the same test number (CPU)."""
    rd = e2e["run_dir"]
    first = json.load(open(rd / "eval" / "test_final.json"))
    (rd / "eval" / "test_final.json").unlink()

    status = train_ft.run_training(e2e["matrix"], "ft_cub_tiny_saga_bs1_f0",
                                   data_root=e2e["fake_cub"],
                                   out_root=str(e2e["out_root"]),
                                   device_str="cpu")
    assert status == "completed"
    lines = (rd / "log.csv").read_text().strip().splitlines()
    assert len(lines) == 3, "fresh restart must truncate the stale log"
    second = json.load(open(rd / "eval" / "test_final.json"))
    assert second["top1"] == first["top1"]
    assert second["best_epoch"] == first["best_epoch"]
