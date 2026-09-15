"""
tests/test_task13_ring_ablation.py — TASK-13 Phase A
====================================================
Covers A4's four mandated checks:
  * the ring-1 index set matches the TASK-07 definition exactly (44 indices,
    pinned as a literal regression AND recomputed from the TASK-07 helper);
  * the three mask conditions are area-matched, and mask_random44 is
    deterministic given a seed;
  * the `full` condition path IS the TASK-08 eval path — asserted
    structurally (same transform objects, same loader construction) and
    numerically (it reproduces a number produced by the TASK-08 path);
  * third-dataset support is absent until a dataset is actually configured.

Everything runs on CPU against fake data.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image

from analysis.address_analysis import (border_distance_map, border_rings,
                                       ring_indices)
from evaluation.e6_finegrained.data.ft_meta import (FTItemsDataset,
                                                    build_eval_transform)
from evaluation.e6_finegrained.tools.build_ft_split import build_split
from evaluation.e6_finegrained.tools import train_ft
from saga.run_registry import file_sha256
from tools import ring_ablation as RA
from tools.model_factory import build_model

REPO = Path(__file__).resolve().parents[1]
TINY = "vit_tiny_patch16_224"
SIDE = 14


# ── A4.1  the ring definition ────────────────────────────────────────────────

def test_ring1_has_44_indices_and_matches_the_task07_definition():
    ring1 = ring_indices(SIDE, 1)
    assert len(ring1) == 44
    # recomputed straight from TASK-07's distance map (the single definition)
    dist = border_distance_map(SIDE)
    assert set(ring1.tolist()) == {
        r * SIDE + c for r in range(SIDE) for c in range(SIDE)
        if dist[r, c] == 1}
    # ring 1 is the 12x12 border one patch inside the grid edge
    expected = {r * SIDE + c for r in range(1, 13) for c in range(1, 13)
                if r in (1, 12) or c in (1, 12)}
    assert set(ring1.tolist()) == expected


def test_ring1_literal_regression():
    """Pinned literally: if the ring ever moves, every TASK-13 number that
    was read against TASK-07's coordinates becomes incomparable."""
    assert ring_indices(SIDE, 1).tolist() == [
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        29, 40, 43, 54, 57, 68, 71, 82, 85, 96, 99, 110,
        113, 124, 127, 138, 141, 152, 155, 166,
        169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180]


def test_ring0_is_the_outermost_and_rings_partition_the_grid():
    assert len(ring_indices(SIDE, 0)) == 52          # 14*4 - 4
    assert set(ring_indices(SIDE, 0).tolist()) == {
        r * SIDE + c for r in range(SIDE) for c in range(SIDE)
        if r in (0, SIDE - 1) or c in (0, SIDE - 1)}
    seen = np.concatenate([ring_indices(SIDE, k) for k in range(SIDE // 2)])
    assert sorted(seen.tolist()) == list(range(SIDE * SIDE))


def test_border_rings_unchanged_by_the_refactor():
    """The helper was extracted OUT of border_rings; border_rings must still
    compute exactly what TASK-07's committed answers were written against."""
    rng = np.random.default_rng(0)
    freq = rng.random(SIDE * SIDE)
    g = freq.reshape(SIDE, SIDE)
    idx = np.arange(SIDE)
    dist = np.minimum(
        np.minimum(idx[:, None], SIDE - 1 - idx[:, None]),
        np.minimum(idx[None, :], SIDE - 1 - idx[None, :]))
    reference = [float(g[dist == k].mean()) for k in range(SIDE // 2)]
    assert border_rings(freq) == reference


# ── A4.2  area matching + determinism ────────────────────────────────────────

def test_mask_conditions_are_area_matched():
    sets = RA.condition_indices(SIDE, seed=0)
    assert set(sets) == set(RA.CONDITIONS)
    assert len(sets["full"]) == 0
    for name in ("mask_ring1", "mask_center44", "mask_random44"):
        assert len(sets[name]) == RA.N_MASK == 44, name
        assert len(set(sets[name].tolist())) == 44, name
        assert sets[name].min() >= 0 and sets[name].max() < SIDE * SIDE


def test_random44_is_deterministic_and_seed_sensitive():
    a = RA.random_indices(SIDE, 44, seed=0)
    b = RA.random_indices(SIDE, 44, seed=0)
    c = RA.random_indices(SIDE, 44, seed=1)
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist()
    assert RA.condition_indices(SIDE, 0)["mask_random44"].tolist() == a.tolist()


def test_center44_is_central_deterministic_and_disjoint_from_ring1():
    centre = RA.center_indices(SIDE, 44)
    assert len(centre) == 44
    assert centre.tolist() == RA.center_indices(SIDE, 44).tolist()
    dist = border_distance_map(SIDE).reshape(-1)
    # the 44 most central positions cannot reach out to ring 1 or ring 0
    assert dist[centre].min() >= 3
    assert not (set(centre.tolist()) & set(ring_indices(SIDE, 1).tolist()))
    assert not (set(centre.tolist()) & set(ring_indices(SIDE, 0).tolist()))
    # rings 6,5,4 (4+12+20 = 36) are wholly inside; the other 8 come from
    # ring 3 by the documented Euclidean tie-break
    for k in (4, 5, 6):
        assert set(ring_indices(SIDE, k).tolist()) <= set(centre.tolist())
    assert len(set(centre.tolist()) & set(ring_indices(SIDE, 3).tolist())) == 8


def test_condition_indices_refuses_a_wrong_sized_set(monkeypatch):
    monkeypatch.setattr(RA, "ring_indices", lambda side, k: np.arange(43))
    with pytest.raises(RuntimeError, match="area-matched"):
        RA.condition_indices(SIDE, seed=0)


# ── A4.3  the `full` path is the TASK-08 path ────────────────────────────────

def test_full_condition_returns_the_task08_transform_itself():
    base = build_eval_transform(224)
    full = RA.masked_eval_transform(224, SIDE, [], fill=[0.5, 0.5, 0.5])
    assert [type(t) for t in full.transforms] == [type(t) for t in base.transforms]
    assert not any(isinstance(t, RA.PatchMask) for t in full.transforms)
    # and the parameters, not just the types
    assert full.transforms[-1].mean == base.transforms[-1].mean
    assert full.transforms[-1].std == base.transforms[-1].std
    assert full.transforms[0].size == base.transforms[0].size
    assert full.transforms[1].size == base.transforms[1].size


def test_masked_transform_differs_only_by_the_mask_before_normalize():
    base = build_eval_transform(224)
    masked = RA.masked_eval_transform(224, SIDE, ring_indices(SIDE, 1),
                                      fill=[0.1, 0.2, 0.3])
    assert len(masked.transforms) == len(base.transforms) + 1
    assert isinstance(masked.transforms[-1], T.Normalize)
    assert isinstance(masked.transforms[-2], RA.PatchMask)
    assert isinstance(masked.transforms[-3], T.ToTensor)
    # everything before the inserted mask is untouched
    assert [type(t) for t in masked.transforms[:-2]] == \
           [type(t) for t in base.transforms[:-1]]


def test_legacy_tail_guard_fires_when_the_pipeline_changes(monkeypatch):
    monkeypatch.setattr(
        RA, "build_eval_transform",
        lambda img_size: T.Compose([T.Resize(256), T.CenterCrop(img_size),
                                    T.ToTensor()]))
    with pytest.raises(RuntimeError, match="Normalize"):
        RA.masked_eval_transform(224, SIDE, [1, 2], fill=[0.0, 0.0, 0.0])


def test_patchmask_masks_exactly_the_named_patch_regions():
    t = torch.ones(3, 224, 224)
    mask = RA.PatchMask([0, 15, 195], fill=[0.25, 0.5, 0.75],
                        img_size=224, side=SIDE)
    out = mask(t.clone())
    # patch 0 -> rows/cols 0:16; patch 15 -> row 1, col 1; patch 195 -> 13,13
    for idx, (r, c) in ((0, (0, 0)), (15, (1, 1)), (195, (13, 13))):
        block = out[:, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        assert torch.allclose(block[0], torch.full((16, 16), 0.25))
        assert torch.allclose(block[1], torch.full((16, 16), 0.50))
        assert torch.allclose(block[2], torch.full((16, 16), 0.75))
    assert (out == 1.0).sum().item() == 3 * (224 * 224 - 3 * 16 * 16)


def test_grid_side_refuses_a_grid_that_is_not_196():
    class _Stub:
        patch_embed = None
    with pytest.raises(RuntimeError, match="196"):
        RA.grid_side(_Stub(), img_size=160)


# ── fixtures for the end-to-end contract ─────────────────────────────────────

def _write_img(path: Path, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)


@pytest.fixture(scope="module")
def fake_cub(tmp_path_factory):
    """3 classes x (10 train + 4 test), official CUB layout."""
    root = tmp_path_factory.mktemp("cub13") / "CUB_200_2011"
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
def ablation_env(fake_cub, tmp_path_factory):
    """A fine-tuned tiny checkpoint plus the eval/test_final.json that the
    TASK-08 path would have written for it — so `full` has a real number to
    reproduce, produced by the real TASK-08 code path."""
    work = tmp_path_factory.mktemp("ring_env")
    run_id = "ft_cub_tiny_saga_bs1_f0"

    split_path = work / "cub_val_split.json"
    split_path.write_text(json.dumps(
        build_split("cub", root=fake_cub, val_frac=0.1, seed=0)))

    torch.manual_seed(0)
    model = build_model(TINY, "saga", num_classes=1000)
    model.head = nn.Linear(model.embed_dim, 3)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    model.eval()

    runs_root = work / "runs"
    run_dir = runs_root / run_id
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir / "eval").mkdir(parents=True)
    ckpt_path = run_dir / "ckpt" / "best.pth"
    torch.save({"model": model.state_dict()}, ckpt_path)

    matrix = {
        "defaults": {"epochs": 2, "batch_size": 2, "workers": 0,
                     "img_size": 224, "split_seed": 0, "val_frac": 0.1},
        "splits": {"cub": str(split_path)},
        "expected_counts": {"cub": {"official_train": 30, "test": 12,
                                    "n_classes": 3}},
        "backbones": {"tiny_saga": {"run": "fake_backbone_run",
                                    "ckpt": str(ckpt_path), "sha256": None}},
        "runs": {run_id: {"dataset": "cub", "arch": TINY, "variant": "saga",
                          "backbone": "tiny_saga", "ft_seed": 0}},
    }
    cfg = train_ft.resolve_run_config(matrix, run_id)
    _tr, _va, test_items, n_classes, split_sha = train_ft.load_items(
        cfg, str(fake_cub))

    # the TASK-08 eval path, verbatim, to produce the committed number
    ds = FTItemsDataset(str(fake_cub), test_items,
                        transform=build_eval_transform(224))
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                         num_workers=0)
    with torch.no_grad():
        top1, top5, n = train_ft.evaluate(model, loader, torch.device("cpu"))

    (run_dir / "eval" / "test_final.json").write_text(json.dumps({
        "run_id": run_id, "dataset": "cub", "arch": TINY, "variant": "saga",
        "ft_seed": 0, "seed": 0, "top1": round(top1, 3), "top5": round(top5, 3),
        "n_images": n, "n_classes": n_classes, "smoke": False,
        "backbone_run": "fake_backbone_run",
        "backbone_sha256": "b" * 64,
        "finetuned_ckpt": "ckpt/best.pth",
        "finetuned_sha256": file_sha256(ckpt_path),
        "val_split": str(split_path), "val_split_sha256": split_sha,
    }))
    return {"work": work, "run_id": run_id, "cfg": cfg, "matrix": matrix,
            "runs_root": runs_root, "run_dir": run_dir, "root": fake_cub,
            "committed_top1": round(top1, 3), "ckpt": ckpt_path}


def _run(env, out_root, seed=0):
    return RA.run_one(env["cfg"], env["run_id"], str(env["root"]),
                      Path(out_root), seed, torch.device("cpu"),
                      batch_size=2, workers=0, runs_root=env["runs_root"])


# ── end-to-end contract ──────────────────────────────────────────────────────

def test_full_condition_reproduces_the_committed_number_exactly(ablation_env):
    out = ablation_env["work"] / "out1"
    assert _run(ablation_env, out) == "ok"
    res = json.loads((out / f"{ablation_env['run_id']}.json").read_text())

    assert res["top1"]["full"] == ablation_env["committed_top1"]
    assert res["full_minus_committed"] == 0.0
    assert res["full_reproduces_test_final"] is True
    assert abs(res["full_minus_committed"]) <= RA.FULL_TOL


def test_output_records_every_masked_index_and_its_provenance(ablation_env):
    out = ablation_env["work"] / "out2"
    _run(ablation_env, out)
    res = json.loads((out / f"{ablation_env['run_id']}.json").read_text())

    assert res["conditions"] == list(RA.CONDITIONS)
    assert set(res["top1"]) == set(RA.CONDITIONS)
    assert res["n_masked_patches"] == {"full": 0, "mask_ring1": 44,
                                       "mask_center44": 44,
                                       "mask_random44": 44}
    assert res["masked_indices"]["mask_ring1"] == ring_indices(SIDE, 1).tolist()
    assert res["masked_indices"]["full"] == []
    assert res["grid_side"] == 14 and res["patch_px"] == 16
    assert res["seed"] == 0
    assert res["finetuned_sha256"] == file_sha256(ablation_env["ckpt"])
    assert res["n_images"] == 12
    assert len(res["fill_value"]) == 3
    assert all(0.0 <= v <= 1.0 for v in res["fill_value"])
    # the fill comes from the TRAIN split, never the test split
    assert res["fill_n_images"] == 27      # 30 official train - 3 carved to val
    assert "train" in res["fill_source"]
    assert res["drop_vs_full"]["full"] == 0.0
    assert res["git_sha"]


def test_masking_reaches_the_model_and_touches_exactly_44_patches(ablation_env):
    """A mask that changed nothing would make the whole ablation vacuous.
    Run a real test image through both transforms and count the pixels that
    actually differ, in the normalized space the model consumes."""
    out = ablation_env["work"] / "out3"
    _run(ablation_env, out)
    res = json.loads((out / f"{ablation_env['run_id']}.json").read_text())

    cfg = ablation_env["cfg"]
    _tr, _va, test_items, _n, _s = train_ft.load_items(cfg,
                                                       str(ablation_env["root"]))
    img_path = Path(ablation_env["root"]) / test_items[0][0]
    raw = Image.open(img_path).convert("RGB")

    plain = build_eval_transform(224)(raw)
    masked = RA.masked_eval_transform(
        224, SIDE, ring_indices(SIDE, 1), res["fill_value"])(raw)

    assert not torch.equal(plain, masked)
    differing = (plain != masked).any(dim=0)          # HxW, any channel
    # every differing pixel must lie inside a ring-1 patch region
    allowed = torch.zeros(224, 224, dtype=torch.bool)
    for i in ring_indices(SIDE, 1):
        r, c = divmod(int(i), SIDE)
        allowed[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = True
    assert bool((differing & ~allowed).sum() == 0), \
        "masking altered pixels outside ring 1"
    # and the masked region really is constant per channel afterwards
    for i in ring_indices(SIDE, 1)[:4]:
        r, c = divmod(int(i), SIDE)
        block = masked[:, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        for ch in range(3):
            assert torch.allclose(block[ch], block[ch].flatten()[0])


def test_rerun_is_a_noop_but_a_new_seed_is_not(ablation_env):
    out = ablation_env["work"] / "out4"
    assert _run(ablation_env, out) == "ok"
    before = (out / f"{ablation_env['run_id']}.json").read_bytes()
    assert _run(ablation_env, out) == "skipped"
    assert (out / f"{ablation_env['run_id']}.json").read_bytes() == before
    # a different seed changes mask_random44, so it must NOT be skipped
    assert _run(ablation_env, out, seed=7) == "ok"
    after = json.loads((out / f"{ablation_env['run_id']}.json").read_text())
    assert after["seed"] == 7
    assert after["masked_indices"]["mask_random44"] != \
        json.loads(before)["masked_indices"]["mask_random44"]
    # the position-independent conditions are unchanged by the seed
    assert after["masked_indices"]["mask_ring1"] == \
        json.loads(before)["masked_indices"]["mask_ring1"]


def test_checkpoint_sha_mismatch_is_refused(ablation_env, tmp_path):
    final = ablation_env["run_dir"] / "eval" / "test_final.json"
    original = final.read_text()
    doctored = json.loads(original)
    doctored["finetuned_sha256"] = "0" * 64
    final.write_text(json.dumps(doctored))
    try:
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            _run(ablation_env, tmp_path / "out5")
    finally:
        final.write_text(original)


def test_smoke_artifact_is_refused(ablation_env, tmp_path):
    final = ablation_env["run_dir"] / "eval" / "test_final.json"
    original = final.read_text()
    doctored = json.loads(original)
    doctored["smoke"] = True
    final.write_text(json.dumps(doctored))
    try:
        with pytest.raises(RuntimeError, match="smoke"):
            _run(ablation_env, tmp_path / "out6")
    finally:
        final.write_text(original)


def test_missing_test_final_is_refused(ablation_env, tmp_path):
    cfg = dict(ablation_env["cfg"])
    with pytest.raises(FileNotFoundError, match="test_final.json"):
        RA.run_one(cfg, "ft_does_not_exist", str(ablation_env["root"]),
                   tmp_path / "out7", 0, torch.device("cpu"),
                   batch_size=2, workers=0,
                   runs_root=ablation_env["runs_root"])


# ── A4.4  third dataset is conditional and not yet configured ────────────────

def test_third_dataset_not_configured_until_a_path_is_confirmed():
    """TASK-13 A3 is conditional on the human confirming a staged path.
    If a cars/flowers row ever appears, its split builder, its committed
    split file and its own tests must appear with it — this guard makes
    adding one silently impossible."""
    matrix = yaml.safe_load((REPO / "configs" / "ft_matrix.yaml").read_text())
    datasets = {r["dataset"] for r in matrix["runs"].values()}
    assert datasets == {"cub", "aircraft"}, (
        f"new dataset(s) {datasets - {'cub', 'aircraft'}} in the matrix: add "
        f"their split builder + split file + tests, then update this guard")
    assert set(matrix["splits"]) == {"cub", "aircraft"}


# ── launcher ─────────────────────────────────────────────────────────────────

def test_ring_ablation_job_file_contract():
    src = (REPO / "scripts" / "jobs" / "ft_ring_ablation.sbatch").read_text()
    assert "--partition=a40" in src and "--gres=gpu:a40:1" in src
    assert "tools/ring_ablation.py" in src
    assert "--all" in src and "--seed 0" in src
    # the import gate must come BEFORE anything is staged
    assert src.index("import torch, timm") < src.index("mkdir -p \"$FT_STAGE\"")
    # scratch is released on every exit path, not just a normal one
    assert "trap 'rm -rf \"$FT_STAGE\"' EXIT" in src
    # set -u only after the env sourcing (pinned repo-wide too)
    assert src.index("source ") < src.index("\nset -u")
