"""TASK-11 — probe set, attention dumps, localization score.

CPU, fake data, tiny real models. No checkpoint, dataset or results file is
read from the repo.
"""

import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from analysis.address_analysis import border_rings
from tools import build_probe_set as bps
from tools import dump_attention as dmp
from tools import localization_score as loc

N_FILLER_CLASSES = 210
N_IMAGES_PER_CLASS = 3


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_fake_imagenet(root):
    """Every synset the curated pools name, plus filler so random200 fits."""
    synsets = sorted({s for pool in bps.CURATED_POOLS.values()
                      for s in pool["synsets"]})
    synsets += [f"n9{i:07d}" for i in range(N_FILLER_CLASSES)]
    for c, syn in enumerate(sorted(synsets)):
        d = root / "val" / syn
        d.mkdir(parents=True)
        for i in range(N_IMAGES_PER_CLASS):
            Image.new("RGB", (8, 8), (c % 256, i * 40 % 256, 7)).save(
                d / f"img{i:02d}_{syn}.JPEG")
    return root


def make_fake_coco_ann(path, n_images=25):
    images, annotations = [], []
    for i in range(1, n_images + 1):
        images.append({"id": i, "file_name": f"{i:012d}.jpg",
                       "width": 640, "height": 480})
        annotations.append({"id": i * 10, "image_id": i, "iscrowd": 0,
                            "bbox": [10.0 * i, 20.0, 100.0, 80.0],
                            "category_id": 1})
    path.write_text(json.dumps({"images": images, "annotations": annotations}),
                    encoding="utf-8")
    return path


def build_imagenet_groups(tmp_path, seed=0):
    root = make_fake_imagenet(tmp_path / "in")
    out = tmp_path / "probe_set.json"
    sys_argv = ["build_probe_set.py", "--groups", "imagenet",
                "--data", str(root), "--out", str(out), "--seed", str(seed)]
    old = sys.argv
    try:
        sys.argv = sys_argv
        bps.main()
    finally:
        sys.argv = old
    return root, out


# ── A4: probe set ────────────────────────────────────────────────────────────

def test_probe_set_is_deterministic_across_builds(tmp_path):
    _, out_a = build_imagenet_groups(tmp_path / "a")
    _, out_b = build_imagenet_groups(tmp_path / "b")
    assert out_a.read_bytes() == out_b.read_bytes()


def test_probe_set_seed_changes_selection(tmp_path):
    _, out_a = build_imagenet_groups(tmp_path / "a", seed=0)
    _, out_b = build_imagenet_groups(tmp_path / "b", seed=1)
    ga = json.loads(out_a.read_text())["groups"]["random200"]["items"]
    gb = json.loads(out_b.read_text())["groups"]["random200"]["items"]
    assert [i["path"] for i in ga] != [i["path"] for i in gb]


def test_groups_are_disjoint_and_paths_resolve(tmp_path):
    root, out = build_imagenet_groups(tmp_path)
    doc = json.loads(out.read_text())

    paths = {}
    for name in ("curated", "random200"):
        for it in doc["groups"][name]["items"]:
            assert (root / it["path"]).is_file(), it["path"]
            assert it["path"].startswith("val/")
            assert it["path"] not in paths, f"{it['path']} in two groups"
            paths[it["path"]] = name

    assert doc["groups"]["curated"]["n"] == 12
    assert doc["groups"]["random200"]["n"] == 200
    assert doc["groups"]["random200"]["n_classes"] == 200


def test_curated_records_a_criterion_per_image(tmp_path):
    _, out = build_imagenet_groups(tmp_path)
    items = json.loads(out.read_text())["groups"]["curated"]["items"]
    counts = {}
    for it in items:
        assert it["criterion"] in bps.CURATED_POOLS
        assert it["criterion_basis"] == "class_prior"
        assert it["criterion_rationale"]
        counts[it["criterion"]] = counts.get(it["criterion"], 0) + 1
    assert counts == {k: 4 for k in bps.CURATED_POOLS}


def test_boxes20_starts_pending_and_is_never_a_silent_substitute(tmp_path):
    _, out = build_imagenet_groups(tmp_path)
    g = json.loads(out.read_text())["groups"]["boxes20"]
    assert g["status"] == "pending" and g["n"] == 0 and g["items"] == []


def test_rebuilding_a_frozen_group_is_refused(tmp_path):
    root, out = build_imagenet_groups(tmp_path)
    old = sys.argv
    try:
        sys.argv = ["build_probe_set.py", "--groups", "imagenet",
                    "--data", str(root), "--out", str(out)]
        with pytest.raises(ValueError, match="already frozen"):
            bps.main()
    finally:
        sys.argv = old


def test_if_missing_makes_a_resubmitted_job_a_noop(tmp_path):
    root, out = build_imagenet_groups(tmp_path)
    before = out.read_bytes()
    old = sys.argv
    try:
        sys.argv = ["build_probe_set.py", "--groups", "imagenet", "--if-missing",
                    "--data", str(root), "--out", str(out)]
        assert bps.main() == 0
    finally:
        sys.argv = old
    assert out.read_bytes() == before


def test_filling_boxes20_cannot_perturb_the_frozen_imagenet_groups(tmp_path):
    _, out = build_imagenet_groups(tmp_path)
    before = json.loads(out.read_text())["groups"]
    ann = make_fake_coco_ann(tmp_path / "instances_val2017.json")

    old = sys.argv
    try:
        sys.argv = ["build_probe_set.py", "--groups", "boxes20",
                    "--coco-ann", str(ann), "--out", str(out)]
        assert bps.main() == 0
    finally:
        sys.argv = old

    after = json.loads(out.read_text())["groups"]
    for name in ("curated", "random200"):
        assert after[name] == before[name]
        assert after[name]["sha256"] == bps.group_sha256(after[name])
    assert after["boxes20"]["status"] == "frozen"
    assert after["boxes20"]["n"] == bps.N_BOXES
    assert all(it["boxes_xywh"] for it in after["boxes20"]["items"])


def test_a_hand_edited_frozen_group_is_detected(tmp_path):
    _, out = build_imagenet_groups(tmp_path)
    doc = json.loads(out.read_text())
    doc["groups"]["curated"]["items"][0]["path"] = "val/tampered/x.JPEG"
    out.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        bps.verify_frozen_unchanged(json.loads(out.read_text()))


# ── A4: the ring, imported from TASK-07 ──────────────────────────────────────

def test_ring_index_map_agrees_with_border_rings(tmp_path):
    n = 196
    idx = np.asarray(loc.ring_index_map(n))
    side = 14
    # independent Chebyshev-distance reference, compared against the map that
    # was derived by probing TASK-07's own function
    a = np.arange(side)
    dist = np.minimum(np.minimum(a[:, None], side - 1 - a[:, None]),
                      np.minimum(a[None, :], side - 1 - a[None, :]))
    assert np.array_equal(idx.reshape(side, side), dist)

    # and the profile of a map built FROM the mask reproduces border_rings
    m = loc.ring_mask(n, 1).astype(np.float64)
    prof = border_rings(m)
    assert prof[1] == pytest.approx(1.0)
    assert all(prof[k] == pytest.approx(0.0) for k in range(len(prof)) if k != 1)


def test_ring1_mass_of_a_uniform_map_is_the_rings_area_fraction():
    n = 196
    p = np.full(n, 1.0 / n)
    m = loc.ring_mask(n, 1)
    assert float(p[m].sum()) == pytest.approx(float(m.mean()))
    # 14x14: ring 1 is the 12x12 border minus the 10x10 interior = 44 patches
    assert int(m.sum()) == 44
    assert float(p[m].sum()) == pytest.approx(44 / 196)


def test_entropy_is_bits_and_maximal_for_uniform():
    n = 196
    assert loc.entropy_bits(np.full(n, 1.0 / n)) == pytest.approx(np.log2(n))
    spike = np.zeros(n); spike[3] = 1.0
    assert loc.entropy_bits(spike) == pytest.approx(0.0)


# ── A4: box geometry, measured against the real transform ────────────────────

@pytest.mark.parametrize("size", [(640, 480), (480, 640), (500, 500),
                                  (333, 500), (1024, 300)])
def test_resized_size_matches_torchvision(size):
    from torchvision.transforms import functional as F
    w, h = size
    img = Image.new("RGB", (w, h))
    out = F.resize(img, 256)
    assert loc.resized_size(w, h, 256) == (out.width, out.height)


def test_box_mapping_lands_where_the_real_transform_puts_the_pixels():
    """A white rectangle is pushed through the ACTUAL Resize+CenterCrop; the
    mapped box must bound the pixels that survive."""
    from torchvision import transforms as T
    W, H = 640, 480
    x, y, bw, bh = 100.0, 60.0, 220.0, 180.0

    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[int(y):int(y + bh), int(x):int(x + bw)] = 255
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.NEAREST),
                    T.CenterCrop(224)])
    got = np.asarray(tf(Image.fromarray(arr)))[:, :, 0] > 127

    (mx1, my1, mx2, my2), = loc.map_boxes_to_input_frame(
        [[x, y, bw, bh]], W, H, img_size=224)

    ys, xs = np.nonzero(got)
    assert xs.size and ys.size
    assert mx1 == pytest.approx(xs.min(), abs=1.5)
    assert mx2 == pytest.approx(xs.max() + 1, abs=1.5)
    assert my1 == pytest.approx(ys.min(), abs=1.5)
    assert my2 == pytest.approx(ys.max() + 1, abs=1.5)


def test_boxes_outside_the_crop_vanish_rather_than_clamping_to_a_sliver():
    # a tall image: the crop keeps the middle, so a box at the very top is gone
    boxes = loc.map_boxes_to_input_frame([[0.0, 0.0, 10.0, 5.0]], 300, 1200)
    assert boxes == []


# ── A4: localization score on synthetic attention ────────────────────────────

def _full_frame_item():
    return {"boxes_xywh": [[0.0, 0.0, 224.0, 224.0]], "width": 224,
            "height": 224}


TL_4x4 = [(0.0, 0.0, 64.0, 64.0)]     # top-left 4x4 patches, ALREADY in the
                                      # 224 input frame (patch_coverage's own
                                      # contract; the mapping into that frame
                                      # is pinned separately, above)


def test_inbox_mass_concentrated_inside_a_known_box_is_one():
    grid = 14
    cov = loc.patch_coverage(TL_4x4, grid)
    assert cov.sum() == pytest.approx(16.0)      # 16 fully covered patches

    p = np.zeros(grid * grid)
    inside = np.nonzero(cov > 0.999)[0]
    p[inside] = 1.0 / inside.size                 # all mass inside the box
    assert float((p * cov).sum()) == pytest.approx(1.0)


def test_inbox_mass_of_uniform_attention_is_the_box_area_fraction():
    grid = 14
    cov = loc.patch_coverage(TL_4x4, grid)
    p = np.full(grid * grid, 1.0 / (grid * grid))
    assert float((p * cov).sum()) == pytest.approx(float(cov.mean()))
    assert float(cov.mean()) == pytest.approx(16 / 196)


def test_a_whole_image_box_covers_the_whole_crop_end_to_end():
    """Composition check: map_boxes_to_input_frame + patch_coverage. A box
    that is the entire source image must cover every patch of the crop."""
    cov = loc.patch_coverage(
        loc.map_boxes_to_input_frame([[0.0, 0.0, 640.0, 480.0]], 640, 480), 14)
    assert cov.min() == pytest.approx(1.0)
    assert cov.mean() == pytest.approx(1.0)


def test_overlapping_boxes_are_a_union_not_a_sum():
    grid = 14
    twice = loc.patch_coverage(TL_4x4 + TL_4x4, grid)
    once = loc.patch_coverage(TL_4x4, grid)
    assert np.allclose(twice, once)
    assert twice.max() <= 1.0


def test_pointing_hit_follows_the_argmax_patch():
    grid = 14
    cx = (0 % grid + 0.5) * 16
    cy = (0 // grid + 0.5) * 16
    assert loc.union_mask_contains(TL_4x4, cx, cy) is True
    assert loc.union_mask_contains(TL_4x4, 200.0, 200.0) is False


def test_patch_distribution_renormalizes_over_patches_only():
    L, P = 3, 196
    cls = np.zeros((L, P), dtype=np.float32)
    cls[-1] = 0.5 / P                # last block holds only half the mass:
    p = loc.patch_distribution(cls)  # the rest sat on prefix tokens
    assert p.sum() == pytest.approx(1.0)


# ── A4: attention dumps ──────────────────────────────────────────────────────

def _write_probe_json(tmp_path, root, items):
    group = {"status": "frozen", "dataset": "imagenet-1k", "split": "val",
             "seed": 0, "n": len(items), "selector": "test", "items": items}
    group["sha256"] = bps.group_sha256(group)
    doc = {"schema_version": 1, "seed": 0, "groups": {"curated": group}}
    out = tmp_path / "probe_small.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _tiny_probe(tmp_path):
    root = tmp_path / "in"
    d = root / "val" / "n00000001"
    d.mkdir(parents=True)
    items = []
    for i in range(2):
        Image.new("RGB", (40, 30), (30 * i, 60, 90)).save(d / f"i{i}.JPEG")
        items.append({"image_id": f"n00000001/i{i}",
                      "path": f"val/n00000001/i{i}.JPEG", "label": 0,
                      "synset": "n00000001"})
    return root, _write_probe_json(tmp_path, root, items)


def _run_dump(tmp_path, root, probe, variant, out_root, extra=(), seed=0):
    from tools.model_factory import build_model
    torch.manual_seed(seed)          # same seed => byte-identical checkpoint,
    model = build_model("vit_tiny_patch16_224", variant,   # so the skip guard
                        num_classes=10)                    # can actually match
    ckpt = tmp_path / f"{variant}.pth"
    torch.save({"model": model.state_dict(), "epoch": 1}, ckpt)

    old = sys.argv
    try:
        sys.argv = ["dump_attention.py", "--run-id", f"r_{variant}",
                    "--arch", "vit_tiny_patch16_224", "--variant", variant,
                    "--ckpt", str(ckpt), "--probe", str(probe),
                    "--groups", "curated", "--data", str(root),
                    "--out-root", str(out_root), "--device", "cpu",
                    "--num-classes", "10", "--batch-size", "2", *extra]
        assert dmp.main() == 0
    finally:
        sys.argv = old
    return out_root / f"r_{variant}" / "attn"


@pytest.mark.parametrize("variant,want_prefix",
                         [("baseline", 1), ("saga", 1), ("registers", 5)])
def test_dump_handles_every_variants_prefix_tokens(tmp_path, variant,
                                                   want_prefix):
    root, probe = _tiny_probe(tmp_path)
    attn_dir = _run_dump(tmp_path, root, probe, variant, tmp_path / "runs")

    files = sorted(attn_dir.glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0], allow_pickle=False) as z:
        prov = json.loads(str(z["provenance"]))
        assert prov["num_prefix_tokens"] == want_prefix
        assert prov["grid_side"] == 14
        assert z["cls_attn_mean"].shape == (12, 196)
        assert z["cls_attn_heads"].shape[0] == 12
        assert z["cls_attn_heads"].shape[2] == 196
        assert z["full_attn_last4"].shape[0] == 4
        assert z["full_attn_last4"].shape[-1] == 196 + want_prefix
        assert list(z["last4_blocks"]) == [8, 9, 10, 11]
        assert z["token_norms"].shape == (12, 196 + want_prefix)
        assert ("gate_sigmoid" in z) == (variant == "saga")
        if variant == "saga":
            assert z["gate_sigmoid"].shape == (12, 196)
            assert (z["gate_sigmoid"] > 0).all() and (z["gate_sigmoid"] < 1).all()


def test_dumped_cls_rows_sum_to_one_with_the_prefix_mass(tmp_path):
    """The stored patch row plus the stored prefix mass is the full softmax
    row — the A4 'rows sum to 1' check, on the artifact rather than in memory."""
    root, probe = _tiny_probe(tmp_path)
    attn_dir = _run_dump(tmp_path, root, probe, "registers", tmp_path / "runs")
    with np.load(sorted(attn_dir.glob("*.npz"))[0], allow_pickle=False) as z:
        total = (z["cls_attn_mean"].astype(np.float64).sum(axis=1)
                 + z["cls_prefix_mass"].astype(np.float64))
    assert np.allclose(total, 1.0, atol=2e-3)   # fp16 storage


def test_dump_is_idempotent_and_skips_on_the_guard(tmp_path, capsys):
    root, probe = _tiny_probe(tmp_path)
    out_root = tmp_path / "runs"
    attn_dir = _run_dump(tmp_path, root, probe, "baseline", out_root, seed=0)
    stamps = {p.name: p.stat().st_mtime_ns for p in attn_dir.glob("*.npz")}

    # same checkpoint -> a resubmitted job burns no GPU
    _run_dump(tmp_path, root, probe, "baseline", out_root, seed=0)
    assert "0 written, 2 skipped" in capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns
            for p in attn_dir.glob("*.npz")} == stamps

    # a DIFFERENT checkpoint at the same path must re-dump: the guard keys on
    # content, never on the filename
    _run_dump(tmp_path, root, probe, "baseline", out_root, seed=1)
    assert "2 written, 0 skipped" in capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns
            for p in attn_dir.glob("*.npz")} != stamps


def test_a_truncated_dump_is_redone_not_trusted(tmp_path):
    root, probe = _tiny_probe(tmp_path)
    out_root = tmp_path / "runs"
    attn_dir = _run_dump(tmp_path, root, probe, "baseline", out_root, seed=0)
    victim = sorted(attn_dir.glob("*.npz"))[0]
    victim.write_bytes(b"not an npz")

    _run_dump(tmp_path, root, probe, "baseline", out_root, seed=0)
    with np.load(victim, allow_pickle=False) as z:      # readable again
        assert z["cls_attn_mean"].shape == (12, 196)


def test_dump_refuses_a_probe_group_that_is_not_frozen(tmp_path):
    root, _ = _tiny_probe(tmp_path)
    doc = {"schema_version": 1, "groups": {"curated": bps.pending_boxes20()}}
    probe = tmp_path / "pending.json"
    probe.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="not frozen"):
        dmp.load_probe_group(probe, "curated")


def test_capture_does_not_change_the_models_output(tmp_path):
    """A4 explicitly reuses the T4 no-op property — re-pinned here for the
    models this task actually dumps."""
    from saga.attn_extract import capture_attention
    from tools.model_factory import build_model
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    for variant in ("baseline", "saga", "registers"):
        model = build_model("vit_tiny_patch16_224", variant,
                            num_classes=10).eval()
        with torch.no_grad():
            ref = model(x)
        with capture_attention(model) as store, torch.no_grad():
            out = model(x)
        assert torch.allclose(ref, out, atol=1e-4)
        for a in store.values():
            s = a.sum(dim=-1)
            assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


# ── A4: the localization CSV ─────────────────────────────────────────────────

def test_localization_csv_schema_and_absent_handling(tmp_path):
    import csv as _csv
    root, probe = _tiny_probe(tmp_path)
    out_root = tmp_path / "runs"
    _run_dump(tmp_path, root, probe, "baseline", out_root)
    out = tmp_path / "F1_localization.csv"

    old = sys.argv
    try:
        sys.argv = ["localization_score.py", "--runs", "r_baseline,r_missing",
                    "--groups", "curated", "--probe", str(probe),
                    "--out-root", str(out_root), "--out", str(out),
                    "--allow-missing"]
        assert loc.main() == 0
    finally:
        sys.argv = old

    with open(out, encoding="utf-8", newline="") as f:
        rows = list(_csv.DictReader(f))
    assert [*rows[0]] == list(loc.CSV_FIELDS)
    assert {r["run_id"] for r in rows} == {"r_baseline"}   # missing != zero
    assert {r["metric"] for r in rows} == {
        "ring1_mass", "uniform_ring1_mass", "attn_entropy_bits"}
    for r in rows:
        float(r["value"])

    meta = json.loads((out.parent / "F1_localization_meta.json").read_text())
    assert meta["runs"]["r_baseline"]["num_prefix_tokens"] == 1
    assert any("r_missing" in a for a in meta["absent"])


# ── A6: F1 archive + figure ──────────────────────────────────────────────────

def _collect_f1(tmp_path, root, probe, out_root, runs, out):
    from analysis import collect_F1
    old = sys.argv
    try:
        sys.argv = ["collect_F1.py", "--runs", runs, "--probe", str(probe),
                    "--data", str(root), "--out-root", str(out_root),
                    "--out", str(out)]
        assert collect_F1.main() == 0
    finally:
        sys.argv = old


def test_collect_F1_records_absent_runs_instead_of_zero_panels(tmp_path):
    root, probe = _tiny_probe(tmp_path)
    out_root = tmp_path / "runs"
    _run_dump(tmp_path, root, probe, "baseline", out_root)
    _run_dump(tmp_path, root, probe, "saga", out_root)
    out = tmp_path / "F1_teaser.npz"
    _collect_f1(tmp_path, root, probe, out_root,
                "r_baseline,r_saga,r_ttr_missing", out)

    with np.load(out, allow_pickle=False) as z:
        index = json.loads(str(z["index"]))
        keys = set(z.files)

    assert index["grid"] == 14
    assert any("r_ttr_missing" in a for a in index["absent"])
    assert not any(k.startswith("attn/r_ttr_missing/") for k in keys)
    assert "attn/r_baseline/n00000001/i0" in keys
    assert "gate/r_saga/8" in keys           # SAGA carries its gate
    assert not any(k.startswith("gate/r_baseline/") for k in keys)
    assert z_shape(out, "ring1_mask") == (14, 14)
    assert index["provenance"]["r_saga"]["variant"] == "saga"


def z_shape(path, key):
    with np.load(path, allow_pickle=False) as z:
        return z[key].shape


def test_plot_F1_renders_with_a_missing_ttr_column(tmp_path):
    root, probe = _tiny_probe(tmp_path)
    out_root = tmp_path / "runs"
    _run_dump(tmp_path, root, probe, "baseline", out_root)
    _run_dump(tmp_path, root, probe, "saga", out_root)
    archive = tmp_path / "F1_teaser.npz"
    _collect_f1(tmp_path, root, probe, out_root, "r_baseline,r_saga", archive)

    csv_path = tmp_path / "F1_localization.csv"
    old = sys.argv
    try:
        sys.argv = ["localization_score.py", "--runs", "r_baseline,r_saga",
                    "--groups", "curated", "--probe", str(probe),
                    "--out-root", str(out_root), "--out", str(csv_path)]
        assert loc.main() == 0

        from plotting import plot_F1
        pdf = tmp_path / "F1_teaser_draft.pdf"
        sys.argv = ["plot_F1.py", "--archive", str(archive),
                    "--localization", str(csv_path),
                    "--images", "n00000001/i0,n00000001/i1",
                    "--columns",
                    "Baseline=r_baseline,Registers=,TTR=,SAGA=r_saga",
                    "--out", str(pdf)]
        assert plot_F1.main() == 0
    finally:
        sys.argv = old

    assert pdf.is_file() and pdf.stat().st_size > 1000


def test_attn_root_separates_bulk_dumps_from_the_repo_run_tree(tmp_path):
    """The ~3 GB of dumps may live on bulk storage while the committed
    diag/*_addr.json sink maps stay in the repo — so collect_F1 must read the
    two from DIFFERENT roots."""
    root, probe = _tiny_probe(tmp_path)
    bulk = tmp_path / "bulk"
    repo_runs = tmp_path / "repo_runs"

    from tools.model_factory import build_model
    torch.manual_seed(0)
    model = build_model("vit_tiny_patch16_224", "baseline", num_classes=10)
    ckpt = tmp_path / "b.pth"
    torch.save({"model": model.state_dict()}, ckpt)

    old = sys.argv
    try:
        sys.argv = ["dump_attention.py", "--run-id", "r_b",
                    "--arch", "vit_tiny_patch16_224", "--variant", "baseline",
                    "--ckpt", str(ckpt), "--probe", str(probe),
                    "--groups", "curated", "--data", str(root),
                    "--out-root", str(repo_runs), "--attn-root", str(bulk),
                    "--device", "cpu", "--num-classes", "10"]
        assert dmp.main() == 0
        # dumps went to the bulk root only
        assert list(bulk.glob("r_b/attn/*.npz"))
        assert not repo_runs.exists() or not list(repo_runs.glob("**/*.npz"))

        # a sink map committed in the REPO tree is still picked up
        diag = repo_runs / "r_b" / "diag"
        diag.mkdir(parents=True, exist_ok=True)
        (diag / "x_last_addr.json").write_text(
            json.dumps({"freq_canon": [1.0 / 196] * 196}), encoding="utf-8")

        from analysis import collect_F1
        out = tmp_path / "F1.npz"
        sys.argv = ["collect_F1.py", "--runs", "r_b", "--probe", str(probe),
                    "--data", str(root), "--out-root", str(repo_runs),
                    "--attn-root", str(bulk), "--out", str(out)]
        assert collect_F1.main() == 0
    finally:
        sys.argv = old

    with np.load(out, allow_pickle=False) as z:
        assert "attn/r_b/n00000001/i0" in z.files    # from bulk
        assert z["sink/r_b"].shape == (14, 14)       # from the repo tree


def test_localization_fails_loudly_when_dumps_are_missing(tmp_path):
    _, probe = _tiny_probe(tmp_path)
    old = sys.argv
    try:
        sys.argv = ["localization_score.py", "--runs", "nope",
                    "--groups", "curated", "--probe", str(probe),
                    "--out-root", str(tmp_path / "runs"),
                    "--out", str(tmp_path / "x.csv")]
        with pytest.raises(SystemExit, match="missing dump"):
            loc.main()
    finally:
        sys.argv = old
