"""
tests/test_task09_dense.py
==========================
TASK-09 (dense prediction toward Gate 2) — CPU tests on fake data.

Coverage, grouped by the bug or contract each block pins:
  B5  double normalization  — the tensor reaching the backbone is the
      dataloader's single-normalized image; re-enabling the detector's own
      ImageNet normalization is detected and refused.
  B6  registers backbone    — the timm register-token model runs through its
      own forward_intermediates() at 800x1216 and returns exactly
      (H/16)*(W/16) patch tokens; the SAGA backbone interpolates gate AND
      pos-embed to the same non-square grid; SAGAViT refuses register models
      outright; register checkpoints strict-load (reg_token really lands).
  COCO scoring              — predictions are mapped back to official
      category ids before scoring; the un-mapped (old) path is provably
      wrong; per-category AP extraction is correct.
  B4  mIoU ignore index     — hand-computed IoU on a synthetic image with an
      ignore border; the old formula is provably different.
  contracts                 — matrix resolves the COMMITTED hyperparameters,
      backbone sha256s match the committed eval JSONs, job files are in sync
      with the matrix, log schema, resume/sanitise, completion fast-path,
      committed 20-image probe list, and the two end-to-end run contracts.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from detection.data.coco_dataset import COCODetectionDataset
from detection.data.transforms import get_val_transforms
from detection.models.backbone import DetectionBackbone
from detection.models.detector import (IDENTITY_MEAN, IDENTITY_STD,
                                       IMAGENET_MEAN, IMAGENET_STD,
                                       build_detector)
from saga.run_registry import file_sha256
from saga.vit import SAGAViT, build_saga_vit
from tools import dense_runtime as dr

ARCH = "vit_base_patch16_224"
TINY = {"depth": 2, "embed_dim": 64, "num_heads": 2}
DENSE_H, DENSE_W = 800, 1216          # task-mandated dense test resolution
DENSE_BASE_PORT = 29850               # scripts/gen_dense_jobs.py --base-port
DENSE_SMOKE_PORTS = (29860, 29861)
DENSE_GRID = (DENSE_H // 16, DENSE_W // 16)     # (50, 76)
DENSE_TOKENS = DENSE_GRID[0] * DENSE_GRID[1]    # 3800


# ── helpers ───────────────────────────────────────────────────────────────────

def tiny_backbone(variant, ckpt=None, expected_sha=None, img_size=224):
    return DetectionBackbone(
        arch=ARCH, gate=(variant == "saga"),
        registers=4 if variant == "registers" else 0,
        ckpt_path=ckpt, fpn_indices=[0, 1], img_size=img_size,
        expected_sha256=expected_sha, model_kwargs=dict(TINY))


def save_classifier_ckpt(path, variant, img_size=224):
    """A trainer-shaped checkpoint ({'model': sd, ...}) built the way the
    classification trainer builds each variant."""
    if variant == "registers":
        import timm
        model = timm.create_model(ARCH, pretrained=False, num_classes=1000,
                                  img_size=img_size, reg_tokens=4, **TINY)
    else:
        model = build_saga_vit(arch=ARCH, gate=(variant == "saga"),
                               img_size=img_size, patch_size=16,
                               num_classes=1000, **TINY)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    torch.save({"model": model.state_dict(), "epoch": 7, "top1": 12.5}, path)
    return model


def fake_pil(h, w, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def coco_ann_file(path, categories, images, annotations):
    with open(path, "w") as f:
        json.dump({"images": images, "categories": categories,
                   "annotations": annotations,
                   "info": {}, "licenses": []}, f)


# ── B5: double normalization ─────────────────────────────────────────────────

def det_cfg(tmp_ckpt, sha, min_size=64, max_size=64, variant="baseline"):
    return {
        "model": {"arch": ARCH, "gate": variant == "saga",
                  "registers": 4 if variant == "registers" else 0,
                  "fpn_indices": [0, 1, 1, 1], "img_size": 224,
                  "embed_dim": 64, "fpn_out_channels": 8,
                  "num_det_classes": 3,
                  "anchor_sizes": [[8], [16], [32], [64]],
                  "anchor_ratios": [0.5, 1.0, 2.0],
                  "model_kwargs": dict(TINY)},
        "train": {"min_size": min_size, "max_size": max_size},
        "backbone": {"ckpt": str(tmp_ckpt), "sha256": sha},
    }


@pytest.fixture(scope="module")
def tiny_det(tmp_path_factory):
    """A real ViTDetector (tiny backbone) configured with the PRODUCTION
    800/1333 detection resolution, so the normalization check runs on the
    same transform geometry the COCO runs will use."""
    tmp = tmp_path_factory.mktemp("det")
    ckpt = tmp / "baseline.pth"
    save_classifier_ckpt(ckpt, "baseline")
    return build_detector(det_cfg(ckpt, file_sha256(ckpt),
                                  min_size=800, max_size=1333))


def dense_val_tensor(h=DENSE_H, w=DENSE_W, seed=1):
    """The dataloader's output for a real-shaped image: resized to
    800/1333 and ImageNet-normalized exactly once."""
    img = fake_pil(h, w, seed=seed)
    target = {"boxes": torch.zeros(0, 4),
              "labels": torch.zeros(0, dtype=torch.long)}
    tensor, _ = get_val_transforms(min_size=800, max_size=1333)(img, target)
    return img, tensor


def test_detector_normalization_is_the_identity(tiny_det):
    """B5: the ONE fix — FasterRCNN's own normalization is disabled."""
    assert list(tiny_det.frcnn.transform.image_mean) == list(IDENTITY_MEAN)
    assert list(tiny_det.frcnn.transform.image_std) == list(IDENTITY_STD)


def test_backbone_input_is_single_normalized(tiny_det):
    """B5: a real-shaped batch reaches the backbone with ImageNet-normalized
    statistics, not double-shifted ones."""
    img, tensor = dense_val_tensor()
    assert tensor.shape == (3, DENSE_H, DENSE_W)   # already at 800 min side
    stats = tiny_det.check_normalization([tensor])

    # independently: normalizing the raw image once gives these statistics
    raw = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    expect_mean = ((raw - mean) / std).mean(dim=(1, 2))
    got = torch.tensor(stats["backbone_input_channel_mean"])
    assert torch.allclose(got, expect_mean, atol=2e-2), (got, expect_mean)

    assert stats["max_abs_elementwise_diff"] < 1e-4
    assert stats["dist_to_double_normalized"] > 1.0
    assert stats["dist_to_single_normalized"] < 1e-3


def test_double_normalization_is_detected(tiny_det):
    """Re-enabling the detector's ImageNet normalization (the B5 bug) must
    fail loudly, on the same code path the trainer runs at step 0."""
    _, tensor = dense_val_tensor(seed=2)
    tr = tiny_det.frcnn.transform
    saved = (list(tr.image_mean), list(tr.image_std))
    try:
        tr.image_mean = list(IMAGENET_MEAN)     # the bug, restored
        tr.image_std = list(IMAGENET_STD)
        with pytest.raises(AssertionError, match="B5"):
            tiny_det.check_normalization([tensor])
    finally:
        tr.image_mean, tr.image_std = saved
    # and the guard is clean again afterwards
    tiny_det.check_normalization([tensor])


def test_check_normalization_needs_images(tiny_det):
    with pytest.raises(ValueError):
        tiny_det.check_normalization([])


def test_check_normalization_handles_a_padded_mixed_size_batch(tiny_det):
    """Real batches hold differently-shaped images, which the detector pads
    to a common size (and to a multiple of 32). The guard must compare only
    the unpadded region — otherwise the zero padding would look like a
    shifted distribution and the check would either false-alarm or, worse,
    mask a real double shift."""
    _, a = dense_val_tensor(h=800, w=1216, seed=3)        # 800x1216
    _, b = dense_val_tensor(h=480, w=640, seed=4)         # -> 800x1067
    assert a.shape[-2:] != b.shape[-2:]
    stats = tiny_det.check_normalization([a, b])
    assert stats["n_images"] == 2
    assert stats["max_abs_elementwise_diff"] < 1e-4
    assert stats["dist_to_double_normalized"] > 1.0


@pytest.mark.parametrize("variant", ["saga", "registers"])
def test_detector_trains_and_infers_at_production_resolution(tmp_path,
                                                             variant):
    """The FPN reshape, MultiScaleRoIAlign's scale inference and the SAGA
    gate/pos-embed interpolation are only exercised at real detection sizes —
    a 64x64 test would pass with a broken grid. Also checks that gradients
    reach the gate through the whole detector."""
    ckpt = tmp_path / f"{variant}.pth"
    save_classifier_ckpt(ckpt, variant)
    det = build_detector(det_cfg(ckpt, file_sha256(ckpt), min_size=800,
                                 max_size=1333, variant=variant))
    img = torch.randn(3, DENSE_H, DENSE_W)
    target = {"boxes": torch.tensor([[10., 20., 200., 300.],
                                     [400., 100., 900., 700.]]),
              "labels": torch.tensor([1, 2])}

    det.train()
    losses = det([img], [target])
    assert set(losses) == {"loss_objectness", "loss_rpn_box_reg",
                           "loss_classifier", "loss_box_reg"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert det.backbone.last_patch_grid == DENSE_GRID
    if variant == "saga":
        grad = det.backbone.vit.blocks[0].attn.gate.phi.grad
        assert grad is not None and torch.isfinite(grad).all()

    det.eval()
    with torch.no_grad():
        out = det([img])
    assert out[0]["boxes"].shape[1] == 4
    assert torch.isfinite(out[0]["boxes"]).all()
    assert out[0]["labels"].numel() == 0 or int(out[0]["labels"].max()) <= 3


# ── B6: registers / SAGA backbones at a real dense resolution ───────────────

def test_registers_backbone_token_count_at_800x1216():
    bb = tiny_backbone("registers")
    assert bb.num_prefix_tokens == 5          # cls + 4 registers
    assert bb.vit.reg_token is not None       # the parameter B6 used to drop
    out = bb(torch.randn(1, 3, DENSE_H, DENSE_W))
    assert len(out) == 2
    for feat in out:
        assert feat.shape == (1, DENSE_TOKENS, 64)
        assert torch.isfinite(feat).all()
    assert bb.last_patch_grid == DENSE_GRID


def test_saga_backbone_interpolates_gate_and_posembed_at_800x1216():
    bb = tiny_backbone("saga")
    # non-trivial gate values, so a dropped interpolation would show
    with torch.no_grad():
        for blk in bb.vit.blocks:
            blk.attn.gate.phi.normal_(0.0, 1.0)
    out = bb(torch.randn(1, 3, DENSE_H, DENSE_W))
    for feat in out:
        assert feat.shape == (1, DENSE_TOKENS, 64)
        assert torch.isfinite(feat).all()
    assert bb.last_patch_grid == DENSE_GRID
    for blk in bb.vit.blocks:
        assert blk.attn.gate.current_grid_size == DENSE_GRID
        # the stored parameter keeps the 224 training grid
        assert blk.attn.gate.get_gate_maps().shape == (2, 14, 14)
    # pos-embed interpolation actually ran and produced the right length
    x = bb.vit.patch_embed(torch.randn(1, 3, DENSE_H, DENSE_W))
    bb.vit.last_patch_grid = DENSE_GRID
    seq = torch.cat([bb.vit.cls_token, x.flatten(1, 2)], dim=1)
    pos = bb.vit._interpolate_pos_embed(seq)
    assert pos.shape == (1, DENSE_TOKENS + 1, 64)
    assert torch.isfinite(pos).all()


def test_baseline_backbone_at_800x1216():
    bb = tiny_backbone("baseline")
    out = bb(torch.randn(1, 3, DENSE_H, DENSE_W))
    assert out[0].shape == (1, DENSE_TOKENS, 64)
    assert bb.last_patch_grid == DENSE_GRID


@pytest.mark.parametrize("variant", ["baseline", "saga", "registers"])
@pytest.mark.parametrize("indices", [[0, 1, 1, 1], [1, 0], [1, 1]])
def test_every_variant_answers_fpn_indices_identically(variant, indices):
    """One feature map per requested index, in the requested ORDER, even for
    repeated or descending index lists. timm's forward_intermediates appends
    one entry per matching block in block order, so the registers path has to
    re-expand — without that it silently hands the neck too few feature maps
    and the anchor generator raises (or, worse, levels get mis-assigned)."""
    bb = DetectionBackbone(arch=ARCH, gate=(variant == "saga"),
                           registers=4 if variant == "registers" else 0,
                           ckpt_path=None, fpn_indices=indices,
                           img_size=224, model_kwargs=dict(TINY))
    out = bb(torch.randn(1, 3, 224, 224))
    assert len(out) == len(indices)
    for feat in out:
        assert feat.shape == (1, 196, 64)
    # entries requested twice must be the same tensor content
    for i, a in enumerate(indices):
        for j, b in enumerate(indices):
            if a == b:
                assert torch.equal(out[i], out[j])
            else:
                assert not torch.equal(out[i], out[j])


def test_sagavit_refuses_register_models():
    """B6 root cause: wrapping a reg-token model dropped reg_token."""
    import timm
    reg = timm.create_model(ARCH, pretrained=False, num_classes=10,
                            img_size=224, reg_tokens=4, **TINY)
    with pytest.raises(ValueError, match="B6"):
        SAGAViT(reg)
    with pytest.raises(ValueError, match="B6"):
        build_saga_vit(arch=ARCH, gate=True, reg_tokens=4, **TINY)


def test_posembed_interpolation_rejects_non_square_source():
    """The [:1]/[1:] split is only valid for one prefix token — the tripwire
    fires if a future edit sneaks a multi-prefix pos_embed in."""
    bb = tiny_backbone("baseline")
    bb.vit.num_prefix_tokens = 5
    with pytest.raises(RuntimeError, match="B6"):
        bb.vit._interpolate_pos_embed(torch.zeros(1, 10, 64))
    bb.vit.num_prefix_tokens = 1
    # a pos_embed carrying 200 patch positions is not a square grid; asking
    # for a different (300-token) input forces the reshape path
    bb.vit.pos_embed = torch.nn.Parameter(torch.zeros(1, 200 + 1, 64))
    bb.vit.last_patch_grid = (10, 30)
    with pytest.raises(RuntimeError, match="square"):
        bb.vit._interpolate_pos_embed(torch.zeros(1, 301, 64))
    # and a grid that disagrees with the token count is refused too
    bb.vit.pos_embed = torch.nn.Parameter(torch.zeros(1, 196 + 1, 64))
    bb.vit.last_patch_grid = (10, 30)
    with pytest.raises(RuntimeError, match="disagrees"):
        bb.vit._interpolate_pos_embed(torch.zeros(1, 251, 64))


@pytest.mark.parametrize("variant", ["baseline", "saga", "registers"])
def test_checkpoint_strict_loads_into_dense_backbone(tmp_path, variant):
    """Every variant's classification checkpoint loads with zero missing /
    zero unexpected keys — and the register tokens really arrive."""
    ckpt = tmp_path / f"{variant}.pth"
    src = save_classifier_ckpt(ckpt, variant)
    bb = tiny_backbone(variant, ckpt=ckpt, expected_sha=file_sha256(ckpt))
    assert bb.ckpt_sha256 == file_sha256(ckpt)
    assert bb.ckpt_meta["top1"] == 12.5
    src_sd = src.state_dict()
    for key in ("reg_token", "blocks.0.attn.gate.phi"):
        if key in src_sd:
            assert torch.equal(dict(bb.vit.state_dict())[key], src_sd[key])
    # a shrunken-but-real forward still works after loading
    assert bb(torch.randn(1, 3, 224, 224))[0].shape == (1, 196, 64)


def test_backbone_sha_mismatch_is_refused(tmp_path):
    ckpt = tmp_path / "baseline.pth"
    save_classifier_ckpt(ckpt, "baseline")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        tiny_backbone("baseline", ckpt=ckpt, expected_sha="0" * 64)


def test_backbone_rejects_bad_fpn_indices():
    with pytest.raises(ValueError, match="out of range"):
        DetectionBackbone(arch=ARCH, gate=False, registers=0, ckpt_path=None,
                          fpn_indices=[0, 99], model_kwargs=dict(TINY))
    with pytest.raises(ValueError, match="mutually exclusive"):
        DetectionBackbone(arch=ARCH, gate=True, registers=4, ckpt_path=None,
                          fpn_indices=[0], model_kwargs=dict(TINY))


def test_superseded_dense_scripts_refuse_to_run():
    """A runnable path to a void number is a defect, not a convenience.
    segmentation/tools/evaluate.py never applied its own ignore_index (B4)
    and averaged per-image IoU; detection/tools/evaluate.py and analyze.py
    scored resized-frame boxes against original-frame ground truth —
    analyze.py computing the very per-category / AP_S numbers Gate 2 turns
    on. All three now hard-exit, with their bodies kept for provenance."""
    import importlib
    for mod, needle in (("segmentation.tools.evaluate", "B4"),
                        ("detection.tools.evaluate", "resized frame"),
                        ("detection.tools.analyze", "resized frame")):
        m = importlib.import_module(mod)
        with pytest.raises(SystemExit) as exc:
            m.main()
        assert "SUPERSEDED" in str(exc.value)
        assert needle in str(exc.value)
        assert hasattr(m, "_disabled_main")     # body kept for provenance


def test_normalization_is_present_exactly_once_in_the_dataloader():
    """The B5 fix depends on the dataloader being the ONE normalizer. A guard
    that only checks "the detector did not change the tensor" would also pass
    if nothing normalized at all, so pin the transform composition itself."""
    from detection.data.transforms import (Normalize, get_train_transforms,
                                           get_val_transforms)
    for build in (get_train_transforms, get_val_transforms):
        norms = [t for t in build(800, 1333).transforms
                 if isinstance(t, Normalize)]
        assert len(norms) == 1, build.__name__
        assert tuple(norms[0].mean) == IMAGENET_MEAN
        assert tuple(norms[0].std) == IMAGENET_STD


def test_segmentation_backbone_is_the_same_object():
    """One implementation — the seg copy used to drift (and carried B6)."""
    from segmentation.models.backbone import DetectionBackbone as SegBB
    assert SegBB is DetectionBackbone


# ── COCO scoring: category ids and per-category AP ──────────────────────────

CATS = [{"id": 1, "name": "a"}, {"id": 3, "name": "b"},
        {"id": 90, "name": "c"}]


@pytest.fixture
def mini_coco(tmp_path):
    """2 annotated images + 1 empty one; deliberately non-contiguous ids."""
    from PIL import Image
    img_dir = tmp_path / "val"
    img_dir.mkdir()
    images, annotations = [], []
    for i in (1, 2, 3):
        Image.fromarray(np.full((64, 64, 3), 40 * i, dtype=np.uint8)).save(
            img_dir / f"{i:06d}.jpg")
        images.append({"id": i, "file_name": f"{i:06d}.jpg",
                       "height": 64, "width": 64})
    boxes = {1: [(1, [8, 8, 16, 16]), (90, [32, 32, 20, 20])],
             2: [(3, [4, 4, 40, 40])]}
    aid = 1
    for img_id, items in boxes.items():
        for cat, box in items:
            annotations.append({"id": aid, "image_id": img_id,
                                "category_id": cat, "bbox": box,
                                "area": box[2] * box[3], "iscrowd": 0})
            aid += 1
    ann = tmp_path / "instances.json"
    coco_ann_file(ann, CATS, images, annotations)
    return img_dir, ann, annotations


def test_category_id_mapping_is_a_bijection(mini_coco):
    img_dir, ann, _ = mini_coco
    ds = COCODetectionDataset(img_dir, ann)
    assert ds.coco_id_to_idx == {1: 1, 3: 2, 90: 3}
    assert ds.idx_to_coco_id == {1: 1, 2: 3, 3: 90}
    for coco_id, idx in ds.coco_id_to_idx.items():
        assert ds.idx_to_coco_id[idx] == coco_id


def test_include_empty_controls_val_coverage(mini_coco):
    img_dir, ann, _ = mini_coco
    assert len(COCODetectionDataset(img_dir, ann)) == 2
    ds = COCODetectionDataset(img_dir, ann, include_empty=True)
    assert len(ds) == 3
    # the annotation-free image must still be usable: correctly SHAPED empty
    # targets, and the val transform must not choke on them
    ds.transforms = get_val_transforms(min_size=64, max_size=64)
    empty = [i for i in range(len(ds))
             if ds.img_ids[i] == 3]
    assert empty, "image 3 has no annotations and should be included"
    img, target = ds[empty[0]]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert img.shape[0] == 3


def test_mapped_predictions_score_ap_100_unmapped_do_not(mini_coco):
    """The scoring bug this task also had to fix: predictions live in the
    contiguous 1-80 label space, the official ground truth in the original
    1-90 one. Mapping back gives a perfect score; not mapping does not."""
    pytest.importorskip("pycocotools")
    from detection.tools.train import score_coco
    img_dir, ann, annotations = mini_coco
    ds = COCODetectionDataset(img_dir, ann, include_empty=True)
    coco_gt = ds.official_coco_gt()
    names = {int(c["id"]): c["name"] for c in CATS}
    img_ids = [1, 2, 3]

    mapped, unmapped = [], []
    for a in annotations:
        model_label = ds.coco_id_to_idx[a["category_id"]]
        mapped.append({"image_id": a["image_id"], "score": 0.9,
                       "bbox": list(a["bbox"]),
                       "category_id": ds.idx_to_coco_id[model_label]})
        unmapped.append({"image_id": a["image_id"], "score": 0.9,
                         "bbox": list(a["bbox"]),
                         "category_id": model_label})

    metrics, per_cat = score_coco(mapped, coco_gt, img_ids, names)
    assert metrics["AP"] == pytest.approx(100.0, abs=1e-6)
    assert metrics["AP50"] == pytest.approx(100.0, abs=1e-6)
    # COCOeval's -1 "no ground truth in this slice" sentinel must never be
    # written as a number (these mini boxes are all small/medium)
    assert metrics["AP_L"] is None
    assert all(v is None or v >= 0.0 for v in metrics.values())
    assert {c["name"] for c in per_cat} == {"a", "b", "c"}
    assert all(c["AP"] == pytest.approx(100.0, abs=1e-6) for c in per_cat)
    # small vs large split is populated where the boxes fall
    assert any(c["AP_S"] is not None for c in per_cat)

    bad, _ = score_coco(unmapped, ds.official_coco_gt(), img_ids, names)
    assert bad["AP"] < 100.0


def test_per_category_ap_is_none_for_absent_classes(mini_coco):
    pytest.importorskip("pycocotools")
    from detection.tools.train import score_coco
    img_dir, ann, annotations = mini_coco
    ds = COCODetectionDataset(img_dir, ann, include_empty=True)
    names = {int(c["id"]): c["name"] for c in CATS}
    # score only image 2 -> categories 1 and 90 have no ground truth there
    only2 = [{"image_id": 2, "score": 0.9, "bbox": [4, 4, 40, 40],
              "category_id": 3}]
    _, per_cat = score_coco(only2, ds.official_coco_gt(), [2], names)
    by_name = {c["name"]: c for c in per_cat}
    assert by_name["b"]["AP"] == pytest.approx(100.0, abs=1e-6)
    assert by_name["a"]["AP"] is None and by_name["c"]["AP"] is None


def test_box_frame_roundtrip_is_the_EXACT_inverse_of_the_resize():
    """boxes_to_original_frame must undo ResizeDetection to float precision,
    not merely to within a pixel. Inverting with a per-axis size ratio
    (orig/round(orig*scale)) instead of the recorded scalar leaves a
    systematic sub-pixel stretch that costs a perfect detector several points
    of AP_S — so the tolerance here is 1e-9, not 1 px."""
    from detection.data.transforms import ResizeDetection
    from detection.tools.train import boxes_to_original_frame
    orig = np.array([[10., 20., 110., 220.], [0., 0., 5., 7.],
                     [300., 200., 480., 400.]])
    worst_ratio_err = 0.0
    for (h, w) in ((480, 640), (640, 480), (333, 500), (800, 1216),
                   (427, 640), (1000, 300), (612, 612), (2048, 300),
                   (640, 638), (639, 641), (801, 1334)):
        boxes = np.clip(orig, 0, [w, h, w, h])
        target = {"boxes": torch.from_numpy(boxes.copy()).double()}
        img = fake_pil(h, w, seed=7)
        img2, target = ResizeDetection(800, 1333)(img, target)
        assert "resize_scale" in target
        back = boxes_to_original_frame(target["boxes"].numpy(),
                                       target["resize_scale"].item())
        assert np.allclose(back, boxes, atol=1e-9, rtol=0), (h, w, back)

        # and show that the size-ratio inversion really is worse
        mh, mw = img2.size[1], img2.size[0]
        ratio = target["boxes"].numpy().copy()
        ratio[:, [0, 2]] *= w / mw
        ratio[:, [1, 3]] *= h / mh
        worst_ratio_err = max(worst_ratio_err,
                              float(np.abs(ratio - boxes).max()))
    assert worst_ratio_err > 1e-3, (
        "expected the per-axis size-ratio inversion to be measurably worse "
        "than the recorded scalar; if it is not, this test has stopped "
        "distinguishing the two")


def test_a_perfect_detector_scores_ap_100_through_the_whole_eval_chain(
        mini_coco, tmp_path):
    """End-to-end pin for the detection metric: a model that returns exactly
    the ground truth (in the frame it was handed, which is what a detector
    does) must score AP = 100. This catches BOTH eval-space bugs at once —
    the contiguous-vs-official category ids and the resized-vs-original box
    coordinates. Before the coordinate fix this returned AP ~ 0."""
    pytest.importorskip("pycocotools")
    from torch.utils.data import DataLoader
    from detection.data.coco_dataset import collate_fn

    img_dir, ann, annotations = mini_coco
    ds = COCODetectionDataset(img_dir, ann, include_empty=True,
                              transforms=get_val_transforms(min_size=800,
                                                            max_size=1333))
    gt_by_image = {}
    for a in annotations:
        gt_by_image.setdefault(a["image_id"], []).append(a)

    class PerfectModel:
        """Returns the ground-truth boxes expressed in the coordinate frame
        of the tensor it is handed — exactly what a detector produces."""

        current_id = None

        def eval(self):
            return self

        def train(self):
            return self

        def __call__(self, images):
            out = []
            for img in images:
                mh, mw = int(img.shape[-2]), int(img.shape[-1])
                ry, rx = mh / 64.0, mw / 64.0   # fixture originals are 64x64
                boxes, labels = [], []
                for a in gt_by_image.get(self.current_id, []):
                    x, y, bw, bh = a["bbox"]
                    boxes.append([x * rx, y * ry, (x + bw) * rx,
                                  (y + bh) * ry])
                    labels.append(ds.coco_id_to_idx[a["category_id"]])
                out.append({
                    "boxes": torch.tensor(boxes).reshape(-1, 4).float(),
                    "scores": torch.ones(len(boxes)),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                })
            return out

    model = PerfectModel()
    base = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                      collate_fn=collate_fn)

    class SequencedLoader:
        """batch_size 1, so the stub can know which image it is looking at."""

        def __iter__(self):
            for images, targets in base:
                model.current_id = int(targets[0]["image_id"].item())
                yield images, targets

    from detection.tools.train import infer_val_shard
    results, image_ids = infer_val_shard(
        model, SequencedLoader(), torch.device("cpu"), ds.idx_to_coco_id)

    assert len(image_ids) == 3
    # every returned box must be back in ORIGINAL coordinates == the GT bbox
    by_key = {(r["image_id"], r["category_id"]): r["bbox"] for r in results}
    for a in annotations:
        got = by_key[(a["image_id"], a["category_id"])]
        # the fixture images are square, so the forward scale is exact and
        # the round trip must land on the ground truth to rounding precision
        assert np.allclose(got, a["bbox"], atol=1e-2), (a, got)

    from detection.tools.train import score_coco
    names = {int(c["id"]): c["name"] for c in CATS}
    metrics, _ = score_coco(results, ds.official_coco_gt(), image_ids, names)
    assert metrics["AP"] == pytest.approx(100.0, abs=1e-6), metrics
    assert metrics["AP50"] == pytest.approx(100.0, abs=1e-6)


def test_scoring_does_not_pollute_the_detections_artifact(mini_coco):
    """pycocotools' loadRes MUTATES the dicts it is handed — it adds a
    box-outline `segmentation` polygon, an `area`, a synthetic `id` and
    `iscrowd`. Those same dicts are what detections_val.json is written
    from, so without a copy the mandated artifact would carry a fabricated
    segmentation mask the model never predicted (and be ~2.2x larger)."""
    pytest.importorskip("pycocotools")
    from detection.tools.train import score_coco
    img_dir, ann, annotations = mini_coco
    ds = COCODetectionDataset(img_dir, ann, include_empty=True)
    results = [{"image_id": a["image_id"],
                "category_id": a["category_id"],
                "bbox": list(a["bbox"]), "score": 0.9}
               for a in annotations]
    before = [dict(r) for r in results]

    score_coco(results, ds.official_coco_gt(), ds.img_ids,
               {int(c["id"]): c["name"] for c in CATS})

    assert results == before, "score_coco mutated the caller's predictions"
    for r in results:
        assert set(r) == {"image_id", "category_id", "bbox", "score"}


def test_legacy_get_coco_api_now_uses_the_model_label_space(mini_coco):
    pytest.importorskip("pycocotools")
    img_dir, ann, _ = mini_coco
    ds = COCODetectionDataset(img_dir, ann)
    gt = ds.get_coco_api()
    assert sorted(gt.getCatIds()) == [1, 2, 3]
    for a in gt.loadAnns(gt.getAnnIds()):
        assert a["category_id"] in (1, 2, 3)
    # nothing dropped: all 3 annotations survive the remap
    assert len(gt.loadAnns(gt.getAnnIds())) == 3


# ── B4: mIoU ignore index ────────────────────────────────────────────────────

def legacy_miou(preds, target, num_classes):
    """The committed (buggy) metric: ignore pixels stay in the union."""
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    for c in range(num_classes):
        pm, gm = preds == c, target == c
        inter[c] = (pm & gm).sum()
        union[c] = (pm | gm).sum()
    iou = inter / (union + 1e-10)
    valid = union > 0
    return float(iou[valid].mean() * 100)


def test_miou_ignores_border_exactly_hand_computed():
    """6x6 image, 1-pixel ignore border, 2 classes. Hand computation:

    Labelled region is the inner 4x4 = 16 pixels: left 2 columns class 0
    (8 px), right 2 columns class 1 (8 px). The prediction gets the whole
    left half right, mislabels 2 of class 1's pixels as class 0, and
    predicts class 0 over the entire ignore border.

      class 0: inter 8, gt 8, pred 8 + 2 = 10  -> union 10  -> IoU 0.8
      class 1: inter 6, gt 8, pred 6           -> union 8   -> IoU 0.75
      mIoU = (0.8 + 0.75) / 2 = 0.775  ->  77.5 %

    The 20 border pixels must appear NOWHERE in those counts.
    """
    from segmentation.tools.train import (summarize_confusion,
                                          update_confusion)
    target = torch.full((6, 6), 255, dtype=torch.long)
    target[1:5, 1:3] = 0
    target[1:5, 3:5] = 1
    preds = torch.zeros((6, 6), dtype=torch.long)   # border predicted as 0
    preds[1:5, 1:3] = 0
    preds[1:5, 3:5] = 1
    preds[1, 3] = 0                                  # 2 class-1 pixels missed
    preds[2, 3] = 0

    conf = torch.zeros(2, 2, dtype=torch.int64)
    update_confusion(conf, preds, target, 2)

    assert conf.sum().item() == 16                   # only labelled pixels
    assert conf.tolist() == [[8, 0], [2, 6]]
    summary, iou, inter, gt, pred, union = summarize_confusion(conf)
    assert inter.tolist() == [8, 6]
    assert gt.tolist() == [8, 8]
    assert pred.tolist() == [10, 6]
    assert union.tolist() == [10, 8]
    assert iou[0] == pytest.approx(0.8)
    assert iou[1] == pytest.approx(0.75)
    assert summary["mIoU"] == pytest.approx(77.5)
    assert summary["n_classes_scored"] == 2
    assert summary["n_labelled_pixels"] == 16
    assert summary["pixel_acc"] == pytest.approx(14 / 16 * 100)
    assert summary["mean_acc"] == pytest.approx((8 / 8 + 6 / 8) / 2 * 100)

    # the old formula counted the 20 ignored pixels into class 0's union
    assert legacy_miou(preds, target, 2) == pytest.approx(
        ((8 / 30) + 0.75) / 2 * 100, abs=1e-4)
    assert abs(legacy_miou(preds, target, 2) - summary["mIoU"]) > 20


def test_update_confusion_rejects_out_of_range_labels():
    from segmentation.tools.train import update_confusion
    conf = torch.zeros(2, 2, dtype=torch.int64)
    target = torch.tensor([[0, 5]], dtype=torch.long)
    with pytest.raises(ValueError, match="outside"):
        update_confusion(conf, torch.zeros_like(target), target, 2)


def test_all_ignore_image_contributes_nothing():
    from segmentation.tools.train import update_confusion
    conf = torch.zeros(3, 3, dtype=torch.int64)
    target = torch.full((4, 4), 255, dtype=torch.long)
    update_confusion(conf, torch.ones_like(target), target, 3)
    assert conf.sum().item() == 0


ADE_REAL_NAMES = [
    "wall",
    "building, edifice",
    "sky",
    "person, individual, someone, somebody, mortal, soul",
]


def test_class_names_survive_the_real_ade20k_file_format(tmp_path):
    """The official objectInfo150.csv is TAB-separated (its Name column
    contains commas). Assuming a comma delimiter would yield 150 MISSING
    names and block Phase C's mandated sky/wall/floor rows."""
    from segmentation.tools.train import load_class_names
    tab = tmp_path / "tab"
    tab.mkdir()
    with open(tab / "objectInfo150.csv", "w", encoding="utf-8") as f:
        f.write("Idx\tRatio\tTrain\tVal\tName\n")
        for i, n in enumerate(ADE_REAL_NAMES, 1):
            f.write(f"{i}\t0.1\t1\t1\t{n}\n")
    names, source = load_class_names(tab, len(ADE_REAL_NAMES))
    assert names == ADE_REAL_NAMES
    assert "tab" in source

    comma = tmp_path / "comma"          # a comma-separated variant still works
    comma.mkdir()
    with open(comma / "objectInfo150.csv", "w", encoding="utf-8") as f:
        f.write("Idx,Ratio,Train,Val,Name\n1,0.1,1,1,wall\n2,0.1,1,1,sky\n")
    names, source = load_class_names(comma, 2)
    assert names == ["wall", "sky"] and "comma" in source

    missing, reason = load_class_names(tmp_path / "nowhere", 150)
    assert missing is None and "not found" in reason
    # wrong class count is reported, never silently truncated
    bad, reason = load_class_names(tab, 150)
    assert bad is None and "150" in reason

    # The class index comes from ROW POSITION. A file re-sorted by Name has
    # the right count and non-empty names, so without checking the Idx
    # column every name in the committed table would be attached to the
    # wrong IoU. It must come back MISSING, not misaligned.
    resorted = tmp_path / "resorted"
    resorted.mkdir()
    with open(resorted / "objectInfo150.csv", "w", encoding="utf-8") as f:
        f.write("Idx\tRatio\tTrain\tVal\tName\n")
        for i, n in sorted(enumerate(ADE_REAL_NAMES, 1), key=lambda t: t[1]):
            f.write(f"{i}\t0.1\t1\t1\t{n}\n")
    names, reason = load_class_names(resorted, len(ADE_REAL_NAMES))
    assert names is None and "Idx order" in reason

    noidx = tmp_path / "noidx"
    noidx.mkdir()
    with open(noidx / "objectInfo150.csv", "w", encoding="utf-8") as f:
        f.write("Ratio\tName\n")
        for n in ADE_REAL_NAMES:
            f.write(f"0.1\t{n}\n")
    names, reason = load_class_names(noidx, len(ADE_REAL_NAMES))
    assert names is None and "no Idx column" in reason


def test_per_class_csv_quotes_names_containing_commas(tmp_path):
    """A naive ",".join would splice an ADE20K name into extra columns and
    shift every number in that row one field right."""
    from segmentation.tools.train import (PER_CLASS_FIELDS,
                                          write_per_class_csv)
    n = len(ADE_REAL_NAMES)
    iou = np.array([0.5, 0.25, np.nan, 1.0])
    inter = np.arange(n) + 1
    union = (np.arange(n) + 1) * 10
    gt = np.arange(n) + 2
    pred = np.arange(n) + 3
    path = tmp_path / "per_class_iou.csv"
    write_per_class_csv(path, iou, inter, gt, pred, union, ADE_REAL_NAMES)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == PER_CLASS_FIELDS
        rows = list(reader)
    assert len(rows) == n
    for i, row in enumerate(rows):
        assert row["name"] == ADE_REAL_NAMES[i]
        assert row["class_index"] == str(i)
        assert row["intersection"] == str(int(inter[i]))
        assert row["union"] == str(int(union[i]))
        assert row["gt_pixels"] == str(int(gt[i]))
        assert row["pred_pixels"] == str(int(pred[i]))
    assert rows[2]["iou_frac"] == "MISSING"   # NaN never printed as a number
    assert rows[0]["iou_frac"] == "0.500000"


def test_class_absent_from_gt_but_predicted_is_scored_zero():
    """A class only ever predicted still has union > 0, so it counts (IoU 0)
    — that is the standard ADE20K convention and it must not be silently
    dropped."""
    from segmentation.tools.train import (summarize_confusion,
                                          update_confusion)
    conf = torch.zeros(3, 3, dtype=torch.int64)
    target = torch.zeros((2, 2), dtype=torch.long)
    update_confusion(conf, torch.full((2, 2), 1, dtype=torch.long), target, 3)
    summary, iou, *_ = summarize_confusion(conf)
    assert summary["n_classes_scored"] == 2          # classes 0 and 1
    assert iou[0] == 0.0 and iou[1] == 0.0
    assert np.isnan(iou[2])
    assert summary["mIoU"] == 0.0


# ── matrix / config contracts ────────────────────────────────────────────────

MATRIX_PATH = REPO_ROOT / "configs" / "dense_matrix.yaml"
EXPECTED_RUNS = {
    "det_vitb_baseline_s1", "det_vitb_saga_s1", "det_vitb_registers_s1",
    "seg_vitb_baseline_s1", "seg_vitb_saga_s1", "seg_vitb_registers_s1",
}


@pytest.fixture(scope="module")
def matrix():
    return yaml.safe_load(open(MATRIX_PATH))


def test_matrix_holds_exactly_the_mandated_six_runs(matrix):
    assert set(matrix["runs"]) == EXPECTED_RUNS
    for run_id, run in matrix["runs"].items():
        assert run["arch"] == ARCH
        assert run["seed"] == 1
        assert run["task"] == ("detection" if run_id.startswith("det_")
                               else "segmentation")
        assert run["variant"] in ("baseline", "saga", "registers")


@pytest.mark.parametrize("run_id", sorted(EXPECTED_RUNS))
def test_resolved_config_is_the_committed_one(matrix, run_id):
    """Nothing about the training schedule may be re-typed in the matrix: the
    resolved config must equal the committed task config, variant switches
    aside."""
    cfg = dr.resolve_dense_config(matrix, run_id)
    committed = yaml.safe_load(open(REPO_ROOT / cfg["task_config_file"]))
    assert cfg["train"] == committed["train"]
    assert cfg["logging"] == committed["logging"]
    for key, value in committed["model"].items():
        if key in ("gate", "registers"):
            continue
        assert cfg["model"][key] == value, key
    # only the eval block is extended, and only with the MS-TTA knobs
    assert set(cfg["eval"]) - set(committed["eval"]) == {"ms_scales",
                                                         "ms_flip"}
    for key, value in committed["eval"].items():
        assert cfg["eval"][key] == value, key

    variant = matrix["runs"][run_id]["variant"]
    assert cfg["model"]["gate"] is (variant == "saga")
    assert cfg["model"]["registers"] == (4 if variant == "registers" else 0)
    assert cfg["seed"] == 1
    assert cfg["out_root"] == f"results/{cfg['task']}"


def test_detection_schedule_is_the_legacy_one(matrix):
    t = dr.resolve_dense_config(matrix, "det_vitb_saga_s1")["train"]
    assert (t["epochs"], t["batch_size"], t["min_size"], t["max_size"]) == \
        (25, 8, 800, 1333)
    assert (float(t["backbone_lr"]), float(t["head_lr"])) == (1e-5, 1e-4)
    assert t["freeze_backbone_epochs"] == 1 and t["amp"] is True


def test_segmentation_schedule_is_the_legacy_one(matrix):
    cfg = dr.resolve_dense_config(matrix, "seg_vitb_saga_s1")
    t = cfg["train"]
    assert (t["epochs"], t["batch_size"], t["input_size"]) == (80, 16, 512)
    assert t["freeze_backbone_epochs"] == 2
    assert cfg["model"]["num_classes"] == 150


def test_backbone_shas_match_the_committed_eval_jsons(matrix):
    """The two seeded backbones are pinned to the hashes in their own
    committed ImageNet eval JSONs — no hand-typed hash."""
    for key, run in (("vitb_baseline", "e2r_vitb_mixup_baseline_s1"),
                     ("vitb_saga", "e2r_vitb_mixup_saga_s1")):
        spec = matrix["backbones"][key]
        assert spec["run"] == run
        assert spec["ckpt"] == f"results/runs/{run}/ckpt/last.pth"
        source = json.load(open(REPO_ROOT / "results" / "runs" / run /
                                "eval" / "imagenet_val_last.json"))
        assert spec["sha256"] == source["ckpt_sha256"]


def test_registers_fallback_matches_the_legacy_manifest(matrix):
    """The registers fallback must be exactly the manifest row TASK-09 names:
    the legacy ViT-B registers nomix-DIRECTORY last.pth, recipe_actual
    mixup."""
    fb = matrix["backbones"]["vitb_registers"]["fallback"]
    with open(REPO_ROOT / "results" / "legacy" /
              "checkpoint_manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["path"] == fb["ckpt"]]
    assert len(rows) == 1, fb["ckpt"]
    row = rows[0]
    assert row["sha256"] == fb["sha256"]
    assert row["arch"] == "vit_base"
    assert row["variant"] == "registers"
    assert row["recipe_actual"] == "mixup"
    assert row["filename"] == "last.pth"
    assert matrix["backbones"]["vitb_registers"]["sha256"] is None


def test_resolve_backbone_requires_a_FINISHED_pinned_primary(tmp_path):
    """TASK-09 says the e2r registers run is used "if FINISHED by launch
    time". The e2r trainer rewrites ckpt/last.pth every epoch, so "the file
    exists" is not that condition: an in-flight run would hand three GPU-days
    of detection an epoch-0 backbone. A primary must be a completed run AND
    have its hash pinned."""
    primary, fallback = tmp_path / "p.pth", tmp_path / "f.pth"
    fallback.write_bytes(b"x")
    gate = tmp_path / "e2r_run"
    gate.mkdir()

    def spec(**over):
        s = {"run": "P", "ckpt": str(primary), "sha256": None,
             "require_complete": str(gate), "require_epochs": 2,
             "fallback": {"run": "F", "ckpt": str(fallback),
                          "sha256": "abc", "note": "legacy"}}
        s.update(over)
        return s

    # (1) primary file missing -> fallback
    got = dr.resolve_backbone(spec())
    assert (got["source"], got["run"], got["sha256"]) == ("fallback", "F",
                                                          "abc")
    # (2) primary exists but hash unpinned -> STILL the fallback
    primary.write_bytes(b"y")
    assert dr.resolve_backbone(spec())["source"] == "fallback"
    # (3) hash pinned but the e2r run is only mid-flight -> still fallback
    dr.append_log_row(gate / "log.csv", FIELDS, {"epoch": 0})
    dr.atomic_json_dump({"end_time": None}, gate / "meta.json")
    assert dr.resolve_backbone(spec(sha256="deadbeef"))["source"] == "fallback"
    # (4) pinned AND the run finished -> the primary finally wins
    dr.append_log_row(gate / "log.csv", FIELDS, {"epoch": 1})
    dr.atomic_json_dump({"end_time": "2026-09-08T00:00:00Z"},
                        gate / "meta.json")
    got = dr.resolve_backbone(spec(sha256="deadbeef"))
    assert (got["source"], got["run"], got["sha256"]) == ("primary", "P",
                                                          "deadbeef")

    # nothing usable at all is an error naming every rejected candidate
    with pytest.raises(FileNotFoundError) as exc:
        dr.resolve_backbone({"ckpt": str(tmp_path / "nope.pth"),
                             "sha256": "x"})
    assert "nope.pth" in str(exc.value)


def test_committed_registers_backbone_is_gated_and_unpinned(matrix, tmp_path):
    """The committed matrix must be in the state TASK-09 describes today:
    the e2r registers run has not finished, its hash is therefore unpinned,
    and the legacy checkpoint is what the runs will actually use."""
    spec = dict(matrix["backbones"]["vitb_registers"])
    assert spec["sha256"] is None
    assert spec["require_complete"] == \
        "results/runs/e2r_vitb_mixup_registers_s1"
    committed = yaml.safe_load(
        open(REPO_ROOT / "classification" / "configs" / "base.yaml"))
    assert spec["require_epochs"] == committed["train"]["epochs"]

    # even if that checkpoint appeared on disk right now, the unpinned hash
    # keeps it out (the fallback, whose hash IS pinned, wins)
    primary = tmp_path / "last.pth"
    primary.write_bytes(b"pretend e2r registers checkpoint")
    fallback = tmp_path / "legacy.pth"
    fallback.write_bytes(b"legacy")
    spec["ckpt"] = str(primary)
    spec["fallback"] = dict(spec["fallback"], ckpt=str(fallback))
    got = dr.resolve_backbone(spec)
    assert got["source"] == "fallback"
    assert got["sha256"] == \
        matrix["backbones"]["vitb_registers"]["fallback"]["sha256"]


def test_repo_path_keeps_posix_absolute_hpc_paths_absolute():
    """The registers fallback is an absolute /home/vault/... HPC path. On
    Windows `Path('/home/...').is_absolute()` is False, so a naive check
    would rebase it under the repo root and could load a same-named local
    file instead of failing."""
    assert str(dr.repo_path("/home/vault/x/last.pth")).replace("\\", "/") \
        == "/home/vault/x/last.pth"
    assert dr.repo_path("results/runs/x").is_absolute()
    assert dr.repo_path("results/runs/x").parts[-2:] == ("runs", "x")


def test_resolve_dense_config_rejects_bad_runs(matrix):
    with pytest.raises(KeyError):
        dr.resolve_dense_config(matrix, "not_a_run")
    bad = {"task_configs": matrix["task_configs"],
           "out_roots": matrix["out_roots"], "backbones": {},
           "runs": {"x": {"task": "nope", "arch": ARCH, "variant": "saga",
                          "backbone": "b", "seed": 1}}}
    with pytest.raises(ValueError, match="task must be"):
        dr.resolve_dense_config(bad, "x")
    bad["runs"]["x"]["task"] = "detection"
    bad["runs"]["x"]["variant"] = "nope"
    with pytest.raises(ValueError, match="variant must be"):
        dr.resolve_dense_config(bad, "x")


def test_job_files_are_in_sync_with_the_matrix(tmp_path):
    """The committed sbatch/submit scripts must be exactly what the generator
    produces from the committed matrix — byte for byte."""
    from scripts import gen_dense_jobs as gen
    written = gen.render(yaml.safe_load(open(MATRIX_PATH)), tmp_path,
                         DENSE_BASE_PORT)
    assert len(written) == 6 * 2 + 2
    for produced in written:
        committed = REPO_ROOT / "scripts" / produced.relative_to(tmp_path)
        assert committed.exists(), committed
        assert (produced.read_bytes().replace(b"\r\n", b"\n")
                == committed.read_bytes().replace(b"\r\n", b"\n")), committed
        text = committed.read_text()
        if produced.suffix == ".sbatch" and produced.name.startswith(
                ("det_", "seg_")):
            # serialization must be in the HEADER too, not only on
            # submit_<run>.sh's sbatch line: `sbatch scripts/jobs/<run>.sbatch`
            # is the documented way to add one more job to a chain, and
            # without this two jobs of one run could overwrite last.pth
            assert "#SBATCH --dependency=singleton" in text
            # a completion + backbone pre-check must run BEFORE staging
            i_check = text.index("tools/dense_done.py")
            i_stage = text.index("stage_coco" if "stage_coco" in text
                                 else "stage_ade20k")
            assert i_check < text.rindex("--resume auto")
            assert "--require_backbone" in text
            assert i_stage > 0
        if produced.suffix == ".sbatch":
            # .gitattributes forces LF on *.sbatch even on a Windows
            # checkout (a CRLF sbatch fails cryptically on the cluster);
            # the submit_*.sh wrappers are not covered by that rule and may
            # legitimately be CRLF in a Windows working tree — they are only
            # ever executed from the HPC checkout, which is LF.
            assert "\r" not in text
            assert "--partition=a100" in text
            assert "--gres=gpu:a100:4" in text
            assert "configs/dense_matrix.yaml" in text


def _launcher_ports():
    """Every master port any committed launcher can use, repo-wide.

    Array launchers compute their port as `$((BASE + SLURM_ARRAY_TASK_ID))`,
    so grepping for a literal port number MISSES them — which is exactly how
    the first version of this suite failed to notice that
    classification/scripts/e2_nomix_alex.sh already owns 29800+. Each such
    base is expanded over that file's own `#SBATCH --array` range.
    """
    ports = {}
    paths = sorted(REPO_ROOT.rglob("*.sh")) + sorted(REPO_ROOT.rglob("*.sbatch"))
    for path in paths:
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")

        hi = 0
        for line in text.splitlines():
            if "--array=" in line and line.lstrip().startswith("#SBATCH"):
                spec = line.split("--array=")[1].split()[0]
                for part in spec.split(","):
                    for bound in part.split("-"):
                        bound = bound.split(":")[0]
                        if bound.isdigit():
                            hi = max(hi, int(bound))
        for line in text.splitlines():
            if "MASTER_PORT=$((" in line:
                base = line.split("$((")[1].split("+")[0].strip()
                if base.isdigit():
                    for p in range(int(base), int(base) + hi + 1):
                        ports.setdefault(p, set()).add(rel)
            if "--master_port=" in line:
                token = line.split("--master_port=")[1].strip().rstrip("\\").strip()
                if token.isdigit():
                    ports.setdefault(int(token), set()).add(rel)
    return ports


def test_dense_done_reports_complete_incomplete_and_bad_backbone(tmp_path):
    """The generated jobs gate on these exit codes BEFORE staging 18 GB of
    COCO: 0 = complete (skip), 3 = no usable backbone (abort in seconds
    rather than minutes into a 4x24h chain), 1 = proceed, 2 = cannot tell."""
    import subprocess
    ckpt = tmp_path / "bb.pth"
    save_classifier_ckpt(ckpt, "saga")
    matrix = _tiny_matrix("detection", "det_vitb_saga_s1", ckpt,
                          file_sha256(ckpt))
    mpath = tmp_path / "m.yaml"
    mpath.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    out_root = tmp_path / "out"

    def run(*extra, m=mpath):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "dense_done.py"),
             "--matrix", str(m), "--run", "det_vitb_saga_s1",
             "--out_root", str(out_root), *extra],
            capture_output=True, text=True, cwd=str(REPO_ROOT))

    assert run().returncode == 1                      # nothing there yet
    assert run("--require_backbone").returncode == 1   # backbone is fine
    assert "backbone ok" in run("--require_backbone").stdout

    ckpt.unlink()                                      # backbone disappears
    bad = run("--require_backbone")
    assert bad.returncode == 3
    assert "NO USABLE BACKBONE" in bad.stdout

    unknown = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "dense_done.py"),
         "--matrix", str(mpath), "--run", "not_a_run"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert unknown.returncode == 2

    # and a completed run reports 0 without needing the backbone at all
    run_dir = out_root / "det_vitb_saga_s1"
    dr.append_log_row(run_dir / "log.csv", ["epoch"], {"epoch": 0})
    dr.atomic_json_dump({"end_time": "2026-09-08T00:00:00Z"},
                        run_dir / "meta.json")
    assert run("--require_backbone").returncode == 0


def test_no_file_tells_the_human_to_bash_a_glob_of_submit_scripts():
    """`bash scripts/submit_det_*.sh` runs ONLY the first script and passes
    the other two as positional arguments (verified) — the submit scripts
    never read "$@", so two of the three Gate-2 chains would silently never
    be queued. Every instruction must be one command per line."""
    import re
    bad = []
    # Both shapes: `bash .../submit_det_*.sh` AND a bare `submit_seg_vitb_*`
    # reference in prose — the second form is how the instruction slipped
    # back in even with this guard in place.
    # Only the DENSE submit-script names, so a wildcard in real code
    # (`f"submit_{run_id}.sh"`) or an unrelated glob is not flagged.
    pattern = re.compile(r"submit_(?:det|seg)[A-Za-z0-9_]*[*{]")
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in (".sh", ".sbatch", ".md", ".py",
                                       ".yaml", ".yml", ".txt"):
            continue
        if path.name == Path(__file__).name:
            continue                      # this file quotes the bad form
        for i, line in enumerate(path.read_text(encoding="utf-8",
                                                errors="replace").splitlines(),
                                 1):
            if pattern.search(line):
                bad.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not bad, "glob/brace submit instructions found:\n" + "\n".join(bad)


def test_master_ports_are_unique_across_every_launcher():
    """A reused NCCL master port makes two concurrent jobs on one node
    collide. TASK-09's ports must be disjoint from every other committed
    launcher, INCLUDING the array-computed ones."""
    ports = _launcher_ports()

    def is_dense(rel):
        name = rel.rsplit("/", 1)[-1]
        return name.startswith(("det_vitb_", "seg_vitb_", "dense_smoke_",
                                "submit_det_", "submit_seg_"))

    dense, foreign = {}, {}
    for port, files in ports.items():
        mine = {f for f in files if is_dense(f)}
        if mine:
            dense[port] = mine
        others = files - mine
        if others:
            foreign[port] = others

    expected = set(range(DENSE_BASE_PORT, DENSE_BASE_PORT + 6)) |         set(DENSE_SMOKE_PORTS)
    assert set(dense) == expected, sorted(dense)
    assert all(len(f) == 1 for f in dense.values()), dense
    clashes = {p: sorted(foreign[p]) for p in dense if p in foreign}
    assert not clashes, f"TASK-09 ports collide with: {clashes}"
    # the specific collision the adversarial review caught: an array-computed
    # base in classification/scripts/e2_nomix_alex.sh owns 29800+
    assert 29800 in foreign and any("e2_nomix_alex.sh" in f
                                    for f in foreign[29800]), (
        "expected classification/scripts/e2_nomix_alex.sh to still hold "
        "29800 — if that changed, re-check the dense port block")


# ── runtime helpers ──────────────────────────────────────────────────────────

FIELDS = ["epoch", "a", "b"]


def test_log_is_append_safe_and_sanitised(tmp_path):
    log = tmp_path / "log.csv"
    for e in range(4):
        dr.append_log_row(log, FIELDS, {"epoch": e, "a": e * 2})
    assert dr.log_epochs(log) == 4
    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["0", "1", "2", "3"]
    assert rows[0]["b"] == ""                       # missing -> empty, not None

    dr.sanitize_log(log, FIELDS, 2)                 # simulate resume at 2
    assert dr.log_epochs(log) == 2
    dr.append_log_row(log, FIELDS, {"epoch": 2, "a": 99})
    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["0", "1", "2"]
    assert rows[-1]["a"] == "99"

    with pytest.raises(KeyError):
        dr.append_log_row(log, FIELDS, {"epoch": 9, "typo": 1})
    dr.sanitize_log(tmp_path / "absent.csv", FIELDS, 0)   # no-op, no crash


def test_run_is_complete_needs_end_time_and_all_rows(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert not dr.run_is_complete(run, 2)
    dr.atomic_json_dump({"end_time": None}, run / "meta.json")
    assert not dr.run_is_complete(run, 2)
    dr.append_log_row(run / "log.csv", FIELDS, {"epoch": 0})
    dr.atomic_json_dump({"end_time": "2026-09-08T00:00:00Z"},
                        run / "meta.json")
    assert not dr.run_is_complete(run, 2)           # only 1 of 2 epochs
    dr.append_log_row(run / "log.csv", FIELDS, {"epoch": 1})
    assert dr.run_is_complete(run, 2)
    (run / "meta.json").write_text("{not json")
    assert not dr.run_is_complete(run, 2)           # torn meta != complete


def test_schedule_geometry_guards():
    with pytest.raises(RuntimeError, match="steps_per_epoch changed"):
        dr.check_schedule_geometry({"steps_per_epoch": 10, "total_epochs": 5},
                                   11, 5)
    assert dr.check_schedule_geometry(
        {"steps_per_epoch": 10, "total_epochs": 5}, 10, 5) is True
    assert dr.check_schedule_geometry(
        {"steps_per_epoch": 10, "total_epochs": 5}, 10, 2) is False
    assert dr.check_schedule_geometry({}, 10, 5) is True


def test_best_artifact_pair_consistency_is_checked(tmp_path):
    """A job killed inside the best-artifact block leaves a half-updated
    pair. Because best_ap comes back from the checkpoint, a re-run of that
    epoch is not a "new best" and would never rewrite it — so the pair has
    to be reconciled on resume and re-checked before the run is marked
    complete."""
    from detection.tools import train as det_train
    run = tmp_path / "run"
    run.mkdir()
    det = run / "detections_val.json"
    best = run / "coco_eval_best.json"

    assert det_train.check_best_artifacts(run, -1) is None   # nothing yet
    assert det_train.check_best_artifacts(run, 4) == \
        "coco_eval_best.json missing for best epoch 4"

    # detections written, JSON not (killed between the two writes)
    dr.atomic_bytes_write(b"[]", det)
    assert "missing for best epoch" in det_train.check_best_artifacts(run, 4)

    # a consistent pair passes
    dr.atomic_json_dump({"epoch": 4,
                         "detections_sha256": file_sha256(det)}, best)
    assert det_train.check_best_artifacts(run, 4) is None

    # ... and every way it can disagree is caught
    assert "checkpoint's best is 7" in det_train.check_best_artifacts(run, 7)
    dr.atomic_bytes_write(b'[{"image_id": 1}]', det)     # dump changed
    assert "does not match its recorded sha256" in \
        det_train.check_best_artifacts(run, 4)
    det.unlink()
    assert "detections_val.json missing" == \
        det_train.check_best_artifacts(run, 4)
    best.write_text("{tor")
    assert "unreadable" in det_train.check_best_artifacts(run, 4)


def test_segmentation_best_artifact_set_consistency_is_checked(tmp_path):
    from segmentation.tools import train as seg_train
    run = tmp_path / "run"
    run.mkdir()
    assert seg_train.check_best_artifacts(run, -1) is None
    assert "missing for best epoch 3" in \
        seg_train.check_best_artifacts(run, 3)

    dr.atomic_json_dump({"epoch": 3}, run / "miou_ss.json")
    assert "conf_matrix.npz missing" == \
        seg_train.check_best_artifacts(run, 3)
    dr.atomic_npz_save(run / "conf_matrix.npz", conf=np.zeros((2, 2)),
                       epoch=np.array(3))
    assert "per_class_iou.csv missing" == \
        seg_train.check_best_artifacts(run, 3)
    dr.atomic_write_text("class_index\n0\n", run / "per_class_iou.csv")
    assert seg_train.check_best_artifacts(run, 3) is None

    # the npz and the JSON must agree on the epoch
    dr.atomic_npz_save(run / "conf_matrix.npz", conf=np.zeros((2, 2)),
                       epoch=np.array(9))
    assert "conf_matrix.npz is for epoch 9" in \
        seg_train.check_best_artifacts(run, 3)


def test_read_meta_quarantines_a_torn_file(tmp_path):
    """finalize_run rewrites meta.json non-atomically (pre-existing,
    e2r-wide). A node death inside that write must not kill the resumed job
    — and with --dependency=afterany, the whole remaining chain with it."""
    run = tmp_path / "run"
    run.mkdir()
    assert dr.read_meta(run) == {}                 # absent is fine
    dr.atomic_json_dump({"end_time": "x", "seed": 1}, run / "meta.json")
    assert dr.read_meta(run)["seed"] == 1
    (run / "meta.json").write_text('{"end_time": "x", "se')   # torn write
    assert dr.read_meta(run, "r1") == {}
    assert not (run / "meta.json").exists()
    quarantined = list(run.glob("meta.json.corrupt.*"))
    assert len(quarantined) == 1                   # bytes kept for forensics
    assert quarantined[0].read_text().startswith('{"end_time"')
    assert not dr.run_is_complete(run, 1)


def test_clear_stale_artifacts_removes_files_and_dirs(tmp_path):
    keep = tmp_path / "keep.json"
    keep.write_text("{}")
    gone_file = tmp_path / "miou_ms.json"
    gone_file.write_text("{}")
    gone_dir = tmp_path / "preds_fixed20"
    gone_dir.mkdir()
    (gone_dir / "a.png").write_bytes(b"x")

    removed = dr.clear_stale_artifacts([gone_file, gone_dir,
                                        tmp_path / "never_existed"])
    assert len(removed) == 2
    assert not gone_file.exists() and not gone_dir.exists()
    assert keep.exists()


def test_sync_buffers_is_a_noop_without_distribution():
    """Single-process runs must not touch torch.distributed at all (the
    dense trainers are also run non-distributed for smokes and tests)."""
    model = torch.nn.BatchNorm2d(3)
    before = model.running_mean.clone()
    assert dr.sync_buffers_from_rank0(model, False) == 0
    assert torch.equal(model.running_mean, before)


def test_shard_indices_is_an_exact_partition():
    n, world = 2000, 4
    shards = [dr.shard_indices(n, r, world) for r in range(world)]
    flat = [i for s in shards for i in s]
    assert sorted(flat) == list(range(n))
    assert len(flat) == len(set(flat)) == n
    assert dr.shard_indices(7, 0, 3) == [0, 3, 6]
    with pytest.raises(ValueError):
        dr.shard_indices(5, 3, 3)


def test_atomic_writers_leave_no_tmp_files(tmp_path):
    dr.atomic_json_dump({"a": 1}, tmp_path / "x.json")
    dr.atomic_npz_save(tmp_path / "y.npz", conf=np.zeros((2, 2)))
    dr.atomic_write_text("hello\n", tmp_path / "z.csv")
    dr.atomic_bytes_write(b"raw", tmp_path / "w.bin")
    dr.atomic_torch_save({"t": torch.zeros(1)}, tmp_path / "v.pth")
    assert json.load(open(tmp_path / "x.json")) == {"a": 1}
    assert np.load(tmp_path / "y.npz")["conf"].shape == (2, 2)
    assert (tmp_path / "z.csv").read_text() == "hello\n"
    assert (tmp_path / "w.bin").read_bytes() == b"raw"
    assert list(tmp_path.glob("*.tmp*")) == []


# ── the committed 20-image probe list ────────────────────────────────────────

def test_committed_fixed20_list_is_valid_and_reproducible():
    from segmentation.tools.build_fixed20 import build_spec
    path = REPO_ROOT / "results" / "probe" / "ade20k_fixed20.json"
    spec = json.load(open(path))
    assert spec["dataset"] == "ade20k" and spec["split"] == "validation"
    assert spec["n"] == 20 == len(spec["stems"]) == len(set(spec["stems"]))
    assert spec["n_val_source"] == 2000
    assert all(1 <= i <= 2000 for i in spec["ids"])
    assert spec["stems"] == [f"ADE_val_{i:08d}" for i in spec["ids"]]
    assert spec["ids"] == sorted(spec["ids"])
    rebuilt = build_spec(n=spec["n"], n_val=spec["n_val_source"],
                         seed=spec["seed"])
    assert rebuilt["stems"] == spec["stems"]        # frozen, reproducible


def test_load_fixed20_refuses_a_list_the_data_does_not_match(tmp_path):
    from segmentation.tools import train as seg_train

    class DS:
        stems = [f"ADE_val_{i:08d}" for i in range(1, 5)]

    probe = tmp_path / "probe.json"
    dr.atomic_json_dump({"stems": ["ADE_val_00000001", "ADE_val_00009999"]},
                        probe)
    old = seg_train.FIXED20
    try:
        seg_train.FIXED20 = str(probe)
        with pytest.raises(FileNotFoundError, match="probe images"):
            seg_train.load_fixed20(tmp_path, DS())
        dr.atomic_json_dump({"stems": ["ADE_val_00000001",
                                       "ADE_val_00000001"]}, probe)
        with pytest.raises(ValueError, match="duplicate"):
            seg_train.load_fixed20(tmp_path, DS())
        dr.atomic_json_dump({"stems": ["ADE_val_00000002"]}, probe)
        stems, idx, sha = seg_train.load_fixed20(tmp_path, DS())
        assert stems == ["ADE_val_00000002"] and idx == [1] and len(sha) == 64
    finally:
        seg_train.FIXED20 = old


# ── end-to-end run contracts (CPU, fake data, tiny models) ──────────────────

def _tiny_matrix(task, run_id, ckpt, sha, extra_overrides=None):
    overrides = {
        "model": {"model_kwargs": dict(TINY), "embed_dim": 64,
                  "fpn_out_channels": 8, "fpn_indices": [0, 1, 1, 1]},
        "train": {"num_workers": 0, "batch_size": 2, "epochs": 1},
    }
    overrides = dr.deep_merge(overrides, extra_overrides or {})
    return {
        "task_configs": {"detection": "detection/configs/base.yaml",
                         "segmentation": "segmentation/configs/base.yaml"},
        "out_roots": {"detection": "results/detection",
                      "segmentation": "results/segmentation"},
        "n_register_tokens": 4,
        "chain": {"detection": 4, "segmentation": 2},
        "backbones": {"bb": {"run": "fake", "ckpt": str(ckpt),
                             "sha256": sha}},
        "runs": {run_id: {"task": task, "arch": ARCH, "variant": "saga",
                          "backbone": "bb", "seed": 1,
                          "overrides": overrides}},
    }


@pytest.fixture
def fake_coco_root(tmp_path):
    from PIL import Image
    root = tmp_path / "coco"
    for split in ("train2017", "val2017"):
        (root / split).mkdir(parents=True)
    (root / "annotations").mkdir()
    images, annotations, aid = [], [], 1
    for i in (1, 2, 3, 4):
        for split in ("train2017", "val2017"):
            Image.fromarray(
                np.random.default_rng(i).integers(
                    0, 256, (80, 100, 3), dtype=np.uint8)).save(
                root / split / f"{i:06d}.jpg")
        images.append({"id": i, "file_name": f"{i:06d}.jpg",
                       "height": 80, "width": 100})
        for cat, box in ((1, [10, 10, 30, 30]), (90, [50, 40, 20, 20])):
            annotations.append({"id": aid, "image_id": i, "category_id": cat,
                                "bbox": box, "area": box[2] * box[3],
                                "iscrowd": 0})
            aid += 1
    for split in ("train2017", "val2017"):
        coco_ann_file(root / "annotations" / f"instances_{split}.json",
                      CATS, images, annotations)
    return root


def test_detection_end_to_end_contract(tmp_path, fake_coco_root, monkeypatch):
    pytest.importorskip("pycocotools")
    from detection.tools import train as det_train

    ckpt = tmp_path / "bb.pth"
    save_classifier_ckpt(ckpt, "saga")
    run_id = "det_vitb_saga_s1"
    matrix = _tiny_matrix("detection", run_id, ckpt, file_sha256(ckpt),
                          {"model": {"num_det_classes": 3,
                                     "anchor_sizes": [[8], [16], [32], [64]]},
                           "train": {"min_size": 64, "max_size": 64},
                           "eval": {"eval_freq": 1}})
    out_root = tmp_path / "out"

    status = det_train.run_training(
        matrix, run_id, fake_coco_root, out_root=out_root, resume="none",
        max_steps=1, device_str="cpu")
    assert status == "completed"

    run_dir = out_root / run_id
    for name in ("meta.json", "config.resolved.yaml", "log.csv",
                 "coco_eval_best.json", "detections_val.json"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "ckpt" / "last.pth").exists()
    assert list((run_dir / "eval_shards").glob("*.json")) == []

    # schema pinned LITERALLY, not against the constant that wrote it
    with open(run_dir / "log.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "epoch", "lr_backbone", "lr_head", "train_loss", "AP", "AP50",
            "AP75", "AP_S", "AP_M", "AP_L", "img_per_sec", "wall_time"]
        rows = list(reader)
    assert len(rows) == 1 and rows[0]["epoch"] == "0"
    assert rows[0]["AP"] != ""

    best = json.load(open(run_dir / "coco_eval_best.json"))
    # the six APs the task mandates, plus the ARs, named literally
    for key in ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L",
                "AR_1", "AR_10", "AR_100", "AR_S", "AR_M", "AR_L"):
        assert key in best
        assert best[key] is None or best[key] >= 0.0, (key, best[key])
    for key in ("per_category", "n_val_images", "n_detections", "seed",
                "epoch", "git_sha", "backbone_run", "backbone_ckpt",
                "backbone_source", "backbone_sha256", "val_protocol",
                "detections_file", "detections_sha256"):
        assert key in best, key
    for entry in best["per_category"]:
        assert set(entry) == {"category_id", "name", "AP", "AP50", "AP_S"}
    # the raw dump must be exactly the four prediction fields — pycocotools'
    # loadRes must not have injected a fabricated segmentation polygon
    for d in json.load(open(run_dir / "detections_val.json")):
        assert set(d) == {"image_id", "category_id", "bbox", "score"}
    assert best["run_id"] == run_id and best["seed"] == 1
    assert best["smoke"] is True                    # max_steps was overridden
    assert best["n_val_images"] == 4                # include_empty val split
    assert len(best["per_category"]) == 3
    assert best["backbone_sha256"] == file_sha256(ckpt)
    assert best["detections_sha256"] == file_sha256(
        run_dir / "detections_val.json")
    assert best["normalization_check"]["max_abs_elementwise_diff"] < 1e-4
    dets = json.load(open(run_dir / "detections_val.json"))
    assert isinstance(dets, list) and best["n_detections"] == len(dets)
    for d in dets:
        assert d["category_id"] in (1, 3, 90)       # official ids, not 1-3

    meta = json.load(open(run_dir / "meta.json"))
    assert meta["end_time"] and meta["seed"] == 1
    assert meta["backbone_sha256_observed"] == file_sha256(ckpt)
    assert meta["backbone_source"] == "primary"
    assert meta["normalization_check"]["dist_to_double_normalized"] > 1.0
    assert meta["task"] == "detection"

    # resubmission of a completed run is a no-op (chain safety)
    assert det_train.run_training(
        matrix, run_id, fake_coco_root, out_root=out_root, resume="auto",
        max_steps=1, device_str="cpu") == "skipped"

    # ... and a resume that finds a TORN meta.json rebuilds the provenance
    # record instead of continuing with nothing (or dying and taking the
    # whole afterany chain with it)
    (run_dir / "meta.json").write_text('{"end_time": "x", "se')
    assert det_train.run_training(
        matrix, run_id, fake_coco_root, out_root=out_root, resume="auto",
        max_epochs=2, max_steps=1, device_str="cpu") == "completed"
    meta = json.load(open(run_dir / "meta.json"))
    assert meta["git_sha"] and meta["seed"] == 1 and meta["end_time"]
    assert meta["backbone_sha256_observed"] == file_sha256(ckpt)
    assert len(list(run_dir.glob("meta.json.corrupt.*"))) == 1

    # a resume whose matrix now names a DIFFERENT backbone must refuse: the
    # checkpoint's weights would be kept while the new backbone got the
    # credit in meta.json and coco_eval_best.json
    # same variant (so the strict load succeeds and we actually reach the
    # resume guard), different weights -> different hash
    other = tmp_path / "other_bb.pth"
    save_classifier_ckpt(other, "saga")
    with torch.no_grad():
        sd = torch.load(other, map_location="cpu", weights_only=True)
        sd["model"]["cls_token"] += 0.123
    torch.save(sd, other)
    assert file_sha256(other) != file_sha256(ckpt)
    swapped = json.loads(json.dumps(matrix))     # deep copy
    swapped["backbones"]["bb"] = {"run": "other", "ckpt": str(other),
                                  "sha256": file_sha256(other)}
    with pytest.raises(RuntimeError, match="backbone changed under a resume") \
            as exc:
        det_train.run_training(swapped, run_id, fake_coco_root,
                               out_root=out_root, resume="auto",
                               max_epochs=3, max_steps=1, device_str="cpu")
    # the message must NAME both hashes — an f-string with doubled braces
    # would render literal placeholders and tell the human nothing
    msg = str(exc.value)
    assert file_sha256(ckpt) in msg and file_sha256(other) in msg
    assert "{" not in msg and "}" not in msg


@pytest.fixture
def fake_ade_root(tmp_path):
    from PIL import Image
    root = tmp_path / "ADEChallengeData2016"
    stems = {"training": ["ADE_train_00000001", "ADE_train_00000002"],
             "validation": ["ADE_val_00000001", "ADE_val_00000002"]}
    for split, names in stems.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i, stem in enumerate(names):
            rng = np.random.default_rng(i)
            Image.fromarray(rng.integers(0, 256, (72, 96, 3),
                                         dtype=np.uint8)).save(
                root / "images" / split / f"{stem}.jpg")
            # raw ADE labels: 0 = unlabelled -> 255, k -> k-1
            mask = rng.integers(0, 5, (72, 96), dtype=np.uint8)
            Image.fromarray(mask, mode="L").save(
                root / "annotations" / split / f"{stem}.png")
    with open(root / "objectInfo150.csv", "w", newline="") as f:
        f.write("Idx,Ratio,Train,Val,Name\n")
        for i, name in enumerate(["wall", "building", "sky", "floor"], 1):
            f.write(f"{i},0.1,1,1,{name}\n")
    return root, stems["validation"]


def test_segmentation_end_to_end_contract(tmp_path, fake_ade_root):
    from segmentation.tools import train as seg_train

    ade_root, val_stems = fake_ade_root
    ckpt = tmp_path / "bb.pth"
    save_classifier_ckpt(ckpt, "saga")
    run_id = "seg_vitb_saga_s1"
    matrix = _tiny_matrix("segmentation", run_id, ckpt, file_sha256(ckpt),
                          {"model": {"num_classes": 4},
                           "train": {"input_size": 64,
                                     "random_scale": [1.0, 1.0]},
                           "eval": {"eval_freq": 1, "input_size": 64,
                                    "ms_scales": [1.0, 0.5],
                                    "ms_flip": True}})
    out_root = tmp_path / "out"

    probe = tmp_path / "probe.json"
    dr.atomic_json_dump({"dataset": "ade20k", "stems": val_stems}, probe)

    # plant a stale multi-scale result from a DISCARDED attempt: the MS eval
    # is gated on that file's existence, so a fresh start has to clear it or
    # the run would finalize carrying an mIoU from weights that no longer
    # exist
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True)
    dr.atomic_json_dump({"mIoU": 99.9, "stale": True}, run_dir / "miou_ms.json")

    old = seg_train.FIXED20
    try:
        seg_train.FIXED20 = str(probe)
        status = seg_train.run_training(
            matrix, run_id, ade_root, out_root=out_root, resume="none",
            max_steps=1, device_str="cpu")
        assert status == "completed"
    finally:
        seg_train.FIXED20 = old

    stale_check = json.load(open(run_dir / "miou_ms.json"))
    assert "stale" not in stale_check and stale_check["mIoU"] != 99.9
    for name in ("meta.json", "config.resolved.yaml", "log.csv",
                 "miou_ss.json", "miou_ms.json", "per_class_iou.csv",
                 "conf_matrix.npz"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "ckpt" / "last.pth").exists()
    assert (run_dir / "ckpt" / "best_model.pth").exists()

    with open(run_dir / "log.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["epoch", "lr_backbone", "lr_head",
                                     "train_loss", "mIoU", "img_per_sec",
                                     "wall_time"]
        rows = list(reader)
    assert len(rows) == 1 and rows[0]["mIoU"] != ""

    ss = json.load(open(run_dir / "miou_ss.json"))
    ms = json.load(open(run_dir / "miou_ms.json"))
    assert ss["eval_mode"] == "single_scale" and ss["ms_scales"] is None
    assert ms["eval_mode"] == "multi_scale" and ms["ms_scales"] == [1.0, 0.5]
    assert ms["ms_flip"] is True
    for payload in (ss, ms):
        assert payload["n_images"] == 2
        assert payload["ignore_index"] == 255
        assert payload["num_classes"] == 4
        assert len(payload["iou_per_class"]) == 4
        assert payload["backbone_sha256"] == file_sha256(ckpt)
        assert payload["smoke"] is True
        assert 0.0 <= payload["mIoU"] <= 100.0
        assert payload["class_names_available"] is True

    conf = np.load(run_dir / "conf_matrix.npz")["conf"]
    assert conf.shape == (4, 4)
    # the confusion matrix must agree with the reported mIoU exactly
    summary, *_ = seg_train.summarize_confusion(torch.from_numpy(conf))
    assert summary["mIoU"] == pytest.approx(ss["mIoU"])

    with open(run_dir / "per_class_iou.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["class_index", "name", "iou_frac",
                                     "intersection", "union", "gt_pixels",
                                     "pred_pixels"]
        per_class = list(reader)
    assert [r["class_index"] for r in per_class] == ["0", "1", "2", "3"]
    assert [r["name"] for r in per_class] == ["wall", "building", "sky",
                                              "floor"]
    for row, c in zip(per_class, range(4)):
        assert int(row["intersection"]) == int(conf[c, c])
        assert int(row["gt_pixels"]) == int(conf[c].sum())
        assert int(row["pred_pixels"]) == int(conf[:, c].sum())

    probe_dir = run_dir / "preds_fixed20"
    from PIL import Image
    for stem in val_stems:
        pred = np.asarray(Image.open(probe_dir / f"{stem}_pred.png"))
        gt = np.asarray(Image.open(probe_dir / f"{stem}_gt.png"))
        assert pred.shape == gt.shape == (64, 64)
        assert pred.max() < 4                        # valid class indices
        assert set(np.unique(gt)) <= set(range(4)) | {255}
        assert (probe_dir / f"{stem}_img.jpg").exists()

    meta = json.load(open(run_dir / "meta.json"))
    assert meta["end_time"] and meta["task"] == "segmentation"
    assert meta["probe_list_sha256"] == file_sha256(probe)
    assert "B4" in meta["miou_definition"]


def test_trainers_reject_the_wrong_task(tmp_path, fake_coco_root):
    from detection.tools import train as det_train
    from segmentation.tools import train as seg_train
    ckpt = tmp_path / "bb.pth"
    save_classifier_ckpt(ckpt, "saga")
    matrix = _tiny_matrix("segmentation", "seg_x", ckpt, file_sha256(ckpt))
    with pytest.raises(ValueError, match="segmentation run"):
        det_train.run_training(matrix, "seg_x", fake_coco_root,
                               out_root=tmp_path / "o", device_str="cpu")
    matrix = _tiny_matrix("detection", "det_x", ckpt, file_sha256(ckpt))
    with pytest.raises(ValueError, match="detection run"):
        seg_train.run_training(matrix, "det_x", fake_coco_root,
                               out_root=tmp_path / "o", device_str="cpu")


# ── the Phase-B smoke checker ────────────────────────────────────────────────

def test_check_smoke_reports_missing_and_present_evidence(tmp_path):
    """tools/check_smoke.py decides whether the HPC smokes passed, so it must
    FAIL on absent evidence rather than assume, and every check must print
    what it actually saw."""
    from tools import check_smoke

    rep = check_smoke.Report()
    rep.check("a", True, "saw 1")
    rep.check("b", False, "saw nothing")
    assert [n for _, n, _ in rep.failures()] == ["b"]

    # a log that is missing must be a FAIL, never a silent pass
    rep2 = check_smoke.Report()
    assert check_smoke.read_text(tmp_path / "nope.log") is None
    assert check_smoke.load_json(tmp_path / "nope.json") is None

    # the log checks key off the strings the trainers actually print
    good = ("deps ok: 2.10.0 1.0.28\n77 passed in 30.00s\n"
            "  train2017: 118287 images (expected: 118287)\n"
            "  B5 normalization check PASSED: max_abs_diff=0.00e+00, x\n"
            "  backbone [primary]: /path/last.pth\n"
            "[1/2] loss=1.0  (no eval)  t=1s\n"
            "[2/2] loss=0.9  AP=1.0  AP50=2.0  AP_S=0.5  t=1s\n"
            "  *** new best AP 1.000 -> coco_eval_best.json\n"
            "Done - det_vitb_saga_s1  best AP 1.000 at epoch 1\n")
    check_smoke.check_log(rep2, check_smoke.PIPELINES["detection"], good, "")
    assert not rep2.failures(), [n for _, n, _ in rep2.failures()]

    # and a run that crashed must not be reported as a pass
    rep3 = check_smoke.Report()
    check_smoke.check_log(rep3, check_smoke.PIPELINES["detection"],
                          good.replace("77 passed", "1 failed, 76 passed")
                          + "Traceback (most recent call last):\n", "")
    names = [n for _, n, _ in rep3.failures()]
    assert "unit suite green ON THE COMPUTE NODE" in names
    assert "no python traceback anywhere in stdout/stderr" in names

    # The a0801 fault must be matched by its SIGNATURE, not by the bare word
    # "TaskProlog", which SLURM also prints on healthy nodes — matching the
    # word alone produced a false FAIL on a smoke that COMPLETED 0:0 with
    # every artifact intact.
    rep4 = check_smoke.Report()
    check_smoke.check_log(rep4, check_smoke.PIPELINES["detection"],
                          good + "TaskProlog: /etc/slurm/slurm.taskprolog\n",
                          "")
    assert not rep4.failures(), [n for _, n, _ in rep4.failures()]

    rep5 = check_smoke.Report()
    check_smoke.check_log(
        rep5, check_smoke.PIPELINES["detection"], good,
        "slurm task_prolog can not be executed "
        "(/etc/slurm/slurm.taskprolog) Permission denied\n"
        "TaskProlog failed status=1\n")
    assert "no a0801-style TaskProlog fault" in \
        [n for _, n, _ in rep5.failures()]


def test_ckpt_root_matrix_key_redirects_every_launcher(tmp_path):
    """hpc measured 101G of a 100G soft quota, and ckpt/ defaults to the run
    dir there. Setting `ckpt_root` must redirect the six chains AND both
    smokes (the smokes write checkpoints too — an earlier claim that they
    did not was wrong), and leaving it null must reproduce the committed
    files byte for byte."""
    from scripts import gen_dense_jobs as gen
    matrix = yaml.safe_load(open(MATRIX_PATH))
    assert "ckpt_root" in matrix, "the escape hatch must be in the matrix"
    assert matrix["ckpt_root"] is None, "default must stay the run dir"

    plain = tmp_path / "plain"
    gen.render(matrix, plain, DENSE_BASE_PORT)
    for f in sorted(plain.rglob("*.sbatch")):
        assert "--ckpt_root" not in f.read_text(), f

    redirected = tmp_path / "redirected"
    gen.render(dict(matrix, ckpt_root="/bulk/dense_ckpt"), redirected,
               DENSE_BASE_PORT)
    sbatch = sorted(redirected.rglob("*.sbatch"))
    assert len(sbatch) == 8, [f.name for f in sbatch]
    for f in sbatch:
        text = f.read_text()
        assert "--ckpt_root /bulk/dense_ckpt" in text, f.name
        # must be a continued argument, not a stray line
        assert "--ckpt_root /bulk/dense_ckpt \\" in text, f.name


def test_runtime_projection_flags_an_undersized_chain(tmp_path):
    """The smoke's throughput is the only dense measurement that exists, and
    it decides whether 25 epochs fit in 4x24 h. A projection that overruns
    the budget must FAIL and say what to raise the chain to."""
    from tools import check_smoke

    def project(ips):
        run = tmp_path / f"r{ips}"
        run.mkdir()
        dr.append_log_row(run / "log.csv",
                          ["epoch", "img_per_sec", "wall_time"],
                          {"epoch": 0, "img_per_sec": ips, "wall_time": 100})
        rep = check_smoke.Report()
        check_smoke.project_runtime(rep, check_smoke.PIPELINES["detection"],
                                    run, MATRIX_PATH)
        return rep.rows[0]

    slow_ok, _, slow_ev = project(2.0)
    fast_ok, _, fast_ev = project(40.0)
    assert not slow_ok and "RAISE chain.detection" in slow_ev
    assert fast_ok and "RAISE" not in fast_ev
    # the projection must be built from the COMMITTED chain length
    assert "budget 96 h" in slow_ev, slow_ev

    # a missing log is a FAIL, never an assumption
    empty = tmp_path / "empty"
    empty.mkdir()
    rep = check_smoke.Report()
    check_smoke.project_runtime(rep, check_smoke.PIPELINES["detection"],
                                empty, MATRIX_PATH)
    assert not rep.rows[0][0]


def test_class_names_parse_the_REAL_objectInfo150_txt_layout(tmp_path):
    """The archive ships the names as objectInfo150.TXT (confirmed on the
    HPC 2026-09-08: ADEChallengeData2016/objectInfo150.txt, 5689 B — there is
    no .csv). Pin the real layout: tab-separated, `Idx Ratio Train Val Name`,
    names containing commas, 150 rows in Idx order."""
    from segmentation.tools.train import load_class_names, write_per_class_csv

    # the first four official ADE20K classes, in their real Idx order
    real = ["wall", "building, edifice", "sky",
            "floor, flooring"]
    root = tmp_path / "ADEChallengeData2016"
    root.mkdir()
    with open(root / "objectInfo150.txt", "w", encoding="utf-8") as f:
        f.write("Idx\tRatio\tTrain\tVal\tName\n")
        for i, n in enumerate(real, 1):
            f.write(f"{i}\t0.1576\t11664\t1172\t{n}\n")

    names, source = load_class_names(root, len(real))
    assert names == real, names
    assert "objectInfo150.txt" in source and "tab" in source
    assert "Idx-verified" in source

    # ... and those names survive into the CSV Phase C reads, un-mangled by
    # the commas inside them
    iou = np.array([0.5, 0.25, 0.75, 0.125])
    counts = np.arange(1, len(real) + 1)
    path = tmp_path / "per_class_iou.csv"
    write_per_class_csv(path, iou, counts, counts, counts, counts, names)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["name"] for r in rows] == real
    by_name = {r["name"].split(",")[0]: int(r["class_index"]) for r in rows}
    assert by_name["wall"] == 0 and by_name["sky"] == 2
    assert by_name["floor"] == 3          # the rows Phase C must surface
