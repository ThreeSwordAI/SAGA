"""
tests/test_I6_external.py
=========================
TASK A / I6 A11 — CPU, tiny fake models, fake data, no network.

What these tests are actually for, in order of how much damage they prevent:

  1. The PREDICTION must be the one in the task file, character for
     character. I6's whole claim is that it was registered before any
     external weight was fetched; a prediction that could be edited
     afterwards to match an outcome is worth nothing.
  2. The download tool must not be able to run inference. Checked on the
     parsed AST, not by trusting the docstring: it must not import timm and
     must not build a model. If it could evaluate, the first look at the
     outcome could precede the commit that registers the claim.
  3. Prefix count and grid side must be read from the MODEL and cross-checked
     against the registry. A register model read with a hard-coded prefix of
     1 puts four register tokens into the patch grid and every ring number
     after that is wrong — quietly.
  4. An unresolvable timm name must be recorded UNAVAILABLE, never
     substituted (TASK A §12).
  5. The outcome must be computed from the registered rule, for every
     group and both branches of each.

Two fake shapes are exercised: a 14x14 grid with 1 prefix token and a 16x16
grid with 5, which is the register variant's geometry.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from analysis.address_analysis import border_distance_map
from saga.frozen import external as ext
from saga.frozen import norms as fnorms
from saga.frozen.stages import ALL_STAGES, EXTERNAL_STAGES, resolve_stage

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "configs" / "frozen" / "I6_models.yaml"
CONDITIONS = REPO / "configs" / "frozen" / "I6_external.yaml"
JOB_FILE = REPO / "scripts" / "jobs" / "frozen_I6.sbatch"
DOWNLOAD_TOOL = REPO / "tools" / "frozen_I6_download.py"
TASK_FILE = REPO / "docs" / "TASK_A_I1_I6.md"

MISSING = "MISSING"


@pytest.fixture(scope="module")
def registry():
    return ext.load_registry(REGISTRY)


# ─────────────────────────────────────────────────────────────────────────────
# 1. The registry and the pre-registered prediction
# ─────────────────────────────────────────────────────────────────────────────

def test_the_registry_parses_and_holds_nine_models(registry):
    assert registry["work_package"] == "I6_external"
    assert len(registry["models"]) == 9
    assert {m["group"] for m in registry["models"]} == set(ext.GROUPS)
    counts = {}
    for m in registry["models"]:
        counts[m["group"]] = counts.get(m["group"], 0) + 1
    assert counts == {"supervised_mixing": 4, "no_mixing": 4,
                      "registers_exploratory": 1}


def test_the_prediction_is_the_task_files_prediction_character_for_character(
        registry):
    """Copied VERBATIM from docs/TASK_A_I1_I6.md §9 — including the
    non-breaking hyphens. If this ever fails, the registry was edited."""
    line = [l for l in TASK_FILE.read_text(encoding="utf-8").splitlines()
            if l.startswith("> Supervised")]
    assert len(line) == 1, "the task file's §9 prediction block moved"
    assert registry["prediction"] == line[0][2:]
    assert "‑" in registry["prediction"], "the non-breaking hyphens are " \
                                               "part of the registered string"


def test_the_registry_declares_how_the_prediction_is_scored(registry):
    """The rule lives beside the claim, so neither can drift from the other."""
    assert set(registry["outcome_rule"]) == set(ext.GROUPS)
    assert registry["registered_at"]
    assert registry["prediction_source"].startswith("docs/TASK_A_I1_I6.md")


def test_every_model_declares_a_recipe_flag_a_group_and_a_citation(registry):
    for m in registry["models"]:
        assert str(m["mixing"]) in ("yes", "no"), \
            f"{m['model_id']}: bare yes/no is a YAML boolean"
        assert m["group"] in ext.GROUPS
        assert len(m["citation"].strip()) > 40, \
            f"{m['model_id']}: the recipe flag needs a paper behind it"
        assert str(m["weight_sha256"]) == "PENDING" or \
            len(str(m["weight_sha256"])) == 64


def test_the_grid_side_follows_from_the_patch_size_at_224(registry):
    """16 for patch-14, 14 for patch-16 — and the registry says so itself."""
    for m in registry["models"]:
        assert m["grid_side"] * m["patch_size"] == registry["input_size"]
    sides = {m["model_id"]: m["grid_side"] for m in registry["models"]}
    assert sides["deit_small_patch16_224"] == 14
    assert sides["vit_small_patch14_dinov2"] == 16
    assert sides["vit_small_patch14_reg4_dinov2"] == 16


@pytest.mark.parametrize("bad,match", [
    ({"prediction": "too short"}, "pre-registered claim"),
    ({"models": []}, "missing or empty"),
    ({"hf_home": ""}, "missing or empty"),
])
def test_a_registry_without_its_prediction_is_refused(tmp_path, registry, bad,
                                                      match):
    import yaml
    doc = dict(registry)
    doc.update(bad)
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ext.ExternalError, match=match):
        ext.load_registry(path)


def test_a_model_in_an_undeclared_group_is_refused(tmp_path, registry):
    import yaml
    doc = json.loads(json.dumps(registry))
    doc["models"][0]["group"] = "supervised_no_mixing_probably"
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ext.ExternalError, match="declares group"):
        ext.load_registry(path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. timm resolution — recorded, never substituted
# ─────────────────────────────────────────────────────────────────────────────

def test_every_registered_name_resolves_or_is_reported_unavailable(registry):
    """Not an assertion that all nine resolve — that depends on the installed
    timm. An unresolvable name must be REPORTED with the version, and the
    report is what the log and the handoff carry."""
    status = ext.availability(registry)
    assert set(status) == {m["model_id"] for m in registry["models"]}
    for model_id, verdict in status.items():
        assert verdict == "available" or verdict.startswith(ext.UNAVAILABLE)
        if verdict != "available":
            assert ext.timm_version() in verdict


def test_an_unresolvable_name_is_refused_and_never_substituted(registry,
                                                               tmp_path):
    import yaml
    doc = json.loads(json.dumps(registry))
    doc["models"][0]["timm_name"] = "vit_nonexistent_patch16_224.no_such_tag"
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    bad = ext.load_registry(path)
    model_id = bad["models"][0]["model_id"]
    assert ext.availability(bad)[model_id].startswith(ext.UNAVAILABLE)
    with pytest.raises(ext.ExternalError, match="does not resolve in timm"):
        ext.build_external(model_id, bad, pretrained=False)


def test_an_architecture_without_the_pretrained_tag_does_not_resolve():
    """A name whose ARCH exists but whose TAG does not would otherwise build
    random weights and measure noise."""
    assert ext.resolves_in_timm("vit_base_patch16_224.augreg_in1k")
    assert not ext.resolves_in_timm("vit_base_patch16_224.no_such_tag")


def test_a_model_not_in_the_registry_is_refused(registry):
    with pytest.raises(ext.ExternalError, match="not in the registry"):
        ext.model_entry(registry, "resnet50")


# ─────────────────────────────────────────────────────────────────────────────
# 3. The adapter's geometry, on fake models
# ─────────────────────────────────────────────────────────────────────────────

def fake_model(*, img_size, patch_size, reg_tokens=0, depth=12):
    torch.manual_seed(0)
    return timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=10, depth=depth,
        embed_dim=24, num_heads=3, img_size=img_size, patch_size=patch_size,
        reg_tokens=reg_tokens).eval()


@pytest.mark.parametrize("patch,reg,side,prefix", [
    (16, 0, 14, 1),     # DeiT / AugReg / CLIP / MAE geometry
    (14, 0, 16, 1),     # DINOv2 S/B at 224
    (14, 4, 16, 5),     # DINOv2 reg4 — FIVE prefix tokens
])
def test_geometry_is_read_from_the_model(patch, reg, side, prefix):
    geom = ext.model_geometry(fake_model(img_size=224, patch_size=patch,
                                         reg_tokens=reg))
    assert geom["grid"] == (side, side)
    assert geom["grid_side"] == side
    assert geom["n_patches"] == side * side
    assert geom["n_prefix"] == prefix, \
        "the prefix count must come from the model, never from a constant"
    assert geom["depth"] == 12


def entry(**kw):
    base = {"model_id": "fake", "grid_side": 14, "n_prefix": 1, "depth": 12}
    base.update(kw)
    return base


def test_a_registry_that_disagrees_with_the_model_fails_loudly():
    geom = ext.model_geometry(fake_model(img_size=224, patch_size=14,
                                         reg_tokens=4))
    ext.check_geometry(geom, entry(grid_side=16, n_prefix=5))    # correct

    with pytest.raises(ext.ExternalError, match="declares n_prefix=1"):
        ext.check_geometry(geom, entry(grid_side=16, n_prefix=1))
    with pytest.raises(ext.ExternalError, match="declares grid_side=14"):
        ext.check_geometry(geom, entry(grid_side=14, n_prefix=5))
    with pytest.raises(ext.ExternalError, match="declares depth=6"):
        ext.check_geometry(geom, entry(grid_side=16, n_prefix=5, depth=6))


def test_a_non_square_token_grid_is_refused():
    geom = {"grid": (14, 16), "grid_side": 14, "n_patches": 224, "depth": 12,
            "n_prefix": 1}
    with pytest.raises(ext.ExternalError, match="not\n?\\s*square|not square"):
        ext.check_geometry(geom, entry(grid_side=14))


def test_a_model_without_a_patch_grid_is_refused():
    class Bare(torch.nn.Module):
        blocks = torch.nn.ModuleList([torch.nn.Identity()])
    with pytest.raises(ext.ExternalError, match="patch_embed.grid_size"):
        ext.model_geometry(Bare())


# ─────────────────────────────────────────────────────────────────────────────
# 4. The analog stages
# ─────────────────────────────────────────────────────────────────────────────

def test_the_analog_stages_are_the_two_block_outputs_under_their_own_names():
    assert ext.EXT_STAGES == ("ext_s11_out", "ext_hist")
    assert resolve_stage("ext_s11_out") == "s11_out"
    assert resolve_stage("ext_hist") == "s12_pre_norm"
    assert set(EXTERNAL_STAGES) <= set(ALL_STAGES)
    assert ext.ext_stage_alias("ext_hist") == "s12_pre_norm"
    with pytest.raises(ext.ExternalError, match="not an external analog"):
        ext.ext_stage_alias("s11_out")


@pytest.mark.parametrize("patch,reg,n_patches", [(16, 0, 196), (14, 4, 256)])
def test_the_analog_stages_capture_patch_rows_only(patch, reg, n_patches):
    """Prefix rows removed with the MODEL's own count, on both geometries."""
    from saga.frozen.stages import capture_stages

    model = fake_model(img_size=224, patch_size=patch, reg_tokens=reg)
    x = torch.randn(2, 3, 224, 224)
    with capture_stages(model, ext.EXT_STAGES) as store:
        model(x)
        for stage in ext.EXT_STAGES:
            assert store[stage].shape[1] == n_patches

    with torch.no_grad():
        t = model.patch_embed(x)
        t = model._pos_embed(t)
        t = model.norm_pre(t)
        outs = []
        for blk in model.blocks:
            t = blk(t)
            outs.append(t)
    n_prefix = ext.model_geometry(model)["n_prefix"]
    assert torch.equal(store["ext_s11_out"], outs[-2][:, n_prefix:, :])
    assert torch.equal(store["ext_hist"], outs[-1][:, n_prefix:, :])


def test_the_conditions_file_declares_one_native_condition():
    from saga.frozen.runner import load_conditions
    doc = load_conditions(CONDITIONS)
    assert doc["work_package"] == "I6_external"
    assert doc["stage"] == "ext_s11_out"
    assert [c["id"] for c in doc["conditions"]] == ["native"]
    assert doc["conditions"][0]["stages"] == list(ext.EXT_STAGES)
    assert doc["thresholds_canon"] == MISSING, \
        "tau_canon is our cohort's value and has no meaning for DeiT or CLIP"


# ─────────────────────────────────────────────────────────────────────────────
# 5. The download tool cannot run inference
# ─────────────────────────────────────────────────────────────────────────────

def test_the_download_tool_cannot_run_a_forward_pass():
    """On the parsed AST. The tool downloads FILES; it never builds a model,
    so there is nothing to call. If it could evaluate, the first look at an
    outcome could precede the commit that registers the prediction."""
    tree = ast.parse(DOWNLOAD_TOOL.read_text(encoding="utf-8"),
                     filename=str(DOWNLOAD_TOOL))
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Attribute):
            called.add(node.attr)
    assert "timm" not in imported, "the download tool must not import timm"
    assert "torch" not in imported, "the download tool must not import torch"
    for banned in ("create_model", "forward", "eval", "predict",
                   "build_external", "no_grad", "backward"):
        assert banned not in called, \
            f"the download tool reaches .{banned}() — it must only fetch files"


def test_the_download_tool_writes_a_sha_without_touching_anything_else(
        tmp_path):
    """The registry is mostly prose — the prediction, the outcome rule, every
    citation. A PyYAML round trip would delete all of it, so the write is
    line-targeted and this test proves nothing else moved."""
    from tools.frozen_I6_download import set_sha_in_registry

    original = REGISTRY.read_text(encoding="utf-8")
    sha = "b" * 64
    updated = set_sha_in_registry(original, "vit_small_patch14_reg4_dinov2",
                                  sha)

    before = original.splitlines()
    after = updated.splitlines()
    assert len(before) == len(after)
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(changed) == 1, f"{len(changed)} lines moved, expected 1"
    assert changed[0][1].strip() == f"weight_sha256: {sha}"
    assert "Supervised ImageNet" in updated, "the prediction must survive"
    assert updated.count("#") == original.count("#"), "comments were lost"

    # and it lands on the RIGHT model
    import yaml
    doc = yaml.safe_load(updated)
    by_id = {m["model_id"]: m for m in doc["models"]}
    assert by_id["vit_small_patch14_reg4_dinov2"]["weight_sha256"] == sha
    assert by_id["deit_small_patch16_224"]["weight_sha256"] == "PENDING"


def test_writing_a_sha_for_an_unknown_model_is_refused():
    from tools.frozen_I6_download import set_sha_in_registry
    with pytest.raises(ext.ExternalError, match="no `model_id"):
        set_sha_in_registry(REGISTRY.read_text(encoding="utf-8"),
                            "not_a_model", "c" * 64)


def test_weights_are_verified_against_the_registry_before_they_are_loaded():
    with pytest.raises(ext.ExternalError, match="Run tools/frozen_I6_download"):
        ext.verify_weight_sha({"model_id": "fake", "hub_id": "timm/whatever",
                               "weight_sha256": "PENDING"})


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scoring the prediction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group,peak,excess,expected", [
    ("supervised_mixing", True, 0.3, "met"),
    ("supervised_mixing", False, 0.3, "not met"),
    ("supervised_mixing", True, -0.1, "not met"),
    ("supervised_mixing", True, float("nan"), MISSING),
    ("no_mixing", False, 0.3, "met"),
    ("no_mixing", True, 0.3, "not met"),
    ("no_mixing", True, -0.1, "not met"),
    ("registers_exploratory", True, 0.3, "not applicable"),
    ("registers_exploratory", False, -0.1, "not applicable"),
])
def test_the_outcome_follows_the_registered_rule(group, peak, excess,
                                                 expected):
    assert ext.prediction_outcome(group, has_ring_1_peak=peak,
                                  gini_excess=excess) == expected


def test_an_unknown_group_cannot_be_scored():
    with pytest.raises(ext.ExternalError, match="unknown prediction group"):
        ext.prediction_outcome("something_else", has_ring_1_peak=True,
                               gini_excess=1.0)


@pytest.mark.parametrize("side", [14, 16])
def test_ring_1_peak_reads_the_profile(side):
    rings = border_distance_map(side).reshape(-1)
    peaked = np.array([0.02 + 0.10 * (r == 1) for r in rings])
    flat = np.full(side * side, 0.05)
    from saga.frozen.prevalence import ring_profile
    assert ext.ring_1_peak(ring_profile(peaked))
    assert not ext.ring_1_peak(ring_profile(flat))


# ─────────────────────────────────────────────────────────────────────────────
# 7. The table
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_i6_results(tmp_path, registry):
    """Maps for three registered models, one per group, in the layout
    `analysis/frozen_I6_external.py` reads — including a 16x16 grid."""
    root = tmp_path / "I6_external"
    rng = np.random.RandomState(0)
    spec = {"deit_small_patch16_224": (14, True),
            "vit_base_patch16_224_mae": (14, False),
            "vit_small_patch14_reg4_dinov2": (16, True)}
    n_images, tau = 120, 1.0
    for model_id, (side, peaked) in spec.items():
        n = side * side
        rings = border_distance_map(side).reshape(-1)
        p = np.full(n, 0.02) + (0.10 * (rings == 1) if peaked
                                else np.zeros(n))
        for stage in ext.EXT_STAGES:
            # Drive the REAL accumulator: tokens whose L2 norm is 2.0 exceed
            # tau=1.0 and tokens at 0.5 do not, so the indicator matrix is
            # exactly `want` on the fixed basis — and on the MAD basis too,
            # since the median norm is 0.5 with a MAD of 0.
            acc = fnorms.StageAccumulator("native", stage, tau_cal=tau)
            want = rng.uniform(size=(n_images, n)) < p[None, :]
            norms = np.where(want, 2.0, 0.5).astype(np.float32)
            patches = torch.zeros(n_images, n, 3)
            patches[:, :, 0] = torch.from_numpy(norms)
            acc.add(patches, [f"img{i:04d}" for i in range(n_images)])
            assert np.array_equal(acc.indicators("fixed_cal"), want)
            fnorms.write_stage_maps_npz(root / model_id, [acc], {
                "run_id": model_id, "model_id": model_id,
                "ckpt_sha256": "a" * 64, "weight_sha256": "a" * 64,
                "git_sha": "c" * 40, "split_name": "evaluation",
                "split_sha256": "d" * 64,
                "threshold_split_name": "calibration",
                "threshold_split_sha256": "e" * 64,
                "tau_cal": {stage: 1.0}, "model_n_prefix": 5 if side == 16 else 1,
                "model_grid_side": side, "model_depth": 12,
                "model_timm_version": "1.0.0", "model_input_size": 224,
                "model_native_input_size": 518 if side == 16 else 224,
                "model_resolution_deviation": side == 16,
                "model_patch_size": 14 if side == 16 else 16,
                "model_norm_mean": [0.5, 0.5, 0.5],
                "model_norm_std": [0.5, 0.5, 0.5], "model_crop_pct": 0.9})
    return root


def test_the_i6_table_is_byte_identical_on_a_rebuild(fake_i6_results,
                                                     tmp_path):
    from analysis.frozen_I6_external import build

    out = {}
    for i in (1, 2):
        d = tmp_path / f"t{i}"
        build(REGISTRY, root=fake_i6_results, out_dir=d,
              git_sha_value="c" * 40, resamples=200)
        out[i] = (d / "T_I6a_external.csv").read_bytes()
    assert out[1] == out[2]


def test_the_table_scores_each_group_by_the_registered_rule(fake_i6_results,
                                                            tmp_path):
    import csv

    from analysis.frozen_I6_external import build

    d = tmp_path / "t"
    result = build(REGISTRY, root=fake_i6_results, out_dir=d,
                   git_sha_value="c" * 40, resamples=200)
    rows = list(csv.DictReader(open(result["table"], encoding="utf-8")))
    assert rows, "no rows built"
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r)

    # a ring-1-peaked supervised_mixing model: prediction MET
    for r in by_model["deit_small_patch16_224"]:
        assert r["ring_1_peak"] == "1"
        assert r["prediction_outcome"] == "met"
        assert float(r["gini_excess"]) > 0
    # a flat no_mixing model: no ring-1 peak, prediction MET
    for r in by_model["vit_base_patch16_224_mae"]:
        assert r["ring_1_peak"] == "0"
        assert r["prediction_outcome"] == "met"
    # the register variant makes no prediction, whatever it shows
    for r in by_model["vit_small_patch14_reg4_dinov2"]:
        assert r["prediction_outcome"] == "not applicable"
        assert r["grid_side"] == "16"
        assert r["ring7_freq"] != MISSING, "a 16x16 grid has 8 rings"
        assert r["n_prefix"] == "5"

    # the six models with no maps are ABSENT, never rows of zeros
    assert len(result["missing"]) == 6
    assert set(by_model) == {"deit_small_patch16_224",
                             "vit_base_patch16_224_mae",
                             "vit_small_patch14_reg4_dinov2"}


def test_the_table_carries_the_provenance_and_the_resolution_deviation(
        fake_i6_results, tmp_path):
    import csv

    from analysis.frozen_I6_external import build

    d = tmp_path / "t"
    result = build(REGISTRY, root=fake_i6_results, out_dir=d,
                   git_sha_value="c" * 40, resamples=200)
    rows = list(csv.DictReader(open(result["table"], encoding="utf-8")))
    for r in rows:
        for col in ("weight_sha256", "split_sha256", "threshold_split_sha256",
                    "timm_version", "git_sha", "stage_resolved"):
            assert r[col] not in ("", MISSING), f"{col} is empty"
        assert r["split_sha256"] != r["threshold_split_sha256"], \
            "I6 calibrates on one split and reports on another, by design"
    dev = {r["model_id"]: r["resolution_deviation"] for r in rows}
    assert dev["vit_small_patch14_reg4_dinov2"] == "1", \
        "DINOv2 is natively 518 and is run at 224 — that must be recorded"
    assert dev["deit_small_patch16_224"] == "0"


def test_nothing_in_the_table_is_pooled_across_models_or_groups(
        fake_i6_results, tmp_path):
    import csv

    from analysis.frozen_I6_external import build

    d = tmp_path / "t"
    result = build(REGISTRY, root=fake_i6_results, out_dir=d,
                   git_sha_value="c" * 40, resamples=200)
    rows = list(csv.DictReader(open(result["table"], encoding="utf-8")))
    ids = {r["model_id"] for r in rows}
    assert "MEAN" not in ids and "ALL" not in ids
    # exactly one row per (model, stage, basis)
    keys = [(r["model_id"], r["stage"], r["basis"]) for r in rows]
    assert len(keys) == len(set(keys)) == 3 * 2 * 2


# ─────────────────────────────────────────────────────────────────────────────
# 8. The job file
# ─────────────────────────────────────────────────────────────────────────────

def test_the_job_file_lists_the_registry_in_array_order(registry):
    text = JOB_FILE.read_text(encoding="utf-8")
    block = text.split("MODEL_IDS=(")[1].split(")")[0]
    listed = [line.split('"')[1] for line in block.splitlines()
              if line.strip().startswith('"')]
    assert listed == [m["model_id"] for m in registry["models"]]
    assert "--array=0-8" in text


def test_the_job_file_keeps_the_hub_cache_off_the_home_quota():
    text = JOB_FILE.read_text(encoding="utf-8")
    assert "/home/woody/" in text, "the hub cache belongs on woody"
    assert "HF_HOME" in text and "TORCH_HOME" in text
    assert "HF_HUB_OFFLINE=1" in text, "compute nodes have no network"


def test_the_job_file_refuses_to_run_against_a_pending_sha():
    text = JOB_FILE.read_text(encoding="utf-8")
    assert "weight_sha256" in text
    assert "frozen_I6_download.py" in text


def test_set_cache_env_points_at_the_registry_path(registry, tmp_path,
                                                   monkeypatch):
    monkeypatch.setenv("SAGA_HF_HOME", str(tmp_path / "cache"))
    root = ext.set_cache_env(registry)
    assert Path(root) == tmp_path / "cache"
    import os
    assert os.environ["HF_HOME"] == root
    assert os.environ["TORCH_HOME"] == root
