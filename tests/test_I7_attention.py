"""
tests/test_I7_attention.py
==========================
TASK C / I7 §8 Phase A' — attention capture, the wording rule, and the TTR
operating curve, on CPU with tiny fake models and fake data.

The models are `vit_tiny_patch16_224` at DEPTH 12, not the 4-block fakes the
I0/I5 tests use: I7 names blocks 10 and 11 by ABSOLUTE index, and a shallower
fake would not have them. Three shapes, as everywhere in this project:
a SAGA ViT (1 prefix), a plain ViT (1 prefix), and a timm 4-register model
(5 prefix), because the register model is the one where a hard-coded prefix
count would silently shift every key position.
"""

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from saga.frozen import attention as ATT
from saga.frozen.edits import state_hash
from saga.frozen.runner import load_conditions
from saga.frozen.stages import (ALL_STAGES, BLOCK_INPUT_STAGES,
                                capture_stages, stage_block_index)
from saga.metrics import infer_num_prefix_tokens, sink_counts_mad, token_norms
from saga.vit import build_saga_vit

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = REPO / "configs" / "frozen" / "I7_attention.yaml"

DEPTH = 12
DIM = 24
HEADS = 3
N_PATCHES = 196


def _structured_gate(model, seed=0):
    """A spatially structured phi — no training, no optimizer, just a fixed
    pattern assigned under `torch.no_grad()` (the I0/I5 convention)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for blk in model.blocks:
            phi = blk.attn.gate.phi
            phi.copy_(torch.randn(phi.shape, generator=g) * 1.5)
    return model


@pytest.fixture(scope="module")
def saga_model():
    torch.manual_seed(0)
    return _structured_gate(build_saga_vit(
        "vit_tiny_patch16_224", gate=True, num_classes=10, depth=DEPTH,
        embed_dim=DIM, num_heads=HEADS).eval())


@pytest.fixture(scope="module")
def baseline_model():
    torch.manual_seed(0)
    return build_saga_vit("vit_tiny_patch16_224", gate=False, num_classes=10,
                          depth=DEPTH, embed_dim=DIM, num_heads=HEADS).eval()


@pytest.fixture(scope="module")
def reg4_model():
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=10, reg_tokens=4, depth=DEPTH,
                             embed_dim=DIM, num_heads=HEADS).eval()


@pytest.fixture(scope="module")
def images():
    torch.manual_seed(1)
    return torch.randn(3, 3, 224, 224)


# ── the new stage, and the alignment it exists for ───────────────────────────

def test_in_b10_is_the_output_of_block_nine(saga_model, images):
    """§5: attention inside blocks[10] is computed from that block's INPUT.

    Compared TENSOR against TENSOR with a manual hook on blocks[9], so the
    convention cannot drift from the docstring. This is the off-by-one that
    would place the effect one block after its cause, silently.
    """
    captured = {}
    handle = saga_model.blocks[9].register_forward_hook(
        lambda m, i, out: captured.__setitem__(
            "out", (out[0] if isinstance(out, tuple) else out).detach()))
    try:
        with torch.no_grad(), capture_stages(saga_model, ("in_b10",)) as store:
            saga_model(images)
            got = store["in_b10"].clone()
    finally:
        handle.remove()

    p = infer_num_prefix_tokens(saga_model)
    assert torch.equal(got, captured["out"][:, p:, :].float())
    assert stage_block_index(saga_model, "in_b10") == 9


def test_in_b10_was_added_additively():
    """The I0 four and TASK A's two are untouched; `in_b10` is a new key."""
    assert BLOCK_INPUT_STAGES["in_b10"] == 10
    for name in ("in_b07", "in_b08"):
        assert name in BLOCK_INPUT_STAGES
    for name in ("s11_out", "s12_pre_norm", "s12_post_norm", "hist"):
        assert name in ALL_STAGES
    assert "in_b10" in ALL_STAGES


def test_the_declared_alignment_is_input_not_output():
    """blocks[10] <-> in_b10, blocks[11] <-> s11_out, and nothing else."""
    assert ATT.BLOCK_ALIGNMENT == {10: "in_b10", 11: "s11_out"}


# ── the capture ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture_name,expect_prefix",
                         [("saga_model", 1), ("baseline_model", 1),
                          ("reg4_model", 5)])
def test_post_softmax_rows_sum_to_one(request, fixture_name, expect_prefix,
                                      images):
    """The captured map is a POST-SOFTMAX attention matrix — every query row
    is a distribution over keys. If this fails, every mass number is wrong."""
    from saga.attn_extract import capture_attention

    model = request.getfixturevalue(fixture_name)
    assert infer_num_prefix_tokens(model) == expect_prefix
    with torch.no_grad(), capture_attention(model, blocks=[10, 11]) as maps:
        model(images)
        for b in (10, 11):
            sums = maps[b].sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.parametrize("fixture_name,expect_prefix,expect_tokens",
                         [("saga_model", 1, 197), ("baseline_model", 1, 197),
                          ("reg4_model", 5, 201)])
def test_summary_shapes_follow_the_models_own_prefix_count(
        request, fixture_name, expect_prefix, expect_tokens, images):
    """The key axis is the FULL token axis (prefix included, because the mass
    landing on a register token is one of the things §5 asks for); the
    exceedance axis is the PATCH axis. 197 vs 201 is the whole point."""
    model = request.getfixturevalue(fixture_name)
    n_prefix = infer_num_prefix_tokens(model)
    assert n_prefix == expect_prefix

    out = ATT.attention_summaries(model, images, [10, 11], n_prefix=n_prefix,
                                  tau_cal=1.0)
    for b in (10, 11):
        for q in ATT.QUANTITIES:
            assert out[b][q].shape == (images.shape[0], expect_tokens), \
                f"{q} at block {b}"
        assert out[b]["exceedance_mad"].shape == (images.shape[0], N_PATCHES)
        assert out[b]["exceedance_tau_cal"].shape == (images.shape[0],
                                                      N_PATCHES)
    assert out[10]["stage"] == "in_b10"
    assert out[11]["stage"] == "s11_out"


def test_incoming_mass_is_a_mean_over_queries_and_sums_to_one_over_keys(
        saga_model, images):
    """`in_mass_patchq` averages a distribution over queries, so it is itself
    a distribution over keys: it sums to 1. That pins it as a MEAN and not a
    sum, which is what makes it comparable across token counts."""
    out = ATT.attention_summaries(saga_model, images, [10, 11], n_prefix=1)
    for b in (10, 11):
        for q in ("in_mass_patchq", "in_mass_clsq"):
            total = out[b][q].sum(dim=-1)
            assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
                f"{q} at block {b} does not sum to 1 over keys"


def test_summaries_against_hand_computed_values(saga_model, images):
    """The reduction, checked against the map it came from.

    `in_mass_patchq[k]` must equal the mean over heads AND patch queries of
    A[..., q, k]; `in_mass_clsq[k]` the mean over heads of the CLS row. Done
    by capturing the map separately and reducing it by hand.
    """
    from saga.attn_extract import capture_attention

    with torch.no_grad(), capture_attention(saga_model, blocks=[11]) as maps:
        saga_model(images)
        a = maps[11].float().clone()

    out = ATT.attention_summaries(saga_model, images, [11], n_prefix=1)
    want_patch = a[:, :, 1:, :].mean(dim=(1, 2))
    want_cls = a[:, :, 0, :].mean(dim=1)
    assert torch.allclose(out[11]["in_mass_patchq"], want_patch, atol=1e-6)
    assert torch.allclose(out[11]["in_mass_clsq"], want_cls, atol=1e-6)


def test_value_norm_matches_the_modules_own_projection(saga_model, images):
    """`value_norm` must come from the SAME qkv unbind order the attention
    does; a different order would be a different quantity with this name."""
    captured = {}
    mod = saga_model.blocks[11].attn
    handle = mod.register_forward_pre_hook(
        lambda m, args: captured.__setitem__("x", args[0].detach()))
    try:
        with torch.no_grad():
            saga_model(images)
    finally:
        handle.remove()

    x = captured["x"]
    got = ATT.value_norms(mod, x)
    B, N, C = x.shape
    H = mod.num_heads
    head_dim = getattr(mod, "head_dim", C // H)
    qkv = mod.qkv(x).reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
    want = qkv.unbind(0)[2].float().norm(dim=-1).mean(dim=1)
    assert torch.allclose(got, want, atol=1e-6)
    assert got.shape == (B, N)


def test_exceedance_flags_equal_saga_metrics(saga_model, images):
    """The flags are the metric the paper reports, not a lookalike."""
    out = ATT.attention_summaries(saga_model, images, [11], n_prefix=1)
    with torch.no_grad(), capture_stages(saga_model, ("s11_out",)) as store:
        saga_model(images)
        patches = store["s11_out"].clone()
    counts = sink_counts_mad(token_norms(patches), k=5.0)
    assert torch.equal(out[11]["exceedance_mad"].sum(dim=1).float(), counts)


def test_tau_cal_flags_are_an_absolute_threshold(saga_model, images):
    """The SECONDARY basis is I2's absolute tau, applied as an absolute
    threshold on the patch norms — not a second MAD rule."""
    with torch.no_grad(), capture_stages(saga_model, ("s11_out",)) as store:
        saga_model(images)
        norms = token_norms(store["s11_out"].clone())
    tau = float(norms.median())
    out = ATT.attention_summaries(saga_model, images, [11], n_prefix=1,
                                  tau_cal=tau)
    assert torch.equal(out[11]["exceedance_tau_cal"], norms > tau)


def test_no_tau_means_no_secondary_basis(saga_model, images):
    """A cell with no calibrated tau still runs: MAD is primary precisely
    because it needs no calibration."""
    out = ATT.attention_summaries(saga_model, images, [11], n_prefix=1,
                                  tau_cal=None)
    assert out[11]["exceedance_tau_cal"] is None
    assert out[11]["exceedance_mad"] is not None


# ── restoration (§5, and what the task file's "fused attention" line means) ──

@pytest.mark.parametrize("fixture_name",
                         ["saga_model", "baseline_model", "reg4_model"])
def test_the_model_is_restored_after_a_capture(request, fixture_name, images):
    """State hash, `fused_attn` flags, and no leaked instance `forward`.

    The task file says fused attention is "disabled for the dump"; this
    repository never disables it — `saga.attn_extract.capture_attention`
    recomputes the explicit map while the fused forward still produces the
    output. What must hold either way is that the model afterwards is
    EXACTLY the model before, and that is what is asserted here, three ways.
    """
    model = request.getfixturevalue(fixture_name)
    before_hash = state_hash(model)
    before_fused = ATT.fused_attn_flags(model)
    assert ATT.patched_forwards(model) == []

    ATT.attention_summaries(model, images, [10, 11],
                            n_prefix=infer_num_prefix_tokens(model))

    assert state_hash(model) == before_hash
    assert ATT.fused_attn_flags(model) == before_fused
    assert ATT.patched_forwards(model) == [], \
        "a capture leaked a patched forward into the model"


def test_capture_leaves_the_model_output_bit_identical(saga_model, images):
    """The capture must not change what the model computes."""
    with torch.no_grad():
        plain = saga_model(images).clone()
    ATT.attention_summaries(saga_model, images, [10, 11], n_prefix=1)
    with torch.no_grad():
        after = saga_model(images).clone()
    assert torch.equal(plain, after)


def test_fused_attn_flags_are_reported_where_they_exist(reg4_model,
                                                        saga_model):
    """timm's Attention carries `fused_attn`; this repo's GatedAttention does
    not. Both are handled, and the test states which is which so a future
    reader is not surprised by an empty dict."""
    assert ATT.fused_attn_flags(reg4_model), \
        "a timm model should expose fused_attn"
    assert ATT.fused_attn_flags(saga_model) == {}, \
        "GatedAttention calls SDPA unconditionally and has no flag"


# ── the conditions contract (the runner hook) ────────────────────────────────

def test_the_committed_conditions_file_parses_and_declares_D10():  # noqa: N802
    doc = load_conditions(CONDITIONS)
    assert doc["work_package"] == "I7_attention"
    assert doc["capture"] == "attention"
    assert doc["blocks"] == [10, 11]
    assert doc["alignment"] == {10: "in_b10", 11: "s11_out"}
    assert doc["exceedance_primary"] == "mad"
    assert doc["bootstrap_seed"] == 0
    assert doc["bootstrap_resamples"] == 10000
    assert [c["id"] for c in doc["conditions"]] == ["native"]


def test_the_runner_refuses_an_undeclared_block(tmp_path):
    import yaml
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["blocks"] = [9, 10, 11]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ATT.AttentionError, match=r"\[9\]"):
        load_conditions(bad)


def test_the_runner_refuses_a_wrong_alignment(tmp_path):
    """Declaring blocks[10] against s11_out would compare a block's incoming
    attention with the norms of its own OUTPUT."""
    import yaml
    doc = yaml.safe_load(CONDITIONS.read_text(encoding="utf-8"))
    doc["alignment"] = {10: "s11_out", 11: "s11_out"}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ATT.AttentionError, match="INPUT tokens"):
        load_conditions(bad)


def test_the_attention_hook_leaves_other_conditions_files_alone():
    for name in ("smoke", "I2_terminal", "I1_spatial", "I6_external",
                 "I5_readout"):
        path = REPO / "configs" / "frozen" / f"{name}.yaml"
        if not path.exists():
            continue
        doc = load_conditions(path)
        assert doc.get("capture") is None
        assert ATT.check_attention_block("x", doc) is doc


def test_summaries_refuse_an_undeclared_block(saga_model, images):
    with pytest.raises(ATT.AttentionError, match="alignment"):
        ATT.attention_summaries(saga_model, images, [5], n_prefix=1)


# ── D10, the wording rule ────────────────────────────────────────────────────

def _cells(ratio, ci_lo, ci_hi=None):
    from analysis.i7_wording import REQUIRED_BLOCKS, REQUIRED_RUN_IDS
    return [{"run_id": r, "block": b, "query_group": "patch",
             "ratio": ratio, "ci_lo": ci_lo,
             "ci_hi": ci_hi if ci_hi is not None else ratio + 1.0}
            for r in REQUIRED_RUN_IDS for b in REQUIRED_BLOCKS]


def test_the_wording_rule_returns_the_sink_verdict_when_it_is_earned():
    from analysis.i7_wording import VERDICT_SINK, decide

    v = decide(_cells(5.2, 4.1))
    assert v.verdict == VERDICT_SINK
    assert v.passed is True
    assert v.failures == []
    assert len(v.checked) == 4
    assert "attention sink" in v.sentence
    assert "5.20" in v.sentence and "4.10" in v.sentence


def test_the_wording_rule_returns_the_outlier_verdict_and_reports_the_ratio():
    from analysis.i7_wording import VERDICT_OUTLIER, decide

    v = decide(_cells(1.8, 1.4))
    assert v.verdict == VERDICT_OUTLIER
    assert v.passed is False
    assert "high-norm outlier tokens" in v.sentence
    # D10: "reports the measured ratio"
    assert "1.80" in v.sentence
    assert v.min_ratio == pytest.approx(1.8)


def test_the_ci_edge_a_lower_bound_of_exactly_three_does_not_exclude_three():
    """The criterion is a CI EXCLUDING 3. A CI whose lower bound IS 3
    contains it. `>` not `>=`, and this is the test that keeps it that way."""
    from analysis.i7_wording import VERDICT_OUTLIER, VERDICT_SINK, decide

    assert decide(_cells(3.5, 3.0)).verdict == VERDICT_OUTLIER
    assert decide(_cells(3.5, 3.0000001)).verdict == VERDICT_SINK
    # ratio exactly at the threshold is allowed (">= 3"), the CI decides
    assert decide(_cells(3.0, 3.01)).verdict == VERDICT_SINK
    assert decide(_cells(2.999, 3.01)).verdict == VERDICT_OUTLIER


def test_one_failing_cell_fails_the_whole_rule():
    """"in EVERY fresh baseline, at BOTH blocks" — so one cell decides it."""
    from analysis.i7_wording import VERDICT_OUTLIER, decide

    cells = _cells(9.0, 8.0)
    cells[2] = dict(cells[2], ratio=1.1, ci_lo=0.9)
    v = decide(cells)
    assert v.verdict == VERDICT_OUTLIER
    assert len(v.failures) == 1


def test_a_missing_measurement_cannot_pass():
    from analysis.i7_wording import VERDICT_OUTLIER, decide

    v = decide(_cells(9.0, 8.0)[:-1])
    assert v.verdict == VERDICT_OUTLIER
    assert v.missing == ["e2r_vits_mixup_baseline_s2|block10"] or v.missing
    v2 = decide([dict(c, ratio="MISSING") for c in _cells(9.0, 8.0)])
    assert v2.verdict == VERDICT_OUTLIER


def test_the_rule_reads_patch_queries_only():
    """D10 names patch queries. A CLS-query row cannot satisfy it."""
    from analysis.i7_wording import VERDICT_OUTLIER, decide

    cls_only = [dict(c, query_group="cls") for c in _cells(9.0, 8.0)]
    assert decide(cls_only).verdict == VERDICT_OUTLIER


def test_the_rule_names_the_two_fresh_baselines_and_both_blocks():
    from analysis.i7_wording import (RATIO_THRESHOLD, REQUIRED_BLOCKS,
                                     REQUIRED_QUERY_GROUP, REQUIRED_RUN_IDS)
    assert RATIO_THRESHOLD == 3.0
    assert REQUIRED_RUN_IDS == ("e2r_vits_mixup_baseline_s1",
                                "e2r_vits_mixup_baseline_s2")
    assert REQUIRED_BLOCKS == (10, 11)
    assert REQUIRED_QUERY_GROUP == "patch"


def test_the_rule_reads_a_committed_table(tmp_path):
    from analysis.i7_wording import VERDICT_SINK, from_table

    path = tmp_path / "T_I7a_incoming.csv"
    fields = ("run_id", "block", "query_group", "ratio", "ratio_ci_lo",
              "ratio_ci_hi")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for c in _cells(6.0, 5.0):
            w.writerow({"run_id": c["run_id"], "block": c["block"],
                        "query_group": "patch", "ratio": c["ratio"],
                        "ratio_ci_lo": c["ci_lo"], "ratio_ci_hi": c["ci_hi"]})
        w.writerow({"run_id": "someone_else", "block": 11,
                    "query_group": "patch", "ratio": 0.1,
                    "ratio_ci_lo": 0.0, "ratio_ci_hi": 0.2})
    v = from_table(path)
    assert v.verdict == VERDICT_SINK, "a non-required row must not count"


# ── the TTR operating curve (§6) ─────────────────────────────────────────────

def test_the_curve_is_built_from_the_committed_files():
    """32 points across two layer ranges; no new runs, nothing interpolated."""
    from analysis.frozen_I7_ttr_curve import build
    import tempfile

    out = tempfile.mkdtemp()
    report = build(out_dir=out)
    assert report["status"] == "OK"
    assert report["n_points"] == 32
    assert report["n_points_by_range"] == {"all": 8, "midlayer": 24}
    assert report["n_neurons_by_range"]["midlayer"] == [0, 8, 10, 12, 16, 24]
    assert report["n_neurons_by_range"]["all"] == [0, 9, 10, 11, 12, 13, 14,
                                                   15]
    assert len(report["cells_by_range"]["midlayer"]) == 4
    assert len(report["cells_by_range"]["all"]) == 1


def test_the_curve_marks_absent_cells_missing_and_never_fills_them():
    from analysis.frozen_I7_ttr_curve import MISSING, build
    import tempfile

    out = tempfile.mkdtemp()
    report = build(out_dir=out)
    assert sorted(report["missing_cells_by_range"]["all"]) == [
        "e2r_vitb_mixup_baseline_s1", "e2r_vits_nomix_baseline_s1",
        "legacy_vits_baseline"]
    assert report["missing_cells_by_range"]["midlayer"] == []

    with open(Path(out) / "T_I7d_ttr_coverage.csv", newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    absent = [r for r in rows if r["status"] == MISSING]
    assert len(absent) == 3
    for r in absent:
        # a MISSING cell carries no grid and no points — never a zero that a
        # later mean would absorb
        assert r["n_neurons_grid"] == MISSING
        assert r["n_points"] == "0"


def test_the_gate_thresholds_are_read_not_restated():
    from analysis.frozen_I7_ttr_curve import build
    import tempfile

    report = build(out_dir=tempfile.mkdtemp())
    assert report["gate"]["agreed"] is True
    assert report["gate"]["min_outlier_reduction"] == 0.5
    assert report["gate"]["max_top1_drop"] == 1.0


def test_the_curve_agrees_with_the_committed_sweep_table():
    """`T_ttr_sweep.csv` is the midlayer aggregate; every row of it must
    match a per-run point. A disagreement is reported, never smoothed."""
    from analysis.frozen_I7_ttr_curve import build
    import tempfile

    report = build(out_dir=tempfile.mkdtemp())
    assert report["cross_check_issues"] == []


def test_the_chosen_operating_point_is_marked_on_the_right_range():
    """Four cells, four chosen points, all at n_neurons=24 in the MIDLAYER
    range — matching on neuron count alone would also mark the all-range
    point, which is a different configuration."""
    from analysis.frozen_I7_ttr_curve import build
    import tempfile

    out = tempfile.mkdtemp()
    report = build(out_dir=out)
    assert report["n_marked_chosen"] == 4
    with open(Path(out) / "T_I7d_ttr_curve.csv", newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chosen = [r for r in rows if r["is_chosen_operating_point"] == "1"]
    assert len(chosen) == 4
    assert {r["layer_range_label"] for r in chosen} == {"midlayer"}
    assert {r["n_neurons"] for r in chosen} == {"24"}


def test_a_fake_sweep_with_a_gap_reports_the_gap(tmp_path, monkeypatch):
    """The MISSING path, on a fabricated tree: a range with one cell and a
    range with two must report the absent one and no others."""
    import analysis.frozen_I7_ttr_curve as CURVE

    roots = {"one": tmp_path / "one", "two": tmp_path / "two"}
    spec = {"one": ["cell_a"], "two": ["cell_a", "cell_b"]}
    for label, cells in spec.items():
        for cell in cells:
            d = roots[label] / cell
            d.mkdir(parents=True)
            (d / "sweep.csv").write_text(
                "n_neurons,n_images,top1,top1_drop,sink_fixed_canon,"
                "sink_mad_k5,oversmooth_pairwise,sink_reduction_frac,"
                "layers_touched,passes\n"
                "0,10,80.0,0.0,10.0,9.0,0.4,,,\n"
                "8,10,79.5,0.5,4.0,3.5,0.39,0.6,\"4,5\",True\n",
                encoding="utf-8")
            (d / "validate.json").write_text(json.dumps({
                "arch": "vit_small", "recipe": "mixup",
                "ckpt_sha256": "e" * 64, "canon_tau": 20.0,
                "layer_range": [0, 12], "git_sha": "f" * 40,
                "scan_stats": {"layer_range": [0, 12]},
                "verdict": {"min_sink_reduction": 0.5, "max_top1_drop": 1.0},
            }), encoding="utf-8")

    monkeypatch.setattr(CURVE, "SWEEP_ROOTS", roots)
    monkeypatch.setattr(CURVE, "SWEEP_TABLE", tmp_path / "absent.csv")
    monkeypatch.setattr(CURVE, "TTR_TABLE", tmp_path / "absent2.csv")
    report = CURVE.build(out_dir=tmp_path / "tables")
    # 3 run directories (one/cell_a, two/cell_a, two/cell_b) x 2 sweep rows
    assert report["n_points"] == 6
    assert report["n_points_by_range"] == {"one": 2, "two": 4}
    assert report["missing_cells_by_range"] == {"one": ["cell_b"], "two": []}
    assert report["cross_check_issues"][0]["issue"].startswith(
        "T_ttr_sweep.csv is absent")


def test_the_curve_npz_marks_missing_values_rather_than_zeroing_them(tmp_path):
    from analysis.frozen_I7_ttr_curve import build

    npz = tmp_path / "F_ttr_curve.npz"
    build(out_dir=tmp_path / "tables", npz_path=npz)
    with np.load(npz, allow_pickle=False) as z:
        assert z["outlier_reduction_frac"].shape == z[
            "outlier_reduction_frac_is_missing"].shape
        # the n_neurons=0 rows have no reduction fraction by construction
        missing = z["outlier_reduction_frac_is_missing"]
        assert missing.sum() >= 5
        assert np.isnan(z["outlier_reduction_frac"][missing]).all()
        assert float(z["gate_min_outlier_reduction"][0]) == 0.5


# ── the analysis ─────────────────────────────────────────────────────────────

def test_the_ratio_is_a_ratio_of_means_not_a_mean_of_ratios():
    """The two estimators differ, and the module must use the one D10 names.

    Constructed so the difference is unmissable: one image has a near-zero
    denominator, which a mean-of-ratios would let dominate.
    """
    from analysis.frozen_I7_analysis import ratio_of_means_ci

    exc = [1.0, 1.0, 1.0]
    other = [0.5, 0.5, 0.001]
    ratio, lo, hi = ratio_of_means_ci(exc, other, resamples=200)
    assert ratio == pytest.approx(3.0 / (0.5 + 0.5 + 0.001), rel=1e-9)
    mean_of_ratios = np.mean([a / b for a, b in zip(exc, other)])
    assert ratio < mean_of_ratios / 10
    assert lo <= ratio <= hi


def test_the_bootstrap_is_deterministic_and_seeded_at_zero():
    from analysis.frozen_I7_analysis import (BOOTSTRAP_RESAMPLES,
                                             BOOTSTRAP_SEED, mean_ci)
    assert (BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES) == (0, 10000)
    values = list(np.random.RandomState(4).rand(50))
    assert mean_ci(values) == mean_ci(values)


def test_auc_is_the_mann_whitney_statistic():
    """Hand values: a perfect separation is 1.0, a reversal 0.0, all-ties
    0.5, and one class absent is MISSING rather than a default."""
    from analysis.frozen_I7_analysis import MISSING, auc_per_image

    scores = np.array([[0.1, 0.2, 0.9, 0.8],
                       [0.9, 0.8, 0.1, 0.2],
                       [0.5, 0.5, 0.5, 0.5],
                       [0.1, 0.2, 0.3, 0.4]])
    labels = np.array([[False, False, True, True],
                       [False, False, True, True],
                       [False, False, True, True],
                       [False, False, False, False]])
    got = auc_per_image(scores, labels)
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(0.0)
    assert got[2] == pytest.approx(0.5)
    assert got[3] is MISSING


def test_tables_are_built_and_stamped(tmp_path, saga_model, reg4_model,
                                      images):
    """End to end on two fake runs: every table exists, every row carries an
    endpoint_class, and the register table only covers register models."""
    from analysis.frozen_I7_analysis import (DESCRIPTIVE, EXPLORATORY,
                                             SECONDARY, build_all)

    root = tmp_path / "runs"
    manifest_rows = []
    for run_id, model, variant in (("fake_baseline", saga_model, "baseline"),
                                   ("fake_registers", reg4_model,
                                    "registers")):
        row = {"run_id": run_id, "arch": "vit_tiny",
               "recipe_actual": "mixup", "variant": variant,
               "ckpt_kind": "last", "ckpt_sha256": "a" * 64}
        _write_fake_npz(root / run_id, model, images, row)
        manifest_rows.append(dict(
            row, family="e2r_300ep", status="eligible",
            provenance_tag="s1", seed_controlled=1))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": manifest_rows}), encoding="utf-8")

    out = tmp_path / "tables"
    report = build_all(root, manifest, out)
    assert report["n_runs_loaded"] == 2

    for name in ("T_I7a_incoming", "T_I7b_registers", "T_I7c_value_norm"):
        with open(out / f"{name}.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"{name} is empty"
        assert {r["endpoint_class"] for r in rows} <= {
            SECONDARY, EXPLORATORY, DESCRIPTIVE}
        if name == "T_I7b_registers":
            assert {r["variant"] for r in rows} == {"registers"}
            assert {r["endpoint_class"] for r in rows} == {DESCRIPTIVE}
        if name == "T_I7c_value_norm":
            assert {r["endpoint_class"] for r in rows} == {EXPLORATORY}

    with open(out / "T_I7a_incoming.csv", newline="", encoding="utf-8") as f:
        a_rows = list(csv.DictReader(f))
    assert {r["block"] for r in a_rows} == {"10", "11"}
    assert {r["query_group"] for r in a_rows} == {"patch", "cls"}
    assert {(r["block"], r["aligned_stage"]) for r in a_rows} == {
        ("10", "in_b10"), ("11", "s11_out")}


def _write_fake_npz(out_dir, model, images, row):
    """One run's incoming_mass.npz, produced by the real capture path."""
    from saga.frozen.attention import run_attention_package

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return images.shape[0]

        def __getitem__(self, i):
            return images[i], 0

    run_attention_package(
        row=row, conditions={"work_package": "I7_attention",
                             "blocks": [10, 11], "precision": "fp32"},
        dataset=_DS(), out_dir=out_dir,
        image_ids=[f"img/{i}" for i in range(images.shape[0])],
        batch_size=2, split_name="fake", split_sha="b" * 64,
        git_sha="c" * 40, tau_cal=1.0, model=model)


def test_the_npz_round_trips_and_carries_its_provenance(tmp_path, saga_model,
                                                        images):
    from analysis.frozen_I7_analysis import load_run

    _write_fake_npz(tmp_path / "fake_run", saga_model, images,
                    {"run_id": "fake_run", "arch": "vit_tiny",
                     "recipe_actual": "mixup", "variant": "saga",
                     "ckpt_kind": "last", "ckpt_sha256": "a" * 64})
    data = load_run(tmp_path, "fake_run")
    assert data is not None
    meta = data["meta"]
    assert meta["n_prefix"] == 1
    assert meta["blocks"] == [10, 11]
    assert meta["alignment"] == {"10": "in_b10", "11": "s11_out"}
    assert meta["exceedance_primary"] == "mad"
    assert meta["state_restored"] is True
    assert meta["split_sha256"] == "b" * 64
    assert data["in_mass_patchq"].shape == (images.shape[0], 2, 197)
    assert data["exceedance_mad"].shape == (images.shape[0], 2, N_PATCHES)
    assert "fused_attn_flags" in meta
    # NO attention map is stored
    assert not any(a.ndim >= 4 for a in
                   (v for k, v in data.items() if isinstance(v, np.ndarray)))


def test_the_writer_is_resubmit_safe(tmp_path, saga_model, images):
    from saga.frozen.attention import is_done

    out = tmp_path / "fake_run"
    _write_fake_npz(out, saga_model, images,
                    {"run_id": "fake_run", "arch": "vit_tiny",
                     "recipe_actual": "mixup", "variant": "saga",
                     "ckpt_kind": "last", "ckpt_sha256": "a" * 64})
    assert is_done(out, "a" * 64)
    assert not is_done(out, "b" * 64)


# ── guards ───────────────────────────────────────────────────────────────────

def test_no_optimizer_or_training_import_in_the_i7_files():
    """§11 / I0 handoff §8.7, extended to this work package's files."""
    banned_modules = ("torch.optim", "torch.optim.lr_scheduler")
    banned_names = {"AdamW", "Adam", "SGD", "backward", "step",
                    "LabelSmoothingCrossEntropy", "Mixup"}
    targets = [
        REPO / "saga" / "frozen" / "attention.py",
        REPO / "tools" / "frozen_I7_attn.py",
        REPO / "analysis" / "frozen_I7_analysis.py",
        REPO / "analysis" / "frozen_I7_ttr_curve.py",
        REPO / "analysis" / "i7_wording.py",
    ]
    assert all(t.exists() for t in targets)
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("torch.optim"), path.name
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "") not in banned_modules, path.name
                for a in node.names:
                    assert a.name not in banned_names, path.name
            elif isinstance(node, ast.Attribute):
                assert node.attr not in ("backward", "zero_grad"), path.name


#: Historical identifiers this project already uses, which a module that READS
#: a committed file cannot avoid naming. Everything else is the paper's
#: wording and belongs in `analysis/i7_wording.py` alone.
ALLOWED_SINK_IDENTIFIERS = (
    "sink_counts_mad", "sink_fixed_canon", "sink_mad_k5", "sink_canon",
    "sink_reduction_frac", "min_sink_reduction", "gate_min_sink_reduction",
    "meets_sink", "sink_canon_removed_abs", "sink_canon_reduction_frac",
    "sink_mad", "sink_address",
)


def test_the_paper_wording_lives_in_exactly_one_module():
    """§11: "sink" appears in code only inside `analysis/i7_wording.py`.

    Historical COLUMN NAMES in committed files (`sink_reduction_frac` and
    friends) are data references, not the paper's wording, and a module that
    reads those files must name them; they are allow-listed by name. The
    PHRASE "attention sink" is the wording itself and is permitted nowhere
    else at all.
    """
    targets = [
        REPO / "saga" / "frozen" / "attention.py",
        REPO / "tools" / "frozen_I7_attn.py",
        REPO / "analysis" / "frozen_I7_analysis.py",
        REPO / "analysis" / "frozen_I7_ttr_curve.py",
        REPO / "configs" / "frozen" / "I7_attention.yaml",
        REPO / "scripts" / "jobs" / "frozen_I7.sbatch",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "attention sink" not in text.lower(), \
            f"{path.name} uses the paper's wording; that belongs in " \
            f"analysis/i7_wording.py"
        stripped = text
        for allowed in ALLOWED_SINK_IDENTIFIERS:
            stripped = stripped.replace(allowed, "")
        assert "sink" not in stripped.lower(), \
            f"{path.name} contains 'sink' outside the allow-listed " \
            f"historical identifiers"

    wording = (REPO / "analysis" / "i7_wording.py").read_text(encoding="utf-8")
    assert "attention sink" in wording.lower()


def test_the_i7_runner_hook_is_one_line():
    """§7: `attention.py` is a new module and the only change to runner.py is
    a single hook line plus its import."""
    src = (REPO / "saga" / "frozen" / "runner.py").read_text(encoding="utf-8")
    calls = [ln for ln in src.splitlines()
             if "_c_check_attention_block(" in ln and "import" not in ln]
    assert len(calls) == 1, f"expected one hook call, found {calls}"
    assert "# TASK C / I7" in calls[0]
    # and Track B's and Track C/I5's hooks are still there
    assert "_resolve_permutations(" in src
    assert "_c_check_readout_block(" in src


def test_the_job_file_lists_the_cell_in_array_order():
    from saga.frozen.runner import eligible_cohort

    text = (REPO / "scripts" / "jobs"
            / "frozen_I7.sbatch").read_text(encoding="utf-8")
    block = text.split("RUN_IDS=(", 1)[1].split("\n)", 1)[0]
    listed = [ln.split('"')[1] for ln in block.splitlines()
              if ln.strip().startswith('"')]
    want = [r["run_id"] for r in eligible_cohort(
        REPO / "results" / "frozen" / "I0_manifest" / "manifest.json")
        if r["arch"] == "vit_small" and r["recipe_actual"] == "mixup"]
    assert listed == want
    assert len(want) == 10, "§2 declares the ViT-S/mixup cell as 10 runs"
    assert f"--array=0-{len(want) - 1}" in text


def test_the_job_file_fixes_the_split_to_sub1k():
    text = (REPO / "scripts" / "jobs"
            / "frozen_I7.sbatch").read_text(encoding="utf-8")
    assert "SPLIT=results/frozen/splits/sub1k.json" in text
    assert "frozen_I7_attn.py" in text


def test_the_declared_split_is_the_one_d10_names():
    split = json.loads(
        (REPO / "results" / "frozen" / "splits" / "sub1k.json").read_text(
            encoding="utf-8"))
    assert split["n"] == 1000
    assert split["name"] == "sub1k"
    assert split["sha256"] == (
        "6a38d1000b32f2ca0c5bcfd1187cde86258b2d45a4b1e4b47cfd5580fe088ef3")
    locked = (REPO / "docs" / "LOCKED_ANALYSIS.md").read_text(encoding="utf-8")
    assert split["sha256"] in locked
