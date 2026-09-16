"""
tests/test_I2_terminal.py
=========================
TASK I2 D8 — the terminal patch-gate sweep, on CPU with tiny fake models and
fake data, in the style of `tests/test_I0_frozen.py`.

What these tests are actually for, in order of how much damage they prevent:

  1. `native` and `term_0.50` must be THE SAME PASS on a checkpoint whose
     terminal phi is 0. If they are not, every "survival" number in I2 is
     measuring an implementation difference and not a gate effect.
  2. `s11_out` cannot depend on the LAST block's gate. A capture that moved
     under a `term_*` condition would mean the whole stage comparison is
     circular.
  3. `count_fixed_canon` must be MISSING off `hist`. The historical tau was
     calibrated at the last block; printed at `s11_out` it would be a number
     with no definition, and a mean over stages would silently absorb it.
  4. `results/diagsplit/fixed_thresholds_canon.json` must not move. Its
     content is pinned here, and the writer refuses to target it by path.
  5. The decision rule must decide. All three branches are exercised on
     synthetic gaps, so the branch cannot be chosen by reading a table.

Two model shapes are exercised everywhere it matters: a tiny SAGA ViT (1
prefix token) and a tiny timm 4-REGISTER ViT (5 prefix tokens), built
directly and never wrapped in SAGAViT.
"""

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from saga.frozen import diag as fdiag
from saga.frozen import records as rec
from saga.frozen import runner as frunner
from saga.frozen.edits import TERMINAL_GATE_VALUES, state_hash
from saga.frozen.stages import HIST_STAGE
from saga.metrics import (infer_num_prefix_tokens, oversmoothing_pairwise,
                          oversmoothing_pairwise_nosink, sink_counts_mad)
from saga.vit import build_saga_vit

REPO = Path(__file__).resolve().parents[1]
CONDITIONS_YAML = REPO / "configs" / "frozen" / "I2_terminal.yaml"
JOB_FILE = REPO / "scripts" / "jobs" / "frozen_I2.sbatch"
MANIFEST = REPO / "results" / "frozen" / "I0_manifest" / "manifest.json"
CANON = REPO / "results" / "diagsplit" / "fixed_thresholds_canon.json"

DEPTH = 4
DIM = 24
HEADS = 3
N_PATCHES = 196
MISSING = "MISSING"

#: The canon thresholds, pinned. See `test_the_canon_thresholds_file_is_pinned`
#: for why the digest is taken over LF-normalised bytes.
CANON_SHA256_LF = (
    "c30f2b9947cf035ee3e7f18265edc97ea5759524be84a2aaab3461d39a617ba0")
CANON_TAU = {"vit_small|mixup": 20.8515625,
             "vit_base|mixup": 127.3125,
             "vit_small|nomix": 22.859375}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _structured_gate(model, seed=0, flat_terminal=False):
    """Spatially structured phi in every block; optionally phi = 0 in the LAST.

    A freshly built SAGA gate has phi = 0 everywhere, so sigma(phi) = 0.5 and
    a terminal override to 0.50 would be trivially equal to native for the
    wrong reason. The trained checkpoints have real structure in the blocks a
    loss reached and phi_L = 0 in the last one (weight decay's fixed point,
    since a CLS-only loss never touches it). `flat_terminal` reproduces that
    shape. No training, no optimizer: this is `torch.no_grad()` assignment.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for i, blk in enumerate(model.blocks):
            phi = blk.attn.gate.phi
            last = (i == len(model.blocks) - 1)
            if flat_terminal and last:
                phi.zero_()
            else:
                phi.copy_(torch.randn(phi.shape, generator=g) * 1.5)
    return model


@pytest.fixture(scope="module")
def saga_model():
    torch.manual_seed(0)
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=10,
                           depth=DEPTH, embed_dim=DIM, num_heads=HEADS).eval()
    return _structured_gate(model)


@pytest.fixture(scope="module")
def saga_model_phi_L_zero():
    """A SAGA model shaped like a real checkpoint: structure everywhere the
    CLS loss reached, and phi = 0 in the terminal block."""
    torch.manual_seed(0)
    model = build_saga_vit("vit_tiny_patch16_224", gate=True, num_classes=10,
                           depth=DEPTH, embed_dim=DIM, num_heads=HEADS).eval()
    return _structured_gate(model, flat_terminal=True)


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


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        torch.manual_seed(i)
        return torch.randn(3, 224, 224), i % 10


def _items(n):
    return [[f"val/n{c:08d}/IMG_{c}.JPEG", c] for c in range(n)]


def _row(**kw):
    base = {"run_id": "fake_run", "arch": "vit_small", "variant": "saga",
            "recipe_actual": "mixup", "ckpt_kind": "last",
            "ckpt_sha256": "a" * 64, "gate_mode": "spatial", "git_dirty": 1,
            "patch_file": MISSING, "provenance_tag": "s1",
            "ckpt_path": "results/runs/fake_run/ckpt/last.pth"}
    base.update(kw)
    return base


def _tau_doc(split_sha="b" * 64, cell="vit_small|mixup"):
    return {"split_sha256": split_sha,
            "tau_cal": {cell: {"s11_out": 1.0, "hist": 1.5,
                               "s12_post_norm": 0.8}}}


def _sweep(model, monkeypatch, tmp_path, *, row=None, n_images=4,
           conditions=None, tau_doc=None):
    """Run the real multi-stage loop on a fake model and fake data."""
    monkeypatch.setattr(frunner, "build_from_row",
                        lambda row, ckpt_path=None, device="cpu": model)
    return frunner.run_work_package_stages(
        row=row or _row(), conditions=conditions or CONDITIONS,
        dataset=FakeDataset(_items(n_images)), out_dir=tmp_path,
        device="cpu", batch_size=2, split_name="calibration",
        split_sha="b" * 64, git_sha="c" * 40, git_dirty=1,
        tau_cal_doc=tau_doc or _tau_doc(),
        canon_doc=fdiag.load_canon_thresholds(CANON))


CONDITIONS = frunner.load_conditions(CONDITIONS_YAML)


# ─────────────────────────────────────────────────────────────────────────────
# The conditions file — the only place conditions are defined (D1)
# ─────────────────────────────────────────────────────────────────────────────

def test_the_shipped_I2_conditions_file_is_valid_and_complete():
    doc = frunner.load_conditions(CONDITIONS_YAML)
    assert doc["work_package"] == "I2_terminal"
    assert doc["stage"] == "hist"
    assert doc["precision"] == "fp32"

    ids = [c["id"] for c in doc["conditions"]]
    assert ids == ["native", "term_0.50", "term_0.25", "term_0.75",
                   "term_1.00"]
    assert doc["conditions"][0]["edit_type"] == "native", \
        "native must be declared FIRST — the logit diff is measured against it"

    # the four declared constants of task §3, and no fifth
    values = {c["params"]["value"] for c in doc["conditions"]
              if c["edit_type"] == "terminal_gate_override"}
    assert values == set(TERMINAL_GATE_VALUES) == {0.25, 0.5, 0.75, 1.0}

    # §3: native on all 19; every term_* on SAGA only
    assert doc["conditions"][0]["applies_to"] == ["baseline", "registers",
                                                  "saga"]
    for c in doc["conditions"][1:]:
        assert c["applies_to"] == ["saga"], c["id"]

    # §3: s11_out is captured ONCE, under native — it cannot depend on the
    # last block's gate, and capturing it again would invite a reader to
    # compare two captures that are equal by construction
    assert doc["conditions"][0]["stages"] == ["s11_out", "hist",
                                              "s12_post_norm"]
    for c in doc["conditions"][1:]:
        assert c["stages"] == ["hist", "s12_post_norm"], c["id"]

    assert doc["thresholds_canon"] == fdiag.CANON_THRESHOLDS
    # a TEMPLATE, not a path: tau_cal belongs to exactly one split
    assert doc["thresholds_cal"] == \
        "results/frozen/I2_terminal/{split_name}/thresholds_cal.json"
    assert doc["calibration_stages"] == ["s11_out", "hist", "s12_post_norm"]
    # LOCKED_ANALYSIS §8 / D6 (default 0): declared, not implied
    assert doc["bootstrap_resamples"] == 10000
    assert doc["bootstrap_seed"] == 0


def test_every_declared_stage_is_calibrated_by_the_conditions_file():
    """A stage measured but not calibrated would write count_fixed_cal
    against a threshold that does not exist."""
    doc = frunner.load_conditions(CONDITIONS_YAML)
    declared = {s for c in doc["conditions"]
                for s in frunner.condition_stages(doc, c)}
    assert declared <= set(doc["calibration_stages"])


def test_an_undeclared_condition_variant_or_stage_is_refused(tmp_path):
    """The runner runs the YAML and only the YAML."""
    import yaml
    p = tmp_path / "c.yaml"

    def write(conditions, **extra):
        p.write_text(yaml.safe_dump(
            {"work_package": "X", "stage": "hist",
             "conditions": conditions, **extra}))

    write([{"id": "a", "edit_type": "teleport"}])
    with pytest.raises(frunner.RunnerError, match="unknown edit_type"):
        frunner.load_conditions(p)

    write([{"id": "a", "edit_type": "native", "applies_to": ["sagaa"]}])
    with pytest.raises(frunner.RunnerError, match="not model variants"):
        frunner.load_conditions(p)

    write([{"id": "a", "edit_type": "native", "stages": ["s13_out"]}])
    with pytest.raises(Exception, match="unknown stage"):
        frunner.load_conditions(p)

    # `hist` and `s12_pre_norm` are ONE stage under two names; declaring both
    # would write one capture as two rows
    write([{"id": "a", "edit_type": "native",
            "stages": ["hist", "s12_pre_norm"]}])
    with pytest.raises(frunner.RunnerError, match="resolve to the same stage"):
        frunner.load_conditions(p)

    # an unlisted gate constant — the operating points are the declared four
    write([{"id": "a", "edit_type": "native"},
           {"id": "b", "edit_type": "terminal_gate_override",
            "params": {"value": 0.6}}])
    with pytest.raises(frunner.RunnerError, match="declared constants"):
        frunner.load_conditions(p)

    # half-declared stages: one row shape per file
    write([{"id": "a", "edit_type": "native", "stages": ["hist"]},
           {"id": "b", "edit_type": "native"}])
    with pytest.raises(frunner.RunnerError, match="single-stage or "
                                                  "multi-stage throughout"):
        frunner.load_conditions(p)


def test_conditions_are_filtered_by_the_checkpoints_own_variant():
    doc = frunner.load_conditions(CONDITIONS_YAML)
    assert [c["id"] for c in frunner.conditions_for(doc, "baseline")] == \
        ["native"]
    assert [c["id"] for c in frunner.conditions_for(doc, "registers")] == \
        ["native"]
    assert len(frunner.conditions_for(doc, "saga")) == 5
    with pytest.raises(frunner.RunnerError, match="unknown model variant"):
        frunner.conditions_for(doc, "sagaa")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic definitions — imported from saga/metrics.py, never re-derived
# ─────────────────────────────────────────────────────────────────────────────

def test_the_mad_threshold_is_the_existing_definition():
    """`mad_threshold` must be the threshold `sink_counts_mad` counts above,
    and the same number `tools/compute_fixed_thr.py` used for every canon
    tau. Both equalities are asserted on random data, not asserted about."""
    g = torch.Generator().manual_seed(7)
    norms = torch.rand(9, N_PATCHES, generator=g) * 10.0
    norms[0, :5] += 50.0                      # a few genuine outliers

    for k in (2.0, 5.0):
        thr = fdiag.mad_threshold(norms, k)
        assert thr.shape == (9,)
        # 1. the counts agree EXACTLY with the project's own count
        assert torch.equal((norms > thr.unsqueeze(1)).sum(dim=1).float(),
                           sink_counts_mad(norms, k=k))
        # 2. the values agree with the canon tool's numpy implementation to
        #    fp32 precision: the torch path works in fp32 (the dtype the
        #    norms are captured in) and the canon tool upcasts to float64, so
        #    the two differ by rounding and by nothing else. The COUNT
        #    equality above is EXACT, and the count is what every recorded
        #    number is built from.
        canon = fdiag.canon_per_image_mad_thresholds(norms.numpy(), k)
        assert np.allclose(thr.numpy(), canon, rtol=1e-6, atol=0)


def test_the_calibration_is_the_canon_tools_lower_median():
    g = torch.Generator().manual_seed(3)
    norms = torch.rand(11, N_PATCHES, generator=g) * 4.0
    per_image = fdiag.canon_per_image_mad_thresholds(norms.numpy(), 5.0)
    tau = fdiag.calibrate_tau(per_image)
    # lower median: the (n-1)//2-th of the sorted values, never an
    # interpolation between two of them (np.median would interpolate)
    assert tau == float(np.sort(per_image)[(per_image.size - 1) // 2])


def test_patch_diagnostics_reuse_the_project_definitions(saga_model, images):
    from saga.frozen.stages import forward_with_stages
    _, store = forward_with_stages(saga_model, images, ("hist",))
    patches = store["hist"]
    values, maps = fdiag.patch_diagnostics(patches, tau_cal=1.0,
                                           tau_canon=2.0)

    assert np.allclose(values["cos_all"],
                       oversmoothing_pairwise(patches).tolist())
    from saga.metrics import effective_rank
    assert np.allclose(values["eff_rank"], effective_rank(patches).tolist())
    # cos_nosink_mad is the per-image form of the project's batch function
    for i in range(patches.shape[0]):
        one, _ = oversmoothing_pairwise_nosink(
            patches[i:i + 1], patches[i:i + 1].norm(dim=-1), k=fdiag.MAD_K)
        assert values["cos_nosink_mad"][i] == pytest.approx(float(one),
                                                            rel=1e-6)
    assert set(maps) == set(fdiag.MAP_BASES)
    for basis in fdiag.MAP_BASES:
        assert maps[basis].shape == (patches.shape[0], N_PATCHES)
        assert maps[basis].dtype == bool
    # the map and the count are the same statement
    assert np.allclose(maps["mad"].sum(axis=1), values["count_mad"])
    assert np.allclose(maps["fixed_cal"].sum(axis=1),
                       values["count_fixed_cal"])


def test_every_declared_diagnostic_key_is_produced(saga_model, images):
    from saga.frozen.stages import forward_with_stages
    _, store = forward_with_stages(saga_model, images, ("hist",))
    values, _ = fdiag.patch_diagnostics(store["hist"], tau_cal=1.0,
                                        tau_canon=2.0)
    assert set(values) == set(fdiag.DIAG_KEYS)
    assert set(fdiag.DIAG_KEYS) <= set(rec.DIAG_STAGE_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# The two invariances the whole work package rests on
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", TERMINAL_GATE_VALUES)
def test_s11_out_is_bit_identical_under_every_terminal_constant(
        saga_model, images, value):
    """s11_out is the output of the SECOND-TO-LAST block; the terminal
    override swaps the LAST block's gate, which is downstream of it. If this
    ever moved, every s11_out comparison in I2 would be circular."""
    targets = torch.zeros(images.shape[0], dtype=torch.long)
    stages = ("s11_out", "hist")
    _, native, _, _ = frunner.run_condition_multistage(
        saga_model, images, targets, {"id": "native", "edit_type": "native"},
        stages)
    _, edited, _, _ = frunner.run_condition_multistage(
        saga_model, images, targets,
        {"id": f"term_{value}", "edit_type": "terminal_gate_override",
         "params": {"value": value}}, stages)

    assert torch.equal(native["s11_out"], edited["s11_out"])
    if value != 0.5:
        # ... while the stage AFTER the edited gate does move, so the
        # equality above is not the trivial one of an edit that did nothing
        assert not torch.equal(native["hist"], edited["hist"])


def test_native_equals_term_050_when_phi_L_is_zero(saga_model_phi_L_zero,
                                                   images):
    """phi_L = 0 is weight decay's fixed point under a CLS-only loss, so
    sigma(phi_L) = 0.50 and `term_0.50` is the SAME computation as `native`.
    Bit-identical, not merely close: both paths multiply the patch rows of
    the SDPA output by the same fp32 0.5."""
    model = saga_model_phi_L_zero
    assert torch.equal(model.blocks[-1].attn.gate.phi,
                       torch.zeros_like(model.blocks[-1].attn.gate.phi))
    targets = torch.zeros(images.shape[0], dtype=torch.long)
    stages = ("s11_out", "hist", "s12_post_norm")

    resp_n, native, _, logits_n = frunner.run_condition_multistage(
        model, images, targets, {"id": "native", "edit_type": "native"},
        stages)
    resp_e, edited, _, logits_e = frunner.run_condition_multistage(
        model, images, targets,
        {"id": "term_0.50", "edit_type": "terminal_gate_override",
         "params": {"value": 0.5}}, stages, native_logits=logits_n)

    assert torch.equal(logits_n, logits_e)
    assert max(resp_e["max_abs_logit_diff_vs_native"]) == 0.0
    assert resp_n["nll"] == resp_e["nll"]
    for stage in stages:
        assert torch.equal(native[stage], edited[stage]), stage

    # and the diagnostics computed from them are identical too
    for stage in stages:
        a, _ = fdiag.patch_diagnostics(native[stage], tau_cal=1.0)
        b, _ = fdiag.patch_diagnostics(edited[stage], tau_cal=1.0)
        assert a == b, stage

    # the same model with a STRUCTURED terminal phi does move, so the
    # equality above is a property of phi_L = 0 and not of the harness
    _structured_gate(model, seed=11)
    try:
        _, moved, _, _ = frunner.run_condition_multistage(
            model, images, targets,
            {"id": "term_0.50", "edit_type": "terminal_gate_override",
             "params": {"value": 0.5}}, ("hist",))
        _, base, _, _ = frunner.run_condition_multistage(
            model, images, targets,
            {"id": "native", "edit_type": "native"}, ("hist",))
        assert not torch.equal(base["hist"], moved["hist"])
    finally:
        _structured_gate(model, flat_terminal=True)


def test_the_terminal_override_restores_the_model(saga_model, images):
    before = state_hash(saga_model)
    targets = torch.zeros(images.shape[0], dtype=torch.long)
    for value in TERMINAL_GATE_VALUES:
        frunner.run_condition_multistage(
            saga_model, images, targets,
            {"id": "x", "edit_type": "terminal_gate_override",
             "params": {"value": value}}, ("hist",))
    assert state_hash(saga_model) == before


# ─────────────────────────────────────────────────────────────────────────────
# The multi-stage loop
# ─────────────────────────────────────────────────────────────────────────────

def _rows(out_dir, kind):
    """The `kind` file as string-valued dicts, parquet OR csv.

    `saga/frozen/records.py` writes parquet when pyarrow is importable and
    CSV otherwise — the HPC has it, a bare login node may not — so a test
    that reads back what the runner wrote must not assume either. Values are
    normalised to `str` so one assertion covers both (CSV gives strings;
    parquet gives typed values).
    """
    from analysis.build_I2_tables import read_rows
    return [{k: (MISSING if v is None else str(v)) for k, v in row.items()}
            for row in read_rows(rec.records_path(out_dir, kind))]


def _diag_rows(out_dir):
    return _rows(out_dir, "diag")


def test_the_sweep_writes_one_diag_row_per_image_condition_stage(
        saga_model, tmp_path, monkeypatch):
    summary = _sweep(saga_model, monkeypatch, tmp_path, n_images=4)
    assert summary["state_restored"] is True
    # 5 conditions x 4 images
    assert summary["n_records"] == 20
    # native at 3 stages + 4 constants at 2 stages = 11, x 4 images
    assert summary["n_diag"] == 44

    rows = _diag_rows(tmp_path)
    seen = {(r["condition_id"], r["stage"]) for r in rows}
    assert seen == {("native", s) for s in ("s11_out", "hist",
                                            "s12_post_norm")} | {
        (f"term_{v:.2f}", s) for v in TERMINAL_GATE_VALUES
        for s in ("hist", "s12_post_norm")}


def test_count_fixed_canon_is_MISSING_off_hist(saga_model, tmp_path,
                                               monkeypatch):
    """§4: the historical tau was calibrated at the LAST BLOCK. At s11_out or
    after the final norm it has no definition, and MISSING is a VALUE — never
    a rescaled guess, never a null that a mean would skip."""
    _sweep(saga_model, monkeypatch, tmp_path, n_images=4)
    rows = _diag_rows(tmp_path)
    tau = CANON_TAU["vit_small|mixup"]
    for r in rows:
        if r["stage"] == "hist":
            assert r["count_fixed_canon"] != MISSING
            assert float(r["count_fixed_canon"]) == \
                int(float(r["count_fixed_canon"]))       # a count, not a rate
            assert float(r["tau_canon_value"]) == tau
        else:
            assert r["count_fixed_canon"] == MISSING, r["stage"]
            assert r["tau_canon_value"] == MISSING
    assert fdiag.canon_defines_stage("hist")
    assert fdiag.canon_defines_stage(HIST_STAGE)
    assert not fdiag.canon_defines_stage("s11_out")
    assert not fdiag.canon_defines_stage("s12_post_norm")


def test_every_row_carries_the_provenance_columns(saga_model, tmp_path,
                                                  monkeypatch):
    _sweep(saga_model, monkeypatch, tmp_path, n_images=4)
    rows = _diag_rows(tmp_path)
    assert set(rows[0]) == set(rec.DIAG_STAGE_COLUMNS)
    for col in rec.PROVENANCE_COLUMNS:
        assert col in rows[0], col
    for r in rows:
        assert r["ckpt_sha256"] == "a" * 64
        assert r["split_sha256"] == "b" * 64
        assert r["split_name"] == "calibration"
        assert r["git_sha"] == "c" * 40
        assert r["precision"] == "fp32"
        assert r["n_prefix"] == "1"
        assert r["n_patches"] == str(N_PATCHES)
        assert r["image_id"].startswith("n0")
        # the declared stage name AND what it resolves to, on every row
        assert r["stage_resolved"] == (HIST_STAGE if r["stage"] == "hist"
                                       else r["stage"])
    records = _rows(tmp_path, "records")
    assert set(records[0]) == set(rec.RECORD_COLUMNS)
    assert {r["stage"] for r in records} == {"hist"}


def test_the_register_model_runs_native_with_the_right_patch_count(
        reg4_model, tmp_path, monkeypatch):
    """5 prefix tokens, not 1. A patch intervention that slid a row would be
    invisible on a CLS-only model and would invalidate every register
    comparison (plan §13.3)."""
    assert infer_num_prefix_tokens(reg4_model) == 5
    row = _row(run_id="fake_registers", variant="registers",
               gate_mode=MISSING)
    summary = _sweep(reg4_model, monkeypatch, tmp_path, row=row, n_images=4)

    # a register model has no gate: only `native` applies
    assert list(summary["conditions"]) == ["native"]
    assert summary["skipped_conditions"] == sorted(
        f"term_{v:.2f}" for v in TERMINAL_GATE_VALUES)
    assert summary["n_records"] == 4
    assert summary["n_diag"] == 12                       # 3 stages x 4 images

    rows = _diag_rows(tmp_path)
    for r in rows:
        assert r["n_prefix"] == "5"
        assert r["n_patches"] == str(N_PATCHES)
        assert r["variant"] == "registers"


def test_maps_npz_has_one_count_vector_per_condition_stage_basis(
        saga_model, tmp_path, monkeypatch):
    _sweep(saga_model, monkeypatch, tmp_path, n_images=4)
    with np.load(tmp_path / "maps.npz", allow_pickle=False) as z:
        keys = sorted(k for k in z.files
                      if "|" in k and not k.startswith("n_images__"))
        assert len(keys) == 11 * len(fdiag.MAP_BASES)
        for key in keys:
            cond, stage, basis = key.split("|")
            assert basis in fdiag.MAP_BASES
            assert z[key].shape == (N_PATCHES,)
            assert z[key].dtype == np.int64
            assert int(z[f"n_images__{key}"]) == 4
            # a per-position count over 4 images can never exceed 4
            assert z[key].max() <= 4
            assert z[key].min() >= 0
        meta = json.loads(str(z["meta_json"]))
        assert meta["split_sha256"] == "b" * 64
        assert meta["ckpt_sha256"] == "a" * 64
        assert meta["hist_stage"] == HIST_STAGE

    # the aggregated map must equal the per-image counts it came from
    rows = _diag_rows(tmp_path)
    with np.load(tmp_path / "maps.npz", allow_pickle=False) as z:
        for (cond, stage) in {("native", "hist"), ("term_1.00", "hist")}:
            total = sum(float(r["count_mad"]) for r in rows
                        if r["condition_id"] == cond and r["stage"] == stage)
            assert int(z[f"{cond}|{stage}|mad"].sum()) == int(total)


def test_a_resubmitted_sweep_adds_nothing_and_rewrites_nothing(
        saga_model, tmp_path, monkeypatch):
    first = _sweep(saga_model, monkeypatch, tmp_path, n_images=4)
    before = (rec.records_path(tmp_path, "diag").read_bytes(),
              (tmp_path / "maps.npz").read_bytes())
    second = _sweep(saga_model, monkeypatch, tmp_path, n_images=4)
    assert first["n_diag"] == 44 and second["n_diag"] == 0
    assert second["n_records"] == 0
    assert rec.records_path(tmp_path, "diag").read_bytes() == before[0]
    assert not list(tmp_path.glob("*.tmp*"))
    assert rec.is_done(tmp_path, "records", "a" * 64)
    assert rec.is_done(tmp_path, "diag", "a" * 64)


def test_a_stage_with_no_calibrated_tau_is_refused(saga_model, tmp_path,
                                                   monkeypatch):
    """An uncalibrated stage is a configuration error, not a MISSING value."""
    doc = _tau_doc()
    del doc["tau_cal"]["vit_small|mixup"]["s11_out"]
    with pytest.raises(frunner.RunnerError, match="no calibrated tau"):
        _sweep(saga_model, monkeypatch, tmp_path, n_images=2, tau_doc=doc)


def test_thresholds_from_another_split_are_refused(tmp_path):
    path = tmp_path / "thresholds_cal.json"
    fdiag.write_thresholds_cal(path, {"split_sha256": "b" * 64,
                                      "tau_cal": {}})
    assert fdiag.load_thresholds_cal(path, split_sha256="b" * 64)
    with pytest.raises(fdiag.DiagError, match="another split"):
        fdiag.load_thresholds_cal(path, split_sha256="z" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# The canon thresholds are historical: read-only, and pinned
# ─────────────────────────────────────────────────────────────────────────────

def test_the_canon_thresholds_file_is_pinned():
    """`results/diagsplit/fixed_thresholds_canon.json` is never recalibrated
    (§10). The digest is taken over LF-NORMALISED bytes: this repo is checked
    out on Windows locally and on Linux on the cluster, so the raw bytes of a
    committed JSON differ by platform while its CONTENT does not."""
    assert fdiag.canon_sha256(CANON) == CANON_SHA256_LF
    doc = fdiag.load_canon_thresholds(CANON)
    assert {c: doc[c] for c in CANON_TAU} == CANON_TAU
    assert doc["k"] == fdiag.MAD_K == 5.0
    # the designations I2 calibrates from, and the checkpoints behind them
    assert doc["source_run"]["vit_small|mixup"] == "e2r_vits_mixup_baseline_s1"
    assert doc["source_run"]["vit_base|mixup"] == "e2r_vitb_mixup_baseline_s1"
    assert doc["source_run"]["vit_small|nomix"] == "e2r_vits_nomix_baseline_s1"


def test_the_threshold_writer_refuses_the_canon_file(tmp_path):
    with pytest.raises(fdiag.DiagError, match="historical calibration"):
        fdiag.write_thresholds_cal(CANON, {"tau_cal": {}})
    # by NAME too, so a relative path or a copy under another directory
    # cannot slip through
    with pytest.raises(fdiag.DiagError, match="historical calibration"):
        fdiag.write_thresholds_cal(tmp_path / CANON.name, {"tau_cal": {}})
    before = CANON.read_bytes()
    fdiag.write_thresholds_cal(tmp_path / "thresholds_cal.json",
                               {"tau_cal": {}, "split_sha256": "b" * 64})
    assert CANON.read_bytes() == before


def test_no_new_I2_file_opens_the_canon_file_for_writing():
    """Checked on the parsed AST, not on the text: every one of these files
    NAMES the canon path, and a substring search would flag the very lines
    that read it."""
    for path in I2_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or \
                getattr(node.func, "attr", None)
            if name not in ("open", "write_text", "write_bytes", "savez",
                            "savez_compressed", "dump"):
                continue
            literals = [a.value for a in node.args
                        if isinstance(a, ast.Constant)
                        and isinstance(a.value, str)]
            literals += [kw.value.value for kw in node.keywords
                         if isinstance(kw.value, ast.Constant)
                         and isinstance(kw.value.value, str)]
            assert not any("fixed_thresholds_canon" in lit
                           for lit in literals), \
                f"{path.name} passes the canon path to {name}()"


def test_the_designated_baselines_come_from_the_canon_file():
    """I2 calibrates from the checkpoint the canon file itself names, so
    tau_cal[hist] and tau_canon are the same recipe on two splits. A second
    implementation of "the designated baseline" could pick another
    checkpoint and make the two incomparable."""
    from tools.frozen_I2_thresholds import designated_baselines
    cohort = frunner.eligible_cohort(MANIFEST)
    canon = fdiag.load_canon_thresholds(CANON)
    picked = designated_baselines(cohort, canon)
    assert {c: r["run_id"] for c, r in picked.items()} == canon["source_run"]
    for cell, row in picked.items():
        assert row["variant"] == "baseline"
        assert row["ckpt_sha256"] == canon["source_ckpt_sha256"][cell]

    # a canon file whose designation is not in the cohort is refused
    bad = dict(canon, source_run=dict(canon["source_run"],
                                      **{"vit_small|mixup": "nope"}))
    with pytest.raises(frunner.RunnerError, match="not an eligible cohort"):
        designated_baselines(cohort, bad)


# ─────────────────────────────────────────────────────────────────────────────
# The cohort and the array order
# ─────────────────────────────────────────────────────────────────────────────

def test_the_eligible_cohort_is_the_nineteen_of_the_handoff():
    cohort = frunner.eligible_cohort(MANIFEST)
    assert len(cohort) == 19
    counts = {}
    for r in cohort:
        counts[r["variant"]] = counts.get(r["variant"], 0) + 1
        assert r["ckpt_kind"] == "last"
        assert r["family"] in frunner.COHORT_FAMILIES
        assert r["status"] in frunner.COHORT_STATUSES
        assert r["ckpt_sha256"] != MISSING
    assert counts == {"baseline": 8, "saga": 8, "registers": 3}
    # no ablation, fine-tuned, dense or TTR row (task §10)
    assert not [r for r in cohort if r["family"] not in
                ("e2r_300ep", "legacy_300ep")]
    # baselines first: every calibration source is inside --array=0-7
    assert [r["variant"] for r in cohort[:8]] == ["baseline"] * 8
    # the order is a contract, so it is stable across calls
    assert frunner.eligible_run_ids(MANIFEST) == \
        frunner.eligible_run_ids(MANIFEST)


def test_pairing_comes_from_build_pooled_tables():
    """The gaps are paired by the provenance tags `build_pooled_tables`
    defines — the module that also carries the recipe erratum remap and the
    exclusion of the VOID legacy ViT-B mixup-dir trio."""
    from analysis.build_I2_tables import cell_tags, pairs_for
    cohort = frunner.eligible_cohort(MANIFEST)

    assert cell_tags("vit_small", "mixup") == [
        "legacy-mixupdir", "legacy-nomixdir", "s1", "s2"]
    assert cell_tags("vit_small", "nomix") == ["s1", "s2"]
    assert cell_tags("vit_base", "mixup") == ["legacy-nomixdir", "s1"]

    # task §7: the voting cell has FOUR pairs; the others have two
    assert len(pairs_for(cohort, "vit_small", "mixup", "saga")) == 4
    assert len(pairs_for(cohort, "vit_small", "nomix", "saga")) == 2
    assert len(pairs_for(cohort, "vit_base", "mixup", "saga")) == 2
    assert len(pairs_for(cohort, "vit_small", "mixup", "registers")) == 2
    assert len(pairs_for(cohort, "vit_base", "mixup", "registers")) == 1

    for tag, base, other in pairs_for(cohort, "vit_small", "mixup", "saga"):
        by_id = {r["run_id"]: r for r in cohort}
        assert by_id[base]["provenance_tag"] == by_id[other]["provenance_tag"]
        assert by_id[base]["variant"] == "baseline"
        assert by_id[other]["variant"] == "saga"

    # the VOID legacy ViT-B mixup-DIRECTORY trio is in no cell
    assert not [r for r in cohort if "mixupdir" in r["run_id"]
                and r["arch"] == "vit_base"]


# ─────────────────────────────────────────────────────────────────────────────
# The tables
# ─────────────────────────────────────────────────────────────────────────────

CELL_RUNS = {
    "legacy-mixupdir": ("legacy_e2_vit_small_mixupdir_baseline",
                        "legacy_e2_vit_small_mixupdir_saga",
                        "legacy_e2_vit_small_mixupdir_registers"),
    "legacy-nomixdir": ("legacy_e2_vit_small_nomixdir_baseline",
                        "legacy_e2_vit_small_nomixdir_saga",
                        "legacy_e2_vit_small_nomixdir_registers"),
    "s1": ("e2r_vits_mixup_baseline_s1", "e2r_vits_mixup_saga_s1", None),
    "s2": ("e2r_vits_mixup_baseline_s2", "e2r_vits_mixup_saga_s2", None),
}

FAKE_SPLIT_SHA = "b" * 64
N_FAKE_IMAGES = 6


#: A fixed per-position exceedance-count pattern. Every fake map is a
#: PERMUTATION of it, so the total mass is identical everywhere and
#: `concentration_null` — 400 Binomial simulations per distinct
#: (mass, n_images, n_positions) — is computed once and cached, while the
#: spatial arrangement still differs between runs.
MAP_PATTERN = np.concatenate([
    np.full(8, 4), np.full(20, 2), np.full(40, 1),
    np.zeros(N_PATCHES - 68, dtype=int)]).astype(np.int64)


def _bool_map_with_counts(counts, n_images):
    """[n_images, N] boolean whose column sums are exactly `counts`."""
    out = np.zeros((n_images, counts.size), dtype=bool)
    for j, c in enumerate(counts):
        out[:int(c), j] = True
    return out


def _fake_run(root, row, *, shift, rng):
    """Write one run's records / diag / maps with controlled diagnostics.

    `shift` is added to every diagnostic at every stage, so a pair's gap is
    known in advance and the decision rule can be driven to a chosen branch.
    """
    out = Path(root) / row["run_id"]
    stages = {"native": ("s11_out", "hist", "s12_post_norm")}
    if row["variant"] == "saga":
        for v in TERMINAL_GATE_VALUES:
            stages[f"term_{v:.2f}"] = ("hist", "s12_post_norm")

    base = rec.provenance(
        run_id=row["run_id"], arch=row["arch"],
        recipe_actual=row["recipe_actual"], variant=row["variant"],
        ckpt_kind="last", ckpt_sha256=row["ckpt_sha256"], n_prefix=1,
        stage="hist", split_name="calibration", split_sha256=FAKE_SPLIT_SHA,
        precision="fp32", git_sha="c" * 40, git_dirty=0)

    ids = [f"n{c:08d}/IMG_{c}" for c in range(N_FAKE_IMAGES)]
    records, diags = [], []
    maps = fdiag.MapAccumulator()
    for cond, cond_stages in stages.items():
        for i, img in enumerate(ids):
            records.append(dict(
                base, image_id=img, condition_id=cond, edit_type="native",
                edit_params="{}", layer=MISSING, epsilon=MISSING,
                measured_perturbation_norm=MISSING,
                nll=1.0 + 0.01 * i, correct=1, top1=i % 10, label=i % 10,
                # native and term_0.50 are the SAME pass on a phi_L = 0
                # checkpoint; the other constants move the patch rows but
                # not the CLS logits (Proposition 2), which is why this
                # column is tiny rather than zero for them
                max_abs_logit_diff_vs_native=(
                    0.0 if cond in ("native", "term_0.50") else 1e-7 * i)))
        for stage in cond_stages:
            bump = 0.0 if cond in ("native", "term_0.50") else 0.5
            for i, img in enumerate(ids):
                v = shift + bump + 0.1 * i
                diags.append(dict(
                    base, image_id=img, condition_id=cond, stage=stage,
                    stage_resolved=HIST_STAGE if stage == "hist" else stage,
                    edit_type="native", n_patches=N_PATCHES,
                    tau_cal_value=1.5,
                    tau_canon_value=(str(CANON_TAU["vit_small|mixup"])
                                     if stage == "hist" else MISSING),
                    count_fixed_canon=(str(int(10 + shift * 10))
                                       if stage == "hist" else MISSING),
                    norm_p50=2.0 + v, norm_p90=3.0 + v, norm_p99=4.0 + v,
                    norm_p999=5.0 + v, norm_max=6.0 + v, mad_thr=1.2 + v,
                    count_fixed_cal=5.0 + v, count_mad=3.0 + v,
                    cos_all=0.1 + v, cos_nosink_mad=0.05 + v,
                    eff_rank=20.0 + v))
            counts = rng.permutation(MAP_PATTERN)
            m = _bool_map_with_counts(counts, N_FAKE_IMAGES)
            maps.add(cond, stage, {"mad": m, "fixed_cal": m.copy()})

    rec.append_rows(out, "records", records)
    rec.append_rows(out, "diag", diags, key=rec.DIAG_STAGE_KEY,
                    columns=rec.DIAG_STAGE_COLUMNS)
    fdiag.write_maps_npz(out, maps, {"run_id": row["run_id"]})
    rec.mark_done(out, "records", {"ckpt_sha256": row["ckpt_sha256"]})
    rec.mark_done(out, "diag", {"ckpt_sha256": row["ckpt_sha256"]})
    return out


@pytest.fixture(scope="module")
def fake_results(tmp_path_factory):
    """A whole ViT-S/mixup cell on disk: 4 baseline/SAGA pairs and the two
    register repeats, with a KNOWN SAGA-minus-baseline gap of +1.0 on every
    diagnostic at every stage under `native`."""
    root = tmp_path_factory.mktemp("I2_terminal")
    by_id = {r["run_id"]: r for r in frunner.eligible_cohort(MANIFEST)}
    rng = np.random.default_rng(0)
    for tag, (base_id, saga_id, reg_id) in CELL_RUNS.items():
        _fake_run(root, by_id[base_id], shift=0.0, rng=rng)
        _fake_run(root, by_id[saga_id], shift=1.0, rng=rng)
        if reg_id:
            _fake_run(root, by_id[reg_id], shift=0.5, rng=rng)
    return root


@pytest.fixture(scope="module")
def built_tables(fake_results, tmp_path_factory):
    from analysis.build_I2_tables import build_all
    out = tmp_path_factory.mktemp("tables")
    summary = build_all(fake_results, out, manifest=MANIFEST,
                        git_sha_value="c" * 40, resamples=200)
    return out, summary


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_tables_are_byte_identical_on_a_rebuild(fake_results,
                                                    tmp_path_factory):
    """A generated table is REGENERATED, never hand-merged (I0 handoff §8.6),
    so two builds from the same inputs must produce the same bytes — no
    timestamp in a CSV, no dict-order row output, and a bootstrap whose seed
    and chunk size fully determine its resamples."""
    from analysis.build_I2_tables import build_all
    a = tmp_path_factory.mktemp("tables_a")
    b = tmp_path_factory.mktemp("tables_b")
    for out in (a, b):
        build_all(fake_results, out, manifest=MANIFEST,
                  git_sha_value="c" * 40, resamples=200)
    names = sorted(p.name for p in a.glob("*.csv"))
    assert names == ["T_I2a_invariance.csv", "T_I2b_sweep.csv",
                     "T_I2c_gaps.csv", "T_I2d_maps.csv",
                     "T_I2e_registers.csv"]
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
        assert b"\r\n" not in (a / name).read_bytes(), name


def test_every_table_carries_the_provenance_columns(built_tables):
    from analysis.build_I2_tables import PROV_FIELDS
    out, _ = built_tables
    for path in sorted(out.glob("*.csv")):
        rows = _read(path)
        assert rows, path.name
        for col in PROV_FIELDS:
            assert col in rows[0], f"{path.name} is missing {col}"
        for r in rows:
            assert r["work_package"] == "I2_terminal"
            assert r["split_sha256"] == FAKE_SPLIT_SHA
            assert r["split_name"] == "calibration"
            assert r["git_sha"] == "c" * 40


def test_the_invariance_table_flags_the_phi_L_zero_equality(built_tables):
    out, _ = built_tables
    rows = _read(out / "T_I2a_invariance.csv")
    assert {r["variant"] for r in rows} == {"saga"}
    assert {r["condition_id"] for r in rows} == {
        f"term_{v:.2f}" for v in TERMINAL_GATE_VALUES}
    for r in rows:
        assert int(r["n_images"]) == N_FAKE_IMAGES
        assert 0.0 <= float(r["top1_agreement"]) <= 1.0
        if r["condition_id"] == "term_0.50":
            # the fake sweep writes term_0.50 identical to native
            assert r["phi_L_zero_bit_identical"] == "1"
            assert float(r["max_abs_cos_all_diff_hist"]) == 0.0
        else:
            assert r["phi_L_zero_bit_identical"] == MISSING


def test_the_sweep_table_reports_canon_counts_only_at_hist(built_tables):
    out, _ = built_tables
    rows = _read(out / "T_I2b_sweep.csv")
    assert {r["stage"] for r in rows} == {"s11_out", "hist", "s12_post_norm"}
    for r in rows:
        if r["stage"] == "hist":
            assert r["count_fixed_canon"] != MISSING
        else:
            assert r["count_fixed_canon"] == MISSING, r["stage"]
    # all 19 cohort rows are eligible for this table; the fixture wrote 10
    assert len({r["run_id"] for r in rows}) == 10
    assert {r["variant"] for r in rows} == {"baseline", "saga", "registers"}


def test_the_gaps_table_recovers_the_planted_gap(built_tables):
    """The fixture plants SAGA = baseline + 1.0 under `native` at every
    stage, and + 1.5 under every constant except term_0.50."""
    out, _ = built_tables
    rows = [r for r in _read(out / "T_I2c_gaps.csv")
            if r["cell"] == "vit_small|mixup"]
    by = {(r["setting"], r["diagnostic"]): r for r in rows}
    assert {r["setting"] for r in rows} == {"hist_native", "hist_term_1.00",
                                            "s11_out_native"}
    for d in ("cos_all", "eff_rank", "count_mad", "count_fixed_cal",
              "cos_nosink_mad"):
        assert float(by[("hist_native", d)]["gap_mean"]) == pytest.approx(1.0)
        assert float(by[("s11_out_native", d)]["gap_mean"]) == \
            pytest.approx(1.0)
        assert float(by[("hist_term_1.00", d)]["gap_mean"]) == \
            pytest.approx(1.5)
        assert int(by[("hist_native", d)]["n_pairs"]) == 4
        assert int(by[("hist_native", d)]["n_images"]) == N_FAKE_IMAGES
        lo = float(by[("hist_native", d)]["ci_lo"])
        hi = float(by[("hist_native", d)]["ci_hi"])
        assert lo <= 1.0 <= hi
        assert by[("hist_native", d)]["per_pair"].count("=") == 4
        assert float(by[("hist_term_1.00", d)]["survival"]) == \
            pytest.approx(1.5)
        assert float(by[("s11_out_native", d)]["survival_s11"]) == \
            pytest.approx(1.0)
    # count_fixed_canon is MISSING at s11_out and must not become a gap there
    assert by[("s11_out_native", "count_fixed_canon")]["gap_mean"] == MISSING
    assert by[("hist_native", "count_fixed_canon")]["gap_mean"] != MISSING


def test_the_maps_table_uses_the_existing_spatial_reference(built_tables):
    out, _ = built_tables
    rows = _read(out / "T_I2d_maps.csv")
    assert rows
    for r in rows:
        assert r["map_basis"] in fdiag.MAP_BASES
        assert int(r["n_positions"]) == N_PATCHES
        # the exact 8 x side^2 dihedral/torus-roll set of
        # analysis.address_analysis.build_perm on a 14x14 grid — fixed and
        # complete, so the p-value needs no seed and can never be 0
        assert int(r["n_transforms"]) == 8 * N_PATCHES
        p = float(r["p_spatial"])
        assert 1.0 / (8 * N_PATCHES) <= p <= 1.0
        assert -1.0 <= float(r["rho_spatial"]) <= 1.0
        assert 0.0 <= float(r["ring1_share_variant"]) <= 1.0
        assert float(r["mass_variant"]) > 0
        # the excess is over the FINITE-SAMPLE null, not the raw statistic
        assert r["excess_gini_variant"] != MISSING
    assert {r["condition_id"] for r in rows} == {"native"} | {
        f"term_{v:.2f}" for v in TERMINAL_GATE_VALUES}
    assert {r["stage"] for r in rows} == {"s11_out", "hist", "s12_post_norm"}


def test_the_registers_table_covers_s11_out_and_hist_only(built_tables):
    out, _ = built_tables
    rows = [r for r in _read(out / "T_I2e_registers.csv")
            if r["cell"] == "vit_small|mixup"]
    assert {r["stage"] for r in rows} == {"s11_out", "hist"}
    assert {r["condition_id"] for r in rows} == {"native"}
    assert {r["variant"] for r in rows} == {"registers"}
    for r in rows:
        if r["diagnostic"] != "count_fixed_canon" or r["stage"] == "hist":
            assert int(r["n_pairs"]) == 2, r          # n is STATED (task §6)
    gap = [r for r in rows if r["diagnostic"] == "cos_all"
           and r["stage"] == "hist"][0]
    assert float(gap["gap_mean"]) == pytest.approx(0.5)


def test_the_bootstrap_is_the_locked_analysis_one():
    from analysis.build_I2_tables import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
                                          paired_bootstrap)
    assert BOOTSTRAP_RESAMPLES == 10000          # LOCKED_ANALYSIS §8
    assert BOOTSTRAP_SEED == 0                   # D6's stated default
    a = np.arange(20, dtype=float)
    first = paired_bootstrap([a, a + 2.0], resamples=300, seed=0)
    again = paired_bootstrap([a, a + 2.0], resamples=300, seed=0)
    other = paired_bootstrap([a, a + 2.0], resamples=300, seed=1)
    assert first == again                        # seeded, reproducible
    assert first[1:] != other[1:]                # and the seed actually moves
    assert first[0] == pytest.approx(a.mean() + 1.0)
    assert first[1] < first[0] < first[2]


def test_a_pair_whose_image_ids_differ_is_dropped_not_aligned(fake_results,
                                                              tmp_path):
    """Two splits that happen to be the same length are not the same split."""
    from analysis.build_I2_tables import _paired_deltas, load_run
    loaded = {}
    for base_id, saga_id, _ in [CELL_RUNS["s1"]]:
        loaded[base_id] = load_run(fake_results, base_id)
        loaded[saga_id] = load_run(fake_results, saga_id)
    pairs = [("s1", CELL_RUNS["s1"][0], CELL_RUNS["s1"][1])]
    got, dropped = _paired_deltas(loaded, pairs, "hist", "native", "cos_all")
    assert len(got) == 1 and not dropped

    g = loaded[CELL_RUNS["s1"][1]][1][("native", "hist")]
    g["image_ids"] = np.array([f"other/{i}" for i in range(N_FAKE_IMAGES)])
    got, dropped = _paired_deltas(loaded, pairs, "hist", "native", "cos_all")
    assert got == [] and dropped == ["s1:image_ids_differ"]


# ─────────────────────────────────────────────────────────────────────────────
# The decision rule (§7) — all three branches, on synthetic gaps
# ─────────────────────────────────────────────────────────────────────────────

def _gaps(native, term, s11):
    from analysis.i2_decision import (PRIMARY_DIAGNOSTICS, SETTING_NATIVE,
                                      SETTING_S11, SETTING_TERM)
    return {d: {SETTING_NATIVE: native, SETTING_TERM: term, SETTING_S11: s11}
            for d in PRIMARY_DIAGNOSTICS}


def test_branch_1_terminal_gate_is_a_minor_contributor():
    from analysis.i2_decision import BRANCH_MINOR, decide
    v = decide(_gaps(1.0, 0.8, 0.2), n_pairs=4)
    assert v["branch"] == BRANCH_MINOR
    assert v["d1_stage"] == "hist"
    assert all(s == pytest.approx(0.8) for s in v["survival"].values())
    assert v["failing_diagnostics_term"] == []
    assert "D1 = `hist`" in v["sentence"]
    assert "minor contributor" in v["sentence"]
    # exactly at the cutoff is INSIDE the branch
    assert decide(_gaps(1.0, 0.70, 0.0), n_pairs=4)["branch"] == BRANCH_MINOR
    # a sign flip fails it however large the ratio
    assert decide(_gaps(1.0, -2.0, 0.9), n_pairs=4)["branch"] != BRANCH_MINOR


def test_branch_2_report_at_s11_out():
    from analysis.i2_decision import BRANCH_STAGE_SHIFT, decide
    v = decide(_gaps(1.0, 0.2, 0.9), n_pairs=4)
    assert v["branch"] == BRANCH_STAGE_SHIFT
    assert v["d1_stage"] == "s11_out"
    assert v["failing_diagnostics_term"] == sorted(v["survival"])
    assert v["failing_diagnostics_s11"] == []
    assert "D1 = `s11_out`" in v["sentence"]
    assert "readout-contaminated control" in v["sentence"]


def test_branch_3_I2_is_a_main_finding():
    from analysis.i2_decision import BRANCH_MAIN_FINDING, decide
    v = decide(_gaps(1.0, 0.1, 0.1), n_pairs=4)
    assert v["branch"] == BRANCH_MAIN_FINDING
    assert v["d1_stage"] == "s11_out"
    assert "main finding" in v["sentence"]
    assert "substantially a terminal-gate effect" in v["sentence"]


def test_one_failing_primary_diagnostic_is_enough_to_leave_branch_1():
    """`for EVERY primary diagnostic` is a conjunction, not a majority."""
    from analysis.i2_decision import BRANCH_MINOR, decide
    gaps = _gaps(1.0, 0.9, 0.95)
    gaps["eff_rank"]["hist_term_1.00"] = 0.69
    v = decide(gaps, n_pairs=4)
    assert v["branch"] != BRANCH_MINOR
    assert v["failing_diagnostics_term"] == ["eff_rank"]


def test_a_missing_or_zero_gap_cannot_certify_a_branch():
    """Survival is MISSING when the denominator is 0 or either side is
    MISSING, and a MISSING survival fails its test rather than being
    dropped from the conjunction."""
    from analysis.i2_decision import (BRANCH_MAIN_FINDING, MISSING as DM,
                                      decide, survival_of)
    assert survival_of(0.0, 1.0) == DM
    assert survival_of(DM, 1.0) == DM
    assert survival_of(1.0, DM) == DM
    assert survival_of(2.0, 1.0) == 0.5

    gaps = _gaps(1.0, 0.9, 0.9)
    gaps["cos_all"]["hist_native"] = 0.0
    v = decide(gaps, n_pairs=4)
    assert v["survival"]["cos_all"] == DM
    assert v["branch"] == BRANCH_MAIN_FINDING
    assert "cos_all" in v["failing_diagnostics_term"]


def test_the_cutoff_and_the_voting_cell_are_named_constants():
    from analysis import i2_decision
    assert i2_decision.SURVIVAL_CUTOFF == 0.70
    assert i2_decision.DECISION_CELL == "vit_small|mixup"
    assert i2_decision.PRIMARY_DIAGNOSTICS == (
        "cos_all", "cos_nosink_mad", "eff_rank", "count_fixed_cal",
        "count_mad")
    # the rule needs every primary diagnostic; a short dict is an error
    with pytest.raises(i2_decision.DecisionError, match="primary diagnostic"):
        i2_decision.decide({"cos_all": {}})


def test_the_decision_reads_the_gaps_table_the_builder_writes(built_tables):
    from analysis.i2_decision import DECISION_CELL, decide, load_gaps
    out, _ = built_tables
    by_cell = load_gaps(out / "T_I2c_gaps.csv")
    assert DECISION_CELL in by_cell
    gaps, n_pairs = by_cell[DECISION_CELL]
    assert n_pairs == 4
    verdict = decide(gaps, cell=DECISION_CELL, n_pairs=n_pairs)
    # the planted gaps are native 1.0, term_1.00 1.5, s11_out 1.0 — survival
    # 1.5 with matching signs, so branch 1
    assert verdict["branch"] == 1
    assert verdict["d1_stage"] == "hist"
    assert verdict["sentence"].startswith("D1 = `hist`")


def test_the_table_builder_and_the_decision_share_one_survival():
    """Two implementations of `gap_term / gap_native` would be two numbers
    that must agree; there is one."""
    import analysis.build_I2_tables as tables
    from analysis.i2_decision import survival_of
    assert tables.survival_of is survival_of


# ─────────────────────────────────────────────────────────────────────────────
# The job file (D7)
# ─────────────────────────────────────────────────────────────────────────────

def _job():
    return JOB_FILE.read_text(encoding="utf-8")


def test_the_job_file_lists_the_cohort_in_array_order():
    """The array index must mean the same run in the job file and in the
    manifest; the job file asserts this again at runtime."""
    src = _job()
    want = frunner.eligible_run_ids(MANIFEST)
    block = src.split("RUN_IDS=(", 1)[1].split(")", 1)[0]
    listed = [line.split('"')[1] for line in block.splitlines()
              if line.strip().startswith('"')]
    assert listed == want
    assert len(listed) == 19
    assert "#SBATCH --array=0-18" in src
    assert "eligible_run_ids" in src, \
        "the job must re-check the cohort against the manifest at runtime"


def test_the_job_file_takes_the_split_as_an_argument_and_defaults_to_nothing():
    """Task §10: the evaluation split is not run before D1 is signed. A
    default would make that a habit rather than a property of the repo."""
    src = _job()
    # `${1:?...}` aborts with the usage message when no split is given
    assert 'SPLIT="${1:?' in src
    assert '--split "$SPLIT"' in src
    # the ONLY split the job ever passes is the one it was given; no
    # assignment and no flag names a split file literally
    assert "SPLIT=results/" not in src
    assert "--split results/" not in src
    for line in src.splitlines():
        if line.lstrip().startswith("#") or "evaluation.json" not in line:
            continue
        # the single permitted mention is the usage message, which TELLS the
        # human that evaluation is B2 and comes after D1 is signed
        assert "only AFTER the human has signed D1" in line, line


def test_no_code_path_defaults_to_the_evaluation_split():
    """Task §10: nothing runs on the evaluation split before D1 is signed.

    A string constant that IS a path — as opposed to prose that mentions one
    — is what could become a default, so that is what is checked: any bare
    `<path>.json` literal anywhere in the new code, the driver or the runner.
    """
    import re
    path_like = re.compile(r"^[\w./-]+\.json$")
    checked = 0
    for path in I2_FILES + [REPO / "tools" / "frozen_eval.py",
                            REPO / "saga" / "frozen" / "runner.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            if not path_like.match(node.value.strip()):
                continue
            checked += 1
            assert "evaluation.json" not in node.value, \
                f"{path.name} names evaluation.json as a path literal"
    assert checked, "the path-literal scan matched nothing — it is not working"


def test_the_job_file_follows_the_established_header_and_ordering():
    src = _job()
    raw = JOB_FILE.read_bytes()
    assert raw.startswith(b"#!/bin/bash")
    assert b"\r\n" not in raw, "a CRLF sbatch fails on the HPC"
    assert b"\nset -u\n" in raw

    for line in ["#SBATCH --partition=a100", "#SBATCH --gres=gpu:a100:1",
                 "#SBATCH --cpus-per-task=4"]:
        assert line in src, line
    assert "#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/" in src
    assert "#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/" in src
    assert ("source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh"
            in src)
    assert "source $CODE_ROOT/scripts/stage_probe.sh" in src
    assert '--data "$PROBE_IN_DIR"' in src

    lines = src.splitlines()

    def first(pred):
        return next((i for i, l in enumerate(lines) if pred(l)), None)

    i_src = first(lambda l: l.strip().startswith("source ")
                  and "env_alex.sh" in l)
    i_setu = first(lambda l: l.strip() == "set -u")
    i_exec = first(lambda l: l.strip().startswith('if [ ! -x "$PY" ]'))
    i_torch = first(lambda l: l.startswith("if ! $PY -c")
                    and "import torch" in l)
    i_arrow = first(lambda l: l.startswith("if ! $PY -c")
                    and "import pyarrow" in l)
    i_trap = first(lambda l: l.strip() == "trap cleanup_probe EXIT")
    i_stage = first(lambda l: l.strip() == "stage_probe_imagenet_val")
    i_thr = first(lambda l: "tools/frozen_I2_thresholds.py" in l)
    i_eval = first(lambda l: "tools/frozen_eval.py" in l)

    # `set -u` after the env sourcing; preflights before the trap and the
    # staging; thresholds before the sweep that uses them
    assert i_src < i_setu < i_exec < i_torch < i_arrow < i_trap < i_stage
    assert i_stage < i_thr < i_eval
    assert src.count("cleanup_probe") == 1
    for i in (i_exec, i_torch, i_arrow):
        assert "exit 1" in "\n".join(lines[i:i + 10])


def test_the_job_file_is_resubmit_safe():
    src = _job()
    assert "--if-missing" in src          # thresholds: a no-op once written
    assert "--skip-if-done" in src        # sweep: a no-op once complete
    assert "configs/frozen/I2_terminal.yaml" in src
    # nothing scientific on the command line: no stage, no gate value, no tau
    for forbidden in ("--stage", "--fixed-thr", "--effrank", "--condition ",
                      "terminal_gate_override"):
        assert forbidden not in src, forbidden


# ─────────────────────────────────────────────────────────────────────────────
# No training, anywhere in what I2 added
# ─────────────────────────────────────────────────────────────────────────────

I2_FILES = [
    REPO / "saga" / "frozen" / "diag.py",
    REPO / "tools" / "frozen_I2_thresholds.py",
    REPO / "analysis" / "build_I2_tables.py",
    REPO / "analysis" / "i2_decision.py",
    REPO / "analysis" / "build_D1_proposal.py",
]


def test_no_optimizer_or_training_import_in_the_new_I2_files():
    """TASK I0 handoff §8.7, extended to this work package's own files.
    Checked on the parsed AST: these files say "no optimizer" in their own
    docstrings and a substring search would flag the rule itself."""
    banned_modules = ("torch.optim", "torch.optim.lr_scheduler")
    banned_names = {"AdamW", "Adam", "SGD", "backward", "step",
                    "LabelSmoothingCrossEntropy", "Mixup"}
    assert len(I2_FILES) == 5
    for path in I2_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith("torch.optim"), \
                        f"{path.name} imports {a.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod not in banned_modules, \
                    f"{path.name} imports from {mod}"
                for a in node.names:
                    assert a.name not in banned_names, \
                        f"{path.name} imports {a.name} from {mod}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in ("backward", "zero_grad"), \
                    f"{path.name} calls .{node.attr}"


# ─────────────────────────────────────────────────────────────────────────────
# One split, one directory — the B2 collision that nearly happened
# ─────────────────────────────────────────────────────────────────────────────

def test_a_sweeps_output_directory_names_its_split():
    """Two splits must not share a run directory.

    They did until B2 was about to be submitted: `out_dir` was
    `<out_root>/<work_package>/<run_id>`, so an evaluation sweep would have
    appended its images into the calibration `records.parquet`, and
    `--skip-if-done` — which keys the completion marker on the CHECKPOINT
    sha, not the split — would have exited 0 for all 19 runs having done
    nothing at all. Neither failure is loud.
    """
    src = (REPO / "tools" / "frozen_eval.py").read_text(encoding="utf-8")
    assert ('out_dir = (Path(args.out_root) / conditions["work_package"] '
            "/ split_name" in src)
    # and the committed calibration results already live under that layout
    assert (RESULTS / "calibration").is_dir()
    assert not any(p.is_dir() and p.name not in ("calibration", "evaluation")
                   for p in RESULTS.iterdir())


def test_the_thresholds_path_is_per_split_and_refuses_a_fixed_one():
    got = fdiag.thresholds_cal_path(
        "results/frozen/I2_terminal/{split_name}/thresholds_cal.json",
        "evaluation")
    assert got == Path(
        "results/frozen/I2_terminal/evaluation/thresholds_cal.json")
    with pytest.raises(fdiag.DiagError, match="split_name"):
        fdiag.thresholds_cal_path("results/frozen/I2_terminal/t.json", "x")


def test_the_evaluation_records_are_git_ignored_and_calibration_is_not():
    """B2's raw per-image records stay on the cluster; B1's stay in git."""
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    for pat in ("results/frozen/I2_terminal/evaluation/*/records.parquet",
                "results/frozen/I2_terminal/evaluation/*/diag.parquet"):
        assert pat in ignore, pat
    assert "results/frozen/I2_terminal/calibration/" not in ignore
    # ... and what IS committed for evaluation must not be ignored by accident
    for keep in ("maps.npz", "run_meta.json", "records.done.json",
                 "thresholds_cal.json"):
        assert f"evaluation/*/{keep}" not in ignore, keep


def test_run_meta_records_the_digest_of_every_file_the_run_wrote():
    """A records file too large to commit must still be identifiable: the
    committed run_meta.json is then the only thing that says which bytes
    produced the tables."""
    from tools.frozen_eval import output_digests
    run = RESULTS / "calibration" / "e2r_vits_mixup_saga_s1"
    got = output_digests(run)
    assert "run_meta.json" not in got
    for name in ("maps.npz", "records.done.json", "diag.done.json"):
        assert name in got, name
        assert len(got[name]["sha256"]) == 64
        assert got[name]["bytes"] > 0
    from saga.frozen.records import records_path
    assert records_path(run, "diag").name in got


def test_the_primary_pack_covers_every_run_condition_stage(tmp_path):
    """figures_data/frozen/I2_evaluation_primary.npz is what makes an
    evaluation number regenerable once the raw parquet stays on the HPC."""
    from analysis.i2_decision import PRIMARY_DIAGNOSTICS
    from tools.frozen_I2_pack_primary import pack
    src = RESULTS / "calibration"
    if not (src / "e2r_vits_mixup_saga_s1").exists():     # pragma: no cover
        pytest.skip("the calibration sweep is not in this checkout")
    arrays, ids, meta = pack(src, MANIFEST)
    # 8 SAGA runs x 11 (condition, stage) + 11 others x 3, times 5 diagnostics
    assert len(arrays) == (8 * 11 + 11 * 3) * len(PRIMARY_DIAGNOSTICS) == 605
    assert len(meta) == 19
    for key, arr in arrays.items():
        run_id, cond, stage, diag = key.split("|")
        assert diag in PRIMARY_DIAGNOSTICS
        assert arr.dtype == np.float32
        assert arr.shape == ids.shape
    assert np.array_equal(ids, np.sort(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Phase C1 — the committed tables and the generated D1 proposal
# ─────────────────────────────────────────────────────────────────────────────

RESULTS = REPO / "results" / "frozen" / "I2_terminal"
CALIBRATION = RESULTS / "calibration"
TABLE_DIR = CALIBRATION / "tables"
PROPOSAL = RESULTS / "D1_proposal.md"

needs_results = pytest.mark.skipif(
    not (TABLE_DIR / "T_I2c_gaps.csv").exists(),
    reason="the I2 sweep results / tables are not present in this checkout")


@needs_results
def test_the_committed_proposal_is_what_the_generator_produces():
    """A generated file under results/ is REGENERATED, never hand-merged
    (I0 handoff §8.6). Two builds must agree with each other AND with what
    is committed — so a hand edit to the note shows up here as a failure."""
    from analysis.build_D1_proposal import build
    once = build(TABLE_DIR, CALIBRATION / "thresholds_cal.json")
    twice = build(TABLE_DIR, CALIBRATION / "thresholds_cal.json")
    assert once == twice, "the generator is not deterministic"
    assert PROPOSAL.read_text(encoding="utf-8") == once, (
        "results/frozen/I2_terminal/D1_proposal.md differs from what "
        "analysis/build_D1_proposal.py produces — regenerate it, never edit "
        "it by hand")


@needs_results
def test_no_generated_markdown_table_row_contains_an_unescaped_pipe():
    """A literal `|` inside a cell silently breaks its row, and every cell
    name in this project is `<arch>|<recipe_actual>`. TASK I0 Phase C hit
    this twice; it cannot come back."""
    text = PROPOSAL.read_text(encoding="utf-8")
    rows = [l for l in text.splitlines()
            if l.startswith("|") and l.rstrip().endswith("|")]
    assert rows, "no markdown table rows found in the proposal"
    widths = {}
    block = 0
    prev_was_row = False
    for line in text.splitlines():
        is_row = line.startswith("|") and line.rstrip().endswith("|")
        if is_row and not prev_was_row:
            block += 1
        prev_was_row = is_row
        if not is_row:
            continue
        # count only the SEPARATORS: an escaped pipe is content
        n = line.replace("\\|", "\x00").count("|") - 1
        widths.setdefault(block, []).append((n, line))
    for cols in widths.values():
        counts = {n for n, _ in cols}
        assert len(counts) == 1, (
            "a table block has rows of differing column counts, which means "
            "an unescaped `|`:\n  "
            + "\n  ".join(f"{n}: {l}" for n, l in cols))


@needs_results
def test_the_proposal_quotes_the_rule_verbatim_whatever_was_signed():
    """The signed wording is the §3(b) alternative, but the RULE's own
    sentence must still appear in the note unedited — the verdict is the
    rule's, and a reader has to be able to see what it actually said."""
    from analysis.i2_decision import DECISION_CELL, decide, load_gaps
    text = PROPOSAL.read_text(encoding="utf-8")
    gaps, n_pairs = load_gaps(TABLE_DIR / "T_I2c_gaps.csv")[DECISION_CELL]
    v = decide(gaps, cell=DECISION_CELL, n_pairs=n_pairs)
    assert "> " + v["sentence"] in text, \
        "the note must quote the rule's sentence verbatim, not a paraphrase"
    assert f"D1 = `{v['d1_stage']}`" in text
    assert "conventional" in text
    assert f"{v['cutoff']:.2f}" in text
    # the disclosure that the signed wording departs from the boilerplate
    assert "overstates what these numbers show" in text


@needs_results
def test_the_signature_lives_in_LOCKED_ANALYSIS_and_the_note_reads_it_back():
    """The note is GENERATED, so it cannot hold a signature of its own: a
    hand-edited status block would be overwritten by the next build and the
    byte-identity test would fail. §1 of LOCKED_ANALYSIS is where the
    signature lives, and `read_signature` is how the note gets it."""
    from analysis.build_D1_proposal import read_signature
    sig = read_signature(REPO / "docs" / "LOCKED_ANALYSIS.md")
    assert sig is not None, "D1 has no signature line in LOCKED_ANALYSIS §1"
    text = PROPOSAL.read_text(encoding="utf-8")
    assert "**SIGNED**" in text
    assert sig["name"] in text and sig["date"] in text
    assert f"`{sig['sha']}`" in text
    assert "`PENDING`" not in text


@needs_results
def test_D1_is_closed_and_the_document_is_still_a_draft():
    """Closing a decision and freezing the document are separate acts: D1
    (and D2/D3/D4/D6/D7) are closed, D5 is open, and the header's three
    signature lines stay PENDING while it is."""
    locked = (REPO / "docs" / "LOCKED_ANALYSIS.md").read_text(encoding="utf-8")
    section = locked.split("## 1. Feature stage")[1].split("## 2.")[0]
    assert "DECISION NEEDED" not in section
    assert "**`s11_out`**" in section
    # the closed set, struck through in §12 the way D8 already was
    for closed in ("~~D1~~", "~~D2~~", "~~D3~~", "~~D4~~", "~~D6~~",
                   "~~D7~~", "~~D8~~"):
        assert closed in locked, closed
    assert "~~D5~~" not in locked, "D5 is NOT closed — I1 owns it"
    assert "| D5 | 5 |" in locked
    # EXACTLY ONE decision still carries the marker, and it is D5's own
    # row in section 5. Every closed decision says so where it is defined.
    body = locked.split("## 1. Feature stage", 1)[1]
    marked = [l for l in body.splitlines() if "DECISION NEEDED" in l]
    assert len(marked) == 1, marked
    assert "Which discovery map" in marked[0]
    # the document itself is NOT frozen
    assert "STATUS: **DRAFT — NOT YET FROZEN**" in locked
    header = locked.split("## 1. Feature stage")[0]
    assert header.count("`PENDING`") == 2


@needs_results
def test_the_committed_tables_are_from_one_split_and_the_whole_cohort():
    import csv as _csv
    names = ["T_I2a_invariance.csv", "T_I2b_sweep.csv", "T_I2c_gaps.csv",
             "T_I2d_maps.csv", "T_I2e_registers.csv"]
    shas = set()
    for name in names:
        with open(TABLE_DIR / name, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert rows, name
        shas |= {r["split_sha256"] for r in rows}
        assert {r["work_package"] for r in rows} == {"I2_terminal"}
    assert len(shas) == 1, f"the tables mix {len(shas)} splits"
    meta = json.loads((TABLE_DIR / "build_meta.json").read_text("utf-8"))
    assert meta["split_name"] == "calibration"
    assert meta["bootstrap_resamples"] == 10000 and meta["bootstrap_seed"] == 0
    assert not meta["runs_absent"], meta["runs_absent"]
    assert len(meta["runs_present"]) == 19


def test_the_new_framework_file_is_inside_the_I0_ast_tests_glob():
    """`saga/frozen/diag.py` is new, and TASK I0's AST test globs
    `saga/frozen/*.py` — so it is already covered there. This asserts the
    file sits where that glob looks, so the coverage cannot be lost by
    moving it somewhere the I0 test does not see."""
    i0 = (REPO / "tests" / "test_I0_frozen.py").read_text(encoding="utf-8")
    assert '(REPO / "saga" / "frozen").glob("*.py")' in i0
    assert (REPO / "saga" / "frozen" / "diag.py").exists()
