"""TASK-07 A1/A2: sink-address maps + concentration stats, canon-tau key
resolution (recipe_actual remap included), idempotency, cross-checks, and
the optional-run matrix/chain-generator prep."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.sink_address import (build_addr, collect_npz, concentration,
                                freq_above_mad, freq_above_tau, gini,
                                process_one)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "classification" / "tools"))
import train as trainer  # classification/tools/train.py

NEW_RUNS = ["e2r_vits_mixup_registers_s1", "e2r_vitb_mixup_registers_s1",
            "e2r_vitb_mixup_baseline_s2", "e2r_vitb_mixup_saga_s2"]


# ── fixtures ─────────────────────────────────────────────────────────────────

# 3 images x 4 positions; hand-checkable
NORMS = np.array([
    [1.0, 10.0, 2.0, 1.0],
    [1.0, 10.0, 1.0, 1.0],
    [1.0,  1.0, 1.0, 1.0],
], dtype=np.float16)


def make_legacy_pair(diag_dir: Path, stem: str, norms=NORMS, sha="aa",
                     arch="vit_small", extra=None):
    diag_dir.mkdir(parents=True, exist_ok=True)
    np.savez(diag_dir / f"{stem}_norms.npz",
             last_block_patch_norms=np.asarray(norms, dtype=np.float16))
    meta = {"arch": arch, "ckpt_sha256": sha, "variant": "saga",
            "n_images": int(np.asarray(norms).shape[0])}
    meta.update(extra or {})
    (diag_dir / f"{stem}.json").write_text(json.dumps(meta))
    return diag_dir / f"{stem}_norms.npz"


def make_run_pair(runs_root: Path, run_id: str, recipe: str, norms=NORMS,
                  sha="bb", arch="vit_small"):
    run = runs_root / run_id
    diag = run / "diag"
    diag.mkdir(parents=True)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(
        {"recipe": recipe, "variant": "saga"}))
    np.savez(diag / "diag_final_last_norms.npz",
             last_block_patch_norms=np.asarray(norms, dtype=np.float16))
    (diag / "diag_final_last.json").write_text(json.dumps(
        {"arch": arch, "ckpt_sha256": sha,
         "n_images": int(np.asarray(norms).shape[0])}))
    return diag / "diag_final_last_norms.npz"


# ── freq maps: known values ──────────────────────────────────────────────────

def test_freq_above_tau_known():
    # tau=5: only position 1 in images 0,1 exceeds
    np.testing.assert_allclose(freq_above_tau(NORMS, 5.0),
                               [0.0, 2 / 3, 0.0, 0.0])
    # tau=0.5: everything exceeds
    np.testing.assert_allclose(freq_above_tau(NORMS, 0.5), [1, 1, 1, 1])
    # strict >: values equal to tau do not count
    np.testing.assert_allclose(freq_above_tau(NORMS, 10.0), [0, 0, 0, 0])


def test_freq_above_mad_known():
    # image 0: sorted [1,1,2,10], lower median = 1 (idx (4-1)//2 = 1);
    #   |v-m| = [0,9,1,0] sorted [0,0,1,9] -> MAD = 0 -> thr = 1 -> [0,1,1,0]
    # image 1: median 1, MAD 0, thr 1 -> only position 1 (10 > 1)
    # image 2: all equal -> MAD 0 -> thr 1 -> none (strict >)
    np.testing.assert_allclose(freq_above_mad(NORMS, k=5.0),
                               [0.0, 2 / 3, 1 / 3, 0.0])


def test_freq_maps_match_bruteforce():
    rng = np.random.RandomState(0)
    v = rng.lognormal(1.0, 1.0, size=(7, 9)).astype(np.float16)
    tau = float(np.median(v))
    f_tau = np.zeros(9)
    f_mad = np.zeros(9)
    for i in range(7):
        row = np.sort(v[i].astype(np.float64))
        med = row[(9 - 1) // 2]
        mad = np.sort(np.abs(v[i].astype(np.float64) - med))[(9 - 1) // 2]
        for p in range(9):
            x = float(v[i, p])
            f_tau[p] += x > tau
            f_mad[p] += x > med + 5.0 * mad
    np.testing.assert_allclose(freq_above_tau(v, tau), f_tau / 7)
    np.testing.assert_allclose(freq_above_mad(v, 5.0), f_mad / 7)


# ── concentration stats: known values ────────────────────────────────────────

def test_concentration_uniform():
    c = concentration(np.full(196, 0.25))
    assert c["entropy_bits"] == pytest.approx(math.log2(196))
    assert c["entropy_normalized"] == pytest.approx(1.0)
    assert c["entropy_uniform_bits"] == pytest.approx(math.log2(196))
    assert c["gini"] == pytest.approx(0.0, abs=1e-12)
    assert c["top5_share"] == pytest.approx(5 / 196)
    assert c["top20_share"] == pytest.approx(20 / 196)
    assert c["top10_positions"] == list(range(10))  # ties -> lower index
    assert c["total_mass"] == pytest.approx(196 * 0.25)


def test_concentration_onehot():
    f = np.zeros(196)
    f[42] = 0.7
    c = concentration(f)
    assert c["entropy_bits"] == pytest.approx(0.0)
    assert c["entropy_normalized"] == pytest.approx(0.0)
    assert c["gini"] == pytest.approx(195 / 196)
    assert c["top5_share"] == pytest.approx(1.0)
    assert c["top20_share"] == pytest.approx(1.0)
    assert c["top10_positions"][0] == 42
    assert c["total_mass"] == pytest.approx(0.7)


def test_concentration_hand_computed():
    # q = [.5, .25, .25, 0]: H = 1.5 bits; gini: sorted [0,.25,.25,.5],
    # 2*(1*0+2*.25+3*.25+4*.5)/(4*1) - 5/4 = 0.375
    c = concentration(np.array([0.5, 0.25, 0.25, 0.0]))
    assert c["entropy_bits"] == pytest.approx(1.5)
    assert c["entropy_normalized"] == pytest.approx(1.5 / 2.0)
    assert c["gini"] == pytest.approx(0.375)
    assert c["top5_share"] == pytest.approx(1.0)  # k > n saturates at 1
    assert c["top10_positions"] == [0, 1, 2, 3]


def test_concentration_zero_mass():
    c = concentration(np.zeros(8))
    assert c["total_mass"] == 0.0
    for key in ("entropy_bits", "entropy_normalized", "gini",
                "top5_share", "top20_share"):
        assert c[key] is None
    assert c["top10_positions"] == []


def test_gini_zero_total():
    assert gini(np.zeros(5)) is None


# ── canon-tau key resolution (incl. recipe_actual remap) ─────────────────────

def test_legacy_nomix_stem_resolves_to_mixup_tau(tmp_path):
    # legacy ViT-S *nomix-dir* run: recipe_actual = mixup (erratum remap)
    npz = make_legacy_pair(tmp_path, "e2_vit_small_nomix_saga_rlast_last")
    status = process_one(npz, {"vit_small|mixup": 5.0})
    assert status.startswith("wrote")
    addr = json.loads(
        (tmp_path / "e2_vit_small_nomix_saga_rlast_last_addr.json").read_text())
    assert addr["canon_key"] == "vit_small|mixup"
    assert addr["tau_canon"] == 5.0
    assert addr["freq_canon"] == pytest.approx([0.0, 2 / 3, 0.0, 0.0])
    assert addr["freq_mad"] == pytest.approx([0.0, 2 / 3, 1 / 3, 0.0])
    assert addr["ckpt_sha256"] == "aa"
    assert addr["n_images"] == 3 and addr["n_positions"] == 4


def test_run_dir_resolves_recipe_from_config(tmp_path):
    npz = make_run_pair(tmp_path, "e2r_vits_nomix_saga_s1", recipe="nomix")
    status = process_one(npz, {"vit_small|nomix": 5.0,
                               "vit_small|mixup": 999.0})
    assert status.startswith("wrote")
    addr = json.loads(npz.with_name("diag_final_last_addr.json").read_text())
    assert addr["canon_key"] == "vit_small|nomix"
    assert addr["tau_canon"] == 5.0


def test_missing_tau_writes_partial(tmp_path):
    npz = make_legacy_pair(tmp_path, "e2_vit_base_mixup_saga_rlast_last",
                           arch="vit_base")
    status = process_one(npz, {"vit_small|mixup": 5.0})
    assert status.startswith("PARTIAL")
    addr = json.loads(
        (tmp_path / "e2_vit_base_mixup_saga_rlast_last_addr.json").read_text())
    assert addr["freq_canon"] is None
    assert addr["mass_canon"] is None
    assert addr["concentration_canon"] is None
    assert "not in thresholds" in addr["canon_skip_reason"]
    assert addr["freq_mad"] == pytest.approx([0.0, 2 / 3, 1 / 3, 0.0])


# ── idempotency / requeue safety ─────────────────────────────────────────────

def test_skip_on_matching_sha_and_tau(tmp_path):
    npz = make_legacy_pair(tmp_path, "e2_vit_small_mixup_saga_rlast_last")
    thr = {"vit_small|mixup": 5.0}
    assert process_one(npz, thr).startswith("wrote")
    assert process_one(npz, thr) == "SKIP (up to date)"
    assert not list(tmp_path.glob("*.tmp"))

    # a tau change rewrites
    assert process_one(npz, {"vit_small|mixup": 0.5}).startswith("wrote")
    addr = json.loads(
        (tmp_path / "e2_vit_small_mixup_saga_rlast_last_addr.json").read_text())
    assert addr["tau_canon"] == 0.5
    assert addr["freq_canon"] == pytest.approx([1, 1, 1, 1])

    # a sha change rewrites
    sib = tmp_path / "e2_vit_small_mixup_saga_rlast_last.json"
    meta = json.loads(sib.read_text())
    meta["ckpt_sha256"] = "new"
    sib.write_text(json.dumps(meta))
    assert process_one(npz, {"vit_small|mixup": 0.5}).startswith("wrote")


def test_partial_self_heals_when_tau_appears(tmp_path):
    npz = make_legacy_pair(tmp_path, "e2_vit_small_mixup_saga_rlast_last")
    assert process_one(npz, {}).startswith("PARTIAL")
    assert process_one(npz, {"vit_small|mixup": 5.0}).startswith("wrote")
    addr = json.loads(
        (tmp_path / "e2_vit_small_mixup_saga_rlast_last_addr.json").read_text())
    assert addr["freq_canon"] is not None


def test_crosscheck_reengages_after_canon_backfill(tmp_path):
    """An addr written before the sibling had sink_fixed_canon is re-run
    (not skipped) once the backfill lands, so the hard cross-check engages."""
    stem = "e2_vit_small_mixup_saga_rlast_last"
    npz = make_legacy_pair(tmp_path, stem)
    thr = {"vit_small|mixup": 5.0}
    assert process_one(npz, thr).startswith("wrote")
    addr = json.loads((tmp_path / f"{stem}_addr.json").read_text())
    assert isinstance(addr["crosscheck_canon"], str)  # not comparable yet

    sib = tmp_path / f"{stem}.json"
    meta = json.loads(sib.read_text())
    meta.update({"sink_fixed_canon": 2 / 3, "canon_thr_value": 5.0})
    sib.write_text(json.dumps(meta))
    assert process_one(npz, thr).startswith("wrote")  # NOT "SKIP"
    addr = json.loads((tmp_path / f"{stem}_addr.json").read_text())
    assert addr["crosscheck_canon"]["abs_diff"] <= 1e-8
    assert process_one(npz, thr) == "SKIP (up to date)"


def test_skip_guard_rechecks_n_images(tmp_path):
    stem = "e2_vit_small_mixup_saga_rlast_last"
    npz = make_legacy_pair(tmp_path, stem)
    thr = {"vit_small|mixup": 5.0}
    assert process_one(npz, thr).startswith("wrote")
    # regenerated npz + sibling with a different image count, same ckpt sha
    make_legacy_pair(tmp_path, stem, norms=np.vstack([NORMS, NORMS]))
    status = process_one(npz, thr)
    assert status.startswith("wrote")
    addr = json.loads((tmp_path / f"{stem}_addr.json").read_text())
    assert addr["n_images"] == 6


def test_single_position_npz_is_error(tmp_path):
    npz = make_legacy_pair(tmp_path, "e2_vit_small_mixup_saga_rlast_last",
                           norms=np.ones((3, 1)))
    assert process_one(npz, {"vit_small|mixup": 5.0}).startswith("ERROR")


# ── cross-checks & guards ────────────────────────────────────────────────────

def test_canon_crosscheck_pass_and_fail(tmp_path):
    # sibling computed from the same npz with the same tau: mean per-image
    # count at tau=5 is (1 + 1 + 0)/3
    good = {"sink_fixed_canon": 2 / 3, "canon_thr_value": 5.0}
    npz = make_legacy_pair(tmp_path, "e2_vit_small_mixup_saga_rlast_last",
                           extra=good)
    status = process_one(npz, {"vit_small|mixup": 5.0})
    assert status.startswith("wrote")
    addr = json.loads(
        (tmp_path / "e2_vit_small_mixup_saga_rlast_last_addr.json").read_text())
    assert addr["crosscheck_canon"]["abs_diff"] <= 1e-8
    assert addr["mass_canon"] == pytest.approx(sum(addr["freq_canon"]))

    bad = {"sink_fixed_canon": 1.5, "canon_thr_value": 5.0}
    npz2 = make_legacy_pair(tmp_path, "e2_vit_small_mixup_baseline_rlast_last",
                            extra=bad)
    status = process_one(npz2, {"vit_small|mixup": 5.0})
    assert status.startswith("ERROR")
    assert not (tmp_path
                / "e2_vit_small_mixup_baseline_rlast_last_addr.json").exists()


def test_mad_crosscheck_recorded(tmp_path):
    npz = make_legacy_pair(tmp_path, "e2_vit_small_mixup_saga_rlast_last",
                           extra={"sink_mad_k5": 1.0})
    assert process_one(npz, {"vit_small|mixup": 5.0}).startswith("wrote")
    addr = json.loads(
        (tmp_path / "e2_vit_small_mixup_saga_rlast_last_addr.json").read_text())
    assert addr["mass_mad"] == pytest.approx(1.0)  # counts 2,1,0 over 3 images
    assert addr["crosscheck_mad"]["diff"] == pytest.approx(0.0)


def test_n_images_mismatch_is_error(tmp_path):
    npz = make_legacy_pair(tmp_path, "e2_vit_small_mixup_saga_rlast_last",
                           extra={"n_images": 999})
    assert process_one(npz, {"vit_small|mixup": 5.0}).startswith("ERROR")


def test_sibling_without_sha_is_skipped(tmp_path):
    diag = tmp_path / "diag"
    diag.mkdir()
    np.savez(diag / "diag_e009_norms.npz",
             last_block_patch_norms=NORMS)
    (diag / "diag_e009.json").write_text(json.dumps({"n_images": 3}))
    assert process_one(diag / "diag_e009_norms.npz", {}) == \
        "SKIP (sibling has no ckpt_sha256)"


def test_collect_npz_covers_both_roots(tmp_path):
    legacy = tmp_path / "legacy_diag"
    make_legacy_pair(legacy, "e2_vit_small_mixup_saga_rlast_last")
    runs = tmp_path / "runs"
    make_run_pair(runs, "e2r_vits_mixup_saga_s1", recipe="mixup")
    found = collect_npz(legacy, runs)
    assert [p.name for p in found] == [
        "e2_vit_small_mixup_saga_rlast_last_norms.npz",
        "diag_final_last_norms.npz"]


def test_build_addr_mass_identities():
    """sum(freq) equals the mean per-image count for both maps."""
    rng = np.random.RandomState(1)
    v = rng.lognormal(1.0, 1.0, size=(11, 13)).astype(np.float16)
    sibling = {"ckpt_sha256": "x", "arch": "vit_small", "n_images": 11}
    addr = build_addr(v, sibling, 4.0, "vit_small|mixup", None, "t.npz")
    v64 = v.astype(np.float64)
    assert addr["mass_canon"] == pytest.approx((v64 > 4.0).sum(1).mean())
    assert sum(addr["freq_canon"]) == pytest.approx(addr["mass_canon"])
    assert sum(addr["freq_mad"]) == pytest.approx(addr["mass_mad"])
    np.testing.assert_allclose(addr["mean_norm"], v64.mean(axis=0))
    np.testing.assert_allclose(addr["p99_norm"],
                               np.percentile(v64, 99, axis=0))


# ── A2: optional-run matrix + chain generation ───────────────────────────────

def test_matrix_new_runs_resolve():
    matrix_path = REPO / "configs" / "e2r_matrix.yaml"
    matrix = yaml.safe_load(open(matrix_path))
    assert len(matrix["runs"]) == 14
    for run_id in NEW_RUNS:
        assert run_id in matrix["runs"]
        cfg = trainer.resolve_run_config(str(matrix_path), run_id)
        run = matrix["runs"][run_id]
        assert cfg["recipe"] == "mixup"
        assert cfg["seed"] == run["seed"]
        assert cfg["model"]["arch"] == run["arch"]
        if run["variant"] == "registers":
            assert cfg["model"]["registers"] == 4
            assert cfg["model"]["gate"] is False
        else:
            assert cfg["model"]["registers"] == 0
            assert cfg["model"]["gate"] == (run["variant"] == "saga")
        assert cfg["instrumentation"]["log_grad_phi"] is False
    # the original 10 runs are untouched
    assert matrix["runs"]["e2r_vits_mixup_saga_s1"] == {
        "arch": "vit_small_patch16_224", "recipe": "mixup",
        "variant": "saga", "seed": 1, "log_grad_phi": True}


def test_gen_slurm_chain_only(tmp_path):
    matrix = {"chain": {"vit_small_patch16_224": 2, "vit_base_patch16_224": 4},
              "runs": {
                  "e2r_old": {"arch": "vit_small_patch16_224", "recipe": "mixup",
                              "variant": "baseline", "seed": 1},
                  "e2r_new_b": {"arch": "vit_base_patch16_224", "recipe": "mixup",
                                "variant": "saga", "seed": 2},
                  "e2r_new_s": {"arch": "vit_small_patch16_224", "recipe": "mixup",
                                "variant": "registers", "seed": 1}}}
    mpath = tmp_path / "m.yaml"
    mpath.write_text(yaml.safe_dump(matrix))
    out = tmp_path / "scripts"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_slurm_chain.py"),
         "--matrix", str(mpath), "--out-dir", str(out),
         "--only", "e2r_new_b,e2r_new_s", "--base-port", "29750"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (out / "jobs").glob("*.sbatch")) == \
        ["e2r_new_b.sbatch", "e2r_new_s.sbatch"]  # e2r_old NOT generated
    assert not (out / "submit_e2r_old.sh").exists()
    b = (out / "jobs" / "e2r_new_b.sbatch").read_text()
    assert "--master_port=29750" in b and "--run e2r_new_b" in b
    s = (out / "jobs" / "e2r_new_s.sbatch").read_text()
    assert "--master_port=29751" in s
    assert "seq 2 4" in (out / "submit_e2r_new_b.sh").read_text()
    assert "seq 2 2" in (out / "submit_e2r_new_s.sh").read_text()


def test_gen_slurm_chain_only_unknown_id(tmp_path):
    matrix = {"runs": {"e2r_old": {"arch": "vit_small_patch16_224",
                                   "recipe": "mixup", "variant": "baseline",
                                   "seed": 1}}}
    mpath = tmp_path / "m.yaml"
    mpath.write_text(yaml.safe_dump(matrix))
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_slurm_chain.py"),
         "--matrix", str(mpath), "--out-dir", str(tmp_path / "s"),
         "--only", "nope"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "nope" in r.stderr


def test_gen_slurm_chain_refuses_implicit_full_regen(tmp_path):
    """A bare invocation must not silently rewrite every job file with
    reshuffled ports now that the matrix has grown (TASK-07)."""
    matrix = {"runs": {"e2r_old": {"arch": "vit_small_patch16_224",
                                   "recipe": "mixup", "variant": "baseline",
                                   "seed": 1}}}
    mpath = tmp_path / "m.yaml"
    mpath.write_text(yaml.safe_dump(matrix))
    out = tmp_path / "s"
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_slurm_chain.py"),
         "--matrix", str(mpath), "--out-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "--all" in r.stderr
    assert not list(out.glob("submit_*.sh"))

    # --all regenerates deliberately
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_slurm_chain.py"),
         "--matrix", str(mpath), "--out-dir", str(out), "--all"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "submit_e2r_old.sh").exists()
    assert "--master_port=29700" in (out / "jobs" / "e2r_old.sbatch").read_text()
