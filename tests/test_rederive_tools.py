"""TASK-02 tools: check_done guard, fixed-thr math, apply idempotency,
job generator output."""

import csv
import json

import numpy as np
import pytest

from tools.apply_fixed_thr import apply_to_file, fixed_count_mean
from tools.check_done import is_done
from tools.compute_fixed_thr import (compute_tau, find_source,
                                     per_image_mad_thresholds)
from tools.gen_rederive_jobs import emit, load_rows


# ── check_done ───────────────────────────────────────────────────────────────

def test_check_done_guard(tmp_path):
    out = tmp_path / "x.json"
    assert not is_done(out, "abc")                    # missing
    out.write_text(json.dumps({"ckpt_sha256": "abc", "top1": 1.0}))
    assert is_done(out, "abc")                        # done, same ckpt
    assert not is_done(out, "OTHER")                  # different ckpt
    out.write_text("{corrupt")
    assert not is_done(out, "abc")                    # corrupt -> re-run


# ── compute_fixed_thr ────────────────────────────────────────────────────────

def test_mad_threshold_math():
    # image 0: nine 10s + one 100 -> med 10, MAD 0 -> thr 10
    # image 1: 1..9 + 100 -> med 5 (lower), MAD 2 -> thr 5 + 5*2 = 15
    norms = np.array([[10.0] * 9 + [100.0],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0]], dtype=np.float16)
    thr = per_image_mad_thresholds(norms, k=5.0)
    assert thr.tolist() == [10.0, 15.0]


def test_compute_tau_is_median_of_thresholds(tmp_path):
    norms = np.stack([np.full(10, 10.0),           # thr 10
                      np.full(10, 20.0),           # thr 20
                      np.full(10, 30.0)]).astype(np.float16)  # thr 30
    p = tmp_path / "e2_vit_small_nomix_baseline_rlast_last_norms.npz"
    np.savez(p, last_block_patch_norms=norms)
    assert compute_tau(p, k=5.0) == 20.0
    assert find_source(tmp_path, "vit_small") == p
    with pytest.raises(FileNotFoundError):
        find_source(tmp_path, "vit_base")          # no match -> error


# ── apply_fixed_thr ──────────────────────────────────────────────────────────

def test_apply_updates_only_the_two_keys(tmp_path):
    norms = np.array([[1.0, 2.0, 50.0], [1.0, 1.0, 1.0]], dtype=np.float16)
    jp = tmp_path / "e2_vit_small_mixup_saga_rlast_last.json"
    original = {"arch": "vit_small", "variant": "saga", "top_secret": 42,
                "sink_fixed_thr": None, "fixed_thr_value": None}
    jp.write_text(json.dumps(original))
    np.savez(tmp_path / (jp.stem + "_norms.npz"),
             last_block_patch_norms=norms)

    thresholds = {"vit_small": 10.0}
    status = apply_to_file(jp, thresholds)
    assert status.startswith("sink_fixed_thr")

    updated = json.loads(jp.read_text())
    assert updated["sink_fixed_thr"] == 0.5        # 1 token > 10 in image 0
    assert updated["fixed_thr_value"] == 10.0
    for key in original:
        if key not in ("sink_fixed_thr", "fixed_thr_value"):
            assert updated[key] == original[key]

    # idempotent: applying again changes nothing
    apply_to_file(jp, thresholds)
    assert json.loads(jp.read_text()) == updated

    # missing npz sibling -> skip, file untouched
    jp2 = tmp_path / "orphan.json"
    jp2.write_text(json.dumps({"arch": "vit_small"}))
    assert apply_to_file(jp2, thresholds) == "no _norms.npz sibling"


def test_fixed_count_is_strict():
    norms = np.array([[10.0, 10.0, 11.0]], dtype=np.float16)
    assert fixed_count_mean(norms, tau=10.0) == 1.0


# ── gen_rederive_jobs ────────────────────────────────────────────────────────

def _write_manifest(path, rows):
    fields = ["path", "filename", "size_bytes", "mtime_iso", "sha256",
              "exp", "arch", "recipe", "variant", "seed"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _row(**kw):
    base = {"path": "/vault/e2/ckpt/ViT-S_SAGA/best.pth",
            "filename": "best.pth", "size_bytes": "1", "mtime_iso": "t",
            "sha256": "aa11", "exp": "e2", "arch": "vit_small",
            "recipe": "mixup", "variant": "saga", "seed": "rlast"}
    base.update(kw)
    return base


def test_generator_emits_guarded_steps(tmp_path):
    m = tmp_path / "manifest.csv"
    _write_manifest(m, [
        _row(),
        _row(path="/vault/e2/ckpt/ViT-B_baseline/last.pth",
             filename="last.pth", sha256="bb22", arch="vit_base",
             recipe="nomix", variant="baseline"),
        _row(path="/vault/e6/x/best.pth", exp="e6", arch="", variant="",
             recipe="", seed=""),                  # other exp: ignored
    ])
    rows = load_rows(m, {"e2"})
    assert len(rows) == 2

    script = emit(rows, data="$STAGE_DIR", batch_eval=256, batch_diag=128)
    stem_a = "e2_vit_small_mixup_saga_rlast_best"
    stem_b = "e2_vit_base_nomix_baseline_rlast_last"
    for stem, sha in [(stem_a, "aa11"), (stem_b, "bb22")]:
        assert f"check_done.py results/legacy/eval/{stem}.json {sha}" in script
        assert f"check_done.py results/legacy/diag/{stem}.json {sha}" in script
        assert f"--out results/legacy/eval/{stem}.json" in script
        assert f"--out results/legacy/diag/{stem}.json" in script
        assert f"eval/{stem} FAILED" in script
        assert f"diag/{stem} FAILED" in script
    assert script.count("tools/eval.py") == 2
    assert script.count("tools/diagnose.py") == 2
    assert "--split-file results/diagsplit/val_diag_split.json" in script


def test_generator_rejects_unfilled_rows(tmp_path):
    m = tmp_path / "manifest.csv"
    _write_manifest(m, [_row(seed="")])            # unfilled seed
    with pytest.raises(SystemExit):
        load_rows(m, {"e2"})
    _write_manifest(m, [_row(arch="vit_large")])   # unsupported arch
    with pytest.raises(SystemExit):
        load_rows(m, {"e2"})
