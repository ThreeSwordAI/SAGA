"""TASK-06 / 06B: derivation driver planning, cell-tau extension, canon
apply with recipe_actual remapping, eval consistency checker."""

import csv
import json
import sys

import numpy as np
import pytest
import yaml

import tools.compute_fixed_thr as compute_fixed_thr
from analysis.check_eval_consistency import check_run
from tools.apply_fixed_thr import apply_to_file, resolve_key
from tools.derive_runs import plan_steps


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_run_dir(root, run_id, arch="vit_small_patch16_224",
                 variant="saga", recipe="mixup", tags=("best", "last")):
    run = root / run_id
    (run / "ckpt").mkdir(parents=True)
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(
        {"model": {"arch": arch}, "variant": variant, "recipe": recipe}))
    (run / "meta.json").write_text(json.dumps({"run_id": run_id}))
    for tag in tags:
        (run / "ckpt" / f"{tag}.pth").write_bytes(b"\x01" * 16)
    return run


def make_run_norms(run, sha, norms):
    diag = run / "diag"
    diag.mkdir(exist_ok=True)
    np.savez(diag / "diag_final_last_norms.npz",
             last_block_patch_norms=np.asarray(norms, dtype=np.float16))
    (diag / "diag_final_last.json").write_text(json.dumps(
        {"arch": "vit_small", "ckpt_sha256": sha}))
    return diag / "diag_final_last_norms.npz"


# ── derive_runs planning ─────────────────────────────────────────────────────

def test_plan_steps_full_run(tmp_path):
    run = make_run_dir(tmp_path, "e2r_vits_mixup_saga_s1")
    steps = plan_steps(run, "/data", "split.json", python="py")
    names = [s[0] for s in steps]
    assert names == [f"{run.name}:eval:best", f"{run.name}:diag:best",
                     f"{run.name}:eval:last", f"{run.name}:diag:last"]
    for name, out_json, ckpt, argv in steps:
        assert ckpt.exists()
        if ":eval:" in name:
            assert out_json.name.startswith("imagenet_val_")
            assert "tools/eval.py" in argv[1]
        else:
            assert out_json.name.startswith("diag_final_")
            assert "--attn" in argv
        assert "--arch" in argv and argv[argv.index("--arch") + 1] == "vit_small"
        assert argv[argv.index("--variant") + 1] == "saga"


def test_plan_steps_missing_ckpt(tmp_path):
    run = make_run_dir(tmp_path, "e2r_x", tags=("last",))
    steps = plan_steps(run, "/data", "split.json")
    assert steps[0][0].endswith("best:MISSING-CKPT") and steps[0][3] is None
    assert len(steps) == 3                          # last still planned


def test_plan_steps_rejects_unknown_arch(tmp_path):
    run = make_run_dir(tmp_path, "e2r_y", arch="vit_huge_patch14_224")
    with pytest.raises(ValueError, match="unsupported arch"):
        plan_steps(run, "/data", "split.json")


# ── compute_fixed_thr --cell (v2 extension + canon creation) ────────────────

def test_add_cell_extends_without_touching_existing(tmp_path, monkeypatch):
    runs = tmp_path / "results" / "runs"
    run = make_run_dir(runs, "e2r_vitb_mixup_baseline_s1")
    # 3 images, all-equal rows -> per-image thresholds [10, 20, 30];
    # LOWER median over an odd count -> tau 20
    npz = make_run_norms(run, "shaB",
                         [[10.0] * 4, [20.0] * 4, [30.0] * 4])

    out = tmp_path / "fixed_thresholds_v2.json"
    existing = {"vit_small|mixup": 20.90625, "definition": "old-def", "k": 5.0,
                "source_ckpt_sha256": {"vit_small|mixup": "legacy-sha"}}
    out.write_text(json.dumps(existing))

    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--cell", f"vit_base|mixup={npz}",
        "--out", str(out)])
    compute_fixed_thr.main()

    d = json.loads(out.read_text())
    assert d["vit_base|mixup"] == 20.0
    assert d["vit_small|mixup"] == 20.90625          # untouched
    assert d["definition"] == "old-def"              # untouched
    assert d["source_ckpt_sha256"]["vit_small|mixup"] == "legacy-sha"
    assert d["source_ckpt_sha256"]["vit_base|mixup"] == "shaB"
    assert d["source_run"]["vit_base|mixup"] == "e2r_vitb_mixup_baseline_s1"

    # idempotent re-add (same tau) succeeds; different tau refused
    compute_fixed_thr.main()
    assert json.loads(out.read_text())["vit_base|mixup"] == 20.0
    npz2 = make_run_norms(make_run_dir(runs, "e2r_other"), "shaX",
                          [[99.0] * 4])
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--cell", f"vit_base|mixup={npz2}",
        "--out", str(out)])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        compute_fixed_thr.main()


def test_add_cell_creates_canon_file(tmp_path, monkeypatch):
    runs = tmp_path / "results" / "runs"
    run = make_run_dir(runs, "e2r_vits_mixup_baseline_s1")
    npz = make_run_norms(run, "shaS", [[7.0] * 4])
    out = tmp_path / "fixed_thresholds_canon.json"
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--cell", f"vit_small|mixup={npz}",
        "--out", str(out), "--definition", "canon"])
    compute_fixed_thr.main()
    d = json.loads(out.read_text())
    assert d["vit_small|mixup"] == 7.0
    assert "recipe_actual" in d["definition"]
    assert d["source_run"]["vit_small|mixup"] == "e2r_vits_mixup_baseline_s1"


def test_add_cell_refuses_v1_default_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--cell", "a|b=x.npz"])
    with pytest.raises(SystemExit, match="v1 default path"):
        compute_fixed_thr.main()


# ── apply_fixed_thr: canon remap + run-dir paths ─────────────────────────────

def _legacy_diag(tmp_path, arch, recipe, norms):
    stem = f"e2_{arch}_{recipe}_saga_rlast_last"
    jp = tmp_path / f"{stem}.json"
    jp.write_text(json.dumps({"arch": arch, "sink_fixed_thr": 1.0,
                              "fixed_thr_value": 9.0,
                              "sink_fixed_v2": 2.0,
                              "fixed_thr_v2_value": 8.0}))
    np.savez(tmp_path / f"{stem}_norms.npz",
             last_block_patch_norms=np.asarray(norms, dtype=np.float16))
    return jp


def test_canon_remaps_legacy_nomix_to_mixup(tmp_path):
    jp = _legacy_diag(tmp_path, "vit_small", "nomix", [[1.0, 50.0]])
    thr = {"vit_small|mixup": 10.0}
    status = apply_to_file(jp, thr, version="canon")
    assert status.startswith("sink_fixed_canon")
    d = json.loads(jp.read_text())
    assert d["sink_fixed_canon"] == 1.0 and d["canon_thr_value"] == 10.0
    # v1/v2 provenance fields untouched
    assert d["sink_fixed_thr"] == 1.0 and d["fixed_thr_value"] == 9.0
    assert d["sink_fixed_v2"] == 2.0 and d["fixed_thr_v2_value"] == 8.0


def test_canon_skips_pending_vitb_nomix(tmp_path):
    jp = _legacy_diag(tmp_path, "vit_base", "nomix", [[1.0]])
    status = apply_to_file(jp, {"vit_base|mixup": 5.0}, version="canon")
    assert "pending" in status
    assert "sink_fixed_canon" not in json.loads(jp.read_text())


def test_run_dir_diag_resolves_recipe_from_config(tmp_path):
    runs = tmp_path / "results" / "runs"
    run = make_run_dir(runs, "e2r_vits_mixup_saga_s1", recipe="mixup")
    make_run_norms(run, "sha", [[1.0, 50.0]])
    jp = run / "diag" / "diag_final_last.json"

    assert resolve_key(jp, "vit_small", "v2") == ("vit_small|mixup", None)
    assert resolve_key(jp, "vit_small", "canon") == ("vit_small|mixup", None)

    status = apply_to_file(jp, {"vit_small|mixup": 10.0}, version="canon")
    assert status.startswith("sink_fixed_canon")
    status = apply_to_file(jp, {"vit_small|mixup": 10.0}, version="v2")
    assert status.startswith("sink_fixed_v2")


# ── eval consistency checker ─────────────────────────────────────────────────

def _run_with_eval(tmp_path, run_id, log_top1, eval_top1):
    run = tmp_path / run_id
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "imagenet_val_last.json").write_text(
        json.dumps({"top1": eval_top1}))
    with open(run / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_top1_full"])
        w.writeheader()
        w.writerow({"epoch": 298, "val_top1_full": 1.0})
        w.writerow({"epoch": 299, "val_top1_full": log_top1})
    return run


def test_consistency_ok_warn_skip(tmp_path):
    ok = _run_with_eval(tmp_path, "a", 79.384, 79.384)
    warn = _run_with_eval(tmp_path, "b", 79.384, 79.484)
    assert check_run(ok, 0.02)[0] == "ok"
    status, msg = check_run(warn, 0.02)
    assert status == "warn" and "WARNING" in msg and "e299" in msg
    (tmp_path / "c").mkdir()
    assert check_run(tmp_path / "c", 0.02)[0] == "skip"


# ── review-driven guards ─────────────────────────────────────────────────────

def test_add_cell_refuses_k_mismatch(tmp_path, monkeypatch):
    runs = tmp_path / "results" / "runs"
    run = make_run_dir(runs, "e2r_a")
    npz = make_run_norms(run, "sha", [[7.0] * 4])
    out = tmp_path / "canon.json"
    out.write_text(json.dumps({"definition": "d", "k": 5.0,
                               "source_ckpt_sha256": {}}))
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--cell", f"x|y={npz}",
        "--out", str(out), "--k", "4"])
    with pytest.raises(SystemExit, match="does not match the file"):
        compute_fixed_thr.main()


def test_add_cell_noop_rerun_leaves_file_bytes(tmp_path, monkeypatch):
    runs = tmp_path / "results" / "runs"
    run = make_run_dir(runs, "e2r_b")
    npz = make_run_norms(run, "sha", [[7.0] * 4])
    out = tmp_path / "canon.json"
    monkeypatch.setattr(sys, "argv", [
        "compute_fixed_thr.py", "--cell", f"x|y={npz}",
        "--out", str(out), "--definition", "canon"])
    compute_fixed_thr.main()
    before = out.read_bytes()
    compute_fixed_thr.main()          # idempotent re-add: no rewrite at all
    assert out.read_bytes() == before


def test_run_completion_gate(tmp_path):
    from tools.derive_runs import run_complete
    run = make_run_dir(tmp_path, "e2r_c")
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(
        {"model": {"arch": "vit_small_patch16_224"}, "variant": "saga",
         "recipe": "mixup", "train": {"epochs": 3}}))
    # mid-training: no end_time
    (run / "meta.json").write_text(json.dumps({"end_time": None}))
    with open(run / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_top1_full"])
        w.writeheader()
        w.writerow({"epoch": 1, "val_top1_full": 1.0})
    ok, reason = run_complete(run)
    assert not ok and "end_time" in reason

    # finished marker but truncated log
    (run / "meta.json").write_text(json.dumps({"end_time": "t"}))
    ok, reason = run_complete(run)
    assert not ok and "final log epoch 1 != expected 2" in reason

    with open(run / "log.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_top1_full"])
        w.writerow({"epoch": 2, "val_top1_full": 2.0})
    assert run_complete(run) == (True, "")


def test_consistency_flags_truncated_run(tmp_path):
    run = _run_with_eval(tmp_path, "trunc", 50.0, 50.0)   # rows end at e299
    # rewrite the log to end early
    with open(run / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_top1_full"])
        w.writeheader()
        w.writerow({"epoch": 249, "val_top1_full": 50.0})
    status, msg = check_run(run, 0.02)
    assert status == "warn" and "truncated" in msg
