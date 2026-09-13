"""
tests/test_task09_phasec.py
===========================
TASK-09 Phase C: the Gate-2 tables, the generated note, the F7 selection,
and the second detection seed.

What these pin, in order of what would hurt most if it broke:
  * every table value traces to its source artifact through an INDEPENDENT
    path (the tests re-read coco_eval_best.json / miou_*.json /
    per_class_iou.csv themselves rather than trusting the builder);
  * the verdict is a pure function of two numbers and a frozen rule, the
    rule maps to PASS/PARTIAL/FAIL exactly as TASK-09 words it, and the
    uncertainty discussion cannot move it;
  * the note is GENERATED — re-rendering it from the committed tables must
    reproduce the committed file byte for byte, so no number in it can be
    hand-typed or go stale;
  * an unverified backbone yields MISSING, never a number;
  * the s2 runs differ from s1 in the seed and NOTHING else.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import build_dense_tables as bdt          # noqa: E402
from analysis import build_gate2_note as bgn            # noqa: E402
from tools import dense_runtime as dr                   # noqa: E402

TABLES = REPO / "results" / "tables"
NOTE = REPO / "results" / "notes" / "gate2_report.md"
DET_ROOT = REPO / "results" / "detection"
SEG_ROOT = REPO / "results" / "segmentation"
MATRIX = REPO / "configs" / "dense_matrix.yaml"
AP_KEYS = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]
VARIANTS = ["baseline", "registers", "saga"]


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(rows, kind, variant):
    return next((r for r in rows
                 if r["kind"] == kind and r["variant"] == variant), None)


@pytest.fixture(scope="module")
def matrix():
    return yaml.safe_load(open(MATRIX))


@pytest.fixture(scope="module")
def t4():
    return read_csv(TABLES / "T4_coco.csv")


@pytest.fixture(scope="module")
def t5():
    return read_csv(TABLES / "T5_ade20k.csv")


# ── tables trace to their sources ────────────────────────────────────────

def test_T4_values_match_the_run_artifacts_independently(t4):
    """Re-read each coco_eval_best.json here; the table must agree."""
    for variant in VARIANTS:
        row = pick(t4, "value", variant)
        assert row is not None, variant
        src = json.load(open(DET_ROOT / row["run_id"]
                             / "coco_eval_best.json"))
        for k in AP_KEYS:
            assert float(row[k]) == pytest.approx(float(src[k]), abs=5e-4), \
                (variant, k)
        assert int(row["epoch"]) == src["epoch"]
        assert int(row["n_val_images"]) == src["n_val_images"]
        assert row["backbone_sha256"] == src["backbone_sha256"]
        assert src["smoke"] is False, "a smoke run must never reach a table"


def test_T4_deltas_are_the_arithmetic_of_the_value_rows(t4):
    base = pick(t4, "value", "baseline")
    for variant in ("saga", "registers"):
        val = pick(t4, "value", variant)
        dl = pick(t4, f"delta_{variant}_minus_baseline", variant)
        for k in AP_KEYS:
            assert float(dl[k]) == pytest.approx(
                float(val[k]) - float(base[k]), abs=1e-6), (variant, k)


def test_T5_values_match_the_run_artifacts_independently(t5):
    for variant in VARIANTS:
        row = pick(t5, "value", variant)
        d = SEG_ROOT / row["run_id"]
        ss = json.load(open(d / "miou_ss.json"))
        ms = json.load(open(d / "miou_ms.json"))
        assert float(row["mIoU_ss"]) == pytest.approx(ss["mIoU"], abs=5e-5)
        assert float(row["mIoU_ms"]) == pytest.approx(ms["mIoU"], abs=5e-5)
        assert float(row["pixel_acc"]) == pytest.approx(ss["pixel_acc"],
                                                        abs=5e-5)
        assert int(row["n_images"]) == ss["n_images"]
        assert int(row["n_classes_scored"]) == ss["n_classes_scored"]
        assert ss["smoke"] is False


def test_T5_per_class_matches_the_confusion_matrices(t5):
    """The strongest available check: recompute every per-class IoU from the
    committed conf_matrix.npz and compare with the table."""
    np = pytest.importorskip("numpy")
    rows = read_csv(TABLES / "T5_ade20k_per_class.csv")
    assert len(rows) == 150
    for variant in VARIANTS:
        run_id = pick(t5, "value", variant)["run_id"]
        conf = np.load(SEG_ROOT / run_id / "conf_matrix.npz")["conf"]
        conf = conf.astype("int64")
        inter = np.diag(conf)
        union = conf.sum(1) + conf.sum(0) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        for i, r in enumerate(rows):
            cell = r[f"{variant}_iou_frac"]
            if cell == bdt.MISSING:
                assert not np.isfinite(iou[i])
                continue
            assert float(cell) == pytest.approx(float(iou[i]), abs=1e-6), i
            # the _pts column must be the same number times 100
            assert float(r[f"{variant}_iou_pts"]) == pytest.approx(
                float(cell) * 100, abs=1e-4)


def test_background_rows_are_the_three_classes_the_task_names():
    rows = read_csv(TABLES / "T5_ade20k_per_class.csv")
    bg = [r for r in rows if r["is_background_class"] == "yes"]
    assert len(bg) == 3
    assert {r["name"].split(",")[0].strip() for r in bg} == {"wall", "sky",
                                                             "floor"}
    assert {int(r["class_index"]) for r in bg} == {0, 2, 3}


def test_per_category_table_covers_all_80_coco_categories():
    rows = read_csv(TABLES / "T4_coco_per_category.csv")
    assert len(rows) == 80
    src = json.load(open(DET_ROOT / "det_vitb_saga_s1"
                         / "coco_eval_best.json"))
    by_id = {c["category_id"]: c for c in src["per_category"]}
    for r in rows:
        cid = int(r["category_id"])
        assert r["name"] == by_id[cid]["name"]
        want = by_id[cid]["AP"]
        if want is None:
            assert r["saga_AP"] == bdt.MISSING
        else:
            assert float(r["saga_AP"]) == pytest.approx(want, abs=5e-4)


def test_an_unverified_backbone_yields_MISSING_not_a_number(tmp_path):
    """A run whose recorded sha does not match the matrix candidate it names
    must not contribute a number to a paper table."""
    det = tmp_path / "det"
    (det / "det_vitb_saga_s1").mkdir(parents=True)
    good = json.load(open(DET_ROOT / "det_vitb_saga_s1"
                          / "coco_eval_best.json"))
    tampered = dict(good, backbone_sha256="0" * 64)
    (det / "det_vitb_saga_s1" / "coco_eval_best.json").write_text(
        json.dumps(tampered), encoding="utf-8")

    m = yaml.safe_load(open(MATRIX))
    loaded = bdt.load_detection(det, m)
    rows = bdt.det_rows(loaded, m)
    row = pick(rows, "value", "saga")
    assert row["AP"] == bdt.MISSING
    assert "UNVERIFIED BACKBONE" in row["note"]
    assert "sha mismatch" in row["note"]


# ── the verdict rule ─────────────────────────────────────────────────────

@pytest.mark.parametrize("ap_s,bg,expected", [
    (+1.0, +1.0, "PASS"),
    (+1.0, -1.0, "PARTIAL"),
    (-1.0, +1.0, "PARTIAL"),
    (-1.0, -1.0, "FAIL"),
    (0.0, -1.0, "FAIL"),      # exactly zero is NOT "in favor"
    (0.0, +1.0, "PARTIAL"),
    (0.0, 0.0, "FAIL"),
])
def test_verdict_rule_maps_as_task09_words_it(ap_s, bg, expected):
    moved_ap_s = ap_s > 0
    moved_bg = bg > 0
    got = ("PASS" if moved_ap_s and moved_bg
           else "PARTIAL" if moved_ap_s or moved_bg else "FAIL")
    assert got == expected


def test_note_verdict_follows_from_the_committed_tables(t4):
    """The note's verdict must be what the committed numbers imply — not a
    value that could drift from them."""
    ap_s = float(pick(t4, "delta_saga_minus_baseline", "saga")["AP_S"])
    per_class = read_csv(TABLES / "T5_ade20k_per_class.csv")
    bg = [float(r["saga_minus_baseline_iou_pts"]) for r in per_class
          if r["is_background_class"] == "yes"]
    assert len(bg) == 3
    bg_mean = sum(bg) / len(bg)

    expected = ("PASS" if ap_s > 0 and bg_mean > 0
                else "PARTIAL" if ap_s > 0 or bg_mean > 0 else "FAIL")
    text = NOTE.read_text(encoding="utf-8")
    assert f"## VERDICT: {expected}" in text, text[:400]
    # and the two criterion numbers appear as rendered, not rounded away
    assert f"{ap_s:+.3f}" in text
    assert f"{bg_mean:+.4f}" in text


def test_note_is_generated_not_hand_typed(tmp_path):
    """Re-render from the committed tables; byte-identical or the note has
    drifted from its inputs."""
    out = tmp_path / "gate2_report.md"
    rc = subprocess.run(
        [sys.executable, str(REPO / "analysis" / "build_gate2_note.py"),
         "--out", str(out)], cwd=REPO, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert out.read_text(encoding="utf-8") == NOTE.read_text(encoding="utf-8")


def test_note_states_the_single_seed_limitation():
    """The n=1 caveat is load-bearing for how the verdict is read; it must
    not quietly disappear from the note."""
    text = NOTE.read_text(encoding="utf-8")
    assert "exactly one run per cell" in text.lower()
    assert "standard error" in text.lower()
    # and the registers backbone mismatch must be disclosed
    assert "not backbone-matched" in text.lower()
    assert "fallback" in text.lower()


def test_within_run_spread_is_read_from_the_log_not_invented():
    spread, n, vals = bgn.within_run_spread(DET_ROOT / "det_vitb_saga_s1",
                                            "AP_S")
    rows = [r for r in read_csv(DET_ROOT / "det_vitb_saga_s1" / "log.csv")
            if r["AP_S"]]
    tail = [float(r["AP_S"]) for r in rows][-bgn.LAST_N_EVALS:]
    assert vals == tail
    assert spread == pytest.approx(max(tail) - min(tail))
    assert f"{spread:.3f}" in NOTE.read_text(encoding="utf-8")


# ── F7 ───────────────────────────────────────────────────────────────────

def test_f7_selection_is_deterministic_and_records_its_rule(tmp_path):
    sel_path = REPO / "results" / "figures_data" / "F7_coco_selection.json"
    committed = json.loads(sel_path.read_text(encoding="utf-8"))
    assert committed["rule"]["n_images"] == 2
    assert committed["rule"]["small_area_max_px2"] == 32.0 * 32.0
    assert "image_id asc" in committed["rule"]["ranking"]

    out = tmp_path / "sel.json"
    rc = subprocess.run(
        [sys.executable, str(REPO / "analysis" / "collect_F7.py"),
         "--out", str(out)], cwd=REPO, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    again = json.loads(out.read_text(encoding="utf-8"))
    assert ([i["image_id"] for i in again["images"]]
            == [i["image_id"] for i in committed["images"]])
    assert again["images"] == committed["images"]


def test_f7_selection_is_write_once():
    rc = subprocess.run(
        [sys.executable, str(REPO / "analysis" / "collect_F7.py")],
        cwd=REPO, capture_output=True, text=True)
    assert rc.returncode != 0
    assert "write-once" in (rc.stdout + rc.stderr)


def test_f7_renders_with_the_coco_half_missing_and_present(tmp_path):
    """The figure must build either way, and must SAY which state it is in
    rather than silently omitting the block."""
    pytest.importorskip("matplotlib")
    out = tmp_path / "F7.pdf"
    rc = subprocess.run(
        [sys.executable, str(REPO / "plotting" / "plot_F7.py"),
         "--out", str(out), "--coco-dir", str(tmp_path / "absent")],
        cwd=REPO, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert out.exists() and out.stat().st_size > 10_000
    assert "MISSING placeholder" in rc.stdout

    # now with a synthetic export present
    from PIL import Image
    coco = tmp_path / "coco"
    coco.mkdir()
    sel = json.loads((REPO / "results" / "figures_data"
                      / "F7_coco_selection.json").read_text(encoding="utf-8"))
    recs = []
    for im in sel["images"]:
        Image.new("RGB", (120, 90), (30, 60, 90)).save(
            coco / f"{im['image_id']}_crop.jpg")
        recs.append({
            "image_id": im["image_id"], "crop_file": f"{im['image_id']}_crop.jpg",
            "n_small": im["n_small"],
            "detections_in_crop": {v: [{"bbox_crop": [5, 5, 20, 20]}]
                                   for v in VARIANTS},
            "gt_in_crop": [{"bbox_crop": [8, 8, 18, 18], "is_small": True}],
        })
    (coco / "crops.json").write_text(json.dumps({"images": recs}),
                                     encoding="utf-8")
    out2 = tmp_path / "F7b.pdf"
    rc = subprocess.run(
        [sys.executable, str(REPO / "plotting" / "plot_F7.py"),
         "--out", str(out2), "--coco-dir", str(coco)],
        cwd=REPO, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert "rendered" in rc.stdout and out2.exists()


def test_f7_ade_panels_use_committed_probe_images():
    """Every ADE panel F7 draws must already be in the repo — the figure
    must not depend on the dataset."""
    probe = json.loads((REPO / "results" / "probe"
                        / "ade20k_fixed20.json").read_text(encoding="utf-8"))
    for stem in probe["stems"][:3]:
        for variant in VARIANTS:
            d = SEG_ROOT / f"seg_vitb_{variant}_s1" / "preds_fixed20"
            assert (d / f"{stem}_pred.png").exists(), (variant, stem)
        base = SEG_ROOT / "seg_vitb_baseline_s1" / "preds_fixed20"
        assert (base / f"{stem}_gt.png").exists()
        assert (base / f"{stem}_img.jpg").exists()


# ── the second detection seed ────────────────────────────────────────────

def test_seed2_runs_differ_from_s1_only_in_the_seed(matrix):
    for variant in ("baseline", "saga"):
        s1 = dr.resolve_dense_config(matrix, f"det_vitb_{variant}_s1")
        s2 = dr.resolve_dense_config(matrix, f"det_vitb_{variant}_s2")
        assert s1["seed"] == 1 and s2["seed"] == 2
        # same backbone FILE, pinned by the same hash
        assert s1["backbone_spec"] == s2["backbone_spec"]
        assert s1["backbone_spec"]["sha256"] is not None
        # identical schedule and model
        assert s1["train"] == s2["train"]
        assert s1["model"] == s2["model"]
        assert s1["eval"] == s2["eval"]
        assert s1["task"] == s2["task"] == "detection"


def test_seed2_launchers_exist_and_are_wired_to_the_right_run(matrix):
    for variant in ("baseline", "saga"):
        run_id = f"det_vitb_{variant}_s2"
        job = REPO / "scripts" / "jobs" / f"{run_id}.sbatch"
        sub = REPO / "scripts" / f"submit_{run_id}.sh"
        assert job.exists() and sub.exists(), run_id
        text = job.read_text()
        assert f"--run {run_id} \\" in text
        assert f"--master_port={matrix['runs'][run_id]['port']}" in text
        assert f"--job-name={run_id}" in text
        assert "--resume auto" in text
        assert f"scripts/jobs/{run_id}.sbatch" in sub.read_text()


def test_seed2_did_not_disturb_the_runs_that_already_executed():
    """Adding runs must not renumber or rewrite a launcher whose results are
    already committed — that is why ports are pinned in the matrix."""
    executed = ["det_vitb_baseline_s1", "det_vitb_saga_s1",
                "det_vitb_registers_s1", "seg_vitb_baseline_s1",
                "seg_vitb_saga_s1", "seg_vitb_registers_s1"]
    rc = subprocess.run(["git", "diff", "--name-only", "HEAD", "--"]
                        + [f"scripts/jobs/{r}.sbatch" for r in executed]
                        + [f"scripts/submit_{r}.sh" for r in executed],
                        cwd=REPO, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert rc.stdout.strip() == "", (
        "the already-executed runs' launchers changed:\n" + rc.stdout)


def test_every_run_pins_a_unique_port(matrix):
    ports = [r["port"] for r in matrix["runs"].values()]
    assert all(isinstance(p, int) for p in ports)
    assert len(set(ports)) == len(ports)


def test_dense_jobs_put_set_u_after_the_env_sourcing():
    """`/etc/profile`'s site scripts reference unset variables, so `set -u`
    above the env sourcing aborts the job. TASK-10's equivalent check
    (tests/test_task10_ttr.py) matches the line `set -u` EXACTLY and
    therefore SKIPS every dense job file, whose line carries a trailing
    comment — so the property is asserted here instead of going unchecked.
    """
    jobs = [j for j in (REPO / "scripts" / "jobs").glob("*.sbatch")
            if j.name.startswith(("det_vitb_", "seg_vitb_", "dense_smoke_"))]
    assert jobs
    for job in jobs:
        lines = job.read_text().splitlines()
        i_setu = next(i for i, l in enumerate(lines)
                      if l.strip().startswith("set -u"))
        i_src = next(i for i, l in enumerate(lines)
                     if l.strip().startswith("source ")
                     and "env_alex.sh" in l)
        assert i_setu > i_src, (
            f"{job.name}: `set -u` on line {i_setu + 1} is above the env "
            f"sourcing on line {i_src + 1}")


# ── prepared-but-not-submitted runs, and path hygiene ────────────────────

def test_prepared_but_unsubmitted_runs_stay_out_of_the_tables(matrix):
    """`submitted: false` means "never started", which is a different fact
    from "expected but absent". A MISSING row for such a run reads as a
    failure, so it is excluded — while a run that IS expected and absent
    must still yield MISSING."""
    unsubmitted = {r for r, v in matrix["runs"].items()
                   if not v.get("submitted", True)}
    assert unsubmitted == {"det_vitb_baseline_s2", "det_vitb_saga_s2"}

    for path in ("T4_coco.csv", "T5_ade20k.csv"):
        rows = read_csv(TABLES / path)
        assert not (unsubmitted & {r["run_id"] for r in rows}), path

    # the MISSING guard is still live for an expected-but-absent run
    m2 = yaml.safe_load(open(MATRIX))
    m2["runs"]["det_vitb_baseline_s2"]["submitted"] = True
    loaded = bdt.load_detection(DET_ROOT, m2)
    rows = bdt.det_rows(loaded, m2)
    ghost = [r for r in rows if r["run_id"] == "det_vitb_baseline_s2"]
    assert ghost and ghost[0]["AP"] == bdt.MISSING
    assert "not found" in ghost[0]["note"]


def test_no_committed_artifact_leaks_an_absolute_path():
    """A committed results file must not carry a machine-specific path."""
    import re
    pattern = re.compile(r"[A-Za-z]:[\/]|/home/[a-z]|/Users/")
    targets = list(TABLES.glob("T4*.csv")) + list(TABLES.glob("T5*.csv")) + [
        NOTE, REPO / "results" / "figures_data" / "F7_coco_selection.json"]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        hits = [l for l in text.splitlines() if pattern.search(l)]
        assert not hits, f"{path.name}: {hits[:2]}"


def test_f7_coco_half_is_present_and_matches_the_frozen_selection():
    """The exported crops must be the images the committed selection names,
    at the geometry it froze — not a re-pick made on the HPC."""
    coco = REPO / "results" / "figures_data" / "f7_coco"
    sel = json.loads((REPO / "results" / "figures_data"
                      / "F7_coco_selection.json").read_text(encoding="utf-8"))
    payload = json.loads((coco / "crops.json").read_text(encoding="utf-8"))

    assert ({r["image_id"] for r in payload["images"]}
            == {i["image_id"] for i in sel["images"]})
    assert payload["selection_rule"] == sel["rule"]
    by_id = {i["image_id"]: i for i in sel["images"]}
    for rec in payload["images"]:
        assert (coco / rec["crop_file"]).exists()
        # the exporter may only CLIP the frozen crop, never move it
        fx, fy, fw, fh = by_id[rec["image_id"]]["crop_xywh"]
        cx, cy, cw, ch = rec["crop_xywh_original"]
        assert cx >= fx - 1e-6 and cy >= fy - 1e-6
        assert cx + cw <= fx + fw + 1e-6 and cy + ch <= fy + fh + 1e-6
        assert rec["n_small"] == by_id[rec["image_id"]]["n_small"]
        # every drawn detection count matches the selection's own tally
        for variant, dets in rec["detections_in_crop"].items():
            assert len(dets) == rec["n_small"][variant]
