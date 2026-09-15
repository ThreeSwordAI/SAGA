"""TASK-12 Phase C — tables, address correlation, note, figure, derive job.

The properties defended here are the ones that would let a wrong number into
the paper: a MISSING cell must never become a value, a delta must never be
computed against a MISSING or mismatched side, a non-spatial arm must never
be given a correlation of 0.0, the note must contain nothing hand-typed, and
the figure must say which quantity it is plotting.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.build_ablation_tables import build_rows, MISSING  # noqa: E402
from analysis import build_ablation_note as note_mod            # noqa: E402

MATRIX = yaml.safe_load((REPO / "configs" / "abl_matrix.yaml").read_text())
T2 = REPO / "results" / "tables" / "T2_ablation.csv"
T2_ADDR = REPO / "results" / "tables" / "T2_ablation_address.csv"
NOTE = REPO / "results" / "notes" / "ablation.md"
ARMS = list(MATRIX["runs"])

# ViT-S/16: the counts TASK-12's design table promises a reviewer
EXPECTED_GATE_PARAMS = {
    "A": (0, 0), "B": (14112, 0), "C": (72, 72),
    "D": (4608, 4608), "E": (14112, 14112), "F": (14112, 14112),
}


# ─────────────────────────────────────────────────────────────────────────────
# synthetic run tree — lets the tests exercise the DERIVED state that does not
# exist locally yet, without inventing a number in a committed file
# ─────────────────────────────────────────────────────────────────────────────

def make_runs(root: Path, *, derived=True, top1=None, sha_mismatch=(),
              canon=True):
    top1 = top1 or {}
    for run_id, run in MATRIX["runs"].items():
        mode = run.get("gate_mode",
                       "spatial" if run["variant"] == "saga" else "none")
        d = root / run_id
        (d / "diag").mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps(
            {"run_id": run_id, "seed": 0, "end_time": "2026-09-14T00:00:00",
             "ckpt_dir": f"/home/woody/abl_ckpt/{run_id}/ckpt"}))
        cfg = {"model": {"arch": "vit_small_patch16_224", "gate_mode": mode,
                         "gate": mode != "none"},
               "variant": run["variant"], "recipe": "mixup", "seed": 0,
               "train": {"epochs": 100},
               "knobs": {"gate_init_logit": float(
                   run.get("gate_init_logit", 0.0))}}
        (d / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
        with open(d / "log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "lr", "train_loss", "val_top1_full",
                        "val_top5_full", "val_loss", "img_per_sec",
                        "wall_time"])
            for e in range(100):
                w.writerow([e, 1e-6, 3.0, 70.0 + e / 100, 90.0, 1.2,
                            8000.0, 160.0])
        slots = {"const": 196, "headscalar": 1, "spatial": 196}.get(mode)
        if slots:
            (d / "gates").mkdir()
            init = float(run.get("gate_init_logit", 0.0))
            rng = np.random.RandomState(abs(hash(run_id)) % 2**31)
            phi = np.full((12, 6, slots), init, dtype=np.float32)
            if mode != "const":
                phi = phi + rng.normal(0, 0.3, phi.shape).astype(np.float32)
            np.savez(d / "gates" / "phi_e099.npz", phi=phi)
        if not derived:
            continue
        sha = "a" * 64 if run_id not in sha_mismatch else "b" * 64
        (d / "eval").mkdir()
        (d / "eval" / "imagenet_val_last.json").write_text(json.dumps(
            {"top1": top1.get(run_id, 76.0), "top5": 93.0,
             "n_images": 50000, "ckpt_sha256": "a" * 64}))
        diag = {"ckpt_sha256": sha, "sink_mad_k5": 5.0,
                "oversmooth_pairwise": 0.28,
                "oversmooth_pairwise_nosink": 0.27, "eff_rank": 120.0}
        if canon:
            diag.update(sink_fixed_canon=4.0, canon_thr_value=20.8515625)
        (d / "diag" / "diag_final_last.json").write_text(json.dumps(diag))
    return root


def rows_for(root):
    return build_rows(root, MATRIX)[0]


# ─────────────────────────────────────────────────────────────────────────────
# T2 — parameter counts, MISSING discipline, deltas
# ─────────────────────────────────────────────────────────────────────────────

def test_committed_table_has_the_six_arms_and_pinned_param_counts():
    rows = list(csv.DictReader(open(T2, newline="", encoding="utf-8")))
    assert [r["arm"] for r in rows] == ["A", "B", "C", "D", "E", "F"]
    for r in rows:
        got = (int(r["gate_params_registered"]),
               int(r["gate_params_trainable"]))
        assert got == EXPECTED_GATE_PARAMS[r["arm"]], r["arm"]
    # the structural check: 0 spatial std where the arm cannot vary
    by_arm = {r["arm"]: r for r in rows}
    assert float(by_arm["B"]["gate_spatial_std_final"]) == 0.0
    assert float(by_arm["C"]["gate_spatial_std_final"]) == 0.0
    assert float(by_arm["E"]["gate_spatial_std_final"]) > 0.0
    assert float(by_arm["F"]["gate_spatial_std_final"]) > 0.0


def test_undrived_runs_yield_MISSING_and_name_the_job(tmp_path):
    rows = rows_for(make_runs(tmp_path, derived=False))
    for r in rows:
        assert r["top1_last"] == MISSING
        assert r["sink_fixed_canon"] == MISSING
        assert r["delta_top1_vs_A"] == MISSING
        assert "abl_derive.sbatch" in r["note"]
        # the bf16 column is still populated — and kept separate
        assert r["top1_last_bf16_log"] != MISSING


def test_deltas_appear_only_when_both_sides_are_present(tmp_path):
    rows = rows_for(make_runs(tmp_path, top1={
        "abl_vits_mixup_baseline_s0": 76.5,
        "abl_vits_mixup_spatial_i0_s0": 76.9}))
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["E"]["delta_top1_vs_A"] == pytest.approx(0.4)
    assert by_arm["A"]["delta_top1_vs_A"] == pytest.approx(0.0)


def test_eval_and_diag_from_different_checkpoints_void_the_row(tmp_path):
    """A results row must never mix two checkpoints. This is the decoy: the
    files are both present and both well-formed, only their shas disagree."""
    root = make_runs(tmp_path, sha_mismatch=("abl_vits_mixup_spatial_i0_s0",))
    by_arm = {r["arm"]: r for r in rows_for(root)}
    assert by_arm["E"]["top1_last"] == MISSING
    assert by_arm["E"]["sink_fixed_canon"] == MISSING
    assert "DIFFERENT checkpoints" in by_arm["E"]["note"]
    assert by_arm["E"]["delta_top1_vs_A"] == MISSING
    assert by_arm["D"]["top1_last"] != MISSING      # neighbours unaffected


def test_intrain_diagnostics_never_leak_into_the_canonical_columns(tmp_path):
    """The trainer's periodic diag and the canonical pass are different
    files. The table may carry both; it may never merge them, and the note
    must say which one it is showing."""
    root = make_runs(tmp_path, derived=False)
    for run_id in ARMS:
        (root / run_id / "diag" / "diag_e099.json").write_text(json.dumps(
            {"epoch": 99, "sink_mad_k5": 7.5, "oversmooth_pairwise": 0.29,
             "oversmooth_pairwise_nosink": 0.28, "eff_rank": 118.0}))
    rows = rows_for(root)
    for r in rows:
        assert r["sink_mad_k5"] == MISSING          # canonical stays MISSING
        assert float(r["sink_mad_k5_intrain"]) == 7.5
        assert int(r["intrain_diag_epoch"]) == 99

    table = tmp_path / "t2.csv"
    from analysis.build_ablation_tables import FIELDS
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    text = note_mod.build(note_mod.load(table), note_mod.load(T2_ADDR),
                          note_mod.load(REPO / "results" / "tables"
                                        / "e2_pooled.csv"), "2026-01-01")
    assert "trainer's OWN periodic diagnostics at epoch 99" in text
    assert "7.5000" in text

    # once the canonical pass exists, the note must show THAT instead
    root2 = make_runs(tmp_path / "derived")
    rows2 = rows_for(root2)
    assert all(r["sink_mad_k5"] != MISSING for r in rows2)
    table2 = tmp_path / "t2b.csv"
    with open(table2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows2)
    text2 = note_mod.build(note_mod.load(table2), note_mod.load(T2_ADDR),
                           note_mod.load(REPO / "results" / "tables"
                                         / "e2_pooled.csv"), "2026-01-01")
    assert "trainer's OWN periodic diagnostics" not in text2


def test_missing_canon_backfill_is_named_not_silently_empty(tmp_path):
    rows = rows_for(make_runs(tmp_path, canon=False))
    for r in rows:
        assert r["sink_fixed_canon"] == MISSING
        assert "apply_fixed_thr" in r["note"]
        assert r["sink_mad_k5"] != MISSING          # the secondary survives


# ─────────────────────────────────────────────────────────────────────────────
# address — the null must be TASK-07's, and a constant gate gets no number
# ─────────────────────────────────────────────────────────────────────────────

def _run_address(root, out):
    return subprocess.run(
        [sys.executable, str(REPO / "analysis" / "ablation_address.py"),
         "--runs-root", str(root), "--matrix", str(REPO / "configs"
                                                   / "abl_matrix.yaml"),
         "--out", str(out)], cwd=str(REPO), capture_output=True, text=True)


def test_address_uses_the_exact_permutation_null_not_the_iid_one(tmp_path):
    """TASK-07 measured the spatial null's sd at 0.157-0.342 against the iid
    0.0716; judging these smooth maps against the iid reference inflates
    significance several-fold."""
    root = make_runs(tmp_path, derived=False)
    rng = np.random.RandomState(0)
    smooth = np.zeros((14, 14))
    smooth[0, :] = smooth[-1, :] = smooth[:, 0] = smooth[:, -1] = 1.0
    freq = (smooth + rng.normal(0, 0.05, (14, 14))).ravel()
    for run_id in ARMS:
        (root / run_id / "diag" / "diag_final_last_addr.json").write_text(
            json.dumps({"freq_canon": freq.tolist(),
                        "freq_mad": freq.tolist()}))
    # the gate must be SPATIALLY SMOOTH too — the permutation null is only
    # wider than the iid one when both maps carry spatial autocorrelation,
    # and the real gates do (see results/figures/F_ablation_draft.pdf).
    # A white-noise gate would legitimately give an iid-sized null.
    ring = np.zeros((14, 14))
    ring[1, 1:-1] = ring[-2, 1:-1] = ring[1:-1, 1] = ring[1:-1, -2] = 1.0
    for run_id, run in MATRIX["runs"].items():
        if run.get("gate_mode") != "spatial":
            continue
        phi = (ring.ravel()[None, None, :] * 0.8
               + rng.normal(0, 0.02, (12, 6, 196))).astype(np.float32)
        np.savez(root / run_id / "gates" / "phi_e099.npz", phi=phi)
    out = tmp_path / "addr.csv"
    res = _run_address(root, out)
    assert res.returncode == 0, res.stdout + res.stderr
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))

    spatial = [r for r in rows if r["arm"] == "E" and r["rho"] != MISSING]
    assert spatial, "the spatial arm must get a correlation"
    iid = 1.0 / np.sqrt(196 - 1)          # 0.0716, the reference TASK-07 rejects
    for r in spatial:
        assert int(r["n_transforms"]) == 8 * 196
        # how far above the iid sd the permutation null sits depends on how
        # smooth the two maps are (TASK-07 measured 0.157-0.342 on the real
        # ones, i.e. 2-5x); this synthetic ring is thinner than a real gate,
        # so the bar here is only that the null is meaningfully wider than iid
        assert float(r["null_sd"]) > iid * 1.2, \
            "null sd looks like the iid reference, not the spatial one"
        assert float(r["p_spatial"]) >= 1.0 / (8 * 196)   # identity is in it


@pytest.mark.parametrize("arm", ["B", "C"])
def test_constant_gate_arms_get_no_correlation_only_words(tmp_path, arm):
    root = make_runs(tmp_path, derived=False)
    freq = np.random.RandomState(1).rand(196)
    for run_id in ARMS:
        (root / run_id / "diag" / "diag_final_last_addr.json").write_text(
            json.dumps({"freq_canon": freq.tolist(),
                        "freq_mad": freq.tolist()}))
    out = tmp_path / "addr.csv"
    assert _run_address(root, out).returncode == 0
    rows = [r for r in csv.DictReader(open(out, newline="", encoding="utf-8"))
            if r["arm"] == arm]
    assert rows
    for r in rows:
        assert r["rho"] == MISSING, "a constant gate must not get a number"
        assert r["p_spatial"] == MISSING
        assert "undefined, not zero" in r["note"]


def test_committed_address_table_separates_its_three_blank_reasons():
    """A blank rho has three causes and they mean different things: the gate
    is constant by construction, the ADDRESS map is degenerate, or one layer
    never moved from its init. Conflating them would either credit or accuse
    an arm wrongly."""
    rows = list(csv.DictReader(open(T2_ADDR, newline="", encoding="utf-8")))
    assert rows
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["gate_mode"], []).append(r)

    # the spatial arms get real correlations on a basis that has a map
    spatial = [r for r in by_mode["spatial"] if r["rho"] != MISSING]
    assert spatial, "the spatial arms must be correlated where a map exists"
    assert all(r["map_basis"] == "mad" for r in spatial), \
        "only the mad basis has a usable map in this cell"

    # a degenerate address map is attributed to the MAP, never to the gate
    degen = [r for r in rows if "ADDRESS MAP is constant" in r["note"]]
    assert degen and all(r["rho"] == MISSING for r in degen)
    assert all(r["map_basis"] == "canon" for r in degen)
    assert any(r["gate_mode"] == "spatial" for r in degen), \
        "the spatial arms also meet the degenerate canon map"

    # a constant-by-construction gate never gets a number
    for mode in ("const", "headscalar"):
        assert all(r["rho"] == MISSING for r in by_mode[mode])
        assert any("undefined, not zero" in r["note"] for r in by_mode[mode])


# ─────────────────────────────────────────────────────────────────────────────
# the note is generated, answers the four questions, and hides nothing
# ─────────────────────────────────────────────────────────────────────────────

def _committed_note_date():
    for line in NOTE.read_text(encoding="utf-8").splitlines():
        if line.startswith("Generated by"):
            return line.rstrip(".").split(" on ")[1].split(".")[0].strip()
    raise AssertionError("no generation date in the committed note")


def test_note_is_reproducible_from_the_committed_tables():
    """Nothing in the note may be hand-typed: re-rendering it from the
    committed tables must give back the committed bytes."""
    text = note_mod.build(note_mod.load(T2), note_mod.load(T2_ADDR),
                          note_mod.load(REPO / "results" / "tables"
                                        / "e2_pooled.csv"),
                          _committed_note_date())
    assert text == NOTE.read_text(encoding="utf-8")


def test_note_answers_the_four_questions_or_says_pending():
    text = NOTE.read_text(encoding="utf-8")
    for heading in ("Q1 — does the spatial arm beat",
                    "Q2 — does the init matter",
                    "Q3 — is the 100-epoch ranking consistent",
                    "Caveats"):
        assert heading in text, heading
    for caveat in ("One seed per arm", "100 epochs, not the headline's 300",
                   "One cell"):
        assert caveat in text, caveat
    # anything still PENDING must name the job that fills it, rather than
    # being answered from a substitute number
    if "PENDING" in text:
        assert "abl_derive.sbatch" in text
    assert "NOT CANONICAL" not in text.split("## 8.")[0], \
        "no provisional value may appear before the provenance section"
    # a saturated threshold is a measured result, not a pending one, and
    # must be labelled as ordering nothing rather than quietly compared
    rows = list(csv.DictReader(open(T2, newline="", encoding="utf-8")))
    if any(str(r["sink_canon_saturated"]).lower() == "true" for r in rows):
        assert "SATURATE" in text
        assert "orders nothing" in text
        assert "Nothing was recalibrated" in text


def test_note_flags_a_sign_inversion_loudly(tmp_path):
    """Synthetic derived state where E lands below A: the note must SHOUT,
    and must also calibrate against the headline cell's own repeat spread."""
    root = make_runs(tmp_path, top1={"abl_vits_mixup_baseline_s0": 76.5,
                                     "abl_vits_mixup_spatial_i0_s0": 76.4})
    table_rows = rows_for(root)
    table = tmp_path / "t2.csv"
    with open(table, "w", newline="", encoding="utf-8") as f:
        from analysis.build_ablation_tables import FIELDS
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(table_rows)
    text = note_mod.build(note_mod.load(table), note_mod.load(T2_ADDR),
                          note_mod.load(REPO / "results" / "tables"
                                        / "e2_pooled.csv"), "2026-01-01")
    assert "SIGN INVERSION AGAINST THE HEADLINE" in text
    assert "was itself negative" in text      # the calibration, not just alarm
    assert "one seed" in text.lower()


def test_note_and_tables_carry_no_machine_specific_paths():
    """TASK-09 leaked an absolute Windows path into a committed table."""
    for path in (T2, T2_ADDR, NOTE):
        text = path.read_text(encoding="utf-8")
        assert ":\\" not in text, path.name
        assert "/Users/" not in text, path.name
        assert "C:/" not in text, path.name


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────

def test_figure_renders_and_says_which_quantity_it_plots(tmp_path):
    out = tmp_path / "fig.pdf"
    res = subprocess.run(
        [sys.executable, str(REPO / "plotting" / "plot_T2.py"),
         "--table", str(T2), "--out", str(out)],
        cwd=str(REPO), capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.exists() and out.stat().st_size > 5000
    assert "NOT CANONICAL" in res.stdout or "fp32" in res.stdout


# ─────────────────────────────────────────────────────────────────────────────
# the derivation job
# ─────────────────────────────────────────────────────────────────────────────

def test_derive_job_contract():
    src = (REPO / "scripts" / "jobs" / "abl_derive.sbatch").read_text(
        encoding="utf-8")
    lines = src.splitlines()

    def first(pred):
        return next(i for i, l in enumerate(lines) if pred(l))

    i_src = first(lambda l: l.strip().startswith("source ")
                  and "env_alex.sh" in l)
    i_setu = first(lambda l: l.strip() == "set -u")
    i_torch = first(lambda l: "import torch, timm" in l)
    i_trap = first(lambda l: l.strip() == "trap cleanup_probe EXIT")
    i_stage = first(lambda l: l.strip() == "stage_probe_imagenet_val")
    assert i_setu > i_src
    assert i_torch < i_trap < i_stage
    assert "/home/vault/iwi5/iwi5359h/envs/saga/bin/python" in src
    assert not [l for l in lines
                if not l.lstrip().startswith("#") and "which python" in l]
    assert src.count("cleanup_probe") == 1

    # the canon backfill must precede the address maps: sink_address.py
    # hard-checks its mass against the field apply_fixed_thr writes
    assert src.index("tools/apply_fixed_thr.py") < \
        src.index("tools/sink_address.py")
    assert "--version canon" in src
    assert "fixed_thresholds_canon.json" in src
    # nothing may recalibrate a threshold here
    assert "compute_fixed_thr" not in src
    assert "--pattern 'abl_*'" in src
