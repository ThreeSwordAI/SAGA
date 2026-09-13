"""
tests/test_task10_phasec.py
===========================
TASK-10 PHASE C: the paired-CI statistics, T_ttr.csv, and the generated note.

The three standing decisions are pinned here so they cannot erode:
  1. the A3 threshold is frozen at 1.00 and the note never recomputes a verdict
  2. every cell appears with its own verdict, and absolute sink counts sit
     beside every percentage
  3. no pooled address correlation
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.build_ttr_tables import CELLS, addr_rho, find_by_sha
from tools.ttr_paired_ci import paired_stats

REPO = Path(__file__).resolve().parents[1]
T_TTR = REPO / "results" / "tables" / "T_ttr.csv"
SWEEP = REPO / "results" / "tables" / "T_ttr_sweep.csv"
NOTE = REPO / "results" / "notes" / "ttr_baseline.md"
MISSING = "MISSING"


def rows():
    return list(csv.DictReader(open(T_TTR, encoding="utf-8")))


# ─────────────────────────────────────────────────────────────────────────────
# The paired statistics (decision 1)
# ─────────────────────────────────────────────────────────────────────────────

def _table(n, b, c, n11):
    unp = np.zeros(n, bool)
    pat = np.zeros(n, bool)
    unp[:b] = True
    pat[b:b + c] = True
    unp[b + c:b + c + n11] = True
    pat[b + c:b + c + n11] = True
    return unp, pat


def test_paired_stats_recovers_the_discordant_table_and_drop():
    n, b, c, n11 = 50000, 600, 90, 40000
    s = paired_stats(*_table(n, b, c, n11), n_boot=20000, seed=0)
    assert s["b_unpatched_correct_patched_wrong"] == b
    assert s["c_unpatched_wrong_patched_correct"] == c
    assert s["n_discordant"] == b + c
    assert s["n_both_correct"] == n11
    assert s["top1_unpatched"] == pytest.approx(100 * (n11 + b) / n)
    assert s["top1_patched"] == pytest.approx(100 * (n11 + c) / n)
    # drop is positive when TTR costs accuracy, matching the gate convention
    assert s["top1_drop"] == pytest.approx(100 * (b - c) / n)


def test_paired_bootstrap_matches_literal_resampling():
    """The multinomial shortcut must equal resampling the paired rows.

    Only (b, c, concordant) carry information, so a bootstrap resample is
    exactly a multinomial draw — this checks that claim rather than assuming
    it.
    """
    n, b, c = 50000, 600, 90
    unp, pat = _table(n, b, c, 40000)
    s = paired_stats(unp, pat, n_boot=50000, seed=0)

    d = pat.astype(np.int8) - unp.astype(np.int8)
    rng = np.random.default_rng(7)
    lit = np.array([-100.0 * d[rng.integers(0, n, n)].mean()
                    for _ in range(2000)])
    se_multinomial = s["bootstrap"]["se"]
    assert lit.std(ddof=1) == pytest.approx(se_multinomial, rel=0.10)
    assert lit.mean() == pytest.approx(s["top1_drop"], abs=4 * se_multinomial)


def test_paired_stats_degenerate_and_symmetric_cases():
    n = 1000
    unp, _ = _table(n, 100, 0, 500)
    same = paired_stats(unp, unp.copy(), n_boot=500, seed=0)
    assert same["top1_drop"] == 0.0
    assert same["n_discordant"] == 0
    assert same["bootstrap"]["ci_low"] == same["bootstrap"]["ci_high"] == 0.0
    assert same["mcnemar"]["p_value"] is None      # no discordant pairs

    # TTR helping is a NEGATIVE drop
    up = paired_stats(*_table(n, 10, 40, 500), n_boot=2000, seed=0)
    assert up["top1_drop"] < 0


def test_paired_stats_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        paired_stats(np.zeros(10, bool), np.zeros(11, bool), 10, 0)


# ─────────────────────────────────────────────────────────────────────────────
# T_ttr.csv (decision 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_every_cell_appears_with_its_own_verdict():
    """Decision 2: excluding failures would be selection on outcome."""
    rs = rows()
    assert len(rs) == len(CELLS)
    verdicts = {r["cell"]: r["gate_verdict"] for r in rs}
    assert set(verdicts.values()) <= {"PASS", "FAIL"}
    assert "PASS" in verdicts.values() and "FAIL" in verdicts.values(), (
        "this table is supposed to contain both outcomes")
    for r in rs:
        assert r["gate_verdict"], f"{r['cell']} has no verdict"


def test_absolute_sink_counts_sit_beside_every_percentage():
    """Decision 2: 42% of 3.70 is not comparable with 55% of 19.71."""
    for r in rows():
        assert r["sink_canon_baseline"] not in ("", MISSING)
        assert r["sink_canon_ttr"] not in ("", MISSING)
        assert r["sink_canon_removed_abs"] not in ("", MISSING)
        removed = float(r["sink_canon_removed_abs"])
        base, ttr = float(r["sink_canon_baseline"]), float(r["sink_canon_ttr"])
        assert removed == pytest.approx(base - ttr, abs=1e-9)
        frac = float(r["sink_canon_reduction_frac"])
        assert frac == pytest.approx(removed / base, abs=1e-9)


def test_table_values_match_their_source_jsons():
    """Independent recompute straight from the run artifacts."""
    for r in rows():
        run = REPO / "results" / "runs" / f"ttr_{r['base_run_id']}"
        ev = json.load(open(run / "eval" / "eval_last.json"))
        dg = json.load(open(run / "diag" / "diag_last.json"))
        assert float(r["top1_ttr"]) == pytest.approx(ev["top1"])
        assert int(r["n_images_eval"]) == ev["n_images"] == 50000
        assert float(r["sink_canon_ttr"]) == pytest.approx(dg["sink_fixed_canon"])
        assert float(r["oversmooth_pairwise_ttr"]) == pytest.approx(
            dg["oversmooth_pairwise"])
        assert float(r["eff_rank_ttr"]) == pytest.approx(dg["eff_rank"])
        assert int(r["num_prefix_tokens_ttr"]) == dg["num_prefix_tokens"] == 2
        assert r["ckpt_sha256"] == ev["ckpt_sha256"] == dg["ckpt_sha256"]
        # the delta is against the sha-matched unpatched baseline
        if r["top1_baseline"] not in ("", MISSING):
            assert float(r["top1_delta_vs_baseline"]) == pytest.approx(
                float(r["top1_ttr"]) - float(r["top1_baseline"]), abs=1e-9)
            assert float(r["top1_drop_vs_baseline"]) == pytest.approx(
                -float(r["top1_delta_vs_baseline"]), abs=1e-9)


def test_tau_is_the_committed_canon_value_and_never_recalibrated():
    canon = json.load(open(REPO / "results/diagsplit/fixed_thresholds_canon.json"))
    for r in rows():
        key = f"{r['arch']}|{r['recipe_actual']}"
        assert float(r["canon_tau"]) == canon[key]
        assert str(r["tau_recalibrated"]).lower() == "false"


def test_missing_is_never_imputed_for_the_paired_ci():
    """Until the HPC run lands, those columns must be MISSING — not zero,
    not blank, not silently dropped."""
    for r in rows():
        cols = ["paired_drop", "paired_ci_low", "paired_ci_high",
                "paired_b_broke", "paired_c_fixed", "mcnemar_p"]
        vals = [r[c] for c in cols]
        assert all(v == MISSING for v in vals) or all(v != MISSING for v in vals), (
            f"{r['cell']}: paired CI columns are partially filled: {vals}")


def test_pairing_is_by_checkpoint_sha_not_filename(tmp_path):
    """A decoy artifact with the wrong sha must not be picked up."""
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "aaa_first.json").write_text(
        json.dumps({"ckpt_sha256": "ff" * 32, "top1": 1.0}))
    (tmp_path / "eval" / "zzz_last.json").write_text(
        json.dumps({"ckpt_sha256": "ab" * 32, "top1": 2.0}))
    hit, err = find_by_sha(tmp_path, "eval", "ab" * 32)
    assert err is None and hit.name == "zzz_last.json"
    none, err = find_by_sha(tmp_path, "eval", "cc" * 32)
    assert none is None and "need exactly 1" in err


def test_sweep_table_carries_absolute_removals_and_the_unpatched_row():
    sweep = list(csv.DictReader(open(SWEEP, encoding="utf-8")))
    assert sweep
    by_cell = {}
    for s in sweep:
        by_cell.setdefault(s["cell"], []).append(s)
    for cell, ss in by_cell.items():
        ns = [int(s["n_neurons"]) for s in ss]
        assert 0 in ns, f"{cell}: no unpatched n=0 reference row"
        for s in ss:
            if int(s["n_neurons"]) > 0:
                assert s["sink_canon_removed_abs"] != MISSING


# ─────────────────────────────────────────────────────────────────────────────
# The address correlation (decision from the human: do not pool)
# ─────────────────────────────────────────────────────────────────────────────

def test_addr_rho_uses_the_spatial_permutation_null_not_iid():
    """TASK-07: the iid reference (sd 0.0716 at n=196) is far too generous
    for these smooth maps; the permutation null's sd is 0.157-0.342."""
    for r in rows():
        if r["addr_rho_canon_null_sd"] in ("", MISSING):
            continue
        sd = float(r["addr_rho_canon_null_sd"])
        assert sd > 0.0716 * 1.5, (
            f"{r['cell']}: null sd {sd} looks like the iid reference")
        p = float(r["addr_rho_canon_p"])
        assert 0.0 < p <= 1.0


def test_addr_rho_is_deterministic_and_symmetric():
    rng = np.random.default_rng(0)
    a = rng.random(196)
    b = rng.random(196)
    r1, p1, sd1 = addr_rho(a, b)
    r2, p2, sd2 = addr_rho(a, b)
    assert (r1, p1, sd1) == (r2, p2, sd2)
    assert addr_rho(a, a)[0] == pytest.approx(1.0)


def test_note_reports_no_pooled_address_correlation():
    text = NOTE.read_text(encoding="utf-8")
    assert "**No pooled number is reported.**" in text
    assert "flat by construction" in text
    # each cell's rho appears individually
    for r in rows():
        if r["addr_rho_canon"] not in ("", MISSING):
            assert f"{float(r['addr_rho_canon']):.4f}".lstrip("-") in text


# ─────────────────────────────────────────────────────────────────────────────
# The note (decision 1: threshold held)
# ─────────────────────────────────────────────────────────────────────────────

def test_note_holds_the_threshold_and_does_not_recompute_a_verdict():
    text = NOTE.read_text(encoding="utf-8")
    assert "is held at 1.00 and is not revisited" in text
    assert "does not reopen it" in text
    assert "does not reinterpret the verdict" in text
    assert "**This does not make it a pass.**" in text
    # the verdicts in the note are the ones the gate recorded
    for r in rows():
        assert f"| {r['cell']} | **{r['gate_verdict']}** |" in text


def test_note_states_the_provenance_unambiguously():
    text = NOTE.read_text(encoding="utf-8")
    assert "REIMPLEMENTED, not vendored" in text
    assert "no line of it was copied" in text
    assert "860df43515c8d8e9e90952af25a46c26e4469570" in text
    assert "mean_abs_act_at_outliers_v1" in text


def test_note_carries_the_replication_variance_sentence():
    text = NOTE.read_text(encoding="utf-8")
    assert "**Replication variance.**" in text
    assert "3 of 5" in text and "5 of 5" in text


def test_note_regenerates_identically_from_the_committed_tables(tmp_path):
    """The note is generated, not edited: re-running must reproduce it.

    Generated into tmp_path via --out, never over the committed file — a
    test that rewrites a results file would dirty the working tree on every
    run (the `*Generated ...*` timestamp alone is enough to do it).
    """
    scratch = tmp_path / "note.md"
    proc = subprocess.run(
        [sys.executable, "analysis/build_ttr_note.py",
         "--out", str(scratch)],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    def strip_ts(t):
        return "\n".join(l for l in t.splitlines()
                         if not l.startswith("*Generated "))
    assert strip_ts(scratch.read_text(encoding="utf-8")) == \
        strip_ts(NOTE.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# tools/ttr_paired_ci.py end to end
# ─────────────────────────────────────────────────────────────────────────────

def test_paired_ci_cli_end_to_end(tmp_path, monkeypatch):
    """The real CLI on fake data: two passes over the SAME images, the npz of
    per-image outcomes, and a JSON whose counts reconstruct the marginals."""
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPO))
    from saga.ttr import CRITERION_MEAN_ABS, write_neurons_json
    from tools.model_factory import build_model

    data = tmp_path / "data"
    items = []
    for c in range(2):
        d = data / "val" / f"n{c:08d}"
        d.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (16, 16), (c * 40, i * 60, 90)).save(
                d / f"img{i:02d}.JPEG")
            items.append(1)

    torch.manual_seed(0)
    ckpt = tmp_path / "last.pth"
    torch.save({"model": build_model("vit_small", "baseline").state_dict()}, ckpt)

    write_neurons_json(
        tmp_path / "neurons.json", [(5, 1, 9.0), (6, 2, 8.0), (7, 3, 7.0)],
        criterion=CRITERION_MEAN_ABS, outlier_tau=20.85,
        tau_key="vit_small|mixup", tau_source="x", seed=0, arch="vit_small",
        variant="baseline", ckpt=str(ckpt), ckpt_sha256="ab" * 32,
        scan_stats=dict(n_images_seen=6, n_images_scored=6,
                        layer_range=[3, 12], num_layers_scanned=9,
                        num_neurons_per_layer=1536, num_prefix_tokens=1,
                        detect_outliers_layer=-1),
        top_n_stored=8)

    import tools.ttr_paired_ci as mod
    monkeypatch.setattr(mod, "IMAGENET_VAL_SIZE", len(items))
    monkeypatch.chdir(REPO)
    out = tmp_path / "paired_ci.json"
    monkeypatch.setattr(sys, "argv", [
        "ttr_paired_ci.py", "--ckpt", str(ckpt), "--arch", "vit_small",
        "--recipe", "mixup", "--neurons-file", str(tmp_path / "neurons.json"),
        "--n-neurons", "3", "--data", str(data), "--out", str(out),
        "--device", "cpu", "--num-workers", "0", "--batch-size", "2",
        "--n-boot", "2000", "--seed", "0"])
    assert mod.main() == 0

    r = json.loads(out.read_text())
    assert r["schema"] == "saga.ttr.paired_ci/v1"
    assert r["n_images"] == len(items)
    # the 2x2 table must partition the sample exactly
    assert (r["n_both_correct"] + r["n_both_wrong"]
            + r["b_unpatched_correct_patched_wrong"]
            + r["c_unpatched_wrong_patched_correct"]) == len(items)
    # and reconstruct both marginals
    n = len(items)
    assert r["top1_unpatched"] == pytest.approx(
        100 * (r["n_both_correct"] + r["b_unpatched_correct_patched_wrong"]) / n)
    assert r["top1_patched"] == pytest.approx(
        100 * (r["n_both_correct"] + r["c_unpatched_wrong_patched_correct"]) / n)
    assert r["bootstrap"]["ci_low"] <= r["top1_drop"] <= r["bootstrap"]["ci_high"]
    # the frozen threshold is recorded, never used to decide anything
    assert r["frozen_threshold"] == 1.00
    assert "not recomputed" in r["note"].lower()
    assert "verdict stands" in r["note"].lower()
    assert r["layer_range"] == [3, 12]

    # per-image outcomes are kept so the CI can be recomputed without a GPU
    with np.load(out.with_suffix(".npz")) as z:
        unp = np.unpackbits(z["unpatched_correct"])[:n].astype(bool)
        pat = np.unpackbits(z["patched_correct"])[:n].astype(bool)
        assert int(z["n_images"][0]) == n
    assert int((unp & ~pat).sum()) == r["b_unpatched_correct_patched_wrong"]
    assert int((~unp & pat).sum()) == r["c_unpatched_wrong_patched_correct"]
