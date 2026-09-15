"""TASK-11 PHASE C — T_localization.csv, teaser.md, the F1 figure.

CPU, no checkpoints. The committed-artifact tests recompute every headline
number from the per-image CSV through an independent path, the way TASK-08's
suite pins T3 against its own run JSONs.
"""

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from analysis import build_localization_tables as blt
from analysis import build_teaser_note as btn

REPO = Path(__file__).resolve().parents[1]
LOC = REPO / "results/figures_data/F1_localization.csv"
TABLE = REPO / "results/tables/T_localization.csv"
NOTE = REPO / "results/notes/teaser.md"
ARCHIVE = REPO / "results/figures_data/F1_teaser.npz"

# the committed table stores 6 significant digits (see
# test_table_stores_six_significant_digits); artifact comparisons are
# made at that precision, not at float precision
SIGFIG = 1e-5


# ── unit: the statistics ─────────────────────────────────────────────────────

def _table(rows_by_key, meta):
    return {k: dict(v) for k, v in rows_by_key.items()}, meta


def test_paired_delta_se_and_2xSE_flag():
    """Hand-computable: deltas 1,1,1,3 -> mean 1.5, sample sd 1.0,
    se 0.5, 2*se 1.0, and 1.5 > 1.0 so the flag is YES."""
    table = {
        ("base", "g", "m"): {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
        ("x", "g", "m"): {"a": 1.0, "b": 1.0, "c": 1.0, "d": 3.0},
    }
    meta = {"base": {"variant": "baseline", "arch": "A"},
            "x": {"variant": "saga", "arch": "A"}}
    rows = blt.build(table, meta, {"A": ["base", "x"]})
    r = next(r for r in rows if r["run_id"] == "x")
    assert float(r["delta_vs_baseline_mean"]) == pytest.approx(1.5)
    assert float(r["delta_sd"]) == pytest.approx(1.0)
    assert float(r["delta_se"]) == pytest.approx(0.5)
    assert r["delta_significant_2xSE"] == "YES"
    assert r["delta_direction"] == "above baseline"


def test_a_delta_inside_2xSE_is_not_flagged():
    """The other side of the same boundary: deltas 1,1,1,-1 -> mean 0.5,
    sd 1.0, se 0.5, 2*se 1.0, and 0.5 < 1.0 so the flag is 'no'."""
    table = {
        ("base", "g", "m"): {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
        ("x", "g", "m"): {"a": 1.0, "b": 1.0, "c": 1.0, "d": -1.0},
    }
    meta = {"base": {"variant": "baseline", "arch": "A"},
            "x": {"variant": "saga", "arch": "A"}}
    rows = blt.build(table, meta, {"A": ["base", "x"]})
    r = next(r for r in rows if r["run_id"] == "x")
    assert float(r["delta_vs_baseline_mean"]) == pytest.approx(0.5)
    assert float(r["delta_se"]) == pytest.approx(0.5)
    assert r["delta_significant_2xSE"] == "no"


def test_table_stores_six_significant_digits():
    """Documents the committed precision, so the tolerance the artifact
    tests below use is a stated property rather than a fudge factor."""
    assert blt.fmt(0.38222949206289054) == "0.382229"
    assert blt.fmt(1 / 3) == "0.333333"
    assert blt.fmt(blt.MISSING) == blt.MISSING


def test_baseline_row_carries_no_delta():
    table = {("base", "g", "m"): {"a": 1.0, "b": 2.0},
             ("x", "g", "m"): {"a": 1.5, "b": 2.5}}
    meta = {"base": {"variant": "baseline", "arch": "A"},
            "x": {"variant": "saga", "arch": "A"}}
    rows = blt.build(table, meta, {"A": ["base", "x"]})
    b = next(r for r in rows if r["run_id"] == "base")
    assert b["delta_vs_baseline_mean"] == blt.MISSING
    assert b["role"] == "baseline"


def test_std_is_MISSING_at_n1_never_zero():
    table = {("base", "g", "m"): {"a": 1.0}, ("x", "g", "m"): {"a": 2.0}}
    meta = {"base": {"variant": "baseline", "arch": "A"},
            "x": {"variant": "saga", "arch": "A"}}
    rows = blt.build(table, meta, {"A": ["base", "x"]})
    for r in rows:
        assert r["std"] == blt.MISSING
        assert r["delta_sd"] == blt.MISSING


def test_a_null_that_differs_between_models_is_refused():
    """box_area_frac depends only on the image. If it ever differs between
    models something is wrong with the frame mapping, and it must not be
    published as a 'reference'."""
    table = {
        ("base", "boxes20", "inbox_mass"): {"a": 0.5},
        ("x", "boxes20", "inbox_mass"): {"a": 0.6},
        ("base", "boxes20", "box_area_frac"): {"a": 0.40},
        ("x", "boxes20", "box_area_frac"): {"a": 0.41},      # <- must raise
    }
    meta = {"base": {"variant": "baseline", "arch": "A"},
            "x": {"variant": "saga", "arch": "A"}}
    with pytest.raises(ValueError, match="model-independent"):
        blt.build(table, meta, {"A": ["base", "x"]})


def test_architectures_are_never_pooled():
    """A ViT-S model must be paired against the ViT-S baseline, never the
    ViT-B one."""
    table = {
        ("sb", "g", "m"): {"a": 1.0}, ("ss", "g", "m"): {"a": 2.0},
        ("bb", "g", "m"): {"a": 10.0}, ("bs", "g", "m"): {"a": 11.0},
    }
    meta = {"sb": {"variant": "baseline", "arch": "S"},
            "ss": {"variant": "saga", "arch": "S"},
            "bb": {"variant": "baseline", "arch": "B"},
            "bs": {"variant": "saga", "arch": "B"}}
    rows = blt.build(table, meta, {"S": ["sb", "ss"], "B": ["bb", "bs"]})
    ss = next(r for r in rows if r["run_id"] == "ss")
    bs = next(r for r in rows if r["run_id"] == "bs")
    assert float(ss["delta_vs_baseline_mean"]) == pytest.approx(1.0)
    assert float(bs["delta_vs_baseline_mean"]) == pytest.approx(1.0)
    assert ss["arch"] == "S" and bs["arch"] == "B"


def test_more_than_one_baseline_per_arch_is_refused():
    table = {("a", "g", "m"): {"i": 1.0}, ("b", "g", "m"): {"i": 1.0}}
    meta = {"a": {"variant": "baseline", "arch": "A"},
            "b": {"variant": "baseline", "arch": "A"}}
    with pytest.raises(ValueError, match="exactly one baseline"):
        blt.build(table, meta, {"A": ["a", "b"]})


# ── the committed artifacts ──────────────────────────────────────────────────

@pytest.mark.skipif(not LOC.exists(), reason="Phase B artifacts not present")
def test_committed_table_matches_an_independent_recompute():
    """Every mean/std/delta/SE/flag in T_localization.csv, recomputed from
    F1_localization.csv without touching the builder."""
    raw = defaultdict(dict)
    with open(LOC, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            raw[(r["run_id"], r["group"], r["metric"])][r["image_id"]] = \
                float(r["value"])

    with open(TABLE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "committed table is empty"

    baselines = {r["arch"]: r["run_id"] for r in rows if r["role"] == "baseline"}
    checked = 0
    for r in rows:
        if r["role"] == "null_reference" or r["n"] in ("0", blt.MISSING):
            continue
        vals_map = raw[(r["run_id"], r["group"], r["metric"])]
        vals = [vals_map[k] for k in sorted(vals_map)]
        assert int(r["n"]) == len(vals)
        assert float(r["mean"]) == pytest.approx(statistics.fmean(vals), rel=SIGFIG)
        if len(vals) > 1:
            assert float(r["std"]) == pytest.approx(
                statistics.stdev(vals), rel=SIGFIG)

        if r["role"] == "baseline":
            assert r["delta_vs_baseline_mean"] == blt.MISSING
            continue

        base = raw[(baselines[r["arch"]], r["group"], r["metric"])]
        keys = sorted(set(vals_map) & set(base))
        d = [vals_map[k] - base[k] for k in keys]
        mu = statistics.fmean(d)
        sd = statistics.stdev(d)
        se = sd / math.sqrt(len(d))
        assert float(r["delta_vs_baseline_mean"]) == pytest.approx(mu, rel=SIGFIG)
        assert float(r["delta_se"]) == pytest.approx(se, rel=SIGFIG)
        assert (r["delta_significant_2xSE"] == "YES") == (abs(mu) > 2 * se)
        checked += 1
    assert checked >= 8, f"only {checked} delta rows verified"


@pytest.mark.skipif(not LOC.exists(), reason="Phase B artifacts not present")
def test_the_null_metrics_really_are_model_independent():
    raw = defaultdict(dict)
    with open(LOC, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            raw[(r["run_id"], r["group"], r["metric"])][r["image_id"]] = \
                float(r["value"])
    for metric, group in (("box_area_frac", "boxes20"),
                          ("uniform_ring1_mass", "random200")):
        series = [v for (run, g, m), v in raw.items()
                  if m == metric and g == group]
        assert len(series) >= 2
        ref = series[0]
        for other in series[1:]:
            assert other.keys() == ref.keys()
            assert max(abs(other[k] - ref[k]) for k in ref) < 1e-9


@pytest.mark.skipif(not LOC.exists(), reason="Phase B artifacts not present")
def test_uniform_ring1_mass_is_the_rings_area_fraction():
    """Ties the committed table back to the TASK-07 ring definition: 44 of
    196 patches on a 14x14 grid."""
    with np.load(ARCHIVE, allow_pickle=False) as z:
        mask = z["ring1_mask"]
    with open(TABLE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ref = [r for r in rows if r["metric"] == "uniform_ring1_mass"]
    assert ref
    for r in ref:
        assert float(r["mean"]) == pytest.approx(mask.mean(), rel=SIGFIG)
        assert float(r["mean"]) == pytest.approx(44 / 196, rel=SIGFIG)


@pytest.mark.skipif(not NOTE.exists(), reason="Phase C note not built")
def test_note_regenerates_byte_identically_from_the_committed_table(tmp_path):
    """Pins that no number in teaser.md was typed by hand — the whole file
    is a function of the committed artifacts."""
    out = tmp_path / "teaser.md"
    old = sys.argv
    try:
        sys.argv = ["build_teaser_note.py", "--table", str(TABLE),
                    "--localization", str(LOC), "--archive", str(ARCHIVE),
                    "--out", str(out)]
        assert btn.main() == 0
    finally:
        sys.argv = old
    a = NOTE.read_text(encoding="utf-8").splitlines()
    b = out.read_text(encoding="utf-8").splitlines()
    # the provenance footer carries the live git sha, which moves with commits
    assert a[:-1] == b[:-1]


@pytest.mark.skipif(not NOTE.exists(), reason="Phase C note not built")
def test_note_states_the_pending_and_absent_panels():
    text = NOTE.read_text(encoding="utf-8")
    assert "TTR — PENDING" in text
    assert "vit_base / registers — ABSENT" in text
    assert "illustrations, not evidence" in text
    # the honest reading must survive, not be softened away
    assert "does not support the framing" in text
    assert "over IMAGES, not over seeds" in text
    # failure cases kept
    assert "Failure cases" in text
    assert "Pointing game missed by EVERY model" in text


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Phase B archive not present")
def test_note_and_figure_choose_the_same_three_images():
    """build_teaser_note.choose_images() and plot_F1's own default must not
    drift apart — the note would then describe panels the figure never drew."""
    from plotting import plot_F1
    with np.load(ARCHIVE, allow_pickle=False) as z:
        index = json.loads(str(z["index"]))
    note_pick = [im["image_id"] for im in btn.choose_images(index)]

    seen, fig_pick = set(), []
    for im in index["images"]:
        if im["criterion"] not in seen:
            seen.add(im["criterion"])
            fig_pick.append(im["image_id"])
    assert note_pick == fig_pick[:3]
    assert len(note_pick) == 3
    assert hasattr(plot_F1, "COLUMN_ORDER")
    assert plot_F1.COLUMN_ORDER == ("Baseline", "Registers", "TTR", "SAGA")
