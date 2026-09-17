"""
tests/test_C_phasec.py
======================
TASK C / Phase C — the committed tables, the figure data, and the handoff.

These tests run against the REAL committed results, not fake data: Phase C's
job is to turn what the cluster produced into tables and one document, and
the thing that can go wrong is a number changing between the records, the
table, the figure and the prose. So every test here asks the same kind of
question — does what is written match what it was derived from.

They skip cleanly when the results are absent, so the suite still passes in a
checkout that has the code but not the Phase B outputs.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
I5T = REPO / "results" / "frozen" / "I5_readout" / "sub2k" / "tables"
I7T = REPO / "results" / "frozen" / "I7_attention" / "sub1k" / "tables"
FIG = REPO / "figures_data" / "frozen"
HANDOFF = REPO / "docs" / "C_HANDOFF.md"
VERDICT = (REPO / "results" / "frozen" / "I7_attention" / "sub1k"
           / "i7_wording_verdict.json")

I5_TABLES = ("T_I5a_readout", "T_I5b_methods", "T_I5c_terminal",
             "T_I5d_positions", "T_I5e_diag_vs_readout")
I7_TABLES = ("T_I7a_incoming", "T_I7b_registers", "T_I7c_value_norm",
             "T_I7d_ttr_curve", "T_I7d_ttr_coverage")

MISSING = "MISSING"


def read(path):
    path = Path(path)
    if not path.exists():
        pytest.skip(f"{path.name} not present (Phase B results absent)")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── every table is stamped, and chance is printed ────────────────────────────

@pytest.mark.parametrize("name", I5_TABLES + I7_TABLES)
def test_every_row_carries_an_endpoint_class(name):
    """§7 / D7: every I5 and I7 comparison is secondary, descriptive or
    exploratory, and the TABLE says which on EVERY row — not in a caption a
    reader may never reach."""
    root = I5T if name.startswith("T_I5") else I7T
    rows = read(root / f"{name}.csv")
    assert rows, f"{name} is empty"
    assert "endpoint_class" in rows[0]
    assert {r["endpoint_class"] for r in rows} <= {
        "secondary", "descriptive", "exploratory"}
    assert all(r["endpoint_class"] for r in rows)


@pytest.mark.parametrize("name", ("T_I5a_readout", "T_I5b_methods",
                                  "T_I5c_terminal", "T_I5d_positions"))
def test_chance_is_printed_on_every_readout_table(name):
    """§3: an accuracy without its chance level is unreadable."""
    rows = read(I5T / f"{name}.csv")
    assert "chance_exact" in rows[0]
    vals = {r["chance_exact"] for r in rows}
    assert MISSING not in vals, f"{name} has a row with no chance"
    for v in vals:
        assert abs(float(v) - 1.0 / 196) < 1e-9


def test_no_i5_or_i7_table_is_stamped_primary():
    """D7 fixed the three primary contrasts in Track B. Nothing here is one."""
    for name in I5_TABLES + I7_TABLES:
        root = I5T if name.startswith("T_I5") else I7T
        for r in read(root / f"{name}.csv"):
            assert r["endpoint_class"] != "primary", name


# ── the §7 controls, ON THE COMMITTED TABLES ─────────────────────────────────

def test_term_1_00_left_s11_out_unchanged_in_the_real_results():
    """The control that the whole T_I5c reading depends on: the bypass swaps
    the LAST block's gate, so `s11_out` cannot move. Asserted on the real
    numbers, not on a fake model."""
    rows = [r for r in read(I5T / "T_I5c_terminal.csv")
            if r["stage"] == "s11_out"]
    assert rows, "no s11_out rows in T_I5c_terminal"
    for r in rows:
        assert float(r["delta"]) == 0.0, r["run_id"]
        assert float(r["ci_lo"]) == 0.0 and float(r["ci_hi"]) == 0.0
        assert float(r["max_abs_s11_diff_vs_native"]) == 0.0


def test_t0_is_the_matchers_ceiling_in_the_real_results():
    """T0 matches an image against itself, so it must score exactly 1."""
    rows = [r for r in read(I5T / "T_I5a_readout.csv")
            if r["transform"] == "T0"]
    assert rows
    for r in rows:
        assert float(r["acc_exact_mean"]) == 1.0, r["run_id"]


def test_the_readout_is_far_above_chance():
    """A readout pinned at chance would measure nothing. This is the sanity
    check that the whole I5 table is informative, stated as a floor rather
    than as a claim about any method."""
    rows = [r for r in read(I5T / "T_I5a_readout.csv")
            if r["transform"] != "T0" and r["condition_id"] == "native"]
    over = [float(r["acc_exact_over_chance"]) for r in rows
            if r["acc_exact_over_chance"] != MISSING]
    assert over and min(over) > 50, f"lowest is {min(over):.1f}x chance"


# ── the wording rule is COPIED, never re-decided ─────────────────────────────

def test_the_handoff_copies_the_wording_sentence_verbatim():
    """§12: the verdict and its sentence are copied into the handoff, never
    edited. The rule was declared before the measurement existed, and
    re-phrasing its output here would quietly undo that."""
    if not (HANDOFF.exists() and VERDICT.exists()):
        pytest.skip("handoff or verdict not present")
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    text = HANDOFF.read_text(encoding="utf-8")
    assert v["sentence"] in text, "the sentence was altered on its way in"
    assert v["verdict"] in text
    assert v["term"] in text


def test_the_wording_verdict_matches_the_table_it_was_computed_from():
    """Re-deriving the verdict from the committed table must reproduce the
    committed verdict — the file and the rule cannot have drifted apart."""
    if not VERDICT.exists():
        pytest.skip("verdict not present")
    from analysis.i7_wording import from_table

    committed = json.loads(VERDICT.read_text(encoding="utf-8"))
    rebuilt = from_table(I7T / "T_I7a_incoming.csv").to_dict()
    assert rebuilt["verdict"] == committed["verdict"]
    assert rebuilt["sentence"] == committed["sentence"]
    assert rebuilt["passed"] == committed["passed"]


# ── the handoff is generated, and says what is true about D9/D10 ─────────────

def test_the_handoff_is_byte_identical_to_its_generator():
    """A generated file under docs/ is REGENERATED, never hand-edited
    (I0 handoff §8.6). If someone improved a sentence by hand, this fails.

    Regenerated at the sha the DOCUMENT records, not at HEAD: the provenance
    line legitimately moves with every later commit, and a test that went red
    on every unrelated commit is one people learn to ignore. Every other byte
    — every number, every word — is pinned exactly.
    """
    if not HANDOFF.exists():
        pytest.skip("handoff not present")
    from analysis.build_C_handoff import build, sha_in

    text = HANDOFF.read_text(encoding="utf-8")
    recorded = sha_in(text)
    assert recorded and len(recorded) >= 7, "no git sha in the handoff"
    assert text == build(sha=recorded)


def test_the_handoff_provenance_line_is_the_only_thing_that_may_move():
    """The seam above must not be a hole: regenerating at a DIFFERENT sha has
    to change the document, or the byte-identity test is checking nothing."""
    if not HANDOFF.exists():
        pytest.skip("handoff not present")
    from analysis.build_C_handoff import build, sha_in

    text = HANDOFF.read_text(encoding="utf-8")
    other = build(sha="0" * 40)
    assert other != text
    # and they differ on exactly one line — the provenance line
    diff = [(a, b) for a, b in zip(text.splitlines(), other.splitlines())
            if a != b]
    assert len(diff) == 1, f"{len(diff)} lines moved, expected 1"
    assert sha_in(text) in diff[0][0]


def test_the_handoff_reports_the_d9_d10_status_truthfully():
    """The handoff must not imply a pre-registration that did not happen —
    nor deny one that did. It reports whichever is the case."""
    if not HANDOFF.exists():
        pytest.skip("handoff not present")
    locked = (REPO / "docs" / "LOCKED_ANALYSIS.md").read_text(encoding="utf-8")
    text = HANDOFF.read_text(encoding="utf-8")
    from saga.frozen.masks import says_frozen
    signed = ("D9" in locked and "D10" in locked and says_frozen(locked))
    if signed:
        assert "signed before these results were inspected" in text
    else:
        assert "not** a signed pre-registration" in text
        assert "were fixed in committed code before any Phase B job ran" in text


def test_the_handoff_states_what_track_c_does_not_settle():
    """§12 requires it, and it is the part a reader most needs."""
    if not HANDOFF.exists():
        pytest.skip("handoff not present")
    text = HANDOFF.read_text(encoding="utf-8")
    for phrase in ("ONE utility, not utility", "is a convention",
                   "n = 2", "EXPLORATORY", "SECONDARY under D7"):
        assert phrase in text, f"the handoff never says {phrase!r}"


# ── figure data ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ("F5B_readout", "F1C_readout_draft",
                                  "F7_attention", "F_ttr_curve"))
def test_figure_data_marks_missing_rather_than_zeroing_it(name):
    path = FIG / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{name}.npz not present")
    with np.load(path, allow_pickle=False) as z:
        assert "meta_json" in z, f"{name} carries no provenance"
        json.loads(str(z["meta_json"]))
        flags = [k for k in z.files if k.endswith("__is_missing")
                 or k.endswith("_is_missing")]
        assert flags, f"{name} has no MISSING bookkeeping"
        for fk in flags:
            base = fk.replace("__is_missing", "").replace("_is_missing", "")
            if base in z.files and z[base].dtype.kind == "f":
                assert np.isnan(z[base][z[fk]]).all(), \
                    f"{name}: {base} has a MISSING entry that is not NaN"


def test_f7_carries_the_verdict_it_was_built_beside():
    path = FIG / "F7_attention.npz"
    if not (path.exists() and VERDICT.exists()):
        pytest.skip("F7 or verdict not present")
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        assert meta["wording_verdict"] == v["verdict"]
        assert meta["wording_sentence"] == v["sentence"]
        assert float(z["threshold"][0]) == v["threshold"]


# ── the TTR curve's MISSING bookkeeping survives into the table ──────────────

def test_the_ttr_coverage_table_names_the_absent_cells():
    rows = read(I7T / "T_I7d_ttr_coverage.csv")
    absent = [r for r in rows if r["status"] == MISSING]
    assert absent, "no MISSING cell recorded — the all-layer range has three"
    for r in absent:
        assert r["n_neurons_grid"] == MISSING
        assert int(r["n_points"]) == 0


def test_the_ttr_curve_never_averages_missing():
    """A grid point that is not in a committed file stays MISSING; it is
    never interpolated from a neighbour or carried over from another range."""
    rows = read(I7T / "T_I7d_ttr_curve.csv")
    zero_n = [r for r in rows if int(r["n_neurons"]) == 0]
    assert zero_n, "the unpatched baseline point should be in the curve"
    for r in zero_n:
        # the unpatched point has no reduction fraction by construction
        assert r["outlier_reduction_frac"] == MISSING
