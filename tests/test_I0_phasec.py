"""
tests/test_I0_phasec.py
=======================
TASK I0 Phase C — the generated handoff, and the split shas in
`docs/LOCKED_ANALYSIS.md`.

The handoff embeds a git sha, so the generator takes `--git-sha` and the
byte-identity test regenerates with the sha the committed document already
records. That makes the document reproducible from its inputs rather than
merely plausible, which is the property the project's other handoffs check
number by number.

Everything else here asserts that a number in the document equals the value
the committed file it came from holds, and that the qualifications a reader
needs are still present.
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HANDOFF = REPO / "docs" / "I0_HANDOFF.md"
LOCKED = REPO / "docs" / "LOCKED_ANALYSIS.md"
MAN = REPO / "results" / "frozen" / "I0_manifest"
SPLITS = REPO / "results" / "frozen" / "splits"
MISSING = "MISSING"


@pytest.fixture(scope="module")
def text():
    return HANDOFF.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((MAN / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def smokes():
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(MAN.glob("smoke_*.json"))]


# ── the document is generated, not hand-typed ───────────────────────────────

def test_handoff_regenerates_byte_identically(text, tmp_path):
    """Re-render with the sha the committed document records; byte-identical
    or it has drifted from its inputs."""
    m = re.search(r"at git `([0-9a-f]+)`", text)
    assert m, "the handoff does not record the git sha it was built at"
    out = tmp_path / "I0_HANDOFF.md"
    rc = subprocess.run(
        [sys.executable, str(REPO / "analysis" / "build_I0_handoff.py"),
         "--out", str(out), "--git-sha", m.group(1)],
        cwd=REPO, capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert out.read_text(encoding="utf-8") == text


def test_every_markdown_table_has_consistent_columns(text):
    """A literal `|` inside a cell silently breaks the table it is in — this
    caught two of them in the first draft."""
    rows, bad = [], []
    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("|"):
            rows.append((i, line.count("|")))
            continue
        if rows and len({w for _, w in rows}) != 1:
            bad.append((rows[0][0], sorted({w for _, w in rows})))
        rows = []
    assert not bad, f"inconsistent table column counts at lines {bad}"


# ── the numbers it quotes are the ones the files hold ───────────────────────

def test_cohort_numbers_match_the_manifest(text, manifest):
    rows = manifest["rows"]
    cohort = [r for r in rows
              if r["family"] in ("e2r_300ep", "legacy_300ep")
              and r["ckpt_kind"] == "last"
              and r["status"] in ("eligible", "eligible_legacy")]
    assert f"**{len(cohort)} completed 300-epoch conditions.**" in text
    assert f"{manifest['n_rows']} rows in total" in text

    by_variant = {}
    for r in cohort:
        by_variant.setdefault(r["variant"], []).append(r)
    # the totals row of the breakdown table
    totals = " | ".join(str(len(by_variant[v])) for v in sorted(by_variant))
    assert f"| **total** | | {totals} | **{len(cohort)}** |" in text

    n_ctrl = sum(1 for r in cohort if str(r["seed_controlled"]) == "1")
    assert f"{n_ctrl} of the {len(cohort)} were trained under the seeded" \
        in text


def test_family_status_table_matches_the_manifest(text, manifest):
    rows = manifest["rows"]
    statuses = sorted({r["status"] for r in rows})
    for fam in sorted({r["family"] for r in rows}):
        fam_rows = [r for r in rows if r["family"] == fam]
        counts = [sum(1 for r in fam_rows if r["status"] == s)
                  for s in statuses]
        cells = " | ".join(str(c) if c else "—" for c in counts)
        assert f"| {fam} | {cells} | {len(fam_rows)} |" in text, fam


def test_split_shas_match_the_split_files(text):
    for name in ("calibration", "evaluation", "sub1k", "sub2k"):
        d = json.loads((SPLITS / f"{name}.json").read_text(encoding="utf-8"))
        assert f"`{d['sha256']}`" in text, name
        assert f"| `{name}.json` | {d['n']:,} |" in text, name
    # and the protocol sentence travels verbatim from the split files
    d = json.loads((SPLITS / "evaluation.json").read_text(encoding="utf-8"))
    assert d["protocol"] in text


def test_hist_stage_and_its_citation_are_the_module_constants(text):
    from saga.frozen.stages import HIST_STAGE, HIST_STAGE_CITATION
    assert f"**`{HIST_STAGE}`**" in text
    assert HIST_STAGE_CITATION in text


def test_smoke_numbers_match_the_smoke_jsons(text, smokes):
    n_pass = sum(s["n_pass"] for s in smokes)
    n_app = sum(s["n_pass"] + s["n_fail"] for s in smokes)
    assert f"**{n_pass} of {n_app} applicable checks PASS.**" in text
    for s in smokes:
        assert (f"| `{s['run_id']}` | {s['variant']} | {s['n_prefix']} | "
                f"{s['n_pass']} | {s['n_fail']} | {s['n_skip']} |") in text
        # the prefix-row measurement, which is the register-model guard
        c = next(c for c in s["checks"]
                 if c["check"] == "prefix_rows_untouched")
        assert f"(n_prefix={c['measured']['n_prefix']})" in text


def test_checkpoint_identity_numbers_match(text, manifest):
    rows = manifest["rows"]
    hashed = sum(1 for r in rows if r["ckpt_sha256"] != MISSING)
    assert f"{hashed} of {len(rows)} rows carry a 64-character" in text
    hashes = json.loads((MAN / "ckpt_hashes.json").read_text(encoding="utf-8"))
    assert f"`{hashes['n_hashed']}` hashed of `{hashes['n_targets']}`" in text

    legacy = {}
    with open(REPO / "results" / "legacy" / "checkpoint_manifest.csv",
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            legacy[r["path"]] = r["sha256"]
    n_leg = sum(1 for r in rows if r["ckpt_path"] in legacy)
    n_ok = sum(1 for r in rows if r["ckpt_path"] in legacy
               and r["ckpt_sha256"] == legacy[r["ckpt_path"]])
    assert f"**{n_ok} of {n_leg}**" in text
    assert n_ok == n_leg                      # and they actually all agree


def test_detection_resolution_table_matches_the_manifest(text, manifest):
    det = [r for r in manifest["rows"]
           if r["family"] == "dense_det" and r["ckpt_kind"] == "best"]
    assert det
    for r in det:
        p = json.loads(r["derived_params"])
        assert (f"| `{r['run_id']}` | {p['best_ap_epoch']} | "
                f"{p['last_epoch']} |") in text


# ── the qualifications a reader needs must not quietly disappear ────────────

def test_handoff_states_that_i0_concluded_nothing_scientific(text):
    assert "concluded nothing scientific" in text


def test_handoff_reports_the_recorded_fail_rather_than_hiding_it(text,
                                                                 smokes):
    """A handoff that dropped the one FAIL would misrepresent Phase B."""
    fails = [(s, c) for s in smokes for c in s["checks"]
             if c["status"] == "FAIL"]
    if not fails:
        pytest.skip("no FAIL recorded in the committed smoke JSONs")
    s, c = fails[0]
    assert f"`{s['run_id']}` records **FAIL** on `{c['check']}`" in text
    # with the measured quantity, and the reason it is not a defect
    assert "per_head_sorted_values_identical" in text
    assert "fp32 addition is not associative" in text
    assert "left exactly as the HPC wrote it" in text


def test_handoff_states_that_locked_analysis_is_still_a_draft(text):
    assert "**DRAFT**" in text
    assert "no I3/I4 Phase-B job may run" in text
    assert "BLOCKS a work package" in text


def test_handoff_states_the_split_protocol_limit(text):
    """The paper must not describe this as an untouched test set."""
    low = text.lower()
    assert "locked evaluation protocol" in low
    assert "not a completely untouched test set" in low
    assert "retrospectively preregistered" in low


def test_handoff_records_the_conditions_contract(text):
    assert "No condition, layer, epsilon, mask or permutation is selectable " \
        "from the command line" in text


# ── LOCKED_ANALYSIS: the Phase-C edit ───────────────────────────────────────

def test_locked_analysis_split_shas_are_filled_in_and_correct():
    locked = LOCKED.read_text(encoding="utf-8")
    assert "PENDING — Phase B" not in locked
    import hashlib
    disc = REPO / "results" / "diagsplit" / "val_diag_split.json"
    assert f"`{hashlib.sha256(disc.read_bytes()).hexdigest()}`" in locked
    for name in ("calibration", "evaluation", "sub1k", "sub2k"):
        d = json.loads((SPLITS / f"{name}.json").read_text(encoding="utf-8"))
        assert f"`{d['sha256']}`" in locked, name


def test_locked_analysis_is_still_unsigned():
    """Phase C fills in FACTS. The three signature lines are the human's, and
    filling them in is what freezes the document."""
    locked = LOCKED.read_text(encoding="utf-8")
    assert "STATUS: **DRAFT — NOT YET FROZEN**" in locked
    # the two machine-checkable signature fields are still unfilled, and the
    # "signed by" line still names the human rather than anyone else
    assert "Date frozen: `PENDING`" in locked
    assert "Git sha at freeze: `PENDING`" in locked
    assert "Signed and dated by: *(the human — nobody else)*" in locked


def test_every_decision_is_still_listed_and_d5_still_blocks():
    """All eight decisions stay in the table, and D5 stays open.

    This asserted "exactly one closed (D8)" when I0 wrote it. That was a
    SNAPSHOT of a state later work packages exist to change: I2 closed D1 by
    measurement, I3-prep closed D3 by generating the permutation lists, and
    D2/D4/D6/D7 were closed at their written defaults. What the test actually
    guards — no decision silently vanishes from the table, and D5 is not
    quietly closed while I1 still owes the map it needs — is unchanged, and
    the closed set is now pinned BY NAME so a future silent closure still
    fails here.
    """
    locked = LOCKED.read_text(encoding="utf-8")
    rows = [ln for ln in locked.splitlines()
            if re.match(r"^\| ~*D\d+~*", ln)]
    assert len(rows) == 8
    closed = {re.search(r"D\d+", ln).group()
              for ln in rows if ln.startswith("| ~~")}
    assert closed == {"D1", "D2", "D3", "D4", "D6", "D7", "D8"}
    d5 = next(ln for ln in rows if ln.startswith("| D5"))
    assert "no default" in d5
    assert "D5" not in closed, "D5 is open until I1 produces the block-input map"
