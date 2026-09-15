"""
tests/test_task13_phasec.py — TASK-13 Phase C
=============================================
Pins the two generated tables and the generated note against the committed
source files, so a table can never drift from the JSONs it claims to
summarize and the note can never drift from the tables.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
RING_DIR = REPO / "results" / "finegrained" / "ring_ablation"
T_RING = REPO / "results" / "tables" / "T_ring_ablation.csv"
T3 = REPO / "results" / "tables" / "T3_finegrained.csv"
NOTE = REPO / "results" / "notes" / "finegrained_ext.md"
MISSING = "MISSING"

sys.path.insert(0, str(REPO))
from analysis import build_ring_tables as BRT           # noqa: E402


def _rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick(rows, **kw):
    return next((r for r in rows if all(r.get(k) == v for k, v in kw.items())),
                None)


@pytest.fixture(scope="module")
def ring_json():
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RING_DIR.glob("*.json"))}


# ── the contrast identity ────────────────────────────────────────────────────

def test_contrast_is_random44_minus_ring1_for_every_run(ring_json):
    """contrast = drop_ring1 - drop_random44 collapses to
    random44 - ring1; the table must agree exactly, or the whole §2 reading
    is against a different quantity than the one it names."""
    rows = [r for r in _rows(T_RING) if r["kind"] == "repeat"]
    assert len(rows) == len(ring_json) == 24
    for r in rows:
        rid = (f"ft_{r['dataset']}_"
               f"{'vits' if 'small' in r['arch'] else 'vitb'}_"
               f"{r['variant']}_bs1_{r['ft_seed']}")
        d = ring_json[rid]
        t = d["top1"]
        assert float(r["top1_full"]) == t["full"]
        assert float(r["drop_ring1"]) == pytest.approx(
            t["full"] - t["mask_ring1"], abs=1e-9)
        assert float(r["drop_random44"]) == pytest.approx(
            t["full"] - t["mask_random44"], abs=1e-9)
        assert float(r["contrast_ring1_minus_random44"]) == pytest.approx(
            float(r["drop_ring1"]) - float(r["drop_random44"]), abs=1e-9)
        assert float(r["contrast_ring1_minus_random44"]) == pytest.approx(
            t["mask_random44"] - t["mask_ring1"], abs=1e-9)


def test_committed_ring_table_matches_the_committed_jsons(ring_json):
    """Independent recompute of every aggregate, not via the generator."""
    import statistics as stat
    rows = _rows(T_RING)
    for ds in ("cub", "aircraft"):
        recs = [d for d in ring_json.values() if d["dataset"] == ds]
        assert len(recs) == 12
        for col, f in (
            ("drop_ring1", lambda d: d["top1"]["full"] - d["top1"]["mask_ring1"]),
            ("drop_random44",
             lambda d: d["top1"]["full"] - d["top1"]["mask_random44"]),
            ("contrast_ring1_minus_random44",
             lambda d: d["top1"]["mask_random44"] - d["top1"]["mask_ring1"]),
        ):
            xs = [f(d) for d in recs]
            m = _pick(rows, dataset=ds, arch="ALL", kind="mean")
            s = _pick(rows, dataset=ds, arch="ALL", kind="se")
            assert float(m[col]) == pytest.approx(stat.mean(xs), abs=1e-9)
            sd = stat.stdev(xs)
            assert float(s[col]) == pytest.approx(
                sd / len(xs) ** 0.5, abs=1e-9)


def test_between_dataset_row_is_cub_minus_aircraft(ring_json):
    rows = _rows(T_RING)
    col = "contrast_ring1_minus_random44"
    d = _pick(rows, kind="between_dataset_difference")
    cub = float(_pick(rows, dataset="cub", arch="ALL", kind="mean")[col])
    air = float(_pick(rows, dataset="aircraft", arch="ALL", kind="mean")[col])
    assert float(d[col]) == pytest.approx(cub - air, abs=1e-9)
    assert "UNPAIRED" in d["flag"], \
        "the between-dataset difference must be labelled unpaired"
    assert "prediction: CUB > Aircraft" in d["flag"]


def test_area_matching_holds_in_every_committed_json(ring_json):
    for rid, d in ring_json.items():
        for c in ("mask_ring1", "mask_center44", "mask_random44"):
            assert d["n_masked_patches"][c] == 44, rid
        assert d["n_masked_patches"]["full"] == 0, rid
    assert len({tuple(d["masked_indices"]["mask_ring1"])
                for d in ring_json.values()}) == 1


def test_all_24_reproduce_the_committed_test_number(ring_json):
    """The acceptance item: `full` must reproduce test_final.json."""
    for rid, d in ring_json.items():
        assert d["full_reproduces_test_final"] is True, rid
        assert abs(d["full_minus_committed"]) <= d["full_tolerance"], rid
        final = json.loads(
            (REPO / "results" / "runs" / rid / "eval" / "test_final.json"
             ).read_text(encoding="utf-8"))
        assert d["committed_test_top1"] == final["top1"], rid
        assert d["finetuned_sha256"] == final["finetuned_sha256"], rid
        assert final.get("smoke") is False, rid


# ── MISSING discipline ───────────────────────────────────────────────────────

def test_a_run_whose_full_did_not_reproduce_yields_MISSING(tmp_path):
    """A drop computed against an unverified baseline must never appear."""
    src = sorted(RING_DIR.glob("*.json"))[0]
    d = json.loads(src.read_text(encoding="utf-8"))
    rid = d["run_id"]
    d["full_reproduces_test_final"] = False
    d["full_minus_committed"] = 0.9
    ring_dir = tmp_path / "ring"
    ring_dir.mkdir()
    (ring_dir / f"{rid}.json").write_text(json.dumps(d), encoding="utf-8")

    matrix = yaml.safe_load(
        (REPO / "configs" / "ft_matrix.yaml").read_text(encoding="utf-8"))
    rows, recs = BRT.build_rows(matrix, ring_dir,
                                REPO / "results" / "runs")
    rep = [r for r in rows if r["kind"] == "repeat"]
    assert len(rep) == 1
    assert rep[0]["drop_ring1"] == MISSING
    assert rep[0]["contrast_ring1_minus_random44"] == MISSING
    assert "did not reproduce" in rep[0]["flag"]
    # and the mean must not have averaged it in
    m = _pick(rows, kind="mean", arch="ALL")
    assert m["contrast_ring1_minus_random44"] == MISSING
    assert m["n"] == 0


def test_std_and_se_are_MISSING_not_zero_for_n_equals_1():
    m, sd, se = BRT.mean_std([1.5])
    assert m == 1.5 and sd == MISSING and se == MISSING
    m, sd, se = BRT.mean_std([])
    assert m == sd == se == MISSING


# ── T3 after the seed fill ───────────────────────────────────────────────────

def test_T3_has_all_24_repeats_and_vitb_at_n3():
    rows = _rows(T3)
    assert len([r for r in rows if r["kind"] == "repeat"]) == 24
    for ds in ("cub", "aircraft"):
        d = _pick(rows, dataset=ds, arch="vit_base_patch16_224",
                  kind="paired_delta_mean", variant="saga")
        se = _pick(rows, dataset=ds, arch="vit_base_patch16_224",
                   kind="paired_delta_se", variant="saga")
        assert d["n"] == "3", f"{ds} ViT-B should be n=3"
        assert se["test_top1"] not in ("", MISSING), \
            f"{ds} ViT-B must now carry an SE"


def test_T3_values_match_the_committed_test_final_jsons():
    """Repeats in the table must equal the run JSONs, via an independent read."""
    rows = [r for r in _rows(T3) if r["kind"] == "repeat"]
    for r in rows:
        arch = "vits" if "small" in r["arch"] else "vitb"
        rid = f"ft_{r['dataset']}_{arch}_{r['variant']}_bs1_{r['ft_seed']}"
        d = json.loads((REPO / "results" / "runs" / rid / "eval" /
                        "test_final.json").read_text(encoding="utf-8"))
        assert float(r["test_top1"]) == pytest.approx(d["top1"], abs=1e-9), rid


# ── the note ─────────────────────────────────────────────────────────────────

def test_note_regenerates_byte_identically_from_the_committed_tables(tmp_path):
    """Every number in the note must come from the tables — if any were
    typed in, regenerating would not reproduce the committed file."""
    out = tmp_path / "regen.md"
    r = subprocess.run(
        [sys.executable, str(REPO / "analysis" / "build_finegrained_ext_note.py"),
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == NOTE.read_text(encoding="utf-8")


def test_note_states_the_legacy_numbers_are_void():
    t = NOTE.read_text(encoding="utf-8")
    assert "+2.19" in t and "+1.29" in t
    assert "VOID" in t
    assert "B7" in t
    assert "never cited" in t


def test_note_records_the_skipped_third_dataset():
    t = NOTE.read_text(encoding="utf-8")
    assert "SKIPPED" in t
    assert "Stanford Cars" in t and "Oxford Flowers-102" in t
    assert "no dataset was downloaded" in t
    assert "no claim of generalization beyond CUB and Aircraft" in t.lower() \
        or "no claim of generalization beyond CUB and Aircraft" in t


def test_note_reports_the_reproduction_check_and_the_prediction_verdict():
    t = NOTE.read_text(encoding="utf-8")
    assert "24 of 24 pass" in t
    # the verdict must be stated in words, in whichever direction it fell
    assert ("predicted ORDERING HOLDS" in t
            or "predicted ORDERING DOES NOT HOLD" in t)
    # and the qualifications that bound it must survive any edit
    assert "Both contrasts are NEGATIVE" in t
    assert "orders the OTHER way" in t


def test_note_makes_no_pooled_cross_architecture_claim():
    t = NOTE.read_text(encoding="utf-8")
    assert "disagree in SIGN" in t
    assert "no pooled cross-architecture claim" in t


# ── the handoff document ─────────────────────────────────────────────────────

HANDOFF = REPO / "docs" / "Task13_handsoff.md"


def test_handoff_headline_numbers_match_the_tables():
    """The handoff embeds a git sha, so byte-equality regeneration is not
    usable; instead every headline number it quotes must appear in it with
    the value the committed tables hold."""
    text = HANDOFF.read_text(encoding="utf-8")
    t3, rg = _rows(T3), _rows(T_RING)

    for ds in ("cub", "aircraft"):
        for arch in ("vit_small_patch16_224", "vit_base_patch16_224"):
            d = _pick(t3, dataset=ds, arch=arch, kind="paired_delta_mean",
                      variant="saga")
            assert f"{float(d['test_top1']):+.3f}" in text, (ds, arch)
        c = _pick(rg, dataset=ds, arch="ALL", kind="mean")
        assert f"{float(c['contrast_ring1_minus_random44']):+.3f}" in text, ds
        assert f"{float(c['drop_ring1']):+.3f}" in text, ds

    diff = _pick(rg, kind="between_dataset_difference")
    assert f"{float(diff['contrast_ring1_minus_random44']):+.4f}" in text


def test_handoff_states_the_scope_limits():
    """A handoff that dropped these would misrepresent what was shown."""
    text = HANDOFF.read_text(encoding="utf-8")
    # the three qualifications that bound the ring verdict
    assert "Both contrasts are NEGATIVE" in text
    assert "orders the OTHER way" in text
    assert "carried by the denominator" in text.lower() \
        or "carried by the DENOMINATOR" in text
    # the skipped sub-goal and its consequence
    assert "SKIPPED" in text
    assert "no claim of generalization beyond CUB and Aircraft" in text
    # the sign disagreement must not be quietly dropped
    assert "disagree in SIGN" in text
    # and the void legacy numbers
    assert "+2.19" in text and "+1.29" in text and "VOID" in text
