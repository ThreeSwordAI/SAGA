"""TASK-02 Phase 2: table builder — verbatim values, deltas, MISSING."""

import csv
import json

from analysis.build_legacy_tables import build_rows, e2_runs

SHAS = {"best": "sha-best", "last": "sha-last"}


def _manifest(tmp_path, variants=("baseline", "saga")):
    p = tmp_path / "manifest.csv"
    fields = ["path", "filename", "size_bytes", "mtime_iso", "sha256",
              "exp", "arch", "recipe", "variant", "seed"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in variants:
            for tag in ("best", "last"):
                w.writerow({"path": f"/x/{v}/{tag}.pth",
                            "filename": f"{tag}.pth", "size_bytes": 1,
                            "mtime_iso": "t", "sha256": SHAS[tag],
                            "exp": "e2", "arch": "vit_small",
                            "recipe": "nomix", "variant": v, "seed": "rlast"})
    return p


def _write_results(tmp_path, variant, top1_best, top1_last, sink):
    ev, dg = tmp_path / "eval", tmp_path / "diag"
    ev.mkdir(exist_ok=True), dg.mkdir(exist_ok=True)
    stem = f"e2_vit_small_nomix_{variant}_rlast"
    for tag, top1 in (("best", top1_best), ("last", top1_last)):
        (ev / f"{stem}_{tag}.json").write_text(json.dumps(
            {"top1": top1, "ckpt_sha256": SHAS[tag]}))
        (dg / f"{stem}_{tag}.json").write_text(json.dumps({
            "ckpt_sha256": SHAS[tag], "sink_mad_k5": sink, "sink_mu3s": 1.0,
            "sink_fixed_thr": 2.0, "oversmooth_pairwise": 0.4,
            "oversmooth_pairwise_nosink": 0.41, "nosink_excluded_mean": 3.0,
            "eff_rank": 100.25, "cls_attn_share": [0.1, 0.2],
            "cls_norm_ratio": [1.0, 1.5], "reg_norm_mean": None}))


def test_rows_verbatim_and_delta(tmp_path):
    m = _manifest(tmp_path)
    _write_results(tmp_path, "baseline", 76.5, 76.25, 10.0)
    _write_results(tmp_path, "saga", 77.125, 77.0, 8.5)

    rows, gaps = build_rows(e2_runs(m), tmp_path / "eval", tmp_path / "diag")
    assert gaps == []
    by_v = {r["variant"]: r for r in rows}

    saga = by_v["saga"]
    assert saga["top1_best"] == 77.125 and saga["top1_last"] == 77.0
    assert saga["top1_best_minus_last"] == 0.125
    assert saga["delta_top1_last_vs_baseline"] == 0.75
    assert saga["sink_mad_k5"] == 8.5
    assert saga["cls_attn_share_lastblock"] == 0.2
    assert saga["cls_norm_ratio_lastblock"] == 1.5
    assert saga["reg_norm_mean"] == ""          # structurally null
    assert saga["ckpt_sha256_last"] == "sha-last"
    assert by_v["baseline"]["delta_top1_last_vs_baseline"] == ""


def test_missing_files_become_MISSING(tmp_path):
    m = _manifest(tmp_path)
    _write_results(tmp_path, "baseline", 76.5, 76.25, 10.0)
    _write_results(tmp_path, "saga", 77.125, 77.0, 8.5)
    (tmp_path / "diag" / "e2_vit_small_nomix_saga_rlast_last.json").unlink()

    rows, gaps = build_rows(e2_runs(m), tmp_path / "eval", tmp_path / "diag")
    saga = {r["variant"]: r for r in rows}["saga"]
    assert len(gaps) == 1
    assert saga["sink_mad_k5"] == "MISSING"
    assert saga["reg_norm_mean"] == "MISSING"
    assert saga["top1_last"] == 77.0            # eval side unaffected


def test_missing_diag_best_is_a_completeness_gap(tmp_path):
    # diag(best) feeds no table column but IS part of the 2.1 completeness
    # check — its absence (or wrong checkpoint) must surface as a gap
    m = _manifest(tmp_path, variants=("baseline",))
    _write_results(tmp_path, "baseline", 76.5, 76.25, 10.0)
    (tmp_path / "diag" / "e2_vit_small_nomix_baseline_rlast_best.json").unlink()

    rows, gaps = build_rows(e2_runs(m), tmp_path / "eval", tmp_path / "diag")
    assert len(gaps) == 1 and "best" in gaps[0]
    assert rows[0]["sink_mad_k5"] == 10.0       # table values unaffected

    _write_results(tmp_path, "baseline", 76.5, 76.25, 10.0)  # restore
    p = tmp_path / "diag" / "e2_vit_small_nomix_baseline_rlast_best.json"
    p.write_text(json.dumps({"ckpt_sha256": "WRONG"}))
    _, gaps = build_rows(e2_runs(m), tmp_path / "eval", tmp_path / "diag")
    assert len(gaps) == 1 and "mismatch" in gaps[0] and "best" in gaps[0]


def test_sha_mismatch_treated_as_gap(tmp_path):
    m = _manifest(tmp_path, variants=("baseline",))
    _write_results(tmp_path, "baseline", 76.5, 76.25, 10.0)
    p = tmp_path / "eval" / "e2_vit_small_nomix_baseline_rlast_last.json"
    p.write_text(json.dumps({"top1": 99.9, "ckpt_sha256": "WRONG"}))

    rows, gaps = build_rows(e2_runs(m), tmp_path / "eval", tmp_path / "diag")
    assert any("mismatch" in g for g in gaps)
    assert rows[0]["top1_last"] == "MISSING"    # never a wrong-ckpt number
