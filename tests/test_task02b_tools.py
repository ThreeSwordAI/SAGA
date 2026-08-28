"""TASK-02B Phase A: robustness builder, F6 collector, norm summarizer."""

import csv
import json
import sys

import numpy as np

from analysis.build_sink_robustness import build_verdict, compare, load_diag_last
from tools.summarize_norms import arch_edges, collect_files, norm_stats
import tools.summarize_norms as summarize_norms

THRESH_FIELDS = {"sink_mad_k5": 3.0, "sink_mu2s": 5.0, "sink_mu3s": 2.0,
                 "sink_mu4s": 1.0, "sink_mu5s": 0.5, "sink_mu6s": 0.25,
                 "sink_fixed_thr": 4.0}


def _write_manifest(path, rows):
    fields = ["path", "filename", "size_bytes", "mtime_iso", "sha256",
              "exp", "arch", "recipe", "variant", "seed"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mrow(variant, sha):
    return {"path": f"/x/{variant}/last.pth", "filename": "last.pth",
            "size_bytes": 1, "mtime_iso": "t", "sha256": sha, "exp": "e2",
            "arch": "vit_small", "recipe": "nomix", "variant": variant,
            "seed": "rlast"}


def _diag(tmp_path, variant, sha, scale=1.0):
    d = {"ckpt_sha256": sha, "arch": "vit_small",
         **{k: v * scale for k, v in THRESH_FIELDS.items()}}
    p = tmp_path / f"e2_vit_small_nomix_{variant}_rlast_last.json"
    p.write_text(json.dumps(d))
    return d


# ── build_sink_robustness ────────────────────────────────────────────────────

def test_verdict_matrix_orders_and_ties(tmp_path):
    m = tmp_path / "manifest.csv"
    _write_manifest(m, [_mrow("baseline", "b1"), _mrow("saga", "s1"),
                        _mrow("registers", "r1")])
    _diag(tmp_path, "baseline", "b1", scale=1.0)
    _diag(tmp_path, "saga", "s1", scale=0.5)        # saga smaller everywhere
    _diag(tmp_path, "registers", "r1", scale=1.0)   # ties baseline everywhere

    diags, gaps = load_diag_last(m, tmp_path)
    assert gaps == [] and len(diags) == 3

    cells, rows = build_verdict(diags)
    assert cells == [("vit_small", "nomix")]
    assert len(rows) == 14                          # 7 thresholds x 2 blocks
    for row in rows:
        if row["comparison"] == "saga_vs_baseline":
            assert row["vit_small/nomix"] == "S<B"
        else:
            assert row["vit_small/nomix"] == "R=B"  # tie marker


def test_sha_mismatch_is_a_gap(tmp_path):
    m = tmp_path / "manifest.csv"
    _write_manifest(m, [_mrow("baseline", "EXPECTED")])
    _diag(tmp_path, "baseline", "STALE")
    diags, gaps = load_diag_last(m, tmp_path)
    assert diags == {} and len(gaps) == 1 and "mismatch" in gaps[0]


def test_compare_missing_propagates():
    assert compare("MISSING", 1.0, "S<B", "S>B") == "MISSING"
    assert compare(1.0, 1.0, "S<B", "S>B") == "S=B"


# ── summarize_norms ──────────────────────────────────────────────────────────

def _norms_fixture(tmp_path, stem, arch, sha, norms):
    np.savez(tmp_path / f"{stem}_norms.npz",
             last_block_patch_norms=np.asarray(norms, dtype=np.float16))
    (tmp_path / f"{stem}.json").write_text(json.dumps(
        {"arch": arch, "ckpt_sha256": sha}))


def test_norm_stats_hand_computed():
    # image 0: [1,1,1,1] -> med 1, MAD 0; image 1: [2,2,2,8] -> med 2 (lower),
    # MAD 0; thresholds [1, 2] -> mean 1.5; median_of_medians lower([1,2]) = 1
    s = norm_stats(np.array([[1, 1, 1, 1], [2, 2, 2, 8.0]]))
    assert s["median_of_medians"] == 1.0
    assert s["mean_mad"] == 0.0
    assert s["mean_threshold_mad_k5"] == 1.5
    assert s["max"] == 8.0
    assert s["n_images"] == 2
    assert s["p50"] == 1.5          # np.percentile linear on the 8 values


def test_summarize_norms_shared_edges_and_idempotency(tmp_path, monkeypatch):
    _norms_fixture(tmp_path, "a_last", "vit_small", "shaA",
                   [[1.0, 2.0], [3.0, 4.0]])
    _norms_fixture(tmp_path, "b_last", "vit_small", "shaB",
                   [[2.0, 8.0], [5.0, 6.0]])

    files = collect_files(tmp_path)
    edges = arch_edges(files)
    assert set(edges) == {"vit_small"}
    assert len(edges["vit_small"]) == 65
    assert edges["vit_small"][0] == 1.0 and edges["vit_small"][-1] == 8.0

    monkeypatch.setattr(sys, "argv",
                        ["summarize_norms.py", "--diag-dir", str(tmp_path)])
    summarize_norms.main()
    out_a = json.loads((tmp_path / "a_last_normstats.json").read_text())
    out_b = json.loads((tmp_path / "b_last_normstats.json").read_text())
    assert out_a["hist_bin_edges"] == out_b["hist_bin_edges"]   # arch-shared
    assert sum(out_a["hist_counts"]) == 4                       # all tokens
    assert len(out_a["hist_counts"]) == 64
    assert out_a["ckpt_sha256"] == "shaA"

    # idempotent second run: outputs byte-identical
    before = (tmp_path / "a_last_normstats.json").read_bytes()
    summarize_norms.main()
    assert (tmp_path / "a_last_normstats.json").read_bytes() == before

    # a new checkpoint sha forces a rewrite
    (tmp_path / "a_last.json").write_text(json.dumps(
        {"arch": "vit_small", "ckpt_sha256": "shaA2"}))
    summarize_norms.main()
    assert json.loads((tmp_path / "a_last_normstats.json").read_text())[
        "ckpt_sha256"] == "shaA2"


# ── collect_F6 (via its CSV contract) ────────────────────────────────────────

def test_collect_f6_long_format(tmp_path, monkeypatch):
    from analysis import collect_F6
    m = tmp_path / "manifest.csv"
    _write_manifest(m, [_mrow("baseline", "b1")])
    d = {"ckpt_sha256": "b1", "cls_norm_ratio": [1.0, 1.5],
         "cls_attn_share": None}
    (tmp_path / "e2_vit_small_nomix_baseline_rlast_last.json").write_text(
        json.dumps(d))
    out = tmp_path / "f6.csv"
    monkeypatch.setattr(sys, "argv",
                        ["collect_F6.py", "--manifest", str(m),
                         "--diag-dir", str(tmp_path), "--out", str(out)])
    collect_F6.main()
    rows = list(csv.DictReader(open(out, newline="")))
    assert len(rows) == 2
    assert rows[1]["block"] == "1" and rows[1]["cls_norm_ratio"] == "1.5"
    assert rows[0]["cls_attn_share"] == ""          # None -> blank, not 0


# ── build_gate1_addendum: reading, saturation, H1/H2 classification ─────────

def test_memo_reading_renders_missing_not_equal():
    from analysis.build_gate1_addendum import sec_robustness
    cells = [("vit_small", "nomix")]
    verdict_rows = [
        {"comparison": "saga_vs_baseline", "threshold": "sink_mad_k5",
         "vit_small/nomix": "MISSING"},
        {"comparison": "saga_vs_baseline", "threshold": "sink_mu3s",
         "vit_small/nomix": "S=B"},
        {"comparison": "registers_vs_baseline", "threshold": "sink_mad_k5",
         "vit_small/nomix": "MISSING"},
    ]
    robustness_rows = [{"arch": "vit_small", "recipe": "nomix",
                        "variant": "saga", "sink_fixed_thr": "1.0"}]
    text = "\n".join(sec_robustness(verdict_rows, cells, robustness_rows))
    assert "MISSING under sink_mad_k5" in text
    assert "equal under sink_mu3s" in text
    assert "equal under sink_mad_k5" not in text   # never fabricated


def test_memo_saturation_spans_are_per_arch():
    from analysis.build_gate1_addendum import sec_robustness
    cells = []
    rob = [
        {"arch": "vit_small", "recipe": "nomix", "variant": "a",
         "sink_fixed_thr": "4.0"},
        {"arch": "vit_small", "recipe": "nomix", "variant": "b",
         "sink_fixed_thr": "18.0"},
        {"arch": "vit_base", "recipe": "nomix", "variant": "a",
         "sink_fixed_thr": "6.0"},
        {"arch": "vit_base", "recipe": "mixup", "variant": "b",
         "sink_fixed_thr": "196.0"},
    ]
    text = "\n".join(sec_robustness([], cells, rob))
    assert "ViT-S/16: 4.0000 (nomix a) to 18.0000 (nomix b)" in text
    assert "ViT-B/16: 6.0000 (nomix a) to 196.0000 (mixup b)" in text
    # no pooled cross-arch span 4.0 -> 196.0 on one line
    assert "4.0000 (nomix a) to 196.0000" not in text
    assert "ViT-B/16/mixup b: 196.0000 / 196" in text   # saturation flagged


def test_memo_h1_h2_classification_three_way(tmp_path):
    from analysis.build_gate1_addendum import sec_normscale

    def cell_stats(recipe, variant, thr, p999, mx):
        p = tmp_path / f"e2_vit_small_{recipe}_{variant}_rlast_last_normstats.json"
        p.write_text(json.dumps({"median_of_medians": 10.0,
                                 "mean_threshold_mad_k5": thr,
                                 "p999": p999, "max": mx}))

    t = [{"arch": "vit_small", "recipe": r}
         for r in ("h1", "h2", "both", "nei")]
    # H1 only: threshold -50%, extremes -10% (both shrank)
    cell_stats("h1", "baseline", 100.0, 100.0, 100.0)
    cell_stats("h1", "saga", 50.0, 90.0, 90.0)
    # H2 only: extremes grew, threshold also grew
    cell_stats("h2", "baseline", 100.0, 100.0, 100.0)
    cell_stats("h2", "saga", 105.0, 110.0, 110.0)
    # both signatures: threshold shrank while extremes grew
    cell_stats("both", "baseline", 100.0, 100.0, 100.0)
    cell_stats("both", "saga", 90.0, 110.0, 110.0)
    # neither: threshold -10%, extremes -30% (shrank MORE than threshold)
    cell_stats("nei", "baseline", 100.0, 100.0, 100.0)
    cell_stats("nei", "saga", 90.0, 70.0, 70.0)

    text = "\n".join(sec_normscale(tmp_path, t))
    per_line = {r: next(l for l in text.splitlines()
                        if f"ViT-S/16/{r}:" in l and "pattern" in l)
                for r in ("h1", "h2", "both", "nei")}
    assert "H1 (threshold shrank more" in per_line["h1"]
    assert "H2 (extremes did not shrink)" in per_line["h2"]
    assert "both signatures" in per_line["both"]
    # the false-H2 branch is gone: shrinking extremes never labeled H2
    assert "neither" in per_line["nei"] and "H2" not in per_line["nei"]
