"""early_handoff.md: numbers sourced from committed tables, scope retained."""

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "early_handoff.md"


def _rows(rel, enc="utf-8"):
    with open(REPO / rel, newline="", encoding=enc) as f:
        return list(csv.DictReader(f))


def _doc():
    return DOC.read_text(encoding="utf-8")


def test_headline_numbers_match_committed_tables():
    t = _doc()
    # classification: the S/mixup paired delta and SE from e2_pooled.csv
    p = _rows("results/tables/e2_pooled.csv")
    d = next(r for r in p if (r["arch"], r["recipe_actual"], r["variant"],
                              r["kind"]) ==
             ("vit_small", "mixup", "saga", "paired_delta_mean"))
    assert f"+{float(d['top1_last']):.3f}" in t
    # detection: the AP_S delta the Gate-2 verdict turns on
    t4 = _rows("results/tables/T4_coco.csv")
    ds = next(r for r in t4 if r["kind"] == "delta_saga_minus_baseline")
    assert ds["AP_S"] in t
    # TTR: the ViT-B full-val drop and its CI bounds
    tt = _rows("results/tables/T_ttr.csv")
    vb = next(r for r in tt if r["arch"] == "vit_base")
    assert f"{float(vb['top1_drop_vs_baseline']):.3f}" in t
    assert f"{float(vb['paired_ci_low']):.3f}" in t
    # ring ablation: the pooled dataset contrasts (arch=ALL rows)
    rg = _rows("results/tables/T_ring_ablation.csv")
    for dsn in ("cub", "aircraft"):
        m = next(r for r in rg if r["dataset"] == dsn and r["kind"] == "mean"
                 and r["arch"] == "ALL")
        assert f"{float(m['contrast_ring1_minus_random44']):+.3f}" in t
    # segmentation: background-class mean uses the is_background_class flag
    pc = _rows("results/tables/T5_ade20k_per_class.csv")
    bg = [float(r["saga_minus_baseline_iou_pts"]) for r in pc
          if r["is_background_class"] == "yes"]
    assert len(bg) == 3
    assert f"{sum(bg) / 3:+.4f}" in t


def test_scope_and_void_statements_retained():
    t = _doc()
    assert "VOID — never cite these" in t
    assert "+2.19 Aircraft / +1.29 CUB" in t          # the void legacy claim
    assert "OPPOSITE directions" in t                 # fine-grained honesty
    assert "UNANSWERED" in t                          # ablation sink saturation
    assert "below SAGA on top-1 in every cell" in t   # TTR verdict
    assert "REGISTERS, not SAGA" in t                 # localization headline
    assert "recipe_actual" in t                       # the erratum doctrine
    assert re.search(r"Gate-1.*PARTIAL|Gate 1.*PARTIAL", t)
    assert re.search(r"Gate-2 verdict: PARTIAL", t)


def test_no_local_absolute_paths_leak():
    t = _doc()
    assert not re.search(r"[A-Za-z]:\\|/Users/|/home/(?!woody|vault|hpc)",
                         t), "absolute local path leaked into the handoff"
