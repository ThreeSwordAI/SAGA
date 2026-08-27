"""T3 — sink counts: hand-known outliers, MAD and mu+k*sigma, edge cases (M1)."""

import torch

from saga.metrics import sink_counts_fixed, sink_counts_gauss, sink_counts_mad


def test_mad_single_outlier():
    # nine 10s and one 100: median=10, MAD=0 -> thr=10 -> only the 100 counts
    v = torch.tensor([[10.0] * 9 + [100.0]])
    assert sink_counts_mad(v, k=5.0).tolist() == [1.0]


def test_mad_graded_vector():
    # v = 1..9 plus 100: lower median = 5, |v-5| sorted has lower median 2,
    # thr(k=5) = 5 + 10 = 15 -> only 100 is above
    v = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 100.]])
    assert sink_counts_mad(v, k=5.0).tolist() == [1.0]
    # k=0 -> thr = median = 5 -> strictly above: 6,7,8,9,100
    assert sink_counts_mad(v, k=0.0).tolist() == [5.0]


def test_gauss_known_sigma():
    # nine 10s + one 100: mu=19, population sigma=27
    # k=2 -> thr=73 -> count 1;  k=3 -> thr=100, strict -> count 0
    v = torch.tensor([[10.0] * 9 + [100.0]])
    assert sink_counts_gauss(v, k=2.0).tolist() == [1.0]
    assert sink_counts_gauss(v, k=3.0).tolist() == [0.0]


def test_all_equal_norms_edge_case():
    # MAD=0 and sigma=0 -> thr = median/mean -> strictly above -> 0, no NaN
    v = torch.full((3, 16), 7.0)
    for counts in (sink_counts_mad(v, k=5.0), sink_counts_gauss(v, k=3.0)):
        assert torch.isfinite(counts).all()
        assert counts.tolist() == [0.0, 0.0, 0.0]


def test_fixed_threshold_is_strict():
    v = torch.tensor([[10.0] * 9 + [100.0]])
    assert sink_counts_fixed(v, tau=5.0).tolist() == [10.0]
    assert sink_counts_fixed(v, tau=10.0).tolist() == [1.0]
    assert sink_counts_fixed(v, tau=100.0).tolist() == [0.0]


def test_batched_rows_independent():
    v = torch.stack([
        torch.tensor([10.0] * 9 + [100.0]),
        torch.full((10,), 7.0),
    ])
    assert sink_counts_mad(v, k=5.0).tolist() == [1.0, 0.0]
