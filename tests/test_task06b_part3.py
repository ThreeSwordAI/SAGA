"""TASK-06B Part 3: pooled-stat math, pairing, gate-agreement NaN handling."""

import numpy as np

from analysis.build_pooled_tables import MISSING, mean_std, welch
from analysis.gate_structure import agreement_rows, pair_group


def test_mean_std_and_missing():
    m, s = mean_std([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and abs(s - 1.29099) < 1e-4
    m, s = mean_std([5.0])                    # n=1: mean but no std
    assert m == 5.0 and s == MISSING
    m, s = mean_std([MISSING, 7.0])           # MISSING never averaged
    assert m == 7.0 and s == MISSING
    assert mean_std([MISSING]) == (MISSING, MISSING)


def test_welch_needs_two_per_side():
    assert welch([1.0], [1.0, 2.0]) == (MISSING, MISSING)
    t, p = welch([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert t < 0 and 0 < p < 1


def test_paired_delta_significance_arithmetic():
    # the exact S/mixup construction: 4 paired deltas -> mean/SE/2xSE flag
    ds = [-0.126, 0.570, 0.490, 0.890]
    m, sd = mean_std(ds)
    se = sd / 2
    assert abs(m - 0.456) < 1e-9
    assert abs(m) > 2 * se                     # significant at 2xSE
    # and a case that must NOT be significant
    ds2 = [-0.5, 0.6]
    m2, sd2 = mean_std(ds2)
    assert abs(m2) <= 2 * (sd2 / np.sqrt(2))


def test_gate_agreement_handles_constant_layer():
    rng = np.random.RandomState(0)
    shared = rng.rand(3, 9)                    # 3 layers, 9 positions
    a = shared + rng.rand(3, 9) * 1e-3
    b = shared + rng.rand(3, 9) * 1e-3
    a[2, :] = 0.5                              # constant final layer
    b[2, :] = 0.5
    rows = agreement_rows({"mixup:s1": a, "mixup:s2": b})
    r = rows[0]
    assert r["group"] == "within-mixup"
    assert r["n_layers_defined"] == 2          # constant layer excluded
    assert r["spearman_layer_mean"] == r["spearman_layer_mean"]  # not NaN
    assert r["spearman_layer_min"] > 0.9       # near-identical maps agree


def test_pair_group_labels():
    assert pair_group("mixup:s1", "mixup:legacy-nomixdir") == "within-mixup"
    assert pair_group("nomix:s1", "nomix:s2") == "within-nomix"
    assert pair_group("mixup:s1", "nomix:s2") == "cross-recipe"


def test_head_mean_gate_is_arithmetic_sigmoid_mean():
    # regression: the documented transform is sigmoid THEN arithmetic head
    # mean — an operator-precedence slip (1/(1+e^-x)).mean vs 1/((1+e^-x).mean)
    # produced different published numbers once
    from analysis.gate_structure import head_mean_gate
    phi = np.array([[[0.0, 2.0], [0.0, -2.0]]])       # [L=1, H=2, N=2]
    got = head_mean_gate(phi)
    sig = 1.0 / (1.0 + np.exp(-phi))
    want = sig.mean(axis=1)
    assert np.allclose(got, want)
    # and it must NOT equal the harmonic-style pooling
    harmonic = 1.0 / (1.0 + np.exp(-phi)).mean(axis=1)
    assert not np.allclose(got, harmonic)
