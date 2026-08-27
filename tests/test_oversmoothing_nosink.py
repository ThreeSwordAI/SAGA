"""T8 — sink-excluded oversmoothing (TASK-02 metric addendum)."""

import torch

from saga.metrics import (oversmoothing_pairwise, oversmoothing_pairwise_nosink,
                          token_norms)


def test_outlier_exclusion_raises_cosine():
    # 9 identical tokens (cos 1 among them) + 1 large-norm near-orthogonal
    # outlier: full pairwise mean = 72/90 = 0.8; nosink drops the outlier -> 1
    base = torch.zeros(10, 4)
    base[:, 0] = 10.0            # 9 copies of 10*e0 ...
    base[9] = 0.0
    base[9, 1] = 1000.0          # ... and one 1000*e1 outlier
    x = base.unsqueeze(0)        # [1, 10, 4]
    norms = token_norms(x)

    full = oversmoothing_pairwise(x)
    nosink, excluded = oversmoothing_pairwise_nosink(x, norms, k=5.0)

    assert torch.allclose(full, torch.tensor([0.8]), atol=1e-6)
    assert torch.allclose(nosink, torch.tensor(1.0), atol=1e-6)
    assert nosink > full[0]
    assert excluded.item() == 1.0


def test_no_outliers_metrics_agree():
    # tokens are random signed basis vectors scaled by exactly 7.0, so every
    # norm is EXACTLY equal (float-normalizing random vectors leaves ~1e-7
    # jitter that a near-zero MAD threshold would flag) -> nothing excluded
    # -> both metrics identical
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 16, (4, 37, 1), generator=g)
    sign = (torch.randint(0, 2, (4, 37, 1), generator=g) * 2 - 1).float()
    x = torch.zeros(4, 37, 16).scatter_(2, idx, 7.0 * sign)
    norms = token_norms(x)
    assert (norms == 7.0).all()

    full = oversmoothing_pairwise(x).mean()
    nosink, excluded = oversmoothing_pairwise_nosink(x, norms, k=5.0)

    assert excluded.item() == 0.0
    assert torch.allclose(nosink, full, atol=1e-6)


def test_too_few_survivors_does_not_crash():
    # N=1: fewer than 2 tokens can ever survive -> every image skipped,
    # mean cosine is NaN by contract, exclusion mean still finite, no crash
    x = torch.full((2, 1, 8), 3.0)
    nosink, excluded = oversmoothing_pairwise_nosink(x, token_norms(x), k=5.0)
    assert torch.isnan(nosink)
    assert torch.isfinite(excluded)

    # mixed batch (N=2): image 0 has norms [1, 5] -> lower median 1, MAD 0,
    # threshold 1 -> only 1 survivor -> skipped; image 1 has two identical
    # tokens -> both kept, cos 1. Mean must come from image 1 only.
    x2 = torch.zeros(2, 2, 4)
    x2[0, 0, 0] = 1.0
    x2[0, 1, 0] = 5.0
    x2[1, :, 1] = 3.0
    nosink2, excl2 = oversmoothing_pairwise_nosink(x2, token_norms(x2), k=5.0)
    assert torch.allclose(nosink2, torch.tensor(1.0), atol=1e-6)
    assert excl2.item() == 0.5   # 1 excluded in image 0, 0 in image 1
