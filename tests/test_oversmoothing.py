"""T2 — closed-form pairwise oversmoothing == O(N^2) brute force (fixes B3)."""

import torch

from saga.metrics import oversmoothing_consecutive_legacy, oversmoothing_pairwise


def brute_force_pairwise(x: torch.Tensor) -> torch.Tensor:
    """Mean cosine over all ordered pairs i != j, per image. fp64 reference."""
    x = x.double()
    xh = x / x.norm(dim=-1, keepdim=True)
    b, n, _ = x.shape
    out = []
    for bi in range(b):
        total = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total += float(xh[bi, i] @ xh[bi, j])
        out.append(total / (n * (n - 1)))
    return torch.tensor(out, dtype=torch.float64)


def test_closed_form_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 37, 16, generator=g, dtype=torch.float64)
    ref = brute_force_pairwise(x)

    closed64 = oversmoothing_pairwise(x)
    assert torch.allclose(closed64, ref, atol=1e-5)

    closed32 = oversmoothing_pairwise(x.float()).double()
    assert torch.allclose(closed32, ref, atol=1e-5)


def test_identical_tokens_give_cosine_one():
    x = torch.ones(2, 5, 3).repeat(1, 1, 1) * torch.tensor([1.0, 2.0, 3.0])
    val = oversmoothing_pairwise(x)
    assert torch.allclose(val, torch.ones(2), atol=1e-6)


def test_legacy_consecutive_metric():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 9, 8, generator=g, dtype=torch.float64)
    xh = x / x.norm(dim=-1, keepdim=True)
    ref = torch.stack([
        torch.stack([xh[b, j] @ xh[b, j + 1] for j in range(8)]).mean()
        for b in range(3)
    ])
    val = oversmoothing_consecutive_legacy(x)
    assert torch.allclose(val, ref, atol=1e-6)
