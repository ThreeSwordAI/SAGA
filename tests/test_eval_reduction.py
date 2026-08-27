"""T5 — eval accumulator: uneven shards sum to the pooled computation (B1)."""

import torch

from tools.eval import counts_to_metrics, shard_counts


def test_uneven_shards_equal_pooled():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(101, 13, generator=g)
    labels = torch.randint(0, 13, (101,), generator=g)

    # 3 uneven shards, together covering every sample exactly once
    bounds = [(0, 17), (17, 60), (60, 101)]
    summed = torch.zeros(4, dtype=torch.float64)
    for lo, hi in bounds:
        summed += shard_counts(logits[lo:hi], labels[lo:hi])

    pooled = shard_counts(logits, labels)
    assert torch.allclose(summed, pooled, rtol=0, atol=1e-6)

    metrics = counts_to_metrics(summed)
    assert metrics["n_images"] == 101

    # counts match a direct computation
    top1_direct = (logits.argmax(dim=1) == labels).sum().item()
    assert summed[0].item() == top1_direct
    top5_direct = (logits.topk(5, dim=1).indices
                   == labels.view(-1, 1)).any(dim=1).sum().item()
    assert summed[1].item() == top5_direct
    assert abs(metrics["top1"] - 100.0 * top1_direct / 101) < 1e-9


def test_counts_are_exact_integers():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(64, 10, generator=g)
    labels = torch.randint(0, 10, (64,), generator=g)
    c = shard_counts(logits, labels)
    assert c[0] == int(c[0]) and c[1] == int(c[1]) and c[3] == 64.0
