#!/usr/bin/env python3
"""
analysis/frozen_I1_stats.py
===========================
The image-level statistics `analysis/frozen_I1_spatial.py` could not compute
before Phase B — TASK A / I1 Phase C.

Everything here needs the per-image INDICATOR matrix `[n_images, N]` that
`saga/frozen/norms.py` packs into each committed `maps_<stage>.npz`. Before
Phase B only per-position counts existed, and a count summed over 10,000
images cannot be un-summed, which is why these columns read `PENDING B` in
the Phase-A tables.

THE UNIT OF ANALYSIS IS THE IMAGE (`docs/LOCKED_ANALYSIS.md` §8). Every
resample here draws IMAGES with replacement, never positions: the 196
positions of one image are one observation of a spatial arrangement, not 196
independent observations of anything. D6 is CLOSED at seed 0.

WHY THE BOOTSTRAP IS CHUNKED. A percentile CI on a per-image mean is
10,000 resamples of a 10,000-long vector — 10^8 gathers per statistic, which
is seconds in a vectorised gather and hours in a Python loop. The η² CI is
worse: each resample needs a fresh frequency MAP, i.e. a weighted mean of
`[n_images, N]`, so it is done as a BLAS matmul of a chunk of multinomial
weight vectors against the indicator matrix. Both are exact bootstraps —
the chunking changes the memory profile, not the estimator.

No training, no optimizer, no probe fitting, no selection by any loss.
"""

import numpy as np

from analysis.address_analysis import ring_indices

#: `docs/LOCKED_ANALYSIS.md` §8; D6 CLOSED at 0.
BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 10000
ALPHA = 0.05

#: Resamples per vectorised chunk. 500 x 10,000 int64 indices is ~40 MB.
CHUNK = 500
#: Resamples per chunk for the map-valued bootstrap, which multiplies a
#: [chunk, n_images] weight matrix against [n_images, N] indicators.
MAP_CHUNK = 250


def percentile_ci(values, *, alpha=ALPHA):
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_weights(n: int, take: int, rng) -> np.ndarray:
    """`[take, n]` float32 resample WEIGHTS — how often each image is drawn.

    Drawing `n` image indices uniformly with replacement and counting them
    IS the multinomial bootstrap; generating it as `randint` + one flat
    `bincount` is about three times faster than `rng.multinomial` with
    10,000 categories, and it is the same estimator, not an approximation.

    Weights rather than gathers: `x[idx]` for a `[500, 10000]` index matrix
    materialises 5,000,000 rows, while `W @ x` is one BLAS call and touches
    `n x k` floats.
    """
    idx = rng.randint(0, n, size=(int(take), int(n)))
    flat = idx + (np.arange(int(take)) * int(n))[:, None]
    return np.bincount(flat.ravel(),
                       minlength=int(take) * int(n)).reshape(int(take),
                                                             int(n)).astype(
                                                                 np.float32)


def bootstrap_mean_ci(per_image, *, seed=BOOTSTRAP_SEED,
                      resamples=BOOTSTRAP_RESAMPLES, alpha=ALPHA):
    """Percentile CI for the mean of a per-image quantity.

    `per_image` may be `[n_images]` or `[n_images, k]`; a 2-D input is
    resampled ONCE and every column carried through the SAME resample, which
    is what makes the CI on `ring1 - ring0` consistent with the CIs on
    `ring1` and `ring0` separately rather than a third independent draw.
    """
    x = np.asarray(per_image, dtype=np.float32)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = x.shape[0]
    if n < 2:
        return (None, None) if squeeze else [(None, None)] * x.shape[1]

    rng = np.random.RandomState(int(seed))
    means = np.empty((int(resamples), x.shape[1]), dtype=np.float64)
    done = 0
    while done < int(resamples):
        take = min(CHUNK, int(resamples) - done)
        means[done:done + take] = (bootstrap_weights(n, take, rng) @ x) / n
        done += take
    out = [percentile_ci(means[:, j], alpha=alpha) for j in range(x.shape[1])]
    return out[0] if squeeze else out


def per_image_ring_means(indicators, side: int) -> np.ndarray:
    """`[n_images, n_rings]` — each image's mean indicator per border ring.

    Collapsing positions into ring means BEFORE resampling is what makes the
    ring CIs cheap: the bootstrap then resamples a small matrix instead of
    re-reducing the whole indicator array 10,000 times.
    """
    ind = np.asarray(indicators)
    side = int(side)
    return np.column_stack([ind[:, ring_indices(side, k)].mean(axis=1)
                            for k in range(side // 2)])


def eta2_from_freq(freq) -> float:
    """η²_pos = Var_P f(P) / (f̄(1 − f̄)), population variance over positions."""
    f = np.asarray(freq, dtype=np.float64)
    fbar = float(f.mean())
    if not 0.0 < fbar < 1.0:
        return float("nan")
    return float(f.var(ddof=0) / (fbar * (1.0 - fbar)))


def bootstrap_eta2_ci(indicators, *, seed=BOOTSTRAP_SEED,
                      resamples=BOOTSTRAP_RESAMPLES, alpha=ALPHA):
    """Percentile CI for η², resampling IMAGES.

    Each resample needs a whole frequency map, so the multinomial weights of
    a chunk of resamples are multiplied against the indicator matrix in one
    BLAS call: `W [chunk, n] @ ind [n, N] / n` gives `chunk` maps at once.
    float32 halves the traffic and is far finer than the 1/n granularity of
    the counts themselves.
    """
    ind = np.asarray(indicators, dtype=np.float32)
    n = ind.shape[0]
    if n < 2:
        return None, None
    rng = np.random.RandomState(int(seed))
    vals = np.empty(int(resamples), dtype=np.float64)
    done = 0
    while done < int(resamples):
        take = min(MAP_CHUNK, int(resamples) - done)
        w = bootstrap_weights(n, take, rng)
        maps = (w @ ind) / float(n)                       # [take, N]
        fbar = maps.mean(axis=1, dtype=np.float64)
        var = maps.var(axis=1, ddof=0, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals[done:done + take] = var / (fbar * (1.0 - fbar))
        done += take
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None, None
    return percentile_ci(vals, alpha=alpha)


def image_stats(indicators, side: int, *, with_rings=False, with_eta2=False,
                seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES) -> dict:
    """Image-level statistics for ONE map, from one pass over its indicators.

    Called once per (run, stage, basis) and its small result cached: the
    indicator matrices are ~2 MB each and there are 336 of them.

    THE BOOTSTRAPS ARE OPTIONAL BECAUSE THEY DOMINATE THE COST. `p_any`, the
    mean count and the split-half reliability are a few array reductions and
    are always computed; the ring CIs (`with_rings`) and the η² CI
    (`with_eta2`) are ~3 s and ~3 s per map, so they are requested only for
    the rows that report them — `T_I1c`'s ring ordering and `T_I1e`'s η²,
    both of which the task file scopes to BASELINE maps.
    """
    from saga.frozen import norms as fnorms
    from saga.frozen import prevalence as P

    ind = np.asarray(indicators)
    counts = ind.sum(axis=1)
    fa, fb = fnorms.split_half_frequency(ind, seed=seed)
    half = P.spatial_rho(fa, fb)

    out = {
        "n_images": int(ind.shape[0]),
        "p_any": float((counts > 0).mean()),
        "count_per_image_mean": float(counts.mean()),
        "split_half_rho": None if half is None else float(half[0]),
        "split_half_p": None if half is None else float(half[1]),
        "split_half_n_transforms": 0 if half is None else int(half[3]),
        "bootstrap_seed": int(seed), "bootstrap_resamples": int(resamples),
    }
    if with_rings:
        rings = per_image_ring_means(ind, side)
        n_rings = rings.shape[1]
        # the ring CIs and the ring1 - ring0 contrast share ONE resample
        columns = (np.column_stack([rings, rings[:, 1] - rings[:, 0]])
                   if n_rings > 1 else rings)
        cis = bootstrap_mean_ci(columns, seed=seed, resamples=resamples)
        out["ring_ci"] = {k: cis[k] for k in range(n_rings)}
        out["ring1_minus_ring0_ci"] = (cis[n_rings] if n_rings > 1
                                       else (None, None))
        out["count_per_image_ci"] = bootstrap_mean_ci(
            counts.astype(np.float32), seed=seed, resamples=resamples)
    if with_eta2:
        out["eta2_ci"] = bootstrap_eta2_ci(ind, seed=seed,
                                           resamples=resamples)
    return out


__all__ = [
    "BOOTSTRAP_SEED", "BOOTSTRAP_RESAMPLES", "percentile_ci",
    "bootstrap_weights", "bootstrap_mean_ci", "per_image_ring_means",
    "eta2_from_freq",
    "bootstrap_eta2_ci", "image_stats",
]
