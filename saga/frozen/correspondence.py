"""
saga/frozen/correspondence.py
=============================
TASK C / I5 §3 — the training-free correspondence readout: descriptors,
the cosine nearest-neighbour matcher, and the per-image record.

THE QUESTION THIS ANSWERS
-------------------------
Outlier counts and cosine similarity are properties of a feature map. They
are not statements about what a consumer of those features can do. I5
supplies a consumer that needs no training at all: after a transform whose
patch correspondence is known exactly, a patch's descriptor should still
identify the same patch. Nearest-neighbour accuracy against that answer key
is a per-image, chance-calibrated readout of local feature usefulness, and
it is computed from the same frozen forward pass as the diagnostics it is
compared with.

WHAT IS AND IS NOT CLAIMED
--------------------------
This is ONE utility, under exact grid-aligned transforms, and it is a
SECONDARY endpoint under D7 — the three primary contrasts of the paper are
C1-C3 in Track B. Nothing here decides a primary claim, and
`analysis/frozen_I5_analysis.py` stamps `secondary` on every table it
writes.

THE MATCHER (§3, D9)
--------------------
For each shared position of the TRANSFORMED image, the cosine nearest
neighbour among ALL grid**2 patches of the ORIGINAL image — not among the
shared ones. Restricting the candidates to the shared set would make T2 and
T3 easier than T1 by construction (fewer distractors) and the three
transforms would stop being comparable.

    acc_exact   the neighbour is the corresponding position
    acc_1       the neighbour is within Chebyshev distance 1 of it
    chance      1/grid**2, and the exact 1-tolerance value, printed on
                every table (saga/frozen/transforms.py)

Two descriptors, both declared in D9:

    l2          the raw patch token, L2-normalized          (PRIMARY)
    centred_l2  the image's mean patch token subtracted     (SECONDARY)
                first, then L2-normalized

`centred_l2` exists because a large shared component across an image's
patches inflates every cosine similarity equally and can make a
nearest-neighbour search look better or worse than the local content
warrants. Subtracting the image mean removes it. Which of the two is
primary was fixed in advance, not chosen from the result.

MAD EXCEEDANCE, ON THE FLY
--------------------------
The per-position MAD exceedance flag of the ORIGINAL image at the matched
stage is computed from the same forward (`saga.metrics` arithmetic, via
`saga.frozen.diag.mad_threshold`, which `tests/test_I5_readout.py` asserts
agrees with `saga.metrics.sink_counts_mad` exactly). Nothing here reads a
Track A output or a threshold file: the split of accuracy into exceedance
and non-exceedance positions has to be available for any checkpoint the
sweep runs, including ones no other work package touches.

WHAT IS STORED
--------------
Per-image records only — no raw features, no attention, no descriptors
(§7). The per-position exceedance flag is used on the fly and REDUCED to
the split it exists for (`n_shared_exc` / `n_correct_exact_exc` / ... vs the
non-exceedance half) before anything is written; the 196-bit mask itself is
not a column, because `corr_records.parquet` is committed and a mask on
every one of ~48,000 rows per run would add tens of megabytes across the
cohort for a quantity the tables only ever consume as that split. This is
the one place this module's record differs from the column list in §3 of
the task file, and it is recorded in `docs/TASK_LOG.md`.

The split is stored as COUNTS, and so is the overall accuracy: every
accuracy here is `n_correct / n_shared` for two small integers, and
`saga/frozen/records.py::CORR_ACCURACY_COUNTS` declares which count divides
which. Exact, and MISSING needs no sentinel (`n_shared_exc == 0` says it).
"""

import numpy as np
import torch

from saga.frozen.diag import MAD_K, mad_threshold
from saga.frozen.transforms import (chebyshev, correspondence, grid_size_for,
                                    is_bijection)

MISSING = "MISSING"

#: The two declared descriptors (D9). PRIMARY FIRST — several tables report
#: only `DESCRIPTORS[0]` and the order is what makes that unambiguous.
DESCRIPTORS = ("l2", "centred_l2")


class CorrespondenceError(ValueError):
    """A descriptor, table or token tensor the matcher refuses."""


# ─────────────────────────────────────────────────────────────────────────────
# Descriptors
# ─────────────────────────────────────────────────────────────────────────────

def descriptor(patches: torch.Tensor, kind: str) -> torch.Tensor:
    """[B, N, D] patch tokens -> [B, N, D] unit-norm descriptors.

    `patches` has the prefix rows ALREADY removed by `capture_stages` using
    the model's own count — this function never sees a prefix row and never
    slices one off itself.

    A zero-norm row (possible after centring, if a patch happens to equal
    the image mean) would divide by zero; it is left at zero instead, which
    gives it cosine 0 against everything rather than a NaN that would
    poison an argmax.
    """
    if kind not in DESCRIPTORS:
        raise CorrespondenceError(
            f"unknown descriptor {kind!r}; the declared set is "
            f"{list(DESCRIPTORS)} (D9) and §11 forbids adding to it")
    if patches.ndim != 3:
        raise CorrespondenceError(
            f"expected [B, N_patches, D] patch rows, got "
            f"{tuple(patches.shape)}")
    x = patches.detach().float()
    if kind == "centred_l2":
        x = x - x.mean(dim=1, keepdim=True)
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > 0, x / norm.clamp_min(torch.finfo(x.dtype).tiny),
                       torch.zeros_like(x))


# ─────────────────────────────────────────────────────────────────────────────
# The matcher
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def match(original: torch.Tensor, transformed: torch.Tensor, transform_id: str,
          descriptor_kind: str, *, grid=None):
    """Nearest-neighbour match one batch. Returns a dict of [B]-long arrays.

    `original` and `transformed` are [B, N, D] patch tokens from the two
    forwards of one pair, in the same image order. Both are turned into
    descriptors HERE, with the same rule, so a caller cannot normalize one
    side and not the other.

    Returned keys: `acc_exact`, `acc_1`, `mean_nn_sim`, and `nn_flat`
    ([B, n_shared] int64) — the neighbour each shared position chose, which
    the exceedance split consumes and nothing writes to disk.
    """
    if original.shape != transformed.shape:
        raise CorrespondenceError(
            f"the two members of a pair have different shapes: "
            f"{tuple(original.shape)} vs {tuple(transformed.shape)}")
    n_patches = int(original.shape[1])
    grid = grid_size_for(n_patches) if grid is None else int(grid)

    t_idx, o_idx = correspondence(transform_id, grid)
    if not is_bijection(t_idx, o_idx, grid):
        raise CorrespondenceError(                       # pragma: no cover
            f"the correspondence table for {transform_id!r} on a {grid}x{grid} "
            f"grid is not a bijection — refusing to score against it")

    o_desc = descriptor(original, descriptor_kind)                 # [B, N, D]
    t_desc = descriptor(transformed, descriptor_kind)[:, t_idx, :]  # [B, S, D]

    # Both sides are unit-norm, so the cosine similarity IS the inner product.
    sim = torch.matmul(t_desc, o_desc.transpose(1, 2))             # [B, S, N]
    best_sim, nn = sim.max(dim=2)                                  # [B, S]

    target = torch.as_tensor(o_idx, dtype=torch.long, device=nn.device)
    exact = (nn == target.unsqueeze(0))
    dist = torch.as_tensor(
        chebyshev(nn.cpu().numpy(), o_idx[None, :], grid), dtype=torch.long)
    within1 = (dist <= 1)

    # COUNTS, not accuracies: `n_correct / n_shared` is derived by the
    # analysis from two exact integers (see records.CORR_ACCURACY_COUNTS).
    return {
        "n_correct_exact": exact.sum(dim=1).cpu().numpy().astype(np.int64),
        "n_correct_1": within1.sum(dim=1).cpu().numpy().astype(np.int64),
        "mean_nn_sim": best_sim.float().mean(dim=1).cpu().numpy(),
        "exact_flags": exact.cpu().numpy(),                # [B, S] bool
        "within1_flags": within1.cpu().numpy(),            # [B, S] bool
        "n_shared": int(t_idx.size),
        "target_idx": o_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAD exceedance, and the split it exists for
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def exceedance_flags(patches: torch.Tensor, k: float = MAD_K) -> torch.Tensor:
    """[B, N] bool — patches whose norm exceeds this image's median + k*MAD.

    The same arithmetic `saga.metrics.sink_counts_mad` counts, reached
    through `saga.frozen.diag.mad_threshold` so there is ONE definition of
    the threshold in the project rather than a fourth copy of it.
    `tests/test_I5_readout.py` asserts `flags.sum(1) == sink_counts_mad(...)`
    exactly, so this cannot drift from the metric the paper reports.
    """
    from saga.metrics import token_norms
    norms = token_norms(patches)                                   # [B, N]
    return norms > mad_threshold(norms, k).unsqueeze(1)


def split_by_exceedance(result: dict, exc: torch.Tensor) -> dict:
    """Correct-counts at exceedance positions vs the rest, per image (T_I5d).

    A position is scored as "exceedance" by the flag of the ORIGINAL image's
    CORRESPONDING patch — the answer key's position, not the query's. That is
    the quantity the table is about: whether a patch whose norm is an outlier
    is one that a consumer can still locate.

    Returns COUNTS, in six integer columns. An image with no exceedance
    position has `n_shared_exc == 0`, and the analysis reports MISSING for
    its accuracy rather than a zero — a mean over an empty set is not zero,
    and MISSING is never averaged (I0 handoff §8.2). Storing the counts says
    that without a string column standing in for an undefined float.
    """
    target = torch.as_tensor(result["target_idx"], dtype=torch.long)
    sel = exc.detach().cpu()[:, target]                    # [B, S] bool
    exact = torch.as_tensor(result["exact_flags"])
    within1 = torch.as_tensor(result["within1_flags"])

    out = {}
    for name, mask in (("exc", sel), ("nonexc", ~sel)):
        out[f"n_shared_{name}"] = mask.sum(dim=1).numpy().astype(np.int64)
        for key, flags in (("n_correct_exact", exact),
                           ("n_correct_1", within1)):
            out[f"{key}_{name}"] = (flags & mask).sum(
                dim=1).numpy().astype(np.int64)
    return out
