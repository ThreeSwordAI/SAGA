"""
saga/frozen/transforms.py
=========================
TASK C / I5 §3 — the three image transforms whose patch correspondence on
the 14x14 grid is EXACT, and the correspondence tables themselves.

    T0   identity                         the reference forward
    T1   horizontal flip                  (r, c) <-> (r, G-1-c), all G*G
    T2   translation by ONE patch         240 -> two 224 crops, 16 px apart
    T3   translation by TWO patches       256 -> two 224 crops, 32 px apart

WHY ONLY THESE FOUR
-------------------
A correspondence readout is only a readout if the answer key is exact. A
scale change, a rotation, or a shift that is not a whole number of patches
maps one patch of the original onto a BLEND of several patches of the
transformed image, so "the nearest neighbour is the corresponding patch"
stops having a defined right answer and the accuracy silently becomes a
measure of how the blend was resolved. Every transform here moves the patch
grid by a whole patch or reflects it, so the answer key is a bijection on
the shared set and nothing is approximate. §11 of the task file forbids the
others, and this docstring is where that rule is written down.

THE TWO-CROP CONSTRUCTION (T2, T3)
----------------------------------
A translation cannot be produced by cropping ONE 224-pixel image: shifting
its content by 16 px would have to invent a 16-px column. So the source is
resized to a larger square S and TWO 224x224 crops are taken from it, offset
horizontally by exactly `dx` pixels:

    left  crop = pixels [0, 224)      <- THE ORIGINAL (task file §3)
    right crop = pixels [dx, dx+224)  <- the transformed image

Both crops come out of the SAME resized image through the SAME normalization,
so a pair differs by a translation and by nothing else. The left crop, not
the standard centre crop, is the original — which is why both members of a
pair pass through identical processing, and which is stated in §3 of the
task file rather than being a choice made here.

    patch column c of the left  crop covers source pixels [16c,      16c+16)
    patch column c' of the right crop covers source pixels [dx+16c', dx+16c'+16)

so the right crop's column c' IS the left crop's column c' + dx/16, and the
shared set is the columns where that lands inside the grid:

    T2  dx = 16, S = 240  ->  c' in [0, 12], c in [1, 13]  ->  14 x 13 = 182
    T3  dx = 32, S = 256  ->  c' in [0, 11], c in [2, 13]  ->  14 x 12 = 168

`dx` must be a whole number of patches and `S` must be `224 + dx`, or the
construction is refused — those are the two conditions that make the table
exact, and they are checked rather than assumed.

GEOMETRY, NOT THE EVALUATION CROP
---------------------------------
T2/T3 resize the source to a SQUARE S x S, which does not preserve the
aspect ratio the way the evaluation transform's short-side resize does. That
is deliberate and it is the price of an exact answer key: the two crops must
be offset by a known pixel count, and a short-side resize makes the offset
depend on the image's shape. What IS shared with the evaluation pipeline is
everything that touches pixel VALUES — bilinear interpolation and the
ImageNet mean/std normalization, both read from the same places
`tools/eval.build_val_transform` reads them. T0 and T1 use the evaluation
transform itself, unchanged.

NOTHING HERE IS SELECTED
------------------------
The four ids, the two offsets and the two resize sizes are fixed in this
module and in D9; no caller may add a transform, and `TRANSFORMS` is the
whole set.
"""

from dataclasses import dataclass

import numpy as np
import torch

#: The patch grid this project's 224/16 models produce. Derived per model at
#: run time (`grid_size_for`), but named here because the tables are built on
#: it and the task file quotes 196/182/168 against it.
DEFAULT_GRID = 14

#: The evaluation input size every transform produces.
INPUT_SIZE = 224


class TransformError(ValueError):
    """A transform, grid or offset whose patch correspondence is not exact."""


@dataclass(frozen=True)
class TransformSpec:
    """One declared transform.

    `kind` is what the pipeline does; `dx_patches` is the horizontal shift in
    PATCHES (0 for identity and flip); `resize` is the square the source is
    resized to before the two crops are taken (None for the single-crop
    transforms, which use the evaluation transform itself).
    """

    id: str
    kind: str                  # "identity" | "hflip" | "translate"
    dx_patches: int = 0
    resize: int = 0
    note: str = ""

    @property
    def dx_pixels(self) -> int:
        return self.dx_patches * patch_size_for_grid(DEFAULT_GRID)


def patch_size_for_grid(grid: int) -> int:
    """The patch side in pixels, from the grid and the fixed input size."""
    if grid <= 0 or INPUT_SIZE % grid != 0:
        raise TransformError(
            f"a {grid}x{grid} grid does not tile a {INPUT_SIZE}-pixel input "
            f"in whole patches; the correspondence would not be exact")
    return INPUT_SIZE // grid


#: THE declared transforms (task file §3, D9). T0 is the reference forward;
#: the other three each have an exact correspondence table below.
TRANSFORMS = {
    "T0": TransformSpec(
        "T0", "identity",
        note="the reference forward; reproduces the plain evaluation pass "
             "bit-for-bit"),
    "T1": TransformSpec(
        "T1", "hflip",
        note="horizontal flip; (r, c) <-> (r, G-1-c), every position shared"),
    "T2": TransformSpec(
        "T2", "translate", dx_patches=1, resize=INPUT_SIZE + 16,
        note="one-patch translation: resize to 240, two 224 crops 16 px apart"),
    "T3": TransformSpec(
        "T3", "translate", dx_patches=2, resize=INPUT_SIZE + 32,
        note="two-patch translation: resize to 256, two 224 crops 32 px apart"),
}

#: The transforms that carry a correspondence table (T0 is the reference).
MATCHED_TRANSFORMS = ("T1", "T2", "T3")


def transform_spec(transform_id: str) -> TransformSpec:
    if transform_id not in TRANSFORMS:
        raise TransformError(
            f"unknown transform {transform_id!r}; the declared set is "
            f"{sorted(TRANSFORMS)} and §11 forbids adding to it (a transform "
            f"without exact grid correspondence has no answer key)")
    return TRANSFORMS[transform_id]


# ─────────────────────────────────────────────────────────────────────────────
# The correspondence tables
# ─────────────────────────────────────────────────────────────────────────────

def flat_index(row, col, grid: int):
    """Row-major flat patch index, the order a ViT emits patch tokens in."""
    return np.asarray(row) * grid + np.asarray(col)


def row_col(flat, grid: int):
    """(row, col) of a flat patch index."""
    flat = np.asarray(flat)
    return flat // grid, flat % grid


def correspondence(transform_id: str, grid: int = DEFAULT_GRID):
    """(transformed_idx, original_idx) — the exact answer key, as flat indices.

    `transformed_idx[i]` is a patch position of the TRANSFORMED image and
    `original_idx[i]` the position of the SAME image content in the ORIGINAL
    image. Both are row-major flat indices on a `grid` x `grid` layout, and
    the pair of arrays is a bijection between two subsets of `range(grid**2)`
    — which `is_bijection` checks and `tests/test_I5_readout.py` asserts for
    every transform.

    The tables are DERIVED from the geometry documented at the top of this
    module, never tabulated by hand: a hand table would be a second source of
    truth for the thing the whole readout depends on.
    """
    spec = transform_spec(transform_id)
    rows, cols = np.divmod(np.arange(grid * grid), grid)

    if spec.kind == "identity":
        t = flat_index(rows, cols, grid)
        return t, t.copy()

    if spec.kind == "hflip":
        # A horizontal flip maps the whole grid onto itself, so every
        # position is shared: the patch at column c of the flipped image is
        # the patch at column G-1-c of the original.
        return flat_index(rows, cols, grid), flat_index(rows, grid - 1 - cols,
                                                        grid)

    # translate: the right crop's column c' is the left crop's column c' + dx.
    dx = int(spec.dx_patches)
    if dx <= 0 or dx >= grid:
        raise TransformError(
            f"{transform_id}: a shift of {dx} patches leaves no shared "
            f"columns on a {grid}-column grid")
    if spec.resize != INPUT_SIZE + dx * patch_size_for_grid(grid):
        raise TransformError(
            f"{transform_id}: resize {spec.resize} does not equal "
            f"{INPUT_SIZE} + {dx} x {patch_size_for_grid(grid)}; the two "
            f"crops would not be offset by a whole number of patches")
    keep = cols < (grid - dx)
    return (flat_index(rows[keep], cols[keep], grid),
            flat_index(rows[keep], cols[keep] + dx, grid))


def is_bijection(transformed_idx, original_idx, grid: int = DEFAULT_GRID
                 ) -> bool:
    """True when the table is a one-to-one map between two subsets of the grid.

    This is the property the readout rests on: if two transformed positions
    claimed the same original position, "the nearest neighbour is the right
    one" would be ill-posed for at least one of them.
    """
    t = np.asarray(transformed_idx)
    o = np.asarray(original_idx)
    if t.shape != o.shape or t.ndim != 1 or t.size == 0:
        return False
    n = grid * grid
    if t.min() < 0 or t.max() >= n or o.min() < 0 or o.max() >= n:
        return False
    return len(set(t.tolist())) == t.size and len(set(o.tolist())) == o.size


def n_shared(transform_id: str, grid: int = DEFAULT_GRID) -> int:
    """How many positions the two images share under `transform_id`."""
    return int(correspondence(transform_id, grid)[0].size)


def chebyshev(a_flat, b_flat, grid: int):
    """Chebyshev (chessboard) distance between two flat positions."""
    ar, ac = row_col(a_flat, grid)
    br, bc = row_col(b_flat, grid)
    return np.maximum(np.abs(ar - br), np.abs(ac - bc))


def chance_exact(grid: int = DEFAULT_GRID) -> float:
    """Chance for `acc_exact`: one right answer among all grid**2 candidates.

    The candidate set is EVERY patch of the original image, not just the
    shared ones — the matcher searches all of them (§3) — so chance does not
    depend on the transform.
    """
    return 1.0 / float(grid * grid)


def chance_within_1(transform_id: str, grid: int = DEFAULT_GRID) -> float:
    """Chance for `acc_1`, computed exactly rather than bounded.

    A uniformly random guess is `correct-1` when it lands in the 3x3
    neighbourhood of the true position — 9 positions in the interior, 6 on an
    edge, 4 in a corner. Averaging the neighbourhood size over the TRUE
    positions this transform actually scores gives the exact chance, which is
    at most 9/grid**2 (the bound the task file quotes) and is smaller
    whenever the shared set touches the border.
    """
    _, original = correspondence(transform_id, grid)
    r, c = row_col(original, grid)
    rows_in = np.minimum(r + 1, grid - 1) - np.maximum(r - 1, 0) + 1
    cols_in = np.minimum(c + 1, grid - 1) - np.maximum(c - 1, 0) + 1
    return float(np.mean(rows_in * cols_in) / float(grid * grid))


# ─────────────────────────────────────────────────────────────────────────────
# The image pipeline
# ─────────────────────────────────────────────────────────────────────────────

def eval_transform(input_size: int = INPUT_SIZE):
    """THE evaluation transform, imported from `tools/eval.py`, not rebuilt.

    T0 and T1 must be the pipeline every historical number in this project
    was produced with; the only way to guarantee that is to call the function
    that produced them.
    """
    from tools.eval import build_val_transform
    return build_val_transform(input_size)


def _imagenet_mean_std():
    """The normalization constants, from the same place `tools/eval.py` reads
    them, so a T2/T3 pair and a T0 forward normalize identically."""
    from timm.data.constants import (IMAGENET_DEFAULT_MEAN,
                                     IMAGENET_DEFAULT_STD)
    return list(IMAGENET_DEFAULT_MEAN), list(IMAGENET_DEFAULT_STD)


def pair_transform(transform_id: str, input_size: int = INPUT_SIZE):
    """A callable PIL image -> (original [3,S,S], transformed [3,S,S]).

    Both members of the returned pair have been through IDENTICAL processing.
    For T2/T3 that is the whole point of the two-crop construction: the pair
    differs by a translation and by nothing else, so a descriptor that fails
    to match cannot be blamed on the two images having been resized, cropped
    or normalized differently.

    T0 returns the evaluation image twice — the reference forward is its own
    "pair", and a test asserts the two tensors are bit-identical.
    """
    spec = transform_spec(transform_id)

    if spec.kind in ("identity", "hflip"):
        base = eval_transform(input_size)

        def _single(img):
            x = base(img)
            if spec.kind == "identity":
                return x, x
            # A horizontal flip of the NORMALIZED tensor equals normalizing a
            # flipped image: the normalization is per-channel and the centre
            # crop is symmetric, so flipping commutes with both. Flipping the
            # tensor keeps the two members of the pair on exactly one
            # decode/resize path, which is what "identical preprocessing"
            # means here.
            return x, torch.flip(x, dims=[-1])

        return _single

    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    mean, std = _imagenet_mean_std()
    dx = spec.dx_pixels
    size = int(spec.resize)
    if size != input_size + dx:
        raise TransformError(
            f"{spec.id}: resize {size} != input {input_size} + offset {dx}; "
            f"the right crop would fall outside the resized image")
    # Bilinear, matching build_val_transform's interpolation='bilinear'.
    resize = T.Resize((size, size),
                      interpolation=T.InterpolationMode.BILINEAR,
                      antialias=True)
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=mean, std=std)

    def _pair(img):
        big = resize(img)
        left = TF.crop(big, 0, 0, input_size, input_size)
        right = TF.crop(big, 0, dx, input_size, input_size)
        return (normalize(to_tensor(left)), normalize(to_tensor(right)))

    return _pair


def grid_size_for(n_patches: int) -> int:
    """The square grid a patch-token count implies, or a refusal.

    Every table in this module is built on a square grid; a token count that
    is not a perfect square would mean the model is not the 224/16 ViT this
    readout is defined for, and guessing a rectangular layout would silently
    produce a wrong answer key.
    """
    grid = int(round(float(n_patches) ** 0.5))
    if grid * grid != int(n_patches):
        raise TransformError(
            f"{n_patches} patch tokens is not a square grid — the I5 "
            f"correspondence tables are defined on a square patch layout")
    return grid
