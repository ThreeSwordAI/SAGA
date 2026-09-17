"""
saga/frozen/features.py
=======================
TASK C / I5 §3 — feature capture for a transform PAIR, the correspondence
sweep loop, and the conditions-file validator the runner hooks.

    check_readout_block(...)          the runner's single hook (§7)
    CorrespondencePairDataset         a frozen split as (original, transformed)
    capture_pair(...)                 two forwards, every declared stage, one edit
    run_correspondence_package(...)   the loop that writes corr_records

ONE FORWARD PER (IMAGE, TRANSFORM) PER SIDE
-------------------------------------------
Every declared stage is captured in the SAME forward, exactly as
`run_condition_multistage` does for I2: `s11_out`, `hist` and
`s12_post_norm` for one image under one edit are three readouts of ONE
computation, never three passes that could differ. The pair costs two
forwards because the two images are different images.

THE STAGES, AND WHY THREE
-------------------------
`s11_out` is D1 — the stage every new patch diagnostic in this project is
reported at (I2 handoff §0). `hist` is where every historical number was
computed, so the readout can be put beside them. `s12_post_norm` is what a
dense head actually consumes, which is the only one of the three that a
"feature consumer" argument can be made about directly. All three come from
the same forward, so a difference between them is a difference of stage and
of nothing else.

THE TWO CONTROLS (§7)
---------------------
    T0          must reproduce the plain evaluation forward BIT-FOR-BIT.
                `t0_matches_plain_forward` in the run summary records the
                measured max |difference| per stage, so the control is a
                number in the output and not an assumption in a docstring.
    term_1.00   cannot change `s11_out` — it swaps the LAST block's gate,
                and s11_out is the output of the second-to-last block. The
                per-row `max_abs_s11_diff_vs_native` carries the measured
                value, so Phase C asserts it on the real records rather than
                trusting the argument.

NO TRAINING, NO FITTING, NO SELECTION
-------------------------------------
Nothing here has an optimizer, a loss, or a fitted parameter: the matcher is
a nearest neighbour and the descriptors are a normalization. The transforms,
stages and descriptors come from the committed YAML and from D9; the sweep
refuses a YAML that names anything else, which is what `check_readout_block`
is for and why it is wired into `saga/frozen/runner.py` rather than into
this package's own driver.
"""

import json
from pathlib import Path

import numpy as np
import torch

from saga.frozen import records as rec
from saga.frozen.correspondence import (DESCRIPTORS, exceedance_flags, match,
                                        split_by_exceedance)
from saga.frozen.stages import capture_stages, resolve_stage
from saga.frozen.transforms import (MATCHED_TRANSFORMS, TRANSFORMS,
                                    chance_exact, chance_within_1,
                                    grid_size_for, pair_transform)

MISSING = "MISSING"

#: The stage D1 fixed for every new patch diagnostic (I2 handoff §0). Recorded
#: on EVERY row as `count_mad_s11` whatever the row's own stage is, so
#: T_I5e can correlate a readout against the diagnostic the paper reports.
D1_STAGE = "s11_out"

#: One row per (condition, transform, stage, descriptor, image).
CORR_KEY = ("condition_id", "transform", "stage", "descriptor", "image_id")


def model_device(model):
    """The device the MODEL'S parameters are on.

    Derived from the model rather than threaded through as an argument: the
    batch has to land wherever the weights already are, and a `device=`
    parameter can silently disagree with that. It did — the first I5a array
    (job 4266049, 2026-09-17) failed 19 of 19 with

        RuntimeError: Input type (torch.FloatTensor) and weight type
        (torch.cuda.FloatTensor) should be the same

    because this loop accepted `device` and never applied it to the images.
    Reading it off the parameters cannot get out of sync that way.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:                                # pragma: no cover
        return torch.device("cpu")


def to_device(batch, device):
    """Move one tensor onto `device`. The seam the tests count calls on.

    A device mismatch cannot be caught by a CPU-only test — both sides are
    `cpu` and everything passes — so `tests/test_I5_readout.py` asserts
    instead that this function is CALLED for every member of every batch,
    which is exactly what was missing.
    """
    return batch.to(device) if hasattr(batch, "to") else batch


def as_missing_str(value) -> str:
    """A float as a round-tripping string, or the literal MISSING.

    The columns in `records.CORR_MISSING_COLUMNS` are undefined for some
    rows — a `native` row has no difference against itself, an image with no
    exceedance position has no accuracy at exceedance positions — and a
    parquet column cannot hold both a float and a string. `repr` of a float
    round-trips exactly under `float()`, so nothing is lost to formatting.
    """
    if value is None or value is MISSING:
        return MISSING
    return repr(float(value))


class FeatureError(ValueError):
    """A readout configuration this module refuses to run."""


# ─────────────────────────────────────────────────────────────────────────────
# The runner hook (§7) — the conditions contract, extended to I5's keys
# ─────────────────────────────────────────────────────────────────────────────

def check_readout_block(conditions_yaml, doc: dict):
    """Validate a conditions document that declares a `readout` (TASK C / I5).

    Called from `saga/frozen/runner.py::load_conditions` — the project's ONE
    place where a committed conditions file is parsed — so I5's transforms
    and descriptors are refused on the login node by the same contract that
    already refuses an unknown edit_type or stage, instead of being read
    loosely by a driver of its own. A document with no `readout` key is
    untouched, which is every work package but this one.

    The point of the contract (I0 handoff §1): no transform, stage or
    descriptor is selectable from the command line, and a name that is not in
    D9 cannot be run at all.
    """
    readout = doc.get("readout")
    if readout is None:
        return doc
    if readout != "correspondence":
        raise FeatureError(
            f"{conditions_yaml}: readout={readout!r}; the only declared "
            f"readout is 'correspondence' (TASK C / I5 §3)")

    transforms = doc.get("transforms")
    if not isinstance(transforms, list) or not transforms:
        raise FeatureError(
            f"{conditions_yaml}: a correspondence readout must declare a "
            f"non-empty `transforms` list, got {transforms!r}")
    unknown = [t for t in transforms if t not in TRANSFORMS]
    if unknown:
        raise FeatureError(
            f"{conditions_yaml}: transforms {unknown} are not declared. The "
            f"set is {sorted(TRANSFORMS)}; a transform without exact grid "
            f"correspondence has no answer key (task file §3, §11)")
    if len(set(transforms)) != len(transforms):
        raise FeatureError(
            f"{conditions_yaml}: `transforms` repeats an entry: {transforms}")
    if not any(t in MATCHED_TRANSFORMS for t in transforms):
        raise FeatureError(
            f"{conditions_yaml}: `transforms` is {transforms}, which scores "
            f"nothing — T0 is the reference forward, so at least one of "
            f"{list(MATCHED_TRANSFORMS)} must be declared")

    descriptors = doc.get("descriptors")
    if not isinstance(descriptors, list) or not descriptors:
        raise FeatureError(
            f"{conditions_yaml}: a correspondence readout must declare a "
            f"non-empty `descriptors` list, got {descriptors!r}")
    unknown = [d for d in descriptors if d not in DESCRIPTORS]
    if unknown:
        raise FeatureError(
            f"{conditions_yaml}: descriptors {unknown} are not declared. The "
            f"set is {list(DESCRIPTORS)} (D9), primary first")
    if len(set(descriptors)) != len(descriptors):
        raise FeatureError(
            f"{conditions_yaml}: `descriptors` repeats an entry: "
            f"{descriptors}")
    for c in doc.get("conditions", []):
        if not c.get("stages"):
            raise FeatureError(
                f"{conditions_yaml}: condition {c.get('id')!r} declares no "
                f"`stages`. A correspondence sweep is multi-stage throughout "
                f"— the readout is reported at every stage it captures")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# The dataset: a frozen split, as pairs
# ─────────────────────────────────────────────────────────────────────────────

class CorrespondencePairDataset(torch.utils.data.Dataset):
    """A frozen split yielding (original, transformed, label) for ONE transform.

    Deliberately parallel to `tools/frozen_eval.FrozenSplitDataset` — same
    split file, same image order, same `items` attribute, so `image_id` comes
    from the path and not from an index — with the single-image transform
    replaced by the pair pipeline of `saga/frozen/transforms.py`.
    """

    def __init__(self, root, items, transform_id: str, input_size: int = 224):
        self.root = Path(root)
        self.items = list(items)
        self.transform_id = transform_id
        self._pair = pair_transform(transform_id, input_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        rel, label = self.items[idx]
        img = Image.open(self.root / rel).convert("RGB")
        original, transformed = self._pair(img)
        return original, transformed, int(label)


# ─────────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def capture_pair(model, original, transformed, stages):
    """({stage: [B,N,D]}, {stage: [B,N,D]}) — one forward per side.

    The caller supplies the edit context (or none); this function does not
    open one, so the SAME edit is in force for both members of a pair by
    construction. Every declared stage of one side comes from one forward.
    """
    wanted = tuple(stages)
    out = []
    for batch in (original, transformed):
        with capture_stages(model, wanted) as store:
            model(batch)
            out.append({s: store[s].clone() for s in wanted})
    return out[0], out[1]


@torch.no_grad()
def plain_forward_stages(model, images, stages):
    """The stage captures of an ordinary forward — the T0 reference (§7)."""
    with capture_stages(model, tuple(stages)) as store:
        model(images)
        return {s: store[s].clone() for s in stages}


# ─────────────────────────────────────────────────────────────────────────────
# The sweep
# ─────────────────────────────────────────────────────────────────────────────

def _edit_context(model, condition):
    """The declared edit for one condition, reusing the runner's table.

    Imported inside the function: `saga/frozen/runner.py` imports this module
    for the hook above, and a module-level import back into it would be a
    cycle.
    """
    from saga.frozen.runner import _edit_context as runner_edit_context
    ctx, _params = runner_edit_context(model, condition)
    return ctx


@torch.no_grad()
def correspondence_rows(model, loader, condition, stages, transform_id, *,
                        descriptors, base_row, image_ids, native_s11=None,
                        max_images=None):
    """Score one (condition, transform) over the loader. Returns (rows, s11).

    `native_s11` is the per-batch `s11_out` of the ORIGINAL side captured
    under `native`; when it is supplied (i.e. for the `term_1.00` condition)
    each row carries `max_abs_s11_diff_vs_native`, the measured evidence for
    the §7 control. The returned dict is this condition's own s11 capture, so
    the caller can pass it forward.
    """
    grid = None
    rows = []
    s11_by_batch = {}
    cursor = 0
    ctx = _edit_context(model, condition)
    with ctx:
        device = model_device(model)
        for b, batch in enumerate(loader):
            original, transformed, _labels = batch
            if max_images is not None:
                if cursor >= max_images:
                    break
                if cursor + original.shape[0] > max_images:
                    keep = max_images - cursor
                    original, transformed = original[:keep], transformed[:keep]

            # BOTH members go where the weights are. Omitting this is what
            # failed the first I5a array 19 of 19 (see `model_device`).
            original = to_device(original, device)
            transformed = to_device(transformed, device)
            orig_st, trans_st = capture_pair(model, original, transformed,
                                             stages)
            n = original.shape[0]
            if grid is None:
                grid = grid_size_for(int(orig_st[stages[0]].shape[1]))

            s11 = orig_st.get(D1_STAGE)
            s11_by_batch[b] = s11
            s11_diff = MISSING
            if native_s11 is not None and s11 is not None:
                prev = native_s11.get(b)
                s11_diff = (float((s11 - prev).abs().max())
                            if prev is not None else MISSING)

            count_s11 = (exceedance_flags(s11).sum(dim=1).cpu().numpy()
                         if s11 is not None else None)

            for stage in stages:
                exc = exceedance_flags(orig_st[stage])
                count_stage = exc.sum(dim=1).cpu().numpy()
                for desc in descriptors:
                    result = match(orig_st[stage], trans_st[stage],
                                   transform_id, desc, grid=grid)
                    split = split_by_exceedance(result, exc)
                    for i in range(n):
                        rows.append(dict(
                            base_row,
                            image_id=image_ids[cursor + i],
                            condition_id=condition["id"],
                            stage=stage,
                            transform=transform_id,
                            descriptor=desc,
                            grid=int(grid),
                            n_shared=int(result["n_shared"]),
                            chance_exact=chance_exact(grid),
                            chance_1=chance_within_1(transform_id, grid),
                            n_correct_exact=int(
                                result["n_correct_exact"][i]),
                            n_correct_1=int(result["n_correct_1"][i]),
                            mean_nn_sim=float(result["mean_nn_sim"][i]),
                            count_mad_stage=int(count_stage[i]),
                            count_mad_s11=as_missing_str(
                                count_s11[i] if count_s11 is not None
                                else MISSING),
                            max_abs_s11_diff_vs_native=as_missing_str(
                                s11_diff),
                            **{k: int(v[i]) for k, v in split.items()}))
            cursor += n
    return rows, s11_by_batch


def run_correspondence_package(*, row, conditions, dataset_for, out_dir,
                               image_ids, device="cpu", batch_size=32,
                               split_name, split_sha, git_sha, git_dirty,
                               patch_file=MISSING, ckpt_path=None,
                               max_images=None, model=None):
    """Run every declared (condition, transform) and write `corr_records`.

    `dataset_for(transform_id)` returns the pair dataset for one transform —
    a callable rather than a dataset, because each transform has its own
    preprocessing and the sweep walks the split once per transform.

    Returns a summary dict; the caller writes it into `run_meta.json`.
    """
    from torch.utils.data import DataLoader

    from saga.frozen.edits import EditError, state_hash
    from saga.frozen.runner import conditions_for
    from saga.metrics import infer_num_prefix_tokens

    if model is None:
        from saga.frozen.runner import build_from_row
        model = build_from_row(row, ckpt_path=ckpt_path, device=device)
    model.eval()

    n_prefix = infer_num_prefix_tokens(model)
    hash_before = state_hash(model)
    declared = conditions_for(conditions, row["variant"])
    if declared[0]["edit_type"] != "native":
        raise FeatureError(
            f"the first condition applying to variant {row['variant']!r} is "
            f"{declared[0]['id']!r} ({declared[0]['edit_type']}). "
            f"`max_abs_s11_diff_vs_native` is measured against the native "
            f"pass, which must therefore be declared first.")

    transforms = list(conditions["transforms"])
    descriptors = list(conditions["descriptors"])

    base = rec.provenance(
        run_id=row["run_id"], arch=row["arch"],
        recipe_actual=row["recipe_actual"], variant=row["variant"],
        ckpt_kind=row["ckpt_kind"], ckpt_sha256=row["ckpt_sha256"],
        n_prefix=n_prefix, stage=conditions["stage"], split_name=split_name,
        split_sha256=split_sha, precision=conditions.get("precision", "fp32"),
        git_sha=git_sha, git_dirty=git_dirty, patch_file=patch_file)

    all_rows = []
    summary = {"conditions": {}, "state_hash_before": hash_before,
               "transforms": transforms, "descriptors": descriptors,
               "skipped_conditions": sorted(
                   {c["id"] for c in conditions["conditions"]}
                   - {c["id"] for c in declared})}

    native_s11 = {}
    for cond in declared:
        stages = tuple(cond["stages"])
        per_transform = {}
        for tid in transforms:
            loader = DataLoader(dataset_for(tid), batch_size=batch_size,
                                shuffle=False, num_workers=0)
            rows, s11 = correspondence_rows(
                model, loader, cond, stages, tid, descriptors=descriptors,
                base_row=base, image_ids=image_ids, max_images=max_images,
                native_s11=(native_s11.get(tid)
                            if cond["edit_type"] != "native" else None))
            all_rows.extend(rows)
            per_transform[tid] = len(rows)
            if cond["edit_type"] == "native":
                native_s11[tid] = s11
        summary["conditions"][cond["id"]] = {
            "edit_type": cond["edit_type"], "stages": list(stages),
            "rows_per_transform": per_transform}

    # §7: T0 must reproduce the plain evaluation forward bit-for-bit. The
    # MEASURED max |difference| per stage goes into the summary — a control
    # that is not a number in the output is not a control.
    summary["t0_matches_plain_forward"] = _t0_control(
        model, dataset_for, declared[0], batch_size, max_images)

    hash_after = state_hash(model)
    summary["state_hash_after"] = hash_after
    summary["state_restored"] = bool(hash_before == hash_after)
    if not summary["state_restored"]:
        raise EditError(
            f"{row['run_id']}: model state changed across the correspondence "
            f"sweep ({hash_before[:12]} -> {hash_after[:12]})")

    out_dir = Path(out_dir)
    n_rows, skipped = rec.append_rows(out_dir, "corr_records", all_rows,
                                      key=CORR_KEY,
                                      columns=rec.CORR_COLUMNS)
    marker = {
        "run_id": row["run_id"], "ckpt_sha256": row["ckpt_sha256"],
        "work_package": conditions["work_package"],
        "stage": conditions["stage"], "split_name": split_name,
        "split_sha256": split_sha, "transforms": transforms,
        "descriptors": descriptors, "n_conditions": len(declared),
        "n_records_appended": n_rows, "n_records_skipped": skipped,
        "git_sha": git_sha,
    }
    rec.mark_done(out_dir, "corr_records", marker)
    summary.update(n_records=n_rows, n_skipped=skipped, out_dir=str(out_dir))
    return summary


@torch.no_grad()
def _t0_control(model, dataset_for, condition, batch_size, max_images):
    """max |T0 capture - plain forward| per stage, on the first batch.

    T0's "pair" is the evaluation image twice, so this compares the sweep's
    own capture path against `capture_stages` on the same tensor. A non-zero
    value would mean the pair pipeline had altered the reference image.
    """
    from torch.utils.data import DataLoader

    if "T0" not in TRANSFORMS:                             # pragma: no cover
        return MISSING
    stages = tuple(condition["stages"])
    loader = DataLoader(dataset_for("T0"), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    try:
        original, transformed, _ = next(iter(loader))
    except StopIteration:                                  # pragma: no cover
        return MISSING
    if max_images is not None:
        original = original[:max_images]
        transformed = transformed[:max_images]
    device = model_device(model)
    original = to_device(original, device)
    transformed = to_device(transformed, device)
    pair_side, _ = capture_pair(model, original, transformed, stages)
    plain = plain_forward_stages(model, original, stages)
    out = {s: float((pair_side[s] - plain[s]).abs().max()) for s in stages}
    out["pair_members_identical"] = bool(
        torch.equal(original, transformed))
    out["n_images"] = int(original.shape[0])
    return out


def summary_json(summary: dict) -> str:
    """The run summary as deterministic JSON (numpy scalars included)."""
    return json.dumps(summary, indent=2, sort_keys=True,
                      default=lambda o: (o.item() if isinstance(o, np.generic)
                                         else str(o)))
