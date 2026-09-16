"""
saga/frozen/runner.py
=====================
The one evaluation loop every frozen work package uses.

    load a manifest row by run_id        -> arch / variant / gate_mode /
                                            recipe_actual / ckpt_path
    verify ckpt_sha256 against the row   -> refuse a checkpoint that moved
    build the model through tools/model_factory.py with the RECORDED
      gate_mode                          -> exactly as the run was trained
    load a split by its recorded sha256  -> refuse a split that changed
    run the declared conditions          -> from a per-work-package YAML,
                                            never from the command line
    write records under results/frozen/<WP>/<run_id>/

Determinism: inference only, `torch.no_grad()`, fp32 for the logit
comparisons, batches in ONE image order keyed by `image_id` (the split's
own order), no shuffling, no DistributedSampler.

Conditions are read from a small YAML per work package. No condition,
layer, epsilon, mask or permutation is selectable from the command line —
that is what keeps "the analysis configuration was fixed before the
evaluation outcomes were inspected" a property of the repository and not a
promise (plan §6.2).

NO training, NO optimizer, NO probe fitting: this module imports none.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saga.frozen import records as rec
from saga.frozen.edits import (TERMINAL_GATE_VALUES, EditError, gate_edit,
                               receiver_perturbation, state_hash,
                               terminal_gate_override)
from saga.frozen.stages import (ALL_STAGES, HIST_STAGE, capture_stages,
                                resolve_stage)
from saga.metrics import (effective_rank, infer_num_prefix_tokens,
                          oversmoothing_pairwise, oversmoothing_pairwise_nosink,
                          sink_counts_fixed, sink_counts_mad, token_norms)

MISSING = "MISSING"

#: Edits a conditions YAML may name, and the callable each maps to.
EDIT_TYPES = {
    "native": None,
    "identity": None,
    "terminal_gate_override": terminal_gate_override,
    "gate_edit": gate_edit,
    "receiver_perturbation": receiver_perturbation,
}

#: Model variants a condition may declare itself applicable to. A condition
#: with no `applies_to` applies to all three.
VARIANTS = ("baseline", "registers", "saga")


class RunnerError(RuntimeError):
    """A configuration the runner refuses to execute."""


# ─────────────────────────────────────────────────────────────────────────────
# Inputs: manifest row, split, conditions
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest_row(manifest_json, run_id: str, ckpt_kind: str = "last"):
    """The manifest row for (run_id, ckpt_kind), or a clear refusal."""
    doc = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    rows = [r for r in doc["rows"]
            if r["run_id"] == run_id and r["ckpt_kind"] == ckpt_kind]
    if not rows:
        raise RunnerError(
            f"no manifest row for run_id={run_id!r} ckpt_kind={ckpt_kind!r} "
            f"in {manifest_json}")
    if len(rows) > 1:
        raise RunnerError(
            f"{len(rows)} manifest rows for run_id={run_id!r} "
            f"ckpt_kind={ckpt_kind!r} — the manifest key is not unique")
    return rows[0]


#: The 300-epoch cohort a frozen work package sweeps (I0 handoff §2): one row
#: per condition, `last` checkpoints only, ablation / fine-tuned / dense / TTR
#: rows excluded by family.
COHORT_FAMILIES = ("e2r_300ep", "legacy_300ep")
COHORT_STATUSES = ("eligible", "eligible_legacy")


def eligible_cohort(manifest_json, *, families=COHORT_FAMILIES,
                    statuses=COHORT_STATUSES, ckpt_kind="last") -> list:
    """The eligible 300-epoch cohort rows, in ONE deterministic order.

    BASELINES FIRST, then registers and saga, each block sorted by
    (arch, recipe_actual, variant, provenance_tag, run_id). The order is a
    contract: a job array indexes into it, so it must not depend on the
    manifest's row order, on a dict iteration, or on a filesystem listing.
    Baselines lead because they are what the per-stage thresholds are
    calibrated from, so a partial array (`--array=0-7`) already covers every
    calibration source.
    """
    doc = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    rows = [r for r in doc["rows"]
            if r.get("family") in families and r.get("status") in statuses
            and r.get("ckpt_kind") == ckpt_kind]
    if not rows:
        raise RunnerError(
            f"{manifest_json} has no rows with family in {list(families)}, "
            f"status in {list(statuses)} and ckpt_kind={ckpt_kind!r}")
    return sorted(rows, key=lambda r: (
        0 if r["variant"] == "baseline" else 1, r["arch"],
        r["recipe_actual"], r["variant"], str(r.get("provenance_tag", "")),
        r["run_id"]))


def eligible_run_ids(manifest_json, **kw) -> list:
    """The cohort's run_ids in the array order `eligible_cohort` defines."""
    return [r["run_id"] for r in eligible_cohort(manifest_json, **kw)]


def verify_checkpoint(row: dict, ckpt_path=None) -> str:
    """sha256 the checkpoint and check it against the manifest row.

    A row whose `ckpt_sha256` is still MISSING (Phase A, before the HPC hash
    pass) is refused: running an intervention against an unidentified
    checkpoint is exactly the provenance hole this task exists to close.
    """
    from saga.run_registry import file_sha256

    expect = row.get("ckpt_sha256", MISSING)
    if expect in (MISSING, "", None):
        raise RunnerError(
            f"manifest row {row['run_id']}/{row['ckpt_kind']} has "
            f"ckpt_sha256={expect!r}. Run tools/frozen_manifest_hashes.py on "
            f"the HPC first — an intervention on an unidentified checkpoint "
            f"is not a result.")
    path = Path(ckpt_path or row["ckpt_path"])
    if not path.exists():
        raise RunnerError(f"checkpoint {path} does not exist")
    got = file_sha256(path)
    if got != expect:
        raise RunnerError(
            f"checkpoint {path} hashes to {got}, manifest says {expect} — "
            f"refusing to evaluate a checkpoint that moved")
    return got


def split_sha256(split_doc: dict) -> str:
    """The split's own recorded sha, or a recompute over its canonical form
    (tools/build_frozen_splits.py writes the same digest)."""
    recorded = split_doc.get("sha256")
    payload = {k: v for k, v in split_doc.items() if k != "sha256"}
    got = hashlib.sha256(
        json.dumps(payload, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    if recorded is not None and recorded != got:
        raise RunnerError(
            f"split {split_doc.get('name')!r} records sha256={recorded} but "
            f"hashes to {got} — it has been edited since it was frozen")
    return got


def load_split(split_json):
    """(items, sha256). Items are [relative_path, label] in the split's own
    fixed order; `image_id` is derived from the path, never from an index."""
    doc = json.loads(Path(split_json).read_text(encoding="utf-8"))
    sha = split_sha256(doc)
    items = doc["items"]
    return doc, items, sha


def image_id_for(rel_path: str) -> str:
    """`<synset>/<stem>` — stable under a re-listing, unlike a dataset index."""
    p = Path(rel_path)
    return f"{p.parent.name}/{p.stem}"


def load_conditions(conditions_yaml):
    """The declared conditions for one work package.

    Shape:
        work_package: I0_smoke
        stage: s12_pre_norm            # or hist
        precision: fp32
        conditions:
          - id: native
            edit_type: native
          - id: terminal_gate_0.50
            edit_type: terminal_gate_override
            params: {value: 0.5}

    A condition may additionally declare (TASK I2 §3):

        applies_to: [saga]                 # default: every variant
        stages: [s11_out, hist]            # default: the document's `stage`

    Both are validated HERE, on the login node, so a misspelt variant or an
    unknown stage costs a parse and not a staged dataset. A work package is
    either single-stage throughout or multi-stage throughout: a half-declared
    document would write two different row shapes into one file.
    """
    import yaml
    doc = yaml.safe_load(Path(conditions_yaml).read_text(encoding="utf-8"))
    for key in ("work_package", "stage", "conditions"):
        if key not in doc:
            raise RunnerError(f"{conditions_yaml}: missing key {key!r}")
    resolve_stage(doc["stage"])
    seen = set()
    for c in doc["conditions"]:
        if "id" not in c or "edit_type" not in c:
            raise RunnerError(
                f"{conditions_yaml}: every condition needs `id` and "
                f"`edit_type`, got {c!r}")
        if c["edit_type"] not in EDIT_TYPES:
            raise RunnerError(
                f"{conditions_yaml}: unknown edit_type {c['edit_type']!r}; "
                f"expected one of {sorted(EDIT_TYPES)}")
        if c["id"] in seen:
            raise RunnerError(
                f"{conditions_yaml}: duplicate condition id {c['id']!r}")
        seen.add(c["id"])
        _check_applies_to(conditions_yaml, c)
        _check_stages(conditions_yaml, c)
        if c["edit_type"] == "terminal_gate_override":
            value = float((c.get("params") or {}).get("value", float("nan")))
            if value not in TERMINAL_GATE_VALUES:
                raise RunnerError(
                    f"{conditions_yaml}: condition {c['id']!r} asks for "
                    f"terminal gate value {value!r}, which is not one of the "
                    f"declared constants {TERMINAL_GATE_VALUES}")
    declared = [("stages" in c) for c in doc["conditions"]]
    if any(declared) and not all(declared):
        raise RunnerError(
            f"{conditions_yaml}: {sum(declared)} of {len(declared)} conditions "
            f"declare `stages`. A work package is single-stage or multi-stage "
            f"throughout — mixing them would write two row shapes into one "
            f"diag file")
    doc.setdefault("precision", "fp32")
    return doc


def _check_applies_to(conditions_yaml, c: dict):
    applies = c.get("applies_to")
    if applies is None:
        return
    if not isinstance(applies, list) or not applies:
        raise RunnerError(
            f"{conditions_yaml}: condition {c['id']!r} has applies_to="
            f"{applies!r}; expected a non-empty list of {list(VARIANTS)}")
    unknown = [v for v in applies if v not in VARIANTS]
    if unknown:
        raise RunnerError(
            f"{conditions_yaml}: condition {c['id']!r} applies_to names "
            f"{unknown}, which are not model variants; expected a subset of "
            f"{list(VARIANTS)}")


def _check_stages(conditions_yaml, c: dict):
    stages = c.get("stages")
    if stages is None:
        return
    if not isinstance(stages, list) or not stages:
        raise RunnerError(
            f"{conditions_yaml}: condition {c['id']!r} has stages={stages!r}; "
            f"expected a non-empty list of {list(ALL_STAGES)}")
    if len(set(stages)) != len(stages):
        raise RunnerError(
            f"{conditions_yaml}: condition {c['id']!r} repeats a stage in "
            f"{stages}")
    real = [resolve_stage(s) for s in stages]        # raises on an unknown one
    if len(set(real)) != len(real):
        raise RunnerError(
            f"{conditions_yaml}: condition {c['id']!r} declares {stages}, but "
            f"two of those names resolve to the same stage ({real}) — one "
            f"capture would be written as two rows")


def condition_stages(conditions: dict, c: dict) -> tuple:
    """The stages one condition is captured at: its own list, or the
    document's single `stage`."""
    return tuple(c.get("stages") or (conditions["stage"],))


def conditions_for(conditions: dict, variant: str) -> list:
    """The DECLARED conditions that apply to `variant`, in file order.

    The only source of conditions is the YAML: this returns a subset of it
    and can never return anything that is not in it (TASK I0 handoff §1, the
    conditions contract).
    """
    if variant not in VARIANTS:
        raise RunnerError(
            f"unknown model variant {variant!r}; expected one of "
            f"{list(VARIANTS)}")
    out = [c for c in conditions["conditions"]
           if variant in (c.get("applies_to") or VARIANTS)]
    if not out:
        raise RunnerError(
            f"no declared condition applies to variant {variant!r} in "
            f"{conditions.get('work_package')!r}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model construction from a manifest row
# ─────────────────────────────────────────────────────────────────────────────

def build_from_row(row: dict, ckpt_path=None, device="cpu"):
    """Build and load exactly what the run trained.

    Register models are built by tools/model_factory.build_model as the timm
    model itself (never wrapped in SAGAViT, which refuses
    num_prefix_tokens != 1 by design).
    """
    from tools.model_factory import build_model, load_checkpoint

    gate_mode = row.get("gate_mode")
    gate_mode = None if gate_mode in (MISSING, "", None) else gate_mode
    model = build_model(row["arch"], row["variant"], gate_mode=gate_mode)
    load_checkpoint(model, ckpt_path or row["ckpt_path"])
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# One condition over one batch
# ─────────────────────────────────────────────────────────────────────────────

def _edit_context(model, condition: dict):
    """The context manager for one declared condition (or a null one)."""
    from contextlib import nullcontext

    kind = condition["edit_type"]
    params = dict(condition.get("params") or {})
    if kind in ("native", "identity"):
        return nullcontext({}), params
    fn = EDIT_TYPES[kind]
    if kind == "terminal_gate_override":
        return fn(model, float(params["value"])), params
    if kind == "gate_edit":
        perm = params.get("perm")
        if isinstance(perm, list):
            perm = np.asarray(perm)
        return fn(model, int(params["layer"]), params["mode"],
                  alpha=params.get("alpha"), perm=perm), params
    if kind == "receiver_perturbation":
        mask = np.asarray(params["mask"])
        return fn(model, int(params["layer"]), mask,
                  float(params["epsilon"]),
                  energy_target=params.get("energy_target")), params
    raise RunnerError(f"no context for edit_type {kind!r}")   # pragma: no cover


@torch.no_grad()
def run_condition(model, images, targets, condition, stage, *,
                  native_logits=None, fixed_thr=None, with_diag=True,
                  with_effrank=False):
    """(per-image response dict, per-image diagnostics dict, edit info).

    Everything is computed in fp32: `tools/eval.py` records amp='off' for
    every historical number, and a logit COMPARISON in bf16 would be
    dominated by the cast.
    """
    real_stage = resolve_stage(stage)
    ctx, params = _edit_context(model, condition)
    with ctx as info, capture_stages(model, (real_stage,)) as store:
        logits = model(images).float()
        patches = store[real_stage]

    nll = F.cross_entropy(logits, targets, reduction="none").double()
    top1 = logits.argmax(dim=1)
    resp = {
        "nll": nll.tolist(),
        "top1": top1.tolist(),
        "correct": (top1 == targets).int().tolist(),
        "label": targets.tolist(),
        "max_abs_logit_diff_vs_native": (
            (logits - native_logits).abs().amax(dim=1).tolist()
            if native_logits is not None else [0.0] * logits.shape[0]),
    }

    diag = {}
    if with_diag:
        norms = token_norms(patches)                       # [B, N_patch]
        q = torch.quantile(
            norms.double(),
            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64),
            dim=1)
        cos_ns, excl = [], []
        for i in range(patches.shape[0]):
            c, e = oversmoothing_pairwise_nosink(patches[i:i + 1],
                                                 norms[i:i + 1], k=5.0)
            cos_ns.append(float(c))
            excl.append(float(e))
        diag = {
            "patch_norm_q05": q[0].tolist(), "patch_norm_q25": q[1].tolist(),
            "patch_norm_median": q[2].tolist(),
            "patch_norm_q75": q[3].tolist(), "patch_norm_q95": q[4].tolist(),
            "patch_norm_max": norms.amax(dim=1).tolist(),
            "patch_norm_mean": norms.mean(dim=1).tolist(),
            "count_mad_k5": sink_counts_mad(norms, k=5.0).tolist(),
            "count_fixed_thr": (sink_counts_fixed(norms, fixed_thr).tolist()
                                if fixed_thr is not None
                                else [MISSING] * norms.shape[0]),
            "fixed_thr_value": [fixed_thr if fixed_thr is not None else MISSING]
                               * norms.shape[0],
            "cos_pairwise": oversmoothing_pairwise(patches).tolist(),
            "cos_pairwise_nosink": cos_ns,
            "nosink_excluded": excl,
            "eff_rank": (effective_rank(patches).tolist() if with_effrank
                         else [MISSING] * norms.shape[0]),
        }
    return resp, diag, (dict(info) if isinstance(info, dict) else {}), logits


@torch.no_grad()
def run_condition_multistage(model, images, targets, condition, stages, *,
                             native_logits=None):
    """(response dict, {stage: patch tensor}, edit info, logits) — ONE forward.

    Every declared stage is captured in the SAME forward pass under the SAME
    edit, so `s11_out` and `s12_pre_norm` for one condition are two readouts
    of one computation and not two runs that could differ. `capture_stages`
    stores a tensor under both the declared name and the stage it resolves
    to, so `hist` and `s12_pre_norm` both index the same captured tensor.

    The functional response is identical to `run_condition`'s: it does not
    depend on the stage, because the classification logits are the model's
    own head output.
    """
    wanted = tuple(stages)
    ctx, _params = _edit_context(model, condition)
    with ctx as info, capture_stages(model, wanted) as store:
        logits = model(images).float()
        captured = {s: store[s] for s in wanted}

    nll = F.cross_entropy(logits, targets, reduction="none").double()
    top1 = logits.argmax(dim=1)
    resp = {
        "nll": nll.tolist(),
        "top1": top1.tolist(),
        "correct": (top1 == targets).int().tolist(),
        "label": targets.tolist(),
        "max_abs_logit_diff_vs_native": (
            (logits - native_logits).abs().amax(dim=1).tolist()
            if native_logits is not None else [0.0] * logits.shape[0]),
    }
    return resp, captured, (dict(info) if isinstance(info, dict) else {}), \
        logits


# ─────────────────────────────────────────────────────────────────────────────
# The loop
# ─────────────────────────────────────────────────────────────────────────────

def run_work_package(*, row, conditions, dataset, out_dir, device="cpu",
                     batch_size=32, fixed_thr=None, with_effrank=False,
                     split_name, split_sha, git_sha, git_dirty,
                     patch_file=MISSING, ckpt_path=None, max_images=None):
    """Run every declared condition over `dataset` and write the two files.

    `dataset` yields (image_tensor, label) in the SPLIT'S OWN ORDER and
    carries `.items` (the [rel_path, label] list) so `image_id` comes from
    the path. Returns a summary dict.
    """
    from torch.utils.data import DataLoader

    model = build_from_row(row, ckpt_path=ckpt_path, device=device)
    n_prefix = infer_num_prefix_tokens(model)
    hash_before = state_hash(model)
    stage = conditions["stage"]

    items = list(dataset.items)
    if max_images is not None:
        items = items[:max_images]
    ids = [image_id_for(p) for p, _ in items]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    base = rec.provenance(
        run_id=row["run_id"], arch=row["arch"],
        recipe_actual=row["recipe_actual"], variant=row["variant"],
        ckpt_kind=row["ckpt_kind"], ckpt_sha256=row["ckpt_sha256"],
        n_prefix=n_prefix, stage=stage, split_name=split_name,
        split_sha256=split_sha, precision=conditions.get("precision", "fp32"),
        git_sha=git_sha, git_dirty=git_dirty, patch_file=patch_file)

    record_rows, diag_rows = [], []
    native_by_batch = {}
    summary = {"conditions": {}, "state_hash_before": hash_before}

    for cond in conditions["conditions"]:
        cursor = 0
        edit_info = {}
        for b, (images, targets) in enumerate(loader):
            if max_images is not None and cursor >= max_images:
                break
            images = images.to(device)
            targets = targets.to(device)
            if max_images is not None and cursor + images.shape[0] > max_images:
                keep = max_images - cursor
                images, targets = images[:keep], targets[:keep]

            native = native_by_batch.get(b)
            resp, diag, info, logits = run_condition(
                model, images, targets, cond, stage,
                native_logits=native, fixed_thr=fixed_thr,
                with_effrank=with_effrank)
            if cond["edit_type"] == "native":
                native_by_batch[b] = logits
            edit_info = info or edit_info

            n = images.shape[0]
            for i in range(n):
                img_id = ids[cursor + i]
                row_common = dict(base, image_id=img_id,
                                  condition_id=cond["id"])
                params = dict(cond.get("params") or {})
                record_rows.append(dict(
                    row_common,
                    edit_type=cond["edit_type"],
                    edit_params=json.dumps(params, sort_keys=True,
                                           separators=(",", ":"),
                                           default=str),
                    layer=params.get("layer", MISSING),
                    epsilon=params.get("epsilon", MISSING),
                    measured_perturbation_norm=(
                        info.get("measured_perturbation_norm",
                                 [MISSING] * n)[i]
                        if isinstance(info, dict)
                        and info.get("measured_perturbation_norm")
                        else MISSING),
                    **{k: v[i] for k, v in resp.items()}))
                if diag:
                    diag_rows.append(dict(
                        row_common, edit_type=cond["edit_type"],
                        layer=params.get("layer", MISSING),
                        **{k: v[i] for k, v in diag.items()}))
            cursor += n
        summary["conditions"][cond["id"]] = {
            "edit_type": cond["edit_type"], "n_images": cursor,
            "edit_info": {k: v for k, v in edit_info.items()
                          if not isinstance(v, (list, tuple))
                          or len(v) <= 16},
        }

    hash_after = state_hash(model)
    summary["state_hash_after"] = hash_after
    summary["state_restored"] = bool(hash_before == hash_after)
    if not summary["state_restored"]:
        raise EditError(
            f"{row['run_id']}: model state changed across the condition "
            f"sweep ({hash_before[:12]} -> {hash_after[:12]})")

    out_dir = Path(out_dir)
    n_rec, skip_rec = rec.append_rows(out_dir, "records", record_rows)
    n_diag, skip_diag = (rec.append_rows(out_dir, "diag", diag_rows)
                         if diag_rows else (0, 0))
    marker = {
        "run_id": row["run_id"], "ckpt_sha256": row["ckpt_sha256"],
        "work_package": conditions["work_package"], "stage": stage,
        "hist_stage": HIST_STAGE, "split_name": split_name,
        "split_sha256": split_sha, "n_conditions": len(conditions["conditions"]),
        "n_records_appended": n_rec, "n_records_skipped": skip_rec,
        "git_sha": git_sha,
    }
    rec.mark_done(out_dir, "records", marker)
    if diag_rows:
        rec.mark_done(out_dir, "diag", dict(marker,
                                            n_records_appended=n_diag,
                                            n_records_skipped=skip_diag))
    summary.update(n_records=n_rec, n_diag=n_diag, out_dir=str(out_dir))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# The multi-stage loop (TASK I2) — one forward per (condition, batch),
# every declared stage measured from it
# ─────────────────────────────────────────────────────────────────────────────

def _tau_table(conditions, cell, tau_cal_doc, canon_doc):
    """{declared stage: tau_cal} and tau_canon for one cell, or a refusal.

    Every stage any condition declares must be calibrated for this cell. An
    uncalibrated stage is a configuration error — not a MISSING value — and
    is refused here rather than written as half a column.
    """
    from saga.frozen import diag as fdiag

    stages = sorted({s for c in conditions["conditions"]
                     for s in condition_stages(conditions, c)})
    tau_cal = {s: fdiag.tau_cal_for(tau_cal_doc, cell, s) for s in stages}
    missing = [s for s, v in tau_cal.items() if not isinstance(v, float)]
    if missing:
        raise RunnerError(
            f"cell {cell!r} has no calibrated tau at stage(s) {missing}. Run "
            f"tools/frozen_I2_thresholds.py for this split first — a stage "
            f"with no threshold cannot record count_fixed_cal.")
    return tau_cal, fdiag.canon_tau_for(cell, canon_doc)


def run_work_package_stages(*, row, conditions, dataset, out_dir, device="cpu",
                            batch_size=32, split_name, split_sha, git_sha,
                            git_dirty, patch_file=MISSING, ckpt_path=None,
                            max_images=None, tau_cal_doc=None, canon_doc=None):
    """Run every DECLARED condition that applies to this row's variant, at
    every stage that condition declares, and write records / diag / maps.

    The multi-stage sibling of `run_work_package`: same inputs, same
    provenance, same append-safe writers, same state-restoration proof. What
    differs is that a diag row is keyed by (condition, STAGE, image) and
    carries the TASK I2 §4 diagnostics, and that the per-position exceedance
    maps are aggregated into `maps.npz`.

    Conditions come only from `conditions`; which of them apply to this
    checkpoint comes only from the row's `variant`.
    """
    from torch.utils.data import DataLoader

    from saga.frozen import diag as fdiag

    model = build_from_row(row, ckpt_path=ckpt_path, device=device)
    n_prefix = infer_num_prefix_tokens(model)
    hash_before = state_hash(model)
    report_stage = conditions["stage"]

    cell = fdiag.cell_key(row["arch"], row["recipe_actual"])
    tau_cal, tau_canon = _tau_table(conditions, cell, tau_cal_doc or {},
                                    canon_doc or {})

    declared = conditions_for(conditions, row["variant"])
    if declared[0]["edit_type"] != "native":
        raise RunnerError(
            f"the first condition applying to variant {row['variant']!r} is "
            f"{declared[0]['id']!r} ({declared[0]['edit_type']}). "
            f"`max_abs_logit_diff_vs_native` is measured against the native "
            f"pass, which must therefore be declared first.")

    items = list(dataset.items)
    if max_images is not None:
        items = items[:max_images]
    ids = [image_id_for(p) for p, _ in items]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    base = rec.provenance(
        run_id=row["run_id"], arch=row["arch"],
        recipe_actual=row["recipe_actual"], variant=row["variant"],
        ckpt_kind=row["ckpt_kind"], ckpt_sha256=row["ckpt_sha256"],
        n_prefix=n_prefix, stage=report_stage, split_name=split_name,
        split_sha256=split_sha, precision=conditions.get("precision", "fp32"),
        git_sha=git_sha, git_dirty=git_dirty, patch_file=patch_file)

    record_rows, diag_rows = [], []
    maps = fdiag.MapAccumulator()
    native_by_batch = {}
    summary = {"conditions": {}, "state_hash_before": hash_before,
               "cell": cell, "tau_cal": tau_cal, "tau_canon": tau_canon,
               "skipped_conditions": sorted(
                   {c["id"] for c in conditions["conditions"]}
                   - {c["id"] for c in declared})}

    for cond in declared:
        stages = condition_stages(conditions, cond)
        cursor = 0
        edit_info = {}
        for b, (images, targets) in enumerate(loader):
            if max_images is not None and cursor >= max_images:
                break
            images = images.to(device)
            targets = targets.to(device)
            if max_images is not None and cursor + images.shape[0] > max_images:
                keep = max_images - cursor
                images, targets = images[:keep], targets[:keep]

            resp, captured, info, logits = run_condition_multistage(
                model, images, targets, cond, stages,
                native_logits=native_by_batch.get(b))
            if cond["edit_type"] == "native":
                native_by_batch[b] = logits
            edit_info = info or edit_info

            n = images.shape[0]
            params = dict(cond.get("params") or {})
            for i in range(n):
                record_rows.append(dict(
                    base, image_id=ids[cursor + i], condition_id=cond["id"],
                    edit_type=cond["edit_type"],
                    edit_params=json.dumps(params, sort_keys=True,
                                           separators=(",", ":"), default=str),
                    layer=params.get("layer", MISSING),
                    epsilon=params.get("epsilon", MISSING),
                    measured_perturbation_norm=MISSING,
                    **{k: v[i] for k, v in resp.items()}))

            for stage in stages:
                patches = captured[stage]
                canon_tau = (tau_canon if fdiag.canon_defines_stage(stage)
                             else MISSING)
                values, stage_maps = fdiag.patch_diagnostics(
                    patches, tau_cal=tau_cal[stage], tau_canon=canon_tau)
                maps.add(cond["id"], stage, stage_maps)
                for i in range(n):
                    diag_rows.append(dict(
                        base, image_id=ids[cursor + i],
                        condition_id=cond["id"], stage=stage,
                        stage_resolved=resolve_stage(stage),
                        edit_type=cond["edit_type"],
                        n_patches=int(patches.shape[1]),
                        tau_cal_value=tau_cal[stage],
                        tau_canon_value=(str(canon_tau)
                                         if isinstance(canon_tau, float)
                                         else MISSING),
                        **{k: v[i] for k, v in values.items()}))
            cursor += n
        summary["conditions"][cond["id"]] = {
            "edit_type": cond["edit_type"], "n_images": cursor,
            "stages": list(stages),
            "edit_info": {k: v for k, v in edit_info.items()
                          if not isinstance(v, (list, tuple)) or len(v) <= 16},
        }

    hash_after = state_hash(model)
    summary["state_hash_after"] = hash_after
    summary["state_restored"] = bool(hash_before == hash_after)
    if not summary["state_restored"]:
        raise EditError(
            f"{row['run_id']}: model state changed across the condition "
            f"sweep ({hash_before[:12]} -> {hash_after[:12]})")

    out_dir = Path(out_dir)
    n_rec, skip_rec = rec.append_rows(out_dir, "records", record_rows)
    n_diag, skip_diag = rec.append_rows(
        out_dir, "diag", diag_rows, key=rec.DIAG_STAGE_KEY,
        columns=rec.DIAG_STAGE_COLUMNS)

    marker = {
        "run_id": row["run_id"], "ckpt_sha256": row["ckpt_sha256"],
        "work_package": conditions["work_package"], "stage": report_stage,
        "hist_stage": HIST_STAGE, "split_name": split_name,
        "split_sha256": split_sha, "n_conditions": len(declared),
        "n_records_appended": n_rec, "n_records_skipped": skip_rec,
        "cell": cell, "tau_cal": tau_cal, "tau_canon": tau_canon,
        "git_sha": git_sha,
    }
    fdiag.write_maps_npz(out_dir, maps, dict(marker, kind="maps"))
    rec.mark_done(out_dir, "records", marker)
    rec.mark_done(out_dir, "diag", dict(marker, n_records_appended=n_diag,
                                        n_records_skipped=skip_diag))
    summary.update(n_records=n_rec, n_diag=n_diag, out_dir=str(out_dir))
    return summary
