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
from saga.frozen.edits import (EditError, gate_edit, receiver_perturbation,
                               state_hash, terminal_gate_override)
from saga.frozen.stages import HIST_STAGE, capture_stages, resolve_stage
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
    doc.setdefault("precision", "fp32")
    return doc


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
