#!/usr/bin/env python3
"""
tools/frozen_I5b_seg_eval.py
============================
TASK C / I5b — the existing ADE20K segmentation heads, FROZEN, re-evaluated
for PAIRED PER-IMAGE records.

    python tools/frozen_I5b_seg_eval.py --data_root <ADEChallengeData2016>

WHY THIS EXISTS
---------------
The historical dense numbers are aggregate mIoU: one number per run, with no
pairing, so "SAGA is 0.4 mIoU above baseline" has no interval and no way to
say on which images it differs. I5b re-runs the SAME evaluation on the SAME
weights and writes one record per image, so the comparison becomes a paired
per-image difference with a bootstrap CI — and, for the SAGA head, a second
pass under the terminal-gate bypass, which asks whether a real feature
consumer can see the constant that I2 proved the classifier cannot.

Nothing is trained, fine-tuned or re-tuned. No head is fitted. The
segmentation training code is not modified and not called.

A THIN WRAPPER, NOT A SECOND METRIC
-----------------------------------
The mIoU this project reports comes from `segmentation/tools/train.py`:
a confusion matrix accumulated over `gt != 255` pixels only, with
intersections and unions both derived from it. That is the TASK-09 fix for
bug B4, and it is the reason `segmentation/tools/evaluate.py` is SUPERSEDED
and REFUSES TO RUN — its mIoU ignores `ignore_index` and averages per-IMAGE
IoU, so any number it printed would be void.

So this wrapper imports `update_confusion`, `iou_from_confusion` and
`summarize_confusion` FROM the trainer rather than reimplementing them. The
per-image records are per-image confusion matrices; the dataset-level mIoU
is their SUM, which is bit-identical to the pooled matrix the trainer would
have built (integer counts add), and `tests/test_I5_readout.py` asserts that
equality rather than assuming it.

    per-image  pixel_acc, miou_present (over classes present in THAT image),
               n_labelled_pixels, n_classes_present
    pooled     miou_ss, per-class IoU for all 150 classes, the confusion
               matrix itself

`miou_present` is a PER-IMAGE quantity and is labelled as one everywhere it
appears. It is NOT the dataset metric — averaging it would reproduce exactly
the bug TASK-09 removed — and `miou_ss` never comes from it.

THE GATE (§2, §4): MATCHED-EPOCH WEIGHTS OR NOTHING
---------------------------------------------------
I5b runs only if the manifest carries ViT-B/mixup baseline AND SAGA
segmentation weights at a MATCHED epoch, each with a recorded sha256. If it
does not, this tool REFUSES — it does not retrain, re-fetch, or fall back to
a mismatched pair — and the work package is recorded as DROPPED with the
reason. The check is `matched_seg_rows()`, which reads the manifest and
nothing else.

The registers segmentation row is NOT backbone-matched: its backbone is
`legacy_e2_vitb_registers_nomixdir`, an unseeded legacy checkpoint, while
the baseline and SAGA heads sit on the seeded e2r ViT-B backbones. It is
therefore excluded unless `--include-registers` is passed, and every row it
writes is flagged `backbone_matched=0` so no table can quietly pool it with
the matched pair.

EVAL ONLY — REFUSED IN TRAINING MODE
------------------------------------
`assert_eval_only()` runs before any forward: the model must be in eval
mode, gradients must be off, and no parameter may require a gradient. There
is no optimizer in this file and no code path that creates one; the AST
guard in `tests/test_I5_readout.py` enforces that at the syntax level.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saga.frozen import records as rec  # noqa: E402
from saga.frozen.edits import state_hash, terminal_gate_override  # noqa: E402
from saga.run_registry import file_sha256, git_sha  # noqa: E402

MISSING = "MISSING"

#: The manifest family these weights live in, and the checkpoint kind
#: diagnostics come from everywhere except fine-tuning (I0 handoff §2).
SEG_FAMILY = "dense_seg"
SEG_CKPT_KIND = "last"

#: The two runs the matched comparison needs, by variant.
MATCHED_VARIANTS = ("baseline", "saga")

#: One row per (condition, image).
SEG_KEY = ("condition_id", "image_id")

#: Per-image segmentation records — defined in `saga/frozen/records.py`
#: beside every other schema this project writes, so a reader finds them all
#: in one file rather than one per tool.
SEG_COLUMNS = rec.SEG_COLUMNS


class SegEvalError(RuntimeError):
    """A configuration this evaluator refuses to run."""


# ─────────────────────────────────────────────────────────────────────────────
# The manifest gate
# ─────────────────────────────────────────────────────────────────────────────

def seg_rows(manifest_json, ckpt_kind: str = SEG_CKPT_KIND) -> dict:
    """{variant: row} for the `dense_seg` family, eligible rows only."""
    doc = json.loads(Path(manifest_json).read_text(encoding="utf-8"))
    out = {}
    for r in doc["rows"]:
        if (r.get("family") == SEG_FAMILY
                and r.get("ckpt_kind") == ckpt_kind
                and r.get("status") == "eligible"):
            out[r["variant"]] = r
    return out


def matched_seg_rows(manifest_json, ckpt_kind: str = SEG_CKPT_KIND):
    """(rows, reason) — the matched baseline/SAGA pair, or (None, why not).

    This function IS the §2 condition, in one place: the pair exists, both
    weights are identified by a sha256, and `epochs_completed` is equal. A
    caller that gets `None` records `I5b: DROPPED (reason)` and stops; it
    does not retrain, re-fetch, or substitute a different epoch.
    """
    rows = seg_rows(manifest_json, ckpt_kind)
    have = [v for v in MATCHED_VARIANTS if v in rows]
    if len(have) != len(MATCHED_VARIANTS):
        missing = [v for v in MATCHED_VARIANTS if v not in rows]
        return None, (f"the manifest has no eligible {SEG_FAMILY}/{ckpt_kind} "
                      f"row for variant(s) {missing}")

    pair = {v: rows[v] for v in MATCHED_VARIANTS}
    for v, r in pair.items():
        sha = r.get("ckpt_sha256", MISSING)
        if sha in (MISSING, "", None):
            return None, (f"{r['run_id']} has ckpt_sha256={sha!r} — weights "
                          f"that are not identified are not evaluated")
    epochs = {v: r.get("epochs_completed") for v, r in pair.items()}
    if len({str(e) for e in epochs.values()}) != 1:
        return None, (f"epochs_completed differ across the pair ({epochs}) — "
                      f"§2 requires a MATCHED epoch")
    if any(e in (None, MISSING, "") for e in epochs.values()):
        return None, f"epochs_completed is not recorded for the pair ({epochs})"
    return pair, (f"matched at epochs_completed="
                  f"{list(epochs.values())[0]}: "
                  + ", ".join(f"{v}={pair[v]['run_id']}"
                              for v in MATCHED_VARIANTS))


# ─────────────────────────────────────────────────────────────────────────────
# Eval-only guards
# ─────────────────────────────────────────────────────────────────────────────

def assert_eval_only(model):
    """Refuse anything that is not a frozen forward pass (§4, §11).

    Checked on the MODEL, before any image is read: a wrapper that trained
    would be a different experiment wearing this one's name.
    """
    if model.training:
        raise SegEvalError(
            "the segmentor is in TRAINING mode. I5b is an evaluation of "
            "existing heads only — no training, no fine-tuning, no head "
            "fitting (task file §4, §11).")
    if torch.is_grad_enabled():
        raise SegEvalError(
            "gradients are enabled. I5b runs under torch.no_grad(); a "
            "forward that builds a graph is not the frozen evaluation this "
            "work package declares.")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise SegEvalError(
            f"{len(trainable)} parameter(s) still require a gradient "
            f"(e.g. {trainable[:3]}). Freeze the model before evaluating it.")
    return True


def backbone_of(model):
    """The ViT inside the segmentor — what a terminal-gate override acts on.

    `ViTSegmentor.backbone` is a `DetectionBackbone`, whose `.vit` is the
    SAGAViT (or timm register model) that carries `.blocks`. Reached by
    attribute, and refused loudly if the structure is not what it says.
    """
    backbone = getattr(model, "backbone", None)
    vit = getattr(backbone, "vit", None)
    if vit is None or not hasattr(vit, "blocks"):
        raise SegEvalError(
            f"{type(model).__name__}.backbone.vit does not expose `.blocks`; "
            f"refusing to guess where the terminal gate lives")
    return vit


# ─────────────────────────────────────────────────────────────────────────────
# The evaluation, per image
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_image(model, loader, device, num_classes, image_ids,
                       max_images=None):
    """(rows, pooled_confusion) — one record per image, and their sum.

    The per-image confusion matrices are accumulated with the TRAINER's
    `update_confusion`, so the ignore-index handling is the committed one and
    not a second implementation of it. Their sum is the pooled matrix the
    dataset-level mIoU comes from.
    """
    from segmentation.tools.train import (iou_from_confusion, update_confusion)

    assert_eval_only(model)
    pooled = torch.zeros(num_classes, num_classes, dtype=torch.int64,
                         device=device)
    rows = []
    seen = 0
    for images, masks in loader:
        if max_images is not None and seen >= max_images:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(images).float().argmax(dim=1)
        for i in range(images.shape[0]):
            if max_images is not None and seen >= max_images:
                break
            conf = torch.zeros(num_classes, num_classes, dtype=torch.int64,
                               device=device)
            update_confusion(conf, preds[i], masks[i], num_classes)
            pooled += conf
            iou, inter, gt, pred, union = iou_from_confusion(conf)
            present = gt > 0
            scored = union > 0
            total = int(gt.sum())
            rows.append({
                "image_id": image_ids[seen],
                "n_labelled_pixels": total,
                "n_classes_present": int(present.sum()),
                "pixel_acc": (float(inter.sum() / total) if total
                              else MISSING),
                # PER-IMAGE mIoU over the classes this image scores. NOT the
                # dataset metric — averaging it is bug B4's sibling, and
                # `miou_ss` never comes from it.
                "miou_present": (float(np.mean(iou[scored]))
                                 if int(scored.sum()) else MISSING),
            })
            seen += 1
    return rows, pooled


def pooled_summary(pooled):
    """mIoU / pixel-acc / per-class IoU from the pooled matrix, via the
    trainer's own `summarize_confusion` — the committed dataset metric."""
    from segmentation.tools.train import summarize_confusion

    summary, iou, inter, gt, pred, union = summarize_confusion(pooled)
    return summary, {
        "iou": iou.tolist(), "intersection": inter.tolist(),
        "gt_pixels": gt.tolist(), "pred_pixels": pred.tolist(),
        "union": union.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model construction (no training code is called)
# ─────────────────────────────────────────────────────────────────────────────

def build_frozen_segmentor(matrix, run_id, ckpt_path, expected_sha, device):
    """The committed segmentor for `run_id`, with its trained head loaded.

    `resolve_dense_config` is the same resolver the trainer uses, so the
    architecture is the one the head was trained on; nothing about the
    training schedule is read and no optimizer is constructed.
    """
    from segmentation.models.segmentor import build_segmentor
    from tools.dense_runtime import resolve_backbone, resolve_dense_config

    cfg = resolve_dense_config(matrix, run_id)
    if cfg["task"] != "segmentation":
        raise SegEvalError(f"run {run_id!r} is a {cfg['task']} run")
    cfg["backbone"] = resolve_backbone(cfg["backbone_spec"])

    model = build_segmentor(cfg)
    got = file_sha256(ckpt_path)
    if expected_sha not in (MISSING, "", None) and got != expected_sha:
        raise SegEvalError(
            f"{ckpt_path} hashes to {got}, the manifest says {expected_sha} — "
            f"refusing to evaluate a checkpoint that moved")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v
             for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, got


def main():
    p = argparse.ArgumentParser(
        description="TASK C / I5b — frozen ADE20K evaluation with per-image "
                    "records. Evaluation only; it refuses a training-mode "
                    "call.")
    p.add_argument("--data_root", required=True,
                   help="staged ADEChallengeData2016/")
    p.add_argument("--matrix", default="configs/dense_matrix.yaml")
    p.add_argument("--manifest",
                   default="results/frozen/I0_manifest/manifest.json")
    p.add_argument("--out-root", default="results/frozen/I5_readout/seg")
    p.add_argument("--include-registers", action="store_true",
                   help="also evaluate the registers head. Its backbone is "
                        "NOT matched (legacy, unseeded), so every row it "
                        "writes is flagged backbone_matched=0 and it is "
                        "reported as context, never pooled with the pair.")
    p.add_argument("--max-images", type=int, default=None,
                   help="smoke only: take the first N validation images")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if (torch.cuda.is_available()
                        or not args.device.startswith("cuda")) else "cpu")

    pair, reason = matched_seg_rows(args.manifest)
    if pair is None:
        raise SystemExit(
            f"I5b: DROPPED — {reason}.\n"
            f"Nothing is retrained and nothing is re-fetched (task file §2, "
            f"§4). Record the drop in docs/TASK_LOG.md and the handoff.")
    print(f"I5b gate: {reason}")

    rows_to_run = dict(pair)
    if args.include_registers:
        extra = seg_rows(args.manifest).get("registers")
        if extra is not None:
            rows_to_run["registers"] = extra
            print(f"I5b: including {extra['run_id']} as UNMATCHED context "
                  f"(backbone {extra.get('base_run_id')})")

    from segmentation.data.ade20k_dataset import ADE20KDataset
    from segmentation.data.transforms import get_val_transforms

    out_root = Path(args.out_root)
    manifest_summary = {"gate_reason": reason, "runs": {}}

    for variant, row in rows_to_run.items():
        run_id = row["run_id"]
        model, cfg, sha = build_frozen_segmentor(
            args.matrix, run_id, row["ckpt_path"], row.get("ckpt_sha256"),
            device)
        num_classes = int(cfg["model"]["num_classes"])
        val_size = int(cfg.get("eval", {}).get("input_size",
                                               cfg["train"]["input_size"]))
        val_ds = ADE20KDataset(args.data_root, split="validation",
                               transforms=get_val_transforms(val_size))
        image_ids = list(val_ds.stems)
        loader = torch.utils.data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0)

        conditions = [("native", None)]
        if variant == "saga":
            # The bypass the whole work package turns on: the head was
            # trained on gated features, so this measures SENSITIVITY and is
            # labelled exploratory wherever it is reported (§4).
            conditions.append(("term_1.00", 1.0))

        out_dir = out_root / run_id
        hash_before = state_hash(model)
        for cond_id, gate_value in conditions:
            from contextlib import nullcontext
            ctx = (nullcontext() if gate_value is None
                   else terminal_gate_override(backbone_of(model), gate_value))
            with ctx:
                per_image, pooled = evaluate_per_image(
                    model, loader, device, num_classes, image_ids,
                    max_images=args.max_images)
            summary, per_class = pooled_summary(pooled)
            print(f"  {run_id} / {cond_id}: mIoU_ss={summary['mIoU']} "
                  f"pixel_acc={summary['pixel_acc']} "
                  f"n_images={len(per_image)}")

            base = {
                "run_id": run_id, "variant": variant, "arch": row["arch"],
                "ckpt_kind": row["ckpt_kind"], "ckpt_sha256": sha,
                "backbone_run_id": row.get("base_run_id", MISSING),
                "backbone_matched": int(variant in MATCHED_VARIANTS),
                "epochs_completed": row.get("epochs_completed", MISSING),
                "condition_id": cond_id, "precision": "fp32",
                "git_sha": git_sha(), "git_dirty": 0,
            }
            rec.append_rows(out_dir, "seg_records",
                            [dict(base, **r) for r in per_image],
                            key=SEG_KEY, columns=SEG_COLUMNS)
            np.savez_compressed(
                out_dir / f"conf_matrix_{cond_id}.npz",
                conf=pooled.cpu().numpy(),
                **{k: np.asarray(v) for k, v in per_class.items()})
            manifest_summary["runs"][f"{run_id}/{cond_id}"] = dict(
                summary, n_images=len(per_image), ckpt_sha256=sha,
                backbone_matched=int(variant in MATCHED_VARIANTS))

        hash_after = state_hash(model)
        if hash_before != hash_after:
            raise SegEvalError(
                f"{run_id}: model state changed across the sweep "
                f"({hash_before[:12]} -> {hash_after[:12]})")
        rec.mark_done(out_dir, "seg_records", {
            "run_id": run_id, "ckpt_sha256": sha,
            "work_package": "I5b_seg", "conditions": [c for c, _ in conditions],
            "n_images": len(per_image), "git_sha": git_sha(),
            "state_restored": True})

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "I5b_summary.json").write_text(
        json.dumps(manifest_summary, indent=2, sort_keys=True, default=str)
        + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out_root / 'I5b_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
