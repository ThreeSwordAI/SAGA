#!/usr/bin/env python3
"""
analysis/build_I0_manifest.py
=============================
TASK I0 D1 — one row per saved checkpoint (or checkpoint-derived condition)
in the whole project, with the provenance every later work package needs to
say what it evaluated.

    python analysis/build_I0_manifest.py
        [--out-dir results/frozen/I0_manifest] [--hashes FILE]

Writes `manifest.csv`, `manifest.json` and the generated `eligibility.md`.
`ckpt_sha256` is MISSING until `tools/frozen_manifest_hashes.py` runs on the
HPC (the checkpoints do not exist locally) and is merged back in through
`--hashes`.

DISCIPLINE (TASK I0 §2, and why each rule is here)
--------------------------------------------------
1. **`recipe_actual` is IMPORTED, never redefined and never read from a
   directory name.** `CELLS`, `LEGACY_MEMBERS`, `E2R_MEMBERS` and `VARIANTS`
   come from `analysis/build_pooled_tables.py`, exactly as TASK-07's
   `analysis/address_analysis.py` takes them, so the recipe erratum
   (`results/notes/recipe_erratum.md`: every legacy `*_nomix` directory was
   trained WITH mixup) and the exclusion of the VOID legacy ViT-B mixup-dir
   trio can never drift between the two tables. For the e2r/abl runs the
   recipe is read from the run's OWN resolved config and CROSS-CHECKED
   against its augmentation block, so a config whose `recipe:` key
   contradicted its own mixup/cutmix alphas would be a hard error, not a
   silent mislabel.
2. **Seeds come from TRAINING metadata** (`meta.json` / `config.resolved.yaml`),
   never from a diagnostics JSON — `tools/diagnose.py:67,106` writes its own
   `--seed` default of 0 into every diag file, which describes the
   evaluation and not the run.
3. **Prefix counts and gate parameter counts come from a MODEL**, built at
   the run's own recorded `gate_mode` through `tools/model_factory.py`, so
   they cannot disagree with the code that trained it.
4. **The historical stage is identified, not assumed**: `hist_stage` is
   `saga.frozen.stages.HIST_STAGE`, whose code citation is in that module's
   docstring and is pinned by `tests/test_I0_frozen.py`.
5. **Diagnostics come from the LAST checkpoint** everywhere except the
   fine-tuning family, which selects on a held-out val split and whose
   `best.pth` is therefore its canonical checkpoint (TASK I0 §3). Every
   other `best` row is `superseded` and never eligible.
6. MISSING is a value. It is never averaged, and never replaced by a guess.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.build_pooled_tables import (CELLS, E2R_MEMBERS,  # noqa: E402
                                          LEGACY_MEMBERS, MISSING, VARIANTS)
from saga.frozen.stages import HIST_STAGE, HIST_STAGE_CITATION  # noqa: E402
from tools.derive_runs import ckpt_dir_for  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

FAMILIES = ("e2r_300ep", "legacy_300ep", "ablation_100ep", "finetune",
            "dense_det", "dense_seg", "ttr_edit")

STATUSES = ("eligible", "eligible_legacy", "invalid", "superseded",
            "exploratory", "derived")

COLUMNS = [
    "run_id", "family", "provenance_tag",
    "arch", "variant", "recipe_actual", "recipe_dirname", "recipe_source",
    "seed", "seed_source", "seed_controlled",
    "epochs_completed", "epochs_configured",
    "ckpt_kind", "ckpt_path", "ckpt_sha256", "ckpt_dir_source",
    "n_prefix", "gate_mode", "gate_init_logit",
    "gate_params_registered", "gate_params_trainable",
    "hist_stage", "hist_eval_precision", "hist_split_sha", "recorded_top1",
    "base_run_id", "base_ckpt_sha256", "derived_params",
    "status", "status_reason",
    "git_sha", "git_dirty", "patch_file",
]

# Legacy checkpoint directory names, keyed the way build_pooled_tables keys
# its members: (arch, recipe_dirname, variant) -> the vault directory.
LEGACY_DIRNAME = {"vit_small": "ViT-S", "vit_base": "ViT-B"}
LEGACY_VARIANT = {"baseline": "baseline", "saga": "SAGA", "registers": "registers"}

DIAG_SPLIT = REPO / "results" / "diagsplit" / "val_diag_split.json"

# What the dense trainers ACTUALLY write (TASK I0 §3: "record what weight
# files actually exist per run"). Read from the trainers, not assumed — the
# Phase-B hash pass found six `best.pth` paths that never existed:
#   detection/tools/train.py    saves ckpt/last.pth ONLY. Its docstring
#       (lines 22-23) records that the "new best AP" branch wrote to
#       last.pth and never to a best file; the best-AP state is the JSON
#       pair coco_eval_best.json + detections_val.json, which is exactly the
#       case §3 warned about.
#   segmentation/tools/train.py:438-439  saves ckpt/last.pth and
#       ckpt/best_model.pth — `best_model.pth`, not `best.pth`.
# None means "this family saves no checkpoint of that kind"; the row still
# exists (so the absence is accounted for) with ckpt_path = MISSING.
DENSE_CKPT_FILENAME = {
    "dense_det": {"last": "last.pth", "best": None},
    "dense_seg": {"last": "last.pth", "best": "best_model.pth"},
}

DENSE_NO_BEST_REASON = (
    "no best checkpoint exists: the detection trainer saves ckpt/last.pth "
    "only and records its best-AP state as the JSON pair coco_eval_best.json "
    "+ detections_val.json. Confirmed three independent ways — the trainer's "
    "OUTPUTS block and its only .pth path (detection/tools/train.py:22-23, "
    "51, 346), docs/Task09_handsoff.md:97 (listed as a defect that was "
    "FIXED: a best branch writing to last.pth made best-AP state "
    "indistinguishable from last state), and a directory listing of "
    "dense_ckpt/det_*/ckpt on woody, which holds last.pth alone "
    "(2026-09-16). MISSING is the value, not a gap")


# ─────────────────────────────────────────────────────────────────────────────
# small readers
# ─────────────────────────────────────────────────────────────────────────────

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_yaml(path):
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def final_log_epoch(run_dir: Path):
    log = Path(run_dir) / "log.csv"
    if not log.exists():
        return None
    try:
        rows = list(csv.DictReader(log.open(newline="", encoding="utf-8")))
    except OSError:
        return None
    eps = [int(r["epoch"]) for r in rows if r.get("epoch") not in (None, "")]
    return max(eps) if eps else None


def diag_split_sha() -> str:
    return file_sha256(DIAG_SPLIT) if DIAG_SPLIT.exists() else MISSING


def ckpt_path_str(path, repo: Path = REPO) -> str:
    """REPO-RELATIVE posix for a checkpoint inside the repo, absolute for one
    outside it (vault, woody).

    The manifest is committed and then read ON THE HPC, so an absolute local
    path for an in-repo checkpoint would be a path that exists on exactly one
    machine. Checkpoints that genuinely live elsewhere — the legacy vault
    tree, the ablation `abl_ckpt` root, the dense `dense_ckpt` root — keep
    the absolute path each run recorded.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(repo.resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


# ─────────────────────────────────────────────────────────────────────────────
# model-derived facts (§2.4 — prefix counts come from the MODEL)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_FACTS: dict = {}


def model_facts(arch_short: str, variant: str, gate_mode):
    """(n_prefix, gate_params_registered, gate_params_trainable).

    Built once per (arch, variant, gate_mode) from tools/model_factory.py —
    the same constructor tools/eval.py and tools/diagnose.py use — so the
    numbers are the model's, never a table typed by hand. Registers come
    back as the timm model itself (never wrapped in SAGAViT), which is why
    n_prefix is 5 there and 1 everywhere else.
    """
    key = (arch_short, variant, gate_mode)
    if key in _MODEL_FACTS:
        return _MODEL_FACTS[key]
    try:
        from saga.metrics import infer_num_prefix_tokens
        from tools.model_factory import build_model
        model = build_model(arch_short, variant, gate_mode=gate_mode)
        n_prefix = int(infer_num_prefix_tokens(model))
        reg = tr = 0
        for name, p in model.named_parameters():
            if ".attn.gate." in name:
                reg += p.numel()
                tr += p.numel() if p.requires_grad else 0
        facts = (n_prefix, reg, tr)
    except Exception as exc:                          # pragma: no cover
        print(f"  WARNING: could not build {key}: {exc}", file=sys.stderr)
        facts = (MISSING, MISSING, MISSING)
    _MODEL_FACTS[key] = facts
    return facts


# ─────────────────────────────────────────────────────────────────────────────
# recipe_actual, from the resolved config and cross-checked against it
# ─────────────────────────────────────────────────────────────────────────────

def recipe_from_resolved_config(cfg: dict, label: str):
    """(recipe_actual, source_note) for a run whose trainer resolved its own
    augmentation block.

    The `recipe:` key and the resolved `augmentation:` block must agree:
    'mixup' means mixup_alpha > 0 or cutmix_alpha > 0, 'nomix' means both
    are 0. A disagreement is the exact failure the recipe erratum documents
    for the legacy resolver, so it is raised here rather than recorded.
    """
    aug = cfg.get("augmentation") or {}
    mix = float(aug.get("mixup_alpha", 0.0) or 0.0)
    cut = float(aug.get("cutmix_alpha", 0.0) or 0.0)
    derived = "mixup" if (mix > 0 or cut > 0) else "nomix"
    declared = cfg.get("recipe")
    if declared is not None and declared != derived:
        raise SystemExit(
            f"{label}: config.resolved.yaml says recipe={declared!r} but its "
            f"own augmentation block (mixup_alpha={mix}, cutmix_alpha={cut}) "
            f"resolves to {derived!r}. Refusing to label a cell from a key "
            f"the config contradicts (results/notes/recipe_erratum.md).")
    return derived, (f"config.resolved.yaml augmentation "
                     f"(mixup_alpha={mix}, cutmix_alpha={cut})")


def legacy_recipe_actual(arch: str, recipe_dirname: str):
    """(recipe_actual, provenance_tag, in_cohort) for a legacy e2 directory.

    Read STRICTLY out of build_pooled_tables.LEGACY_MEMBERS: a (arch,
    recipe_actual) cell lists the directory names that belong to it. A
    directory that appears in no cell is not in the cohort — which is how
    the VOID ViT-B mixup-dir trio stays out without being named here.
    """
    for (a, rec), members in LEGACY_MEMBERS.items():
        if a != arch:
            continue
        for tag, dirname in members:
            if dirname == recipe_dirname:
                return rec, tag, True
    return MISSING, f"legacy-{recipe_dirname}dir", False


# ─────────────────────────────────────────────────────────────────────────────
# row builders, one per family
# ─────────────────────────────────────────────────────────────────────────────

def blank_row(**kw):
    row = {c: MISSING for c in COLUMNS}
    row["hist_stage"] = MISSING
    row.update(kw)
    return row


def classification_rows(runs_root: Path, pattern: str, family: str):
    """e2r_* (300 ep) and abl_* (100 ep): the same trainer, the same
    config.resolved.yaml shape, different budgets and different cohorts."""
    rows = []
    eligible_e2r = {p.format(v=v): (arch, rec, tag)
                    for (arch, rec), members in E2R_MEMBERS.items()
                    for tag, p in members
                    for v in VARIANTS}
    split_sha = diag_split_sha()

    for run_dir in sorted(d for d in runs_root.glob(pattern) if d.is_dir()):
        run_id = run_dir.name
        cfg = read_yaml(run_dir / "config.resolved.yaml") or {}
        meta = read_json(run_dir / "meta.json") or {}
        if not cfg:
            rows.append(blank_row(run_id=run_id, family=family,
                                  status="invalid",
                                  status_reason="no config.resolved.yaml"))
            continue

        arch_full = (cfg.get("model") or {}).get("arch", MISSING)
        arch = {"vit_small_patch16_224": "vit_small",
                "vit_base_patch16_224": "vit_base"}.get(arch_full, arch_full)
        variant = cfg.get("variant", MISSING)
        gate_mode = (cfg.get("model") or {}).get("gate_mode")
        recipe_actual, recipe_src = recipe_from_resolved_config(cfg, run_id)

        seed_cfg = cfg.get("seed")
        seed_meta = meta.get("seed")
        if seed_cfg is not None and seed_meta is not None and seed_cfg != seed_meta:
            raise SystemExit(
                f"{run_id}: config.resolved.yaml seed={seed_cfg} but "
                f"meta.json seed={seed_meta} — refusing to pick one")
        seed = seed_cfg if seed_cfg is not None else (
            seed_meta if seed_meta is not None else MISSING)

        epochs_cfg = (cfg.get("train") or {}).get("epochs", MISSING)
        last_ep = final_log_epoch(run_dir)
        epochs_done = (last_ep + 1) if last_ep is not None else MISSING
        complete = (epochs_done != MISSING and epochs_cfg != MISSING
                    and epochs_done == epochs_cfg
                    and meta.get("end_time") is not None)

        ckpt_dir = ckpt_dir_for(run_dir)
        ckpt_dir_source = ("meta.json[\"ckpt_dir\"]" if meta.get("ckpt_dir")
                           else "<run_dir>/ckpt (no ckpt_dir recorded)")
        n_prefix, greg, gtr = model_facts(arch, variant, gate_mode)

        if family == "e2r_300ep":
            cell = eligible_e2r.get(run_id)
            tag = cell[2] if cell else f"s{seed}"
            in_cohort = cell is not None
        else:
            tag = f"s{seed}"
            in_cohort = True

        for kind in ("last", "best"):
            ev = read_json(run_dir / "eval" / f"imagenet_val_{kind}.json") or {}
            if not complete:
                # an unfinished run has no usable checkpoint of ANY kind; a
                # `best` row here would read as "valid but not selected"
                status = "invalid"
                reason = (f"training incomplete: {epochs_done} of "
                          f"{epochs_cfg} epochs, end_time="
                          f"{meta.get('end_time')!r}")
            elif kind == "last":
                if family == "ablation_100ep":
                    status = "eligible"
                    reason = ("100-epoch ablation arm; a separate stratum, "
                              "never pooled with the 300-epoch cohort")
                elif in_cohort:
                    status = "eligible"
                    reason = (f"300-epoch cohort member "
                              f"{arch}|{recipe_actual}|{variant} ({tag})")
                else:
                    status = "exploratory"
                    reason = ("complete, but not a cell member in "
                              "analysis/build_pooled_tables.py")
            else:
                status = "superseded"
                reason = ("diagnostics come from the LAST checkpoint "
                          "(TASK I0 §2.9); best.pth is recorded as existing "
                          "and is never eligible")

            rows.append(blank_row(
                run_id=run_id, family=family, provenance_tag=tag, arch=arch,
                variant=variant, recipe_actual=recipe_actual,
                recipe_dirname=cfg.get("recipe", MISSING),
                recipe_source=recipe_src, seed=seed,
                seed_source="config.resolved.yaml + meta.json (training)",
                seed_controlled=1, epochs_completed=epochs_done,
                epochs_configured=epochs_cfg, ckpt_kind=kind,
                ckpt_path=ckpt_path_str(ckpt_dir / f"{kind}.pth"),
                ckpt_dir_source=ckpt_dir_source, n_prefix=n_prefix,
                gate_mode=gate_mode if gate_mode else MISSING,
                gate_init_logit=(meta.get("knobs") or {}).get(
                    "gate_init_logit", MISSING),
                gate_params_registered=greg, gate_params_trainable=gtr,
                hist_stage=HIST_STAGE if (run_dir / "diag").is_dir() else MISSING,
                hist_eval_precision=ev.get("amp", MISSING),
                recorded_top1=ev.get("top1", MISSING),
                hist_split_sha=(split_sha if (run_dir / "diag").is_dir()
                                else MISSING),
                status=status, status_reason=reason,
                git_sha=meta.get("git_sha", MISSING),
                git_dirty=int(bool(meta.get("git_dirty"))),
                patch_file=MISSING))
    return rows


def legacy_rows(manifest_csv: Path, forensics_csv: Path):
    """One row per legacy e2 checkpoint, from the committed manifest.

    Cohort membership comes out of LEGACY_MEMBERS; the VOID ViT-B mixup-dir
    trio therefore lands as `invalid` without being named here, and the
    forensics file supplies the epoch that proves it never finished.
    """
    rows = []
    forensics = {}
    if forensics_csv.exists():
        for r in csv.DictReader(forensics_csv.open(newline="",
                                                   encoding="utf-8")):
            forensics[(r["arch"], r["recipe"], r["variant"], r["filename"])] = r

    split_sha = diag_split_sha()
    for r in csv.DictReader(manifest_csv.open(newline="", encoding="utf-8")):
        if r["exp"] != "e2" or not r["arch"]:
            continue                       # e1/e3/e6 rows are other families
        arch, dirname, variant = r["arch"], r["recipe"], r["variant"]
        kind = Path(r["filename"]).stem                      # best | last
        recipe_actual, tag, in_cohort = legacy_recipe_actual(arch, dirname)
        run_id = f"legacy_e2_{arch}_{dirname}dir_{variant}"
        stem = f"e2_{arch}_{dirname}_{variant}_rlast_{kind}"

        fo = forensics.get((arch, dirname, variant, r["filename"]), {})
        epoch = fo.get("epoch")
        epochs_done = (int(epoch) + 1) if epoch not in (None, "") else MISSING

        cfg_name = f"{LEGACY_DIRNAME[arch]}_{LEGACY_VARIANT[variant]}" + \
                   ("_nomix" if dirname == "nomix" else "")
        cfg_path = REPO / "results" / "legacy" / "resolved_configs" / \
            f"{cfg_name}.yaml"
        cfg = read_yaml(cfg_path) or {}
        epochs_cfg = (cfg.get("train") or {}).get("epochs", MISSING)

        ev = read_json(REPO / "results" / "legacy" / "eval" /
                       f"{stem}.json") or {}
        has_diag = (REPO / "results" / "legacy" / "diag" /
                    f"{stem}.json").exists()

        n_prefix, greg, gtr = model_facts(
            arch, variant, "spatial" if variant == "saga" else None)

        if not in_cohort:
            # VOID for BOTH kinds: a run that never finished has no usable
            # checkpoint, and a `best` row marked merely `superseded` would
            # read as "valid but not selected"
            status = "invalid"
            reason = (f"VOID: the legacy {arch} {dirname}-dir trio never "
                      f"finished training (last.pth at epoch "
                      f"{epoch if epoch not in (None, '') else 'MISSING'} of "
                      f"{epochs_cfg}); it is a member of no cell in "
                      f"analysis/build_pooled_tables.py and appears in no "
                      f"eligible cohort (results/notes/bmixup_forensics.md)")
        elif kind == "best":
            status = "superseded"
            reason = ("diagnostics come from the LAST checkpoint "
                      "(TASK I0 §2.9); best.pth is recorded, never eligible")
        else:
            status = "eligible_legacy"
            reason = (f"300-epoch cohort member {arch}|{recipe_actual}|"
                      f"{variant} ({tag}); recipe_actual from the erratum "
                      f"remap, never the '{dirname}' directory name")

        rows.append(blank_row(
            run_id=run_id, family="legacy_300ep", provenance_tag=tag,
            arch=arch, variant=variant, recipe_actual=recipe_actual,
            recipe_dirname=dirname,
            recipe_source=("analysis/build_pooled_tables.py LEGACY_MEMBERS + "
                           "results/notes/recipe_erratum.md"),
            seed=MISSING,
            seed_source=("MISSING: the legacy trainer recorded no seed; the "
                         "manifest's `seed` column holds the repeat tag "
                         f"{r['seed']!r}"),
            seed_controlled=0, epochs_completed=epochs_done,
            epochs_configured=epochs_cfg, ckpt_kind=kind,
            ckpt_path=r["path"],
            ckpt_sha256=r["sha256"],
            ckpt_dir_source="results/legacy/checkpoint_manifest.csv",
            n_prefix=n_prefix,
            gate_mode=("spatial" if variant == "saga" else MISSING),
            gate_init_logit=(0.0 if variant == "saga" else MISSING),
            gate_params_registered=greg, gate_params_trainable=gtr,
            hist_stage=HIST_STAGE if has_diag else MISSING,
            hist_eval_precision=ev.get("amp", MISSING),
            recorded_top1=ev.get("top1", MISSING),
            hist_split_sha=split_sha if has_diag else MISSING,
            status=status, status_reason=reason,
            git_sha=ev.get("git_sha", MISSING), git_dirty=MISSING,
            patch_file=MISSING))
    return rows


def finetune_rows(runs_root: Path):
    """ft_*: 24 runs. This family SELECTS ON VAL, so best.pth is its
    canonical checkpoint (TASK I0 §3) — the one family where `best` is
    eligible and `last` is not."""
    rows = []
    for run_dir in sorted(d for d in runs_root.glob("ft_*") if d.is_dir()):
        run_id = run_dir.name
        cfg = read_yaml(run_dir / "config.resolved.yaml") or {}
        meta = read_json(run_dir / "meta.json") or {}
        ev = read_json(run_dir / "eval" / "test_final.json") or {}
        arch_full = cfg.get("arch", MISSING)
        arch = {"vit_small_patch16_224": "vit_small",
                "vit_base_patch16_224": "vit_base"}.get(arch_full, arch_full)
        variant = cfg.get("variant", MISSING)
        bb = cfg.get("backbone") or {}
        last_ep = final_log_epoch(run_dir)
        n_prefix, greg, gtr = model_facts(
            arch, variant, "spatial" if variant == "saga" else None)

        for kind in ("best", "last"):
            if kind == "best":
                status, reason = "eligible", (
                    "fine-tuning selects on the frozen val split, so best.pth "
                    "is this family's canonical checkpoint (TASK I0 §3); a "
                    "SEPARATE STRATUM from the 300-epoch cohort")
                sha = ev.get("finetuned_sha256", MISSING)
            else:
                status, reason = "superseded", (
                    "the val-selected best.pth is canonical for this family; "
                    "last.pth is recorded as existing, never eligible")
                sha = MISSING
            rows.append(blank_row(
                run_id=run_id, family="finetune",
                provenance_tag=f"f{cfg.get('ft_seed', MISSING)}",
                arch=arch, variant=variant,
                recipe_actual=MISSING,
                recipe_dirname=cfg.get("dataset", MISSING),
                recipe_source=("no ImageNet recipe: a downstream fine-tune of "
                               "the backbone named in base_run_id"),
                seed=cfg.get("ft_seed", meta.get("ft_seed", MISSING)),
                seed_source="config.resolved.yaml ft_seed + meta.json",
                seed_controlled=0,
                epochs_completed=(last_ep + 1) if last_ep is not None
                else MISSING,
                epochs_configured=cfg.get("epochs", MISSING),
                ckpt_kind=kind,
                ckpt_path=ckpt_path_str(run_dir / "ckpt" / f"{kind}.pth"),
                ckpt_sha256=sha,
                ckpt_dir_source="<run_dir>/ckpt (evaluation/e6_finegrained)",
                n_prefix=n_prefix,
                gate_mode=("spatial" if variant == "saga" else MISSING),
                gate_init_logit=MISSING,
                gate_params_registered=greg, gate_params_trainable=gtr,
                hist_stage=MISSING,
                hist_eval_precision=MISSING, hist_split_sha=MISSING,
                base_run_id=bb.get("run", meta.get("backbone_run", MISSING)),
                base_ckpt_sha256=bb.get("sha256",
                                        meta.get("backbone_sha256", MISSING)),
                derived_params=json.dumps(
                    {"dataset": cfg.get("dataset"),
                     "val_split": cfg.get("val_split"),
                     "val_split_sha256": meta.get("val_split_sha256"),
                     "best_epoch": meta.get("best_epoch")},
                    sort_keys=True, separators=(",", ":")),
                status=status, status_reason=reason,
                git_sha=meta.get("git_sha", MISSING),
                git_dirty=int(bool(meta.get("git_dirty"))),
                patch_file=MISSING))
    return rows


def detection_best_epoch(run_dir: Path):
    """(best_AP_epoch, last_epoch) for a detection run, or (MISSING, MISSING).

    Detection saves no best checkpoint, so the natural question — asked by
    the human on 2026-09-16 — is whether `last.pth` happens to BE the best-AP
    weights. That is answerable from the committed files: `coco_eval_best.json`
    records the best-AP epoch and `log.csv` the last one. It is answered PER
    RUN rather than assumed, because the trainer rewrites last.pth every
    epoch, so a run that peaked earlier would make the two differ.
    """
    best = read_json(run_dir / "coco_eval_best.json") or {}
    epoch = best.get("epoch", best.get("best_epoch"))
    last = final_log_epoch(run_dir)
    return (MISSING if epoch is None else int(epoch),
            MISSING if last is None else int(last))


def dense_rows(root: Path, family: str):
    """det_*/seg_*: record what weight files the run ACTUALLY declares.

    The detection rewrite may have saved only JSON pairs for some runs, so
    a checkpoint this builder cannot see from the committed metadata is
    `MISSING`, which is a value (TASK I0 §3). `submitted: false` runs from
    configs/dense_matrix.yaml are NOT rows: they were never started.
    """
    rows = []
    if not root.is_dir():
        return rows
    for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        meta = read_json(run_dir / "meta.json")
        if meta is None:
            continue
        cfg = read_yaml(run_dir / "config.resolved.yaml") or {}
        arch_full = meta.get("arch", cfg.get("arch", MISSING))
        arch = {"vit_small_patch16_224": "vit_small",
                "vit_base_patch16_224": "vit_base"}.get(arch_full, arch_full)
        variant = meta.get("variant", MISSING)
        ckpt_dir = meta.get("ckpt_dir")
        n_prefix, greg, gtr = model_facts(
            arch, variant, "spatial" if variant == "saga" else None)
        complete = meta.get("end_time") is not None
        last_ep = final_log_epoch(run_dir)
        epochs = (last_ep + 1) if last_ep is not None else MISSING
        epochs_cfg = meta.get("epochs_configured", meta.get("epochs", MISSING))

        for kind in ("last", "best"):
            filename = DENSE_CKPT_FILENAME[family][kind]
            path = (MISSING if filename is None
                    else (f"{ckpt_dir}/{filename}" if ckpt_dir
                          else run_dir / "ckpt" / filename))
            if not complete:
                status = "invalid"
                reason = "meta.json end_time is null (run did not finish)"
            elif kind == "last":
                status = "eligible"
                reason = (f"{family} head on backbone "
                          f"{meta.get('backbone_run', MISSING)}; a SEPARATE "
                          f"STRATUM from the classification cohort")
            elif filename is None:
                status = "superseded"
                be, le = detection_best_epoch(run_dir)
                coincide = (be != MISSING and le != MISSING and be == le)
                if coincide:
                    # HUMAN'S CALL, 2026-09-16, after the alternative (leave
                    # the row empty) was put to them: resolve this row to
                    # last.pth rather than lose a usable checkpoint. It is
                    # not an assumption — detection_best_epoch READ both
                    # epochs from this run's own coco_eval_best.json and
                    # log.csv and they coincide, so last.pth IS the best-AP
                    # weights for this run. TASK-09's defect was that one
                    # could not TELL; here it is computed, recorded, and
                    # re-checked on every build.
                    path = (f"{ckpt_dir}/last.pth" if ckpt_dir
                            else run_dir / "ckpt" / "last.pth")
                reason = DENSE_NO_BEST_REASON + (
                    f". RESOLVED TO ckpt/last.pth for this run: its best-AP "
                    f"epoch ({be}) is also its last epoch ({le}), read from "
                    f"coco_eval_best.json and log.csv, so last.pth carries "
                    f"the best-AP weights. This row therefore names the SAME "
                    f"FILE as the `last` row and shares its sha256 — the "
                    f"`last` row is the eligible one, and any count of "
                    f"checkpoints must de-duplicate on sha256 "
                    f"(derived_params.resolves_to_ckpt_kind says so). A run "
                    f"whose best-AP epoch differed would NOT be resolved this "
                    f"way"
                    if coincide else
                    f". NOT resolvable for this run: its best-AP epoch ({be}) "
                    f"and last epoch ({le}) DIFFER, so the best-AP weights "
                    f"were never saved and are not recoverable; only the "
                    f"best-AP predictions and metrics are")
            else:
                status = "superseded"
                reason = ("diagnostics come from the LAST checkpoint "
                          "(TASK I0 §2.9); this family's best weights are "
                          f"`{filename}`, not `best.pth`")
            rows.append(blank_row(
                run_id=run_dir.name, family=family,
                provenance_tag=f"s{meta.get('seed', MISSING)}",
                arch=arch, variant=variant, recipe_actual=MISSING,
                recipe_dirname=meta.get("task", MISSING),
                recipe_source=("no ImageNet recipe: a dense head on the "
                               "backbone named in base_run_id"),
                seed=meta.get("seed", MISSING),
                seed_source="meta.json seed (dense trainer)",
                seed_controlled=0,
                epochs_completed=epochs, epochs_configured=epochs_cfg,
                ckpt_kind=kind,
                ckpt_path=(MISSING if path is MISSING
                           else ckpt_path_str(path)),
                ckpt_sha256=MISSING,
                ckpt_dir_source=("meta.json[\"ckpt_dir\"]" if ckpt_dir
                                 else "<run_dir>/ckpt (no ckpt_dir recorded)"),
                n_prefix=n_prefix,
                gate_mode=("spatial" if variant == "saga" else MISSING),
                gate_init_logit=MISSING,
                gate_params_registered=greg, gate_params_trainable=gtr,
                hist_stage=MISSING, hist_eval_precision=MISSING,
                hist_split_sha=MISSING,
                base_run_id=meta.get("backbone_run", MISSING),
                base_ckpt_sha256=meta.get("backbone_sha256_observed",
                                          meta.get("backbone_sha256", MISSING)),
                derived_params=json.dumps(
                    {"task": meta.get("task"),
                     "backbone_source": meta.get("backbone_source"),
                     "n_train_images": meta.get("n_train_images"),
                     "n_val_images": meta.get("n_val_images"),
                     # detection saves no best checkpoint, so record whether
                     # last.pth nonetheless holds the best-AP weights, and —
                     # when it does — that this row deliberately resolves to
                     # the SAME FILE as the `last` row, so any count of
                     # checkpoints can de-duplicate on sha256
                     **(dict(zip(("best_ap_epoch", "last_epoch"),
                                 detection_best_epoch(run_dir)),
                             best_weights_are_last_pth=(
                                 detection_best_epoch(run_dir)[0]
                                 == detection_best_epoch(run_dir)[1]),
                             resolves_to_ckpt_kind=(
                                 "last" if (kind == "best" and filename is None
                                            and detection_best_epoch(run_dir)[0]
                                            == detection_best_epoch(run_dir)[1])
                                 else None))
                        if family == "dense_det" else {})},
                    sort_keys=True, separators=(",", ":")),
                status=status, status_reason=reason,
                git_sha=meta.get("git_sha", MISSING),
                git_dirty=int(bool(meta.get("git_dirty"))),
                patch_file=MISSING))
    return rows


def ttr_rows(runs_root: Path):
    """ttr_*: DERIVED edits of a baseline checkpoint, not models.

    A TTR row has no checkpoint of its own: it is (base run_id, neurons file
    sha, layer range, n_neurons) applied at inference by saga/ttr.py. The
    `seed` in its meta.json is the preparation tool's, not a training seed,
    and the row says so.
    """
    rows = []
    for run_dir in sorted(d for d in runs_root.glob("ttr_*") if d.is_dir()):
        cfg = read_yaml(run_dir / "config.resolved.yaml") or {}
        meta = read_json(run_dir / "meta.json") or {}
        ttr = cfg.get("ttr") or {}
        arch = cfg.get("arch", MISSING)
        variant = cfg.get("variant", MISSING)
        n_extra = int(ttr.get("n_extra_tokens", 0) or 0)
        base_n_prefix, greg, gtr = model_facts(arch, variant, None)
        layers = ttr.get("layers_touched") or []
        ev = read_json(run_dir / "eval" / "eval_last.json") or {}
        has_diag = (run_dir / "diag" / "diag_last.json").exists()

        rows.append(blank_row(
            run_id=run_dir.name, family="ttr_edit",
            provenance_tag=ttr.get("base_run_id", MISSING),
            arch=arch, variant=variant,
            recipe_actual=cfg.get("recipe", MISSING),
            recipe_dirname=MISSING,
            recipe_source=("inherited from the base run named in "
                           "base_run_id (tools/ttr_prepare_run.py)"),
            seed=meta.get("seed", MISSING),
            seed_source=("tools/ttr_prepare_run.py --seed: the EDIT's "
                         "construction seed, NOT a training seed; the model "
                         "is the base run's checkpoint unchanged"),
            seed_controlled=0,
            epochs_completed=MISSING, epochs_configured=MISSING,
            ckpt_kind="derived",
            ckpt_path=MISSING,
            ckpt_dir_source=("no checkpoint of its own: an inference-time "
                             "edit of base_run_id's last.pth"),
            n_prefix=(base_n_prefix + n_extra
                      if base_n_prefix != MISSING else MISSING),
            gate_mode=MISSING, gate_init_logit=MISSING,
            gate_params_registered=greg, gate_params_trainable=gtr,
            hist_stage=HIST_STAGE if has_diag else MISSING,
            hist_eval_precision=ev.get("amp", MISSING),
            recorded_top1=ev.get("top1", MISSING),
            hist_split_sha=diag_split_sha() if has_diag else MISSING,
            base_run_id=ttr.get("base_run_id", MISSING),
            base_ckpt_sha256=MISSING,
            derived_params=json.dumps({
                "neurons_file": ttr.get("neurons_file"),
                "neurons_file_sha256": ttr.get("neurons_file_sha256"),
                "n_neurons": ttr.get("n_neurons"),
                "n_extra_tokens": n_extra,
                "layer_range": ([min(layers), max(layers)] if layers
                                else None),
                "criterion": ttr.get("criterion"),
                "outlier_tau": ttr.get("outlier_tau"),
                "tau_key": ttr.get("tau_key"),
            }, sort_keys=True, separators=(",", ":")),
            status="derived",
            status_reason=("an inference-time edit of a baseline checkpoint "
                           "(test-time registers), not a trained model; "
                           "never a member of a training cohort"),
            git_sha=meta.get("git_sha", MISSING),
            git_dirty=int(bool(meta.get("git_dirty"))),
            patch_file=MISSING))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_rows(repo: Path = REPO):
    runs = repo / "results" / "runs"
    rows = []
    rows += classification_rows(runs, "e2r_*", "e2r_300ep")
    rows += legacy_rows(repo / "results" / "legacy" / "checkpoint_manifest.csv",
                        repo / "results" / "legacy" / "ckpt_forensics.csv")
    rows += classification_rows(runs, "abl_*", "ablation_100ep")
    rows += finetune_rows(runs)
    rows += dense_rows(repo / "results" / "detection", "dense_det")
    rows += dense_rows(repo / "results" / "segmentation", "dense_seg")
    rows += ttr_rows(runs)
    for r in rows:
        if r["status"] not in STATUSES:
            raise SystemExit(f"{r['run_id']}: bad status {r['status']!r}")
        if r["family"] not in FAMILIES:
            raise SystemExit(f"{r['run_id']}: bad family {r['family']!r}")
    return rows


def merge_hashes(rows, hashes_json):
    """Fold tools/frozen_manifest_hashes.py output into `ckpt_sha256`.

    NEW VALUES ONLY: a row that already carries a sha (the legacy manifest
    supplies its own) must AGREE, or the merge fails. A historical value is
    never overwritten (TASK I0 §2.6).
    """
    doc = json.loads(Path(hashes_json).read_text(encoding="utf-8"))
    by_path = doc["sha256_by_path"]
    filled = agreed = 0
    for r in rows:
        got = by_path.get(r["ckpt_path"])
        if got in (None, MISSING):
            continue
        if r["ckpt_sha256"] in (MISSING, "", None):
            r["ckpt_sha256"] = got
            filled += 1
        elif r["ckpt_sha256"] != got:
            raise SystemExit(
                f"{r['run_id']}/{r['ckpt_kind']}: committed sha256 "
                f"{r['ckpt_sha256']} disagrees with the hash pass {got} for "
                f"{r['ckpt_path']} — refusing to overwrite recorded "
                f"provenance")
        else:
            agreed += 1
    return filled, agreed


# ─────────────────────────────────────────────────────────────────────────────
# the generated eligibility note
# ─────────────────────────────────────────────────────────────────────────────

def cohort_counts(rows):
    """{(family, status): n} and the 300-epoch breakdown the tests pin."""
    per = {}
    for r in rows:
        per[(r["family"], r["status"])] = per.get(
            (r["family"], r["status"]), 0) + 1
    cohort = [r for r in rows
              if r["family"] in ("e2r_300ep", "legacy_300ep")
              and r["status"] in ("eligible", "eligible_legacy")]
    breakdown = {}
    for r in cohort:
        key = (r["arch"], r["recipe_actual"], r["variant"])
        breakdown[key] = breakdown.get(key, 0) + 1
    return per, breakdown, len(cohort)


def render_eligibility(rows, generated_by: str) -> str:
    per, breakdown, n_cohort = cohort_counts(rows)
    L = []
    A = L.append
    A("# I0 cohort — eligibility")
    A("")
    A(f"Generated by `{generated_by}` from the committed run metadata; every "
      f"number below is counted from `manifest.csv`, none is typed by hand. "
      f"Re-render and diff to check it has not drifted "
      f"(`tests/test_I0_manifest.py`).")
    A("")
    A("## The historical feature stage")
    A("")
    A(f"Every historical patch diagnostic in this project was computed at "
      f"**`{HIST_STAGE}`** — the output of the LAST transformer block, "
      f"**before** the final LayerNorm. Identified by reading the code, not "
      f"assumed:")
    A("")
    A(f"> {HIST_STAGE_CITATION}")
    A("")
    A(f"`saga/frozen/stages.py` exposes that stage as the alias `hist`, and "
      f"`manifest.csv`'s `hist_stage` column carries it for every row whose "
      f"run has a `diag/` directory. Rows with no historical diagnostics "
      f"(fine-tuning, dense heads) carry `MISSING`.")
    A("")
    A("## Rows per family and status")
    A("")
    A("| family | status | rows |")
    A("|---|---|---|")
    for family in FAMILIES:
        for status in STATUSES:
            n = per.get((family, status), 0)
            if n:
                A(f"| {family} | {status} | {n} |")
    A(f"| **total** | | **{len(rows)}** |")
    A("")
    A("## The 300-epoch classification cohort")
    A("")
    A(f"**{n_cohort} eligible conditions** "
      f"(`family in {{e2r_300ep, legacy_300ep}}`, `ckpt_kind=last`, "
      f"`status in {{eligible, eligible_legacy}}`), by cell:")
    A("")
    A("| arch | recipe_actual | variant | conditions |")
    A("|---|---|---|---|")
    for key in sorted(breakdown):
        A(f"| {key[0]} | {key[1]} | {key[2]} | {breakdown[key]} |")
    A("")
    per_variant = {}
    for (arch, rec, variant), n in breakdown.items():
        per_variant[variant] = per_variant.get(variant, 0) + n
    A("By variant: " + ", ".join(f"**{v}** {per_variant.get(v, 0)}"
                                 for v in VARIANTS) + ".")
    A("")
    A(f"Cell membership is IMPORTED from `analysis/build_pooled_tables.py` "
      f"(`CELLS`, `LEGACY_MEMBERS`, `E2R_MEMBERS`), so the recipe erratum "
      f"remap and the exclusion of the VOID legacy ViT-B mixup-dir trio "
      f"cannot drift between this manifest and `results/tables/e2_pooled.csv`. "
      f"`recipe_actual` is never read from a directory name.")
    A("")
    A("### Seed control")
    A("")
    controlled = [r for r in rows
                  if r["family"] in ("e2r_300ep", "legacy_300ep")
                  and r["status"] in ("eligible", "eligible_legacy")
                  and str(r["seed_controlled"]) == "1"]
    A(f"{len(controlled)} of the {n_cohort} eligible conditions were trained "
      f"under the seeded e2r trainer (`seed_controlled=1`); the remaining "
      f"{n_cohort - len(controlled)} are legacy repeats whose trainer "
      f"recorded no seed (`seed_controlled=0`, `seed=MISSING`). Seeds come "
      f"from training metadata only: a diagnostics JSON's `seed` field is "
      f"`tools/diagnose.py`'s evaluation default of 0 and is never read here.")
    A("")
    A("## Excluded rows, with reasons")
    A("")
    A("| run_id | ckpt_kind | status | reason |")
    A("|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["family"], r["run_id"],
                                         r["ckpt_kind"])):
        if r["status"] in ("invalid",):
            A(f"| {r['run_id']} | {r['ckpt_kind']} | {r['status']} | "
              f"{r['status_reason']} |")
    A("")
    A("`superseded` rows (every `best.pth` outside the fine-tuning family) "
      "are omitted from the table above for length; they are in "
      "`manifest.csv` with their own one-line reason. The fine-tuning family "
      "is the one exception: it selects on a frozen val split, so its "
      "`best.pth` is canonical and its `last.pth` is the superseded row.")
    A("")
    A("## Prefix tokens")
    A("")
    npx = {}
    for r in rows:
        if r["n_prefix"] != MISSING:
            npx.setdefault(str(r["n_prefix"]), set()).add(r["variant"])
    for n in sorted(npx):
        A(f"- `n_prefix = {n}`: {', '.join(sorted(npx[n]))}")
    A("")
    A("Counted from a model built through `tools/model_factory.py` at each "
      "run's own recorded `gate_mode`, then `saga.metrics."
      "infer_num_prefix_tokens` — never a hard-coded 1 (audit bug B2). "
      "Register models are the timm model itself: `SAGAViT` refuses "
      "`num_prefix_tokens != 1` by design (`saga/vit.py:125-136`), and the "
      "frozen framework therefore never wraps them.")
    A("")
    A("## Checkpoint hashes")
    A("")
    missing_sha = sum(1 for r in rows if r["ckpt_sha256"] == MISSING)
    A(f"{len(rows) - missing_sha} of {len(rows)} rows carry a "
      f"`ckpt_sha256`; **{missing_sha}** are `MISSING` and are filled by "
      f"`tools/frozen_manifest_hashes.py` on the HPC, where the checkpoints "
      f"live. `saga/frozen/runner.py` REFUSES to evaluate a row whose "
      f"`ckpt_sha256` is still MISSING.")
    A("")
    return "\n".join(L) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def atomic_write(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description="TASK I0 D1 — cohort manifest.")
    p.add_argument("--out-dir", default="results/frozen/I0_manifest")
    p.add_argument("--hashes", default=None,
                   help="JSON from tools/frozen_manifest_hashes.py (Phase B)")
    p.add_argument("--repo", default=str(REPO))
    args = p.parse_args()

    repo = Path(args.repo)
    rows = build_rows(repo)
    if args.hashes:
        filled, agreed = merge_hashes(rows, args.hashes)
        print(f"hashes: {filled} filled, {agreed} already agreed")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "manifest.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    per, breakdown, n_cohort = cohort_counts(rows)
    doc = {
        "generated_by": "analysis/build_I0_manifest.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hist_stage": HIST_STAGE,
        "hist_stage_citation": HIST_STAGE_CITATION,
        "columns": COLUMNS,
        "n_rows": len(rows),
        "n_cohort_300ep": n_cohort,
        "cohort_breakdown": {"|".join(k): v for k, v in sorted(breakdown.items())},
        "rows_per_family_status": {"|".join(k): v for k, v in sorted(per.items())},
        "rows": rows,
    }
    atomic_write(json.dumps(doc, indent=2, sort_keys=False) + "\n",
                 out_dir / "manifest.json")
    atomic_write(render_eligibility(rows, "analysis/build_I0_manifest.py"),
                 out_dir / "eligibility.md")

    print(f"wrote {out_dir}/manifest.csv: {len(rows)} rows")
    print(f"wrote {out_dir}/manifest.json")
    print(f"wrote {out_dir}/eligibility.md")
    print(f"300-epoch eligible conditions: {n_cohort}")
    for key in sorted(breakdown):
        print(f"  {key[0]}|{key[1]}|{key[2]}: {breakdown[key]}")
    missing = sum(1 for r in rows if r["ckpt_sha256"] == MISSING)
    print(f"ckpt_sha256 MISSING on {missing}/{len(rows)} rows "
          f"(filled by tools/frozen_manifest_hashes.py on the HPC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
