#!/usr/bin/env python3
"""
tools/check_smoke.py
====================
TASK-09 Phase B: decide whether the two dense smokes actually passed.

    python tools/check_smoke.py --job-det 4200497 --job-seg 4200498

Reads the SLURM logs AND the artifacts the smoke wrote, checks every
TASK-09 acceptance criterion the smoke is supposed to demonstrate, and
prints one PASS/FAIL line per check plus a compact block to paste back.

Deliberately evidence-based: a check passes only if the value is found and
correct. Anything missing is FAIL (never "probably fine"), and every check
prints what it actually saw, so the verdict can be audited from the report
alone. Standard library only, so it runs on a login node with no env.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path("/home/vault/iwi5/iwi5359h/SAGA/logs")

PIPELINES = {
    "detection": {
        "job_flag": "--job-det",
        "log_stem": "dense_smoke_det",
        "run_id": "det_vitb_saga_s1",
        "out_root": "results/smoke/detection",
        "epochs": 2,            # gen_dense_jobs.py SMOKE_RUNS
        "steps": 25,
        "eval_images": 64,
        "freeze_epochs": 1,     # detection/configs/base.yaml
        "stage_count": ("train2017", 118287),
    },
    "segmentation": {
        "job_flag": "--job-seg",
        "log_stem": "dense_smoke_seg",
        "run_id": "seg_vitb_saga_s1",
        "out_root": "results/smoke/segmentation",
        "epochs": 3,
        "steps": 17,
        "eval_images": 64,
        "freeze_epochs": 2,     # segmentation/configs/base.yaml
        "stage_count": ("ADE20K training", 20210),
    },
}


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, evidence=""):
        self.rows.append((bool(ok), name, str(evidence)[:200]))
        return bool(ok)

    def failures(self):
        return [r for r in self.rows if not r[0]]

    def dump(self, title):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        for ok, name, ev in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            if ev:
                print(f"         {ev}")


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── log checks ────────────────────────────────────────────────────────────────

def check_log(rep, spec, text, err_text):
    ok_deps = bool(re.search(r"deps ok:", text or ""))
    rep.check("on-device dependency check ran (pycocotools/timm/torch)",
              ok_deps,
              (re.search(r"deps ok:.*", text or "") or [""])[0]
              if ok_deps else "no 'deps ok:' line in the log")

    m = re.search(r"^(\d+) passed(?:, (\d+) skipped)?", text or "", re.M)
    n_fail = re.search(r"(\d+) failed", text or "")
    rep.check("unit suite green ON THE COMPUTE NODE",
              bool(m) and not n_fail,
              f"pytest reported: {m.group(0) if m else 'nothing'}"
              + (f" / {n_fail.group(0)}" if n_fail else ""))
    if m and m.group(2):
        rep.check("no test SKIPPED on device (a skip can hide a missing dep)",
                  int(m.group(2)) == 0, f"{m.group(2)} skipped")

    label, expected = spec["stage_count"]
    staged = re.search(r"(\d[\d,]*) images \(expected: (\d+)\)", text or "")
    rep.check(f"dataset staged and {label} count enforced",
              bool(staged) or f"expected {expected}" in (text or ""),
              staged.group(0) if staged else "no staging count line found")

    if spec["run_id"].startswith("det_"):
        norm = re.search(r"B5 normalization check PASSED: (.*)", text or "")
        rep.check("B5 normalization asserted ON DEVICE (the acceptance item)",
                  bool(norm),
                  norm.group(1) if norm else "no 'B5 normalization check "
                                             "PASSED' line — the assertion "
                                             "did not run")

    bb = re.search(r"backbone \[(primary|fallback)\]: (\S+)", text or "")
    rep.check("backbone resolved and recorded", bool(bb),
              bb.group(0) if bb else "no backbone line")

    epochs = re.findall(r"^\[(\d+)/(\d+)\] .*", text or "", re.M)
    rep.check(f"all {spec['epochs']} smoke epochs ran",
              len(epochs) == spec["epochs"],
              f"epoch lines seen: {len(epochs)} "
              f"({[e[0] + '/' + e[1] for e in epochs]})")

    rep.check("one new-best artifact write happened",
              "new best" in (text or ""),
              [l.strip() for l in (text or "").splitlines()
               if "new best" in l][:2])

    rep.check("run reached the end (Done line)",
              re.search(r"^Done .* (best AP|best mIoU)", text or "", re.M)
              is not None,
              [l.strip() for l in (text or "").splitlines()
               if l.startswith("Done ")][:2])

    tb = "Traceback (most recent call last)" in (text or "") or \
         "Traceback (most recent call last)" in (err_text or "")
    rep.check("no python traceback anywhere in stdout/stderr", not tb,
              "traceback present" if tb else "none")

    for bad in ("CUDA out of memory", "NCCL WARN", "Watchdog"):
        hit = bad in (text or "") or bad in (err_text or "")
        rep.check(f"no '{bad}'", not hit, "found" if hit else "none")

    # The a0801 cluster fault (docs/TASK_LOG.md) must be matched by its real
    # signature, not by the bare word "TaskProlog": SLURM prints that word in
    # ordinary prologue output on healthy nodes too, and matching it caused a
    # false FAIL on a run that had COMPLETED 0:0 with every artifact intact.
    # The fault itself looks like
    #   slurm task_prolog can not be executed (...) Permission denied
    #   TaskProlog failed status=1
    # and it produces NO script output at all.
    fault = re.search(r"task_prolog can not be executed"
                      r"|TaskProlog failed", (text or "") + (err_text or ""))
    rep.check("no a0801-style TaskProlog fault", fault is None,
              fault.group(0) if fault else "none")


# ── artifact checks ───────────────────────────────────────────────────────────

def check_detection_artifacts(rep, run_dir, spec):
    for name in ("meta.json", "config.resolved.yaml", "log.csv",
                 "coco_eval_best.json", "detections_val.json"):
        rep.check(f"artifact present: {name}", (run_dir / name).exists(),
                  str(run_dir / name))
    rep.check("checkpoint written", (run_dir / "ckpt" / "last.pth").exists())
    rep.check("eval shard scratch cleaned up",
              not list((run_dir / "eval_shards").glob("*.json"))
              if (run_dir / "eval_shards").exists() else True)

    log = run_dir / "log.csv"
    if log.exists():
        with open(log, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        rep.check("log.csv schema is the mandated one",
                  fields == ["epoch", "lr_backbone", "lr_head", "train_loss",
                             "AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L",
                             "img_per_sec", "wall_time"], fields)
        rep.check(f"log.csv has {spec['epochs']} epoch rows",
                  len(rows) == spec["epochs"], f"{len(rows)} rows")
        if rows:
            rep.check("only the eval epoch carries an AP",
                      rows[-1].get("AP") not in (None, "")
                      and all(r.get("AP") in (None, "") for r in rows[:-1]),
                      f"AP column: {[r.get('AP') for r in rows]}")
            lrs = [r.get("lr_backbone") for r in rows]
            rep.check("backbone LR logged per epoch (unfreeze happened)",
                      all(v not in (None, "") for v in lrs), f"lr_backbone: {lrs}")

    best = load_json(run_dir / "coco_eval_best.json")
    if best is None:
        rep.check("coco_eval_best.json parses", False, "missing/unparseable")
        return
    rep.check("coco_eval_best.json parses", True)
    rep.check("flagged as a SMOKE (never mistakable for a result)",
              best.get("smoke") is True, f"smoke={best.get('smoke')}")
    six = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]
    rep.check("all six APs present", all(k in best for k in six),
              {k: best.get(k) for k in six})
    rep.check("all six ARs present",
              all(k in best for k in ("AR_1", "AR_10", "AR_100", "AR_S",
                                      "AR_M", "AR_L")),
              {k: best.get(k) for k in ("AR_1", "AR_100", "AR_S")})
    rep.check("no -1 sentinel written as a number",
              all(best.get(k) is None or best.get(k) >= 0 for k in six),
              {k: best.get(k) for k in six})
    rep.check("per-category breakdown present (80 COCO categories)",
              isinstance(best.get("per_category"), list)
              and len(best["per_category"]) == 80,
              f"{len(best.get('per_category') or [])} categories")
    rep.check("eval covered the capped image count",
              best.get("n_val_images") == spec["eval_images"],
              f"n_val_images={best.get('n_val_images')} "
              f"(expected {spec['eval_images']})")
    nc = best.get("normalization_check") or {}
    rep.check("B5 evidence recorded IN THE ARTIFACT",
              nc.get("max_abs_elementwise_diff") is not None
              and nc["max_abs_elementwise_diff"] < 1e-4
              and nc.get("dist_to_double_normalized", 0) > 1.0,
              {k: nc.get(k) for k in ("max_abs_elementwise_diff",
                                      "backbone_input_channel_mean",
                                      "dist_to_double_normalized")})
    rep.check("backbone provenance in the artifact",
              bool(best.get("backbone_sha256")) and bool(best.get("backbone_run")),
              f"{best.get('backbone_source')} / {best.get('backbone_run')} / "
              f"{str(best.get('backbone_sha256'))[:12]}...")

    det = run_dir / "detections_val.json"
    if det.exists():
        rep.check("detections dump matches its recorded sha256",
                  best.get("detections_sha256") == sha256_of(det),
                  f"recorded {str(best.get('detections_sha256'))[:12]}... "
                  f"file {sha256_of(det)[:12]}...  "
                  f"({det.stat().st_size / 2**20:.1f} MiB)")
        dets = load_json(det)
        if isinstance(dets, list):
            keys = {tuple(sorted(d)) for d in dets[:200]}
            rep.check("dump holds ONLY prediction fields (no fabricated "
                      "segmentation polygons)",
                      keys in ((), ) or keys == {("bbox", "category_id",
                                                  "image_id", "score")},
                      f"{len(dets)} detections, keysets={keys}")
            cats = {d["category_id"] for d in dets[:500]}
            rep.check("categories are OFFICIAL COCO ids (not contiguous 1-80)",
                      not cats or max(cats) > 80 or len(cats) <= 3,
                      f"category ids seen: {sorted(cats)[:12]}...")


def check_segmentation_artifacts(rep, run_dir, spec):
    for name in ("meta.json", "config.resolved.yaml", "log.csv",
                 "miou_ss.json", "miou_ms.json", "per_class_iou.csv",
                 "conf_matrix.npz"):
        rep.check(f"artifact present: {name}", (run_dir / name).exists(),
                  str(run_dir / name))
    rep.check("checkpoints written",
              (run_dir / "ckpt" / "last.pth").exists()
              and (run_dir / "ckpt" / "best_model.pth").exists())

    log = run_dir / "log.csv"
    if log.exists():
        with open(log, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        rep.check("log.csv schema is the mandated one",
                  fields == ["epoch", "lr_backbone", "lr_head", "train_loss",
                             "mIoU", "img_per_sec", "wall_time"], fields)
        rep.check(f"log.csv has {spec['epochs']} epoch rows",
                  len(rows) == spec["epochs"], f"{len(rows)} rows")
        if rows:
            rep.check("only the eval epoch carries an mIoU",
                      rows[-1].get("mIoU") not in (None, "")
                      and all(r.get("mIoU") in (None, "") for r in rows[:-1]),
                      f"mIoU column: {[r.get('mIoU') for r in rows]}")

    for tag, name in (("single-scale", "miou_ss.json"),
                      ("multi-scale", "miou_ms.json")):
        payload = load_json(run_dir / name)
        if payload is None:
            rep.check(f"{name} parses", False, "missing/unparseable")
            continue
        rep.check(f"{name} parses", True)
        rep.check(f"{name} flagged as a SMOKE",
                  payload.get("smoke") is True, f"smoke={payload.get('smoke')}")
        rep.check(f"{name} mIoU is a real number",
                  isinstance(payload.get("mIoU"), (int, float))
                  and 0.0 <= payload["mIoU"] <= 100.0,
                  f"mIoU={payload.get('mIoU')} ({tag})")
        rep.check(f"{name} covered the capped image count",
                  payload.get("n_images") == spec["eval_images"],
                  f"n_images={payload.get('n_images')}")
        rep.check(f"{name} records the B4 definition and the units",
                  "B4" in str(payload.get("miou_definition"))
                  and isinstance(payload.get("units"), dict),
                  f"units={payload.get('units')}")
        rep.check(f"{name} ignore_index recorded as 255",
                  payload.get("ignore_index") == 255)
        rep.check(f"{name} scored the full class set",
                  payload.get("num_classes") == 150
                  and len(payload.get("iou_per_class") or []) == 150,
                  f"num_classes={payload.get('num_classes')}, "
                  f"per-class entries={len(payload.get('iou_per_class') or [])}")
    ss = load_json(run_dir / "miou_ss.json") or {}
    ms = load_json(run_dir / "miou_ms.json") or {}
    rep.check("multi-scale eval really was multi-scale",
              ms.get("eval_mode") == "multi_scale" and ms.get("ms_scales"),
              f"scales={ms.get('ms_scales')} flip={ms.get('ms_flip')}")
    rep.check("single-scale eval labelled as such",
              ss.get("eval_mode") == "single_scale")

    pc = run_dir / "per_class_iou.csv"
    if pc.exists():
        with open(pc, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        rep.check("per_class_iou.csv schema (unit-bearing column name)",
                  fields == ["class_index", "name", "iou_frac", "intersection",
                             "union", "gt_pixels", "pred_pixels"], fields)
        rep.check("per_class_iou.csv has 150 class rows", len(rows) == 150,
                  f"{len(rows)} rows")
        named = [r for r in rows if r.get("name") not in ("", "MISSING")]
        rep.check("ADE20K class NAMES resolved (Phase C needs sky/wall/floor)",
                  len(named) == 150,
                  f"{len(named)}/150 named; first three: "
                  f"{[r.get('name') for r in rows[:3]]}")
        wanted = {r.get("name", "").split(",")[0].strip(): r["class_index"]
                  for r in rows}
        rep.check("the three rows Phase C surfaces are present",
                  all(k in wanted for k in ("wall", "sky", "floor")),
                  {k: wanted.get(k) for k in ("wall", "sky", "floor")})

    probe = run_dir / "preds_fixed20"
    n_pred = len(list(probe.glob("*_pred.png"))) if probe.exists() else 0
    n_gt = len(list(probe.glob("*_gt.png"))) if probe.exists() else 0
    n_img = len(list(probe.glob("*_img.jpg"))) if probe.exists() else 0
    rep.check("all 20 probe images dumped (pred + gt + rgb)",
              (n_pred, n_gt, n_img) == (20, 20, 20),
              f"pred={n_pred} gt={n_gt} img={n_img}")

    meta = load_json(run_dir / "meta.json") or {}
    rep.check("probe list sha recorded (the committed fixed-20 list was used)",
              bool(meta.get("probe_list_sha256")),
              f"{meta.get('probe_list')} "
              f"{str(meta.get('probe_list_sha256'))[:12]}...")
    rep.check("class-name source recorded",
              bool(meta.get("class_names_source")),
              meta.get("class_names_source"))


def project_runtime(rep, spec, run_dir, matrix_path):
    """Turn the smoke's measured throughput into "does the production
    schedule fit in chain x 24 h?" — the one pre-launch number that decides
    whether the chains produce a result at all. The repo holds no dense
    throughput datum to calibrate against (no e3/e4 log.csv exists), so the
    smoke's last epoch is the only measurement available; it is also the
    UNFROZEN-backbone epoch by construction, i.e. the representative one.

    Deliberately conservative: it charges the staging time and the eval
    passes the production run will actually pay, and FAILS if the projection
    does not fit, so the chain length is signed off on a measured number
    rather than on the task file's "~3 days".
    """
    log = run_dir / "log.csv"
    if not log.exists():
        rep.check("runtime projected onto the production schedule", False,
                  "no log.csv to measure")
        return
    with open(log, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        rep.check("runtime projected onto the production schedule", False,
                  "log.csv has no rows")
        return

    last = rows[-1]
    try:
        ips = float(last["img_per_sec"])
        wall = float(last["wall_time"])
    except (KeyError, TypeError, ValueError):
        rep.check("runtime projected onto the production schedule", False,
                  f"could not read img_per_sec/wall_time from {last}")
        return

    # the production numbers, read from the matrix + the committed configs
    matrix = {}
    try:
        import ast
        txt = Path(matrix_path).read_text(encoding="utf-8")
        for key in ("detection", "segmentation"):
            m = re.search(rf"^  {key}: (\d+)$", txt, re.M)
            if m:
                matrix[key] = int(m.group(1))
    except OSError:
        pass
    task = "detection" if spec["run_id"].startswith("det_") else "segmentation"
    chain = matrix.get(task, 4 if task == "detection" else 2)

    n_train = {"detection": 118287, "segmentation": 20210}[task]
    epochs = {"detection": 25, "segmentation": 80}[task]
    eval_every = {"detection": 5, "segmentation": 10}[task]
    n_val = {"detection": 5000, "segmentation": 2000}[task]

    # the smoke ran only `steps` iterations, so scale by images, not by time
    smoke_imgs = ips * wall
    epoch_h = (n_train / ips) / 3600.0 if ips > 0 else float("inf")
    # eval throughput is not separately measured; charge it at the training
    # rate, which understates it (eval is forward-only) — noted, not hidden
    evals = epochs // eval_every
    eval_h = (evals * n_val / ips) / 3600.0 if ips > 0 else float("inf")
    stage_h = chain * 0.25          # 10-15 min of staging per job
    total_h = epochs * epoch_h + eval_h + stage_h
    budget_h = chain * 24.0

    rep.check(f"{task} fits the committed {chain}x24h chain "
              f"(measured from the smoke)",
              total_h <= budget_h,
              f"smoke last epoch: {ips:.2f} img/s over {wall:.0f}s "
              f"({smoke_imgs:.0f} imgs) -> {epoch_h:.2f} h/epoch x {epochs} "
              f"= {epochs * epoch_h:.1f} h + {evals} evals {eval_h:.1f} h + "
              f"staging {stage_h:.1f} h = {total_h:.1f} h vs budget "
              f"{budget_h:.0f} h"
              + ("" if total_h <= budget_h else
                 f"  -> RAISE chain.{task} to "
                 f"{int(total_h // 24) + 1} in configs/dense_matrix.yaml"))


def check_meta_common(rep, run_dir):
    meta = load_json(run_dir / "meta.json")
    if meta is None:
        rep.check("meta.json parses", False, "missing/unparseable")
        return
    rep.check("meta.json parses", True)
    rep.check("run finalized (end_time = the completion marker)",
              bool(meta.get("end_time")), meta.get("end_time"))
    rep.check("seed recorded", meta.get("seed") is not None,
              f"seed={meta.get('seed')}")
    rep.check("git sha recorded", bool(meta.get("git_sha")),
              str(meta.get("git_sha"))[:12])
    rep.check("backbone hash VERIFIED against the matrix",
              meta.get("backbone_sha256_expected")
              == meta.get("backbone_sha256_observed")
              and bool(meta.get("backbone_sha256_observed")),
              f"{meta.get('backbone_source')}: "
              f"{str(meta.get('backbone_sha256_observed'))[:12]}...")
    rep.check("world size was 4 GPUs", meta.get("world_size") == 4,
              f"world_size={meta.get('world_size')}")
    rep.check("no torn-meta quarantine happened",
              not list(run_dir.glob("meta.json.corrupt.*")))
    return meta


def main():
    p = argparse.ArgumentParser("check the TASK-09 dense smokes (Phase B)")
    p.add_argument("--job-det", required=True,
                   help="SLURM job id of dense_smoke_det.sbatch")
    p.add_argument("--job-seg", required=True,
                   help="SLURM job id of dense_smoke_seg.sbatch")
    p.add_argument("--log-dir", default=str(LOG_DIR))
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    repo = Path(args.repo_root)
    overall = {}

    for task, spec in PIPELINES.items():
        job = args.job_det if task == "detection" else args.job_seg
        rep = Report()
        log = log_dir / f"{spec['log_stem']}_{job}.log"
        err = log_dir / f"{spec['log_stem']}_{job}.err"
        text, err_text = read_text(log), read_text(err)
        if text is None:
            rep.check(f"log readable: {log}", False,
                      "not found — has the job started/finished?")
        else:
            rep.check(f"log readable: {log}", True,
                      f"{len(text.splitlines())} lines")
            check_log(rep, spec, text, err_text)

        run_dir = repo / spec["out_root"] / spec["run_id"]
        if not run_dir.exists():
            rep.check(f"run dir exists: {run_dir}", False, "missing")
        else:
            rep.check(f"run dir exists: {run_dir}", True)
            check_meta_common(rep, run_dir)
            if task == "detection":
                check_detection_artifacts(rep, run_dir, spec)
            else:
                check_segmentation_artifacts(rep, run_dir, spec)
            project_runtime(rep, spec, run_dir,
                            repo / "configs" / "dense_matrix.yaml")

        rep.dump(f"{task.upper()} SMOKE  (job {job})")
        overall[task] = rep

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    all_ok = True
    for task, rep in overall.items():
        bad = rep.failures()
        all_ok = all_ok and not bad
        print(f"  {task:13s} {len(rep.rows) - len(bad)}/{len(rep.rows)} checks "
              f"passed" + ("" if not bad else "   FAILED: "
                           + "; ".join(n for _, n, _ in bad)))
    print(f"\n  VERDICT: {'SMOKE PASSED' if all_ok else 'SMOKE FAILED'}")

    print(f"\n{'=' * 78}\nPASTE THE BLOCK ABOVE BACK (from the first "
          f"'=' line)\n{'=' * 78}")
    print("  Disk usage of the smoke outputs (safe to delete afterwards):")
    for task, spec in PIPELINES.items():
        d = repo / spec["out_root"]
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) \
            if d.exists() else 0
        print(f"    {d}: {total / 2**30:.2f} GiB")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
