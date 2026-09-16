#!/bin/bash
#SBATCH --job-name=saga_frozen_I1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-18
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I1_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I1_%A_%a.err

# TASK A / I1 A10 — per-token norms, per-image exceedance indicators and the
# frequency maps at s11_out, in_b07, in_b08 and hist (plus hist under
# term_1.00 for the 8 SAGA runs), over the 19 eligible 300-epoch conditions,
# one array task per run_id.
#
#     sbatch scripts/jobs/frozen_I1.sbatch results/diagsplit/val_diag_split.json
#     sbatch scripts/jobs/frozen_I1.sbatch results/frozen/splits/evaluation.json
#
# THE SPLIT IS AN ARGUMENT, NOT A DEFAULT, and the two runs mean different
# things:
#
#   discovery   (results/diagsplit/val_diag_split.json) — SELECTION material.
#               D5's masks are built from it and from nothing else.
#               saga/frozen/prevalence.py refuses to build a mask from any
#               other split, by sha.
#   evaluation  (results/frozen/splits/evaluation.json) — REPORTING. Every
#               table the paper shows comes from it.
#
# Neither is a fallback for the other. A default here would make "selection
# and reporting never mix" a habit rather than a property of the repository.
#
# WHAT THIS JOB DOES NOT RECOMPUTE. TASK I2 already produced the exceedance
# maps at s11_out and hist on both splits, and Phase C loads them. This job
# exists for what I2 did not store: the per-token NORM FIELD (the scale-only
# null rescales it and re-thresholds it) and the per-image INDICATORS (every
# image-level bootstrap and the split-half reliability need them), and for
# the two BLOCK-INPUT stages, which no committed artifact contains. s11_out
# and hist ride along in the same forward passes at no extra cost, and their
# counts must reproduce I2's — Phase C checks that they do.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1 / cpus-per-task=4, the strings of the repo's own a100
# array jobs (scripts/jobs/frozen_I2.sbatch, ft_finegrained_array.sbatch). a40
# also works and is the cheaper queue for an eval-only job — switch on the
# COMMAND LINE rather than copying this file (docs/HPC_WORKFLOW.md
# "Partitions"):
#     sbatch --partition=a40 --gres=gpu:a40:1 scripts/jobs/frozen_I1.sbatch <split>
# Keep --exclude=a0801 on the command line, where the repo's other a100
# launchers put it.
#
# --time=04:00:00: staging ImageNet val is ~10 min (How to Run.md §5); the
# sweep itself is ONE forward pass over the split for a baseline/register run
# (four stages from one pass) and TWO for a SAGA run (native + term_1.00).
# That is fewer passes than I2 needed and there is no batched SVD here, so
# the wall time is dominated by data loading. A task that hits the wall
# leaves NOTHING extra — every npz is written to a temp file and renamed, and
# the completion marker is written last — so it is simply resubmitted.
#
# ── DISK ─────────────────────────────────────────────────────────────────────
# norms_<stage>.npz is float32 [n_images x 196]: ~8 MB per stage per run on
# the 10,000-image splits, so ~32 MB per baseline run and ~40 MB per SAGA run,
# ~700 MB per split over the cohort. It is GIT-IGNORED and stays here.
# maps_<stage>.npz is ~270 kB and IS committed.
#
# ── RESUBMISSION ─────────────────────────────────────────────────────────────
# Idempotent by construction: `--skip-if-done` exits 0 for a run whose
# completion marker already names this exact checkpoint, and
# frozen_I2_thresholds.py --if-missing is a no-op once every cell is
# calibrated for this split. Resubmit the whole array; it adds what is
# missing and rewrites nothing.

SPLIT="${1:?usage: sbatch [--array=0-18] scripts/jobs/frozen_I1.sbatch <split.json>
       discovery:  sbatch scripts/jobs/frozen_I1.sbatch results/diagsplit/val_diag_split.json
       evaluation: sbatch scripts/jobs/frozen_I1.sbatch results/frozen/splits/evaluation.json
       Both are needed: discovery for D5's masks, evaluation for every table.}"

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
# TASK-11's committed val-only stager. stage_imagenet.sh would also pull the
# five train shards — 10x the copy for a job that never reads them.
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

# `set -u` ONLY AFTER the env sourcing: env_alex.sh sources /etc/profile,
# whose site scripts dereference unset variables (debuginfod.sh line 8,
# DEBUGINFOD_URLS), and under `set -u` bash aborts the CALLING script before
# one line of the job body runs (docs/HPC_WORKFLOW.md, TASK-10 failure #2).
set -u

CONDITIONS=configs/frozen/I1_spatial.yaml
MANIFEST=results/frozen/I0_manifest/manifest.json
OUT_ROOT=results/frozen

# ── Interpreter preflight: FATAL, and BEFORE any staging ─────────────────────
# Never `which python`: if the module load fails, PATH still resolves to
# /usr/bin/python, which has no torch. An unchecked import cost TASK-10 a job
# that staged 50 000 images and then died in Python.
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
echo "split:  $SPLIT"
echo "task:   ${SLURM_ARRAY_TASK_ID:-<none>}"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is not executable. The conda env location is in" >&2
    echo "       How to Run.md sections 1/2. If it moved, resubmit with" >&2
    echo "       SAGA_PY=/path/to/python sbatch $0 $SPLIT" >&2
    exit 1
fi
if ! $PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm — refusing to stage 50 000" >&2
    echo "       images for a job that cannot run." >&2
    exit 1
fi

# ── The cohort, and the array index that selects from it ─────────────────────
# The 19 eligible 300-epoch conditions in the order
# saga.frozen.runner.eligible_run_ids defines: BASELINES FIRST (indices 0-7),
# then registers and saga. The list is spelled out so the array indices are
# auditable from the job file alone, and
# tests/test_I1_spatial.py::test_the_job_file_lists_the_cohort_in_array_order
# pins it against the function; the runtime check below refuses to run if the
# manifest and this list have drifted apart.
RUN_IDS=(
    "legacy_e2_vit_base_nomixdir_baseline"        # 0   vit_base|mixup  baseline
    "e2r_vitb_mixup_baseline_s1"                  # 1   vit_base|mixup  baseline
    "legacy_e2_vit_small_mixupdir_baseline"       # 2   vit_small|mixup baseline
    "legacy_e2_vit_small_nomixdir_baseline"       # 3   vit_small|mixup baseline
    "e2r_vits_mixup_baseline_s1"                  # 4   vit_small|mixup baseline
    "e2r_vits_mixup_baseline_s2"                  # 5   vit_small|mixup baseline
    "e2r_vits_nomix_baseline_s1"                  # 6   vit_small|nomix baseline
    "e2r_vits_nomix_baseline_s2"                  # 7   vit_small|nomix baseline
    "legacy_e2_vit_base_nomixdir_registers"       # 8   vit_base|mixup  registers
    "legacy_e2_vit_base_nomixdir_saga"            # 9   vit_base|mixup  saga
    "e2r_vitb_mixup_saga_s1"                      # 10  vit_base|mixup  saga
    "legacy_e2_vit_small_mixupdir_registers"      # 11  vit_small|mixup registers
    "legacy_e2_vit_small_nomixdir_registers"      # 12  vit_small|mixup registers
    "legacy_e2_vit_small_mixupdir_saga"           # 13  vit_small|mixup saga
    "legacy_e2_vit_small_nomixdir_saga"           # 14  vit_small|mixup saga
    "e2r_vits_mixup_saga_s1"                      # 15  vit_small|mixup saga
    "e2r_vits_mixup_saga_s2"                      # 16  vit_small|mixup saga
    "e2r_vits_nomix_saga_s1"                      # 17  vit_small|nomix saga
    "e2r_vits_nomix_saga_s2"                      # 18  vit_small|nomix saga
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#RUN_IDS[@]}" ]; then
    echo "ERROR: array index $IDX is outside 0..$(( ${#RUN_IDS[@]} - 1 ))" >&2
    exit 1
fi
RUN_ID="${RUN_IDS[$IDX]}"

if ! $PY -c "
import sys
sys.path.insert(0, '.')
from saga.frozen.runner import eligible_run_ids
want = eligible_run_ids('$MANIFEST')
# [*] and not [@]: inside double quotes [@] would expand to SEPARATE words
# and split this single -c argument into twenty of them
have = '''${RUN_IDS[*]}'''.split()
if want != have:
    print('MANIFEST COHORT DRIFT', file=sys.stderr)
    print('  job file:', have, file=sys.stderr)
    print('  manifest:', want, file=sys.stderr)
    sys.exit(1)
"; then
    echo "ERROR: scripts/jobs/frozen_I1.sbatch and the manifest disagree" >&2
    echo "       about the eligible cohort. Rebuild the manifest or fix the" >&2
    echo "       job file — an array index must mean the same run on both" >&2
    echo "       sides." >&2
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the wall
# clock (TASK-09's lesson: a plain call at the end only runs on a normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

# ── The per-stage calibrated thresholds, once per split ──────────────────────
# tools/frozen_I2_thresholds.py is REUSED UNCHANGED. It reads the stages
# (`calibration_stages`), the output path template and the canon path from
# whichever conditions YAML it is handed, so pointing it at I1_spatial.yaml
# calibrates s11_out, in_b07, in_b08 and hist into
# results/frozen/I1_spatial/<split>/thresholds_cal.json. Duplicating it as a
# frozen_I1_thresholds.py would give the canon recipe a second implementation,
# which is exactly what saga/frozen/diag.py exists to prevent.
#
# `--if-missing` makes this a no-op after the first array task has written
# every cell, and merge_thresholds_cal refuses to overwrite a calibration
# already on disk, so concurrent array tasks cannot corrupt it. The three
# designated baselines are array indices 1, 4 and 6.
echo ""
echo "=============================================================="
echo "  thresholds_cal (per stage, from each cell's designated baseline)"
echo "=============================================================="
$PY tools/frozen_I2_thresholds.py \
    --conditions "$CONDITIONS" \
    --split "$SPLIT" \
    --data "$PROBE_IN_DIR" \
    --manifest "$MANIFEST" \
    --device cuda --seed 0 --if-missing
THR_RC=$?
if [ $THR_RC -ne 0 ]; then
    echo "ERROR: threshold calibration failed (exit $THR_RC) — refusing to" >&2
    echo "       build a fixed_cal map against a threshold that does not" >&2
    echo "       exist." >&2
    exit $THR_RC
fi

# ── The extraction for this array task's run ─────────────────────────────────
echo ""
echo "=============================================================="
echo "  frozen_I1_norms: $RUN_ID   (array index $IDX)"
echo "=============================================================="
$PY tools/frozen_I1_norms.py \
    --run-id "$RUN_ID" \
    --conditions "$CONDITIONS" \
    --split "$SPLIT" \
    --data "$PROBE_IN_DIR" \
    --manifest "$MANIFEST" \
    --out-root "$OUT_ROOT" \
    --batch-size 64 \
    --device cuda --seed 0 --skip-if-done
RC=$?
echo "  -> exit $RC"

echo ""
echo "=============================================================="
if [ $RC -eq 0 ]; then
    echo "  DONE: $RUN_ID"
    echo ""
    echo "  After BOTH arrays (discovery AND evaluation) finish, commit and"
    echo "  push these, then stop:"
    echo "    results/frozen/I1_spatial/*/thresholds_cal.json"
    echo "    results/frozen/I1_spatial/*/*/maps_*.npz"
    echo "    results/frozen/I1_spatial/*/*/maps.done.json"
    echo "    results/frozen/I1_spatial/*/*/run_meta.json"
    echo ""
    echo "  Do NOT commit results/frozen/I1_spatial/*/*/norms_*.npz — they are"
    echo "  git-ignored on purpose (~700 MB per split) and stay on the"
    echo "  cluster. run_meta.json records their sha256 and byte size."
else
    echo "  FAILED: $RUN_ID (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC
