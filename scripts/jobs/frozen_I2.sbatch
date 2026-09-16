#!/bin/bash
#SBATCH --job-name=saga_frozen_I2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-18
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I2_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I2_%A_%a.err

# TASK I2 D7 — the terminal patch-gate sweep over the 19 eligible 300-epoch
# conditions, one array task per run_id.
#
#     sbatch scripts/jobs/frozen_I2.sbatch results/frozen/splits/calibration.json
#
# THE SPLIT IS AN ARGUMENT, NOT A DEFAULT. Nothing in this file or in the code
# it calls falls back to evaluation.json: B1 runs on calibration, B2 runs on
# evaluation only after the human has SIGNED D1 (task file §10). A default
# would make "we did not look at the evaluation split first" a habit rather
# than a property of the repository.
#
# Everything else the sweep needs — the conditions, the stages, the gate
# constants, the threshold files — is in configs/frozen/I2_terminal.yaml,
# committed before this job runs. The only command-line knobs here are WHICH
# split and (via the array) WHICH run.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1 / cpus-per-task=4, the strings of task file §9 and of the
# repo's own a100 array job (ft_finegrained_array.sbatch). a40 also works and
# is the cheaper queue for an eval-only job — switch on the COMMAND LINE
# rather than copying this file (docs/HPC_WORKFLOW.md "Partitions"):
#     sbatch --partition=a40 --gres=gpu:a40:1 scripts/jobs/frozen_I2.sbatch <split>
# §9 also carries `--exclude=a0801`; keep it on the command line, where the
# repo's other a100 launchers put it.
#
# --time=04:00:00: staging ImageNet val is ~10 min (How to Run.md §5); the
# sweep itself is 3 forward passes over the split for a baseline/register run
# and 11 for a SAGA run (5 conditions x their declared stages), and the slow
# part is the batched SVD behind `eff_rank`. A task that hits the wall leaves
# NOTHING extra — the writers append complete rows and write the completion
# marker last — so it is simply resubmitted.
#
# ── RESUBMISSION ─────────────────────────────────────────────────────────────
# Idempotent by construction: `--skip-if-done` exits 0 for a run whose
# completion marker already names this exact checkpoint, `append_rows` skips
# rows already present on (condition, stage, image), and
# frozen_I2_thresholds.py --if-missing is a no-op once every cell is
# calibrated for this split. Resubmit the whole array; it adds what is
# missing and rewrites nothing.

SPLIT="${1:?usage: sbatch [--array=0-18] scripts/jobs/frozen_I2.sbatch <split.json>
       e.g. sbatch scripts/jobs/frozen_I2.sbatch results/frozen/splits/calibration.json
       B2 (evaluation.json) only AFTER the human has signed D1.}"

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

CONDITIONS=configs/frozen/I2_terminal.yaml
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

# ── pyarrow preflight: a CSV sweep is a git accident waiting to happen ───────
# saga/frozen/records.py writes parquet when pyarrow is importable and CSV
# otherwise, with the SAME schema either way. The I2 diag file is one row per
# image x condition x stage — ~22 000 rows per SAGA run — and every row
# carries the §2.7 provenance columns (two 64-char shas among them). As
# parquet that dictionary-encodes to a few MB for the whole cohort, which is
# what task file §8 budgets; as CSV it is well over a hundred MB of text that
# must then not be committed. Refuse rather than produce it.
if ! $PY -c "import pyarrow, pyarrow.parquet"; then
    echo "ERROR: pyarrow is not installed in $PY, so the sweep would write" >&2
    echo "       CSV records — over a hundred MB for the cohort, which must" >&2
    echo "       not enter git. Install it once into the env and resubmit:" >&2
    echo "         $PY -m pip install pyarrow" >&2
    echo "       (it is declared in requirements.txt)." >&2
    exit 1
fi

# ── The cohort, and the array index that selects from it ─────────────────────
# The 19 eligible 300-epoch conditions in the order
# saga.frozen.runner.eligible_run_ids defines: BASELINES FIRST (indices 0-7),
# then registers and saga. The list is spelled out so the array indices are
# auditable from the job file alone, and
# tests/test_I2_terminal.py::test_the_job_file_lists_the_cohort_in_array_order
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
    echo "ERROR: scripts/jobs/frozen_I2.sbatch and the manifest disagree" >&2
    echo "       about the eligible cohort. Rebuild the manifest or fix the" >&2
    echo "       job file — an array index must mean the same run on both" >&2
    echo "       sides." >&2
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the wall
# clock (TASK-09's lesson: a plain call at the end only runs on a normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

# ── D2: the per-stage calibrated thresholds, once per split ──────────────────
# `--if-missing` makes this a no-op after the first array task has written
# every cell, and merge_thresholds_cal refuses to overwrite a calibration
# that is already on disk, so concurrent array tasks cannot corrupt it. The
# three designated baselines are array indices 1, 4 and 6, so submitting
# `--array=0-7` first and the rest with `--dependency=afterok:` avoids the
# duplicated calibration entirely — an optimisation, never a requirement.
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
    echo "       record count_fixed_cal against a threshold that does not" >&2
    echo "       exist." >&2
    exit $THR_RC
fi

# ── D3: the sweep for this array task's run ──────────────────────────────────
echo ""
echo "=============================================================="
echo "  frozen_eval: $RUN_ID   (array index $IDX)"
echo "=============================================================="
$PY tools/frozen_eval.py \
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
    echo "  After the WHOLE array finishes, commit and push these, then stop:"
    echo "    results/frozen/I2_terminal/thresholds_cal.json"
    echo "    results/frozen/I2_terminal/*/records.parquet"
    echo "    results/frozen/I2_terminal/*/diag.parquet"
    echo "    results/frozen/I2_terminal/*/maps.npz"
    echo "    results/frozen/I2_terminal/*/records.done.json"
    echo "    results/frozen/I2_terminal/*/diag.done.json"
    echo "    results/frozen/I2_terminal/*/run_meta.json"
    echo "  Do NOT run the evaluation split until D1 is signed."
else
    echo "  FAILED: $RUN_ID (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC
