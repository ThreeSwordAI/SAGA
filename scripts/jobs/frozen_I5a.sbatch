#!/bin/bash
#SBATCH --job-name=saga_frozen_I5a
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-18
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I5a_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I5a_%A_%a.err

# TASK C / I5a — the correspondence readout over the 19 eligible 300-epoch
# conditions, one array task per run_id.
#
#     sbatch scripts/jobs/frozen_I5a.sbatch
#
# THE SPLIT IS FIXED TO sub2k IN THIS FILE, not taken as an argument. I5a is
# declared on `results/frozen/splits/sub2k.json` (2,000 evaluation images) in
# task file §2 and in D9; unlike I2 — whose whole design was calibration
# first, evaluation only after D1 was signed — there is no second split this
# work package may legitimately run on, so making it a knob would only create
# a way to run the wrong one.
#
# Everything else the sweep needs — the transforms, the stages, the
# descriptors, the conditions — is in configs/frozen/I5_readout.yaml,
# committed before this job runs. saga/frozen/runner.py::load_conditions
# REFUSES a name that is not in that file and not in D9.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1 / cpus-per-task=4, copied from scripts/jobs/frozen_I2.sbatch
# (the repo's other frozen array job). a40 also works and is the cheaper queue
# for an eval-only job — switch on the COMMAND LINE rather than copying this
# file (docs/HPC_WORKFLOW.md "Partitions"):
#     sbatch --partition=a40 --gres=gpu:a40:1 scripts/jobs/frozen_I5a.sbatch
# Task file §10 carries `--exclude=a0801`; keep it on the command line, where
# the repo's other a100 launchers put it.
#
# --time=04:00:00: staging ImageNet val is ~10 min (How to Run.md §5). The
# sweep itself is 2 forwards per (image, transform) — 4 transforms x 2,000
# images x 2 sides = 16,000 forwards for a baseline or register run, doubled
# for a SAGA run (native + term_1.00). There is no SVD anywhere in I5, so
# this is markedly cheaper than the I2 sweep at the same walltime. A task
# that hits the wall leaves NOTHING extra — the writer appends complete rows
# and writes the completion marker last — so it is simply resubmitted.
#
# ── RESUBMISSION ─────────────────────────────────────────────────────────────
# Idempotent by construction: `--skip-if-done` exits 0 for a run whose
# completion marker already names this exact checkpoint, and `append_rows`
# skips rows already present on
# (condition, transform, stage, descriptor, image). Resubmit the whole array;
# it adds what is missing and rewrites nothing.

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

CONDITIONS=configs/frozen/I5_readout.yaml
MANIFEST=results/frozen/I0_manifest/manifest.json
SPLIT=results/frozen/splits/sub2k.json
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
    echo "       SAGA_PY=/path/to/python sbatch $0" >&2
    exit 1
fi
if ! $PY -c "import torch, timm, torchvision; print('torch', torch.__version__, 'timm', timm.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm/torchvision — refusing to" >&2
    echo "       stage 50 000 images for a job that cannot run. I5 needs" >&2
    echo "       torchvision for the T2/T3 two-crop pipeline." >&2
    exit 1
fi

# ── pyarrow preflight: a CSV sweep is a git accident waiting to happen ───────
# saga/frozen/records.py writes parquet when pyarrow is importable and CSV
# otherwise, with the SAME schema either way. corr_records is one row per
# (condition, transform, stage, descriptor, image) — 36,000 rows for a
# baseline run and 72,000 for a SAGA run — and every row carries the §2.7
# provenance columns (two 64-char shas among them). As parquet that
# dictionary-encodes to a few MB, which is what task file §8 budgets for a
# COMMITTED file; as CSV it is well over a hundred MB of text. Refuse rather
# than produce it.
if ! $PY -c "import pyarrow, pyarrow.parquet"; then
    echo "ERROR: pyarrow is not installed in $PY, so the sweep would write" >&2
    echo "       CSV records — far too large to commit. Install it once" >&2
    echo "       into the env and resubmit:" >&2
    echo "         $PY -m pip install pyarrow" >&2
    echo "       (it is declared in requirements.txt)." >&2
    exit 1
fi

# ── The cohort, and the array index that selects from it ─────────────────────
# The 19 eligible 300-epoch conditions in the order
# saga.frozen.runner.eligible_run_ids defines: BASELINES FIRST (indices 0-7),
# then registers and saga. The list is spelled out so the array indices are
# auditable from the job file alone, and
# tests/test_I5_readout.py::test_the_job_file_lists_the_cohort_in_array_order
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
    echo "ERROR: scripts/jobs/frozen_I5a.sbatch and the manifest disagree" >&2
    echo "       about the eligible cohort. Rebuild the manifest or fix the" >&2
    echo "       job file — an array index must mean the same run on both" >&2
    echo "       sides." >&2
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the wall
# clock (TASK-09's lesson: a plain call at the end only runs on a normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

echo ""
echo "=============================================================="
echo "  frozen_I5_corr: $RUN_ID   (array index $IDX)"
echo "=============================================================="
$PY tools/frozen_I5_corr.py \
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
    echo "    results/frozen/I5_readout/sub2k/*/corr_records.parquet"
    echo "    results/frozen/I5_readout/sub2k/*/corr_records.done.json"
    echo "    results/frozen/I5_readout/sub2k/*/run_meta.json"
    echo ""
    echo "  Every run_meta.json carries t0_matches_plain_forward — the T0"
    echo "  identity control. If any stage there is non-zero, say so in the"
    echo "  report rather than pushing quietly."
else
    echo "  FAILED: $RUN_ID (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC
