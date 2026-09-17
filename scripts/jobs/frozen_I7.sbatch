#!/bin/bash
#SBATCH --job-name=saga_frozen_I7
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I7_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I7_%A_%a.err

# TASK C / I7 — incoming attention at the exceedance positions, over the
# ViT-S/mixup cell (4 baseline, 4 SAGA, 2 registers), one array task per run.
#
#     sbatch scripts/jobs/frozen_I7.sbatch
#
# THE SPLIT IS FIXED TO sub1k IN THIS FILE. §2 and D10 declare I7 on
# `results/frozen/splits/sub1k.json` (1,000 evaluation images); there is no
# second split this work package may legitimately run on, so a knob would
# only create a way to run the wrong one.
#
# Everything else — the two blocks, the alignment, the exceedance bases — is
# in configs/frozen/I7_attention.yaml, committed before this job runs.
# saga/frozen/runner.py::load_conditions REFUSES a block that is not in that
# file and not in D10.
#
# NO ATTENTION MAPS ARE WRITTEN. The explicit [B, H, T, T] map exists inside
# one forward and is reduced to per-image, per-key summaries before the next
# batch. What lands on disk is `incoming_mass.npz`, ~5 MB per run, and it IS
# committed (§8).
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1 / cpus-per-task=4, the strings of scripts/jobs/frozen_I2.sbatch
# and frozen_I5a.sbatch. a40 also works and is the cheaper queue for an
# eval-only job — switch on the COMMAND LINE rather than copying this file
# (docs/HPC_WORKFLOW.md "Partitions"):
#     sbatch --partition=a40 --gres=gpu:a40:1 scripts/jobs/frozen_I7.sbatch
# Task file §10 carries `--exclude=a0801`; keep it on the command line, where
# the repo's other a100 launchers put it.
#
# --time=02:00:00: staging ImageNet val is ~10 min (How to Run.md §5), then
# ONE forward over 1,000 images per run. Materialising the explicit attention
# map for two blocks is the cost, which is why --batch-size is 16 rather than
# 64; it is still minutes, not hours. Half the I5a wall because the split is
# half the size and there is one pass instead of eight.
#
# ── RESUBMISSION ─────────────────────────────────────────────────────────────
# `--skip-if-done` exits 0 for a run whose completion marker already names
# this exact checkpoint. The npz is written tmp+rename and its marker LAST,
# so a killed task leaves either the previous complete file or nothing extra.
# Resubmit the whole array; it adds what is missing and rewrites nothing.

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

CONDITIONS=configs/frozen/I7_attention.yaml
MANIFEST=results/frozen/I0_manifest/manifest.json
SPLIT=results/frozen/splits/sub1k.json
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
if ! $PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm — refusing to stage 50 000" >&2
    echo "       images for a job that cannot run." >&2
    exit 1
fi

# ── The I7 cell, and the array index that selects from it ────────────────────
# The ViT-S/mixup cell (§2): 4 baseline, 4 SAGA, 2 registers = 10 runs, in the
# order saga.frozen.runner.eligible_cohort defines RESTRICTED to that cell —
# baselines first, then registers, then saga. Spelled out so the array indices
# are auditable from the job file alone;
# tests/test_I7_attention.py::test_the_job_file_lists_the_cell_in_array_order
# pins it against the function, and the runtime check below refuses to run if
# the manifest and this list have drifted apart.
RUN_IDS=(
    "legacy_e2_vit_small_mixupdir_baseline"       # 0  baseline  (legacy)
    "legacy_e2_vit_small_nomixdir_baseline"       # 1  baseline  (legacy)
    "e2r_vits_mixup_baseline_s1"                  # 2  baseline  FRESH s1
    "e2r_vits_mixup_baseline_s2"                  # 3  baseline  FRESH s2
    "legacy_e2_vit_small_mixupdir_registers"      # 4  registers (legacy)
    "legacy_e2_vit_small_nomixdir_registers"      # 5  registers (legacy)
    "legacy_e2_vit_small_mixupdir_saga"           # 6  saga      (legacy)
    "legacy_e2_vit_small_nomixdir_saga"           # 7  saga      (legacy)
    "e2r_vits_mixup_saga_s1"                      # 8  saga      s1
    "e2r_vits_mixup_saga_s2"                      # 9  saga      s2
)
# Indices 2 and 3 are the two FRESH baselines D10's wording rule is decided
# on. Submitting `--array=2,3` first gives the verdict's evidence soonest.

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#RUN_IDS[@]}" ]; then
    echo "ERROR: array index $IDX is outside 0..$(( ${#RUN_IDS[@]} - 1 ))" >&2
    exit 1
fi
RUN_ID="${RUN_IDS[$IDX]}"

if ! $PY -c "
import sys
sys.path.insert(0, '.')
from saga.frozen.runner import eligible_cohort
want = [r['run_id'] for r in eligible_cohort('$MANIFEST')
        if r['arch'] == 'vit_small' and r['recipe_actual'] == 'mixup']
# [*] and not [@]: inside double quotes [@] would expand to SEPARATE words
# and split this single -c argument into ten of them
have = '''${RUN_IDS[*]}'''.split()
if want != have:
    print('MANIFEST CELL DRIFT', file=sys.stderr)
    print('  job file:', have, file=sys.stderr)
    print('  manifest:', want, file=sys.stderr)
    sys.exit(1)
"; then
    echo "ERROR: scripts/jobs/frozen_I7.sbatch and the manifest disagree" >&2
    echo "       about the ViT-S/mixup cell. Rebuild the manifest or fix the" >&2
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
echo "  frozen_I7_attn: $RUN_ID   (array index $IDX)"
echo "=============================================================="
$PY tools/frozen_I7_attn.py \
    --run-id "$RUN_ID" \
    --conditions "$CONDITIONS" \
    --split "$SPLIT" \
    --data "$PROBE_IN_DIR" \
    --manifest "$MANIFEST" \
    --out-root "$OUT_ROOT" \
    --batch-size 16 \
    --device cuda --seed 0 --skip-if-done
RC=$?
echo "  -> exit $RC"

echo ""
echo "=============================================================="
if [ $RC -eq 0 ]; then
    echo "  DONE: $RUN_ID"
    echo ""
    echo "  After the WHOLE array finishes, commit and push these, then stop:"
    echo "    results/frozen/I7_attention/sub1k/*/incoming_mass.npz"
    echo "    results/frozen/I7_attention/sub1k/*/incoming_mass.done.json"
    echo "    results/frozen/I7_attention/sub1k/*/run_meta.json"
    echo ""
    echo "  Each run_meta.json records state_restored and the fused_attn"
    echo "  flags before the capture. If any run reports state_restored"
    echo "  false, say so in the report rather than pushing quietly."
else
    echo "  FAILED: $RUN_ID (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC
