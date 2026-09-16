#!/bin/bash
#SBATCH --job-name=saga_frozen_I3
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I3_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I3_%A_%a.err

# TASK B / I3 B5 — the gate-edit sweep: 61 conditions per checkpoint over the
# 8 eligible SAGA checkpoints, one array task per run_id.
#
#     sbatch scripts/jobs/frozen_I3.sbatch
#
# THE SPLIT IS NOT AN ARGUMENT HERE, and that is the difference from
# frozen_I1/I2.sbatch. TASK B §2 fixes it: everything Track B reports comes
# from results/frozen/splits/evaluation.json, and NOTHING in Track B runs on
# discovery or calibration. I1 needed both (discovery for D5's masks,
# evaluation for its tables) so its split had to be chosen per submission; I3
# has exactly one, so it is written down here rather than left to a command
# line that could pass the wrong one at 2 a.m.
#
# ── WHAT RUNS, AND WHICH INDEX IS WHICH ──────────────────────────────────────
# The 8 SAGA checkpoints in the order saga.frozen.runner.eligible_cohort
# defines, filtered to variant == saga. TASK B §9's own comment says "array
# 0-3 = the 4 ViT-S/mixup SAGA checkpoints, array 4-7 = exploratory"; that is
# NOT this repository's cohort order, which sorts by (arch, recipe_actual,
# variant, provenance_tag, run_id) and therefore puts ViT-B first. The
# COMMAND is unchanged (--array=0-7 runs all 8); only the index-to-scope map
# below is, and it is the one the manifest actually produces.
#
#   index  run_id                              cell              scope
#   -----  ----------------------------------  ----------------  --------------
#     0    legacy_e2_vit_base_nomixdir_saga    vit_base|mixup    exploratory
#     1    e2r_vitb_mixup_saga_s1              vit_base|mixup    exploratory
#     2    legacy_e2_vit_small_mixupdir_saga   vit_small|mixup   legacy (reported)
#     3    legacy_e2_vit_small_nomixdir_saga   vit_small|mixup   legacy (reported)
#     4    e2r_vits_mixup_saga_s1              vit_small|mixup   FRESH — decides
#     5    e2r_vits_mixup_saga_s2              vit_small|mixup   FRESH — decides
#     6    e2r_vits_nomix_saga_s1              vit_small|nomix   exploratory
#     7    e2r_vits_nomix_saga_s2              vit_small|nomix   exploratory
#
# To run only the pair that decides: --array=4,5. To run the primary cell:
# --array=2-5. The scope label is recomputed in the tables from the manifest,
# never from an array index, so a partial array never mislabels a row.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1 / cpus-per-task=4, the strings of the repo's own a100
# array jobs (scripts/jobs/frozen_I1.sbatch, frozen_I2.sbatch). a40 also works
# and is the cheaper queue for an eval-only job — switch on the COMMAND LINE
# rather than copying this file (docs/HPC_WORKFLOW.md "Partitions"):
#     sbatch --partition=a40 --gres=gpu:a40:1 scripts/jobs/frozen_I3.sbatch
# Keep --exclude=a0801 on the command line, where the repo's other a100
# launchers put it.
#
# --time=04:00:00, the value frozen_I1 and frozen_I2 use. 61 forward passes
# over 10,000 images is more than I2's 5, but every pass is a plain ViT-S
# inference in fp32 with no batched SVD and no norm dump: TASK B §9 estimates
# well under two hours per ViT-S checkpoint on one A100. ViT-B (indices 0-1)
# is the slow pair. A task that hits the wall leaves NOTHING extra — the
# records file is written to a temp file and renamed, the completion marker
# last — so it is simply resubmitted.
#
# ── RESUBMISSION ─────────────────────────────────────────────────────────────
# Idempotent by construction: `--skip-if-done` exits 0 for a run whose
# completion marker already names this exact checkpoint, and `append_rows`
# skips rows already present on (condition, image). Resubmit the whole array;
# it adds what is missing and rewrites nothing.
#
# ── WHAT THIS JOB DOES NOT DO ────────────────────────────────────────────────
# It does NOT calibrate thresholds. configs/frozen/I3_gate_edits.yaml points
# `thresholds_cal` at TASK I2's committed evaluation file and declares no
# `calibration_stages`, so tau_cal[s11_out] is CONSUMED. Running
# tools/frozen_I2_thresholds.py here would write a second calibration with
# the same key into I2's directory, which is the exact drift
# saga/frozen/diag.py exists to prevent.

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

SPLIT=results/frozen/splits/evaluation.json
CONDITIONS=configs/frozen/I3_gate_edits.yaml
PERMUTATIONS=configs/frozen/permutations_14x14.json
MANIFEST=results/frozen/I0_manifest/manifest.json
OUT_ROOT=results/frozen

# ── Interpreter preflight: FATAL, and BEFORE any staging ─────────────────────
# Never `which python`: if the module load fails, PATH still resolves to
# /usr/bin/python, which has no torch. An unchecked import cost TASK-10 a job
# that staged 50 000 images and then died in Python.
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
echo "split:  $SPLIT   (fixed; TASK B runs on evaluation and nothing else)"
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

# ── The cohort, and the array index that selects from it ─────────────────────
# Spelled out so the array indices are auditable from the job file alone;
# tests/test_I3_gate_edits.py::test_the_job_file_lists_the_saga_cohort_in_array_order
# pins it against the function, and the runtime check below refuses to run if
# the manifest and this list have drifted apart.
RUN_IDS=(
    "legacy_e2_vit_base_nomixdir_saga"            # 0  vit_base|mixup   exploratory
    "e2r_vitb_mixup_saga_s1"                      # 1  vit_base|mixup   exploratory
    "legacy_e2_vit_small_mixupdir_saga"           # 2  vit_small|mixup  legacy
    "legacy_e2_vit_small_nomixdir_saga"           # 3  vit_small|mixup  legacy
    "e2r_vits_mixup_saga_s1"                      # 4  vit_small|mixup  FRESH
    "e2r_vits_mixup_saga_s2"                      # 5  vit_small|mixup  FRESH
    "e2r_vits_nomix_saga_s1"                      # 6  vit_small|nomix  exploratory
    "e2r_vits_nomix_saga_s2"                      # 7  vit_small|nomix  exploratory
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
from saga.frozen.runner import eligible_cohort
want = [r['run_id'] for r in eligible_cohort('$MANIFEST')
        if r['variant'] == 'saga']
# [*] and not [@]: inside double quotes [@] would expand to SEPARATE words
# and split this single -c argument into eight of them
have = '''${RUN_IDS[*]}'''.split()
if want != have:
    print('MANIFEST COHORT DRIFT', file=sys.stderr)
    print('  job file:', have, file=sys.stderr)
    print('  manifest:', want, file=sys.stderr)
    sys.exit(1)
"; then
    echo "ERROR: scripts/jobs/frozen_I3.sbatch and the manifest disagree" >&2
    echo "       about the eligible SAGA cohort. Rebuild the manifest or fix" >&2
    echo "       the job file — an array index must mean the same run on both" >&2
    echo "       sides." >&2
    exit 1
fi

# ── The permutations file, before anything is staged ─────────────────────────
# D3 closed by naming this file AND its sha256 in docs/LOCKED_ANALYSIS.md. The
# conditions file resolves every `perm: {source, index}` out of it and writes
# the digest onto every record row; checking it here means a job that would
# produce rows stamped with the wrong digest never starts.
if ! $PY -c "
import hashlib, pathlib, re, sys
raw = pathlib.Path('$PERMUTATIONS').read_bytes()
digest = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
locked = pathlib.Path('docs/LOCKED_ANALYSIS.md').read_text(encoding='utf-8')
print('permutations sha256:', digest)
if digest not in locked:
    print('LOCKED_ANALYSIS.md does not record this digest for', '$PERMUTATIONS',
          file=sys.stderr)
    sys.exit(1)
"; then
    echo "ERROR: $PERMUTATIONS does not match the digest D3 closed in" >&2
    echo "       docs/LOCKED_ANALYSIS.md. The ten position and ten" >&2
    echo "       within-ring permutations are FIXED; a file that changed is" >&2
    echo "       a different experiment." >&2
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the wall
# clock (TASK-09's lesson: a plain call at the end only runs on a normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

# ── The sweep for this array task's run ──────────────────────────────────────
echo ""
echo "=============================================================="
echo "  frozen_I3: $RUN_ID   (array index $IDX)"
echo "  61 conditions x $(basename $SPLIT)"
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
    echo "  After the whole array finishes, commit and push these, then stop:"
    echo "    results/frozen/I3_gate_edits/evaluation/*/records.done.json"
    echo "    results/frozen/I3_gate_edits/evaluation/*/diag.done.json"
    echo "    results/frozen/I3_gate_edits/evaluation/*/maps.npz"
    echo "    results/frozen/I3_gate_edits/evaluation/*/run_meta.json"
    echo ""
    echo "  Do NOT commit records.parquet / diag.parquet — 61 x 10,000 rows"
    echo "  per run stays on the cluster (the I2 policy). run_meta.json"
    echo "  records their sha256 and byte size, and the per-image primary"
    echo "  pack figures_data/frozen/I3_primary.npz (built in Phase C) is"
    echo "  what every reported number regenerates from."
else
    echo "  FAILED: $RUN_ID (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC
