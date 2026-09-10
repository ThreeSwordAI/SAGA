#!/bin/bash
#SBATCH --job-name=saga_ttr_validate
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_validate_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_validate_%j.err

# TASK-10 PHASE B, STEP 1 — the A3 validation gate for test-time registers.
#
# This job answers ONE question: does TTR do anything to a supervised ViT-S?
# It sweeps --n-neurons on the ViT-S/mixup baseline and prints a PASS/FAIL
# line. **Report that line back before submitting ttr_matrix.sbatch.**
# On FAIL for every n the full matrix must NOT be run — a validated negative
# is the deliverable (TASK-10 A3), and the exit code says which happened:
#   0 = PASS, 3 = FAIL (honestly measured), anything else = the tool broke.
#
# PARTITION PROVENANCE: --partition=a40 / --gres=gpu:a40:1 / --cpus-per-task=4
# are copied from scripts/jobs/probe_attention.sbatch (TASK-11), which took
# them from the human's own working a40 job on this cluster (2026-09-09); the
# same strings are recorded in docs/HPC_WORKFLOW.md "Partitions".
# --time=02:00:00 is dominated by ImageNet val staging (~10 min per
# How to Run.md §5): the compute itself is 7 passes over 5 000 images plus a
# 500-image scan on a ViT-S, i.e. minutes. The a40 wall limit is not
# documented — lower this if the partition refuses it.

set -u

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
# Reuses TASK-11's committed val-only stager rather than adding a third one:
# stage_probe_imagenet_val extracts ImageNet val to node-local scratch and
# enforces 1000 classes / 50000 images, which is exactly what eval and the
# diag split need. stage_imagenet.sh would also pull train/ — 10x the copy
# for a job that never touches it.
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

PY=$(which python)
echo "python: $PY"
$PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"

# Release node-local scratch on EVERY exit path, including a kill at the
# wall clock (TASK-09's lesson: a plain call at the end only runs on a
# normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

CKPT=$CODE_ROOT/results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth
if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found — the A3 gate needs the seeded ViT-S/mixup"
    echo "       baseline, which is also the cell whose canon tau it uses."
    exit 1
fi

echo "======================================================"
echo "  TASK-10 A3 — TTR validation gate"
echo "  base cell: vit_small|mixup   ckpt: $CKPT"
echo "======================================================"

$PY tools/ttr_validate.py \
    --ckpt "$CKPT" \
    --arch vit_small --recipe mixup --variant baseline \
    --n-neurons 4,8,16,32,64 \
    --data $PROBE_IN_DIR \
    --out-root results/ttr \
    --device cuda --seed 0
RC=$?

echo "======================================================"
echo "  exit code $RC  (0 = PASS, 3 = FAIL, other = tool error)"
echo "  Commit and push these two, then report the PASS/FAIL line:"
echo "    results/ttr/e2r_vits_mixup_baseline_s1/neurons.json"
echo "    results/ttr/e2r_vits_mixup_baseline_s1/sweep.csv"
echo "    results/ttr/e2r_vits_mixup_baseline_s1/validate.json"
echo "  Do NOT submit ttr_matrix.sbatch unless this said PASS."
echo "======================================================"
cat results/ttr/e2r_vits_mixup_baseline_s1/sweep.csv 2>/dev/null

exit $RC
