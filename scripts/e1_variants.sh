#!/bin/bash
#SBATCH --job-name=saga_e1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-15
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_e1_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_e1_%A_%a.err

# ── E1 variant study — wave 1 ─────────────────────────────────────────────────
# Wave 1 (16 component ablations):  sbatch --array=0-15  scripts/e1_variants.sh
# Wave 2 (position/granularity/hp): sbatch --array=16-23 scripts/e1_variants.sh
#
# Monitor:  squeue -u iwi5359h
# Watch:    tail -f $OUT_ROOT/logs/saga_e1_ARRAYID_TASKID.log
# Results:  ls $OUT_ROOT/checkpoints/e1_variants/

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh

cd $CODE_ROOT

echo "======================================================"
echo "  SAGA E1 — Variant $SLURM_ARRAY_TASK_ID / 23"
echo "  Array job ID: $SLURM_ARRAY_JOB_ID"
echo "  Node: $SLURMD_NODENAME  |  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  $(date)"
echo "======================================================"

# ── Stage data ────────────────────────────────────────────────────────────────
stage_imagenet

# ── Train ─────────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  Starting training — Variant $SLURM_ARRAY_TASK_ID"
echo "  $(date)"
echo "======================================================"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    tools/train.py \
        --config     configs/e1_variants.yaml \
        --variant_id $SLURM_ARRAY_TASK_ID \
        --data_root  $STAGE_DIR \
        --out_dir    $OUT_ROOT/checkpoints/e1_variants

if [ $? -ne 0 ]; then
    echo "ERROR: Variant $SLURM_ARRAY_TASK_ID failed."
    cleanup_imagenet
    exit 1
fi

echo ""
echo "======================================================"
echo "  Variant $SLURM_ARRAY_TASK_ID DONE"
echo "  $(date)"
echo "======================================================"

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup_imagenet