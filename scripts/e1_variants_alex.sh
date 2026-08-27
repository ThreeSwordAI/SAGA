#!/bin/bash
#SBATCH --job-name=saga_e1_alex
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-15
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_e1_alex_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_e1_alex_%A_%a.err

# ── E1 variant study on Alex A100 ─────────────────────────────────────────────
# Wave 1 (16 ablations):  sbatch --array=0-15  scripts/e1_variants_alex.sh
# Wave 2 (pos/gran/hp):   sbatch --array=16-23 scripts/e1_variants_alex.sh
#
# Auto-resume: if a job hits the 24hr limit, just resubmit the same command.
# The script detects last.pth and resumes automatically.

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh

cd $CODE_ROOT

# ── Unique port per task — avoids NCCL bind conflicts when tasks run on ───────
# the same node sequentially. Base port 29500 + task ID gives each job
# a distinct port (29500 for V00, 29501 for V01, ..., 29515 for V15).
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "======================================================"
echo "  SAGA E1 (Alex) — Variant $SLURM_ARRAY_TASK_ID / 23"
echo "  Array job: $SLURM_ARRAY_JOB_ID"
echo "  Node: $SLURMD_NODENAME  |  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Master port: $MASTER_PORT"
echo "  $(date)"
echo "======================================================"

# ── Auto-detect resume checkpoint ─────────────────────────────────────────────
VARIANT_NAMES=(
    "V00_baseline"
    "V01_A"
    "V02_B"
    "V03_C"
    "V04_D"
    "V05_AB"
    "V06_AC"
    "V07_AD"
    "V08_BC"
    "V09_BD"
    "V10_CD"
    "V11_ABC"
    "V12_ABD"
    "V13_ACD"
    "V14_BCD"
    "V15_ABCD_full"
    "V16_pos_G1"
    "V17_pos_G2"
    "V18_pos_G3"
    "V19_pos_G5"
    "V20_gran_head"
    "V21_gran_elem"
    "V22_lambda_low"
    "V23_lambda_high"
)

VARIANT_NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
LAST_CKPT=$OUT_ROOT/checkpoints/e1_variants/${VARIANT_NAME}/last.pth
RESUME_ARG=""

if [ -f "$LAST_CKPT" ]; then
    echo "  Found checkpoint: $LAST_CKPT"
    echo "  Resuming from previous run..."
    RESUME_ARG="--resume $LAST_CKPT"
else
    echo "  No checkpoint found — starting from scratch"
fi

# ── Stage data ────────────────────────────────────────────────────────────────
stage_imagenet

# ── Train ─────────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  Starting training — Variant $SLURM_ARRAY_TASK_ID ($VARIANT_NAME)"
echo "  $(date)"
echo "======================================================"

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    tools/train.py \
        --config     configs/e1_variants.yaml \
        --variant_id $SLURM_ARRAY_TASK_ID \
        --data_root  $STAGE_DIR \
        --out_dir    $OUT_ROOT/checkpoints/e1_variants \
        $RESUME_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Variant $SLURM_ARRAY_TASK_ID ($VARIANT_NAME) failed."
    cleanup_imagenet
    exit 1
fi

echo ""
echo "======================================================"
echo "  Variant $SLURM_ARRAY_TASK_ID ($VARIANT_NAME) DONE"
echo "  $(date)"
echo "======================================================"

cleanup_imagenet