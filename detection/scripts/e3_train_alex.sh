#!/bin/bash
#SBATCH --job-name=saga_e3
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e3/logs/e3_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e3/logs/e3_%A_%a.err

# ── E3 Detection — 3 variants, 25 epochs each ─────────────────────────────────
#
# Submit all:    sbatch --array=0-2 detection/scripts/e3_train_alex.sh
# Submit one:    sbatch --array=2   detection/scripts/e3_train_alex.sh
#
# Each run fits in one 24h job (~18-20h for 25 epochs).
# Auto-resume: resubmit same command if job times out.
#
# Variant mapping:
#   0 = ViT-B_baseline_det
#   1 = ViT-B_registers_det
#   2 = ViT-B_SAGA_det

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/detection/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/detection/scripts/stage_coco.sh

cd $CODE_ROOT

MASTER_PORT=$((29910 + SLURM_ARRAY_TASK_ID))

VARIANT_NAMES=(
    "ViT-B_baseline_det"
    "ViT-B_registers_det"
    "ViT-B_SAGA_det"
)

VARIANT_NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
LAST_CKPT=/home/vault/iwi5/iwi5359h/SAGA/e3/checkpoints/${VARIANT_NAME}/last.pth
RESUME_ARG=""

echo "======================================================"
echo "  SAGA E3 — $VARIANT_NAME"
echo "  Node: $SLURMD_NODENAME  |  Port: $MASTER_PORT"
echo "  $(date)"
echo "======================================================"

if [ -f "$LAST_CKPT" ]; then
    echo "  Resuming from: $LAST_CKPT"
    RESUME_ARG="--resume $LAST_CKPT"
else
    echo "  Starting from scratch"
fi

stage_coco

echo "======================================================"
echo "  Starting training  $(date)"
echo "======================================================"

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    detection/tools/train.py \
        --config    detection/configs/variants.yaml \
        --id        $SLURM_ARRAY_TASK_ID \
        --paths     detection/configs/paths.yaml \
        --data_root $STAGE_DIR \
        $RESUME_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: $VARIANT_NAME failed."
    cleanup_coco
    exit 1
fi

echo ""
echo "======================================================"
echo "  $VARIANT_NAME DONE — $(date)"
echo "======================================================"

cleanup_coco