#!/bin/bash
#SBATCH --job-name=saga_e4
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e4/logs/e4_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e4/logs/e4_%A_%a.err

# ── E4 Segmentation — 3 variants, 80 epochs each ─────────────────────────────
#
# Submit all:    sbatch --array=0-2 segmentation/scripts/e4_train_alex.sh
# Submit one:    sbatch --array=2   segmentation/scripts/e4_train_alex.sh
# Auto-resume:   resubmit same command after 24h timeout
#
# Variant mapping:
#   0 = ViT-B_baseline_seg
#   1 = ViT-B_registers_seg
#   2 = ViT-B_SAGA_seg

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/segmentation/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/segmentation/scripts/stage_ade20k.sh

cd $CODE_ROOT

MASTER_PORT=$((29920 + SLURM_ARRAY_TASK_ID))

VARIANT_NAMES=(
    "ViT-B_baseline_seg"
    "ViT-B_registers_seg"
    "ViT-B_SAGA_seg"
)

VARIANT_NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
LAST_CKPT=/home/vault/iwi5/iwi5359h/SAGA/e4/checkpoints/${VARIANT_NAME}/last.pth
RESUME_ARG=""

echo "======================================================"
echo "  SAGA E4 — $VARIANT_NAME"
echo "  Node: $SLURMD_NODENAME  |  Port: $MASTER_PORT"
echo "  $(date)"
echo "======================================================"

if [ -f "$LAST_CKPT" ]; then
    echo "  Resuming from: $LAST_CKPT"
    RESUME_ARG="--resume $LAST_CKPT"
else
    echo "  Starting from scratch"
fi

stage_ade20k

echo "======================================================"
echo "  Starting training  $(date)"
echo "======================================================"

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    segmentation/tools/train.py \
        --config    segmentation/configs/variants.yaml \
        --id        $SLURM_ARRAY_TASK_ID \
        --paths     segmentation/configs/paths.yaml \
        --data_root $DATA_ROOT \
        $RESUME_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: $VARIANT_NAME failed."
    cleanup_ade20k
    exit 1
fi

echo ""
echo "======================================================"
echo "  $VARIANT_NAME DONE — $(date)"
echo "======================================================"

cleanup_ade20k