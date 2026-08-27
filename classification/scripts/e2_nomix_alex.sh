#!/bin/bash
#SBATCH --job-name=saga_e2_nomix
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e2/logs/e2_nomix_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e2/logs/e2_nomix_%A_%a.err

# ── ViT-S without MixUp/CutMix — 3 variants ───────────────────────────────
# Purpose: isolate the effect of label-mixing augmentation on SAGA gate.
# Hypothesis: without MixUp/CutMix, SAGA > baseline because φ_h receives
#             clean positional gradient signal from unblended images.
#
# Submit all 3:  sbatch --array=0-2 scripts/e2_nomix_alex.sh
# Auto-resume:   resubmit same command if job hits 24h limit
#
# Variant mapping:
#   0 = ViT-S_baseline_nomix
#   1 = ViT-S_registers_nomix
#   2 = ViT-S_SAGA_nomix

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/classification/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh

cd $CODE_ROOT

MASTER_PORT=$((29800 + SLURM_ARRAY_TASK_ID))

VARIANT_NAMES=(
    "ViT-S_baseline_nomix"
    "ViT-S_registers_nomix"
    "ViT-S_SAGA_nomix"
    "ViT-B_baseline_nomix"
    "ViT-B_registers_nomix"
    "ViT-B_SAGA_nomix"
)

VARIANT_NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
LAST_CKPT=$OUT_ROOT/e2/checkpoints/${VARIANT_NAME}/last.pth
RESUME_ARG=""

echo "======================================================"
echo "  SAGA E2 nomix — $VARIANT_NAME"
echo "  Node: $SLURMD_NODENAME  |  Port: $MASTER_PORT"
echo "  $(date)"
echo "======================================================"

if [ -f "$LAST_CKPT" ]; then
    echo "  Resuming from: $LAST_CKPT"
    RESUME_ARG="--resume $LAST_CKPT"
else
    echo "  Starting from scratch"
fi

stage_imagenet

echo "======================================================"
echo "  Starting training  $(date)"
echo "======================================================"

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    classification/tools/train.py \
        --config    classification/configs/variants_nomix.yaml \
        --id        $SLURM_ARRAY_TASK_ID \
        --paths     classification/configs/paths.yaml \
        --data_root $STAGE_DIR \
        $RESUME_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: $VARIANT_NAME failed."
    cleanup_imagenet
    exit 1
fi

echo ""
echo "======================================================"
echo "  $VARIANT_NAME DONE — $(date)"
echo "======================================================"

cleanup_imagenet