#!/bin/bash
#SBATCH --job-name=saga_e2
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e2/logs/e2_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e2/logs/e2_%A_%a.err

# ── E2 training — 9 variants (3 scales × 3 configs) ───────────────────────────
#
# Submit all 9:    sbatch --array=0-8  scripts/e2_train_alex.sh
# Submit S only:   sbatch --array=0-2  scripts/e2_train_alex.sh
# Submit B only:   sbatch --array=3-5  scripts/e2_train_alex.sh
# Submit L only:   sbatch --array=6-8  scripts/e2_train_alex.sh
# Submit one:      sbatch --array=5    scripts/e2_train_alex.sh
#
# Each run is 300 epochs. With 4×A100:
#   ViT-S: ~30 hrs  (2 submissions of 24h with auto-resume)
#   ViT-B: ~55 hrs  (3 submissions)
#   ViT-L: ~90 hrs  (4 submissions)
#
# Auto-resume: if last.pth exists for this variant, training resumes
# from that epoch automatically. Just resubmit the same command.
#
# Variant mapping:
#   0 = ViT-S_baseline   3 = ViT-B_baseline   6 = ViT-L_baseline
#   1 = ViT-S_registers  4 = ViT-B_registers  7 = ViT-L_registers
#   2 = ViT-S_SAGA       5 = ViT-B_SAGA       8 = ViT-L_SAGA

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/classification/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh

cd $CODE_ROOT

# ── Unique port per task (avoids NCCL conflicts on shared nodes) ──────────────
MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))

# ── Variant names for checkpoint detection ────────────────────────────────────
VARIANT_NAMES=(
    "ViT-S_baseline"
    "ViT-S_registers"
    "ViT-S_SAGA"
    "ViT-B_baseline"
    "ViT-B_registers"
    "ViT-B_SAGA"
    "ViT-L_baseline"
    "ViT-L_registers"
    "ViT-L_SAGA"
)

VARIANT_NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}
LAST_CKPT=$OUT_ROOT/e2/checkpoints/${VARIANT_NAME}/last.pth
RESUME_ARG=""

echo "======================================================"
echo "  SAGA E2 — Variant $SLURM_ARRAY_TASK_ID ($VARIANT_NAME)"
echo "  Array job: $SLURM_ARRAY_JOB_ID"
echo "  Node: $SLURMD_NODENAME  |  Port: $MASTER_PORT"
echo "  $(date)"
echo "======================================================"

if [ -f "$LAST_CKPT" ]; then
    echo "  Found checkpoint — resuming from: $LAST_CKPT"
    RESUME_ARG="--resume $LAST_CKPT"
else
    echo "  No checkpoint — starting from scratch"
fi

# ── Stage ImageNet ─────────────────────────────────────────────────────────────
stage_imagenet

# ── Train ──────────────────────────────────────────────────────────────────────
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
        --config    classification/configs/variants.yaml \
        --id        $SLURM_ARRAY_TASK_ID \
        --paths     classification/configs/paths.yaml \
        --data_root $STAGE_DIR \
        $RESUME_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Variant $SLURM_ARRAY_TASK_ID ($VARIANT_NAME) failed."
    cleanup_imagenet
    exit 1
fi

echo ""
echo "======================================================"
echo "  $VARIANT_NAME DONE — $(date)"
echo "======================================================"

cleanup_imagenet