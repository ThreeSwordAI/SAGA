#!/bin/bash
#SBATCH --job-name=saga_e2_sanity
#SBATCH --time=03:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-2
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e2/logs/e2_sanity_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e2/logs/e2_sanity_%A_%a.err

# ── E2 Sanity check — 3 epochs each for ViT-S variants (ids 0, 1, 2) ──────────
# Verifies all three configurations (baseline, registers, SAGA) run correctly
# on 4×A100 with the full data pipeline before committing to 300 epochs.
#
# Submit: sbatch --array=0-2 classification/scripts/e2_sanity_alex.sh
# Watch:  tail -f $OUT_ROOT/e2/logs/e2_sanity_ARRAYID_TASKID.log
#
# Expected time: ~30 min per variant (3 epochs × ~8 min/epoch at ViT-S scale)

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/classification/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh

cd $CODE_ROOT

MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

VARIANT_NAMES=("ViT-S_baseline" "ViT-S_registers" "ViT-S_SAGA")
VARIANT_NAME=${VARIANT_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "======================================================"
echo "  E2 Sanity — $VARIANT_NAME (3 epochs)"
echo "  Node: $SLURMD_NODENAME  |  Port: $MASTER_PORT"
echo "  $(date)"
echo "======================================================"

stage_imagenet

echo "======================================================"
echo "  Starting training"
echo "  $(date)"
echo "======================================================"

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    classification/tools/train.py \
        --config     classification/configs/variants.yaml \
        --id         $SLURM_ARRAY_TASK_ID \
        --paths      classification/configs/paths.yaml \
        --data_root  $STAGE_DIR \
        --max_epochs 3

if [ $? -ne 0 ]; then
    echo "ERROR: Sanity check failed for $VARIANT_NAME"
    cleanup_imagenet
    exit 1
fi

echo ""
echo "======================================================"
echo "  $VARIANT_NAME PASSED — $(date)"
echo "======================================================"

cleanup_imagenet
EOF