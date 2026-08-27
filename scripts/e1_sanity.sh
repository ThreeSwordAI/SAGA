#!/bin/bash
#SBATCH --job-name=saga_sanity
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_sanity_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_sanity_%j.err

# ── Sanity check: V00 baseline for 5 epochs ──────────────────────────────────
# Runs on 1 GPU with batch_size=64 (safe for RTX 2080 Ti 11GB)
# Full training (E1 wave 1) uses 4 GPUs with batch_size=1024
# Submit:  sbatch scripts/e1_sanity.sh
# Watch:   tail -f $OUT_ROOT/logs/saga_sanity_JOBID.log

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh

cd $CODE_ROOT

echo "======================================================"
echo "  SAGA Sanity Check — V00 baseline, 5 epochs"
echo "  $(date)"
echo "======================================================"

# ── Stage data ────────────────────────────────────────────────────────────────
stage_imagenet

# ── Train ─────────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  Starting training"
echo "  $(date)"
echo "======================================================"

torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    tools/train.py \
        --config     configs/e1_variants.yaml \
        --variant_id 0 \
        --max_epochs 5 \
        --batch_size 64 \
        --data_root  $STAGE_DIR \
        --out_dir    $OUT_ROOT/checkpoints/e1_sanity

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed."
    cleanup_imagenet
    exit 1
fi

echo ""
echo "======================================================"
echo "  Sanity check PASSED"
echo "  Checkpoint: $OUT_ROOT/checkpoints/e1_sanity/V00_baseline/"
echo "  Next step:  sbatch --array=0-15 scripts/e1_variants.sh"
echo "  $(date)"
echo "======================================================"

cleanup_imagenet