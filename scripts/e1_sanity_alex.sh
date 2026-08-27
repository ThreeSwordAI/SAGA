#!/bin/bash
#SBATCH --job-name=saga_sanity_alex
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_sanity_alex_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_sanity_alex_%j.err

# ── Sanity check on Alex A100 — V00 baseline, 5 epochs ───────────────────────
# A100 (40GB) is much faster than RTX 2080 Ti — use batch_size=256
# Submit:  sbatch scripts/e1_sanity_alex.sh
# Watch:   tail -f $OUT_ROOT/logs/saga_sanity_alex_JOBID.log

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh

cd $CODE_ROOT

echo "======================================================"
echo "  SAGA Sanity Check (Alex A100) — V00, 5 epochs"
echo "  $(date)"
echo "  Node: $SLURMD_NODENAME  |  GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# ── Stage data ────────────────────────────────────────────────────────────────
stage_imagenet

# ── Train ─────────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  Starting training"
echo "  $(date)"
echo "======================================================"

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    tools/train.py \
        --config     configs/e1_variants.yaml \
        --variant_id 0 \
        --max_epochs 5 \
        --batch_size 256 \
        --data_root  $STAGE_DIR \
        --out_dir    $OUT_ROOT/checkpoints/e1_sanity_alex

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed."
    cleanup_imagenet
    exit 1
fi

echo ""
echo "======================================================"
echo "  Sanity check PASSED"
echo "  Next step: sbatch --array=0-15 scripts/e1_variants_alex.sh"
echo "  $(date)"
echo "======================================================"

cleanup_imagenet