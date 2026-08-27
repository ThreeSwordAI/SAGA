#!/bin/bash
#SBATCH --job-name=saga_e3_sanity
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e3/logs/e3_sanity_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e3/logs/e3_sanity_%j.err

# ── E3 Sanity check — 1 epoch of ViT-B_SAGA_det (variant 2) ──────────────────
# Verifies full pipeline: staging → data loading → forward pass → COCO eval.
# Expected time: ~30 min (1 epoch ViT-B on COCO at 800×1333)
#
# Submit: sbatch detection/scripts/e3_sanity_alex.sh

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/detection/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/detection/scripts/stage_coco.sh

cd $CODE_ROOT

echo "======================================================"
echo "  E3 Sanity — ViT-B_SAGA_det, 1 epoch"
echo "  Node: $SLURMD_NODENAME"
echo "  $(date)"
echo "======================================================"

stage_coco

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29900 \
    detection/tools/train.py \
        --config    detection/configs/variants.yaml \
        --id        2 \
        --paths     detection/configs/paths.yaml \
        --data_root $STAGE_DIR \
        --max_epochs 1

if [ $? -ne 0 ]; then
    echo "ERROR: E3 sanity check failed."
    cleanup_coco
    exit 1
fi

echo ""
echo "======================================================"
echo "  Sanity PASSED — ready to submit full E3"
echo "  Next: sbatch --array=0-2 detection/scripts/e3_train_alex.sh"
echo "  $(date)"
echo "======================================================"

cleanup_coco