#!/bin/bash
#SBATCH --job-name=saga_e4_sanity
#SBATCH --time=01:30:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e4/logs/e4_sanity_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e4/logs/e4_sanity_%j.err

# ── E4 Sanity check — 1 epoch of ViT-B_SAGA_seg (variant 2) ──────────────────
# Verifies: staging → data loading → forward pass → loss → mIoU eval.
# Expected time: ~20 min (1 epoch on ADE20K at 512×512)
#
# Submit: sbatch segmentation/scripts/e4_sanity_alex.sh

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/segmentation/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/segmentation/scripts/stage_ade20k.sh

cd $CODE_ROOT

echo "======================================================"
echo "  E4 Sanity — ViT-B_SAGA_seg, 1 epoch"
echo "  Node: $SLURMD_NODENAME"
echo "  $(date)"
echo "======================================================"

stage_ade20k

/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29920 \
    segmentation/tools/train.py \
        --config    segmentation/configs/variants.yaml \
        --id        2 \
        --paths     segmentation/configs/paths.yaml \
        --data_root $DATA_ROOT \
        --max_epochs 1

if [ $? -ne 0 ]; then
    echo "ERROR: E4 sanity check failed."
    cleanup_ade20k
    exit 1
fi

echo ""
echo "======================================================"
echo "  Sanity PASSED — ready for full E4"
echo "  Next: sbatch --array=0-2 segmentation/scripts/e4_train_alex.sh"
echo "  $(date)"
echo "======================================================"

cleanup_ade20k