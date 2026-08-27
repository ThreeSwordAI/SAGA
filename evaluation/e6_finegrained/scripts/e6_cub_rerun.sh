#!/bin/bash
#SBATCH --job-name=saga_e6_cub
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e6/logs/e6_cub_rerun_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e6/logs/e6_cub_rerun_%j.err

# ── E6 CUB rerun — 4 variants, 30 epochs each ────────────────────────────────
#
# Why 30 epochs: original 100-epoch run showed both models peak at epoch 15-20
# and then overfit. 30 epochs captures the real peak cleanly.
#
# Why ViT-S: smaller capacity fits CUB-200 (5994 images) better.
# Adds scale comparison consistent with E2 classification results.
#
# Runs sequentially:
#   ViT-B baseline  30ep  (~30 min)
#   ViT-B SAGA      30ep  (~30 min)
#   ViT-S baseline  30ep  (~15 min)
#   ViT-S SAGA      30ep  (~15 min)
# Total: ~90 min, well within 6h wall time.
#
# Submit: sbatch evaluation/e6_finegrained/scripts/e6_cub_rerun.sh

source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/saga

export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
export PYTHONUNBUFFERED=1

cd $CODE_ROOT

PATHS=evaluation/e6_finegrained/configs/paths.yaml

echo "======================================================"
echo "  SAGA E6 — CUB-200 Rerun (30 epochs, ViT-B + ViT-S)"
echo "  Node: $SLURMD_NODENAME"
echo "  $(date)"
echo "======================================================"

run() {
    DATASET=$1; BACKBONE=$2; ARCH=$3; EPOCHS=$4
    NAME="${DATASET}_${BACKBONE}_${ARCH}_${EPOCHS}ep"
    echo ""
    echo "── $NAME ──────────────────────────────────────────"
    echo "  Start: $(date)"

    python3 evaluation/e6_finegrained/tools/train.py \
        --dataset   $DATASET \
        --backbone  $BACKBONE \
        --arch      $ARCH \
        --paths     $PATHS \
        --epochs    $EPOCHS \
        --run_name  $NAME

    if [ $? -ne 0 ]; then
        echo "ERROR: $NAME failed."
        exit 1
    fi
    echo "  Done: $(date)"
}

# ViT-B reruns (30 epochs — fixes overfitting)
run cub baseline vit_base_patch16_224 30
run cub saga     vit_base_patch16_224 30

# ViT-S runs (30 epochs — scale comparison)
run cub baseline vit_small_patch16_224 30
run cub saga     vit_small_patch16_224 30

# Summary
echo ""
echo "======================================================"
echo "  E6 CUB Rerun Summary"
echo "======================================================"
python3 -c "
import json, glob
from pathlib import Path

res_dir = Path('/home/vault/iwi5/iwi5359h/SAGA/e6/results')
print(f'  {\"Run\":<40} {\"Best top-1\":>12}')
print(f'  {\"-\"*52}')
for f in sorted(res_dir.glob('*cub*.json')):
    d = json.load(open(f))
    print(f'  {d[\"run\"]:<40} {d[\"best_top1\"]:>11.2f}%')
"

echo ""
echo "======================================================"
echo "  Done — $(date)"
echo "======================================================"