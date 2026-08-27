#!/bin/bash
#SBATCH --job-name=saga_e5
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e5/logs/e5_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e5/logs/e5_%j.err

# ── E5 LOST Evaluation — runs on TinyGPU (single GPU, no DDP) ────────────────
#
# Pipeline:
#   1. Stage VOC 2007 from tar files (~870MB total, ~2 min)
#   2. Extract patch features for all 4 models (~5 min each, ~20 min total)
#   3. Run LOST algorithm on saved features (~5 min, CPU only)
#   4. Print CorLoc% table
#
# Submit: sbatch evaluation/e5_lost/scripts/e5_run_tinyx.sh

source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/saga

export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
export PYTHONUNBUFFERED=1

mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e5/{results,features,logs}

cd $CODE_ROOT

PATHS=evaluation/e5_lost/configs/paths.yaml

echo "======================================================"
echo "  SAGA E5 — LOST Evaluation on VOC 2007"
echo "  Node: $SLURMD_NODENAME"
echo "  $(date)"
echo "======================================================"

# Step 1+2: Extract features for all models
echo ""
echo "── Extracting features ──────────────────────────────"
python3 evaluation/e5_lost/tools/extract_features.py \
    --paths   $PATHS \
    --models  baseline registers saga pretrained \
    --split   test \
    --batch   32 \
    --workers 4

if [ $? -ne 0 ]; then
    echo "ERROR: Feature extraction failed."
    exit 1
fi

echo ""
echo "── Running LOST ──────────────────────────────────────"

# Step 3: Run LOST
python3 evaluation/e5_lost/tools/run_lost.py \
    --paths   $PATHS \
    --models  baseline registers saga pretrained \
    --split   test \
    --threshold  0.0 \
    --iou_thresh 0.5

if [ $? -ne 0 ]; then
    echo "ERROR: LOST evaluation failed."
    exit 1
fi

echo ""
echo "======================================================"
echo "  E5 DONE — $(date)"
echo "  Results: /home/vault/iwi5/iwi5359h/SAGA/e5/results/"
echo "======================================================"