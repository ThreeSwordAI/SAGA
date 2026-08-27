#!/bin/bash
#SBATCH --job-name=saga_e6
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e6/logs/e6_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e6/logs/e6_%j.err

# ── E6 Fine-grained Recognition — 4 runs on single TinyGPU ───────────────────
#
# Runs sequentially: CUB baseline → CUB SAGA → Aircraft baseline → Aircraft SAGA
# Time estimate: ~2h per run × 4 = ~8h total (fits in 12h wall time)
#
# Submit: sbatch evaluation/e6_finegrained/scripts/e6_run_tinyx.sh

source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/saga

export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
export PYTHONUNBUFFERED=1

mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e6/{checkpoints,results,logs}

cd $CODE_ROOT

PATHS=evaluation/e6_finegrained/configs/paths.yaml

echo "======================================================"
echo "  SAGA E6 — Fine-grained Recognition"
echo "  Node: $SLURMD_NODENAME"
echo "  $(date)"
echo "======================================================"

run_experiment() {
    DATASET=$1
    BACKBONE=$2
    echo ""
    echo "── $DATASET $BACKBONE ──────────────────────────────────"
    echo "  Start: $(date)"

    python3 evaluation/e6_finegrained/tools/train.py \
        --dataset  $DATASET \
        --backbone $BACKBONE \
        --paths    $PATHS \
        --epochs   100 \
        --batch    64 \
        --workers  4

    if [ $? -ne 0 ]; then
        echo "ERROR: $DATASET $BACKBONE failed."
        exit 1
    fi
    echo "  Done: $(date)"
}

# Run all 4 variants
run_experiment cub      baseline
run_experiment cub      saga
run_experiment aircraft baseline
run_experiment aircraft saga

# Print summary table
echo ""
echo "======================================================"
echo "  E6 Summary"
echo "======================================================"
python3 -c "
import json, glob
from pathlib import Path

res_dir = Path('/home/vault/iwi5/iwi5359h/SAGA/e6/results')
print(f'  {\"Run\":<28} {\"Best top-1\":>12} {\"Final top-1\":>12}')
print(f'  {\"-\"*52}')
for f in sorted(res_dir.glob('*.json')):
    d = json.load(open(f))
    print(f'  {d[\"run\"]:<28} {d[\"best_top1\"]:>11.2f}% {d[\"final_top1\"]:>11.2f}%')
"

echo ""
echo "======================================================"
echo "  E6 DONE — $(date)"
echo "======================================================"