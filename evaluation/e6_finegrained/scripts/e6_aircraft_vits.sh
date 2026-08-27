#!/bin/bash
#SBATCH --job-name=saga_e6_aircraft_s
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e6/logs/e6_aircraft_vits_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e6/logs/e6_aircraft_vits_%j.err

# ── E6 Aircraft ViT-S — 2 runs, 30 epochs each ───────────────────────────────
# Adds scale comparison for Aircraft consistent with CUB S/B table.
# Time estimate: ~20 min per run × 2 = ~40 min total.
#
# Submit: sbatch evaluation/e6_finegrained/scripts/e6_aircraft_vits.sh

source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/saga

export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
export PYTHONUNBUFFERED=1

cd $CODE_ROOT

PATHS=evaluation/e6_finegrained/configs/paths.yaml

echo "======================================================"
echo "  SAGA E6 — Aircraft ViT-S (30 epochs)"
echo "  Node: $SLURMD_NODENAME"
echo "  $(date)"
echo "======================================================"

run() {
    BACKBONE=$1
    NAME="aircraft_${BACKBONE}_vit_small_patch16_224_30ep"
    echo ""
    echo "── $NAME ──────────────────────────────────────────"
    echo "  Start: $(date)"

    python3 evaluation/e6_finegrained/tools/train.py \
        --dataset   aircraft \
        --backbone  $BACKBONE \
        --arch      vit_small_patch16_224 \
        --paths     $PATHS \
        --epochs    30 \
        --run_name  $NAME

    if [ $? -ne 0 ]; then
        echo "ERROR: $NAME failed."
        exit 1
    fi
    echo "  Done: $(date)"
}

run baseline
run saga

echo ""
echo "======================================================"
echo "  Results"
echo "======================================================"
python3 -c "
import json
from pathlib import Path

res_dir = Path('/home/vault/iwi5/iwi5359h/SAGA/e6/results')
names = [
    'aircraft_baseline_vit_small_patch16_224_30ep',
    'aircraft_saga_vit_small_patch16_224_30ep',
]
print(f'  {\"Run\":<48} {\"Best top-1\":>12}')
print(f'  {\"-\"*60}')
for name in names:
    f = res_dir / f'{name}.json'
    if f.exists():
        d = json.load(open(f))
        print(f'  {name:<48} {d[\"best_top1\"]:>11.2f}%')
"

echo ""
echo "======================================================"
echo "  Done — $(date)"
echo "======================================================"