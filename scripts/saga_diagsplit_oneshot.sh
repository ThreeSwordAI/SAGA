#!/bin/bash
#SBATCH --job-name=saga_diagsplit
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_diagsplit_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_diagsplit_%j.err

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh
cd $CODE_ROOT
stage_imagenet
python tools/build_diag_split.py --data $STAGE_DIR --n-per-class 10 --seed 0 \
    --out results/diagsplit/val_diag_split.json
cleanup_imagenet