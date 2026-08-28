#!/bin/bash
#SBATCH --job-name=saga_rederive_e2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_rederive_e2_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_rederive_e2_%j.err

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh
cd $CODE_ROOT
stage_imagenet
bash scripts/rederive_e2.sh
STATUS=$?
cleanup_imagenet
exit $STATUS