#!/bin/bash
# segmentation/scripts/env_alex.sh
# ─────────────────────────────────────────────────────────────
# E4 environment for Alex cluster.
# Source this in every sbatch script.
# ─────────────────────────────────────────────────────────────

export PATH=$HOME/.local/bin:$PATH
source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/saga

export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

# ADE20K zip path
export ADE_ZIP=/home/woody/iwi5/iwi5359h/Data/ADE20K/ADEChallengeData2016.zip

# Scratch staging directory
export STAGE_DIR=/scratch/iwi5359h/ade20k_${SLURM_JOB_ID}

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

# Output directories
mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e4/checkpoints
mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e4/results
mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e4/logs