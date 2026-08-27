#!/bin/bash
# detection/scripts/env_alex.sh
# ─────────────────────────────────────────────────────────────
# E3 environment for Alex cluster.
# Source this in every sbatch script.
# ─────────────────────────────────────────────────────────────

export PATH=$HOME/.local/bin:$PATH
source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/saga

export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

# COCO zip paths (read by stage_coco.sh)
export COCO_TRAIN_ZIP=/home/woody/iwi5/iwi5359h/Data/COCO/train2017.zip
export COCO_VAL_ZIP=/home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip
export COCO_ANN_ZIP=/home/woody/iwi5/iwi5359h/Data/COCO/annotations_trainval2017.zip

# Scratch staging directory
export STAGE_DIR=/scratch/iwi5359h/coco_${SLURM_JOB_ID}

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# Output directories
mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e3/checkpoints
mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e3/results
mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e3/logs