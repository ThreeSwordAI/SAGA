#!/bin/bash
# ── SAGA: shared environment ───────────────────────────────────────────────────
# Source this in every sbatch script: source $CODE_ROOT/scripts/env.sh
# Do NOT submit directly.

# ── Paths ─────────────────────────────────────────────────────────────────────
export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export OUT_ROOT=/home/vault/iwi5/iwi5359h/SAGA

# ImageNet: 7 tar.gz shards on janus (shared, read-only, no file quota used)
export JANUS_DATA=/home/janus/iwi5-datasets/imagenet/imagenet-1k/data

# Staging: each job gets its own folder on /scratch using job ID
# /scratch has ~990 GB free — fast SSD, node-local
export STAGE_DIR=/scratch/iwi5359h/imagenet_${SLURM_JOB_ID}

# ── Modules ───────────────────────────────────────────────────────────────────
module load python/pytorch2.6py3.12
module load cuda/12.4.1

# ── Settings ──────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
export NCCL_DEBUG=WARN

# ── Output directories (on vault — permanent) ─────────────────────────────────
mkdir -p $OUT_ROOT/checkpoints
mkdir -p $OUT_ROOT/results
mkdir -p $OUT_ROOT/logs
mkdir -p $OUT_ROOT/figures