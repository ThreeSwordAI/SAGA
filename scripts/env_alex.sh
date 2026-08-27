#!/bin/bash
# ── SAGA: Alex environment ─────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
export PATH=$HOME/.local/bin:$PATH
export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export OUT_ROOT=/home/vault/iwi5/iwi5359h/SAGA
export JANUS_DATA=/home/janus/iwi5-datasets/imagenet/imagenet-1k/data
export STAGE_DIR=/scratch/iwi5359h/imagenet_${SLURM_JOB_ID}

# ── Conda — must use full path in non-interactive sbatch jobs ─────────────────
source /etc/profile
module load python/3.12-conda

# Activate using full path — 'conda activate' does not work in sbatch
source activate /home/vault/iwi5/iwi5359h/envs/saga

# ── Settings ──────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
export NCCL_DEBUG=WARN

# ── Output directories ────────────────────────────────────────────────────────
mkdir -p $OUT_ROOT/checkpoints
mkdir -p $OUT_ROOT/results
mkdir -p $OUT_ROOT/logs
mkdir -p $OUT_ROOT/figures

export TORCHRUN="/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run"