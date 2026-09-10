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
# The site RENAMED this module. 'python/3.12-conda' resolved until at least
# 2026-09-08 (TASK-08's ft array completed all 16 runs through this file) and
# was GONE by 2026-09-10, taking every job in the repo with it: the load
# failed, 'source activate' then failed, and jobs silently ran
# /usr/bin/python with no torch. 'module load python' is what works now
# (confirmed on the cluster by the human, 2026-09-10).
# Try the pinned name first so a restored modulefile is still preferred, fall
# back to the generic one, and otherwise say so LOUDLY — a job that limps on
# with the system python wastes its whole allocation.
if module load python/3.12-conda 2>/dev/null; then
    echo "env_alex: module python/3.12-conda"
elif module load python 2>/dev/null; then
    echo "env_alex: module python  (python/3.12-conda no longer exists)"
else
    echo "env_alex: WARNING - no python module loaded; PATH python is $(command -v python || echo none)"
    echo "env_alex:           use /home/vault/iwi5/iwi5359h/envs/saga/bin/python directly"
fi

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