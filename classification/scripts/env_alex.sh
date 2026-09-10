#!/bin/bash
# classification/scripts/env_alex.sh
# ─────────────────────────────────────────────────────────────
# Alex cluster environment for E2.
# Source this in every sbatch script. Do NOT submit directly.
# ─────────────────────────────────────────────────────────────

# ── Python / conda ─────────────────────────────────────────────
export PATH=$HOME/.local/bin:$PATH
source /etc/profile
# The site RENAMED this module: 'python/3.12-conda' resolved until at least
# 2026-09-08 and was GONE by 2026-09-10, taking every job in the repo with it
# (the load failed, 'source activate' then failed, and jobs silently ran
# /usr/bin/python with no torch). 'module load python' is what works now
# (confirmed on the cluster by the human, 2026-09-10). Pinned name first so a
# restored modulefile is still preferred; otherwise say so LOUDLY.
if module load python/3.12-conda 2>/dev/null; then
    echo "env_alex: module python/3.12-conda"
elif module load python 2>/dev/null; then
    echo "env_alex: module python  (python/3.12-conda no longer exists)"
else
    echo "env_alex: WARNING - no python module loaded; PATH python is $(command -v python || echo none)"
    echo "env_alex:           use /home/vault/iwi5/iwi5359h/envs/saga/bin/python directly"
fi
source activate /home/vault/iwi5/iwi5359h/envs/saga

# ── Code root ──────────────────────────────────────────────────
export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH

# ── Output roots (read from paths.yaml via Python — set here for shell) ────
export OUT_ROOT=/home/vault/iwi5/iwi5359h/SAGA
export JANUS_DATA=/home/janus/iwi5-datasets/imagenet/imagenet-1k/data

# ── Scratch — job-specific staging directory ───────────────────
export STAGE_DIR=/scratch/iwi5359h/imagenet_${SLURM_JOB_ID}

# ── Settings ───────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

# ── Output directories (create if missing) ─────────────────────
mkdir -p $OUT_ROOT/e2/checkpoints
mkdir -p $OUT_ROOT/e2/results
mkdir -p $OUT_ROOT/e2/logs
mkdir -p /home/woody/iwi5/iwi5359h/saga_e2_figures