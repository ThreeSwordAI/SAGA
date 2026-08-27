#!/bin/bash
# classification/scripts/env_alex.sh
# ─────────────────────────────────────────────────────────────
# Alex cluster environment for E2.
# Source this in every sbatch script. Do NOT submit directly.
# ─────────────────────────────────────────────────────────────

# ── Python / conda ─────────────────────────────────────────────
export PATH=$HOME/.local/bin:$PATH
source /etc/profile
module load python/3.12-conda
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