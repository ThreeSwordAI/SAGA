#!/bin/bash
#SBATCH --job-name=saga_ft_ring
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ft_ring_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ft_ring_%j.err

# TASK-13 A2 — ring ablation over every fine-tuned ft_* run that has a
# committed eval/test_final.json. EVAL-ONLY: no training, no checkpoint is
# written, no model selection happens.
#
# PARTITION: a100, at the human's explicit instruction (2026-09-15). Every
# SLURM value below is then copied from the ft pipeline's OWN committed
# single-GPU launcher, which completed all 16 TASK-08 fine-tunes on a100 —
# so nothing here is invented and nothing is carried over from an a40 header.
#
# PROVENANCE OF EVERY SLURM VALUE:
#   --partition=a100 / --gres=gpu:a100:1  How to Run.md §6 + the HEADER in
#                                         scripts/gen_ft_jobs.py, i.e. the
#                                         committed ft_finegrained_array.sbatch
#   --cpus-per-task=8                     same ft a100 header (NOT the a40
#                                         precedent's 4 — cpus are matched to
#                                         the partition they were measured on)
#   --time=06:00:00                       same ft a100 header. Generous here:
#                                         this job is eval-only (16 runs x 4
#                                         full-test passes, minutes each) plus
#                                         two dataset stagings, against a
#                                         header sized for a 100-epoch
#                                         fine-tune.
#   log paths, env sourcing, staging      How to Run.md §1/§5 + that same
#                                         committed launcher
# If a100 is busy, override without editing this file (HPC_WORKFLOW.md) —
# but a40's committed precedent uses --cpus-per-task=4, so set that too:
#   sbatch --partition=a40 --gres=gpu:a40:1 --cpus-per-task=4 scripts/jobs/ft_ring_ablation.sbatch

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
cd $CODE_ROOT

# set -u only AFTER the env sourcing: env_alex.sh sources /etc/profile, whose
# site scripts dereference unset variables and would abort this script.
set -u

# Address the env interpreter directly and GATE ON IT BEFORE STAGING — a
# missing torch/timm otherwise surfaces only after the tarballs are extracted
# (docs/HPC_WORKFLOW.md: an unchecked import cost TASK-10 a staged job).
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
if ! $PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm — not staging anything." >&2
    exit 1
fi

FT_STAGE=/scratch/iwi5359h/ft_ring_${SLURM_JOB_ID}
mkdir -p "$FT_STAGE"

# Release node-local scratch on EVERY exit path, including a kill at the wall
# (TASK-09's lesson: a plain cleanup at the end of the script never runs when
# the job is killed, which is an expected ending).
trap 'rm -rf "$FT_STAGE"' EXIT

# --all skips any run without a committed test_final.json, and re-skips a run
# whose output already matches this checkpoint/seed/index sets, so this job is
# safe to resubmit and will pick up the TASK-13 seed-fill runs once they land.
$PY tools/ring_ablation.py \
    --matrix configs/ft_matrix.yaml \
    --all \
    --stage_base "$FT_STAGE" \
    --seed 0
STATUS=$?

echo "======================================================"
echo "  DONE — commit the ring-ablation JSONs:"
echo "    results/finegrained/ring_ablation/*.json"
ls -la results/finegrained/ring_ablation/ 2>/dev/null
echo "======================================================"
exit $STATUS
