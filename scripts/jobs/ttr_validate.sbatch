#!/bin/bash
#SBATCH --job-name=saga_ttr_validate
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_validate_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_validate_%j.err

# TASK-10 PHASE B, STEP 1 — the A3 validation gate for test-time registers.
#
# This job answers ONE question: does TTR do anything to a supervised ViT-S?
# It sweeps --n-neurons on the ViT-S/mixup baseline and prints a PASS/FAIL
# line. **Report that line back before submitting ttr_matrix.sbatch.**
# On FAIL for every n the full matrix must NOT be run — a validated negative
# is the deliverable (TASK-10 A3), and the exit code says which happened:
#   0 = PASS, 3 = FAIL (honestly measured), anything else = the tool broke.
#
# PARTITION PROVENANCE: --partition=a40 / --gres=gpu:a40:1 / --cpus-per-task=4
# are copied from scripts/jobs/probe_attention.sbatch (TASK-11), which took
# them from the human's own working a40 job on this cluster (2026-09-09); the
# same strings are recorded in docs/HPC_WORKFLOW.md "Partitions".
# --time=02:00:00 is dominated by ImageNet val staging (~10 min per
# How to Run.md §5): the compute itself is 7 passes over 5 000 images plus a
# 500-image scan on a ViT-S, i.e. minutes. The a40 wall limit is not
# documented — lower this if the partition refuses it.

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
# Reuses TASK-11's committed val-only stager rather than adding a third one:
# stage_probe_imagenet_val extracts ImageNet val to node-local scratch and
# enforces 1000 classes / 50000 images, which is exactly what eval and the
# diag split need. stage_imagenet.sh would also pull train/ — 10x the copy
# for a job that never touches it.
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

# `set -u` ONLY AFTER the env sourcing, never before it.
# env_alex.sh sources /etc/profile, which runs the site scripts in
# /etc/profile.d/. One of them dereferences an unset variable
# (debuginfod.sh line 8: DEBUGINFOD_URLS), and under `set -u` that aborts
# the whole job during env setup — before one line of this script runs.
# Measured on Alex: the .err contained exactly
#   /etc/profile.d/debuginfod.sh: line 8: DEBUGINFOD_URLS: unbound variable
# and nothing else. Every job file in this repo that has completed a real
# run (TASK-08 ft_*, TASK-09 det_*/seg_*/dense_smoke_*) puts `set -u` here,
# after the sourcing; a test pins that ordering repo-wide.
set -u

# ── Interpreter preflight: FATAL, and before any staging ─────────────────────
# Never `which python` here. If env_alex.sh's `module load python/3.12-conda`
# fails, PATH still resolves to /usr/bin/python, which has no torch.
# Measured on Alex 2026-09-10 (job 4211911): that modulefile was gone,
# `source activate` then failed with "activate: No such file or directory",
# `which python` gave /usr/bin/python, THIS check raised
# ModuleNotFoundError — and because its exit status was never tested the job
# carried on and spent 68 s staging 50 000 images before dying.
# The env interpreter by absolute path is the repo's own committed pattern
# (scripts/env_alex.sh's TORCHRUN line); it does not need the module at all.
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is not executable. The conda env location is documented"
    echo "       in How to Run.md §1/§2. If it moved, resubmit with"
    echo "       SAGA_PY=/path/to/python sbatch $0"
    exit 1
fi
if ! $PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm — refusing to stage 50 000"
    echo "       images for a job that cannot run. Check 'module avail python'"
    echo "       and the env at /home/vault/iwi5/iwi5359h/envs/saga."
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the
# wall clock (TASK-09's lesson: a plain call at the end only runs on a
# normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

CKPT=$CODE_ROOT/results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth
if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found — the A3 gate needs the seeded ViT-S/mixup"
    echo "       baseline, which is also the cell whose canon tau it uses."
    exit 1
fi

# The sweep grid. The default is TASK-10 A2.3's example, but it is a
# factor-of-2 grid, and the 2026-09-10 run (job 4212808) showed the whole
# sink/accuracy transition happening INSIDE one of its gaps: n=8 gave a 22%
# sink reduction for -0.28 top-1, n=16 gave 79% for -2.90. Refining the grid
# there is a measurement-resolution fix, not a moved goalpost — the PASS
# thresholds (--min-sink-reduction / --max-top1-drop) are untouched:
#     N_NEURONS=9,10,11,12,13,14,15 sbatch scripts/jobs/ttr_validate.sbatch
# NOTE: a rerun OVERWRITES results/ttr/<run_id>/{sweep.csv,validate.json}
# (neurons.json is deterministic and rewrites identically bar its
# timestamp), so COMMIT the previous sweep before launching another.
N_NEURONS=${N_NEURONS:-4,8,16,32,64}

echo "======================================================"
echo "  TASK-10 A3 — TTR validation gate"
echo "  base cell: vit_small|mixup   ckpt: $CKPT"
echo "  sweep:     $N_NEURONS"
echo "======================================================"

$PY tools/ttr_validate.py \
    --ckpt "$CKPT" \
    --arch vit_small --recipe mixup --variant baseline \
    --n-neurons "$N_NEURONS" \
    --data $PROBE_IN_DIR \
    --out-root results/ttr \
    --device cuda --seed 0
RC=$?

echo "======================================================"
echo "  exit code $RC  (0 = PASS, 3 = FAIL, other = tool error)"
echo "  Commit and push these two, then report the PASS/FAIL line:"
echo "    results/ttr/e2r_vits_mixup_baseline_s1/neurons.json"
echo "    results/ttr/e2r_vits_mixup_baseline_s1/sweep.csv"
echo "    results/ttr/e2r_vits_mixup_baseline_s1/validate.json"
echo "  Do NOT submit ttr_matrix.sbatch unless this said PASS."
echo "======================================================"
cat results/ttr/e2r_vits_mixup_baseline_s1/sweep.csv 2>/dev/null

exit $RC
