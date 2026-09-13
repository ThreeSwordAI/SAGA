#!/bin/bash
#SBATCH --job-name=saga_ttr_paired_ci
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_paired_ci_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_paired_ci_%j.err

# TASK-10 PHASE C, decision 1 — paired uncertainty on the ViT-B knife edge.
#
#     sbatch scripts/jobs/ttr_paired_ci.sbatch
#     sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/ttr_paired_ci.sbatch
#
# This job exists because the tool needs ImageNet val STAGED: `--data` must
# point at a real val/ tree, and $PROBE_IN_DIR is defined only inside a job
# that has sourced scripts/stage_probe.sh. Running the command bare on a
# login node expands it to nothing (argparse: "--data: expected one
# argument") and would in any case put two 50 000-image ViT-B passes on a
# node with no GPU.
#
# The threshold stays frozen at 1.00 and no verdict is recomputed. This
# measures how precisely the drop behind the existing FAIL was measured:
# the discordant counts b/c, a paired bootstrap CI, and McNemar exact/mid-p.
#
# Defaults do decision 1's endorsed run (ViT-B/mixup at the matrix's n=24).
# The other cells are overridable but were NOT endorsed — see decision 3.
#
# PARTITION PROVENANCE: same a40 header as scripts/jobs/probe_attention.sbatch
# (TASK-11), from the human's own working a40 job (2026-09-09); recorded in
# docs/HPC_WORKFLOW.md "Partitions".
# --time=02:00:00 is dominated by staging (~10 min): the compute is two
# forward passes over 50 000 images on one ViT-B, plus a CPU bootstrap.

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

# `set -u` ONLY AFTER the env sourcing — env_alex.sh sources /etc/profile,
# whose site scripts dereference unset variables and would abort the job
# during setup. See docs/HPC_WORKFLOW.md.
set -u

# Interpreter by absolute path, and a preflight that actually aborts, before
# any staging. Both lessons from the 2026-09-10 failures.
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is not executable. Resubmit with SAGA_PY=/path/to/python"
    exit 1
fi
if ! $PY -c "import torch, timm, scipy; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm/scipy — refusing to stage 50 000"
    echo "       images for a job that cannot run. scipy is needed for the"
    echo "       exact McNemar test (requirements.txt pins scipy>=1.10.0)."
    exit 1
fi

CELL=${CELL:-e2r_vitb_mixup_baseline_s1}
ARCH=${ARCH:-vit_base}
RECIPE=${RECIPE:-mixup}
N_BEST=${N_BEST:-24}
TTR_OUT_ROOT=${TTR_OUT_ROOT:-results/ttr_midlayer}
N_BOOT=${N_BOOT:-100000}

CKPT=${CKPT:-$CODE_ROOT/results/runs/$CELL/ckpt/last.pth}
NEURONS=$TTR_OUT_ROOT/$CELL/neurons.json
OUT=$TTR_OUT_ROOT/$CELL/paired_ci.json

for f in "$CKPT" "$NEURONS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required input not found: $f"
        exit 1
    fi
done

trap cleanup_probe EXIT
stage_probe_imagenet_val

echo "======================================================"
echo "  TASK-10 paired CI — $CELL ($ARCH | $RECIPE)"
echo "  n_neurons $N_BEST   bootstrap $N_BOOT resamples"
echo "  threshold FROZEN at 1.00; no verdict is recomputed"
echo "======================================================"

$PY tools/ttr_paired_ci.py \
    --ckpt "$CKPT" \
    --arch "$ARCH" --recipe "$RECIPE" --variant baseline \
    --neurons-file "$NEURONS" \
    --n-neurons "$N_BEST" \
    --data $PROBE_IN_DIR \
    --out "$OUT" \
    --n-boot "$N_BOOT" \
    --device cuda --seed 0
RC=$?

echo "======================================================"
echo "  exit code $RC"
echo "  Commit BOTH (the npz is ~6 KB of packed per-image outcomes,"
echo "  so the CI can be recomputed later without a GPU):"
echo "    $OUT"
echo "    ${OUT%.json}.npz"
echo "  Then locally: re-run analysis/build_ttr_tables.py and"
echo "  analysis/build_ttr_note.py — section 7 fills itself in."
echo "======================================================"
ls -la "$TTR_OUT_ROOT/$CELL/"

exit $RC
