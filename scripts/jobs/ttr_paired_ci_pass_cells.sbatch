#!/bin/bash
#SBATCH --job-name=saga_ttr_ci_pass
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_ci_pass_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_ci_pass_%j.err

# TASK-10 — paired CIs for the two cells that PASSED the A3 gate, so every
# reported delta shares one statistical footing.
#
#     sbatch scripts/jobs/ttr_paired_ci_pass_cells.sbatch
#     sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/ttr_paired_ci_pass_cells.sbatch
#
# ViT-B/mixup already has its CI (it was the knife-edge FAIL that decision 1
# asked about). This covers the two PASS cells:
#   e2r_vits_mixup_baseline_s1  and  legacy_vits_baseline
# both ViT-S/mixup, both at n=24, which is each one's own gate best_n.
# ViT-S/nomix is deliberately NOT included: it was not endorsed, and it fails
# on the sink bar rather than the accuracy bar, so an accuracy CI would not
# bear on its verdict.
#
# Nothing here recomputes a verdict. The threshold stays frozen at 1.00 and
# the launcher cannot set it. These runs only attach measurement precision to
# deltas that are already reported.
#
# Both cells run in ONE job because each is ~7 min of compute against ~10 min
# of shared staging; splitting them would stage ImageNet val twice.
#
# PARTITION PROVENANCE: same a40 header as scripts/jobs/probe_attention.sbatch
# (TASK-11), from the human's own working a40 job (2026-09-09); recorded in
# docs/HPC_WORKFLOW.md "Partitions". --time=03:00:00 = staging + two ViT-S
# double passes over 50 000 images, with headroom.

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

# `set -u` ONLY AFTER the env sourcing — see docs/HPC_WORKFLOW.md.
set -u

PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is not executable. Resubmit with SAGA_PY=/path/to/python"
    exit 1
fi
if ! $PY -c "import torch, timm, scipy; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm/scipy — refusing to stage 50 000"
    echo "       images for a job that cannot run."
    exit 1
fi

TTR_OUT_ROOT=${TTR_OUT_ROOT:-results/ttr_midlayer}
N_BOOT=${N_BOOT:-100000}

# cell | arch | recipe | n_neurons | checkpoint
# The legacy checkpoint is on vault, not under results/runs — its path is the
# manifest's (results/legacy/checkpoint_manifest.csv, sha cd926200127b98cb...).
CELLS=(
  "e2r_vits_mixup_baseline_s1|vit_small|mixup|24|$CODE_ROOT/results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth"
  "legacy_vits_baseline|vit_small|mixup|24|/home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints/ViT-S_baseline/last.pth"
)

# Fail before staging if anything is missing — a 10-minute stage for a job
# that cannot run is the mistake this repo has already made once.
for entry in "${CELLS[@]}"; do
    IFS='|' read -r CELL ARCH RECIPE NB CKPT <<< "$entry"
    for f in "$CKPT" "$TTR_OUT_ROOT/$CELL/neurons.json"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: required input not found for $CELL: $f"
            exit 1
        fi
    done
done

trap cleanup_probe EXIT
stage_probe_imagenet_val

DONE=()
for entry in "${CELLS[@]}"; do
    IFS='|' read -r CELL ARCH RECIPE NB CKPT <<< "$entry"
    OUT=$TTR_OUT_ROOT/$CELL/paired_ci.json
    echo ""
    echo "=============================================================="
    echo "  $CELL  ($ARCH | $RECIPE)  n_neurons $NB"
    echo "  threshold FROZEN at 1.00; no verdict is recomputed"
    echo "=============================================================="
    $PY tools/ttr_paired_ci.py \
        --ckpt "$CKPT" \
        --arch "$ARCH" --recipe "$RECIPE" --variant baseline \
        --neurons-file "$TTR_OUT_ROOT/$CELL/neurons.json" \
        --n-neurons "$NB" \
        --data $PROBE_IN_DIR \
        --out "$OUT" \
        --n-boot "$N_BOOT" \
        --device cuda --seed 0
    DONE+=("$CELL")
done

echo ""
echo "======================================================"
echo "  DONE — commit the JSON *and* the npz for each cell"
echo "  (the npz is ~6 KB of packed per-image outcomes, so the"
echo "   CI can be recomputed later without a GPU):"
for c in "${DONE[@]}"; do
    echo "    $TTR_OUT_ROOT/$c/paired_ci.json"
    echo "    $TTR_OUT_ROOT/$c/paired_ci.npz"
done
echo "  Then locally: re-run analysis/build_ttr_tables.py and"
echo "  analysis/build_ttr_note.py — the tables and note fill themselves in."
echo "======================================================"
ls -la "$TTR_OUT_ROOT"/*/paired_ci.* 2>/dev/null
