#!/bin/bash
#SBATCH --job-name=abl_smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/abl_smoke_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/abl_smoke_%j.err

# TASK-12 A4 — 2-epoch smoke of ALL SIX ablation arms, in one job.
#
# WHY THIS IS NOT OPTIONAL: arms B-F are gate code paths that have never
# trained. const (frozen phi), headscalar ([H,1] phi) and layerscale
# (per-channel gamma at G1) are new modules; spatial at gate_init_logit=4.0
# has never been run either. Six 100-epoch chains is ~72 GPU-days of A100 —
# they do not get launched on untested code.
#
# The arms run SEQUENTIALLY inside this one job, so ImageNet is staged once
# (~20-40 min per scripts/stage_imagenet.sh) and one master port serves all
# six. Budget from the MEASURED cost of this configuration — median
# wall_time 161.4 s/epoch over the 300 epochs in
# results/runs/e2r_vits_mixup_saga_s1/log.csv: ~6 min of training per
# 2-epoch arm, plus two full-val passes and process start-up, ~50-60 min for
# six, ~1.5 h with staging. --time=04:00:00 is that plus a wide margin.
#
# Output goes to results/runs_smoke/ (NOT results/runs/), so a smoke can
# never be mistaken for a run; ckpt/ under it is git-ignored, the contract
# files are not. Verify with:
#     python tools/check_abl_smoke.py --runs-root results/runs_smoke
#
# Submit:  sbatch scripts/jobs/abl_smoke.sbatch
# (prefix --partition=a40 --gres=gpu:a40:4 only if a100 is unavailable; the
#  production chains are a100, so smoking on a100 is the like-for-like test)

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh
set -u
cd $CODE_ROOT

SAGA_PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}

# ── preflight: FATAL, and before staging ──────────────────────────────────
# TASK-10 lost a job to a `which python` fallback and to an import check
# whose exit status was never tested: it staged 50 000 images for a run that
# could not possibly work. Both guards are fatal here, and both run BEFORE
# the cleanup trap is armed, so nothing fires for a stage dir that was never
# created.
if [ ! -x "$SAGA_PY" ]; then
    echo "FATAL: interpreter $SAGA_PY is missing or not executable" >&2
    exit 1
fi
"$SAGA_PY" -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
    echo "FATAL: $SAGA_PY cannot import torch/timm" >&2
    exit 1
fi

# the gate-mode unit tests, ON THE COMPUTE NODE (TASK-09's lesson: a missing
# dependency otherwise shows up as a skipped test and a green report)
"$SAGA_PY" -m pytest -q tests/test_task12_ablation.py
if [ $? -ne 0 ]; then
    echo "FATAL: TASK-12 gate-mode tests fail on this node" >&2
    exit 1
fi

trap cleanup_imagenet EXIT
stage_imagenet

ARMS="abl_vits_mixup_baseline_s0 \
abl_vits_mixup_const05_s0 \
abl_vits_mixup_headscalar_s0 \
abl_vits_mixup_layerscale_s0 \
abl_vits_mixup_spatial_i0_s0 \
abl_vits_mixup_spatial_i4_s0"

FAILED=""
for RUN in $ARMS; do
    echo ""
    echo "=================== SMOKE: $RUN ==================="
    "$SAGA_PY" -m torch.distributed.run \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29769 \
        classification/tools/train.py \
            --matrix configs/abl_matrix.yaml \
            --run $RUN \
            --data_root $STAGE_DIR \
            --resume auto \
            --out_root results/runs_smoke \
            --max_epochs 2
    if [ $? -ne 0 ]; then
        echo "SMOKE FAILED: $RUN"
        FAILED="$FAILED $RUN"
    fi
done

echo ""
echo "=================== SMOKE SUMMARY ==================="
"$SAGA_PY" tools/check_abl_smoke.py --runs-root results/runs_smoke
CHECK=$?

if [ -n "$FAILED" ]; then
    echo "arms that did not finish:$FAILED"
    exit 1
fi
exit $CHECK
