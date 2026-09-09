#!/bin/bash
#SBATCH --job-name=saga_probe_attn
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_probe_attn_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_probe_attn_%j.err

# TASK-11 PHASE B — freeze the probe set, dump its attention, score it.
#
# PARTITION PROVENANCE: --partition=a40 / --gres=gpu:a40:1 are the human's
# own strings from a working a40 job on this cluster (2026-09-09).
# --cpus-per-task=4 comes from that same header. Everything else
# (log paths, env sourcing, staging, set -u) follows How to Run.md and the
# committed e2r/ft/dense job files. --time=06:00:00 is the ONE value with no
# source: TASK-11 estimates 2-4 h and staging adds ~30-40 min; lower it if
# the a40 partition has a shorter wall limit.

set -u

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

PY=$(which python)
echo "python: $PY"
$PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"

# Release node-local scratch on EVERY exit path — a job killed at the wall
# would otherwise leave the extracted copies behind (TASK-09's lesson).
trap cleanup_probe EXIT

# ── 1. ImageNet val + the two ImageNet probe groups ──────────────────────────
stage_probe_imagenet_val
$PY tools/build_probe_set.py --groups imagenet --if-missing \
    --data $PROBE_IN_DIR --out results/probe/probe_set.json

# ── 2. COCO val + the boxes20 group ──────────────────────────────────────────
stage_probe_coco_val
$PY tools/build_probe_set.py --groups boxes20 --if-missing \
    --coco-ann $PROBE_COCO_DIR/annotations/instances_val2017.json \
    --out results/probe/probe_set.json

# ── 3. Attention dumps ───────────────────────────────────────────────────────
# run_id | arch | variant | checkpoint
# A checkpoint that is absent is SKIPPED loudly, never silently substituted:
# the legacy registers row and the TASK-10 TTR row are both allowed to be
# missing, and every other consumer reports them as ABSENT rather than zero.
MODELS=(
  "e2r_vits_mixup_baseline_s1|vit_small|baseline|$CODE_ROOT/results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth"
  "e2r_vits_mixup_saga_s1|vit_small|saga|$CODE_ROOT/results/runs/e2r_vits_mixup_saga_s1/ckpt/last.pth"
  "e2r_vitb_mixup_baseline_s1|vit_base|baseline|$CODE_ROOT/results/runs/e2r_vitb_mixup_baseline_s1/ckpt/last.pth"
  "e2r_vitb_mixup_saga_s1|vit_base|saga|$CODE_ROOT/results/runs/e2r_vitb_mixup_saga_s1/ckpt/last.pth"
  "legacy_vits_registers|vit_small|registers|/home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints/ViT-S_registers/last.pth"
)

DUMPED=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r RUN_ID ARCH VARIANT CKPT <<< "$entry"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP $RUN_ID — checkpoint not found: $CKPT"
        continue
    fi
    echo "=== dumping $RUN_ID ($ARCH/$VARIANT) ==="
    $PY tools/dump_attention.py --run-id "$RUN_ID" --arch "$ARCH" \
        --variant "$VARIANT" --ckpt "$CKPT" \
        --groups curated,random200 --data $PROBE_IN_DIR
    $PY tools/dump_attention.py --run-id "$RUN_ID" --arch "$ARCH" \
        --variant "$VARIANT" --ckpt "$CKPT" \
        --groups boxes20 --coco-root $PROBE_COCO_DIR
    DUMPED+=("$RUN_ID")
done

if [ ${#DUMPED[@]} -eq 0 ]; then
    echo "ERROR: no checkpoint was dumpable — nothing downstream can run."
    exit 1
fi
RUNS=$(IFS=,; echo "${DUMPED[*]}")
echo "dumped runs: $RUNS"

# ── 4. Localization statistics (small, committed) ────────────────────────────
$PY tools/localization_score.py --runs "$RUNS" --groups random200,boxes20 \
    --out results/figures_data/F1_localization.csv

# ── 5. Figure arrays (small, committed) ──────────────────────────────────────
$PY analysis/collect_F1.py --runs "$RUNS" --data $PROBE_IN_DIR \
    --out results/figures_data/F1_teaser.npz

echo "======================================================"
echo "  DONE — commit these three, and NOT the attn dumps:"
echo "    results/probe/probe_set.json"
echo "    results/figures_data/F1_localization.csv (+ _meta.json)"
echo "    results/figures_data/F1_teaser.npz"
echo "  The attn/*.npz stay here (git-ignored) — Phase C needs them."
ls -la results/figures_data/F1_* results/probe/probe_set.json
du -sh results/runs/*/attn 2>/dev/null
echo "======================================================"
