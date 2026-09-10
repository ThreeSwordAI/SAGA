#!/bin/bash
#SBATCH --job-name=saga_ttr_matrix
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_matrix_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_ttr_matrix_%j.err

# TASK-10 PHASE B, STEP 2 — the four baseline cells with TTR applied.
#
# SUBMIT ONLY AFTER ttr_validate.sbatch PRINTED **PASS**, and pass the sweep
# value it chose:
#
#     N_BEST=16 sbatch scripts/jobs/ttr_matrix.sbatch
#
# For each of the four baseline checkpoints (TASK-10 A5.3) this runs:
#   1. tools/ttr_validate.py   — scans that cell's OWN register neurons and
#      records its own sweep (each model has its own register neurons; only
#      the CHOSEN n is shared across cells)
#   2. tools/ttr_prepare_run.py — results/runs/ttr_<id>/ with a
#      config.resolved.yaml naming the BASE cell, which is how
#      apply_fixed_thr --version canon later resolves the right tau
#   3. tools/ttr_derive.py     — full 50k eval + canonical diagnostics
# then, once over everything:
#   4. tools/apply_fixed_thr.py --version canon — the BASE cell's tau,
#      never anything recalibrated on a patched model
#   5. tools/sink_address.py   — the address maps, from the fresh norms
#
# Order matters in 4 before 5: sink_address.py hard-checks its computed
# canon mass against the sibling diag's sink_fixed_canon, which step 4
# writes. Running 5 first would leave every map without its cross-check.
#
# PARTITION PROVENANCE: identical header to scripts/jobs/probe_attention.sbatch
# (TASK-11), from the human's own working a40 job (2026-09-09); also recorded
# in docs/HPC_WORKFLOW.md "Partitions".
# --time=08:00:00 against TASK-10's own "~4-6 h on ONE A40" estimate for the
# eval-only matrix, plus ~10 min staging. Every step is idempotent, so if the
# wall clock kills this, resubmitting the same command finishes the rest.

# Fail before staging if N_BEST is missing. ${VAR:?} does not need `set -u`
# — it aborts on unset OR empty regardless — which is why this guard can
# stay up here while `set -u` has to wait until after the env sourcing.
: "${N_BEST:?set N_BEST to the --n-neurons value the A3 gate chose, e.g. N_BEST=16 sbatch $0}"

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
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

PY=$(which python)
echo "python: $PY"
echo "N_BEST: $N_BEST"
$PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"

trap cleanup_probe EXIT
stage_probe_imagenet_val

# base_run_id | arch | recipe_actual | checkpoint
#
# The fourth row is the LEGACY ViT-S baseline. Both legacy ViT-S baseline
# checkpoints carry recipe_actual=mixup (the nomix DIRECTORY is an additional
# repeat of the mixup cell — see results/notes/recipe_erratum.md), so
# "the legacy ViT-S baseline" is ambiguous in TASK-10 A5.3; the mixup-dir one
# is used here. Its sibling is
#   /home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints/ViT-S_baseline_nomix/last.pth
# (manifest sha 8d980da27a4b571a...), available as a second repeat of the same
# cell if the human wants both.
CELLS=(
  "e2r_vits_mixup_baseline_s1|vit_small|mixup|$CODE_ROOT/results/runs/e2r_vits_mixup_baseline_s1/ckpt/last.pth"
  "e2r_vitb_mixup_baseline_s1|vit_base|mixup|$CODE_ROOT/results/runs/e2r_vitb_mixup_baseline_s1/ckpt/last.pth"
  "e2r_vits_nomix_baseline_s1|vit_small|nomix|$CODE_ROOT/results/runs/e2r_vits_nomix_baseline_s1/ckpt/last.pth"
  "legacy_vits_baseline|vit_small|mixup|/home/vault/iwi5/iwi5359h/SAGA/e2/checkpoints/ViT-S_baseline/last.pth"
)

DONE=()
for entry in "${CELLS[@]}"; do
    IFS='|' read -r RUN_ID ARCH RECIPE CKPT <<< "$entry"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP $RUN_ID — checkpoint not found: $CKPT"
        continue
    fi
    echo ""
    echo "=============================================================="
    echo "  $RUN_ID   ($ARCH | $RECIPE)"
    echo "=============================================================="

    # 1. this cell's own register neurons + its own sweep. Deterministic for
    #    a fixed seed and data order, so a resubmission rewrites byte-identical
    #    content apart from the timestamp field.
    $PY tools/ttr_validate.py \
        --ckpt "$CKPT" --arch "$ARCH" --recipe "$RECIPE" --variant baseline \
        --n-neurons 4,8,16,32,64 \
        --data $PROBE_IN_DIR --base-run-id "$RUN_ID" \
        --out-root results/ttr --device cuda --seed 0 || \
        echo "  (gate returned non-zero for $RUN_ID — recorded, continuing)"

    # 2. the run dir the collectors will find
    $PY tools/ttr_prepare_run.py \
        --base-run-id "$RUN_ID" --arch "$ARCH" --recipe "$RECIPE" \
        --variant baseline \
        --neurons-file "results/ttr/$RUN_ID/neurons.json" \
        --n-neurons "$N_BEST" --runs-root results/runs --seed 0

    # 3. full 50k eval + diagnostics of the PATCHED model
    $PY tools/ttr_derive.py \
        --ckpt "$CKPT" --arch "$ARCH" --recipe "$RECIPE" --variant baseline \
        --neurons-file "results/ttr/$RUN_ID/neurons.json" \
        --n-neurons "$N_BEST" --base-run-id "$RUN_ID" \
        --data $PROBE_IN_DIR \
        --run-dir "results/runs/ttr_$RUN_ID" \
        --attn --device cuda --seed 0
    DONE+=("ttr_$RUN_ID")
done

if [ ${#DONE[@]} -eq 0 ]; then
    echo "ERROR: no checkpoint was derivable — nothing downstream can run."
    exit 1
fi
echo ""
echo "derived: ${DONE[*]}"

# 4. the BASE cell's canon tau, via the EXISTING backfill (unmodified).
#    --diag-dir points at the legacy tree, which is already fully backfilled,
#    so this is a no-op there and only the new TTR run dirs gain fields.
echo ""
echo "== canon tau backfill =="
$PY tools/apply_fixed_thr.py --version canon \
    --thr-file results/diagsplit/fixed_thresholds_canon.json \
    --diag-dir results/legacy/diag \
    --runs-root results/runs

# 5. address maps from the fresh norms (needs step 4's field for its
#    hard cross-check)
echo ""
echo "== sink address maps =="
$PY tools/sink_address.py \
    --diag-dir results/legacy/diag \
    --runs-root results/runs \
    --thresholds results/diagsplit/fixed_thresholds_canon.json

echo ""
echo "======================================================"
echo "  DONE — commit these (all small), and NOT the *_norms.npz:"
for r in "${DONE[@]}"; do
    echo "    results/runs/$r/{config.resolved.yaml,meta.json}"
    echo "    results/runs/$r/eval/eval_last.json"
    echo "    results/runs/$r/diag/diag_last.json"
    echo "    results/runs/$r/diag/diag_last_addr.json"
done
echo "    results/ttr/*/{neurons.json,sweep.csv,validate.json}"
echo "======================================================"
ls -la results/ttr/*/ results/runs/ttr_*/eval/ results/runs/ttr_*/diag/*.json 2>/dev/null
