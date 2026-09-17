#!/bin/bash
#SBATCH --job-name=saga_frozen_I5b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I5b_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I5b_%j.err

# TASK C / I5b — the EXISTING ADE20K segmentation heads, frozen, re-evaluated
# for PAIRED PER-IMAGE records.
#
#     sbatch scripts/jobs/frozen_I5b.sbatch
#
# NOT A TRAINING JOB. No optimizer is constructed anywhere in the path this
# script runs; `tools/frozen_I5b_seg_eval.py::assert_eval_only` refuses a
# model that is in training mode, that has gradients enabled, or that has a
# parameter still requiring one. segmentation/tools/train.py is IMPORTED for
# its confusion-matrix functions — the TASK-09 B4-correct metric — and is
# never called.
#
# ── THE GATE ─────────────────────────────────────────────────────────────────
# The job REFUSES ITSELF unless the manifest carries ViT-B/mixup baseline AND
# SAGA segmentation weights at a MATCHED epoch, each with a recorded sha256
# (task file §2). As of the manifest committed at I0 Phase C that condition
# HOLDS — seg_vitb_baseline_s1 and seg_vitb_saga_s1, both `last`, both
# epochs_completed=80, both hashed — and the preflight below re-checks it on
# the cluster rather than trusting this comment. If it ever stops holding,
# the tool exits non-zero with the reason and NOTHING is retrained or
# re-fetched; I5b is then recorded as DROPPED.
#
# The registers segmentation row is NOT backbone-matched (its backbone is the
# unseeded legacy_e2_vitb_registers_nomixdir), so it is excluded. Add
# `--include-registers` only to produce clearly flagged context; every row it
# writes carries backbone_matched=0.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1, ONE GPU and no torch.distributed: the evaluation is
# 2,000 validation images at batch size 1 (the committed val pipeline's own
# batch size), and a single-rank run removes the all-reduce that
# evaluate_confusion needs when sharded — the per-image records must not
# depend on which rank an image landed on. The training jobs used 4 GPUs
# because they were training.
# --cpus-per-task=8 matches the seg jobs' dataloader worker budget.
# --time=04:00:00: ADE20K staging is ~10-15 min, then 2,000 forwards per
# condition (3 conditions total: baseline native, saga native, saga
# term_1.00).

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/segmentation/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/segmentation/scripts/stage_ade20k.sh
set -u   # after env sourcing: unset vars (e.g. no SLURM context) fail loudly
cd $CODE_ROOT

PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
MANIFEST=results/frozen/I0_manifest/manifest.json
OUT_ROOT=results/frozen/I5_readout/seg

echo "python: $PY"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is not executable (How to Run.md sections 1/2)." >&2
    exit 1
fi

# ── THE GATE, BEFORE STAGING ─────────────────────────────────────────────────
# Checked first, for the reason the seg chain jobs check completion first: a
# job that is going to refuse itself must not unzip ADE20K to do it.
if ! $PY -c "
import sys
sys.path.insert(0, '.')
from tools.frozen_I5b_seg_eval import matched_seg_rows
pair, reason = matched_seg_rows('$MANIFEST')
print(('I5b GATE OK: ' if pair else 'I5b GATE REFUSED: ') + reason)
sys.exit(0 if pair else 1)
"; then
    echo "" >&2
    echo "ERROR: I5b is DROPPED — the manifest does not carry matched-epoch" >&2
    echo "       ViT-B/mixup baseline and SAGA segmentation weights." >&2
    echo "       Nothing is retrained and nothing is re-fetched (§2, §4)." >&2
    echo "       Record 'I5b: DROPPED (reason)' in docs/TASK_LOG.md and in" >&2
    echo "       docs/C_HANDOFF.md, and do not resubmit this job." >&2
    exit 1
fi

stage_ade20k

# Node-local scratch is per job and must be released on EVERY exit path,
# including the SIGTERM at the wall clock.
trap 'cleanup_ade20k' EXIT

# The committed stage function enforces the VAL count, which is the only
# split this job reads. Assert it rather than assume it: a truncated
# extraction would silently produce per-image records for a subset and a
# dataset mIoU over the wrong denominator.
N_VAL=$(find $DATA_ROOT/images/validation -name "*.jpg" | wc -l)
if [ "$N_VAL" -ne 2000 ]; then
    echo "ERROR: ADE20K validation has $N_VAL images, expected 2000" >&2
    exit 1
fi

echo ""
echo "=============================================================="
echo "  frozen_I5b_seg_eval — frozen ADE20K, per-image records"
echo "=============================================================="
$PY tools/frozen_I5b_seg_eval.py \
    --data_root $DATA_ROOT \
    --matrix configs/dense_matrix.yaml \
    --manifest "$MANIFEST" \
    --out-root "$OUT_ROOT" \
    --device cuda --seed 0
RC=$?
echo "  -> exit $RC"

echo ""
echo "=============================================================="
if [ $RC -eq 0 ]; then
    echo "  DONE"
    echo ""
    echo "  Commit and push these, then stop:"
    echo "    results/frozen/I5_readout/seg/*/seg_records.parquet"
    echo "    results/frozen/I5_readout/seg/*/seg_records.done.json"
    echo "    results/frozen/I5_readout/seg/*/conf_matrix_*.npz"
    echo "    results/frozen/I5_readout/seg/I5b_summary.json"
else
    echo "  FAILED (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC   # the EXIT trap above releases the staged data
