#!/bin/bash
#SBATCH --job-name=abl_derive
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/abl_derive_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/abl_derive_%j.err

# TASK-12 PHASE C, STEP 0 — derive the canonical numbers for the six
# ablation runs. Nothing in Phase C's table exists without this.
#
#     sbatch scripts/jobs/abl_derive.sbatch
#
# The six runs already carry per-epoch full-val top-1 in their log.csv, but
# those are the trainer's bf16 values, and their periodic diag carries no
# _norms.npz and no fixed-threshold field. The paper's numbers come from:
#   1. tools/derive_runs.py  — tools/eval.py (exact fp32 full 50k val) and
#      tools/diagnose.py --attn (canonical diagnostics + _norms.npz) for
#      best and last of every abl_* run, then tools/summarize_norms.py.
#      Idempotent: each step is skipped when its output JSON exists AND its
#      recorded ckpt_sha256 still matches the checkpoint, so a wall-clock
#      kill is recovered by resubmitting this same file.
#   2. tools/apply_fixed_thr.py --version canon — the ViT-S/mixup cell's
#      tau from the COMMITTED fixed_thresholds_canon.json (20.8515625),
#      resolved from each run's own config.resolved.yaml. Never recalibrated.
#   3. tools/sink_address.py — the address maps Phase C's gate-vs-address
#      correlation needs.
#
# Order matters in 2 before 3: sink_address.py hard-checks its computed canon
# mass against the sibling diag's sink_fixed_canon, which step 2 writes.
# (TASK-10's ttr_matrix.sbatch carries the same ordering, for the same reason.)
#
# CHECKPOINTS ARE ON WOODY. The ablation runs were trained with --ckpt_root
# /home/woody/iwi5/iwi5359h/SAGA/abl_ckpt, and each run's meta.json records
# the resolved path; tools/derive_runs.py reads that field, so nothing here
# needs to name it.
#
# PARTITION PROVENANCE: the a40 header is the one in
# scripts/jobs/ttr_matrix.sbatch (TASK-10), itself from the human's own
# working a40 job (2026-09-09) and recorded in docs/HPC_WORKFLOW.md
# "Partitions". a40 is the right partition here: this is eval-only on one
# GPU. If a40 is busy, do NOT copy this file — override on the command line:
#     sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/abl_derive.sbatch
# --time=08:00:00 against tools/derive_runs.py's own "~3-5 h for 6 runs"
# estimate plus ~10 min of val-only staging; every step is idempotent, so a
# kill at the wall is recovered by resubmitting.

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

# `set -u` ONLY AFTER the env sourcing — env_alex.sh sources /etc/profile,
# whose site scripts dereference unset variables (debuginfod.sh line 8), and
# under set -u that aborts the job before one line of this script runs.
# Measured on Alex 2026-09-10; pinned repo-wide by a test.
set -u

# ── interpreter preflight: FATAL, and before any staging ────────────────────
# `which python` silently falls back to /usr/bin/python when the module load
# fails, and an unchecked `import torch` lets a job stage 50 000 images before
# it dies (TASK-10, job 4211911). Both are fatal here, and both run BEFORE the
# cleanup trap is armed, so nothing fires for a stage dir never created.
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is not executable. The env location is documented in"
    echo "       How to Run.md §1/§2. If it moved, resubmit with"
    echo "       SAGA_PY=/path/to/python sbatch $0"
    exit 1
fi
if ! $PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm — refusing to stage 50 000"
    echo "       images for a job that cannot run."
    exit 1
fi

# The threshold file must be the COMMITTED canon one, and it must already
# carry this cell's key. Fail here rather than after four hours of eval.
CANON=results/diagsplit/fixed_thresholds_canon.json
if ! $PY -c "import json,sys; d=json.load(open('$CANON')); sys.exit(0 if 'vit_small|mixup' in d else 1)"; then
    echo "ERROR: $CANON has no 'vit_small|mixup' key — the ablation cell's"
    echo "       tau is missing and nothing may be recalibrated here."
    exit 1
fi
echo "canon tau (vit_small|mixup): $($PY -c "import json;print(json.load(open('$CANON'))['vit_small|mixup'])")"

trap cleanup_probe EXIT
stage_probe_imagenet_val

# 1. eval + diagnose for best and last of every abl_* run, then the norm
#    summaries. Failures are logged and the driver continues; the log is
#    printed at the end and makes the job exit non-zero.
echo ""
echo "== derive: eval + diagnose over the six ablation runs =="
$PY tools/derive_runs.py \
    --runs-root results/runs \
    --pattern 'abl_*' \
    --data $PROBE_IN_DIR \
    --split-file results/diagsplit/val_diag_split.json
DERIVE_STATUS=$?

# 2. canon tau backfill. --diag-dir points at the legacy tree, which is
#    already fully backfilled, so this is a no-op there; only the new abl
#    diag JSONs gain sink_fixed_canon / canon_thr_value.
echo ""
echo "== canon tau backfill =="
$PY tools/apply_fixed_thr.py --version canon \
    --thr-file $CANON \
    --diag-dir results/legacy/diag \
    --runs-root results/runs

# 3. address maps from the fresh norms (needs step 2's field for its hard
#    cross-check)
echo ""
echo "== sink address maps =="
$PY tools/sink_address.py \
    --diag-dir results/legacy/diag \
    --runs-root results/runs \
    --thresholds $CANON

echo ""
echo "======================================================"
echo "  COMMIT THESE (all small; the *_norms.npz are git-ignored and must"
echo "  NOT be deleted — Phase C's figures read them):"
echo "    results/runs/abl_*/eval/imagenet_val_{best,last}.json"
echo "    results/runs/abl_*/diag/diag_final_{best,last}.json"
echo "    results/runs/abl_*/diag/diag_final_{best,last}_addr.json"
echo "    results/runs/abl_*/diag/*_normstats.json"
echo ""
echo "    git add results/runs/abl_*/eval results/runs/abl_*/diag"
echo "    git commit -m '[TASK-12] phase C derivation: eval + diag + address'"
echo "    I_AM_HUMAN=1 git push origin main"
echo "======================================================"
ls -la results/runs/abl_*/eval/ 2>/dev/null
ls results/runs/abl_*/diag/*.json 2>/dev/null

exit $DERIVE_STATUS
