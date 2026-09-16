#!/bin/bash
#SBATCH --job-name=saga_frozen_I6
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I6_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I6_%A_%a.err

# TASK A / I6 A8 — the nine external public checkpoints, one array task per
# model. Thresholds on CALIBRATION, maps + per-image indicators on
# EVALUATION, at both analog stages, under fixed_cal and MAD.
#
#     python tools/frozen_I6_download.py       # LOGIN NODE, first, once
#     git add -A && git commit                 # the weight shas
#     sbatch scripts/jobs/frozen_I6.sbatch
#
# THE DOWNLOAD IS NOT PART OF THIS JOB and cannot be. Compute nodes on this
# cluster have no outbound network, so the weights must already be in the
# woody cache the registry names. This job verifies each weight file's sha256
# against the committed registry BEFORE loading it, and fails loudly if the
# registry still says PENDING — which is what it says until the download tool
# has run and its output has been committed.
#
# WHY THE TWO SPLITS DIFFER HERE. A public checkpoint is its own reference:
# there is no designated baseline in its cell to borrow a threshold from, the
# way tools/frozen_I2_thresholds.py does for our cohort. Calibrating on
# `calibration` and reporting on `evaluation` is what keeps the threshold off
# the images it is reported on. Both shas go into
# results/frozen/I6_external/thresholds_cal.json so the split difference is
# recorded rather than inferred.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1 / cpus-per-task=4, the strings of the repo's other a100
# array jobs. a40 also works and is the cheaper queue for an eval-only job —
# switch on the COMMAND LINE rather than copying this file
# (docs/HPC_WORKFLOW.md "Partitions"):
#     sbatch --partition=a40 --gres=gpu:a40:1 scripts/jobs/frozen_I6.sbatch
# Keep --exclude=a0801 on the command line, where the repo's other a100
# launchers put it.
#
# --time=04:00:00: staging ImageNet val is ~10 min (How to Run.md §5); then
# one forward pass over 2,000 calibration images and one over 10,000
# evaluation images per model, with no SVD and no gate. The ViT-B models are
# the slow ones. A task that hits the wall leaves NOTHING extra — every npz
# is written to a temp file and renamed, and the completion marker is written
# last — so it is simply resubmitted.
#
# ── RESUBMISSION ─────────────────────────────────────────────────────────────
# `--skip-if-done` exits 0 for a model whose completion marker already names
# the same weight sha, and a model already present in thresholds_cal.json is
# not recalibrated. Resubmit the whole array; it adds what is missing and
# rewrites nothing.

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
# TASK-11's committed val-only stager. stage_imagenet.sh would also pull the
# five train shards — 10x the copy for a job that never reads them.
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

# `set -u` ONLY AFTER the env sourcing: env_alex.sh sources /etc/profile,
# whose site scripts dereference unset variables (debuginfod.sh line 8,
# DEBUGINFOD_URLS), and under `set -u` bash aborts the CALLING script before
# one line of the job body runs (docs/HPC_WORKFLOW.md, TASK-10 failure #2).
set -u

REGISTRY=configs/frozen/I6_models.yaml
CONDITIONS=configs/frozen/I6_external.yaml
CAL_SPLIT=results/frozen/splits/calibration.json
EVAL_SPLIT=results/frozen/splits/evaluation.json
OUT_ROOT=results/frozen/I6_external

# The hub cache is on woody, never in $HOME (the small quota — How to Run.md
# §3). saga/frozen/external.py::set_cache_env reads this, falling back to the
# registry's `hf_home`. Exported here as well so anything timm imports sees it
# before it decides where to look.
export SAGA_HF_HOME=${SAGA_HF_HOME:-/home/woody/iwi5/iwi5359h/SAGA/hf_cache}
export HF_HOME=$SAGA_HF_HOME
export TORCH_HOME=$SAGA_HF_HOME
# Compute nodes have no network: fail fast and loudly instead of hanging on a
# hub request that cannot complete.
export HF_HUB_OFFLINE=1

# ── Interpreter preflight: FATAL, and BEFORE any staging ─────────────────────
# Never `which python`: if the module load fails, PATH still resolves to
# /usr/bin/python, which has no torch. An unchecked import cost TASK-10 a job
# that staged 50 000 images and then died in Python.
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python:  $PY"
echo "HF_HOME: $HF_HOME"
echo "task:    ${SLURM_ARRAY_TASK_ID:-<none>}"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is not executable. The conda env location is in" >&2
    echo "       How to Run.md sections 1/2. If it moved, resubmit with" >&2
    echo "       SAGA_PY=/path/to/python sbatch $0" >&2
    exit 1
fi
if ! $PY -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"; then
    echo "ERROR: $PY cannot import torch/timm — refusing to stage 50 000" >&2
    echo "       images for a job that cannot run." >&2
    exit 1
fi

# ── The registry, and the array index that selects from it ───────────────────
# The nine model_ids in registry order: the four supervised-with-mixing
# models first (indices 0-3), then the four no-mixing models (4-7), then the
# exploratory register variant (8). The list is spelled out so the array
# indices are auditable from the job file alone, and
# tests/test_I6_external.py::test_the_job_file_lists_the_registry_in_array_order
# pins it against configs/frozen/I6_models.yaml.
MODEL_IDS=(
    "deit_small_patch16_224"            # 0  supervised_mixing      14x14
    "deit3_small_patch16_224"           # 1  supervised_mixing      14x14
    "deit3_base_patch16_224"            # 2  supervised_mixing      14x14
    "vit_base_patch16_224_augreg"       # 3  supervised_mixing      14x14
    "vit_small_patch14_dinov2"          # 4  no_mixing              16x16
    "vit_base_patch14_dinov2"           # 5  no_mixing              16x16
    "vit_base_patch16_clip_224_openai"  # 6  no_mixing              14x14
    "vit_base_patch16_224_mae"          # 7  no_mixing              14x14
    "vit_small_patch14_reg4_dinov2"     # 8  registers_exploratory  16x16
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#MODEL_IDS[@]}" ]; then
    echo "ERROR: array index $IDX is outside 0..$(( ${#MODEL_IDS[@]} - 1 ))" >&2
    exit 1
fi
MODEL_ID="${MODEL_IDS[$IDX]}"

if ! $PY -c "
import sys
sys.path.insert(0, '.')
from saga.frozen.external import load_registry
want = [m['model_id'] for m in load_registry('$REGISTRY')['models']]
# [*] and not [@]: inside double quotes [@] would expand to SEPARATE words
# and split this single -c argument into ten of them
have = '''${MODEL_IDS[*]}'''.split()
if want != have:
    print('REGISTRY DRIFT', file=sys.stderr)
    print('  job file:', have, file=sys.stderr)
    print('  registry:', want, file=sys.stderr)
    sys.exit(1)
"; then
    echo "ERROR: scripts/jobs/frozen_I6.sbatch and $REGISTRY disagree about" >&2
    echo "       the model list. An array index must mean the same model on" >&2
    echo "       both sides." >&2
    exit 1
fi

# ── The weight shas must already be committed ────────────────────────────────
# The registry ships with weight_sha256: PENDING. The download tool fills it
# on the login node. A job that ran against PENDING would measure whatever
# happened to be in the cache, which is exactly the traceability this
# project's own checkpoint rules forbid.
if ! $PY -c "
import sys
sys.path.insert(0, '.')
from saga.frozen.external import load_registry, model_entry
e = model_entry(load_registry('$REGISTRY'), '$MODEL_ID')
sha = str(e.get('weight_sha256', 'PENDING'))
if sha in ('PENDING', '', 'None'):
    print('weight_sha256 is ' + sha, file=sys.stderr)
    sys.exit(1)
print('registry sha', sha[:16] + '...')
"; then
    echo "ERROR: $MODEL_ID has no recorded weight_sha256. Run this on the" >&2
    echo "       LOGIN NODE first, commit the registry, then resubmit:" >&2
    echo "         python tools/frozen_I6_download.py" >&2
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the wall
# clock (TASK-09's lesson: a plain call at the end only runs on a normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

echo ""
echo "=============================================================="
echo "  frozen_I6_maps: $MODEL_ID   (array index $IDX)"
echo "=============================================================="
$PY tools/frozen_I6_maps.py \
    --model-id "$MODEL_ID" \
    --registry "$REGISTRY" \
    --conditions "$CONDITIONS" \
    --calibration-split "$CAL_SPLIT" \
    --evaluation-split "$EVAL_SPLIT" \
    --data "$PROBE_IN_DIR" \
    --out-root "$OUT_ROOT" \
    --batch-size 64 \
    --device cuda --seed 0 --skip-if-done
RC=$?
echo "  -> exit $RC"

echo ""
echo "=============================================================="
if [ $RC -eq 0 ]; then
    echo "  DONE: $MODEL_ID"
    echo ""
    echo "  After the WHOLE array finishes, commit and push these, then stop:"
    echo "    results/frozen/I6_external/thresholds_cal.json"
    echo "    results/frozen/I6_external/*/maps_ext_*.npz"
    echo "    results/frozen/I6_external/*/maps.done.json"
    echo "    results/frozen/I6_external/*/run_meta.json"
    echo "    configs/frozen/I6_models.yaml   (if the shas are not in yet)"
else
    echo "  FAILED: $MODEL_ID (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC
