#!/bin/bash
#SBATCH --job-name=saga_frozen_I4
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I4_%A_%a.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_I4_%A_%a.err

# TASK B / I4 B8 — the masked receiver-perturbation sweep: 65 conditions per
# checkpoint over the 10 ViT-S/mixup checkpoints, one array task per run_id.
#
#     sbatch scripts/jobs/frozen_I4.sbatch
#
# ── EMBARGOED. DO NOT SUBMIT THIS YET. ───────────────────────────────────────
# I4 Phase B may not run until BOTH of these are true:
#
#   1. D5 is CLOSED — TASK A / I1 Phase C has written
#      configs/frozen/I4_masks.json from the DISCOVERY split, and
#      docs/LOCKED_ANALYSIS.md records that file's sha256 against a
#      struck-through ~~D5~~; and
#   2. docs/LOCKED_ANALYSIS.md carries the header line `STATUS: FROZEN` with
#      a date and a git sha.
#
# Neither is true as this file is written. tools/frozen_eval.py REFUSES any
# configs/frozen/I4_*.yaml until both are, and the check below refuses before
# a single image is staged. Submitting today costs a scheduling round trip
# and produces a named refusal, not a result — which is the intended
# behaviour, not a failure.
#
# Why the embargo is a guard and not a note: I4 is the experiment in this
# project that could most easily be tuned into a result. The 16 coordinates
# it perturbs are the whole question, so the masks are selected on discovery
# data and fixed in a signed document BEFORE any evaluation loss is seen. A
# mask chosen after looking would make the result circular, and the only way
# "we did not" stays true under maintenance is if the code refuses the
# alternative.
#
# ── THE SPLIT ────────────────────────────────────────────────────────────────
# Fixed to evaluation.json inside this file, as in frozen_I3.sbatch. TASK B
# §2: nothing in Track B runs on discovery or calibration. The MASKS come
# from discovery — that is a different file, read by digest, and
# saga/frozen/prevalence.py refuses to build one from any other split.
#
# ── WHAT RUNS, AND WHICH INDEX IS WHICH ──────────────────────────────────────
# The ViT-S/mixup cell and nothing else: D5's masks are defined for that cell,
# so no other cell is in I4 (TASK B §2). In the order
# saga.frozen.runner.eligible_cohort defines, filtered to that cell.
#
#   index  run_id                                   variant     scope
#   -----  ---------------------------------------  ----------  --------------
#     0    legacy_e2_vit_small_mixupdir_baseline    baseline    legacy
#     1    legacy_e2_vit_small_nomixdir_baseline    baseline    legacy
#     2    e2r_vits_mixup_baseline_s1               baseline    FRESH — decides
#     3    e2r_vits_mixup_baseline_s2               baseline    FRESH — decides
#     4    legacy_e2_vit_small_mixupdir_registers   registers   n = 2, labelled
#     5    legacy_e2_vit_small_nomixdir_registers   registers   n = 2, labelled
#     6    legacy_e2_vit_small_mixupdir_saga        saga        legacy
#     7    legacy_e2_vit_small_nomixdir_saga        saga        legacy
#     8    e2r_vits_mixup_saga_s1                   saga        FRESH — decides
#     9    e2r_vits_mixup_saga_s2                   saga        FRESH — decides
#
# C3 is decided on the fresh BASELINES (2, 3) and the fresh SAGA checkpoints
# (8, 9) SEPARATELY (TASK B §6). The registers are n = 2 and are labelled as
# such in every table; they never decide.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# a100 / gpu:a100:1 / cpus-per-task=4, the strings of the repo's own a100
# array jobs. a40 also works and is the cheaper queue for an eval-only job —
# switch on the COMMAND LINE rather than copying this file:
#     sbatch --partition=a40 --gres=gpu:a40:1 scripts/jobs/frozen_I4.sbatch
# Keep --exclude=a0801 on the command line, where the repo's other a100
# launchers put it.
#
# --time=04:00:00, as frozen_I1/I2/I3 use. 65 forwards over 10,000 images is
# four more than I3 and every one is a plain ViT-S inference in fp32. The
# energy-matched controls cost nothing extra: the target is read from the
# primary's own pass, which has already run.
#
# ── RESUBMISSION ─────────────────────────────────────────────────────────────
# Idempotent: `--skip-if-done` exits 0 for a run whose completion marker
# already names this exact checkpoint, and `append_rows` skips rows already
# present on (condition, image). Resubmit the whole array; it adds what is
# missing and rewrites nothing.

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

SPLIT=results/frozen/splits/evaluation.json
CONDITIONS=configs/frozen/I4_perturbation.yaml
MASKS=configs/frozen/I4_masks.json
MANIFEST=results/frozen/I0_manifest/manifest.json
OUT_ROOT=results/frozen

# ── Interpreter preflight: FATAL, and BEFORE any staging ─────────────────────
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
echo "split:  $SPLIT   (fixed; TASK B runs on evaluation and nothing else)"
echo "masks:  $MASKS   (D5, selected on DISCOVERY, read by digest)"
echo "task:   ${SLURM_ARRAY_TASK_ID:-<none>}"
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

# ── THE FREEZE GUARD, before anything is staged ──────────────────────────────
# The same check tools/frozen_eval.py makes, run here so that an embargoed
# submission costs seconds rather than a staged dataset. It names every
# missing piece at once.
# The refusal is printed to STDOUT as well as stderr, so the .log a person
# actually opens explains itself. Measured 2026-09-17 on job 4265735: the
# guard fired correctly, but every line of the reason went to the .err and
# the .log jumped from the torch version straight to JOB_STATISTICS — a job
# that stopped for no visible cause. A guard whose message is in the other
# file is a guard nobody reads.
if ! $PY -c "
import sys
sys.path.insert(0, '.')
from tools.frozen_eval import assert_not_embargoed
try:
    state = assert_not_embargoed('$CONDITIONS')
except Exception as exc:
    print(exc)                      # stdout -> the .log
    print(exc, file=sys.stderr)     # stderr -> the .err
    sys.exit(1)
print('LOCKED_ANALYSIS is FROZEN:', state['date'], state['git_sha'])
print('D5 masks sha256:', state['masks_sha'])
"; then
    for stream in 1 2; do
        echo ""                                                        >&$stream
        echo "ERROR: I4 is EMBARGOED (TASK B section 7). D5 must be"    >&$stream
        echo "       closed and docs/LOCKED_ANALYSIS.md must be FROZEN" >&$stream
        echo "       before this job may run. The refusal above names"  >&$stream
        echo "       what is missing."                                  >&$stream
        echo "       This is the guard working, not a bug. Nothing was" >&$stream
        echo "       staged and nothing was written. I3 is unaffected." >&$stream
    done
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the wall
# clock (TASK-09's lesson: a plain call at the end only runs on a normal exit).
trap cleanup_probe EXIT

# ── The cohort, and the array index that selects from it ─────────────────────
# tests/test_I4_perturbation.py::test_the_job_file_lists_the_cell_in_array_order
# pins this against the function; the runtime check below refuses to run if
# the manifest and this list have drifted apart.
RUN_IDS=(
    "legacy_e2_vit_small_mixupdir_baseline"       # 0  baseline   legacy
    "legacy_e2_vit_small_nomixdir_baseline"       # 1  baseline   legacy
    "e2r_vits_mixup_baseline_s1"                  # 2  baseline   FRESH
    "e2r_vits_mixup_baseline_s2"                  # 3  baseline   FRESH
    "legacy_e2_vit_small_mixupdir_registers"      # 4  registers  n = 2
    "legacy_e2_vit_small_nomixdir_registers"      # 5  registers  n = 2
    "legacy_e2_vit_small_mixupdir_saga"           # 6  saga       legacy
    "legacy_e2_vit_small_nomixdir_saga"           # 7  saga       legacy
    "e2r_vits_mixup_saga_s1"                      # 8  saga       FRESH
    "e2r_vits_mixup_saga_s2"                      # 9  saga       FRESH
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#RUN_IDS[@]}" ]; then
    echo "ERROR: array index $IDX is outside 0..$(( ${#RUN_IDS[@]} - 1 ))" >&2
    exit 1
fi
RUN_ID="${RUN_IDS[$IDX]}"

if ! $PY -c "
import sys
sys.path.insert(0, '.')
from saga.frozen.runner import eligible_cohort
want = [r['run_id'] for r in eligible_cohort('$MANIFEST')
        if (r['arch'], r['recipe_actual']) == ('vit_small', 'mixup')]
# [*] and not [@]: inside double quotes [@] would expand to SEPARATE words
# and split this single -c argument into ten of them
have = '''${RUN_IDS[*]}'''.split()
if want != have:
    print('MANIFEST COHORT DRIFT', file=sys.stderr)
    print('  job file:', have, file=sys.stderr)
    print('  manifest:', want, file=sys.stderr)
    sys.exit(1)
"; then
    echo "ERROR: scripts/jobs/frozen_I4.sbatch and the manifest disagree" >&2
    echo "       about the ViT-S/mixup cell. Rebuild the manifest or fix the" >&2
    echo "       job file — an array index must mean the same run on both" >&2
    echo "       sides." >&2
    exit 1
fi

stage_probe_imagenet_val

# ── The sweep for this array task's run ──────────────────────────────────────
echo ""
echo "=============================================================="
echo "  frozen_I4: $RUN_ID   (array index $IDX)"
echo "  65 conditions x $(basename $SPLIT)"
echo "=============================================================="
$PY tools/frozen_eval.py \
    --run-id "$RUN_ID" \
    --conditions "$CONDITIONS" \
    --split "$SPLIT" \
    --data "$PROBE_IN_DIR" \
    --manifest "$MANIFEST" \
    --out-root "$OUT_ROOT" \
    --batch-size 64 \
    --device cuda --seed 0 --skip-if-done
RC=$?
echo "  -> exit $RC"

echo ""
echo "=============================================================="
if [ $RC -eq 0 ]; then
    echo "  DONE: $RUN_ID"
    echo ""
    echo "  Check the energy match before trusting anything:"
    echo "    run_meta.json -> energy_rel_error_max  (must be < 0.01)"
    echo "    run_meta.json -> n_zero_norm_images    (reported, never dropped)"
    echo ""
    echo "  After the whole array finishes, commit and push these, then stop:"
    echo "    results/frozen/I4_perturbation/evaluation/*/records.done.json"
    echo "    results/frozen/I4_perturbation/evaluation/*/diag.done.json"
    echo "    results/frozen/I4_perturbation/evaluation/*/maps.npz"
    echo "    results/frozen/I4_perturbation/evaluation/*/run_meta.json"
    echo ""
    echo "  records.parquet / diag.parquet stay on the cluster — .gitignore"
    echo "  excludes them by name, and run_meta.json records their sha256"
    echo "  and byte size."
else
    echo "  FAILED: $RUN_ID (exit $RC) — read the error before resubmitting" >&2
fi
echo "=============================================================="
exit $RC
