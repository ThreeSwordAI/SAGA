#!/bin/bash
#SBATCH --job-name=saga_frozen_smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_smoke_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/saga_frozen_smoke_%j.err

# TASK I0 D5 — the nine correctness checks of plan section 13.3 on a fixed
# 32-image subset, on three checkpoints: one ViT-S/mixup baseline s1, its
# matched SAGA s1, and the legacy ViT-S registers checkpoint. The register
# model is the one that matters for the prefix-row check: it has FIVE prefix
# rows, not one, and a patch intervention that slid a row would be invisible
# on a CLS-only model.
#
# Exit 0 = every check PASSed on all three. Exit 3 = at least one FAILed.
#
# ── HEADER PROVENANCE ────────────────────────────────────────────────────────
# `set -u` AFTER the env sourcing, and an interpreter preflight that is
# actually tested, copied from job files that have COMPLETED real runs
# (TASK-08 ft_finegrained_array.sbatch, TASK-10 ttr_validate.sbatch) — never
# from probe_attention.sbatch's ORIGINAL header, whose `set -u` sat above the
# sourcing and killed every job that used it (docs/HPC_WORKFLOW.md, TASK-10
# failure #2). --partition=a40 / --gres=gpu:a40:1 / --cpus-per-task=4 are the
# strings recorded in docs/HPC_WORKFLOW.md "Partitions". a40 is plenty: this
# is 32 images x 3 checkpoints, and the two hours are dominated by staging
# ImageNet val (~10 min per How to Run.md section 5). The a40 wall limit is
# not documented — lower --time if the partition refuses it. If a40 is busy,
# switch on the COMMAND LINE rather than copying this file:
#     sbatch --partition=a100 --gres=gpu:a100:1 scripts/jobs/frozen_smoke.sbatch
#
# ── WHY THE SPLIT BUILD IS IN HERE AND NOT ON A LOGIN NODE ───────────────────
# TASK I0 section 9 puts `tools/build_frozen_splits.py` on the login node. It
# cannot run there: the builder needs an EXTRACTED `val/<synset>/*.JPEG` tree
# to list, and this cluster keeps ImageNet only as tarballs on janus
# ($JANUS_DATA) which every job stages to node-local /scratch (How to Run.md
# section 5; there is no persistent extracted copy). So the build happens
# here, immediately after staging, with --if-missing so a resubmit is a
# no-op. Everything else in section 9 is unchanged.

source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
# TASK-11's committed val-only stager: ImageNet val to node-local scratch,
# with a 1000-class / 50000-image assertion. stage_imagenet.sh would also
# pull the five train shards — 10x the copy for a job that never reads them.
source $CODE_ROOT/scripts/stage_probe.sh
cd $CODE_ROOT

# `set -u` ONLY AFTER the env sourcing. env_alex.sh sources /etc/profile,
# whose site scripts dereference unset variables (debuginfod.sh line 8,
# DEBUGINFOD_URLS); under `set -u` bash aborts the CALLING script and not one
# line of the job body runs.
set -u

# ── Interpreter preflight: FATAL, and BEFORE any staging ─────────────────────
# Never `which python`: if the module load fails, PATH still resolves to
# /usr/bin/python, which has no torch. An unchecked import cost TASK-10 a job
# that staged 50 000 images and then died in Python.
PY=${SAGA_PY:-/home/vault/iwi5/iwi5359h/envs/saga/bin/python}
echo "python: $PY"
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

# The manifest must already carry checkpoint hashes: frozen_smoke.py refuses
# a row whose ckpt_sha256 is MISSING, and hashing ~90 multi-GB checkpoints is
# a login-node job, not a GPU one.
MANIFEST=results/frozen/I0_manifest/manifest.json
if ! $PY -c "
import json, sys
rows = json.load(open('$MANIFEST'))['rows']
need = {'e2r_vits_mixup_baseline_s1', 'e2r_vits_mixup_saga_s1'}
bad = [r['run_id'] for r in rows
       if r['run_id'] in need and r['ckpt_kind'] == 'last'
       and r['ckpt_sha256'] == 'MISSING']
sys.exit(1 if bad else 0)
"; then
    echo "ERROR: $MANIFEST still has ckpt_sha256=MISSING for the smoke's" >&2
    echo "       checkpoints. Run these two on the LOGIN NODE first:" >&2
    echo "         python tools/frozen_manifest_hashes.py" >&2
    echo "         python analysis/build_I0_manifest.py \\" >&2
    echo "             --hashes results/frozen/I0_manifest/ckpt_hashes.json" >&2
    exit 1
fi

# Release node-local scratch on EVERY exit path, including a kill at the wall
# clock (TASK-09's lesson: a plain call at the end only runs on a normal exit).
trap cleanup_probe EXIT

stage_probe_imagenet_val

# ── D2: freeze the splits (write-once; --if-missing makes a resubmit a no-op)
echo ""
echo "=============================================================="
echo "  build_frozen_splits (write-once)"
echo "=============================================================="
$PY tools/build_frozen_splits.py --data "$PROBE_IN_DIR" --if-missing
SPLIT_RC=$?
if [ $SPLIT_RC -ne 0 ]; then
    echo "ERROR: split build failed (exit $SPLIT_RC) — refusing to run the" >&2
    echo "       smoke against a split that does not exist." >&2
    exit $SPLIT_RC
fi

# ── D5: the nine checks, on the three checkpoints of TASK I0 section 7 ───────
RUN_IDS=(
    "e2r_vits_mixup_baseline_s1"                 # ViT-S/mixup baseline, seeded s1
    "e2r_vits_mixup_saga_s1"                     # its matched SAGA, seeded s1
    "legacy_e2_vit_small_mixupdir_registers"     # 5 prefix tokens
)

STATUS=0
for RUN_ID in "${RUN_IDS[@]}"; do
    echo ""
    echo "=============================================================="
    echo "  frozen_smoke: $RUN_ID"
    echo "=============================================================="
    $PY tools/frozen_smoke.py \
        --run-id "$RUN_ID" \
        --data "$PROBE_IN_DIR" \
        --split results/frozen/splits/evaluation.json \
        --device cuda --seed 0
    RC=$?
    echo "  -> exit $RC  (0 = all PASS, 3 = at least one FAIL)"
    if [ $RC -ne 0 ]; then STATUS=3; fi
done

echo ""
echo "=============================================================="
if [ $STATUS -eq 0 ]; then
    echo "  ALL SMOKE CHECKS PASSED on all three checkpoints"
else
    echo "  SMOKE FAILED — read the measured quantities before anything else" >&2
fi
echo ""
echo "  COMMIT AND PUSH these, then stop:"
echo "    results/frozen/splits/*.json"
echo "    results/frozen/splits/README.md"
echo "    results/frozen/I0_manifest/ckpt_hashes.json"
echo "    results/frozen/I0_manifest/manifest.csv"
echo "    results/frozen/I0_manifest/manifest.json"
echo "    results/frozen/I0_manifest/eligibility.md"
echo "    results/frozen/I0_manifest/smoke_*.json"
echo "=============================================================="
exit $STATUS
