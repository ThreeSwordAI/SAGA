#!/bin/bash
# scripts/sync_results.sh — TASK-05 A3, extended for the dense runs (TASK-09)
# Stage ONLY the small per-run artifacts and commit them, so training
# progress is inspectable locally. Run on the HPC every day or two, then
# push with:  I_AM_HUMAN=1 git push
#
# NOTE (TASK-09): results/detection/*/detections_val.json is the raw
# prediction dump — a mandated deliverable (Phase C reads it for the
# small-object crops) and the one LARGE small-artifact here. It is staged
# only once its run is COMPLETE (it is rewritten at every new-best epoch, so
# syncing it mid-run would push the same blob into history repeatedly).
# Under DENSE_DETECTIONS_MB (default 25) it goes in as-is; above that the
# gzip goes in instead and the raw one is untracked, so a run can never ship
# two dumps from different epochs. The raw file always stays on the HPC.
set -eu
cd "$(dirname "$0")/.."

DENSE_DETECTIONS_MB=${DENSE_DETECTIONS_MB:-25}

shopt -s nullglob
git add --ignore-errors \
    results/runs/*/log.csv \
    results/runs/*/meta.json \
    results/runs/*/config.resolved.yaml \
    results/runs/*/diag/*.json \
    results/runs/*/gates/*.npz \
    results/runs/*/eval/*.json \
    results/runs/*/grads/*.csv \
    results/detection/*/log.csv \
    results/detection/*/meta.json \
    results/detection/*/config.resolved.yaml \
    results/detection/*/coco_eval_best.json \
    results/segmentation/*/log.csv \
    results/segmentation/*/meta.json \
    results/segmentation/*/config.resolved.yaml \
    results/segmentation/*/miou_ss.json \
    results/segmentation/*/miou_ms.json \
    results/segmentation/*/per_class_iou.csv \
    results/segmentation/*/conf_matrix.npz \
    results/segmentation/*/preds_fixed20/*.png \
    results/segmentation/*/preds_fixed20/*.jpg \
    results/detection/*/meta.json.corrupt.* \
    results/segmentation/*/meta.json.corrupt.* \
    2>/dev/null || true

LIMIT=$(( DENSE_DETECTIONS_MB * 1048576 ))
for f in results/detection/*/detections_val.json; do
    run_id=$(basename "$(dirname "$f")")

    # Only ship the prediction dump once the run is FINISHED. It is rewritten
    # at every new-best epoch, so syncing it mid-run would commit a
    # multi-megabyte blob several times over into git history for no benefit
    # (Phase C only ever reads the final one).
    if ! python tools/dense_done.py --run "$run_id" --quiet 2>/dev/null; then
        echo "sync_results: $run_id still running -- deferring $f"
        echo "  (it is rewritten at every new best; it will be staged once the run completes)"
        continue
    fi

    bytes=$(wc -c < "$f")
    if [ "$bytes" -le "$LIMIT" ]; then
        # raw fits: make sure no stale .gz counterpart stays tracked
        git rm --cached -q --ignore-unmatch "$f.gz" 2>/dev/null || true
        rm -f "$f.gz"
        echo "sync_results: staging $f ($((bytes / 1048576)) MB)"
        git add --ignore-errors "$f" || true
    else
        # too big for git raw -> commit a gzip of it, and untrack the raw one
        # so a run can never ship two dumps from different epochs
        gzip -9 -c "$f" > "$f.gz"
        gz=$(wc -c < "$f.gz")
        git rm --cached -q --ignore-unmatch "$f" 2>/dev/null || true
        echo "sync_results: $f is $((bytes / 1048576)) MB (> ${DENSE_DETECTIONS_MB} MB)"
        echo "  -> staging $f.gz ($((gz / 1048576)) MB) instead; the raw file stays on the HPC"
        echo "     Phase C reads it with:  gzip -dc <file>.gz"
        git add --ignore-errors "$f.gz" || true
    fi
done

if git diff --cached --quiet; then
    echo "sync_results: nothing new to commit"
else
    git commit -m "[RUNS] progress sync"
    echo "sync_results: committed. Push with:  I_AM_HUMAN=1 git push"
fi
