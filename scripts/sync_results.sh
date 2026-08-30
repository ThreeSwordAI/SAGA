#!/bin/bash
# scripts/sync_results.sh — TASK-05 A3
# Stage ONLY the small per-run artifacts of the e2r runs and commit them,
# so training progress is inspectable locally. Run on the HPC every day or
# two, then push with:  I_AM_HUMAN=1 git push
set -eu
cd "$(dirname "$0")/.."

shopt -s nullglob
git add --ignore-errors \
    results/runs/*/log.csv \
    results/runs/*/meta.json \
    results/runs/*/config.resolved.yaml \
    results/runs/*/diag/*.json \
    results/runs/*/gates/*.npz \
    results/runs/*/eval/*.json \
    results/runs/*/grads/*.csv \
    2>/dev/null || true

if git diff --cached --quiet; then
    echo "sync_results: nothing new to commit"
else
    git commit -m "[RUNS] progress sync"
    echo "sync_results: committed. Push with:  I_AM_HUMAN=1 git push"
fi
