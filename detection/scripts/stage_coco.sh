#!/bin/bash
# detection/scripts/stage_coco.sh
# ─────────────────────────────────────────────────────────────
# Source this in SLURM scripts — provides stage_coco() and
# cleanup_coco() functions.
# ─────────────────────────────────────────────────────────────

stage_coco() {
    echo "======================================================"
    echo "  Staging COCO to $STAGE_DIR"
    echo "  $(date)"
    echo "======================================================"

    mkdir -p $STAGE_DIR

    # Annotations (~240MB — always extract)
    echo "  Extracting annotations..."
    unzip -q $COCO_ANN_ZIP -d $STAGE_DIR
    echo "  Annotations done."

    # Val images (~1GB — always extract for evaluation)
    echo "  Extracting val2017..."
    unzip -q $COCO_VAL_ZIP -d $STAGE_DIR
    echo "  Val done."

    # Train images (~18GB)
    echo "  Extracting train2017 (18GB — may take 10-15 min)..."
    unzip -q $COCO_TRAIN_ZIP -d $STAGE_DIR
    echo "  Train done."

    # Verify
    N_TRAIN=$(find $STAGE_DIR/train2017 -name "*.jpg" 2>/dev/null | wc -l)
    N_VAL=$(find $STAGE_DIR/val2017 -name "*.jpg" 2>/dev/null | wc -l)
    echo "  train2017: $N_TRAIN images (expected: 118287)"
    echo "  val2017:   $N_VAL images (expected: 5000)"

    if [ "$N_VAL" -lt 4900 ]; then
        echo "ERROR: val2017 extraction incomplete ($N_VAL images)"
        exit 1
    fi

    echo "  Staging complete — $(date)"
    echo "======================================================"
}

cleanup_coco() {
    echo "======================================================"
    echo "  Cleaning up $STAGE_DIR"
    rm -rf $STAGE_DIR
    echo "  Cleanup done — $(date)"
    echo "======================================================"
}