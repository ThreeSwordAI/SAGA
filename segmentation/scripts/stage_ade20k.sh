#!/bin/bash
# segmentation/scripts/stage_ade20k.sh
# ─────────────────────────────────────────────────────────────
# Source this in SLURM scripts — provides stage_ade20k()
# and cleanup_ade20k() functions.
# ─────────────────────────────────────────────────────────────

stage_ade20k() {
    echo "======================================================"
    echo "  Staging ADE20K to $STAGE_DIR"
    echo "  $(date)"
    echo "======================================================"

    mkdir -p $STAGE_DIR

    echo "  Extracting ADEChallengeData2016.zip (~900MB)..."
    unzip -q $ADE_ZIP -d $STAGE_DIR
    echo "  Extraction done."

    # Verify counts
    N_TRAIN=$(find $STAGE_DIR/ADEChallengeData2016/images/training \
              -name "*.jpg" | wc -l)
    N_VAL=$(find $STAGE_DIR/ADEChallengeData2016/images/validation \
            -name "*.jpg" | wc -l)

    echo "  training: $N_TRAIN images (expected: 20210)"
    echo "  validation: $N_VAL images (expected: 2000)"

    if [ "$N_VAL" -lt 1900 ]; then
        echo "ERROR: ADE20K extraction incomplete ($N_VAL val images)"
        exit 1
    fi

    # Set DATA_ROOT for use in training scripts
    export DATA_ROOT=$STAGE_DIR/ADEChallengeData2016

    echo "  Staging complete — $(date)"
    echo "======================================================"
}

cleanup_ade20k() {
    echo "======================================================"
    echo "  Cleaning up $STAGE_DIR"
    rm -rf $STAGE_DIR
    echo "  Cleanup done — $(date)"
    echo "======================================================"
}