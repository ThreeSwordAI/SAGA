#!/bin/bash
# scripts/stage_probe.sh
# ─────────────────────────────────────────────────────────────
# Source this in the TASK-11 probe job — provides
# stage_probe_imagenet_val(), stage_probe_coco_val() and cleanup_probe().
#
# WHY THIS EXISTS instead of the committed stagers: the probe set touches
# ONLY validation images, but scripts/stage_imagenet.sh extracts 5 train
# shards (~20-40 min) and detection/scripts/stage_coco.sh extracts
# train2017.zip (~18 GB). Both bodies below are COPIED from those files with
# the train lines removed and the stage directory renamed — no new staging
# logic, and the val halves stay line-for-line comparable with their originals.
#
# Separate per-dataset stage dirs (both keyed by SLURM_JOB_ID, the committed
# convention) so ImageNet and COCO can be staged in the SAME job without
# either overwriting $STAGE_DIR.
# ─────────────────────────────────────────────────────────────

export PROBE_IN_DIR=${PROBE_IN_DIR:-/scratch/iwi5359h/probe_imagenet_${SLURM_JOB_ID}}
export PROBE_COCO_DIR=${PROBE_COCO_DIR:-/scratch/iwi5359h/probe_coco_${SLURM_JOB_ID}}

# COCO zip paths — the same three exported by detection/scripts/env_alex.sh
export COCO_VAL_ZIP=${COCO_VAL_ZIP:-/home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip}
export COCO_ANN_ZIP=${COCO_ANN_ZIP:-/home/woody/iwi5/iwi5359h/Data/COCO/annotations_trainval2017.zip}


stage_probe_imagenet_val() {
    echo "======================================================"
    echo "  Staging ImageNet VAL ONLY to $PROBE_IN_DIR"
    echo "  Source: $JANUS_DATA"
    echo "  $(date)"
    echo "======================================================"

    mkdir -p $PROBE_IN_DIR/val

    echo "  Extracting val shard..."
    tar xzf $JANUS_DATA/val_images.tar.gz -C $PROBE_IN_DIR/val

    # Organize flat files into synset subfolders.
    # Copied from scripts/stage_imagenet.sh; the only changes are the env var
    # (STAGE_DIR -> PROBE_IN_DIR) and the split list (['train','val'] -> ['val']).
    echo "  Organizing into class folders..."
    python3 - << 'PYEOF'
import os, shutil, glob

stage = os.environ['PROBE_IN_DIR']

for split in ['val']:
    folder = os.path.join(stage, split)
    files  = glob.glob(os.path.join(folder, '*.JPEG'))
    print(f'  {split}: {len(files)} images found', flush=True)

    for i, f in enumerate(files):
        bn       = os.path.basename(f)
        root     = os.path.splitext(bn)[0]       # remove .JPEG
        synset   = root.rsplit('_', 1)[1]        # synset is after last underscore
        dest_dir = os.path.join(folder, synset)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(f, os.path.join(dest_dir, bn))
        if (i + 1) % 10000 == 0:
            print(f'    {split}: {i+1}/{len(files)} done', flush=True)

    n_classes = len([d for d in os.listdir(folder)
                     if os.path.isdir(os.path.join(folder, d))])
    print(f'  {split}: {n_classes} classes organized', flush=True)

print('  Staging complete.', flush=True)
PYEOF

    if [ $? -ne 0 ]; then
        echo "ERROR: ImageNet val staging failed."
        exit 1
    fi

    N_CLASSES=$(ls $PROBE_IN_DIR/val | wc -l)
    N_VAL=$(find $PROBE_IN_DIR/val -name "*.JPEG" | wc -l)
    echo "  Val classes: $N_CLASSES  (expected: 1000)"
    echo "  Val images:  $N_VAL      (expected: 50000)"
    if [ "$N_CLASSES" -ne 1000 ] || [ "$N_VAL" -ne 50000 ]; then
        echo "ERROR: incomplete ImageNet val extraction — the frozen probe set"
        echo "       must never be built from a partial directory listing."
        exit 1
    fi
    echo "  $(date)"
    echo "======================================================"
}


stage_probe_coco_val() {
    echo "======================================================"
    echo "  Staging COCO VAL ONLY to $PROBE_COCO_DIR"
    echo "  $(date)"
    echo "======================================================"

    mkdir -p $PROBE_COCO_DIR

    # Both lines copied from detection/scripts/stage_coco.sh (train2017 omitted)
    echo "  Extracting annotations..."
    unzip -q $COCO_ANN_ZIP -d $PROBE_COCO_DIR
    echo "  Extracting val2017..."
    unzip -q $COCO_VAL_ZIP -d $PROBE_COCO_DIR

    N_VAL=$(find $PROBE_COCO_DIR/val2017 -name "*.jpg" 2>/dev/null | wc -l)
    echo "  val2017: $N_VAL images (expected: 5000)"
    if [ "$N_VAL" -ne 5000 ]; then
        echo "ERROR: val2017 extraction incomplete ($N_VAL images)"
        exit 1
    fi
    if [ ! -f "$PROBE_COCO_DIR/annotations/instances_val2017.json" ]; then
        echo "ERROR: instances_val2017.json missing after extraction"
        exit 1
    fi
    echo "  Staging complete — $(date)"
    echo "======================================================"
}


cleanup_probe() {
    echo "======================================================"
    echo "  Cleaning up probe stage dirs"
    rm -rf $PROBE_IN_DIR $PROBE_COCO_DIR
    echo "  Cleanup done — $(date)"
    echo "======================================================"
}
