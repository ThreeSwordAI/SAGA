#!/bin/bash
# ── Stage ImageNet from janus to /scratch ─────────────────────────────────────
# Call this function at the start of every training job.
# Extracts 5 train shards + val shard, then organizes into class subfolders.
# Runtime: ~20-40 minutes depending on /scratch speed.
#
# HuggingFace format: files named {image_id}_{synset_id}.JPEG  (synset is LAST)
# After staging:
#   $STAGE_DIR/train/n01440764/xxx_n01440764.JPEG
#   $STAGE_DIR/val/n01440764/xxx_n01440764.JPEG
# ─────────────────────────────────────────────────────────────────────────────

stage_imagenet() {
    echo "======================================================"
    echo "  Staging ImageNet to $STAGE_DIR"
    echo "  Source: $JANUS_DATA"
    echo "  $(date)"
    echo "======================================================"

    mkdir -p $STAGE_DIR/train $STAGE_DIR/val

    # Extract 5 train shards in parallel (one process per shard)
    echo "  Extracting 5 train shards in parallel..."
    ls -1 $JANUS_DATA/train_images_*.tar.gz | \
        xargs -P 5 -I{} tar xzf {} -C $STAGE_DIR/train

    echo "  Extracting val shard..."
    tar xzf $JANUS_DATA/val_images.tar.gz -C $STAGE_DIR/val

    # Organize flat files into synset subfolders
    # HuggingFace naming: {image_id}_{synset_id}.JPEG → synset is after last _
    echo "  Organizing into class folders..."
    python3 - << 'PYEOF'
import os, shutil, glob, sys

stage = os.environ['STAGE_DIR']

for split in ['train', 'val']:
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
        if (i + 1) % 50000 == 0:
            print(f'    {split}: {i+1}/{len(files)} done', flush=True)

    n_classes = len([d for d in os.listdir(folder)
                     if os.path.isdir(os.path.join(folder, d))])
    print(f'  {split}: {n_classes} classes organized', flush=True)

print('  Staging complete.', flush=True)
PYEOF

    if [ $? -ne 0 ]; then
        echo "ERROR: Staging failed. Cleaning up."
        rm -rf $STAGE_DIR
        exit 1
    fi

    echo "  Train classes: $(ls $STAGE_DIR/train | wc -l)  (expected: 1000)"
    echo "  Val classes:   $(ls $STAGE_DIR/val   | wc -l)  (expected: 1000)"
    echo "  $(date)"
    echo "======================================================"
}

cleanup_imagenet() {
    echo "======================================================"
    echo "  Cleaning up $STAGE_DIR"
    rm -rf $STAGE_DIR
    echo "  Cleanup done — $(date)"
    echo "======================================================"
}