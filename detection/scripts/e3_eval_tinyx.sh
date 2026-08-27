#!/bin/bash
#SBATCH --job-name=saga_e3_eval
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/e3/logs/e3_eval_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/e3/logs/e3_eval_%j.err

# ── E3 Standalone Evaluation — runs on single TinyGPU ────────────────────────
# Evaluates all 3 saved best.pth checkpoints on COCO val2017.
# No DDP, no DistributedSampler — all 4952 val images covered correctly.
# No queue wait on Alex.
#
# Submit: sbatch detection/scripts/e3_eval_tinyx.sh

source /etc/profile
module load python/3.12-conda
source activate /home/vault/iwi5/iwi5359h/envs/saga

export CODE_ROOT=/home/hpc/iwi5/iwi5359h/my_repos/SAGA
export PYTHONPATH=$CODE_ROOT:$PYTHONPATH
export PYTHONUNBUFFERED=1

COCO_ZIP=/home/woody/iwi5/iwi5359h/Data/COCO/val2017.zip
COCO_ANN=/home/woody/iwi5/iwi5359h/Data/COCO/annotations_trainval2017.zip
STAGE_DIR=/tmp/coco_e3_eval_${SLURM_JOB_ID}
CKPT_BASE=/home/vault/iwi5/iwi5359h/SAGA/e3/checkpoints

mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e3/logs
mkdir -p /home/vault/iwi5/iwi5359h/SAGA/e3/results

cd $CODE_ROOT

echo "======================================================"
echo "  E3 Standalone Evaluation — TinyGPU"
echo "  Node: $SLURMD_NODENAME"
echo "  $(date)"
echo "======================================================"

# Stage COCO val only (~1GB, ~1-2 min)
echo ""
echo "── Staging COCO val ─────────────────────────────────"
mkdir -p $STAGE_DIR
unzip -q $COCO_ANN -d $STAGE_DIR
unzip -q $COCO_ZIP -d $STAGE_DIR
echo "  Val images: $(ls $STAGE_DIR/val2017/*.jpg | wc -l) (expected 5000)"

run_eval() {
    VARIANT_ID=$1
    VARIANT_NAME=$2
    CKPT=$CKPT_BASE/$VARIANT_NAME/last.pth

    echo ""
    echo "── $VARIANT_NAME ────────────────────────────────────"

    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: checkpoint not found: $CKPT — skipping"
        return
    fi

    echo "  Checkpoint: $CKPT"
    echo "  Start: $(date)"

    python3 detection/tools/evaluate.py \
        --config    detection/configs/variants.yaml \
        --id        $VARIANT_ID \
        --paths     detection/configs/paths.yaml \
        --ckpt      $CKPT \
        --data_root $STAGE_DIR \
        --split     val \
        --batch_size 4 \
        --num_workers 4

    echo "  Done: $(date)"
}

run_eval 0 ViT-B_baseline_det
run_eval 1 ViT-B_registers_det
run_eval 2 ViT-B_SAGA_det

# Print summary
echo ""
echo "======================================================"
echo "  E3 Evaluation Summary"
echo "======================================================"
python3 -c "
import json
from pathlib import Path

res_dir = Path('/home/vault/iwi5/iwi5359h/SAGA/e3/results')
names = ['ViT-B_baseline_det', 'ViT-B_registers_det', 'ViT-B_SAGA_det']
print(f'  {\"Variant\":<28} {\"AP\":>6} {\"AP50\":>6} {\"AP75\":>6} {\"AP_S\":>6} {\"AP_M\":>6} {\"AP_L\":>6}')
print(f'  {\"-\"*70}')
for name in names:
    f = res_dir / f'{name}_eval_val.json'
    if f.exists():
        d = json.load(open(f))
        print(f'  {name:<28} {d[\"AP\"]:>6.2f} {d[\"AP50\"]:>6.2f} '
              f'{d[\"AP75\"]:>6.2f} {d[\"AP_S\"]:>6.2f} '
              f'{d[\"AP_M\"]:>6.2f} {d[\"AP_L\"]:>6.2f}')
    else:
        print(f'  {name:<28}  no results yet')
"

# Cleanup
echo ""
rm -rf $STAGE_DIR
echo "  Cleaned up: $STAGE_DIR"
echo ""
echo "======================================================"
echo "  Done — $(date)"
echo "======================================================"