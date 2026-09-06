#!/bin/bash
#SBATCH --job-name=e2r_smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/e2r_smoke_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/e2r_smoke_%j.err
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh
cd $CODE_ROOT
stage_imagenet
/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29699 \
    classification/tools/train.py \
        --matrix configs/e2r_matrix.yaml --run e2r_vits_mixup_saga_s1 \
        --data_root $STAGE_DIR --resume auto \
        --out_root results/runs_smoke --max_epochs 2
STATUS=$?
cleanup_imagenet
exit $STATUS