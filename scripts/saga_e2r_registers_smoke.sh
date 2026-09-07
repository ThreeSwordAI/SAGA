#!/bin/bash
#SBATCH --job-name=e2r_reg_smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/vault/iwi5/iwi5359h/SAGA/logs/e2r_reg_smoke_%j.log
#SBATCH --error=/home/vault/iwi5/iwi5359h/SAGA/logs/e2r_reg_smoke_%j.err
# TASK-07 A2: registers has NEVER run through classification/tools/train.py.
# This 2-epoch smoke must PASS (and its contract files must be checked)
# BEFORE either registers chain is submitted. Mirrors saga_e2r_smoke.sh.
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/env_alex.sh
source /home/hpc/iwi5/iwi5359h/my_repos/SAGA/scripts/stage_imagenet.sh
cd $CODE_ROOT
stage_imagenet
/home/vault/iwi5/iwi5359h/envs/saga/bin/python -m torch.distributed.run \
    --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29698 \
    classification/tools/train.py \
        --matrix configs/e2r_matrix.yaml --run e2r_vits_mixup_registers_s1 \
        --data_root $STAGE_DIR --resume auto \
        --out_root results/runs_smoke --max_epochs 2
STATUS=$?
cleanup_imagenet
exit $STATUS
