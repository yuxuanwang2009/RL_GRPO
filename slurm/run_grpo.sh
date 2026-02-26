#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=grpo_%j.out
#SBATCH --error=grpo_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --time=07:00:00
#SBATCH --ntasks=1

module load conda
conda activate grpo

cd /home/yuxuan.wang/RL_GRPO
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/train.py --mix_oneshot 1 --run_name hs3600 --num_iterations 3600 --eval_every 50
