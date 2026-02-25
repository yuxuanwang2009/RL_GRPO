#!/bin/bash
#SBATCH --job-name=eval_os3600_zs
#SBATCH --output=eval_os3600_zs_verbose_%j.out
#SBATCH --error=eval_os3600_zs_verbose_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

module load conda
conda activate grpo

cd /home/yuxuan.wang/RL_GRPO

# Dump every raw completion for val 3num zero-shot to diagnose eval collapse.

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/eval_compare.py \
    --models "os3600:os3600" \
    --zeroshot_only \
    --verbose \
    --num_problems 100 \
    --seed 142
