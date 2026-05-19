#!/bin/bash
#SBATCH --job-name=svr_diff
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=log_svr_diff_%j.out
#SBATCH --error=log_svr_diff_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen

python Registration/train_baseline.py \
    --baseline_model svr_diff \
    --smoothness 4500 \
    --epochs 350 \
    --num_masks 10 \
    --lr 1e-5 \
    --device cuda:0 \
    --beta_lambda 1.0 \
    --beta_alpha 1.1 \
    --beta_beta 1.0 \
    --beta_prior_mode repo_logbeta
