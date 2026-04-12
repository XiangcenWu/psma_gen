#!/bin/bash
#SBATCH --job-name=ddpm_train
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=ddpm_%j.out
#SBATCH --error=ddpm_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen

python Generation/train_ddpm.py \
    --data-dirs /data2/xiangcen/data/pet_gen/processed/batch1_h5_v2 /data2/xiangcen/data/pet_gen/processed/batch2_h5_v2 \
    --val-counts 40 40 \
    --input-key psma_ct \
    --target-key psma_pt \
    --epochs 500 \
    --batch-size 8 \
    --lr 1e-4 \
    --save-dir /data1/xiangcen/models/generation_ddpm/psma_ct_to_psma_pt \
    --device cuda:0
