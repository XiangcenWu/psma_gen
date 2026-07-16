#!/bin/bash
#SBATCH --job-name=ddpm_train_warped_fdg_100
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=ddpm_warped_fdg_100_%j.out
#SBATCH --error=ddpm_warped_fdg_100_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen

python Generation/train_ddpm.py \
    --input-key psma_ct \
    --target-key psma_pt \
    --fdg-condition warped_fdg \
    --data-dirs /data2/xiangcen/data/pet_gen/data_h5_100/h5_data_100/ \
    --val-counts 20 \
    --epochs 500 \
    --batch-size 8 \
    --lr 1e-4 \
    --save-dir /data1/xiangcen/models/generation_ddpm/psma_ct_to_psma_pt_100 \
    --device cuda:0
