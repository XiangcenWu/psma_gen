#!/bin/bash
#SBATCH --job-name=voxelmorph_ctsmooth
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=log_voxelmorph_ctsmooth_%j.out
#SBATCH --error=log_voxelmorph_ctsmooth_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen

python Registration/train.py \
    --registration_model voxelmorph \
    --smoothness 7000 \
    --epochs 350 \
    --num_masks 10 \
    --lr 1e-5 \
    --ct_smoothness \
    --ct_smoothness_margin 3000 \
    --ct_smoothness_gamma 2.0 \
    --device cuda:0 \
    # --use_ct_input
