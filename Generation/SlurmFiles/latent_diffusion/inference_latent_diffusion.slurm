#!/bin/bash
#SBATCH --job-name=latent_inference
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=latent_inference_%j.out
#SBATCH --error=latent_inference_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen

MODEL_DIR=/data1/xiangcen/models/generation_latent_diffusion/psma_ct_warped_fdg_pet_to_psma_pt_128x128x384
OUTPUT_DIR=~/results/generation_latent_diffusion/psma_ct_warped_fdg_pet_to_psma_pt_128x128x384

python Generation/inference_latent_diffusion.py \
    --input-key psma_ct \
    --target-key psma_pt \
    --fdg-condition warped_fdg \
    --image-size 128 128 384 \
    --target-vae-checkpoint ${MODEL_DIR}/target_vae_psma_pt.pth \
    --condition-vae-checkpoint ${MODEL_DIR}/condition_vae_psma_ct_warped_fdg_pet.pth \
    --diffusion-checkpoint ${MODEL_DIR}/latent_psma_ct_warped_fdg_pet_to_psma_pt.pth \
    --output-dir ${OUTPUT_DIR} \
    --num-inference-steps 1000 \
    --batch-size 1 \
    --amp \
    --data-dirs \
        /data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch1_h5_v2 \
        /data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch2_h5_v2 \
        /data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5/batch3_h5_v2 \
    --val-counts 40 40 20 \
    --device cuda:0
