#!/bin/bash
#SBATCH --job-name=latent_train_infer_100
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=latent_train_infer_100_%j.out
#SBATCH --error=latent_train_infer_100_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen

MODEL_DIR=/data1/xiangcen/models/generation_latent_diffusion/psma_ct_warped_fdg_pet_to_psma_pt_128x128x384_100
OUTPUT_DIR=~/results/generation_latent_diffusion/psma_ct_warped_fdg_pet_to_psma_pt_128x128x384_100
DATA_DIR=/data2/xiangcen/data/pet_gen/data_h5_100/h5_data_100/

echo ">>> Starting latent diffusion training"
python Generation/latent_diffusion.py \
    --stage all \
    --input-key psma_ct \
    --target-key psma_pt \
    --fdg-condition warped_fdg \
    --image-size 128 128 384 \
    --target-latent-channels 4 \
    --condition-latent-channels 4 \
    --target-latent-scale 1.0 \
    --condition-latent-scale 1.0 \
    --vae-channels 8 16 32 64 128 \
    --diffusion-channels 32 64 128 \
    --vae-epochs 100 \
    --diffusion-epochs 500 \
    --vae-lr 1e-4 \
    --diffusion-lr 1e-4 \
    --kl-weight 1e-6 \
    --num-train-timesteps 1000 \
    --amp \
    --batch-size 1 \
    --data-dirs ${DATA_DIR} \
    --val-counts 20 \
    --save-dir ${MODEL_DIR} \
    --device cuda:0

echo ">>> Training completed; starting inference"
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
    --data-dirs ${DATA_DIR} \
    --val-counts 20 \
    --device cuda:0

echo ">>> Training and inference completed"
