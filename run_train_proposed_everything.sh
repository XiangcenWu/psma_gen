#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=log_%A_%a.out
#SBATCH --error=log_%A_%a.err
#SBATCH --array=0-8   # 3 smooth × 3 gamma = 9 jobs


# ---------------- hyperparameters ----------------
SMOOTH_LIST=(5000 8000 12000)
GAMMA_LIST=(1.2 1.5 2)

NUM_GAMMA=${#GAMMA_LIST[@]}

# Decode which pair this job should run
SMOOTH_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_GAMMA))
GAMMA_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_GAMMA))

CURRENT_SMOOTH=${SMOOTH_LIST[$SMOOTH_INDEX]}
CURRENT_GAMMA=${GAMMA_LIST[$GAMMA_INDEX]}


# ---------------- environment ----------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen


# ---------------- info ----------------
echo "======================================"
echo "SLURM TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Smoothness: $CURRENT_SMOOTH"
echo "Gamma: $CURRENT_GAMMA"
echo "======================================"


# ---------------- run training ----------------
python Registration/train.py \
    --smoothness $CURRENT_SMOOTH \
    --epochs 350 \
    --lr 1e-5 \
    --num_masks 10 \
    --ct_smoothness \
    --ct_smoothness_margin 3000 \
    --ct_smoothness_gamma $CURRENT_GAMMA
