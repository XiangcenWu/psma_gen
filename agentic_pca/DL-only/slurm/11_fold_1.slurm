#!/bin/bash
#SBATCH --job-name=dl_swin_fold1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=48:00:00
#SBATCH --output=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_swin_fold1_%j.out
#SBATCH --error=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_swin_fold1_%j.err

set -euo pipefail

PROJECT_ROOT="/share/home/xcwu/projects/psma_gen"
CACHE="${CACHE:-agentic_pca/DL-only/cache/pet_128x128x384}"
OUTPUT="${OUTPUT:-agentic_pca/retrieval_agent_inference/outputs/dl_swin3d_fdg_psma_5fold}"
MODEL="${MODEL:-swin_tiny_3d}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
WORKERS="${WORKERS:-4}"

source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

python -u agentic_pca/DL-only/train_fold.py \
  --manifest "${CACHE}/manifest.json" \
  --output-dir "${OUTPUT}" \
  --fold 1 \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --accumulation-steps "${ACCUMULATION_STEPS}" \
  --workers "${WORKERS}" \
  --device cuda
