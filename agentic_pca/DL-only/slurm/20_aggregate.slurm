#!/bin/bash
#SBATCH --job-name=dl_pet_aggregate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_pet_aggregate_%j.out
#SBATCH --error=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_pet_aggregate_%j.err

set -euo pipefail

PROJECT_ROOT="/share/home/xcwu/projects/psma_gen"
CACHE="${CACHE:-agentic_pca/DL-only/cache/pet_128x128x384}"
OUTPUT="${OUTPUT:-agentic_pca/retrieval_agent_inference/outputs/dl_swin3d_fdg_psma_5fold}"

source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u agentic_pca/DL-only/aggregate_results.py \
  --manifest "${CACHE}/manifest.json" \
  --output-dir "${OUTPUT}"
