#!/bin/bash
#SBATCH --job-name=dl_pet_prepare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_pet_prepare_%j.out
#SBATCH --error=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_pet_prepare_%j.err

set -euo pipefail

PROJECT_ROOT="/share/home/xcwu/projects/psma_gen"
CACHE="${CACHE:-agentic_pca/DL-only/cache/pet_128x128x384}"
WORKERS="${WORKERS:-4}"

source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Preparing paired PET cache"
echo "cache=${CACHE}"
echo "workers=${WORKERS}"

python -u agentic_pca/DL-only/prepare_dataset.py \
  --cache-dir "${CACHE}" \
  --workers "${WORKERS}"
