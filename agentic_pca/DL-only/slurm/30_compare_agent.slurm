#!/bin/bash
#SBATCH --job-name=dl_vs_agent
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_vs_agent_%j.out
#SBATCH --error=/share/home/xcwu/projects/psma_gen/agentic_pca/DL-only/logs/dl_vs_agent_%j.err

set -euo pipefail

PROJECT_ROOT="/share/home/xcwu/projects/psma_gen"
OUTPUT="${OUTPUT:-agentic_pca/retrieval_agent_inference/outputs/dl_swin3d_fdg_psma_5fold}"
AGENT_OUTPUT="${AGENT_OUTPUT:-agentic_pca/retrieval_agent_inference/outputs/trajectory_rag_full_run_Qwen3.59BonQwen3.59B}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-${OUTPUT}/agent_comparison_qwen35_9b.json}"

source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u agentic_pca/DL-only/compare_with_agent.py \
  --dl-output "${OUTPUT}" \
  --agent-output "${AGENT_OUTPUT}" \
  --output "${COMPARISON_OUTPUT}"

