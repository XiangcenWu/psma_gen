#!/bin/bash
#SBATCH --job-name=pet_trajectory_rag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=/share/home/xcwu/projects/psma_gen/agentic_pca/retrieval_agent_inference/logs/pet_trajectory_rag_%j.out
#SBATCH --error=/share/home/xcwu/projects/psma_gen/agentic_pca/retrieval_agent_inference/logs/pet_trajectory_rag_%j.err

set -euo pipefail

source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd /share/home/xcwu/projects/psma_gen

TRAJECTORY_RAG_DIR="${TRAJECTORY_RAG_DIR:-agentic_pca/retrieval_agent_inference/outputs/full_run}"
OUTPUT_DIR="${OUTPUT_DIR:-agentic_pca/retrieval_agent_inference/outputs/trajectory_rag_full_run}"
TOP_K="${TOP_K:-5}"
MAX_PER_CASE="${MAX_PER_CASE:-1}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-5}"
MAX_JSON_RETRIES="${MAX_JSON_RETRIES:-3}"
INVALID_REPORT_POLICY="${INVALID_REPORT_POLICY:-warn}"
MAX_PATIENT_ARGS=()

if [[ -n "${MAX_PATIENTS:-}" ]]; then
  MAX_PATIENT_ARGS=(--max-patients "${MAX_PATIENTS}")
fi

python -u agentic_pca/retrieval_agent_inference/infer.py \
  --device cuda \
  --trajectory-rag-dir "${TRAJECTORY_RAG_DIR}" \
  --trajectory-rag-top-k "${TOP_K}" \
  --trajectory-rag-max-per-case "${MAX_PER_CASE}" \
  --num-trajectories "${NUM_TRAJECTORIES}" \
  --max-json-retries "${MAX_JSON_RETRIES}" \
  --invalid-report-policy "${INVALID_REPORT_POLICY}" \
  --temperature 0.7 \
  --top-p 0.9 \
  --output-dir "${OUTPUT_DIR}" \
  "${MAX_PATIENT_ARGS[@]}"
