#!/bin/bash
#SBATCH --job-name=pet_rag_gpt_oss
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=/share/home/xcwu/projects/psma_gen/agentic_pca/retrieval_agent_inference/logs/pet_rag_gpt_oss_%j.out
#SBATCH --error=/share/home/xcwu/projects/psma_gen/agentic_pca/retrieval_agent_inference/logs/pet_rag_gpt_oss_%j.err

set -euo pipefail

source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd /share/home/xcwu/projects/psma_gen

MODEL_PATH="llm_models/gpt-oss-20b"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-5}"
MAX_JSON_RETRIES="${MAX_JSON_RETRIES:-3}"
INVALID_REPORT_POLICY="${INVALID_REPORT_POLICY:-fail}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-agentic_pca/retrieval_agent_inference/outputs/gpt_oss_${TIMESTAMP}}"

MAX_PATIENT_ARGS=()
if [[ -n "${MAX_PATIENTS:-}" ]]; then
  MAX_PATIENT_ARGS=(--max-patients "${MAX_PATIENTS}")
fi

python -u agentic_pca/retrieval_agent_inference/infer.py \
  --model-path "${MODEL_PATH}" \
  --device cuda \
  --num-trajectories "${NUM_TRAJECTORIES}" \
  --max-json-retries "${MAX_JSON_RETRIES}" \
  --invalid-report-policy "${INVALID_REPORT_POLICY}" \
  --temperature 0.7 \
  --top-p 0.9 \
  --output-dir "${OUTPUT_DIR}" \
  "${MAX_PATIENT_ARGS[@]}"
