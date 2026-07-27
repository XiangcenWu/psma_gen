#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: run_direct.sh BASELINE_ID" >&2
  exit 2
fi

BASELINE_ID="$1"
case "${BASELINE_ID}" in
  clinical_only|suv_only|clinical_suv_no_retrieval) ;;
  *)
    echo "Unknown baseline ID: ${BASELINE_ID}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/xcwu/projects/psma_gen}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/llm_models/Qwen3.5-9B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/agentic_pca/retrieval_agent_inference/outputs/comparison_baselines_Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${BASELINE_ID}}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-5}"
MAX_JSON_RETRIES="${MAX_JSON_RETRIES:-3}"
MAX_PATIENT_ARGS=()
MOCK_ARGS=()

if [[ -n "${MAX_PATIENTS:-}" ]]; then
  MAX_PATIENT_ARGS=(--max-patients "${MAX_PATIENTS}")
fi
if [[ "${MOCK_MODEL:-0}" == "1" ]]; then
  MOCK_ARGS=(--mock-model --mock-invalid-first)
fi

source /share/home/xcwu/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

echo "baseline_id=${BASELINE_ID}"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-local}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "model_path=${MODEL_PATH}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u \
  agentic_pca/retrieval_agent_inference/comparison_baselines/infer_baseline.py \
  --baseline-id "${BASELINE_ID}" \
  --model-path "${MODEL_PATH}" \
  --device cuda \
  --output-dir "${OUTPUT_DIR}" \
  --num-trajectories "${NUM_TRAJECTORIES}" \
  --max-json-retries "${MAX_JSON_RETRIES}" \
  --invalid-report-policy warn \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 20260725 \
  "${MAX_PATIENT_ARGS[@]}" \
  "${MOCK_ARGS[@]}"
