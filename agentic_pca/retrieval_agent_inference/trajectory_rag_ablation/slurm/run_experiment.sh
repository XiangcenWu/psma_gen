#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: run_experiment.sh EXPERIMENT_ID" >&2
  exit 2
fi

EXPERIMENT_ID="$1"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/xcwu/projects/psma_gen}"
SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/agentic_pca/retrieval_agent_inference/outputs/full_run_Qwen3.5_9B}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/llm_models/Qwen3.5-9B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/agentic_pca/retrieval_agent_inference/outputs/trajectory_rag_ablation_Qwen3.5-9B_on_Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ID}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-5}"
MAX_JSON_RETRIES="${MAX_JSON_RETRIES:-3}"
INVALID_REPORT_POLICY="${INVALID_REPORT_POLICY:-warn}"
ABLATION_SEED="${ABLATION_SEED:-20260725}"
BASE_SEED="${BASE_SEED:-20260725}"
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

echo "experiment_id=${EXPERIMENT_ID}"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-local}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "source_run=${SOURCE_RUN}"
echo "model_path=${MODEL_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "base_seed=${BASE_SEED}"
echo "ablation_seed=${ABLATION_SEED}"
git rev-parse HEAD 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python -u \
  agentic_pca/retrieval_agent_inference/trajectory_rag_ablation/infer_ablation.py \
  --experiment-id "${EXPERIMENT_ID}" \
  --model-path "${MODEL_PATH}" \
  --device cuda \
  --trajectory-rag-dir "${SOURCE_RUN}" \
  --num-trajectories "${NUM_TRAJECTORIES}" \
  --max-json-retries "${MAX_JSON_RETRIES}" \
  --invalid-report-policy "${INVALID_REPORT_POLICY}" \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed "${BASE_SEED}" \
  --ablation-seed "${ABLATION_SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "${MAX_PATIENT_ARGS[@]}" \
  "${MOCK_ARGS[@]}"
