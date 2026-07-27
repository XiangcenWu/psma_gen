#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/share/home/xcwu/projects/psma_gen"
SLURM_DIR="${PROJECT_ROOT}/agentic_pca/retrieval_agent_inference/comparison_baselines/slurm"

cd "${PROJECT_ROOT}"

validation_submission="$(
  sbatch --parsable "${SLURM_DIR}/00_validate.slurm"
)"
validation_job="${validation_submission%%;*}"

direct_submission="$(
  sbatch \
    --parsable \
    --dependency="afterok:${validation_job}" \
    "${SLURM_DIR}/10_direct_array.slurm"
)"
direct_job="${direct_submission%%;*}"

case_knn_submission="$(
  sbatch \
    --parsable \
    --dependency="afterok:${validation_job}" \
    "${SLURM_DIR}/20_case_knn.slurm"
)"
case_knn_job="${case_knn_submission%%;*}"

suv_ml_submission="$(
  sbatch \
    --parsable \
    --dependency="afterok:${validation_job}" \
    "${SLURM_DIR}/21_structured_suv_ml.slurm"
)"
suv_ml_job="${suv_ml_submission%%;*}"

evaluation_submission="$(
  sbatch \
    --parsable \
    --dependency="afterany:${direct_job}:${case_knn_job}" \
    "${SLURM_DIR}/30_evaluate.slurm"
)"
evaluation_job="${evaluation_submission%%;*}"

echo "validation_job=${validation_job}"
echo "direct_qwen_array_job=${direct_job}"
echo "case_knn_job=${case_knn_job}"
echo "structured_suv_ml_job=${suv_ml_job}"
echo "evaluation_job=${evaluation_job}"
