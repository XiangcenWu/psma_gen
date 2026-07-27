#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/share/home/xcwu/projects/psma_gen"
SLURM_DIR="${PROJECT_ROOT}/agentic_pca/DL-only/slurm"
LOG_DIR="${PROJECT_ROOT}/agentic_pca/DL-only/logs"
CACHE="${CACHE:-/data2/xiangcen/cache/pet_128x128x384}"
TARGET_NODE="${TARGET_NODE:-gpu02}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

export CACHE

cache_check_submission="$(
  sbatch \
    --parsable \
    --partition=gpu \
    --nodelist="${TARGET_NODE}" \
    --export=ALL \
    "${SLURM_DIR}/05_validate_cache.slurm"
)"
cache_check_job="${cache_check_submission%%;*}"

folds_submission="$(
  sbatch \
    --parsable \
    --partition=gpu \
    --nodelist="${TARGET_NODE}" \
    --export=ALL \
    --dependency="afterok:${cache_check_job}" \
    "${SLURM_DIR}/15_folds_array.slurm"
)"
folds_job="${folds_submission%%;*}"

aggregate_submission="$(
  sbatch \
    --parsable \
    --partition=gpu \
    --nodelist="${TARGET_NODE}" \
    --export=ALL \
    --dependency="afterok:${folds_job}" \
    "${SLURM_DIR}/20_aggregate.slurm"
)"
aggregate_job="${aggregate_submission%%;*}"

comparison_submission="$(
  sbatch \
    --parsable \
    --partition=gpu \
    --nodelist="${TARGET_NODE}" \
    --export=ALL \
    --dependency="afterok:${aggregate_job}" \
    "${SLURM_DIR}/30_compare_agent.slurm"
)"
comparison_job="${comparison_submission%%;*}"

echo "cache=${CACHE}"
echo "target_node=${TARGET_NODE}"
echo "cache_check_job=${cache_check_job}"
echo "folds_array_job=${folds_job}"
echo "aggregate_job=${aggregate_job}"
echo "comparison_job=${comparison_job}"
