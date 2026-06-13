#!/bin/bash
#SBATCH --job-name=ants_warp_pet
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=log_ants_warp_pet_%j.out
#SBATCH --error=log_ants_warp_pet_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/projects/psma_gen

# ANTs registration runs on CPU. Match its thread count to the Slurm allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"

python registerANTs_warped_pet.py \
  --input_dirs \
    /data2/xiangcen/data/pet_gen/processed/batch1_h5_v2 \
    /data2/xiangcen/data/pet_gen/processed/batch2_h5_v2 \
    /data2/xiangcen/data/pet_gen/processed/batch3_h5_v2 \
  --output_dir /data2/xiangcen/data/pet_gen/processed/warped_fdg_pet_h5 \
  --dataset_name warped_fdg_pet \
  --type_of_transform SyN \
  --interpolator linear
