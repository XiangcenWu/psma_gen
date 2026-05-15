#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=log_%A_%a.out
#SBATCH --error=log_%A_%a.err


# ---------------- environment ----------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen


# ---------------- info ----------------
echo "======================================"
echo "SLURM TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "======================================"


# ---------------- run training ----------------
python llm_Registration/toy_example.py \
    --hf_model_dir /data2/xiangcen/hf_models/ \
    --data_dirs /data2/xiangcen/data/pet_gen/processed/batch1_h5_v2 /data2/xiangcen/data/pet_gen/processed/batch2_h5_v2 \
    --num_validations 40 40 \
    --batch_size 10 \
    --epochs 350 \
    --max_prompt_organs 5 \
    --lr 1e-5 \
    --device cuda:0 \
    --save_path /data2/xiangcen/llm_regsitration_model/toy.ptm
