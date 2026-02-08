#!/bin/bash
#SBATCH --job-name=train_job       # 作业名称
#SBATCH --partition=gpu            # 队列分区（根据你集群实际分区名修改，通常是 gpu）
#SBATCH --nodelist=gpu02           # 指定在 gpu02 节点运行
#SBATCH --nodes=1                  # 使用 1 个节点
#SBATCH --ntasks=1                 # 运行 1 个任务
#SBATCH --cpus-per-task=8          # 每个任务 8 个 CPU
#SBATCH --mem=32G                  # 内存 32G
#SBATCH --gres=gpu:1               # 使用 1 块 GPU
#SBATCH --output=log_%A_%a.out        # 标准输出日志 (%j 会自动替换为作业 ID)
#SBATCH --error=log_%A_%a.err         # 错误日志
#SBATCH --array=0-2

SMOOTH_LIST=(5000 8000 12000)


CURRENT_SMOOTH=${SMOOTH_LIST[$SLURM_ARRAY_TASK_ID]}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gen
cd ~/projects/psma_gen

echo "Running job with smoothness: $CURRENT_SMOOTH"

python Registration/train.py \
    --smoothness $CURRENT_SMOOTH \
    --epochs 350 \
    --lr 1e-5 \
    --num_masks 10 \
    --ct_smoothness \
    --ct_smoothness_margin 3000
