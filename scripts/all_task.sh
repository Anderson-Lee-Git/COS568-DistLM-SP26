#!/bin/bash
#SBATCH --job-name=all_tasks
#SBATCH --time=03:00:00          # 3x ~1hr each, with buffer
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --output=./logs/all_tasks.out
#SBATCH --error=./logs/all_tasks.err
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/COS568-DistLM-SP26

mkdir -p logs

export GLUE_DIR=./glue_data
export TASK_NAME=RTE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)

# ---------- Task 2a ----------
mkdir -p ./$TASK_NAME/task_2a
MASTER_PORT=12720
srun bash -c 'uv run python3 task_4/run_glue_task_2a.py \
  --model_type bert \
  --model_name_or_path /scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--google-bert--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e \
  --task_name '"$TASK_NAME"' \
  --do_train --do_eval \
  --data_dir '"$GLUE_DIR/$TASK_NAME"' \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./'"$TASK_NAME"'/task_2a \
  --overwrite_output_dir \
  --world_size '"$SLURM_NTASKS"' \
  --local_rank $SLURM_PROCID \
  --master_ip '"$MASTER_ADDR"' \
  --master_port '"$MASTER_PORT"''

# ---------- Task 2b ----------
mkdir -p ./$TASK_NAME/task_2b
srun bash -c 'uv run python3 task_4/run_glue_task_2b.py \
  --model_type bert \
  --model_name_or_path /scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--google-bert--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e \
  --task_name '"$TASK_NAME"' \
  --do_train --do_eval \
  --data_dir '"$GLUE_DIR/$TASK_NAME"' \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./'"$TASK_NAME"'/task_2b \
  --overwrite_output_dir \
  --world_size '"$SLURM_NTASKS"' \
  --local_rank $SLURM_PROCID \
  --master_ip '"$MASTER_ADDR"' \
  --master_port '"$MASTER_PORT"''

# ---------- Task 3 ----------
mkdir -p ./$TASK_NAME/task_3
srun bash -c 'uv run python3 task_4/run_glue_task_3.py \
  --model_type bert \
  --model_name_or_path /scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--google-bert--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e \
  --task_name '"$TASK_NAME"' \
  --do_train --do_eval \
  --data_dir '"$GLUE_DIR/$TASK_NAME"' \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./'"$TASK_NAME"'/task_3 \
  --overwrite_output_dir \
  --world_size '"$SLURM_NTASKS"' \
  --local_rank $SLURM_PROCID \
  --master_ip '"$MASTER_ADDR"' \
  --master_port '"$MASTER_PORT"''