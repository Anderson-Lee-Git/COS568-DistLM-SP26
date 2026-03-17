#!/bin/bash
#SBATCH --job-name=task_3
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4               # 4 ranks total on this node
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --output=./logs/task_3.out
#SBATCH --error=./logs/task_3.err
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/COS568-DistLM-SP26

mkdir -p logs

export GLUE_DIR=./glue_data
export TASK_NAME=RTE

mkdir -p ./$TASK_NAME/task_3

# master address/port: use first node name + fixed port
MASTER_PORT=12347
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_PROCID: $SLURM_PROCID"

# optionally pin threading so each rank uses 1 thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun bash -c 'uv run python3 run_glue_task_3.py \
  --model_type bert \
  --model_name_or_path /scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--google-bert--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e \
  --task_name '"$TASK_NAME"' \
  --do_train \
  --do_eval \
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
