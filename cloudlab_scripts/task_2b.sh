#!/bin/bash
#SBATCH --job-name=task_2b
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=4               # 4 ranks total on this node
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --output=./logs/task_2a.out
#SBATCH --error=./logs/task_2a.err
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/COS568-DistLM-SP26

mkdir -p logs

export GLUE_DIR=./glue_data
export TASK_NAME=RTE

mkdir -p ./$TASK_NAME/task_2b

# master address/port: use first node name + fixed port
MASTER_ADDR="10.10.1.2"
MASTER_PORT=12346
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# optionally pin threading so each rank uses 1 thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOCAL_RANK="${1:?Usage: $0 <local_rank>}"

python3 run_glue_task_2b.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./$TASK_NAME/task_2b \
  --overwrite_output_dir \
  --world_size 4 \
  --local_rank $LOCAL_RANK \
  --master_ip $MASTER_ADDR \
  --master_port $MASTER_PORT
