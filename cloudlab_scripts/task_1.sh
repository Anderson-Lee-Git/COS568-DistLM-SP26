#!/bin/bash
#SBATCH --job-name=task_1
#SBATCH --output=task_1.out
#SBATCH --error=task_1.err
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=./logs/task_1.out
#SBATCH --error=./logs/task_1.err
#SBATCH --chdir=/scratch/gpfs/KOROLOVA/cl6486/COS568-DistLM-SP26


export GLUE_DIR=./glue_data
export TASK_NAME=RTE

uv run python3 run_glue_task_1.py --model_type bert --model_name_or_path /scratch/gpfs/KOROLOVA/cl6486/.cache/huggingface/hub/models--google-bert--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e --task_name $TASK_NAME --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 3 --output_dir "./$TASK_NAME/task_1" --overwrite_output_dir