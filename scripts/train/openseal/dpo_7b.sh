#!/bin/bash

source /localhome/tansang/envs/open-instruct/bin/activate

# Ensure the open_instruct package (one level up from the package dir) is on PYTHONPATH
export PYTHONPATH=/localhome/tansang/open-instruct:${PYTHONPATH}

echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# Memory optimization for CUDA
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Set NCCL timeout to 1 hour (from default 30 min)
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --mixed_precision bf16 \
    --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --dynamo_backend no \
    /localhome/tansang/open-instruct/open_instruct/dpo_tune_cache.py \
    --exp_name trial_openseal_dpo \
    --model_name_or_path /localhome/tansang/llm/openseal-sft/openseal-sailor2ds-stage1 \
    --tokenizer_name /localhome/tansang/llm/openseal-sft/openseal-sailor2ds-stage1 \
    --output_dir /localhome/tansang/llm/trial_openseal_dpo \
    --do_not_randomize_output_dir True \
    --local_cache_dir /localhome/tansang/open-instruct/scripts/train/openseal/local_dataset_cache/trial_openseal_dpo \
    --use_slow_tokenizer False \
    --mixer_list sailor2/sea-ultrafeedback 1.00 \
    --use_flash_attn \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --logging_steps 1 \
    --beta 5 \
    --checkpointing_steps 5000 \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False \
    --try_launch_beaker_eval_jobs False \
    --with_tracking False \
    --loss_type dpo_norm \
    --packing True 