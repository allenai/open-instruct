#!/bin/bash

source /localhome/tansang/envs/open-instruct/bin/activate

# Ensure the open_instruct package (one level up from the package dir) is on PYTHONPATH
export PYTHONPATH=/localhome/tansang/open-instruct:${PYTHONPATH}

echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"

# Memory optimization for CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600  # Increase timeout to 1 hour

CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --num_machines 1 \
    --dynamo_backend no \
    --use_deepspeed \
    --deepspeed_config_file /localhome/tansang/open-instruct/configs/ds_configs/stage3_no_offloading_accelerate.conf \
    /localhome/tansang/open-instruct/open_instruct/finetune.py \
    --exp_name trial_openseal_tulu \
    --model_name_or_path /localhome/tansang/llm/nus-olmo/parallel_only_7B_ckpt/step8290-unsharded-hf \
    --tokenizer_name /localhome/tansang/llm/nus-olmo/parallel_only_7B_ckpt/step8290-unsharded-hf \
    --output_dir /localhome/tansang/llm/openseal-sft \
    --use_slow_tokenizer False \
    --chat_template_name olmo \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 0.001 \
    --use_flash_attn \
    --gradient_checkpointing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --seed 8