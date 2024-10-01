#!/bin/bash

python /net/nfs.cirrascale/mosaic/nouhad/projects/open-instruct/mason.py \
    --cluster ai2/allennlp-cirrascale \
    --image nathanl/open_instruct_auto \
    --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --priority high \
    --budget ai2/allennlp \
    --preemptible \
    --gpus 4 \
    -- accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B" \
    --use_flash_attn \
    --max_seq_length 2048 \
    --dataset_mixer_list "natolambert/tulu-v2-sft-mixture-flan 50000 natolambert/tulu-v2-sft-mixture-cot 49747 ai2-adapt-dev/personahub_math_v2_79975 79975 AI-MO/NuminaMath-TIR 72441" \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir /output/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
