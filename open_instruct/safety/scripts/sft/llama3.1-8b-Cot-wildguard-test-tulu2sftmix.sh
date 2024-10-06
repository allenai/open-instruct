#!/bin/bash

python /net/nfs.cirrascale/mosaic/nouhad/projects/open-instruct/mason.py \
    --cluster ai2/jupiter-cirrascale-2  \
    --image nathanl/open_instruct_auto \
    --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --budget ai2/allennlp \
    --preemptible \
    --gpus 8 \
    --gpus 8 -- accelerate launch --config_file configs/ds_configs/deepspeed_zero3.yaml \
 open_instruct/finetune.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B" \
    --use_flash_attn \
    --max_seq_length 2048 \
    --dataset_mixer_list "open_instruct/wildguard_responses_test.json 1699 allenai/tulu-v2-sft-mixture 2000" \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1   \
    --output_dir models/sft_safety_gc/ \
    --with_tracking \
    --report_to tensorboard \
    --gradient_checkpointing \
    --push_to_hub \
    --logging_steps 1