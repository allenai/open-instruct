#!/bin/bash
# Qwen3-4B SFT with balanced data mix (natural proportions)
MIX_NAME=balanced

python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority high \
    --budget ai2/oe-other \
    --num_nodes 1 \
    --gpus 8 \
    --non_resumable \
    --preemptible \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    -- torchrun \
    --nproc_per_node=8 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --numpy_dataset_path /weka/oe-training-default/saumyam/qwen3_sft_mix_${MIX_NAME}.yaml \
    --max_seq_length 8192 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --num_epochs 2 \
    --activation_memory_budget 0.5 \
    --compile_model true \
    --checkpointing_steps 1000 \
    --ephemeral_save_interval 500 \
    --with_tracking \
    --wandb_project saumyam-qwen3-4b-sft \
    --wandb_entity ai2-llm \
    --logging_steps 1 \
    --seed 33333 \
    --run_name qwen3-4b-sft-${MIX_NAME} \
    --output_dir /weka/oe-training-default/saumyam/qwen3-4b-sft/${MIX_NAME}
