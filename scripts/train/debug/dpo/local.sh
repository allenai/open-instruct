#!/bin/bash
export TORCH_LOGS="graph_breaks,recompiles"
uv run torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --tokenizer_name_or_path allenai/OLMo-2-0425-1B \
    --attn_backend flash_2 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/dpo_local_test/ \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 50 \
    --chat_template_name olmo \
    --exp_name "dpo-local-debug-$(date +%s)" \
    --seed 123 \
    --push_to_hub false
