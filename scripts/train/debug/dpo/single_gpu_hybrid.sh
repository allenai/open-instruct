#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "Single GPU DPO run with hybrid OLMo3-7B, for debugging purposes." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env 'TORCH_LOGS=graph_breaks,recompiles' \
    --gpus 1 -- torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model_name_or_path "$MODEL_NAME" \
    --config_name olmo3_7B_hybrid \
    --vocab_size 100352 \
    --tokenizer_name_or_path allenai/dolma-2-tokenizer-olmo-3-instruct-final \
    --chat_template_name olmo123 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --activation_memory_budget 0.1 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 3 \
    --output_dir output/dpo_hybrid_single_gpu_debug/ \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --seed 123 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --try_auto_save_to_beaker false \
    --with_tracking
