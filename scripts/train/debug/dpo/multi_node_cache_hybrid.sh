#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf
LR=1e-6
EXP_NAME=hybrid-7b-DPO-cache-debug-1k-budget025-${LR}

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "2 node DPO cache run with hybrid model, 1k seq len, budget=0.25." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --gpus 8 -- torchrun \
    --nnodes=2 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/dpo_tune_cache.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --trust_remote_code \
    --chat_template_name olmo \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate "$LR" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/dpo_cache_hybrid_debug/ \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 1000 \
    --seed 123 \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --activation_memory_budget 0.25 \
    --with_tracking
