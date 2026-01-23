#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=allenai/OLMo-2-1124-7B

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "DPO cache forward-pass benchmark: $MODEL_NAME, 2 nodes, 16k seq" \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env "REFERENCE_LOGPROBS_CACHE_PATH=/tmp/benchmark_cache_\$(date +%s)" \
    --gpus 8 -- torchrun \
    --nnodes=2 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/dpo.py \
    --model_name_or_path "$MODEL_NAME" \
    --chat_template_name olmo \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/benchmark_dpo_cache/ \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 1000 \
    --seed 123 \
    --use_flash_attn \
    --loss_type dpo_norm \
    --beta 5 \
    --gradient_checkpointing \
    --cache_logprobs_only
