#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=allenai/Olmo-3-1025-7B
LR=1e-6
EXP_NAME=olmo3-7b-DPO-debug-${LR}

uv run python mason.py \
    --cluster ai2/jupiter \
    --description "Multi-node DPO run with OLMo3-7B, 1k sequence length." \
    --workspace ai2/open-instruct-dev \
    --priority high \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- torchrun \
    --rdzv_conf='read_timeout=420' \
    --rdzv_id=12347 \
    --node_rank \$BEAKER_REPLICA_RANK \
    --nnodes 4 \
    --nproc_per_node 8 \
    --rdzv_backend=static \
    --rdzv_endpoint \$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
    open_instruct/dpo.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --tokenizer_name_or_path "$MODEL_NAME" \
    --chat_template_name olmo \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate "$LR" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/dpo_olmo3_debug/ \
    --dataset_mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 1000 \
    --seed 123 \
    --use_flash_attn \
    --logging_steps 1 \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --gradient_checkpointing \
    --with_tracking
