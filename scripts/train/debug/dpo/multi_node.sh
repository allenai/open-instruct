#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=allenai/OLMo-2-1124-7B
LR=1e-6
EXP_NAME=olmo2-7b-DPO-debug-16k-packing-bs16-tp2-${LR}

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "2 node DPO run with OLMo2-7B, 16k seq len, bs=16 (packing + compile)." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env 'TORCH_LOGS=graph_breaks,recompiles' \
    --gpus 8 -- torchrun \
    --nnodes=2 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/dpo.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --chat_template_name olmo \
    --max_seq_length 16384 \
    --per_device_train_batch_size 16 \
    --packing \
    --gradient_accumulation_steps 1 \
    --learning_rate "$LR" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/dpo_olmo2_debug_16k_baseline/ \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 7680 \
    --seed 123 \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --activation_memory_budget 0.1 \
    --profiling \
    --with_tracking \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --shard_degree 4 \
    --num_replicas 2 \
    --tensor_parallel_degree 2 \
    --try_auto_save_to_beaker false
