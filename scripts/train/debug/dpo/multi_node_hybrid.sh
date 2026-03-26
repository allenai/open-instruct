#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=allenai/Olmo-Hybrid-Instruct-DPO-7B
LR=1e-6
EXP_NAME=hybrid-7b-DPO-test-${LR}-$(date +%s)

uv run python mason.py \
    --cluster ai2/jupiter \
    --description "Hybrid 7B DPO test, 2 nodes, 16k seq, ZeRO-3." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --max_retries 0 \
    --preemptible \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --timeout 60m \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env NCCL_IB_HCA=^=mlx5_bond_0 \
    --env NCCL_SOCKET_IFNAME=ib \
    --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    --env TORCH_DIST_INIT_BARRIER=1 \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --trust_remote_code \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 7680 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --zero_hpz_partition_size 1 \
    --learning_rate "$LR" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --packing \
    --use_flash_attn \
    --activation_memory_budget 0.5 \
    --chat_template_name olmo123 \
    --with_tracking \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --try_auto_save_to_beaker false
