#!/bin/bash
# Final DPO run for Olmo Hybrid 7B Instruct.
#
# Usage (with pre-built image, no Docker build needed):
#   bash scripts/train/olmo-hybrid/7b_instruct_dpo.sh
#
# Usage (with build_image_and_launch.sh, slow ~1hr Docker build):
#   ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_instruct_dpo.sh
#
# Nominal performance: ~10s/step, ~6hrs total (2031 steps).
# Monitor via wandb perf/seconds_per_step. Expect MFU ~3%.

BEAKER_IMAGE="${1:-finbarrt/hybrid-dpo-stable}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --description "Hybrid 7B Instruct DPO final, LR=1e-6, 4 nodes, 16k seq, ZeRO-3." \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --max_retries 0 \
    --preemptible \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env NCCL_IB_HCA=^=mlx5_bond_0 \
    --env NCCL_SOCKET_IFNAME=ib \
    --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    --env TORCH_DIST_INIT_BARRIER=1 \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
    --env TRITON_PRINT_AUTOTUNING=1 \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name "hybrid-7b-DPO-instruct-final" \
    --model_name_or_path allenai/OLMo-3.2-Hybrid-7B-Instruct-SFT \
    --trust_remote_code \
    --mixer_list allenai/Dolci-Instruct-DPO-fixed 259922 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --zero_hpz_partition_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --checkpointing_steps 500 \
    --keep_last_n_checkpoints -1 \
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
    --with_tracking
