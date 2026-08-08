#!/bin/bash
# Backend A/B: DPO on DeepSpeed ZeRO-3 (dpo_tune_cache.py). Pair: ab_dpo_olmocore.sh.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"
MODEL_PATH="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --description "Backend A/B: DPO DeepSpeed (dpo_tune_cache.py), Olmo-3-7B, 4 nodes, 16k seq." \
    --max_retries 0 \
    --preemptible \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env NCCL_IB_HCA=^=mlx5_bond_0 \
    --env NCCL_SOCKET_IFNAME=ib \
    --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    --env TORCH_DIST_INIT_BARRIER=1 \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
    --num_nodes 4 \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name ab_dpo_deepspeed \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --use_slow_tokenizer False \
    --mixer_list allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 30000 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --zero_hpz_partition_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --checkpointing_steps 500 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --max_train_steps 150 \
    --seed 42 \
    --logging_steps 1 \
    --packing \
    --activation_memory_budget 0.5 \
    --chat_template_name olmo123 \
    --push_to_hub False \
    --try_launch_beaker_eval_jobs False \
    --try_auto_save_to_beaker False \
    --with_tracking
