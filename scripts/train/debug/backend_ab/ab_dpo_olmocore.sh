#!/bin/bash
# Backend A/B: DPO on OLMo-core FSDP (dpo.py). Pair: ab_dpo_deepspeed.sh.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"
MODEL_PATH="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --description "Backend A/B: DPO OLMo-core (dpo.py), Olmo-3-7B, 4 nodes, 16k seq." \
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
    --gpus 8 -- torchrun \
    --nnodes=4 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/dpo.py \
    --exp_name ab_dpo_olmocore \
    --model_name_or_path "$MODEL_PATH" \
    --config_name olmo3_7B \
    --chat_template_name olmo123 \
    --mixer_list allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 30000 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --fsdp_shard_degree 32 \
    --fsdp_num_replicas 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --checkpointing_steps 500 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --max_train_steps 150 \
    --seed 42 \
    --logging_steps 1 \
    --activation_checkpointing_mode selected_modules \
    --activation_checkpointing_modules 'blocks.*' \
    --compile_model true \
    --push_to_hub False \
    --try_launch_beaker_eval_jobs False \
    --try_auto_save_to_beaker False \
    --with_tracking
