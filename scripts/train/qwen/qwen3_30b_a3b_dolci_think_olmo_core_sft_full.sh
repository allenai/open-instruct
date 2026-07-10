#!/bin/bash
set -euo pipefail

NUM_NODES="${NUM_NODES:-16}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"

EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-dolci-think-olmo-core-sft-full}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
PROJECT_ROOT="/weka/oe-adapt-default/jacobm/olmoe3/post-training"
MODEL_PATH="${PROJECT_ROOT}/checkpoints/qwen3-30b-a3b-base-olmo"
DATASET_PATH="${PROJECT_ROOT}/datasets/Dolci-Think-SFT-32B/qwen3-30b-a3b-olmo_thinker/full"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_NAME}"

torchrun_args=(--nproc_per_node=8)
if [[ "$NUM_NODES" == "1" ]]; then
    torchrun_args+=(--standalone)
else
    torchrun_args+=(
        --nnodes="$NUM_NODES"
        --node_rank=\$BEAKER_REPLICA_RANK
        --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME
        --master_port=29400
    )
fi

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
    --cluster ai2/holmes \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --max_retries 5 \
    --timeout 18h \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --non_resumable \
    --no_auto_dataset_cache \
    --num_nodes "$NUM_NODES" \
    --gpus 8 \
    --env OLMO_SHARED_FS=1 \
    --env OLMO_DDP_INIT_SYNC=0 \
    --env OLMO_EP_MP_HIGH_PRIORITY_GROUP=0 \
    --env NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
    --env NVSHMEM_HCA_LIST=^mlx5_bond_0 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -- torchrun \
    "${torchrun_args[@]}" \
    open_instruct/olmo_core_finetune.py \
    --run_name "$RUN_NAME" \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_PATH" \
    --config_name Qwen/Qwen3-30B-A3B-Base \
    --tokenizer_name_or_path Qwen/Qwen3-30B-A3B \
    --pretokenized_dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --attn_implementation flash_4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs 1 \
    --learning_rate 6e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --moe_expert_parallel_degree 8 \
    --moe_expert_parallel_path rowwise_nvshmem \
    --moe_expert_parallel_capacity_factor 2.0 \
    --moe_recompute_each_block true \
    --moe_checkpoint_block_internals false \
    --compile_model false \
    --activation_memory_budget 1.0 \
    --checkpointing_enabled true \
    --checkpointing_steps 1000000 \
    --ephemeral_save_interval 1000 \
    --keep_last_n_checkpoints 1 \
    --logging_steps 10 \
    --seed 33333 \
    --data_loader_seed 34521 \
    --with_tracking \
    --wandb_entity ai2-llm \
    --wandb_project jacobm-qwen3-30b-a3b-sft
