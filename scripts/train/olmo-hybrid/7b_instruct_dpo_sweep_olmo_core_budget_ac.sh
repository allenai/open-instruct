#!/bin/bash
# DPO sweep for hybrid instruct models, using OLMo-core (dpo.py) instead of
# dpo_tune_cache.py (Accelerate + DeepSpeed ZeRO-3).
#
# Variant of 7b_instruct_dpo_sweep_olmo_core.sh that uses the standard
# budget-based activation checkpointing (torch.compile partitioner) instead of
# selected_modules, to test whether the fla tilelang backend lets the budget
# partitioner checkpoint through the GDN kernels.
#
# Usage (with pre-built image, no Docker build needed):
#   bash scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core_budget_ac.sh
#
# Usage (with build_image_and_launch.sh, slow ~1hr Docker build):
#   ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_instruct_dpo_sweep_olmo_core_budget_ac.sh
#
# NOTE: dpo.py builds the model with OLMo-core's native TransformerConfig, so the
# hybrid architecture must be resolvable from --config_name. See OLMO_MODEL_CONFIG_MAP
# / get_transformer_config in open_instruct/olmo_core_utils.py.

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

# MFU-tuning knobs.
FSDP_SHARD_DEGREE="${FSDP_SHARD_DEGREE:-32}"
FSDP_NUM_REPLICAS="${FSDP_NUM_REPLICAS:-1}"
ACTIVATION_CHECKPOINTING_MODE="${ACTIVATION_CHECKPOINTING_MODE:-budget}"
# Budget mode only activates when < 1.0 (see build_ac_config in olmo_core_utils.py).
ACTIVATION_MEMORY_BUDGET="${ACTIVATION_MEMORY_BUDGET:-0.5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EXP_TAG="${EXP_TAG:-}"
PROFILING="${PROFILING:-false}"
PROFILING_FLAG=""
if [ "$PROFILING" = "true" ]; then PROFILING_FLAG="--profiling"; fi
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_TRAIN_STEPS_FLAG=""
if [ -n "$MAX_TRAIN_STEPS" ]; then MAX_TRAIN_STEPS_FLAG="--max_train_steps $MAX_TRAIN_STEPS"; fi
TENSOR_PARALLEL_DEGREE="${TENSOR_PARALLEL_DEGREE:-1}"

SFT_MODELS=(
    allenai/Olmo-Hybrid-Instruct-SFT-7B
)

DPO_LRS=(
    1e-6
)

# OLMo-core TransformerConfig preset for the hybrid 7B model. Must be a config
# name registered with olmo-core's TransformerConfig (see olmo_core_utils.py).
CONFIG_NAME=olmo3_hybrid_7B

for MODEL_PATH in "${SFT_MODELS[@]}"; do
    for LR in "${DPO_LRS[@]}"; do
        EXP_NAME="hybrid-7b-DPO-oc-budget-ac-0219-SFT-public-LR-${LR}${EXP_TAG}"
        echo "====================================="
        echo "Launching: ${EXP_NAME}"
        echo "  SFT model: ${MODEL_PATH}"
        echo "  DPO LR: ${LR}"
        echo "====================================="

        uv run python mason.py \
            --cluster ai2/jupiter \
            --description "Hybrid 7B DPO sweep (OLMo-core, budget AC), LR=${LR}, 4 nodes, 16k seq." \
            --workspace ai2/linear-rnns \
            --priority urgent \
            --max_retries 0 \
            --artifact_ttl 1d \
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
            --num_nodes 4 \
            --gpus 8 -- torchrun \
            --nnodes=4 \
            --node_rank=\$BEAKER_REPLICA_RANK \
            --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
            --master_port=29400 \
            --nproc_per_node=8 \
            open_instruct/dpo.py \
            --exp_name "$EXP_NAME" \
            --model_name_or_path "$MODEL_PATH" \
            --config_name "$CONFIG_NAME" \
            --chat_template_name olmo123 \
            --mixer_list allenai/Dolci-Instruct-DPO-fixed 259922 \
            --max_seq_length 16384 \
            --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
            --fsdp_shard_degree "$FSDP_SHARD_DEGREE" \
            --fsdp_num_replicas "$FSDP_NUM_REPLICAS" \
            --tensor_parallel_degree "$TENSOR_PARALLEL_DEGREE" \
            --learning_rate "$LR" \
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
            --push_to_hub False \
            --try_launch_beaker_eval_jobs False \
            --activation_checkpointing_mode "$ACTIVATION_CHECKPOINTING_MODE" \
            --activation_memory_budget "$ACTIVATION_MEMORY_BUDGET" \
            --compile_model true \
            $PROFILING_FLAG \
            $MAX_TRAIN_STEPS_FLAG \
            --with_tracking
    done
done
