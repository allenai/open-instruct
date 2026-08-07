#!/bin/bash

# GRPO (OLMo-core grpo.py) gsm8k run for the OLMo-hybrid small suite.
# GPUS is split evenly between training (FSDP) and generation (vLLM).
# Requires a Beaker image built from the transformers / vLLM / OLMo-core
# maintainer forks that implement the `olmo_hybrid_small` architecture (see
# pyproject.toml [tool.uv.sources]).
# Points at the HF-converted checkpoint (-hf suffix); grpo.py deserializes the
# sibling olmo-core-native checkpoint's own TransformerConfig to build the model.
#
# Usage: ./olmo_hybrid_gsm8k.sh BEAKER_IMAGE MODEL_PATH EXP_NAME [GPUS]
# (BEAKER_IMAGE is required here, not optional-with-default, since with a
# variable number of following args there's no reliable way to tell an
# omitted image apart from MODEL_PATH.)
#
# e.g.: ./olmo_hybrid_gsm8k.sh \
#         michaeln/open-instruct-integration-test-hybrid-small-suite-cuda12 \
#         /weka/oe-training-default/ai2-llm/checkpoints/saumyam/hybrid-small-sft-think-450m-lr8e-4/step24834-hf \
#         olmo_hybrid_450m_gsm8k_8gpu 8

DATASETS="ai2-adapt-dev/rlvr_gsm8k_zs 1.0"
EVAL_DATASETS="mnoukhov/gsm8k-platinum-openinstruct 1.0"

BEAKER_IMAGE="$1"
MODEL_PATH="$2"
EXP_NAME="$3"
GPUS="${4:-8}"
if [[ -z "$BEAKER_IMAGE" || -z "$MODEL_PATH" || -z "$EXP_NAME" ]]; then
    echo "Usage: $0 BEAKER_IMAGE MODEL_PATH EXP_NAME [GPUS]"
    exit 1
fi
shift $(( $# < 4 ? $# : 4 ))
LEARNERS=$((GPUS / 2))
VLLM_ENGINES=$((GPUS / 2))

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Model: $MODEL_PATH"
echo "Exp name: $EXP_NAME"
echo "GPUs: $GPUS ($LEARNERS learners, $VLLM_ENGINES vLLM engines)"

uv run python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/saturn ai2/jupiter ai2/ceres \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-other \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --no_auto_dataset_cache \
    --artifact_ttl 1d \
    --gpus "$GPUS" \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo.py \
    --exp_name ${EXP_NAME} \
    --run_name ${EXP_NAME} \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $EVAL_DATASETS \
    --dataset_mixer_eval_list_splits test \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --active_sampling \
    --filter_zero_std_samples True \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --max_prompt_token_length 512 \
    --response_length 8192 \
    --pack_length 16384 \
    --model_name_or_path "$MODEL_PATH" \
    --chat_template_name olmo_thinker \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --fsdp_shard_degree $LEARNERS \
    --fsdp_num_replicas 1 \
    --num_learners_per_node $LEARNERS \
    --vllm_num_engines $VLLM_ENGINES \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 200 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --mask_truncated_completions False \
    --load_ref_policy True \
    --with_tracking \
    --wandb_entity ai2-llm \
    --push_to_hub False "$@"
