#!/bin/bash

# GRPO (OLMo-core grpo.py) gsm8k run for the OLMo-hybrid small suite.
# GPUS is split evenly between training (FSDP) and generation (vLLM).
# Requires a Beaker image built from the transformers / vLLM / OLMo-core
# maintainer forks that implement the `olmo_hybrid_small` architecture (see
# pyproject.toml [tool.uv.sources]).
# Points at the HF-converted checkpoint (-hf suffix); grpo.py deserializes the
# sibling olmo-core-native checkpoint's own TransformerConfig to build the model.
#
# MODE selects the chat template (see CHAT_TEMPLATES in dataset_transformation.py):
#   think            - generation prompt opens with "<think>", i.e. the checkpoint's
#                       normal reasoning behavior.
#   open_close_think - generation prompt pre-fills a closed, empty "<think></think>",
#                       the standard way to turn a thinking model into instruct-only
#                       (cf. Qwen3's enable_thinking=False).
# Both use a corrected default system prompt matching this checkpoint's actual
# tokenizer_config.json (see research.md#olmo-hybrid-275m-small-suite-grpo-on-gsm8k
# for why "olmo_thinker"'s default system prompt is wrong for this checkpoint).
#
# EVAL_ONLY=true runs a single training step (learning_rate=0, so weights don't
# move) with an eval collected at step 1, to get a baseline GSM8K number for the
# unmodified checkpoint under the given MODE before spending compute on RL.
#
# Usage: ./gsm8k_grpo.sh BEAKER_IMAGE MODEL_PATH EXP_NAME MODE [EVAL_ONLY] [GPUS] [CLUSTERS]
# (BEAKER_IMAGE is required here, not optional-with-default, since with a
# variable number of following args there's no reliable way to tell an
# omitted image apart from MODEL_PATH.)
#
# e.g.: ./gsm8k_grpo.sh \
#         michaeln/open-instruct-integration-test-hybrid-small-suite-cuda12 \
#         /weka/oe-training-default/ai2-llm/checkpoints/saumyam/hybrid-small-sft-think-450m-lr8e-4/step24834-hf \
#         olmo_hybrid_450m_gsm8k_think 8 "ai2/jupiter"

DATASETS="ai2-adapt-dev/rlvr_gsm8k_zs 1.0"
EVAL_DATASETS="mnoukhov/gsm8k-platinum-openinstruct 1.0"

BEAKER_IMAGE="$1"
MODEL_PATH="$2"
EXP_NAME="$3"
MODE="$4"
EVAL_ONLY="${5:-false}"
GPUS="${6:-8}"
CLUSTERS="${7:-ai2/saturn ai2/jupiter ai2/ceres}"
if [[ -z "$BEAKER_IMAGE" || -z "$MODEL_PATH" || -z "$EXP_NAME" || -z "$MODE" ]]; then
    echo "Usage: $0 BEAKER_IMAGE MODEL_PATH EXP_NAME MODE [EVAL_ONLY] [GPUS] [CLUSTERS]"
    echo "  MODE: think | open_close_think"
    exit 1
fi
case "$MODE" in
    think) CHAT_TEMPLATE_NAME="olmo_hybrid_small_think" ;;
    open_close_think) CHAT_TEMPLATE_NAME="olmo_hybrid_small_open_close_think" ;;
    *)
        echo "Unknown MODE '$MODE', expected 'think' or 'open_close_think'"
        exit 1
        ;;
esac
shift $(( $# < 7 ? $# : 7 ))
LEARNERS=$((GPUS / 2))
VLLM_ENGINES=$((GPUS / 2))

NUM_SAMPLES_PER_PROMPT_ROLLOUT=16
NUM_UNIQUE_PROMPTS_ROLLOUT=16
if [[ "$EVAL_ONLY" == "true" ]]; then
    # One training step's worth of episodes, learning_rate=0 so the single step is a
    # no-op, plus --eval_on_step_0 to collect the eval at that first (only) step.
    TOTAL_EPISODES=$((NUM_SAMPLES_PER_PROMPT_ROLLOUT * NUM_UNIQUE_PROMPTS_ROLLOUT))
    LEARNING_RATE=0
else
    TOTAL_EPISODES=512000
    LEARNING_RATE=1e-6
fi

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Model: $MODEL_PATH"
echo "Exp name: $EXP_NAME"
echo "Mode: $MODE (chat template: $CHAT_TEMPLATE_NAME)"
echo "Eval only: $EVAL_ONLY (total_episodes=$TOTAL_EPISODES, learning_rate=$LEARNING_RATE)"
echo "GPUs: $GPUS ($LEARNERS learners, $VLLM_ENGINES vLLM engines)"
echo "Clusters: $CLUSTERS"

uv run python mason.py \
    --task_name ${EXP_NAME} \
    --cluster $CLUSTERS \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-other \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --no_auto_dataset_cache \
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
    --num_samples_per_prompt_rollout $NUM_SAMPLES_PER_PROMPT_ROLLOUT \
    --num_unique_prompts_rollout $NUM_UNIQUE_PROMPTS_ROLLOUT \
    --num_mini_batches 1 \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size 1 \
    --max_prompt_token_length 512 \
    --response_length 16384 \
    --pack_length 32768 \
    --model_name_or_path "$MODEL_PATH" \
    --chat_template_name "$CHAT_TEMPLATE_NAME" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes $TOTAL_EPISODES \
    --fsdp_shard_degree $LEARNERS \
    --fsdp_num_replicas 1 \
    --num_learners_per_node $LEARNERS \
    --vllm_num_engines $VLLM_ENGINES \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --eval_on_step_0 \
    --save_freq 100 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --mask_truncated_completions False \
    --load_ref_policy True \
    --with_tracking \
    --wandb_entity mnoukhov \
    --push_to_hub False "$@"
