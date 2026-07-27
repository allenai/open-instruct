#!/bin/bash

EXP="${EXP:-}"

TRAIN_SCRIPT="open_instruct/grpo.py"
BACKEND_ARGS="--fsdp_shard_degree 4 --fsdp_num_replicas 1 --activation_memory_budget 0.5"

EXP_NAME="${EXP_NAME:-deepcoder_1_5b_${EXP}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${NUM_GPUS:-8}"
BEAKER_IMAGE="${BEAKER_IMAGE:-nathanl/open_instruct_auto}"

CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
NODES="${NODES:-1}"

# Datasets converted from agentica-org/DeepCoder-Preview-Dataset by
# scripts/data/create_deepcoder_data.py. lcbv5_test and codeforces_test are
# reserved as held-out eval and never appear in the train mixer. humanevalplus_test
# is converted from evalplus/humanevalplus by scripts/data/create_humanevalplus_data.py
# and is eval-only (function-signature style, graded by the "code" verifier, not "code_stdio").
# Each eval source is tagged with a distinct dataset name so eval metrics
# (eval/pass_at_1/<name>, etc.) report separately per source instead of pooling.
TRAIN_DATASETS="mnoukhov/deepcoder_lcbv5 1.0 mnoukhov/deepcoder_primeintellect 1.0 mnoukhov/deepcoder_taco 1.0"
EVAL_DATASETS="mnoukhov/deepcoder_lcbv5_test 1.0 mnoukhov/deepcoder_codeforces_test 1.0 mnoukhov/humanevalplus_test 1.0"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ${WORKSPACE} \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes ${NODES} \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --gpus $NUM_GPUS \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& source configs/beaker_configs/code_api_setup.sh \
\&\& uv run ${TRAIN_SCRIPT} \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --vllm_top_p 1.0 \
    --eval_temperature 0.6 \
    --beta 0.001 \
    --async_steps 2 \
    --active_sampling \
    --inflight_updates \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --temperature 1.0 \
    --dataset_mixer_list $TRAIN_DATASETS \
    --dataset_mixer_list_splits "train" \
    --dataset_mixer_eval_list $EVAL_DATASETS \
    --dataset_mixer_eval_list_splits "train" \
    --max_prompt_token_length 2048 \
    --response_length 32768 \
    --pack_length 34816 \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --non_stop_penalty False \
    --total_episodes 64000 \
    ${BACKEND_ARGS} \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --send_slack_alerts \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --max_grad_norm 1.0 \
    --mask_truncated_completions True \
    --code_api_url \$CODE_API_URL/test_program \
    --code_pass_rate_reward_threshold 0.0 \
    --load_ref_policy True \
    --keep_last_n_checkpoints 3 \
    --push_to_hub False "$@"
