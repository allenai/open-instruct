#!/bin/bash
# OLMo-core/FSDP Holmes smoke version of 32b_think_rl.sh.
# Uses open_instruct/grpo.py with one 8-GPU learner node and one TP=8 vLLM generator.

set -euo pipefail

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
export exp_name=holmes_smoke_olmo3_32b_think_dpo_olmocore_rl_${RANDOM}
export data_mix="hamishivi/math_rlvr_mixture_dpo 1.0 hamishivi/code_rlvr_mixture_dpo 1.0 hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter 30186 allenai/rlvr_general_mix-keyword-filtered 21387"
export model_path=/weka/oe-adapt-default/jacobm/olmo-core-checkpoints/Olmo-3-32B-Think-DPO
export judge_base_url=${JUDGE_BASE_URL:?Set JUDGE_BASE_URL to the hosted Qwen3-32B judge endpoint, e.g. http://holmes-cs-aus-000.reviz.ai2.in:8001/v1}
export total_episodes=${TOTAL_EPISODES:-512}

uv run python mason.py \
    --cluster ai2/holmes \
    --image $BEAKER_IMAGE \
    --pure_docker_mode \
    --workspace ai2/holmes-testing \
    --priority urgent \
    --preemptible \
    --num_nodes 2 \
    --gpus 8 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    --env OPEN_INSTRUCT_VLLM_ENGINE_INIT_TIMEOUT_S=7500 \
    --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64 \
    --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
    --env HOSTED_VLLM_API_BASE=${judge_base_url} \
    --no_auto_dataset_cache \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 2e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list ${data_mix} \
        --dataset_mixer_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path ${model_path} \
        --config_name olmo3_32B \
        --chat_template_name olmo_thinker \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes ${total_episodes} \
        --num_learners_per_node 8 \
        --fsdp_shard_degree 8 \
        --fsdp_num_replicas 1 \
        --activation_memory_budget 0.1 \
        --vllm_num_engines 1 \
        --vllm_tensor_parallel_size 8 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --save_freq 25 \
        --gradient_checkpointing \
        --with_tracking \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --checkpoint_state_freq 100 \
        --backend_timeout 1200 \
        --active_sampling
