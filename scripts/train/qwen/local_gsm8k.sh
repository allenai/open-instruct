#!/bin/bash

EXP_NAME="qwen25_05b_it_gsm8k"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct"
DATASETS="ai2-adapt-dev/rlvr_gsm8k_zs 1.0"

LOCAL_EVALS="mnoukhov/gsm8k-platinum-openinstruct 2"
LOCAL_EVAL_SPLITS="test"

# EVALS="aime:2024::justrl,aime:2025::justrl"

# BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="michaeln/open_instruct"
cluster=ai2/saturn

# Check if the first argument starts with the value of $BEAKER_NAME
# if [[ "$1" == "$BEAKER_USER"* ]]; then
#     BEAKER_IMAGE="$1"
#     shift
# fi

# uv run mason.py \
#     --task_name ${EXP_NAME} \
#     --cluster ${cluster} \
#     --workspace ai2/oe-adapt-code \
#     --priority high \
#     --pure_docker_mode \
#     --image ${BEAKER_IMAGE} \
#     --preemptible \
#     --num_nodes 1 \
#     --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
#     --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
#     --gpus 1 \
#     --budget ai2/oe-adapt \
#     -- source configs/beaker_configs/ray_node_setup.sh \
# \&\& 
#!/bin/bash
export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="FLASHINFER"
uv run --active open_instruct/grpo_fast.py \
    --output_dir results \
    --run_name test \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 4 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 512 \
    --response_length 2048 \
    --pack_length 4096 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name qwen_tulu_gsm8k \
    --system_prompt_remove True \
    --non_stop_penalty False \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --total_episodes 128 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 1 \
    --save_freq 200 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --single_gpu_mode \
    --vllm_enforce_eager \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --eval_on_step_0 True \
    --eval_pass_at_k 4 \
    --with_tracking False \
    --push_to_hub False $@

    # --checkpoint_state_freq 200 \
    # --keep_last_n_checkpoints -1 \

    # --eval_priority normal \
    # --try_launch_beaker_eval_jobs_on_weka True \
    # --oe_eval_max_length 32768 \
    # --oe_eval_gpu_multiplier 2  \
    # --oe_eval_beaker_image michaeln/oe_eval_internal \
    # --oe_eval_tasks $EVALS \
