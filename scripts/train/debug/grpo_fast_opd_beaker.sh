#!/bin/bash

# Beaker OPD smoke test #1 (Qwen3, cheap): Qwen3-0.6B student distills from a
# frozen Qwen3-1.7B teacher on gsm8k, 1 node x 2 GPUs (1 learner + 1 engine),
# exercising the production loss path (DPPO + liger tiled loss + vLLM logprobs
# + ZeRO-3) with pure-OPD advantages. ~16 training steps then exits.
# Beaker twin of scripts/train/debug/grpo_fast_opd.sh.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --cluster ai2/ceres \
       --image "$BEAKER_IMAGE" \
       --description "OPD smoke: Qwen3-0.6B student + Qwen3-1.7B teacher, gsm8k, DPPO+liger" \
       --pure_docker_mode \
       --workspace ai2/oe-agents \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --budget ai2/oe-omai \
       --gpus 2 \
       --no_auto_dataset_cache \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --opd_teacher_model_name_or_path Qwen/Qwen3-1.7B \
    --opd_kl_coef 1.0 \
    --opd_pure \
    --filter_zero_std_samples false \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 512 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --load_ref_policy false \
    --beta 0.0 \
    --use_vllm_logprobs true \
    --truncated_importance_sampling_ratio_cap 0.0 \
    --loss_fn dppo \
    --dppo_divergence_type tv \
    --dppo_divergence_threshold 0.1 \
    --use_liger_grpo_loss \
    --liger_grpo_loss_chunk_size 8 \
    --advantage_normalization_type centered \
    --seed 3 \
    --local_eval_every -1 \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --with_tracking \
    --wandb_project oe-general-agents \
    --exp_name opd_smoke_qwen3_gsm8k \
    --push_to_hub false
