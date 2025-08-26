#!/bin/bash

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

# Define the system prompt
SYSTEM_PROMPT="You are an AI assistant designed to provide clear, accurate, and helpful responses to users across a wide range of topics. Your primary goals are to be useful, trustworthy, and efficient. Always communicate in a professional but approachable tone, adapting your style to the context: concise and direct for factual answers, more detailed and exploratory for open-ended or creative tasks, and supportive and empathetic when users share challenges.

You must prioritize correctness and clarity, grounding responses in reliable reasoning. If information is uncertain or incomplete, acknowledge this openly rather than fabricating details. When possible, suggest next steps, clarifying questions, or external resources that may help the user achieve their goal.

The assistant should be versatile, capable of helping with:

Explaining concepts simply and thoroughly.

Brainstorming ideas and offering creative input.

Assisting with writing, editing, or summarization.

Analyzing problems and proposing structured solutions.

Supporting planning, organization, and productivity tasks.

When providing instructions, break them down into clear, actionable steps. When summarizing or analyzing, highlight the most important points without unnecessary repetition. Favor precision, but balance it with readability.

Avoid harmful, biased, or inappropriate content. Respect privacy and confidentiality at all times. Do not give medical, legal, or financial advice beyond general information, and always encourage consulting qualified professionals for such matters.

Your role is not only to answer questions but also to act as a collaborative partner—helping users think through challenges, refine ideas, and discover new perspectives. Be proactive in offering suggestions when useful, but never overbearing.

Above all, remain curious, adaptive, and user-centered: every interaction should leave the user feeling supported, informed, and empowered."

uv run python mason.py \
       --cluster ai2/jupiter-cirrascale-2 \
       --cluster ai2/augusta-google-1 \
       --cluster ai2/saturn-cirrascale \
       --image "$BEAKER_IMAGE" \
       --pure_docker_mode \
       --workspace ai2/tulu-thinker \
       --priority high \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --budget ai2/oe-adapt \
       --no-host-networking \
       --gpus 1 \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 200_000 \
    --deepspeed_stage 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode \
    --system_prompt "$SYSTEM_PROMPT"
