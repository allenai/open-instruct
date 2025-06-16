# uv run python mason.py \
#     --cluster ai2/neptune-cirrascale \
#     --workspace ai2/michaeln \
#     --priority low \
#     --preemptible \
#     --num_nodes 1 \
#     --max_retries 0 \
#     --budget ai2/oe-adapt \
#     --gpus 1 -- uv run \
#     python open_instruct/grpo_fast.py \
#     --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 512 \
#     --max_prompt_token_length 512 \
#     --response_length 512 \
#     --pack_length 1024 \
#     --per_device_train_batch_size 1 \
#     --num_unique_prompts_rollout 1 \
#     --num_samples_per_prompt_rollout 16 \
#     --model_name_or_path allenai/OLMo-2-0425-1B-Instruct \
#     --stop_strings "</answer>" \
#     --apply_r1_style_format_reward \
#     --apply_verifiable_reward true \
#     --temperature 0.7 \
#     --ground_truths_key ground_truth \
#     --chat_template_name r1_simple_chat_postpend_think \
#     --learning_rate 3e-7 \
#     --total_episodes 64 \
#     --deepspeed_stage 2 \
#     --num_epochs 1 \
#     --num_learners_per_node 1 \
#     --vllm_tensor_parallel_size 1 \
#     --beta 0.01 \
#     --seed 3 \
#     --num_evals 20 \
#     --vllm_sync_backend gloo \
#     --vllm_gpu_memory_utilization 0.5 \
#     --save_traces \
#     --vllm_enforce_eager \
#     --gradient_checkpointing \
#     --single_gpu_mode $@

uv run open_instruct/grpo_fast.py \
    --exp_name test \
    --output_dir output/dummy \
    --dataset_mixer_list nouhad/multiplication_test_100_2x2 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list nouhad/multiplication_test_100_2x2 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 256 \
    --response_length 256 \
    --pack_length 512 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path HuggingFaceTB/SmolLM2-135M \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward False \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 10000 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --num_evals 20 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --with_tracking False 