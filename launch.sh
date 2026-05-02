for i in 1 2 3; do 
    export NAME="ngu_256c"

    BEAKER_IMAGE=michaeln/open-instruct-integration-test-michaeln-merge \
    EXP="${NAME}_seed${i}" \
    bash scripts/train/qwen/qwen2.5_0.5b_gsm8k_buckets.sh \
    --wandb_group_name $NAME \
    --vllm_num_engines 3 \
    --num_learners_per_node 1 \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 64 \
    --max_samples_multiplier 8 \
    --max_grad_norm 25 \
    --active_sampling \
    --never_give_up 0.95 \
    --num_response_completions_rollout 256

done
