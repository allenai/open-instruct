for i in 5; do 
# i=4
    export NAME="ngu_comp4_age_4"


    WORKSPACE=ai2/olmo-instruct \
    PRIORITY=urgent \
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
    --num_response_tokens_rollout 50000 \
    --maintain_pending_ngu_completions True \
    --maintain_pending_ngu_counts False \
    --maintain_pending_ngu_age 4

done
