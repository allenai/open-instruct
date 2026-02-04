#!/bin/bash
# OLMo3-7B DPO Training on Tillicum (2-node × 8 GPUs = 16 GPUs)
#
# This is a Tillicum/Slurm conversion of scripts/train/olmo3/7b_instruct_dpo.sh
# Scaled from 4 nodes (32 GPUs) to 2 nodes (16 GPUs) to fit normal QOS limits.
#
# Requirements:
# - QOS 'normal' (no special approval needed)
# - Model: allenai/Olmo-3-7B-Instruct-SFT on HuggingFace Hub
# - HuggingFace token: Set HF_TOKEN environment variable
# - WandB token: Set WANDB_API_KEY environment variable
#
# Usage:
#   # First ensure tokens are set (one-time setup):
#   export HF_TOKEN="hf_your_token_here"
#   export WANDB_API_KEY="your_wandb_key_here"
#
#   # Then run:
#   bash scripts/train/tillicum/olmo3_7b_instruct_dpo_2node.sh
#
# Estimated cost: 16 GPUs × 8 hours × $0.90/GPU-hour = $115.20
# Expected runtime: ~8 hours
#
# Outputs:
# - Logs: /gpfs/scrubbed/$USER/experiments/olmo3_7b_dpo_2node_*/logs/
# - Model: /gpfs/scrubbed/$USER/experiments/olmo3_7b_dpo_2node_*/output/
# - Checkpoints: /gpfs/scrubbed/$USER/experiments/olmo3_7b_dpo_2node_*/checkpoints/

# Check for required tokens
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set!"
    echo ""
    echo "Please set your HuggingFace token before running this script:"
    echo "  export HF_TOKEN='hf_your_token_here'"
    echo ""
    echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
    echo "  echo 'export HF_TOKEN=\"hf_your_token_here\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set!"
    echo ""
    echo "Please set your Weights & Biases API key before running this script:"
    echo "  export WANDB_API_KEY='your_wandb_key_here'"
    echo ""
    echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
    echo "  echo 'export WANDB_API_KEY=\"your_wandb_key_here\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    exit 1
fi

echo "✓ Tokens verified: HF_TOKEN and WANDB_API_KEY are set"
echo ""

# Model configuration
MODEL_NAME=allenai/Olmo-3-7B-Instruct-SFT

# Learning rate (from original script)
LR=1e-6

# Experiment name
EXP_NAME=olmo3-7b-DPO-tillicum-2node-${LR}

uv run python tillicum.py \
    --gpus 8 \
    --nodes 2 \
    --time 08:00:00 \
    --qos normal \
    --job_name olmo3_7b_dpo_2node \
    --module gcc/13.4.0 \
    --module cuda/13.0.0 \
    "$@" \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer False \
    --mixer_list allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 125000 \
        allenai/dpo-yolo1-200k-gpt4.1-2w2s-maxdelta_reje-426124-rm-gemma3-kwd-ftd-ch-ftd-topic-ftd-dedup5-lbc100 125000 \
        allenai/related-query_qwen_pairs_filtered_lbc100 1250 \
        allenai/paraphrase_qwen_pairs_filtered_lbc100 938 \
        allenai/repeat_qwen_pairs_filtered_lbc100 312 \
        allenai/self-talk_qwen_pairs_filtered_lbc100 2500 \
        allenai/related-query_gpt_pairs_filtered_lbc100 1250 \
        allenai/paraphrase_gpt_pairs_filtered_lbc100 938 \
        allenai/repeat_gpt_pairs_filtered_lbc100 312 \
        allenai/self-talk_gpt_pairs_filtered_lbc100 2500 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --zero_hpz_partition_size 1 \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --use_flash_attn \
    --gradient_checkpointing \
    --chat_template_name olmo123 \
    --with_tracking \
    --output_dir '$EXPERIMENT_DIR/output' \
    --dataset_local_cache_dir '$DATASET_LOCAL_CACHE_DIR'
    # NOTE: The following Beaker evaluation args are not supported on Tillicum.
    # They are preserved here for reference only. Run evaluations separately.
    # --eval_workspace ai2/olmo-instruct \
    # --eval_priority urgent \
    # --oe_eval_max_length 32768 \
    # --oe_eval_gpu_multiplier 2 \
    # --oe_eval_tasks "omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek"
