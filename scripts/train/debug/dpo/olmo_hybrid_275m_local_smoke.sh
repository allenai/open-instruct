#!/bin/bash
# LOCAL (no Beaker) single-GPU DPO smoke test for the OLMo-hybrid 275M -hf checkpoint.
# Runs dpo.py directly against the local .venv on the current GPU.
set -uo pipefail

export REPO_PATH=/weka/oe-adapt-default/michaeln/nit-open-instruct
cd "$REPO_PATH"
export PYTHONPATH="$REPO_PATH"
export TORCH_LOGS="graph_breaks,recompiles"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206-hf

.venv/bin/torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$MODEL_PATH" \
    --attn_backend flash_2 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/dpo_olmo_hybrid_275m_local_smoke/ \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 50 \
    --chat_template_name olmo_thinker \
    --exp_name "dpo-olmo-hybrid-275m-local-smoke-$(date +%s)" \
    --seed 123 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
