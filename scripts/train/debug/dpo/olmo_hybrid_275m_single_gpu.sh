#!/bin/bash

# Single GPU DPO debug run for the OLMo-hybrid small suite (275M SFT checkpoint).
# Requires a Beaker image built from the transformers fork that implements the
# `olmo_hybrid_small` architecture (see pyproject.toml [tool.uv.sources]).
# Points at the HF-converted checkpoint (-hf suffix); the olmo-core checkpoint
# (step23206/) is not directly loadable by transformers.

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-sft-think-275M-lr2e-4/step23206-hf

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "Single GPU DPO run for the OLMo-hybrid small suite 275M, for debugging purposes." \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-other \
    --no_auto_dataset_cache \
    --no-host-networking \
    --env 'TORCH_LOGS=graph_breaks,recompiles' \
    --gpus 1 -- torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$MODEL_PATH" \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 3 \
    --output_dir output/dpo_olmo_hybrid_275m_debug/ \
    --logging_steps 1 \
    --mixer_list allenai/tulu-3-wildchat-reused-on-policy-8b 100 \
    --chat_template_name olmo_thinker \
    --exp_name "dpo-olmo-hybrid-275m-single-gpu-debug-$(date +%s)" \
    --seed 123 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --try_auto_save_to_beaker false \
    --with_tracking
