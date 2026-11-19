#!/bin/bash
# SMOKE on the SECOND base family: Qwen3-8B SFT on capability (tmax) + SafeComp safety data.
#
# Purpose: replicate the 9B smoke's mixture (tmax 0.2 ~3.3k + SafeComp 1,045) on Qwen3-8B, the base
# of the other before/after pair (comparison ③ in agent-safety/evals/OAS_SAFETY_BASELINES.md). Prior
# work shows SFT lifts capability on this base; the question is whether the SafeComp mixture repeats
# the 9B result (safety up, capability preserved) across a base-family change.
#
# This is `sft_qwen35_9b_safecomp_smoke.sh` with TWO substantive changes:
#   1. MODEL/TOKENIZER: hamishivi/Qwen3.5-9B -> hamishivi/Qwen3-8B. Weights are byte-identical to
#      Qwen/Qwen3-8B (baseline ③'s base — comparison stays valid); the difference is the chat
#      template: stock Qwen3 STRIPS <think> from non-final assistant turns, hamishivi's keeps
#      reasoning_content everywhere — required to actually train on the thinking tmax data.
#   2. Image: shashankg/open-instruct-integration-test-omni_agent (build commit 19abd73fb,
#      2026-07-12; verified to contain --dataset_mixer_list_config_names AND the agent fork's
#      `tools` column support). Newer trainer than the 9B smoke's image (aeb75710c) — flag when
#      comparing smokes across families.
# plus the exp_name/output_dir rename. Data mixture, LR, seq, SP, epochs=1, seed, and
# no-push/no-auto-eval are kept identical so the two smokes differ only in the base.
#
# Post-train serving differs from the 9B (see agent-safety/evals/OAS_EVAL_GUIDE.md §2, Qwen3 pair):
# plain Qwen3ForCausalLM -> NO -servable rebuild, NO --language-model-only; parser `hermes` (not
# qwen3_xml); 65536 ctx via YaRN x2 --hf-overrides; llm_config max_input_tokens 52000.
BEAKER_IMAGE="shashankg/open-instruct-integration-test-omni_agent"
echo "Using Beaker image: $BEAKER_IMAGE"

MODEL=hamishivi/Qwen3-8B
TOKENIZER=hamishivi/Qwen3-8B
EXP_NAME=sft_qwen3_8b_safecomp_smoke

# capability half (identical to the 9B smoke)
DATASET=osieosie/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all
CAP_FRAC=0.2

# safety half (identical file to the 9B smoke)
SAFECOMP=/weka/oe-adapt-default/mingqianz/agent-safety/data/sft/sft_comp_smoke1k.jsonl

OUT=/weka/oe-adapt-default/mingqianz/models/qwen3-8b-safecomp-smoke

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "SFT Qwen3-8B: capability(tmax 0.2) + SafeComp safety smoke (~1k)" \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --budget ai2/oe-omai \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL \
    --tokenizer_name $TOKENIZER \
    --sequence_parallel_size 4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --dataset_mixer_list \
        $DATASET $CAP_FRAC \
        $SAFECOMP 1.0 \
    --dataset_mixer_list_config_names \
        $DATASET_CONFIG none \
    --dataset_mixer_list_splits \
        train train \
    --output_dir $OUT \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --logging_steps 1 \
    --seed 42 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
    # ^ same rationale as the 9B smoke: our image lacks oe-eval-internal (auto-eval would crash at
    # finetune.py:1115) and we run OAS ourselves and serve from weka.
