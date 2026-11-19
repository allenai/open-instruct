#!/bin/bash
# Rung-1 SMOKE: Qwen3.5-9B SFT on capability (tmax) + SafeComp safety data mixed in.
#
# Purpose: get the whole loop working end-to-end (train -> serve -> OAS eval -> paired-compare to
# base Qwen3.5-9B) and take a first peek at signal. NOT the attribution experiment — there is no
# capability-only B0 control here; that is deferred (see agent-safety/evals/TRAINING_PLAN.md).
#
# This is `sft_qwen35_9b_our_sft.sh` with exactly three changes, so the capability half stays a
# faithful match to the eventual B0 control:
#   1. SafeComp added to --dataset_mixer_list  (local jsonl, config `none`, 100% of it)
#   2. --num_train_epochs 2 -> 1               (smoke: one pass over the mixture)
#   3. --output_dir on weka                    (so vLLM-on-Beaker can serve the checkpoint)
#
# Mix at the current settings: tmax 0.2 (~3.3k examples) + SafeComp (~1k) => safety ~= 23% of the
# mixture. That is a heavier dose than the eventual 10-15% target, deliberately, to maximize
# visibility on the smoke; dial via SAFETY / CAP_FRAC below. NB "1.0" = 100% of the file, "1" = 1
# sample (open-instruct amount convention — a real footgun).

# NOTE (2026-07-21): the first launch failed because the OLD default image shashankg/open_instruct_auto
# (built 2026-04-30) predates the `--dataset_mixer_list_config_names` arg (added 2026-05-18 in
# 7447d65f1), so the training job's arg-parser rejected it — even though the local cache step passed
# (it ran the current venv). Fix: use a fresher image whose build commit INCLUDES the feature. Verified:
# this image = commit aeb75710c (2026-06-18), which contains 7447d65f1 (arg present 5x in its
# finetune.py). So standard --pure_docker_mode is fine; no code-mount workaround needed.
BEAKER_IMAGE="${1:-mingqianz/open-instruct-integration-test-agent-safety}"
echo "Using Beaker image: $BEAKER_IMAGE"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

# capability half (unchanged from the base recipe)
DATASET=osieosie/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all
CAP_FRAC=0.2

# safety half (SafeComp, produced by safecomp/pipeline + to_openinstruct_sft.py)
SAFECOMP=/weka/oe-adapt-default/mingqianz/agent-safety/data/sft/sft_comp_smoke1k_repaired.jsonl

OUT=/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-smoke-repaired

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "SFT Qwen3.5-9B: capability(tmax 0.2) + SafeComp safety smoke ~1k, REPAIRED endings (honest-closing ablation)" \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
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
    --exp_name sft_qwen35_9b_safecomp_smoke_repaired \
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
    # ^ 2026-07-21: the base capability recipe leaves these at their True defaults, which is fine on
    # an image that ships oe-eval-internal + HF creds. Our fresher image lacks oe-eval-internal, so
    # try_launch_beaker_eval_jobs=True crashed at finetune.py:1115 (before training). We don't want
    # open-instruct's auto-eval or an HF push anyway — we run OAS ourselves and serve from weka.
    # Mirrors the working sft_oas_qwen35_9b_safety.sh.
