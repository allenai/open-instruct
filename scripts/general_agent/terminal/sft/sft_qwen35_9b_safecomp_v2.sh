#!/bin/bash
# SafeComp v2: Qwen3.5-9B SFT on capability (tmax) + the REGENERATED SafeComp safety data.
#
# Same experiment shape as `sft_qwen35_9b_safecomp_smoke.sh` (rung-1 smoke, 2026-07-25) so the two
# are directly comparable; the ONLY intended difference is the safety half's provenance:
#
#   smoke (2026-07-20): 450 tmax trajectories -> 1,045 pairs, pre-six-stage pipeline. Its rows carry
#     legacy self-authored observations, tail-authored closings, and the deterministic-checker
#     confirmation regime.
#   v2   (2026-08-06): 520 tmax trajectories on the six-stage pipeline — grounded critic, interleaved
#     author<->environment loop with independently simulated observations, independent closer,
#     separate safety and utility judges, `--gate stage5` passing.
#
# Capability half is byte-identical to smoke (same dataset, config, CAP_FRAC 0.2 ~ 3.3k rows), which
# is what makes a smoke-vs-v2 comparison a data-quality comparison rather than a mixture comparison.
#
# Amount convention (open-instruct footgun): "1.0" = 100% OF THE FILE, "1" = ONE sample. Do not drop
# the decimal point. Verify after launch by step arithmetic: steps x 16 = rows trained.
BEAKER_IMAGE="${1:-mingqianz/open-instruct-integration-test-agent-safety}"
echo "Using Beaker image: $BEAKER_IMAGE"

MODEL=hamishivi/Qwen3.5-9B
TOKENIZER=hamishivi/Qwen3.5-9B

# capability half — unchanged from smoke and from the B0 controls
DATASET=osieosie/tmax-sft-skill-tax-20260505-2.2k-combined-balanced-qwen3.6-27b-thinking
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all
CAP_FRAC=0.2

# safety half — regenerated corpus (safecomp/pipeline + to_openinstruct_sft.py --tag comp1k_v2)
SAFECOMP=/weka/oe-adapt-default/mingqianz/agent-safety/data/sft/sft_comp1k_v2.jsonl

OUT=/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "SFT Qwen3.5-9B: capability(tmax 0.2) + SafeComp v2 regenerated (~1k)" \
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
    --exp_name sft_qwen35_9b_safecomp_v2 \
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
    # try_launch_beaker_eval_jobs must stay false: this image lacks oe-eval-internal and the job
    # crashes at finetune.py:1115 before training. We run OAS ourselves and serve from weka.
