#!/bin/bash
# Wandb: https://wandb.ai/ai2-llm/open_instruct_internal/runs/07o8dec7
# Beaker: https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KA9G0T8C8Y4RVN691AEETNJD
# Commit: 2fd104e7

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
MODEL_NAME=/weka/oe-adapt-default/jacobm/olmo3/32b-merge-configs/instruct-checkpoints/olmo3-32b-instruct-SFT-1114-fix-8e-5-seed_33333/step814-hf
NUM_NODES=8
LR=1e-6
EXP_NAME="olmo3-32b-DPO-1116-match-7b-${LR}"
DATASET_MIXER_LIST="
allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 125000
allenai/dpo-yolo1-200k-gpt4.1-2w2s-maxdelta_reje-426124-rm-gemma3-kwd-ftd-ch-ftd-topic-ftd-dedup5-lbc100 125000
allenai/general_responses_dev_8maxturns_truncate-9fbef8-enrejected-kwd-ftd-cn-ftd-topic-filt-lb100 1250
allenai/paraphrase_train_dev_8maxturns_truncated-6e031f-dorejected-kwd-ftd-cn-ftd-topic-filt-lb100 938
allenai/repeat_tulu_5maxturns_big_truncated2048_victoriagrejected-kwd-ftd-cn-ftd-topic-filt-lb100 312
allenai/self-talk_gpt3.5_gpt4o_prefpairs_truncat-1848c9-agrejected-kwd-ftd-cn-ftd-topic-filt-lb100 2500
allenai/general-responses-truncated-gpt-dedup-topic-filt-lb100 1250
allenai/paraphrase-truncated-gpt-dedup-topic-filt-lb100 938
allenai/repeat-truncated-gpt-dedup-topic-filt-lb100 312
allenai/self-talk-truncated-gpt-deduped-topic-filt-lb100 2500
"
OE_EVAL_TASKS="omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek"

uv run python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --image $BEAKER_IMAGE --pure_docker_mode \
    --num_nodes $NUM_NODES \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 64 \
    --num_machines 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer False \
    --dataset_mixer_list $DATASET_MIXER_LIST \
    --max_seq_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --zero_hpz_partition_size 1 \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --use_flash_attn \
    --gradient_checkpointing \
    --report_to wandb \
    --chat_template_name olmo123 \
    --with_tracking \
    --eval_workspace ai2/olmo-instruct \
    --eval_priority urgent \
    --oe_eval_max_length 32768 \
    --oe_eval_gpu_multiplier 4 \
    --oe_eval_tasks "$OE_EVAL_TASKS" \
    --hf_entity allenai \
    --wandb_entity ai2-llm