MODEL_NAME=/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115-retrofit
for LR in 8e-7
do
    EXP_NAME=olmo3-7b-DPO-1115-newbase-tpc-dedup5-${LR}-j
    python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
        --cluster ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
        --gs_model_name $EXP_NAME \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --image scottg/open_instruct_dev --pure_docker_mode \
        --preemptible \
        --num_nodes 2 \
        --budget ai2/oe-adapt \
        --no_auto_dataset_cache \
        --gpus 8 -- accelerate launch \
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
        --dataset_mixer_list allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5yolo_victoria_hates_code 125000 \
            allenai/dpo-yolo1-200k-gpt4.1-2w2s-maxdelta_reje-426124-rm-gemma3-kwd-ftd-ch-ftd-topic-ftd-dedup5 125000 \
            allenai/general_responses_dev_8maxturns_truncate-9fbef8-enrejected-keyword-filtered-cn-ftd-topic-filt 1250 \
            allenai/paraphrase_train_dev_8maxturns_truncated-6e031f-dorejected-keyword-filtered-cn-ftd-topic-filt 938 \
            allenai/repeat_tulu_5maxturns_big_truncated2048_victoriagrejected-keyword-filtered-cn-ftd-topic-filt 312 \
            allenai/self-talk_gpt3.5_gpt4o_prefpairs_truncat-1848c9-agrejected-keyword-filtered-cn-ftd-topic-filt 2500 \
            allenai/general-responses-truncated-gpt-dedup-topic-filt 1250 \
            allenai/paraphrase-truncated-gpt-dedup-topic-filt 938 \
            allenai/repeat-truncated-gpt-dedup-topic-filt 312 \
            allenai/self-talk-truncated-gpt-deduped-topic-filt 2500 \
        --max_seq_length 16384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
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
        --eval_workspace olmo-instruct \
        --eval_priority urgent \
        --oe_eval_max_length 32768 \
        --oe_eval_gpu_multiplier 2
done