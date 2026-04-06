#!/bin/bash
# SFT Qwen3.5-9B on tmax-sft-full-20260403 with default truncation.

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
EXP_NAME="${EXP_NAME:-sft_qwen35_9b_tmax_0403}"

DATASETS="hamishivi/tmax-sft-full-20260403 1.0"
DATASET_SPLITS="train"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "${EXP_NAME}" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name ${EXP_NAME} \
    --model_name_or_path Qwen/Qwen3.5-9B \
    --tokenizer_name Qwen/Qwen3.5-9B \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --dataset_transform_fn sft_tulu_tokenize_and_truncate_v1 sft_tulu_filter_v1 \
    --max_seq_length 32768 \
    --sequence_parallel_size 2 \
    --use_liger_kernel \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --seed 123 \
    --report_to wandb \
    --with_tracking \
    --hf_entity allenai \
    --wandb_entity ai2-llm \
    --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/hamishivi/ \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
