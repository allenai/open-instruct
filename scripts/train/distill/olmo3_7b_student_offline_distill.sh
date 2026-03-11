#!/bin/bash
set -euo pipefail

# Example: distill Olmo-3-7B-Instruct from precomputed Olmo-3-32B-Instruct logits.
# This mirrors key settings from scripts/train/olmo3/7b_instruct_sft.sh:
# - seq_len=32768
# - lr=8e-5
# - num_train_epochs=2
# - 8 GPUs with DeepSpeed ZeRO-3
# - wandb tracking enabled by default

TEACHER_NAME="allenai/Olmo-3-32B-Instruct"
STUDENT_NAME="allenai/Olmo-3-7B-Instruct"
MODEL_REVISION="main"
SEQ_LEN=32768
NUM_GPUS=8
LR=8e-5
SEED=543210

# Point this to the dataset produced by olmo3_32b_teacher_sample_logits.sh.
# It can be a local path or a hosted dataset path supported by get_cached_dataset_tulu.
DISTILL_DATASET_MIXER_LIST="./output/logits/olmo3_32b_instruct_teacher_logits_32k 1.0"

DS_CONFIG_FILE="configs/ds_configs/stage3_no_offloading_accelerate.conf"
EXP_NAME="olmo3_7b_instruct_offline_distill_from_32b_32k"

COMPRESSION_CONFIG="configs/compression/default.yml"
LOSS_FUNCTIONS='[{"function":"cross_entropy","weight":1.0},{"function":"kl","weight":1.0,"temperature":1.0,"missing_probability_handling":"zero","sparse_chunk_length":1024}]'

LAUNCH_CMD="accelerate launch \
  --mixed_precision bf16 \
  --num_processes ${NUM_GPUS} \
  --use_deepspeed \
  --deepspeed_config_file ${DS_CONFIG_FILE} \
  --deepspeed_multinode_launcher standard \
  open_instruct/offline_distill.py \
  --exp_name ${EXP_NAME} \
  --model_name_or_path ${STUDENT_NAME} \
  --model_revision ${MODEL_REVISION} \
  --tokenizer_name_or_path ${STUDENT_NAME} \
  --tokenizer_revision ${MODEL_REVISION} \
  --trust_remote_code \
  --dataset_mixer_list ${DISTILL_DATASET_MIXER_LIST} \
  --dataset_transform_fn distill_pretokenized_v1 distill_pretokenized_filter_v1 \
  --dataset_target_columns input_ids attention_mask labels compressed_logprobs bytepacked_indices \
  --dataset_cache_mode local \
  --max_seq_length ${SEQ_LEN} \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --fused_optimizer \
  --low_cpu_mem_usage \
  --gradient_checkpointing \
  --learning_rate ${LR} \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --num_train_epochs 2 \
  --output_dir ./output/ \
  --logging_steps 1 \
  --seed ${SEED} \
  --timeout 3600 \
  --with_tracking \
  --report_to wandb \
  --wandb_project_name olmo3-distill \
  --compression_config ${COMPRESSION_CONFIG} \
  --loss_functions '${LOSS_FUNCTIONS}'"

echo "Teacher model: ${TEACHER_NAME}"
echo "Student model: ${STUDENT_NAME}"
echo "Running offline distillation command:"
echo "${LAUNCH_CMD}"
uv run ${LAUNCH_CMD}
