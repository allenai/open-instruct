#!/bin/bash
set -euo pipefail

# Example: sample teacher logits with Olmo-3-32B-Instruct.
# Matches core SFT settings from scripts/train/olmo3/7b_instruct_sft.sh where applicable:
# - sequence length: 32768
# - learning setup target: 2 epochs (applies to distill stage)
# - distributed scale target: 8 GPUs

MODEL_NAME="allenai/Olmo-3-32B-Instruct"
MODEL_REVISION="main"
SEQ_LEN=32768
NUM_GPUS=8
SEED=543210

# Use the same dataset used by 7b_instruct_sft.sh.
DATASET_MIXER_LIST="allenai/tulu-3-sft-personas-algebra 1.0"
DATASET_LOCAL_CACHE_DIR="local_dataset_cache"

# Output location for compressed teacher signals.
OUTPUT_DIR="./output/logits"
EXP_NAME="olmo3_32b_instruct_teacher_logits_32k"

# Compression config used for saved top-k teacher distributions.
COMPRESSION_CONFIG="configs/compression/default.yml"

LAUNCH_CMD="python open_instruct/sample_logits_vllm.py \
  --model_name_or_path ${MODEL_NAME} \
  --model_revision ${MODEL_REVISION} \
  --tokenizer_name_or_path ${MODEL_NAME} \
  --tokenizer_revision ${MODEL_REVISION} \
  --trust_remote_code \
  --compression_config ${COMPRESSION_CONFIG} \
  --dtype bfloat16 \
  --tensor_parallel_size ${NUM_GPUS} \
  --gpu_memory_utilization 0.9 \
  --dataset_mixer_list ${DATASET_MIXER_LIST} \
  --dataset_transform_fn sft_tulu_tokenize_and_truncate_v1 sft_tulu_filter_v1 \
  --dataset_target_columns input_ids attention_mask labels messages text \
  --dataset_cache_mode local \
  --dataset_local_cache_dir ${DATASET_LOCAL_CACHE_DIR} \
  --max_seq_length ${SEQ_LEN} \
  --macrobatch_size 256 \
  --seed ${SEED} \
  --output_dir ${OUTPUT_DIR} \
  --exp_name ${EXP_NAME} \
  --use_flat_logprobs"

echo "Running teacher logit sampling command:"
echo "${LAUNCH_CMD}"
uv run ${LAUNCH_CMD}
