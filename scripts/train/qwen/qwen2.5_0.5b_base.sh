#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-qwen2.5_0.5b_base_gsm8k_pass1buckets}" \
bash "${SCRIPT_DIR}/qwen2.5_0.5b_gsm8k.sh" \
--eval_pass_at_k 32 \
--dataset_mixer_list mnoukhov/gsm8k-platinum-qwen2.5-0.5b-base-1024samples 1.0 \
--dataset_mixer_list_splits test \
--dataset_mixer_eval_list mnoukhov/gsm8k-platinum-qwen2.5-0.5b-base-1024samples-buckets 1.0 "$@"
