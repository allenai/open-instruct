#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXP="${EXP:-}" \
bash "${SCRIPT_DIR}/qwen2.5_0.5b_gsm8k.sh" \
--log_train_solve_rate_metrics \
--eval_pass_at_k 32 \
--dataset_mixer_list mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-userprompt-quartiles 1.0 \
--dataset_mixer_list_splits test \
--dataset_mixer_eval_list mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-pass1-buckets 1.0 \
"$@"
