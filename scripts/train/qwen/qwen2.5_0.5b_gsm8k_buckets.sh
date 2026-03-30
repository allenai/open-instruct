#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-qwen2.5_0.5b_instruct_gsm8k_buckets8}" \
LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-pass1-buckets 1.0}" \
bash "${SCRIPT_DIR}/qwen2.5_0.5b_gsm8k.sh" --eval_pass_at_k 32 "$@"
