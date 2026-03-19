#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="${EXP_NAME:-qwen2.5_1.5b_instruct_gsm8k}" \
bash "${SCRIPT_DIR}/qwen2.5_0.5b_gsm8k.sh" --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct "$@"
