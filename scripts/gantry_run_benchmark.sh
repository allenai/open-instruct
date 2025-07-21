#!/bin/bash
# Runs the benchmark on gantry. Takes one argument which is the response length.
# Usage: ./gantry_run_benchmark.sh [response_length]
# E.g. $ ./gantry_run_benchmark.sh 64000
set -e

# Set default value for response_length
response_length=64000

# If first argument exists and is a number, use it as response_length
if [[ "$1" =~ ^[0-9]+$ ]]; then
  response_length="$1"
  shift
fi

gantry run \
       --name open_instruct-benchmark_generators \
       --workspace ai2/oe-eval \
       --weka=oe-eval-default:/weka \
       --gpus 1 \
       --beaker-image nathanl/open_instruct_auto \
       --cluster ai2/jupiter-cirrascale-2 \
       --budget ai2/oe-eval \
       --install 'pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install packaging
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt' \
       -- python -m open_instruct.benchmark_generators \
    --model_name_or_path "hamishivi/qwen2_5_openthoughts2" \
    --tokenizer_name_or_path "hamishivi/qwen2_5_openthoughts2" \
    --dataset_mixer_list "hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2" "1.0" \
    --dataset_mixer_list_splits "train" \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --temperature 1.0 \
    --response_length "$response_length" \
    --vllm_top_p 0.9 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 16 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --pack_length 40000 \
    --chat_template_name "tulu_thinker" \
    --trust_remote_code \
    --seed 42 \
    --dataset_local_cache_dir "benchmark_cache" \
    --dataset_cache_mode "local" \
    --dataset_transform_fn "rlvr_tokenize_v1" "rlvr_filter_v1" \
    "$@"
