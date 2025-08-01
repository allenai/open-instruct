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
num_prompts=13686
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
       --model_name_or_path "Qwen/Qwen2.5-7B" \
       --dataset_mixer_list saurabh5/rlvr_acecoder_filtered ${num_prompts} saurabh5/synthetic2-rlvr-code-compressed ${num_prompts} \
       --dataset_mixer_list_splits "train" \
       --dataset_mixer_eval_list "saurabh5/rlvr_acecoder_filtered 8 saurabh5/synthetic2-rlvr-code-compressed 8" \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --temperature 1.0 \
    --response_length "$response_length" \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 256 \
    --vllm_num_engines 1 \
    --vllm_enable_prefix_caching \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --pack_length 70000 \
    --chat_template_name "tulu_thinker" \
    --trust_remote_code \
    --seed 42 \
    --dataset_skip_cache True
