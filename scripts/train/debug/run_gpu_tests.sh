#!/bin/bash

BEAKER_IMAGE="${1:-finbarrt/open-instruct-integration-test}"

# Test vLLM directly without Ray actors to see if it works
VLLM_TEST='python3 -c "
import os
os.environ[\"VLLM_LOGGING_LEVEL\"] = \"DEBUG\"

import vllm
print(f\"vLLM version: {vllm.__version__}\")

# Create a simple LLM (not AsyncLLMEngine)
llm = vllm.LLM(
    model=\"Qwen/Qwen3-0.6B\",
    enforce_eager=True,
    max_model_len=512,
    gpu_memory_utilization=0.3,
)

# Generate
outputs = llm.generate([\"What is 2 + 2?\"], vllm.SamplingParams(max_tokens=10))
print(f\"Output: {outputs[0].outputs[0].text}\")
print(\"SUCCESS: vLLM works without Ray!\")
"'

uv run python mason.py \
    --cluster ai2/saturn \
    --image "$BEAKER_IMAGE" \
    --description "GPU test: vLLM direct test" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 30m \
    --budget ai2/oe-adapt \
    --gpus 1 \
    -- "$VLLM_TEST"
