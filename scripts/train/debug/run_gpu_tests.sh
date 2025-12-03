#!/bin/bash

BEAKER_IMAGE="${1:-finbarrt/open-instruct-integration-test}"

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
    -- 'export VLLM_LOGGING_LEVEL=DEBUG && python3 -c "import vllm; print(vllm.__version__); llm = vllm.LLM(model=\"Qwen/Qwen3-0.6B\", enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.3); outputs = llm.generate([\"What is 2 + 2?\"], vllm.SamplingParams(max_tokens=10)); print(outputs[0].outputs[0].text); print(\"SUCCESS\")"'
