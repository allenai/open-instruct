#!/usr/bin/env python3
"""
Minimal reproduction of the get_kv_cache_spec hang on hybrid models.

The issue: When using vLLM v1 with a hybrid model (OLMo 3.5 hybrid),
calling `engine.collective_rpc("get_kv_cache_spec")` hangs indefinitely.
This is due to multi-dtype serialization issues in vLLM v1.

Usage:
    python scripts/debug/repro_kv_cache_hang.py

Expected behavior: Should print KV cache specs
Actual behavior: Hangs at the collective_rpc call
"""

import asyncio
import signal
import sys

import torch
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs


MODEL_PATH = "/weka/oe-training-default/ai2-llm/checkpoints/willm/linear-rnns/OLMo3.1-7B-6T-30h-long-context-drope/step23842-hf"
TIMEOUT_SECONDS = 30


def timeout_handler(signum, frame):
    print(f"\n\nTIMEOUT: get_kv_cache_spec hung for {TIMEOUT_SECONDS} seconds")
    print("This confirms the multi-dtype serialization issue in vLLM v1")
    sys.exit(1)


async def main():
    print(f"Loading hybrid model from: {MODEL_PATH}")
    print(f"Timeout set to {TIMEOUT_SECONDS} seconds\n")

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )

    print("Creating AsyncLLMEngine...")
    engine = vllm.AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=False)
    print("Engine created successfully\n")

    print("Attempting to call collective_rpc('get_kv_cache_spec')...")
    print(f"(Will timeout after {TIMEOUT_SECONDS} seconds if it hangs)\n")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        kv_cache_specs = await engine.collective_rpc("get_kv_cache_spec")
        signal.alarm(0)
        print("SUCCESS: Got KV cache specs:")
        print(kv_cache_specs)
    except Exception as e:
        signal.alarm(0)
        print(f"ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
