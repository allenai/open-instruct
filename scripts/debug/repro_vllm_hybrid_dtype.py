"""Minimal repro for vLLM hybrid model dtype serialization bug.

vllm 0.18.0 crashes when generating with allenai/Olmo-Hybrid-Instruct-DPO-7B
via the async engine with TP=2 and VLLM_ENABLE_V1_MULTIPROCESSING=0 (the path
used by Ray actors in GRPO training):
  msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`

The bug is in vllm/v1/kv_cache_interface.py where MambaSpec.dtypes is typed as
tuple[torch.dtype] (exactly 1 element), but hybrid models have 2 state tensors
with different dtypes. The multiprocess executor serializes this via msgspec
which enforces the tuple length.

Key conditions to trigger:
  1. VLLM_ENABLE_V1_MULTIPROCESSING=0 (set automatically inside Ray actors)
  2. tensor_parallel_size >= 2 (triggers multiprocess executor)
  3. A hybrid model with multiple Mamba state dtypes

Run on a GPU node with 2+ GPUs:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/debug/repro_vllm_hybrid_dtype.py
"""

import asyncio
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import vllm
from vllm import AsyncEngineArgs, AsyncLLMEngine


async def main():
    engine_args = AsyncEngineArgs(
        model="allenai/Olmo-Hybrid-Instruct-DPO-7B",
        enforce_eager=True,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    params = vllm.SamplingParams(max_tokens=16)
    result = await anext(engine.generate("Hello, world!", params, request_id="test"))
    print(result.outputs[0].text)


if __name__ == "__main__":
    asyncio.run(main())
