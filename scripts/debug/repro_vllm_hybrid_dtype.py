"""Minimal repro for vLLM hybrid model dtype serialization bug.

vllm 0.18.0 crashes when generating with allenai/Olmo-Hybrid-Instruct-DPO-7B
via the async engine with TP=2 (the path used by Ray actors in GRPO training):
  msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`

Note: TP=1 sync path works fine. The bug is in the multiprocess serialization.

Run on a GPU node with 2+ GPUs:
  python scripts/debug/repro_vllm_hybrid_dtype.py
"""

import asyncio

import vllm
from vllm import AsyncEngineArgs, AsyncLLMEngine


async def main():
    engine_args = AsyncEngineArgs(
        model="allenai/Olmo-Hybrid-Instruct-DPO-7B",
        enforce_eager=True,
        tensor_parallel_size=2,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    params = vllm.SamplingParams(max_tokens=16)
    result = await anext(engine.generate("Hello, world!", params, request_id="test"))
    print(result.outputs[0].text)


if __name__ == "__main__":
    asyncio.run(main())
