"""Minimal repro for vLLM hybrid model dtype serialization bug.

vllm 0.18.0 crashes when calling collective_rpc("get_kv_cache_spec") on
allenai/Olmo-Hybrid-Instruct-DPO-7B with TP=2. This is the code path used
by our LLMRayActor.get_kv_cache_info() during engine initialization:
  msgspec.ValidationError: Expected `array` of length 1, got 2 - at `$.dtypes`

The bug is in vllm/v1/kv_cache_interface.py where MambaSpec.dtypes is typed as
tuple[torch.dtype] (exactly 1 element), but hybrid models have 2 state tensors
with different dtypes. The collective_rpc result is deserialized via msgspec's
"utility result" path which enforces the tuple length.

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
    kv_cache_specs = await engine.collective_rpc("get_kv_cache_spec")
    print(f"KV cache specs: {kv_cache_specs}")


if __name__ == "__main__":
    asyncio.run(main())
