"""Compute max KV cache concurrency for different sequence lengths.

Loads each model once, profiles GPU memory, then computes the maximum
number of concurrent requests that fit in KV cache at each sequence length.

Usage:
    python scripts/benchmarking/kv_cache_concurrency.py \
        --model_name_or_path <model1> [model2] ... \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9 \
        --trust_remote_code \
        --sequence_lengths 1024 4096 8192 16384 32768
"""

import argparse
import gc
import os

# Force in-process engine core so we can access internals directly.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402
from vllm import LLM  # noqa: E402
from vllm.v1.core import kv_cache_utils  # noqa: E402


def profile_model(model_path, args, sequence_lengths):
    """Profile a single model and print its KV cache concurrency results."""
    max_seq_len = max(sequence_lengths)

    # Create LLM with the largest sequence length to profile GPU memory once.
    # enforce_eager=True skips CUDA graph warmup for faster initialization.
    llm = LLM(
        model=model_path,
        max_model_len=max_seq_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
    )

    # Access EngineCore through: LLM -> LLMEngine -> InprocClient -> EngineCore
    engine_core = llm.llm_engine.engine_core.engine_core

    # Retrieve the KV cache specs and profiled available memory.
    kv_cache_specs = engine_core.model_executor.get_kv_cache_specs()
    available_memory = [
        engine_core.available_gpu_memory_for_kv_cache
    ] * len(kv_cache_specs)
    vllm_config = engine_core.vllm_config

    # Build the KVCacheConfig once (num_blocks is independent of max_model_len).
    kv_cache_configs = kv_cache_utils.get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_memory
    )
    kv_cache_config = kv_cache_configs[0]

    original_max_model_len = vllm_config.model_config.max_model_len

    print("\n" + "=" * 60)
    print(f"Model: {model_path}")
    print(f"TP: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Available KV cache memory: {available_memory[0] / 1e9:.2f} GB")
    print(f"Num KV cache blocks: {kv_cache_config.num_blocks}")
    print(f"Num KV cache groups: {len(kv_cache_config.kv_cache_groups)}")
    for i, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        print(
            f"  Group {i}: {type(spec).__name__}, "
            f"{len(group.layer_names)} layers, "
            f"page_size={spec.page_size_bytes} bytes"
        )
    print("=" * 60)

    for seq_len in sequence_lengths:
        vllm_config.model_config.max_model_len = seq_len
        max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config(
            vllm_config, kv_cache_config
        )
        print(f"  max_model_len={seq_len:>6}: max_concurrency = {max_concurrency:.2f}x")

    print("=" * 60)

    # Restore original value.
    vllm_config.model_config.max_model_len = original_max_model_len

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Compute max KV cache concurrency at various sequence lengths."
    )
    parser.add_argument("--model_name_or_path", nargs="+", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--sequence_lengths",
        nargs="+",
        type=int,
        default=[1024, 4096, 8192, 16384, 32768],
    )
    args = parser.parse_args()

    sequence_lengths = sorted(args.sequence_lengths)

    for model_path in args.model_name_or_path:
        profile_model(model_path, args, sequence_lengths)


if __name__ == "__main__":
    main()
