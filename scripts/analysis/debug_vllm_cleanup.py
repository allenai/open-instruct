#!/usr/bin/env python3
"""
Debug vLLM memory cleanup between sequential loads.

Tests different cleanup approaches to find the best way to release GPU memory
between vLLM instances (needed for bf16→fp32 sequential runs).
"""
import gc
import os
import time

import torch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL = "Qwen/Qwen2.5-0.5B"  # Small model for fast testing
GPU_MEM = 0.5  # Use 50% to test cleanup


def get_memory_stats():
    """Get current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return allocated, reserved


def test_cleanup_method(method: str):
    """Test a specific cleanup method."""
    import vllm

    print(f"\n{'='*60}")
    print(f"Testing: {method}")
    print(f"{'='*60}")

    # Load vLLM
    print("\nLoading vLLM...")
    llm = vllm.LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=True,
    )

    # Generate something
    outputs = llm.generate(["Hello world"], vllm.SamplingParams(max_tokens=10))
    print(f"Generated: {outputs[0].outputs[0].text[:50]}...")

    alloc1, res1 = get_memory_stats()
    print(f"After load: allocated={alloc1:.2f}GB, reserved={res1:.2f}GB")

    # Apply cleanup method
    print(f"\nApplying cleanup: {method}")

    if method == "basic":
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    elif method == "shutdown_first":
        try:
            llm.llm_engine.shutdown()
        except Exception as e:
            print(f"  shutdown() error: {e}")
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    elif method == "force_sync":
        try:
            llm.llm_engine.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    elif method == "reset_peak":
        try:
            llm.llm_engine.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    elif method == "ipc_collect":
        try:
            llm.llm_engine.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    elif method == "aggressive":
        # Most aggressive cleanup
        try:
            llm.llm_engine.shutdown()
        except Exception:
            pass
        del llm

        # Multiple gc passes
        for _ in range(3):
            gc.collect()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

        # Wait for async cleanup
        time.sleep(2)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    time.sleep(1)
    alloc2, res2 = get_memory_stats()
    print(f"After cleanup: allocated={alloc2:.2f}GB, reserved={res2:.2f}GB")
    print(f"Freed: allocated={alloc1-alloc2:.2f}GB, reserved={res1-res2:.2f}GB")

    return alloc2, res2


def main():
    print("vLLM Memory Cleanup Debug")
    print(f"Model: {MODEL}")
    print(f"GPU Memory Utilization: {GPU_MEM}")

    methods = [
        "basic",
        "shutdown_first",
        "force_sync",
        "reset_peak",
        "ipc_collect",
        "aggressive",
    ]

    results = {}
    for method in methods:
        # Reset GPU state between tests
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

        alloc, res = test_cleanup_method(method)
        results[method] = (alloc, res)

        # Extra cleanup between tests
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Allocated (GB)':<15} {'Reserved (GB)':<15}")
    print("-"*50)
    for method, (alloc, res) in results.items():
        print(f"{method:<20} {alloc:<15.3f} {res:<15.3f}")


if __name__ == "__main__":
    main()
