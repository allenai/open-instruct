#!/usr/bin/env python3
"""
Benchmark script for testing vLLM generator performance.

This script loads datasets in the same way as grpo_fast.py, sets up a generator
like in test_grpo_fast.py, and streams results to/from the generator to measure
performance.
"""

import collections
import csv
import dataclasses
import gc
import json
import logging
import pathlib
import queue
import threading
import time
from typing import Any

import datasets
import numpy as np
import ray
import torch
import torch.utils.flop_counter
import transformers
import vllm

from open_instruct import dataset_transformation, grpo_fast, model_utils, utils, vllm_utils3

# For FLOPS, we assume bf16 and ignore sparsity.
GPU_SPECS = {
    "a100": {"flops": 312e12, "memory_size": 80e9},
    "b200": {"flops": 2250e12, "memory_size": 192e9},
    "h100": {"flops": 990e12, "memory_size": 80e9},
    "a6000": {"flops": 155e12, "memory_size": 48e9},
    "l40s": {"flops": 362e12, "memory_size": 48e9},
}


logger = logging.getLogger(__name__)


# Determine data directory
if pathlib.Path("/weka").exists():
    DATA_DIR = pathlib.Path("/weka") / "finbarrt" / "open_instruct_generators_benchmark"
else:
    DATA_DIR = pathlib.Path("/tmp") / "open_instruct_generators_benchmark"


def save_completion_lengths(batch_results: list[dict], timestamp: int, batch_idx: int):
    """
    Save completion lengths to CSV file.

    Args:
        batch_results: List of batch result dictionaries
        timestamp: Unix timestamp
    """
    csv_path = DATA_DIR / f"completion_lengths_{timestamp}.csv"

    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = ["batch_num", "prompt_num", "completion_length"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for batch_result in batch_results:
            response_lengths = batch_result["response_lengths"]
            for i, length in enumerate(response_lengths):
                writer.writerow({"batch_num": batch_idx, "prompt_num": i, "completion_length": length})
    logger.info(f"Saved completion lengths to {csv_path}.")


def save_config(args, tokenizer_config, model_config, timestamp: int):
    """
    Save configuration to JSON file.

    Args:
        args: Args dataclass
        tokenizer_config: TokenizerConfig dataclass
        model_config: ModelConfig dataclass
        timestamp: Unix timestamp
    """
    config_path = DATA_DIR / f"config_{timestamp}.json"

    # Convert dataclasses to dicts
    config_dict = {
        "args": dataclasses.asdict(args),
        "tokenizer_config": dataclasses.asdict(tokenizer_config),
        "model_config": dataclasses.asdict(model_config),
        "timestamp": timestamp,
    }

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Saved config to {config_path}")


def free_all_gpu_memory(device: int | str = 0) -> None:
    """
    Aggressively free GPU memory used by PyTorch.

    What it does
    ------------
    1. Runs Python's garbage collector to drop un-referenced tensors.
    2. Calls torch.cuda.empty_cache() to let CUDA release cached blocks
       back to the driver.
    3. Calls torch.cuda.ipc_collect() to close any shared-memory handles
       created by multiprocessing.
    4. (Optional) Resets PyTorch's memory-stats counters so you start
       with a clean slate for new measurements.

    Parameters
    ----------
    device : int | str, default 0
        GPU index or explicit device string like "cuda:1".
    """
    dev = torch.device(device if isinstance(device, str) else f"cuda:{device}")

    # 1. Clear Python references & collect garbage
    gc.collect()

    # 2. Synchronize to make sure all kernels are done
    torch.cuda.synchronize(dev)

    # 3. Release cached blocks held by the CUDA caching allocator
    torch.cuda.empty_cache()

    # 4. Destroy any lingering CUDA IPC shared-memory handles
    torch.cuda.ipc_collect()

    # Optional: print the end result
    free, total = torch.cuda.mem_get_info(dev)
    gib = 1024**3
    logger.info(f"[GPU {dev.index}] {free / gib:.2f} GiB free of {total / gib:.2f} GiB after cleanup")


def calculate_model_usage_per_token(model_path: str) -> int:
    """
    Calculate actual FLOPs per token for a transformer model using torch FlopCounterMode.

    Args:
        model_path: Path to the actual model for precise measurement

    Returns:
        FLOPs per token as integer.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )

    # Create a single token input
    input_ids = torch.tensor([[1]], device=model.device)  # Single token

    model.eval()  # Set model to evaluation mode for consistent FLOPs counting
    flop_counter = torch.utils.flop_counter.FlopCounterMode(display=False, depth=None)
    with flop_counter:
        model(input_ids)

    return flop_counter.get_total_flops()


def get_device_name(device_name: str) -> str:
    processed_device_name = [
        val for val in device_name.lower().split(" ") if val not in ["nvidia", "80gb", "hbm3", "rtx"]
    ]
    if len(processed_device_name) != 1 or processed_device_name[0] not in GPU_SPECS:
        raise ValueError(f"Unsupported device name: {device_name}.")
    return processed_device_name[0]


def setup_dataset(args: grpo_fast.Args, tokenizer_config: dataset_transformation.TokenizerConfig) -> datasets.Dataset:
    """Set up the dataset using the same pipeline as grpo_fast.py."""
    logger.info("Loading and processing dataset...")

    # Transform function arguments
    transform_fn_args = [
        {},  # For rlvr_tokenize_v1
        {
            "max_token_length": args.max_token_length,
            "max_prompt_token_length": args.max_prompt_token_length,
        },  # For rlvr_filter_v1
    ]

    # Load dataset
    dataset = dataset_transformation.get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tokenizer_config,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )

    # Shuffle dataset
    dataset = dataset.shuffle(seed=args.seed)
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    return dataset


def setup_vllm_engines(
    args: grpo_fast.Args, model_config: model_utils.ModelConfig, max_model_len: int = 20480
) -> list[ray.actor.ActorHandle]:
    """Set up vLLM engines."""
    logger.info("Setting up vLLM engines...")

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True, runtime_env={"excludes": ["/benchmark_cache/"]})

    # Create placement group for multiple engines
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.vllm_num_engines)]
    pg = ray.util.placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    # Create vLLM engines
    vllm_engines = vllm_utils3.create_vllm_engines(
        num_engines=args.vllm_num_engines,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        enforce_eager=True,
        tokenizer_name_or_path=model_config.model_name_or_path,
        pretrain=model_config.model_name_or_path,
        revision=model_config.model_revision,
        seed=args.seed,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        single_gpu_mode=False,
        pg=pg,
        tools={},
        max_tool_calls=[0],
    )

    logger.info("vLLM engines ready")

    return vllm_engines


def get_batch_data(
    dataset: datasets.Dataset, batch_size: int, batch_idx: int
) -> tuple[list[list[int]], list[str], list[str]]:
    """Get a batch of data from the dataset."""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))

    batch_data = dataset[start_idx:end_idx]
    prompts = batch_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    return prompts


def run_generation_batch(
    inference_results_Q: queue.Queue, param_prompt_Q: queue.Queue, prompts: list[list[int]]
) -> dict[str, Any]:
    """Run generation for a batch of prompts and measure performance."""

    start_time = time.time()
    param_prompt_Q.put((None, prompts))
    result = inference_results_Q.get()
    generation_time = time.time() - start_time

    response_ids, finish_reasons, _, _ = result
    collated_finish_reasons = collections.Counter(finish_reasons)

    new_tokens = sum(len(response) for response in response_ids)
    tokens_per_second = new_tokens / generation_time
    return {
        "tokens_per_second": tokens_per_second,
        "generation_time": generation_time,
        "num_new_tokens": new_tokens,
        # dict mapping string reasons to counts.
        "finish_reasons": collated_finish_reasons,
        "response_lengths": [len(response) for response in response_ids],
        "prompt_lengths": [len(prompt) for prompt in prompts],  # Original unique prompts
    }


def run_benchmark(
    dataset: datasets.Dataset,
    vllm_engines: list[ray.actor.ActorHandle],
    args: grpo_fast.Args,
    model_config: model_utils.ModelConfig,
    timestamp: int,
    num_batches: int = 5,
) -> list[dict[str, Any]]:
    """Run the full benchmark."""
    logger.info(f"Starting benchmark with {num_batches} batches of size {args.num_unique_prompts_rollout}")

    # Create persistent queues
    inference_results_Q = queue.Queue(maxsize=10)
    param_prompt_Q = queue.Queue(maxsize=10)
    evaluation_inference_results_Q = queue.Queue(maxsize=10)

    # Create sampling parameters with 'n' for multiple samples per prompt
    generation_config = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.response_length,
        top_p=args.vllm_top_p,
        n=args.num_samples_per_prompt_rollout,
    )

    eval_generation_config = vllm.SamplingParams(temperature=0.0, max_tokens=args.response_length, n=1)

    # We have to do this before starting vLLM as otherwise we get OOM errors.
    flops_per_token = calculate_model_usage_per_token(model_config.model_name_or_path)

    # Unclear why we need this. We didn't need it before torch 2.7.0.
    free_all_gpu_memory()

    # Start persistent vLLM generation thread
    def wrapped_vllm_generate_thread() -> None:
        grpo_fast.vllm_generate_thread(
            vllm_engines,
            generation_config,
            eval_generation_config,
            inference_results_Q,
            param_prompt_Q,
            num_batches,  # num_training_steps
            None,  # eval_prompt_token_ids
            evaluation_inference_results_Q,
            num_batches + 1,  # eval_freq (avoid evaluation)
            1,  # resume_training_step
            False,  # tool_use
        )

    thread = threading.Thread(target=wrapped_vllm_generate_thread)
    thread.start()

    # Wait for thread to be ready
    time.sleep(0.1)

    results = []
    total_start_time = time.time()
    device_name = get_device_name(torch.cuda.get_device_name(0))
    device_flops = GPU_SPECS[device_name]["flops"]
    for batch_idx in range(num_batches):
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

        prompts = get_batch_data(dataset, args.num_unique_prompts_rollout, batch_idx)

        # Run generation
        result = run_generation_batch(inference_results_Q, param_prompt_Q, prompts)
        result["mfu"] = 100 * result["tokens_per_second"] * flops_per_token / device_flops
        # We incrementally save completion lengths so even if the job dies, we still have data.
        save_completion_lengths([result], timestamp, batch_idx)
        results.append(result)
        logger.info(
            f"Batch {batch_idx + 1} completed: "
            f"{result['tokens_per_second']:.2f} new tokens/sec, "
            f"MFU: {result['mfu']:.2f}%, "
            f"{result['generation_time']:.2f}s"
        )

    total_time = time.time() - total_start_time

    # Send stop signal
    param_prompt_Q.put(None)

    # Wait for thread to finish
    thread.join(timeout=10)

    print_summary(results, total_time, args, model_config)


def average_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate average metrics from results."""
    averaged_results = {
        "mfu": 0.0,
        "tokens_per_second": 0.0,
        "generation_time": 0.0,
        "num_new_tokens": 0,
        "finish_reasons": collections.defaultdict(int),
        "response_lengths": [],
        "prompt_lengths": [],
    }
    for result in results:
        for key, value in result.items():
            if key in ["mfu", "tokens_per_second", "generation_time", "num_new_tokens"]:
                averaged_results[key] += value
            elif key == "finish_reasons":
                for reason, count in value.items():
                    averaged_results["finish_reasons"][reason] += count
            elif key == "response_lengths":
                averaged_results["response_lengths"].extend(value)
            elif key == "prompt_lengths":
                averaged_results["prompt_lengths"].extend(value)
    return {k: v / len(results) if isinstance(v, (int, float)) else v for k, v in averaged_results.items()}


def print_summary(
    results: list[dict[str, Any]], total_time: float, args: grpo_fast.Args, model_config: model_utils.ModelConfig
) -> None:
    """Print benchmark summary statistics."""

    # Calculate metrics for all batches
    total_tokens = sum(r["num_new_tokens"] for r in results)
    total_generation_time = sum(r["generation_time"] for r in results)

    # Skip the first batch as it's unrepresentative thanks to warmup time.
    # fields needed:
    avg_results = average_results(results[1:])

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model: {model_config.model_name_or_path}")
    print(f"Total batches: {len(results)}")
    print(f"Batch size: {args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout}")
    print(f"Unique prompts per batch: {args.num_unique_prompts_rollout}")
    print(f"Num rollouts: {args.num_unique_prompts_rollout}")
    print(f"Max tokens: {args.response_length}")
    print("-" * 60)
    print(f"Total time: {total_time:.2f}s ({total_generation_time / total_time:.2f}% generating)")
    print(f"Total new tokens generated: {total_tokens}")
    print("-" * 60)
    print("Results (excluding first batch):")
    print(f"Average tokens/second: {avg_results['tokens_per_second']:.2f}")
    print(f"Average MFU: {avg_results['mfu']:.2f}%")
    print(f"Average generation time per batch: {avg_results['generation_time']:.2f}s")
    print(f"Average new tokens per sample: {avg_results['num_new_tokens']} tokens")

    max_length = np.max(avg_results["response_lengths"])
    mean_length = np.mean(avg_results["response_lengths"])
    wasted_compute = 100 * (max_length - mean_length) / max_length
    print(f"Wasted compute % (variable response length): {wasted_compute:.2%}%")

    print("-" * 60)
    print("HARDWARE SPECIFICATIONS:")
    gpu_specs = GPU_SPECS[get_device_name(torch.cuda.get_device_name(0))]
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU peak FLOPs: {gpu_specs['flops'] / 1e12:.0f} TFLOPs")
    print(f"GPU memory size: {gpu_specs['memory_size'] / 1e9:.0f} GB")

    print("-" * 60)
    print("COMPLETION LENGTH STATISTICS:")
    print(f"Total completions: {len(avg_results['response_lengths'])}")
    print("\nResponse lengths:")
    print(f"- Min: {min(avg_results['response_lengths'])} tokens")
    print(f"- Max: {max(avg_results['response_lengths'])} tokens")
    print(f"- Mean: {np.mean(avg_results['response_lengths']):.2f} tokens")
    print(f"- Median: {np.median(avg_results['response_lengths']):.2f} tokens")

    # Calculate percentiles for valid tokens
    print("\nResponse length percentiles:")
    print(f"- 25th percentile: {np.percentile(avg_results['response_lengths'], 25):.2f} tokens")
    print(f"- 50th percentile: {np.percentile(avg_results['response_lengths'], 50):.2f} tokens")
    print(f"- 75th percentile: {np.percentile(avg_results['response_lengths'], 75):.2f} tokens")
    print(f"- 90th percentile: {np.percentile(avg_results['response_lengths'], 90):.2f} tokens")
    print(f"- 95th percentile: {np.percentile(avg_results['response_lengths'], 95):.2f} tokens")
    print(f"- 99th percentile: {np.percentile(avg_results['response_lengths'], 99):.2f} tokens")

    print("=" * 60)


def cleanup(vllm_engines: list[ray.actor.ActorHandle]) -> None:
    """Clean up resources."""
    for engine in vllm_engines:
        ray.kill(engine)
    if ray.is_initialized():
        ray.shutdown()


def main() -> None:
    """Main benchmark function."""
    # Parse arguments using ArgumentParserPlus
    parser = utils.ArgumentParserPlus(
        (grpo_fast.Args, dataset_transformation.TokenizerConfig, model_utils.ModelConfig)
    )

    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = setup_dataset(args, tokenizer_config)
    vllm_engines = setup_vllm_engines(args, model_config)

    # Create the timestamp here so we use it for both filenames.
    timestamp = int(time.time())
    save_config(args, tokenizer_config, model_config, timestamp)
    run_benchmark(dataset, vllm_engines, args, model_config, timestamp)

    cleanup(vllm_engines)


if __name__ == "__main__":
    main()
