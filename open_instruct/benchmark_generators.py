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
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import ray
import torch
import torch.utils.flop_counter
import transformers
import vllm

from open_instruct import dataset_transformation, grpo_fast, vllm_utils3
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args
from open_instruct.model_utils import ModelConfig
from open_instruct.utils import ArgumentParserPlus

# Set logging level based on DEBUG environment variable
log_level = logging.DEBUG
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


# Determine data directory
if Path("/weka").exists():
    DATA_DIR = Path("/weka") / "finbarrt" / "open_instruct_generators_benchmark"
else:
    DATA_DIR = Path("/tmp") / "open_instruct_generators_benchmark"


def save_completion_lengths(batch_results: List[Dict], timestamp: int):
    """
    Save completion lengths to CSV file.

    Args:
        batch_results: List of batch result dictionaries
        timestamp: Unix timestamp
    """
    csv_path = DATA_DIR / f"completion_lengths_{timestamp}.csv"

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["batch_num", "prompt_num", "completion_length"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for batch_result in batch_results:
            if "error" in batch_result:
                continue

            batch_idx = batch_result["batch_idx"]
            response_lengths = batch_result["response_lengths"]

            # response_lengths contains all responses (flattened from n samples per prompt)
            # We need to map them back to prompt indices
            for i, length in enumerate(response_lengths):
                writer.writerow({"batch_num": batch_idx, "prompt_num": i, "completion_length": length})

    logger.info(f"Saved completion lengths to {csv_path}")


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


# For FLOPS, we assume bf16 and ignore sparsity.
GPU_SPECS = {
    "a100": {"flops": 312e12, "memory_bandwidth": 1.6e12, "memory_size": 80e9},
    "b200": {"flops": 2250e12, "memory_bandwidth": 8.0e12, "memory_size": 192e9},
    "h100": {"flops": 990e12, "memory_bandwidth": 3.35e12, "memory_size": 80e9},
    "a6000": {"flops": 155e12, "memory_bandwidth": 768e9, "memory_size": 48e9},
    "l40s": {"flops": 362e12, "memory_bandwidth": 864e9, "memory_size": 48e9},
}


@dataclasses.dataclass
class ModelUsage:
    flops_per_token: int
    bytes_per_token: int

    def __str__(self):
        return (
            f"ModelUsage(flops_per_token={self.flops_per_token // 1e9} GFLOPs, "
            f"bytes_per_token={self.bytes_per_token // 1e12} TB/s)"
        )


def calculate_model_usage_per_token(model_path: str, sequence_length: int) -> ModelUsage:
    """
    Calculate actual FLOPs per token for a transformer model using torch FlopCounterMode.

    Args:
        model_path: Path to the actual model for precise measurement

    Returns:
        ModelUsage object.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    bytes_per_token = calculate_memory_bandwidth_bytes_per_token(model, sequence_length)

    # Create a single token input
    input_ids = torch.tensor([[1]], device=model.device)  # Single token

    model.eval()  # Set model to evaluation mode for consistent FLOPs counting
    flop_counter = torch.utils.flop_counter.FlopCounterMode(display=False, depth=None)
    with flop_counter:
        model(input_ids)

    flops = flop_counter.get_total_flops()

    # Clean up
    del model
    torch.cuda.empty_cache()

    return ModelUsage(flops_per_token=flops, bytes_per_token=bytes_per_token)


def get_device_name(device_name: str) -> str:
    logging.info(f"Original device name: {device_name}")
    processed_device_name = [
        val for val in device_name.lower().split(" ") if val not in ["nvidia", "80gb", "hbm3", "rtx"]
    ]
    if len(processed_device_name) != 1 or processed_device_name[0] not in GPU_SPECS:
        raise ValueError(f"Unsupported device name: {device_name}.")
    return processed_device_name[0]


def get_gpu_specs() -> float:
    """
    Get theoretical peak FLOPs for the current GPU.

    Returns:
        float: Peak FLOPs per second
    """
    device_name = get_device_name(torch.cuda.get_device_name(0))
    return GPU_SPECS[device_name]


def calculate_memory_bandwidth_bytes_per_token(model: torch.nn.Module, seq_len: int = 0) -> float:
    """
    Estimate memory‑bandwidth utilisation (MBU) **per GPU** for autoregressive
    decoding, including KV‑cache traffic.

    Parameters
    ----------
    model : torch.nn.Module
        The shard that lives on this GPU.
    seq_len : int, default 0
        Context length that the new token attends to (after the prompt).

    Returns
    -------
    ModelUsage object.
    """
    # --- infer dtype size ----------------------------------------------------
    dtype = next(model.parameters()).dtype
    if dtype in (torch.float16, torch.bfloat16):
        bytes_per_param = 2
    elif dtype == torch.float32:
        bytes_per_param = 4
    else:
        bytes_per_param = torch.finfo(dtype).bits // 8

    # --- cheap model introspection ------------------------------------------
    cfg = getattr(model, "config", None)
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
    n_layers = (
        getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
    )
    if d_model is None or n_layers is None:
        raise ValueError(
            "Cannot infer d_model / n_layers – pass a HuggingFace transformers model or supply these numbers manually."
        )

    # --- bytes that must cross HBM per *generated* token --------------------
    # 1. stream weights once
    params_per_gpu = sum(p.numel() for p in model.parameters())
    weight_bytes = params_per_gpu * bytes_per_param

    # 2. KV‑cache: write new k,v and read existing ones
    kv_write = 2 * n_layers * d_model * bytes_per_param
    kv_read = kv_write * seq_len
    bytes_per_token = weight_bytes + kv_write + kv_read

    return bytes_per_token


def calculate_wasted_time_percentage(prompt_lengths_by_batch: List[List[int]]) -> float:
    """
    Calculate the average wasted computation time due to variable prompt lengths.

    In vLLM, each batch processes tokens up to the max prompt length in that batch.
    Shorter prompts effectively waste compute by padding.

    Args:
        prompt_lengths_by_batch: List of lists, where each inner list contains
                                prompt lengths for a single batch

    Returns:
        float: Average wasted time percentage across all batches (0-100)
    """
    if not prompt_lengths_by_batch:
        return 0.0

    wasted_percentages = []

    for batch_prompt_lengths in prompt_lengths_by_batch:
        if not batch_prompt_lengths:
            continue

        max_prompt_length = max(batch_prompt_lengths)

        # Calculate wasted compute for this batch
        # Each prompt wastes (max_length - its_length) tokens worth of compute
        total_wasted_tokens = sum(max_prompt_length - length for length in batch_prompt_lengths)
        total_possible_tokens = max_prompt_length * len(batch_prompt_lengths)

        if total_possible_tokens > 0:
            batch_waste_percentage = (total_wasted_tokens / total_possible_tokens) * 100
            wasted_percentages.append(batch_waste_percentage)

    # Return average waste across all batches
    return sum(wasted_percentages) / len(wasted_percentages) if wasted_percentages else 0.0


def calculate_response_wasted_computation(masks_by_batch: List[List[List[int]]], max_tokens: int) -> float:
    """
    Calculate the average wasted computation due to responses finishing early.

    When responses finish before max_tokens (due to EOS token), the remaining
    token positions up to max_tokens represent wasted computation.

    Args:
        masks_by_batch: List of batches, where each batch contains a list of masks
                       (each mask is a list of 1s and 0s indicating valid tokens)
        max_tokens: The max_tokens parameter used during generation

    Returns:
        float: Average wasted computation percentage across all responses (0-100)
    """
    if not masks_by_batch or max_tokens <= 0:
        return 0.0

    total_wasted_tokens = 0
    total_possible_tokens = 0

    for batch_masks in masks_by_batch:
        for mask in batch_masks:
            # Count actual valid tokens (1s in mask)
            valid_tokens = sum(mask)
            # Wasted tokens = max_tokens - actual valid tokens
            wasted_tokens = max(0, max_tokens - valid_tokens)

            total_wasted_tokens += wasted_tokens
            total_possible_tokens += max_tokens

    if total_possible_tokens > 0:
        return (total_wasted_tokens / total_possible_tokens) * 100
    return 0.0


def setup_tokenizer(
    model_config: ModelConfig,
) -> Tuple[transformers.PreTrainedTokenizer, transformers.PretrainedConfig, float, dict[str, float]]:
    """Set up the tokenizer and model config."""
    logger.info(f"Loading tokenizer and config: {model_config.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def setup_dataset(args: Args, tokenizer_config: TokenizerConfig) -> datasets.Dataset:
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
    args: Args, model_config: ModelConfig, max_model_len: int = 20480
) -> List[ray.actor.ActorHandle]:
    """Set up vLLM engines."""
    logger.info("Setting up vLLM engines...")

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True,
             runtime_env={'excludes': ['/benchmark_cache/']})

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
    dataset: datasets.Dataset, args: Args, batch_idx: int
) -> Tuple[List[List[int]], List[str], List[str]]:
    """Get a batch of data from the dataset."""
    start_idx = batch_idx * args.num_unique_prompts_rollout
    end_idx = min(start_idx + args.num_unique_prompts_rollout, len(dataset))

    batch_data = dataset[start_idx:end_idx]

    # Extract prompts and ground truths
    prompts = batch_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    ground_truths = batch_data[dataset_transformation.GROUND_TRUTHS_KEY]
    datasets_list = batch_data[dataset_transformation.DATASET_SOURCE_KEY]

    # Don't duplicate prompts here - we'll use the 'n' parameter in SamplingParams
    return prompts, ground_truths, datasets_list


def run_generation_batch(
    inference_results_Q: queue.Queue,
    param_prompt_Q: queue.Queue,
    args: Args,
    model_usage: ModelUsage,
    gpu_specs: dict[str, float],
    prompts: List[List[int]],
    batch_idx: int,
) -> Dict[str, Union[float, int, List[str], List[int]]]:
    """Run generation for a batch of prompts and measure performance."""

    # Measure timing from prompt submission to result retrieval
    start_time = time.time()

    # Send prompts
    param_prompt_Q.put((None, prompts))

    # Get results
    result = inference_results_Q.get()

    end_time = time.time()
    generation_time = end_time - start_time

    if result[0] == "ERROR":
        logger.error(f"Generation failed: {result[1]}")
        return {"error": result[1]}

    response_ids, finish_reasons, masks, info = result
    collated_finish_reasons = collections.Counter(finish_reasons)
    logger.info("finish reasons: " + ", ".join(f"{k}: {v}" for k, v in collated_finish_reasons.items()))

    # Calculate tokens generated (response_ids only contains newly generated tokens)
    # When using n parameter, vLLM returns flattened responses as if prompts were duplicated
    total_new_tokens = sum(len(response) for response in response_ids)
    total_prompt_tokens = sum(len(prompt) for prompt in prompts) * args.num_samples_per_prompt_rollout
    total_tokens_generated = total_new_tokens + total_prompt_tokens

    tokens_per_second = total_new_tokens / generation_time if generation_time > 0 else 0
    total_tokens_per_second = total_tokens_generated / generation_time if generation_time > 0 else 0

    # Calculate MFU
    mfu_percentage = 100 * tokens_per_second * model_usage.flops_per_token / gpu_specs["flops"]
    memory_bandwidth_usage = tokens_per_second * model_usage.bytes_per_token
    mbu_percentage = 100 * memory_bandwidth_usage / gpu_specs["memory_bandwidth"]

    logger.info(f"  - Memory bandwidth usage: {memory_bandwidth_usage / 1e9:.2f} GB/s")
    logger.info(f"  - GPU memory bandwidth: {gpu_specs['memory_bandwidth'] / 1e12:.2f} TB/s")
    logger.info(f"  - MBU: {mbu_percentage:.2f}%")

    return {
        "batch_idx": batch_idx,
        "batch_size": len(response_ids),  # Total number of responses generated
        "generation_time": generation_time,
        "total_tokens_generated": total_tokens_generated,
        "total_new_tokens": total_new_tokens,
        "tokens_per_second": tokens_per_second,
        "total_tokens_per_second": total_tokens_per_second,
        "mfu_percentage": mfu_percentage,
        "mbu_percentage": mbu_percentage,
        "memory_bandwidth_usage_gb_s": memory_bandwidth_usage / 1e9,
        "avg_new_tokens_per_sample": total_new_tokens / len(response_ids) if response_ids else 0,
        "finish_reasons": finish_reasons,
        "response_lengths": [len(response) for response in response_ids],
        "prompt_lengths": [len(prompt) for prompt in prompts],  # Original unique prompts
        "masks": masks,  # Include masks for wasted computation calculation
        "max_tokens": args.response_length,  # Include max_tokens setting
    }


def run_benchmark(
    dataset: datasets.Dataset,
    vllm_engines: List[ray.actor.ActorHandle],
    args: Args,
    model_usage: ModelUsage,
    gpu_specs: dict[str, float],
    num_batches: int = 5,
) -> List[Dict[str, Union[float, int, List[str], List[int]]]]:
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

    all_results = []
    total_start_time = time.time()

    for batch_idx in range(num_batches):
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

        # Get batch data
        prompts, ground_truths, datasets_list = get_batch_data(dataset, args, batch_idx)

        if not prompts:
            logger.warning(f"No prompts in batch {batch_idx}, skipping")
            continue

        # Run generation
        batch_result = run_generation_batch(
            inference_results_Q, param_prompt_Q, args, model_usage, gpu_specs, prompts, batch_idx
        )

        if "error" not in batch_result:
            all_results.append(batch_result)
            logger.info(
                f"Batch {batch_idx + 1} completed: "
                f"{batch_result['tokens_per_second']:.2f} new tokens/sec, "
                f"MFU: {batch_result['mfu_percentage']:.2f}%, "
                f"MBU: {batch_result['mbu_percentage']:.2f}%, "
                f"{batch_result['generation_time']:.2f}s"
            )
        else:
            logger.error(f"Batch {batch_idx + 1} failed: {batch_result['error']}")

    total_time = time.time() - total_start_time

    # Send stop signal
    param_prompt_Q.put(None)

    # Wait for thread to finish
    thread.join(timeout=10)

    if thread.is_alive():
        logger.warning("Thread did not shutdown gracefully")

    # Calculate summary statistics
    if all_results:
        print_summary(all_results, total_time, args, model_usage, gpu_specs)
    else:
        logger.error("No successful batches completed")

    return all_results


def print_summary(
    results: List[Dict[str, Union[float, int, List[str], List[int]]]],
    total_time: float,
    args: Args,
    model_usage: ModelUsage,
    gpu_specs: dict[str, float],
) -> None:
    """Print benchmark summary statistics."""

    # Calculate metrics for all batches
    total_samples = sum(r["batch_size"] for r in results)
    total_new_tokens = sum(r["total_new_tokens"] for r in results)
    total_tokens = sum(r["total_tokens_generated"] for r in results)
    total_generation_time = sum(r["generation_time"] for r in results)

    # Collect all prompt lengths for statistics
    all_prompt_lengths = []
    all_response_lengths = []
    all_valid_token_counts = []  # Based on masks
    for r in results:
        all_prompt_lengths.extend(r["prompt_lengths"])
        all_response_lengths.extend(r["response_lengths"])
        # Calculate valid token counts from masks
        for mask in r["masks"]:
            all_valid_token_counts.append(sum(mask))

    avg_new_tokens_per_second = total_new_tokens / total_generation_time if total_generation_time > 0 else 0
    avg_total_tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
    avg_generation_time = total_generation_time / len(results)

    # Calculate average MFU and MBU
    avg_mfu = sum(r["mfu_percentage"] for r in results) / len(results) if results else 0
    avg_mbu = sum(r["mbu_percentage"] for r in results) / len(results) if results else 0

    throughput_samples_per_second = total_samples / total_time

    # Calculate metrics excluding the first batch (N-1 batches)
    if len(results) > 1:
        last_n_minus_1_results = results[1:]  # Skip first batch
        last_n_minus_1_samples = sum(r["batch_size"] for r in last_n_minus_1_results)
        last_n_minus_1_new_tokens = sum(r["total_new_tokens"] for r in last_n_minus_1_results)
        last_n_minus_1_tokens = sum(r["total_tokens_generated"] for r in last_n_minus_1_results)
        last_n_minus_1_generation_time = sum(r["generation_time"] for r in last_n_minus_1_results)

        avg_new_tokens_per_second_last_n_minus_1 = (
            last_n_minus_1_new_tokens / last_n_minus_1_generation_time if last_n_minus_1_generation_time > 0 else 0
        )
        avg_total_tokens_per_second_last_n_minus_1 = (
            last_n_minus_1_tokens / last_n_minus_1_generation_time if last_n_minus_1_generation_time > 0 else 0
        )
        avg_generation_time_last_n_minus_1 = last_n_minus_1_generation_time / len(last_n_minus_1_results)
        avg_mfu_last_n_minus_1 = sum(r["mfu_percentage"] for r in last_n_minus_1_results) / len(last_n_minus_1_results)
        avg_mbu_last_n_minus_1 = sum(r["mbu_percentage"] for r in last_n_minus_1_results) / len(last_n_minus_1_results)

        # Calculate % wasted time due to variable prompt lengths
        # Collect prompt lengths by batch for the last N-1 batches
        prompt_lengths_by_batch = [r["prompt_lengths"] for r in last_n_minus_1_results]
        prompt_wasted_time_percentage = calculate_wasted_time_percentage(prompt_lengths_by_batch)

        # Calculate % wasted computation due to responses finishing early
        # Collect masks by batch for the last N-1 batches
        masks_by_batch = [r["masks"] for r in last_n_minus_1_results]
        max_tokens = last_n_minus_1_results[0]["max_tokens"] if last_n_minus_1_results else args.response_length
        response_wasted_computation_percentage = calculate_response_wasted_computation(masks_by_batch, max_tokens)
    else:
        # If only one batch, use the same metrics
        avg_new_tokens_per_second_last_n_minus_1 = avg_new_tokens_per_second
        avg_total_tokens_per_second_last_n_minus_1 = avg_total_tokens_per_second
        avg_generation_time_last_n_minus_1 = avg_generation_time
        avg_mfu_last_n_minus_1 = avg_mfu
        avg_mbu_last_n_minus_1 = avg_mbu
        prompt_wasted_time_percentage = 0
        response_wasted_computation_percentage = 0

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model: {args.dataset_mixer_list[0] if args.dataset_mixer_list else 'Unknown'}")
    print(f"Total batches: {len(results)}")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout}")
    print(f"Unique prompts per batch: {args.num_unique_prompts_rollout}")
    print(f"Num rollouts: {args.num_unique_prompts_rollout}")
    print(f"Max tokens: {args.response_length}")
    print(f"Temperature: {args.temperature}")
    print("-" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total generation time: {total_generation_time:.2f}s")
    print(f"Total new tokens generated: {total_new_tokens}")
    print(f"Total tokens processed: {total_tokens}")
    print("-" * 60)
    if len(results) > 1:
        print("LAST N-1 BATCHES (excluding first batch):")
        print(f"Average new tokens/second: {avg_new_tokens_per_second_last_n_minus_1:.2f}")
        print(f"Average total tokens/second: {avg_total_tokens_per_second_last_n_minus_1:.2f}")
        print(f"Average MFU: {avg_mfu_last_n_minus_1:.2f}%")
        print(f"Average MBU: {avg_mbu_last_n_minus_1:.2f}%")
        print(f"Average generation time per batch: {avg_generation_time_last_n_minus_1:.2f}s")
        print(f"Wasted time (variable prompt lengths): {prompt_wasted_time_percentage:.2f}%")
        print(f"Wasted computation (early response termination): {response_wasted_computation_percentage:.2f}%")
        print(f"Average new tokens per sample: {last_n_minus_1_new_tokens / last_n_minus_1_samples:.2f}")
    else:
        print("RESULTS:")
        print(f"Average new tokens/second: {avg_new_tokens_per_second:.2f}")
        print(f"Average total tokens/second: {avg_total_tokens_per_second:.2f}")
        print(f"Average MFU: {avg_mfu:.2f}%")
        print(f"Average MBU: {avg_mbu:.2f}%")
        print(f"Average generation time per batch: {avg_generation_time:.2f}s")
        print(f"Throughput (samples/second): {throughput_samples_per_second:.2f}")
        print(f"Average new tokens per sample: {total_new_tokens / total_samples:.2f}")

    print("-" * 60)
    print("HARDWARE SPECIFICATIONS:")
    print(f"GPU peak FLOPs: {gpu_specs['flops'] / 1e12:.0f} TFLOPs")
    print(f"GPU memory bandwidth: {gpu_specs['memory_bandwidth'] / 1e12:.2f} TB/s")
    print(f"GPU memory size: {gpu_specs['memory_size'] / 1e9:.0f} GB")
    print("-" * 60)
    print("MODEL SPECIFICATIONS:")
    print(f"Model FLOPs per token: {model_usage.flops_per_token / 1e9:.2f} GFLOPs")
    print(f"Model bytes per token: {model_usage.bytes_per_token / 1e9:.2f} TB/s")

    # Bottleneck analysis
    print("-" * 60)
    print("BOTTLENECK ANALYSIS:")
    bottleneck_mfu = avg_mfu_last_n_minus_1 if len(results) > 1 else avg_mfu
    bottleneck_mbu = avg_mbu_last_n_minus_1 if len(results) > 1 else avg_mbu

    if bottleneck_mfu < bottleneck_mbu:
        print(f"System is COMPUTE BOUND (MFU: {bottleneck_mfu:.1f}% < MBU: {bottleneck_mbu:.1f}%)")
    else:
        print(f"System is MEMORY BANDWIDTH BOUND (MBU: {bottleneck_mbu:.1f}% < MFU: {bottleneck_mfu:.1f}%)")

    # Print completion length statistics
    if all_response_lengths:
        print("-" * 60)
        print("COMPLETION LENGTH STATISTICS:")
        print(f"Total completions: {len(all_response_lengths)}")

        # Raw response lengths (including padding)
        print("\nRaw response lengths (including any padding):")
        print(f"Min: {min(all_response_lengths)} tokens")
        print(f"Max: {max(all_response_lengths)} tokens")
        print(f"Mean: {np.mean(all_response_lengths):.2f} tokens")
        print(f"Median: {np.median(all_response_lengths):.2f} tokens")
        print(f"Std dev: {np.std(all_response_lengths):.2f} tokens")

        # Valid token counts (based on masks)
        if all_valid_token_counts:
            print("\nValid token counts (based on masks):")
            print(f"Min: {min(all_valid_token_counts)} tokens")
            print(f"Max: {max(all_valid_token_counts)} tokens")
            print(f"Mean: {np.mean(all_valid_token_counts):.2f} tokens")
            print(f"Median: {np.median(all_valid_token_counts):.2f} tokens")
            print(f"Std dev: {np.std(all_valid_token_counts):.2f} tokens")

            # Calculate percentiles for valid tokens
            print("\nValid token percentiles:")
            p25 = np.percentile(all_valid_token_counts, 25)
            p75 = np.percentile(all_valid_token_counts, 75)
            p90 = np.percentile(all_valid_token_counts, 90)
            p95 = np.percentile(all_valid_token_counts, 95)
            p99 = np.percentile(all_valid_token_counts, 99)

            print(f"25th percentile: {p25:.2f} tokens")
            print(f"75th percentile: {p75:.2f} tokens")
            print(f"90th percentile: {p90:.2f} tokens")
            print(f"95th percentile: {p95:.2f} tokens")
            print(f"99th percentile: {p99:.2f} tokens")

    print("=" * 60)


def cleanup(vllm_engines: Optional[List[ray.actor.ActorHandle]]) -> None:
    """Clean up resources."""
    if vllm_engines:
        for engine in vllm_engines:
            ray.kill(engine)
    if ray.is_initialized():
        ray.shutdown()


def compute_summary_stats(results: List[Dict[str, Union[float, int, List[str], List[int]]]]) -> Dict[str, float]:
    """
    Compute summary statistics from benchmark results.

    Returns a dictionary with:
    - mfu: Average MFU percentage (from last N-1 batches if available)
    - tokens_per_second: Average new tokens per second (from last N-1 batches if available)
    - wall_clock_time: Average generation time per batch (from last N-1 batches if available)
    - mbu: Average MBU percentage (from last N-1 batches if available)
    """
    if not results:
        return {"mfu": 0.0, "tokens_per_second": 0.0, "wall_clock_time": 0.0, "mbu": 0.0}

    # If we have more than one batch, use last N-1 batches (excluding first batch)
    if len(results) > 1:
        last_n_minus_1_results = results[1:]  # Skip first batch

        # Calculate averages from last N-1 batches
        total_new_tokens = sum(r["total_new_tokens"] for r in last_n_minus_1_results)
        total_generation_time = sum(r["generation_time"] for r in last_n_minus_1_results)

        avg_new_tokens_per_second = total_new_tokens / total_generation_time if total_generation_time > 0 else 0
        avg_generation_time = total_generation_time / len(last_n_minus_1_results)
        avg_mfu = sum(r["mfu_percentage"] for r in last_n_minus_1_results) / len(last_n_minus_1_results)
        avg_mbu = sum(r["mbu_percentage"] for r in last_n_minus_1_results) / len(last_n_minus_1_results)
    else:
        # Use the single batch results
        total_new_tokens = sum(r["total_new_tokens"] for r in results)
        total_generation_time = sum(r["generation_time"] for r in results)

        avg_new_tokens_per_second = total_new_tokens / total_generation_time if total_generation_time > 0 else 0
        avg_generation_time = total_generation_time / len(results)
        avg_mfu = sum(r["mfu_percentage"] for r in results) / len(results)
        avg_mbu = sum(r["mbu_percentage"] for r in results) / len(results)

    return {
        "mfu": avg_mfu,
        "tokens_per_second": avg_new_tokens_per_second,
        "wall_clock_time": avg_generation_time,
        "mbu": avg_mbu,
    }


def main() -> None:
    """Main benchmark function."""
    # Parse arguments using ArgumentParserPlus
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))

    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()

    # Generate timestamp for this run
    timestamp = int(time.time())

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save config first
    save_config(args, tokenizer_config, model_config, timestamp)

    dataset = setup_dataset(args, tokenizer_config)
    vllm_engines = setup_vllm_engines(args, model_config)

    model_usage = calculate_model_usage_per_token(
        model_config.model_name_or_path, sequence_length=args.response_length
    )
    gpu_specs = get_gpu_specs()

    # Run benchmark and get results
    results = run_benchmark(dataset, vllm_engines, args, model_usage, gpu_specs)

    # Save completion lengths
    if results:
        save_completion_lengths(results, timestamp)

    cleanup(vllm_engines)


if __name__ == "__main__":
    main()
