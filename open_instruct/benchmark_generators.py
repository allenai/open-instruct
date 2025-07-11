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
import os
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


# Peak FLOPs for common GPUs (bfloat16/float16)
GPU_PEAK_FLOPS = {
    "a100": 312e12,  # 312 TFLOPs for bf16
    "b200": 2250e12,  # 2250 TFLOPS for bf16.
    "h100": 990e12,  # 990 TFLOPs for bf16
    "a6000": 155e12,  # 155 TFLOPS for bf16
    "l40s": 362e12,
}

# Memory bandwidth for common GPUs (bytes/second)
GPU_MEMORY_BANDWIDTH = {
    "a100": 1.6e12,  # 1.6 TB/s for A100 40GB/80GB
    "b200": 8.0e12,  # 8.0 TB/s for B200
    "h100": 3.35e12,  # 3.35 TB/s for H100
    "a6000": 768e9,  # 768 GB/s for A6000
    "l40s": 864e9,  # 864 GB/s for L40S
}

# GPU memory sizes (bytes)
GPU_MEMORY_SIZE = {
    "a100": 80e9,  # 80GB variant (also 40GB variant exists)
    "b200": 192e9,  # 192GB for B200
    "h100": 80e9,  # 80GB for H100
    "a6000": 48e9,  # 48GB for A6000
    "l40s": 48e9,  # 48GB for L40S
}


def calculate_model_flops_per_token(model_config: transformers.PretrainedConfig, model_path: str) -> float:
    """
    Calculate actual FLOPs per token for a transformer model using torch FlopCounterMode.

    Args:
        model_config: HuggingFace model config
        model_path: Path to the actual model for precise measurement

    Returns:
        float: FLOPs per token
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

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

    return flops


def get_device_name(device_name: str) -> str:
    logging.debug(f"Original device name: {device_name}")
    return device_name.lower().split(" ")[-1].replace(" ", "").replace("-", "")


def get_gpu_peak_flops() -> float:
    """
    Get theoretical peak FLOPs for the current GPU.

    Returns:
        float: Peak FLOPs per second
    """
    if not torch.cuda.is_available():
        return 0.0

    device_name = torch.cuda.get_device_name(0).lower().replace(" ", "").replace("-", "")

    device_name = get_device_name(torch.cuda.get_device_name(0))
    return GPU_PEAK_FLOPS[device_name]


def get_gpu_memory_bandwidth() -> float:
    """
    Get theoretical memory bandwidth for the current GPU.

    Returns:
        float: Memory bandwidth in bytes per second
    """
    if not torch.cuda.is_available():
        return 0.0

    device_name = get_device_name(torch.cuda.get_device_name(0))
    return GPU_MEMORY_BANDWIDTH[device_name]


def get_gpu_memory_size() -> float:
    """
    Get memory size for the current GPU.

    Returns:
        float: GPU memory in bytes
    """
    if not torch.cuda.is_available():
        return 0.0

    device_name = get_device_name(torch.cuda.get_device_name(0))
    return GPU_MEMORY_SIZE[device_name]


def calculate_mfu(tokens_per_second: float, model_flops_per_token: float, gpu_peak_flops: float) -> float:
    """
    Calculate Model FLOPs Utilization (MFU).

    Args:
        tokens_per_second: Actual token generation rate
        model_flops_per_token: Theoretical FLOPs per token for the model
        gpu_peak_flops: Peak FLOPs of the GPU

    Returns:
        float: MFU as a percentage (0-100)
    """
    actual_flops_per_second = tokens_per_second * model_flops_per_token
    mfu_percentage = (actual_flops_per_second / gpu_peak_flops) * 100

    return mfu_percentage


def estimate_kv_cache_memory_per_token(
    model_config: transformers.PretrainedConfig,
    dtype_bytes: int = 2,  # 2 bytes for bf16/fp16
) -> float:
    """
    Estimate KV cache memory requirement per token.

    Args:
        model_config: Model configuration with architecture details
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32)

    Returns:
        float: KV cache memory in bytes per token
    """
    # Get model dimensions
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_hidden_layers

    # Handle different attention head configurations
    if hasattr(model_config, "num_key_value_heads"):
        # For models with multi-query or grouped-query attention
        num_kv_heads = model_config.num_key_value_heads
    else:
        # Standard multi-head attention
        num_kv_heads = model_config.num_attention_heads

    # KV cache per token = 2 (K and V) * num_layers * hidden_size * (kv_heads/total_heads) * dtype_bytes
    # Note: hidden_size is already the full model dimension, we need per-head dimension
    head_dim = hidden_size // model_config.num_attention_heads
    kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

    return kv_cache_per_token


def calculate_memory_bandwidth_usage(
    tokens_per_second: float,
    kv_cache_per_token: float,
    model_size_bytes: float,
    avg_sequence_length: float,
    batch_size: int,
) -> float:
    """
    Calculate actual memory bandwidth usage.

    Args:
        tokens_per_second: Token generation rate
        kv_cache_per_token: KV cache memory per token in bytes
        model_size_bytes: Total model size in bytes
        avg_sequence_length: Average sequence length being processed
        batch_size: Batch size

    Returns:
        float: Memory bandwidth usage in bytes per second
    """
    # KV cache reads/writes per token generated
    # For each token: read K,V for all layers, write new K,V
    kv_bandwidth = tokens_per_second * kv_cache_per_token * 2  # read + write

    # Model weights are read for each forward pass
    # Each token generation requires a full forward pass
    weight_bandwidth = model_size_bytes * tokens_per_second / batch_size

    # Additional memory access patterns
    # Including: attention scores, intermediate activations, layer norms, etc.
    # Estimated as 4x the KV cache size (more realistic for transformer models)
    activation_bandwidth = kv_bandwidth * 4

    total_bandwidth = kv_bandwidth + weight_bandwidth + activation_bandwidth

    return total_bandwidth


def calculate_mbu(memory_bandwidth_usage: float, gpu_memory_bandwidth: float) -> float:
    """
    Calculate Memory Bandwidth Utilization (MBU).

    Args:
        memory_bandwidth_usage: Actual memory bandwidth usage in bytes/second
        gpu_memory_bandwidth: Theoretical peak memory bandwidth in bytes/second

    Returns:
        float: MBU as a percentage (0-100)
    """
    mbu_percentage = (memory_bandwidth_usage / gpu_memory_bandwidth) * 100
    return mbu_percentage


def estimate_model_size_bytes(model_config: transformers.PretrainedConfig, dtype_bytes: int = 2) -> float:
    """
    Estimate model size in bytes based on config.

    Args:
        model_config: Model configuration
        dtype_bytes: Bytes per parameter (2 for bf16/fp16)

    Returns:
        float: Estimated model size in bytes
    """
    # Rough estimation based on model architecture
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_hidden_layers
    vocab_size = model_config.vocab_size

    # Embedding layers
    embedding_params = vocab_size * hidden_size * 2  # input and output embeddings

    # Attention layers (Q, K, V, O projections)
    attention_params = num_layers * 4 * hidden_size * hidden_size

    # MLP layers (typically 4x hidden size for intermediate)
    intermediate_size = getattr(model_config, "intermediate_size", 4 * hidden_size)
    mlp_params = num_layers * 2 * hidden_size * intermediate_size

    # Layer norms and other small components (rough estimate)
    other_params = num_layers * 2 * hidden_size * 2

    total_params = embedding_params + attention_params + mlp_params + other_params
    total_bytes = total_params * dtype_bytes

    return total_bytes


def estimate_max_batch_size(
    model_config: transformers.PretrainedConfig,
    max_sequence_length: int,
    gpu_memory_size: float,
    gpu_memory_utilization: float = 0.9,
    dtype_bytes: int = 2,
) -> int:
    """
    Estimate maximum batch size that can fit in GPU memory.

    Args:
        model_config: Model configuration
        max_sequence_length: Maximum sequence length
        gpu_memory_size: Total GPU memory in bytes
        gpu_memory_utilization: Fraction of GPU memory to use (0-1)
        dtype_bytes: Bytes per element

    Returns:
        int: Estimated maximum batch size
    """
    # Calculate model size
    model_size = estimate_model_size_bytes(model_config, dtype_bytes)

    # Calculate KV cache per token
    kv_per_token = estimate_kv_cache_memory_per_token(model_config, dtype_bytes)

    # Available memory after loading model
    available_memory = gpu_memory_size * gpu_memory_utilization - model_size

    # Memory per sequence (KV cache for full sequence + activation memory)
    # Activation memory is roughly 20% of KV cache (empirical estimate)
    memory_per_sequence = kv_per_token * max_sequence_length * 1.2

    # Maximum batch size
    max_batch_size = int(available_memory / memory_per_sequence)

    # Ensure at least 1
    return max(1, max_batch_size)


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
) -> Tuple[transformers.PreTrainedTokenizer, transformers.PretrainedConfig, float, float, float, float]:
    """Set up the tokenizer and model config."""
    logger.info(f"Loading tokenizer and config: {model_config.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model config for FLOPs calculation
    hf_model_config = transformers.AutoConfig.from_pretrained(model_config.model_name_or_path)
    model_flops_per_token = calculate_model_flops_per_token(hf_model_config, model_config.model_name_or_path)
    gpu_peak_flops = get_gpu_peak_flops()
    gpu_memory_bandwidth = get_gpu_memory_bandwidth()
    gpu_memory_size = get_gpu_memory_size()

    logger.info(f"Model FLOPs per token: {model_flops_per_token / 1e9:.2f} GFLOPs")
    logger.info(f"GPU peak FLOPs: {gpu_peak_flops / 1e12:.0f} TFLOPs")
    logger.info(f"GPU memory bandwidth: {gpu_memory_bandwidth / 1e12:.2f} TB/s")
    logger.info(f"GPU memory size: {gpu_memory_size / 1e9:.0f} GB")

    return tokenizer, hf_model_config, model_flops_per_token, gpu_peak_flops, gpu_memory_bandwidth, gpu_memory_size


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
    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)

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
    model_flops_per_token: float,
    gpu_peak_flops: float,
    gpu_memory_bandwidth: float,
    hf_model_config: transformers.PretrainedConfig,
    prompts: List[List[int]],
    batch_idx: int,
) -> Dict[str, Union[float, int, List[str], List[int]]]:
    """Run generation for a batch of prompts and measure performance."""

    # Measure timing from prompt submission to result retrieval
    start_time = time.time()

    # Send prompts
    param_prompt_Q.put((None, prompts))

    # Get results
    result = inference_results_Q.get(timeout=24000)

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
    mfu_percentage = calculate_mfu(tokens_per_second, model_flops_per_token, gpu_peak_flops)

    # Calculate memory bandwidth utilization
    kv_cache_per_token = estimate_kv_cache_memory_per_token(hf_model_config)
    model_size_bytes = estimate_model_size_bytes(hf_model_config)

    # Fix: Use actual batch size (unique prompts) not total responses
    actual_batch_size = len(prompts)  # Number of unique prompts

    # Fix: Calculate average sequence length more accurately
    # Each prompt generates multiple responses, but they all share the same prompt length
    avg_prompt_length = sum(len(prompt) for prompt in prompts) / len(prompts)
    avg_response_length = total_new_tokens / len(response_ids) if response_ids else 0
    avg_sequence_length = avg_prompt_length + avg_response_length

    # Log intermediate values for debugging
    logger.info(f"MBU Calculation Details:")
    logger.info(f"  - Tokens per second: {tokens_per_second:.2f}")
    logger.info(f"  - KV cache per token: {kv_cache_per_token / 1024:.2f} KB")
    logger.info(f"  - Model size: {model_size_bytes / 1e9:.2f} GB")
    logger.info(f"  - Actual batch size: {actual_batch_size}")
    logger.info(
        f"  - Avg sequence length: {avg_sequence_length:.0f} (prompt: {avg_prompt_length:.0f}, response: {avg_response_length:.0f})"
    )

    memory_bandwidth_usage = calculate_memory_bandwidth_usage(
        tokens_per_second, kv_cache_per_token, model_size_bytes, avg_sequence_length, actual_batch_size
    )
    mbu_percentage = calculate_mbu(memory_bandwidth_usage, gpu_memory_bandwidth)

    logger.info(f"  - Memory bandwidth usage: {memory_bandwidth_usage / 1e9:.2f} GB/s")
    logger.info(f"  - GPU memory bandwidth: {gpu_memory_bandwidth / 1e12:.2f} TB/s")
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
        "kv_cache_per_token_bytes": kv_cache_per_token,
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
    hf_model_config: transformers.PretrainedConfig,
    model_flops_per_token: float,
    gpu_peak_flops: float,
    gpu_memory_bandwidth: float,
    gpu_memory_size: float,
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
            inference_results_Q,
            param_prompt_Q,
            args,
            model_flops_per_token,
            gpu_peak_flops,
            gpu_memory_bandwidth,
            hf_model_config,
            prompts,
            batch_idx,
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
        print_summary(
            all_results,
            total_time,
            args,
            hf_model_config,
            model_flops_per_token,
            gpu_peak_flops,
            gpu_memory_bandwidth,
            gpu_memory_size,
        )
    else:
        logger.error("No successful batches completed")

    return all_results


def print_summary(
    results: List[Dict[str, Union[float, int, List[str], List[int]]]],
    total_time: float,
    args: Args,
    hf_model_config: transformers.PretrainedConfig,
    model_flops_per_token: float,
    gpu_peak_flops: float,
    gpu_memory_bandwidth: float,
    gpu_memory_size: float,
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
    print(f"GPU peak FLOPs: {gpu_peak_flops / 1e12:.0f} TFLOPs")
    print(f"GPU memory bandwidth: {gpu_memory_bandwidth / 1e12:.2f} TB/s")
    print(f"GPU memory size: {gpu_memory_size / 1e9:.0f} GB")
    print("-" * 60)
    print("MODEL SPECIFICATIONS:")
    print(f"Model FLOPs per token: {model_flops_per_token / 1e9:.2f} GFLOPs")
    print(f"KV cache per token: {estimate_kv_cache_memory_per_token(hf_model_config) / 1024:.2f} KB")
    print(f"Estimated model size: {estimate_model_size_bytes(hf_model_config) / 1e9:.2f} GB")

    # Calculate and display maximum batch size estimation
    max_batch_size_est = estimate_max_batch_size(
        hf_model_config, args.response_length, gpu_memory_size, args.vllm_gpu_memory_utilization
    )
    print(f"Estimated max batch size (@ {args.response_length} tokens): {max_batch_size_est}")

    # Bottleneck analysis
    print("-" * 60)
    print("BOTTLENECK ANALYSIS:")
    bottleneck_mfu = avg_mfu_last_n_minus_1 if len(results) > 1 else avg_mfu
    bottleneck_mbu = avg_mbu_last_n_minus_1 if len(results) > 1 else avg_mbu

    if bottleneck_mfu > bottleneck_mbu:
        print(f"System is COMPUTE BOUND (MFU: {bottleneck_mfu:.1f}% > MBU: {bottleneck_mbu:.1f}%)")
        print("Suggestions:")
        print("  - Consider using a GPU with higher FLOPs")
        print("  - Optimize model architecture or use smaller model")
        print("  - Enable tensor parallelism if not already used")
    else:
        print(f"System is MEMORY BANDWIDTH BOUND (MBU: {bottleneck_mbu:.1f}% > MFU: {bottleneck_mfu:.1f}%)")
        print("Suggestions:")
        print("  - Consider using a GPU with higher memory bandwidth")
        print("  - Reduce batch size to fit more in cache")
        print("  - Use Flash Attention or other memory-efficient attention mechanisms")
        print("  - Consider quantization to reduce memory bandwidth requirements")

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


def run_benchmark_programmatic(
    model_name_or_path: str,
    num_unique_prompts_rollout: int,
    num_samples_per_prompt_rollout: int,
    vllm_num_engines: int,
    response_length: int,
    **kwargs,
) -> Dict[str, float]:
    """
    Run benchmark programmatically and return summary statistics.

    This is a wrapper function for use by other scripts like benchmark_loop.py.

    Args:
        model_name_or_path: Model to benchmark
        num_unique_prompts_rollout: Number of unique prompts per batch
        num_samples_per_prompt_rollout: Number of samples per prompt
        vllm_num_engines: Number of vLLM engines
        response_length: Maximum response length
        **kwargs: Additional arguments to override defaults

    Returns:
        Dictionary with mfu, tokens_per_second, wall_clock_time, mbu
    """
    # Set up default arguments
    default_args = {
        "tokenizer_name_or_path": model_name_or_path,
        "dataset_mixer_list": [
            "hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2",
            "1.0",
        ],
        "dataset_mixer_list_splits": ["train"],
        "max_token_length": 10240,
        "max_prompt_token_length": 2048,
        "temperature": 1.0,
        "vllm_top_p": 0.9,
        "vllm_tensor_parallel_size": 1,
        "vllm_gpu_memory_utilization": 0.9,
        "pack_length": 20480,
        "chat_template_name": "tulu_thinker",
        "trust_remote_code": True,
        "seed": 42,
        "dataset_local_cache_dir": "benchmark_cache",
        "dataset_cache_mode": "local",
        "dataset_transform_fn": ["rlvr_tokenize_v1", "rlvr_filter_v1"],
    }

    # Update with provided kwargs
    default_args.update(kwargs)

    # Create dataclass instances
    args = Args(
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=default_args["tokenizer_name_or_path"],
        dataset_mixer_list=default_args["dataset_mixer_list"],
        dataset_mixer_list_splits=default_args["dataset_mixer_list_splits"],
        max_token_length=default_args["max_token_length"],
        max_prompt_token_length=default_args["max_prompt_token_length"],
        temperature=default_args["temperature"],
        response_length=response_length,
        vllm_top_p=default_args["vllm_top_p"],
        num_unique_prompts_rollout=num_unique_prompts_rollout,
        num_samples_per_prompt_rollout=num_samples_per_prompt_rollout,
        vllm_num_engines=vllm_num_engines,
        vllm_tensor_parallel_size=default_args["vllm_tensor_parallel_size"],
        vllm_gpu_memory_utilization=default_args["vllm_gpu_memory_utilization"],
        pack_length=default_args["pack_length"],
        chat_template_name=default_args["chat_template_name"],
        trust_remote_code=default_args["trust_remote_code"],
        seed=default_args["seed"],
        dataset_local_cache_dir=default_args["dataset_local_cache_dir"],
        dataset_cache_mode=default_args["dataset_cache_mode"],
        dataset_skip_cache=False,
        dataset_transform_fn=default_args["dataset_transform_fn"],
    )

    tokenizer_config = TokenizerConfig()
    model_config = ModelConfig(
        model_name_or_path=model_name_or_path, model_revision=None, trust_remote_code=default_args["trust_remote_code"]
    )

    # Run the benchmark
    tokenizer, hf_model_config, model_flops_per_token, gpu_peak_flops, gpu_memory_bandwidth, gpu_memory_size = (
        setup_tokenizer(model_config)
    )
    dataset = setup_dataset(args, tokenizer_config)
    vllm_engines = setup_vllm_engines(args, model_config)

    try:
        results = run_benchmark(
            dataset,
            vllm_engines,
            args,
            hf_model_config,
            model_flops_per_token,
            gpu_peak_flops,
            gpu_memory_bandwidth,
            gpu_memory_size,
        )

        # Compute summary statistics
        summary_stats = compute_summary_stats(results)

        return summary_stats
    finally:
        cleanup(vllm_engines)


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

    tokenizer, hf_model_config, model_flops_per_token, gpu_peak_flops, gpu_memory_bandwidth, gpu_memory_size = (
        setup_tokenizer(model_config)
    )
    dataset = setup_dataset(args, tokenizer_config)
    vllm_engines = setup_vllm_engines(args, model_config)

    # Run benchmark and get results
    results = run_benchmark(
        dataset,
        vllm_engines,
        args,
        hf_model_config,
        model_flops_per_token,
        gpu_peak_flops,
        gpu_memory_bandwidth,
        gpu_memory_size,
    )

    # Save completion lengths
    if results:
        save_completion_lengths(results, timestamp)

    cleanup(vllm_engines)


if __name__ == "__main__":
    main()
