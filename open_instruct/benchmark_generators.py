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
import pathlib
import threading
import time
from concurrent import futures
from typing import Any, Optional

import datasets
import numpy as np
import ray
import torch
import torch.utils.flop_counter
import transformers
import vllm
from ray.util import queue as ray_queue

from open_instruct import dataset_transformation, grpo_fast, logger_utils, model_utils, utils, vllm_utils3
from open_instruct.queue_types import PromptRequest

# For FLOPS, we assume bf16 and ignore sparsity.
GPU_SPECS = {
    "a100": {"flops": 312e12, "memory_size": 80e9},
    "b200": {"flops": 2250e12, "memory_size": 192e9},
    "h100": {"flops": 990e12, "memory_size": 80e9},
    "a6000": {"flops": 155e12, "memory_size": 48e9},
    "l40s": {"flops": 362e12, "memory_size": 48e9},
}


logger = logger_utils.setup_logger(__name__)


# Determine data directory
if pathlib.Path("/weka").exists():
    DATA_DIR = pathlib.Path("/weka") / "finbarrt" / "open_instruct_generators_benchmark"
elif pathlib.Path("/root").exists():
    DATA_DIR = pathlib.Path("/root") / "finbarrt" / "open_instruct_generators_benchmark"
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


def get_git_commit() -> str:
    """Get the current git commit hash."""
    git_dir = pathlib.Path(".git")
    if not git_dir.exists():
        return "unknown"

    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return "unknown"

    with head_file.open() as f:
        content = f.read().strip()

    if not content.startswith("ref:"):
        # Detached HEAD
        return content[:8]

    # HEAD points to a branch
    ref_path = git_dir / content[5:]

    with ref_path.open() as ref_f:
        return ref_f.read().strip()[:8]  # First 8 chars


def save_benchmark_results_to_csv(
    results: list[dict[str, Any]], total_time: float, args: grpo_fast.Args, model_config: model_utils.ModelConfig
) -> None:
    """Save benchmark results to CSV file."""
    git_commit = get_git_commit()
    avg_results = average_results(results)
    row_data = {
        "git_commit": git_commit,
        "model": model_config.model_name_or_path,
        "total_batches": len(results),
        "batch_size": args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout,
        "num_unique_prompts_rollout": args.num_unique_prompts_rollout,
        "num_samples_per_prompt_rollout": args.num_samples_per_prompt_rollout,
        "response_length": args.response_length,
        "total_time": total_time,
        "total_generation_time": avg_results["total_generation_time"],
        "generation_time_percentage": (avg_results["total_generation_time"] / total_time) * 100,
        "total_tokens": avg_results["total_num_new_tokens"],
        "avg_tokens_per_second": true_avg_tokens_per_second,
        "avg_mfu": avg_results["avg_mfu"],
        "avg_generation_time_per_batch": avg_results["avg_generation_time"],
        "avg_new_tokens_per_sample": avg_results["total_num_new_tokens"]
        / (len(results) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout),
    }

    csv_path: pathlib.Path = DATA_DIR / "generator_benchmark_results.csv"
    csv_dir = csv_path.parent
    csv_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row_data.keys()))
        if not csv_path.exists():
            writer.writeheader()
        writer.writerow(row_data)
    logger.info(f"Saved benchmark results to {csv_path}")


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
    tokens = device_name.lower().replace("-", " ").split()

    filtered = [val for val in tokens if val not in ["nvidia", "80gb", "40gb", "48gb", "hbm3", "rtx", "sxm4", "pcie"]]

    for token in filtered:
        if token in GPU_SPECS:
            return token

    raise ValueError(f"Unsupported device name: {device_name}. Expected one of: {list(GPU_SPECS.keys())}")


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
) -> tuple[list[ray.actor.ActorHandle], ray_queue.Queue, ray_queue.Queue]:
    """Set up vLLM engines and queues."""
    logger.info("Setting up vLLM engines...")

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True, runtime_env={"excludes": ["/benchmark_cache/"]})

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.vllm_num_engines)]
    pg = ray.util.placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    param_prompt_Q = ray_queue.Queue(maxsize=10)
    inference_results_Q = ray_queue.Queue(maxsize=10)

    actor_manager = vllm_utils3.ActorManager.remote()

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
        prompt_queue=param_prompt_Q,
        results_queue=inference_results_Q,
        actor_manager=actor_manager,
    )

    logger.info("vLLM engines ready")

    return vllm_engines, param_prompt_Q, inference_results_Q, actor_manager


def get_batch_data(
    dataset: datasets.Dataset, batch_size: int, batch_idx: int
) -> tuple[list[list[int]], list[str], list[str]]:
    """Get a batch of data from the dataset."""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))

    batch_data = dataset[start_idx:end_idx]
    prompts = batch_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    logger.info(f"get_batch_data: batch_idx={batch_idx}, returning {len(prompts)} unique prompts")
    return prompts


def generate_thread(vllm_engines: list[ray.actor.ActorHandle], stop_event: threading.Event) -> None:
    """Thread that repeatedly calls process_from_queue on vllm engines."""
    logger.info("[Generate Thread] Starting generation thread")
    while not stop_event.is_set():
        processed_results = ray.get([engine.process_from_queue.remote(timeout=20) for engine in vllm_engines])
        num_processed = sum(int(result) for result in processed_results)
        if num_processed == 0:
            time.sleep(1)
        else:
            logger.debug(f"[Generate Thread] Processed {num_processed} requests")


def submission_thread(
    param_prompt_Q: ray_queue.Queue,
    all_prompts: list,
    generation_config: vllm.SamplingParams,
    stop_event: threading.Event,
) -> None:
    """Thread that submits prompts to the queue."""
    logger.info("[Submission Thread] Starting prompt submission")
    for batch_idx, prompts in enumerate(all_prompts):
        if stop_event.is_set():
            logger.info("[Submission Thread] Stopped due to stop event")
            break
        param_prompt_Q.put(
            PromptRequest(prompts=prompts, dataset_index=batch_idx, generation_config=generation_config)
        )
    logger.info(f"[Submission Thread] All {len(all_prompts)} prompts submitted")


def run_benchmark(
    dataset: datasets.Dataset,
    vllm_engines: list[ray.actor.ActorHandle],
    param_prompt_Q: ray_queue.Queue,
    inference_results_Q: ray_queue.Queue,
    args: grpo_fast.Args,
    model_config: model_utils.ModelConfig,
    timestamp: int,
    flops_per_token: int,
    num_batches: int = 5,
) -> list[dict[str, Any]]:
    """Run the full benchmark."""
    logger.info(
        f"Starting benchmark with 1 warmup batch + {num_batches - 1} main batches of size {args.num_unique_prompts_rollout}"
    )

    # Create sampling parameters with 'n' for multiple samples per prompt
    generation_config = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.response_length,
        top_p=args.vllm_top_p,
        n=args.num_samples_per_prompt_rollout,
        # IMPORTANT: Set output_kind to FINAL_ONLY to ensure vLLM V1 properly handles n>1
        # With the default CUMULATIVE mode, vLLM V1 returns separate outputs for each
        # completion, making it difficult to aggregate them correctly. FINAL_ONLY mode
        # ensures all n completions are returned together in a single output.
        output_kind=vllm.sampling_params.RequestOutputKind.FINAL_ONLY,
    )

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="benchmark")

    generation_future = executor.submit(generate_thread, vllm_engines, stop_event)

    results = []
    device_name = get_device_name(torch.cuda.get_device_name(0))
    device_flops = GPU_SPECS[device_name]["flops"]

    logger.info(f"Preparing {num_batches} batches...")
    all_prompts = [
        get_batch_data(dataset, args.num_unique_prompts_rollout, batch_idx) for batch_idx in range(num_batches)
    ]

    # Submit warmup batch first
    logger.info("Submitting warmup batch...")
    warmup_prompts = all_prompts[0]
    param_prompt_Q.put(PromptRequest(prompts=warmup_prompts, dataset_index=0, generation_config=generation_config))

    try:
        logger.info("Running warmup batch...")

        warmup_result = inference_results_Q.get()
        logger.info(f"Warmup batch completed with {len(warmup_result.responses)} responses")
        logger.info(f"Expected {args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout} responses")
        logger.info(f"Submitting {num_batches - 1} batches for main benchmark...")
        submission_future = executor.submit(
            submission_thread, param_prompt_Q, all_prompts[1:], generation_config, stop_event
        )
        last_completion_time = time.time()

        # Process remaining batches with timing
        for batch_idx in range(1, num_batches):
            # Quick health check!
            [future.result() for future in [submission_future, generation_future] if future.done()]
            result = inference_results_Q.get()

            completion_time = time.time()
            batch_generation_time = completion_time - last_completion_time
            last_completion_time = completion_time

            # Debug logging to understand response count
            num_responses = len(result.responses)
            expected_responses = args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
            logger.info(
                f"Batch {batch_idx}: Got {num_responses} responses, expected {expected_responses} "
                f"({args.num_unique_prompts_rollout} prompts × {args.num_samples_per_prompt_rollout} rollouts)"
            )

            new_tokens = sum(len(response) for response in result.responses)
            tokens_per_second = new_tokens / batch_generation_time if batch_generation_time > 0 else 0

            result_dict = {
                "tokens_per_second": tokens_per_second,
                "generation_time": batch_generation_time,
                "num_new_tokens": new_tokens,
                "finish_reasons": collections.Counter(result.finish_reasons),
                "response_lengths": [len(response) for response in result.responses],
                "batch_idx": result.dataset_index,
            }
            result_dict["mfu"] = 100 * result_dict["tokens_per_second"] * flops_per_token / device_flops

            save_completion_lengths([result_dict], timestamp, result.dataset_index)
            results.append(result_dict)
            logger.info(
                f"Batch {batch_idx}/{num_batches - 1}: "
                f"{result_dict['tokens_per_second']:.2f} new tokens/sec, "
                f"MFU: {result_dict['mfu']:.2f}%, "
                f"generation time: {batch_generation_time:.2f}s, "
                f"total new tokens: {new_tokens}"
            )

        # Calculate total time for main benchmark only
        main_benchmark_time = sum(r["generation_time"] for r in results)

        print_summary(results, main_benchmark_time, args, model_config)
        save_benchmark_results_to_csv(results, main_benchmark_time, args, model_config)

    finally:
        stop_event.set()
        executor.shutdown(wait=True)
        logger.info("Threads cleaned up")


def average_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate total and aggregated metrics from results."""
    aggregated_results = {
        "total_mfu": 0.0,
        "total_tokens_per_second": 0.0,
        "total_generation_time": 0.0,
        "total_num_new_tokens": 0,
        "finish_reasons": collections.defaultdict(int),
        "response_lengths": [],
        "prompt_lengths": [],
    }
    for result in results:
        for key, value in result.items():
            if key == "mfu":
                aggregated_results["total_mfu"] += value
            elif key == "tokens_per_second":
                aggregated_results["total_tokens_per_second"] += value
            elif key == "generation_time":
                aggregated_results["total_generation_time"] += value
            elif key == "num_new_tokens":
                aggregated_results["total_num_new_tokens"] += value
            elif key == "finish_reasons":
                for reason, count in value.items():
                    aggregated_results["finish_reasons"][reason] += count
            elif key == "response_lengths":
                aggregated_results["response_lengths"].extend(value)
            elif key == "prompt_lengths":
                aggregated_results["prompt_lengths"].extend(value)

    # Calculate true averages where needed
    num_results = len(results)
    aggregated_results["avg_mfu"] = aggregated_results["total_mfu"] / num_results if num_results > 0 else 0
    aggregated_results["avg_generation_time"] = (
        aggregated_results["total_generation_time"] / num_results if num_results > 0 else 0
    )

    return aggregated_results


def print_summary(
    results: list[dict[str, Any]], total_time: float, args: grpo_fast.Args, model_config: model_utils.ModelConfig
) -> None:
    """Print benchmark summary statistics."""

    # Calculate metrics only for the main benchmark batches (excluding warmup)
    avg_results = average_results(results)
    total_tokens = avg_results["total_num_new_tokens"]
    total_generation_time = avg_results["total_generation_time"]

    # Calculate true average tokens per second
    true_avg_tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0

    # Calculate average new tokens per sample
    total_samples = len(results) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    avg_new_tokens_per_sample = total_tokens / total_samples if total_samples > 0 else 0

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model: {model_config.model_name_or_path}")
    print(f"Main benchmark batches: {len(results)} (after 1 warmup batch)")
    print(f"Batch size: {args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout}")
    print(f"Unique prompts per batch: {args.num_unique_prompts_rollout}")
    print(f"Num rollouts: {args.num_samples_per_prompt_rollout}")
    print(f"Max tokens: {args.response_length}")
    print("-" * 60)
    print(f"Total time (main benchmark): {total_generation_time:.2f}s")
    print(f"Total new tokens generated: {total_tokens}")
    print("-" * 60)
    print(f"Average results over {len(results)} main benchmark batches:")
    print(f"Average tokens/second: {true_avg_tokens_per_second:.2f}")
    print(f"Average MFU: {avg_results['avg_mfu']:.2f}%")
    print(f"Average generation time per batch: {avg_results['avg_generation_time']:.2f}s")
    print(f"Average new tokens per sample: {avg_new_tokens_per_sample:.2f} tokens")

    max_length = np.max(avg_results["response_lengths"])
    mean_length = np.mean(avg_results["response_lengths"])
    wasted_compute = (max_length - mean_length) / max_length
    print(f"Wasted compute % (variable response length): {wasted_compute:.2%}")

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


def cleanup(vllm_engines: list[ray.actor.ActorHandle], actor_manager: Optional[ray.actor.ActorHandle] = None) -> None:
    """Clean up resources."""
    if actor_manager:
        try:
            ray.get(actor_manager.set_should_stop.remote(True))
            logger.info("Signaled all engines to stop via actor manager")
        except Exception as e:
            logger.warning(f"Error signaling actor manager: {e}")

    for engine in vllm_engines:
        try:
            ray.kill(engine)
        except Exception as e:
            logger.warning(f"Error killing engine: {e}")

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

    # Calculate flops per token before starting vLLM
    logger.info("Calculating model FLOPs per token...")
    flops_per_token = calculate_model_usage_per_token(model_config.model_name_or_path)
    logger.info(f"Model FLOPs per token: {flops_per_token:,}")

    # Free GPU memory after calculating FLOPs and before starting vLLM
    logger.info("Freeing GPU memory before starting vLLM...")
    free_all_gpu_memory()

    dataset = setup_dataset(args, tokenizer_config)
    vllm_engines, param_prompt_Q, inference_results_Q, actor_manager = setup_vllm_engines(args, model_config)

    # Create the timestamp here so we use it for both filenames.
    timestamp = int(time.time())
    save_config(args, tokenizer_config, model_config, timestamp)
    run_benchmark(
        dataset, vllm_engines, param_prompt_Q, inference_results_Q, args, model_config, timestamp, flops_per_token
    )

    cleanup(vllm_engines, actor_manager)


if __name__ == "__main__":
    main()
