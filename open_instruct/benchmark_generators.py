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
import os
import pathlib
import threading
import time
from concurrent import futures
from typing import Any, cast

import datasets
import numpy as np
import ray
import torch
import torch.utils.flop_counter
import vllm
from ray.util import queue as ray_queue

from open_instruct import dataset_transformation, grpo_fast, logger_utils, model_utils, utils, vllm_utils
from open_instruct.actor_manager import ActorManager
from open_instruct.data_types import PromptRequest

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
    agg_results = aggregate_results(results)
    csv_path: pathlib.Path = DATA_DIR / "generator_benchmark_results.csv"

    row_data = {
        "git_commit": git_commit,
        "model": model_config.model_name_or_path,
        "total_batches": len(results),
        "batch_size": args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout,
        "num_unique_prompts_rollout": args.num_unique_prompts_rollout,
        "num_samples_per_prompt_rollout": args.num_samples_per_prompt_rollout,
        "response_length": args.response_length,
        "total_time": total_time,
        "total_generation_time": agg_results["total_generation_time"],
        "total_weight_sync_time": agg_results["total_weight_sync_time"],
        "generation_time_percentage": (agg_results["total_generation_time"] / total_time) * 100,
        "weight_sync_time_percentage": (agg_results["total_weight_sync_time"] / total_time) * 100
        if total_time > 0
        else 0,
        "total_tokens": agg_results["total_num_new_tokens"],
        "avg_tokens_per_second": agg_results["avg_tokens_per_second"],
        "avg_mfu": agg_results["avg_mfu"],
        "avg_mbu": agg_results["avg_mbu"],
        "avg_generation_time_per_batch": agg_results["avg_generation_time"],
        "avg_weight_sync_time_per_batch": agg_results["avg_weight_sync_time"],
        "avg_new_tokens_per_sample": agg_results["total_num_new_tokens"]
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


def setup_dataset(args: grpo_fast.Args, tokenizer_config: dataset_transformation.TokenizerConfig) -> datasets.Dataset:
    """Set up the dataset using the same pipeline as grpo_fast.py."""
    logger.info("Loading and processing dataset...")

    # Transform function arguments
    transform_fn_args = [
        {},  # For rlvr_tokenize_v1
        {"max_prompt_token_length": args.max_prompt_token_length},  # For rlvr_filter_v1
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
    args: grpo_fast.Args,
    tokenizer_config: dataset_transformation.TokenizerConfig,
    model_config: model_utils.ModelConfig,
    max_model_len: int,
) -> tuple[list[ray.actor.ActorHandle], ray_queue.Queue, ray_queue.Queue, ray.actor.ActorHandle]:
    """Set up vLLM engines and queues."""
    ray.init(ignore_reinit_error=True, runtime_env={"excludes": ["/benchmark_cache/"], "env_vars": dict(os.environ)})

    param_prompt_Q = ray_queue.Queue(maxsize=10)
    inference_results_Q = ray_queue.Queue(maxsize=10)

    queues_to_monitor = {"Param Prompt Queue": param_prompt_Q, "Inference Results Queue": inference_results_Q}
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args)

    tokenizer_name_or_path = tokenizer_config.tokenizer_name_or_path or model_config.model_name_or_path
    assert tokenizer_name_or_path is not None
    assert model_config.model_name_or_path is not None

    vllm_engines = vllm_utils.create_vllm_engines(
        num_engines=args.vllm_num_engines,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        enforce_eager=args.vllm_enforce_eager,
        tokenizer_name_or_path=tokenizer_name_or_path,
        pretrain=model_config.model_name_or_path,
        revision=model_config.model_revision,
        seed=args.seed,
        enable_prefix_caching=args.vllm_enable_prefix_caching,
        max_model_len=max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        single_gpu_mode=args.single_gpu_mode,
        pg=None,
        tools={},
        max_tool_calls=args.max_tool_calls,
        prompt_queue=param_prompt_Q,
        results_queue=inference_results_Q,
        actor_manager=actor_manager,
        inflight_updates=args.inflight_updates,
    )

    logger.info("vLLM engines ready")

    return vllm_engines, param_prompt_Q, inference_results_Q, actor_manager


def simulate_weight_sync(
    actor_manager: ray.actor.ActorHandle, vllm_engines: list[ray.actor.ActorHandle], args: grpo_fast.Args
) -> float:
    """Simulate weight sync by pausing all actors.

    Returns:
        float: Time taken for the weight sync simulation in seconds
    """
    sync_start = time.perf_counter()

    # Set actors to stop
    ray.get(actor_manager.set_should_stop.remote(True))
    logger.debug("Set should_stop to True for weight sync simulation")

    utils.ray_get_with_progress(
        [engine.check_background_threads.remote() for engine in vllm_engines], "Health check on background threads."
    )

    # Sleep for 1 second to simulate weight sync time (from wandb metrics)
    time.sleep(1.0)

    # Resume actors
    ray.get(actor_manager.set_should_stop.remote(False))
    logger.debug("Set should_stop to False after weight sync simulation")

    sync_time = time.perf_counter() - sync_start
    logger.info(f"Weight sync simulation took {sync_time:.2f}s")

    return sync_time


def submission_thread(
    param_prompt_Q: ray_queue.Queue,
    dataset: datasets.Dataset,
    generation_config: vllm.SamplingParams,
    stop_event: threading.Event,
    batch_size: int,
    start_batch_idx: int,
    num_batches: int,
) -> None:
    """Thread that submits prompts to the queue."""
    logger.info("[Submission Thread] Starting prompt submission")
    for batch_idx in range(start_batch_idx, start_batch_idx + num_batches):
        if stop_event.is_set():
            logger.info("[Submission Thread] Stopped due to stop event")
            break

        # Get batch data from dataset
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_data = dataset[start_idx:end_idx]
        prompts = batch_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]

        # Create individual PromptRequest for each prompt in the batch
        for i, prompt in enumerate(prompts):
            dataset_index = start_idx + i
            param_prompt_Q.put(
                PromptRequest(
                    prompt=prompt,
                    dataset_index=dataset_index,
                    prompt_id=f"batch_{batch_idx}_prompt_{i}",
                    generation_config=generation_config,
                )
            )
    logger.info(f"[Submission Thread] All {num_batches} batches submitted")


def run_benchmark(
    dataset: datasets.Dataset,
    vllm_engines: list[ray.actor.ActorHandle],
    param_prompt_Q: ray_queue.Queue,
    inference_results_Q: ray_queue.Queue,
    actor_manager: ray.actor.ActorHandle,
    args: grpo_fast.Args,
    model_config: model_utils.ModelConfig,
    timestamp: int,
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
        min_tokens=args.response_length,
        top_p=args.vllm_top_p,
        n=args.num_samples_per_prompt_rollout,
        seed=args.seed,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        logprobs=1,
        # IMPORTANT: Set output_kind to FINAL_ONLY to ensure vLLM V1 properly handles n>1
        # With the default CUMULATIVE mode, vLLM V1 returns separate outputs for each
        # completion, making it difficult to aggregate them correctly. FINAL_ONLY mode
        # ensures all n completions are returned together in a single output.
        output_kind=vllm.sampling_params.RequestOutputKind.FINAL_ONLY,
    )

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark")

    results = []
    # Get the model dimensions from one of the engines without loading weights
    model_dims = ray.get(vllm_engines[0].get_model_dims.remote())

    # Submit warmup batch first
    logger.info("Submitting warmup batch...")
    warmup_start_idx = 0
    warmup_end_idx = min(args.num_unique_prompts_rollout, len(dataset))
    warmup_data = dataset[warmup_start_idx:warmup_end_idx]
    warmup_prompts = warmup_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    # Create individual PromptRequest for each warmup prompt
    for i, prompt in enumerate(warmup_prompts):
        dataset_index = warmup_start_idx + i
        param_prompt_Q.put(
            PromptRequest(
                prompt=prompt,
                dataset_index=dataset_index,
                prompt_id=f"warmup_prompt_{i}",
                generation_config=generation_config,
            )
        )

    utils.ray_get_with_progress([engine.ready.remote() for engine in vllm_engines], "Checking if engines are ready.")
    try:
        logger.info("Running warmup batch...")

        # Collect all warmup results (one per prompt)
        warmup_batch_size = warmup_end_idx - warmup_start_idx
        warmup_results = [inference_results_Q.get() for _ in range(warmup_batch_size)]

        total_warmup_responses = sum(len(result.responses) for result in warmup_results)
        logger.info(
            f"Warmup batch completed with {total_warmup_responses} total responses from {len(warmup_results)} prompts"
        )
        logger.info(f"Submitting {num_batches - 1} batches for main benchmark...")
        submission_future = executor.submit(
            submission_thread,
            param_prompt_Q,
            dataset,
            generation_config,
            stop_event,
            args.num_unique_prompts_rollout,
            1,
            num_batches - 1,
        )
        # Process remaining batches with timing
        for batch_idx in range(1, num_batches):
            # Quick health check!
            if submission_future.done():
                submission_future.result()

            # Collect all results for this batch (one per prompt)
            batch_results = [inference_results_Q.get() for _ in range(args.num_unique_prompts_rollout)]

            # Simulate weight sync between batches
            weight_sync_time = simulate_weight_sync(actor_manager, vllm_engines, args)
            completion_time = time.perf_counter()

            # Calculate generation time from the earliest request start time (inclusive of sync)
            earliest_start_time = min(result.start_time for result in batch_results if result.start_time)
            batch_generation_time = completion_time - earliest_start_time if earliest_start_time else 0

            # Aggregate metrics across all results in the batch
            total_new_tokens = sum(len(response) for result in batch_results for response in result.responses)
            tokens_per_second = total_new_tokens / batch_generation_time if batch_generation_time > 0 else 0

            # Collect all finish reasons and response lengths
            all_finish_reasons = []
            all_response_lengths = []
            all_dataset_indices = []
            all_prompt_lengths = []

            for result in batch_results:
                all_finish_reasons.extend(result.finish_reasons)
                all_response_lengths.extend([len(response) for response in result.responses])
                all_dataset_indices.append(result.dataset_index)

                # Get prompt length for this result
                prompt_data = dataset[result.dataset_index]
                prompt = prompt_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
                all_prompt_lengths.append(len(prompt))

            result_dict = {
                "tokens_per_second": tokens_per_second,
                "generation_time": batch_generation_time,
                "weight_sync_time": weight_sync_time,
                "num_new_tokens": total_new_tokens,
                "finish_reasons": collections.Counter(all_finish_reasons),
                "response_lengths": all_response_lengths,
                "prompt_lengths": all_prompt_lengths,
                "dataset_indices": all_dataset_indices,
            }

            num_engines = args.vllm_num_engines
            num_gpus_per_engine = args.vllm_tensor_parallel_size
            num_inference_gpus = num_engines * num_gpus_per_engine

            result_dict["mfu"] = model_dims.calculate_mfu(
                all_prompt_lengths,
                batch_generation_time,
                response_lengths=all_response_lengths,
                samples_per_prompt=args.num_samples_per_prompt_rollout,
                num_gpus=num_inference_gpus,
            )

            result_dict["mbu"] = model_dims.calculate_mbu(
                all_prompt_lengths,
                batch_generation_time,
                response_lengths=all_response_lengths,
                samples_per_prompt=args.num_samples_per_prompt_rollout,
                num_engines=num_engines,
                num_gpus_per_engine=num_gpus_per_engine,
            )

            save_completion_lengths([result_dict], timestamp, batch_idx)
            results.append(result_dict)
            logger.info(
                f"Batch {batch_idx}/{num_batches - 1}: "
                f"{result_dict['tokens_per_second']:.2f} new tokens/sec, "
                f"MFU: {result_dict['mfu']:.2f}%, "
                f"MBU: {result_dict['mbu']:.2f}%, "
                f"generation time: {batch_generation_time:.2f}s, "
                f"weight sync time: {weight_sync_time:.2f}s, "
                f"total new tokens: {total_new_tokens}"
            )

        # Calculate total time for main benchmark only
        main_benchmark_time = sum(r["generation_time"] for r in results)

        print_summary(results, main_benchmark_time, args, model_config, model_dims)
        save_benchmark_results_to_csv(results, main_benchmark_time, args, model_config)

    finally:
        stop_event.set()
        executor.shutdown(wait=True)
        logger.info("Threads cleaned up")

    return results


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate total and aggregated metrics from results."""
    total_mfu = 0.0
    total_mbu = 0.0
    total_tokens_per_second = 0.0
    total_generation_time = 0.0
    total_weight_sync_time = 0.0
    total_num_new_tokens = 0
    finish_reasons: collections.defaultdict[str, int] = collections.defaultdict(int)
    response_lengths: list[int] = []
    prompt_lengths: list[int] = []

    for result in results:
        total_mfu += result["mfu"]
        total_mbu += result["mbu"]
        total_tokens_per_second += result["tokens_per_second"]
        total_generation_time += result["generation_time"]
        total_weight_sync_time += result["weight_sync_time"]
        total_num_new_tokens += result["num_new_tokens"]
        for reason, count in result["finish_reasons"].items():
            finish_reasons[reason] += count
        response_lengths.extend(result["response_lengths"])
        prompt_lengths.extend(result["prompt_lengths"])

    num_results = len(results)
    avg_tokens_per_second = total_num_new_tokens / total_generation_time if total_generation_time > 0 else 0
    avg_mfu = total_mfu / num_results
    avg_mbu = total_mbu / num_results
    avg_generation_time = total_generation_time / num_results
    avg_weight_sync_time = total_weight_sync_time / num_results

    return {
        "total_mfu": total_mfu,
        "total_mbu": total_mbu,
        "total_tokens_per_second": total_tokens_per_second,
        "total_generation_time": total_generation_time,
        "total_weight_sync_time": total_weight_sync_time,
        "total_num_new_tokens": total_num_new_tokens,
        "finish_reasons": finish_reasons,
        "response_lengths": response_lengths,
        "prompt_lengths": prompt_lengths,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_mfu": avg_mfu,
        "avg_mbu": avg_mbu,
        "avg_generation_time": avg_generation_time,
        "avg_weight_sync_time": avg_weight_sync_time,
    }


def print_summary(
    results: list[dict[str, Any]],
    total_time: float,
    args: grpo_fast.Args,
    model_config: model_utils.ModelConfig,
    model_dims: utils.ModelDims,
) -> None:
    """Print benchmark summary statistics."""

    agg_results = aggregate_results(results)
    total_samples = len(results) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    avg_new_tokens_per_sample = agg_results["total_num_new_tokens"] / total_samples

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
    print(f"Total time (main benchmark): {agg_results['total_generation_time']:.2f}s")
    print(f"Total weight sync time: {agg_results['total_weight_sync_time']:.2f}s")
    print(f"Total new tokens generated: {agg_results['total_num_new_tokens']}")
    print("-" * 60)
    print(f"Average results over {len(results)} main benchmark batches:")
    print(f"Average tokens/second: {agg_results['avg_tokens_per_second']:.2f}")
    print(f"Average MFU: {agg_results['avg_mfu']:.2f}%")
    print(f"Average MBU: {agg_results['avg_mbu']:.2f}%")
    print(f"Average generation time per batch: {agg_results['avg_generation_time']:.2f}s")
    print(f"Average weight sync time per batch: {agg_results['avg_weight_sync_time']:.2f}s")
    print(f"Average new tokens per sample: {avg_new_tokens_per_sample:.2f} tokens")

    max_length = np.max(agg_results["response_lengths"])
    mean_length = np.mean(agg_results["response_lengths"])
    wasted_compute = (max_length - mean_length) / max_length
    print(f"Wasted compute % (variable response length): {wasted_compute:.2%}")

    print("-" * 60)
    print("HARDWARE SPECIFICATIONS:")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU peak FLOPs: {model_dims.device_flops / 1e12:.0f} TFLOPs")
    gpu_specs = utils.GPU_SPECS[model_dims.device_name]
    print(f"GPU memory size: {gpu_specs['memory_size'] / 1e9:.0f} GB")
    print(f"GPU memory bandwidth: {model_dims.device_memory_bandwidth / 1e12:.2f} TB/s")

    print("-" * 60)
    print("COMPLETION LENGTH STATISTICS:")
    print(f"Total completions: {len(agg_results['response_lengths'])}")
    print("\nResponse lengths:")
    print(f"- Min: {min(agg_results['response_lengths'])} tokens")
    print(f"- Max: {max(agg_results['response_lengths'])} tokens")
    print(f"- Mean: {np.mean(agg_results['response_lengths']):.2f} tokens")
    print(f"- Median: {np.median(agg_results['response_lengths']):.2f} tokens")

    # Calculate percentiles for valid tokens
    print("\nResponse length percentiles:")
    print(f"- 25th percentile: {np.percentile(agg_results['response_lengths'], 25):.2f} tokens")
    print(f"- 50th percentile: {np.percentile(agg_results['response_lengths'], 50):.2f} tokens")
    print(f"- 75th percentile: {np.percentile(agg_results['response_lengths'], 75):.2f} tokens")
    print(f"- 90th percentile: {np.percentile(agg_results['response_lengths'], 90):.2f} tokens")
    print(f"- 95th percentile: {np.percentile(agg_results['response_lengths'], 95):.2f} tokens")
    print(f"- 99th percentile: {np.percentile(agg_results['response_lengths'], 99):.2f} tokens")

    print("=" * 60)


def cleanup(vllm_engines: list[ray.actor.ActorHandle], actor_manager: ray.actor.ActorHandle | None = None) -> None:
    """Clean up resources."""
    for engine in vllm_engines:
        ray.kill(engine)

    ray.shutdown()


def main() -> None:
    """Main benchmark function."""
    # Parse arguments using ArgumentParserPlus
    parser = utils.ArgumentParserPlus(
        (grpo_fast.Args, dataset_transformation.TokenizerConfig, model_utils.ModelConfig)  # type: ignore[arg-type]
    )

    args, tokenizer_config, model_config = cast(
        tuple[grpo_fast.Args, dataset_transformation.TokenizerConfig, model_utils.ModelConfig],
        parser.parse_args_into_dataclasses(),
    )

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate flops per token before starting vLLM
    logger.info("Calculating model FLOPs per token...")

    # Free GPU memory after calculating FLOPs and before starting vLLM
    logger.info("Freeing GPU memory before starting vLLM...")
    free_all_gpu_memory()

    dataset = setup_dataset(args, tokenizer_config)
    max_model_len = args.max_prompt_token_length + args.response_length
    vllm_engines, param_prompt_Q, inference_results_Q, actor_manager = setup_vllm_engines(
        args, tokenizer_config, model_config, max_model_len
    )

    # Create the timestamp here so we use it for both filenames.
    timestamp = int(time.time())
    save_config(args, tokenizer_config, model_config, timestamp)
    run_benchmark(
        dataset, vllm_engines, param_prompt_Q, inference_results_Q, actor_manager, args, model_config, timestamp
    )

    cleanup(vllm_engines, actor_manager)


if __name__ == "__main__":
    main()
