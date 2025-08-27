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
from typing import Any, ClassVar, Optional, Sequence

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
    results: list[dict[str, Any]],
    total_time: float,
    overall_tokens_per_second: float,
    args: grpo_fast.Args,
    model_config: model_utils.ModelConfig,
) -> None:
    """Save benchmark results to CSV file."""
    git_commit = get_git_commit()
    agg_results = aggregate_results(results)
    csv_path: pathlib.Path = DATA_DIR / "generator_benchmark_results.csv"

    # Calculate overall MFU using ModelDims
    device_name = get_device_name(torch.cuda.get_device_name(0))
    device_flops = GPU_SPECS[device_name]["flops"]

    # Load model dims for FLOPS calculation
    model_dims = load_model_dims(model_config.model_name_or_path)

    # Collect all prompt and response lengths from results
    all_prompt_lengths = []
    all_response_lengths = []
    for result in results:
        if "prompt_lengths" in result:
            all_prompt_lengths.extend(result["prompt_lengths"])
        if "response_lengths" in result:
            all_response_lengths.extend(result["response_lengths"])

    # Calculate total FLOPs for all prompts and responses
    total_flops = model_dims.flops(
        all_prompt_lengths, all_response_lengths, samples_per_prompt=args.num_samples_per_prompt_rollout
    )

    # Calculate overall MFU
    overall_mfu = 100 * (total_flops / total_time) / device_flops

    # Prepare row data
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
        "generation_time_percentage": (agg_results["total_generation_time"] / total_time) * 100,
        "total_tokens": agg_results["total_num_new_tokens"],
        "avg_tokens_per_second": agg_results["avg_tokens_per_second"],
        "avg_mfu": agg_results["avg_mfu"],
        "avg_generation_time_per_batch": agg_results["avg_generation_time"],
        "avg_new_tokens_per_sample": agg_results["total_num_new_tokens"]
        / (len(results) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout),
        "overall_mfu": overall_mfu,
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


@dataclasses.dataclass
class ModelDims:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attn_heads: int
    num_kv_heads: Optional[int] = None

    # Conventions (fixed; not switches)
    FLOP_PER_MAC: ClassVar[int] = 2
    # Approximate softmax cost per attention score:
    # ~4 scalar ops/score: exp + subtract max (stabilization) + sum + divide.
    SOFTMAX_FLOPS_PER_SCORE: ClassVar[int] = 4

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attn_heads

        assert self.hidden_size % self.num_attn_heads == 0, "hidden_size must be divisible by num_attn_heads"
        assert self.num_attn_heads % self.num_kv_heads == 0, (
            "num_attn_heads must be divisible by num_kv_heads (GQA/MQA)"
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attn_heads

    def attn_flops(self, query_len: int, kv_len: int) -> int:
        """FLOPs for one layer of self-attention given query_len and kv_len.

        Assumptions:
          - 1 MAC = 2 FLOPs (FLOP_PER_MAC).
          - Efficient GQA/MQA K/V projections with width = num_kv_heads * head_dim.
          - Softmax ≈ 4 FLOPs per score (see SOFTMAX_FLOPS_PER_SCORE).
          - LayerNorms and minor ops ignored (dominated by matmuls).
        """
        d = self.head_dim
        mul = self.FLOP_PER_MAC

        # Projections for the query_len new tokens
        q_proj = mul * query_len * self.hidden_size * self.hidden_size
        kv_proj = mul * 2 * query_len * self.hidden_size * (self.num_kv_heads * d)  # GQA/MQA

        # Scores and attention-weighted values
        qk = mul * self.num_attn_heads * query_len * kv_len * d
        softmax = self.SOFTMAX_FLOPS_PER_SCORE * self.num_attn_heads * query_len * kv_len
        av = mul * self.num_attn_heads * query_len * kv_len * d

        # Output projection
        out_proj = mul * query_len * self.hidden_size * self.hidden_size

        return q_proj + kv_proj + qk + softmax + av + out_proj

    def mlp_flops(self, seq_len: int) -> int:
        """Two matmuls dominate; activation cost under-counted on purpose."""
        mul = self.FLOP_PER_MAC
        first = mul * seq_len * self.hidden_size * self.intermediate_size
        act = seq_len * self.intermediate_size  # under-counted on purpose
        second = mul * seq_len * self.intermediate_size * self.hidden_size
        return first + act + second

    def prefill_flops(self, prompt_lengths: Sequence[int]) -> int:
        """Prefill builds the KV cache; logits are computed once after each prompt."""
        total = 0
        for L in prompt_lengths:
            total += self.num_layers * (self.attn_flops(L, L) + self.mlp_flops(L))
            # Always include a single LM head after prefill (next-token logits)
            total += self.FLOP_PER_MAC * self.hidden_size * self.vocab_size
        return total

    def decode_flops(
        self, prompt_lengths: Sequence[int], response_lengths: Sequence[int], samples_per_prompt: int = 1
    ) -> int:
        """Decode/generation FLOPs.

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt

        Embedding lookups are ignored by design.
        """
        assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
            f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
        )

        total = 0
        response_idx = 0
        for P in prompt_lengths:
            # Process all samples for this prompt
            for _ in range(samples_per_prompt):
                R = response_lengths[response_idx]
                total += R * self.num_layers * self.mlp_flops(seq_len=1)
                for t in range(R):
                    kv_len = P + t + 1  # prompt + generated so far + current
                    total += self.num_layers * self.attn_flops(query_len=1, kv_len=kv_len)
                total += R * self.FLOP_PER_MAC * self.hidden_size * self.vocab_size
                response_idx += 1
        return total

    def flops(
        self,
        prompt_lengths: Sequence[int],
        response_lengths: Optional[Sequence[int]] = None,
        samples_per_prompt: int = 1,
    ) -> int:
        """Total FLOPs for prefill and (optionally) decode.

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
        """
        total = self.prefill_flops(prompt_lengths)
        if response_lengths is not None:
            total += self.decode_flops(prompt_lengths, response_lengths, samples_per_prompt)
        return total


def load_model_dims(model_name: str) -> ModelDims:
    cfg = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return ModelDims(
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_attn_heads=cfg.num_attention_heads,
        num_kv_heads=getattr(cfg, "num_key_value_heads", None),
    )


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
    args: grpo_fast.Args, model_config: model_utils.ModelConfig, max_model_len: int = 20480, num_batches: int = 5
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

    # Queue size needs to accommodate all individual prompts across all batches.
    # Each batch has num_unique_prompts_rollout prompts, and we submit them individually.
    # Total individual prompts = num_unique_prompts_rollout * num_batches
    queue_size = args.num_unique_prompts_rollout * num_batches
    param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)

    actor_manager = ray.remote(vllm_utils3.ActorManager).remote()

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
        inference_batch_size=args.inference_batch_size,
        pg=pg,
        tools={},
        max_tool_calls=[0],
        prompt_queue=param_prompt_Q,
        results_queue=inference_results_Q,
        actor_manager=actor_manager,
    )

    logger.info("vLLM engines ready")

    return vllm_engines, param_prompt_Q, inference_results_Q, actor_manager


def generate_thread(vllm_engines: list[ray.actor.ActorHandle], stop_event: threading.Event) -> None:
    """Thread that repeatedly calls process_from_queue on vllm engines."""
    logger.info("[Generate Thread] Starting generation thread")
    while not stop_event.is_set():
        try:
            # Use ray.wait with a timeout to allow checking stop_event periodically
            futures = [engine.process_from_queue.remote(timeout=20) for engine in vllm_engines]
            ready, not_ready = ray.wait(futures, timeout=1.0)  # Check every 1 second

            if not ready and stop_event.is_set():
                logger.info("[Generate Thread] Stopping due to stop event")
                break

            # Get all results (this will block if not all are ready)
            if ready:
                processed_results = ray.get(ready)
                # Cancel any not ready futures if stop event is set
                if not_ready and stop_event.is_set():
                    break
                # Wait for remaining if not stopping
                if not_ready:
                    remaining = ray.get(not_ready)
                    processed_results.extend(remaining)
            else:
                continue  # No results ready yet, loop again

            num_processed = sum(int(result) for result in processed_results)
            if num_processed == 0:
                time.sleep(1)
            else:
                logger.debug(f"[Generate Thread] Processed {num_processed} requests")
        except Exception as e:
            if stop_event.is_set():
                logger.info("[Generate Thread] Interrupted while stopping")
                break
            logger.error(f"[Generate Thread] Error: {e}")
            raise
    logger.info("[Generate Thread] Exiting")


def submission_thread(
    param_prompt_Q: ray_queue.Queue,
    dataset: datasets.Dataset,
    generation_config: vllm.SamplingParams,
    stop_event: threading.Event,
    batch_size: int,
    start_batch_idx: int,
    num_batches: int,
) -> None:
    """Thread that submits individual prompts to the queue."""
    logger.info("[Submission Thread] Starting prompt submission")
    total_prompts_submitted = 0
    # Generate batches of prompts from the dataset
    for batch_idx in range(num_batches):
        if stop_event.is_set():
            logger.info("[Submission Thread] Stopped due to stop event")
            break
        # Submit each prompt individually, matching grpo_fast.py behavior
        for prompt_idx in range(batch_size):
            # Calculate the actual dataset index
            dataset_idx = start_batch_idx * batch_size + batch_idx * batch_size + prompt_idx
            if dataset_idx >= len(dataset):
                break

            prompt = dataset[dataset_idx][dataset_transformation.INPUT_IDS_PROMPT_KEY]
            # Note: batch_idx here is relative to the main benchmark batches
            actual_batch_idx = start_batch_idx + batch_idx
            actual_dataset_index = dataset_idx
            param_prompt_Q.put(
                PromptRequest(
                    prompt=prompt,
                    generation_config=generation_config,
                    training_step=actual_batch_idx,
                    dataset_index=actual_dataset_index,
                    is_eval=False,
                    start_time=time.perf_counter(),
                )
            )
            total_prompts_submitted += 1
    logger.info(f"[Submission Thread] All {total_prompts_submitted} individual prompts submitted")


def run_benchmark(
    dataset: datasets.Dataset,
    vllm_engines: list[ray.actor.ActorHandle],
    param_prompt_Q: ray_queue.Queue,
    inference_results_Q: ray_queue.Queue,
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
        top_p=args.vllm_top_p,
        n=args.num_samples_per_prompt_rollout,
        # IMPORTANT: Set output_kind to FINAL_ONLY to ensure vLLM V1 properly handles n>1
        # With the default CUMULATIVE mode, vLLM V1 returns separate outputs for each
        # completion, making it difficult to aggregate them correctly. FINAL_ONLY mode
        # ensures all n completions are returned together in a single output.
        output_kind=vllm.sampling_params.RequestOutputKind.FINAL_ONLY,
    )

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=len(vllm_engines) + 1, thread_name_prefix="benchmark")

    generation_future = executor.submit(generate_thread, vllm_engines, stop_event)
    submission_future = None  # Initialize to None for access in finally block

    results = []
    device_name = get_device_name(torch.cuda.get_device_name(0))
    device_flops = GPU_SPECS[device_name]["flops"]

    # Load model dims for FLOPS calculations
    model_dims = load_model_dims(model_config.model_name_or_path)

    logger.info("Submitting warmup batch...")
    for prompt_idx in range(args.num_unique_prompts_rollout):
        param_prompt_Q.put(
            PromptRequest(
                prompt=dataset[prompt_idx][dataset_transformation.INPUT_IDS_PROMPT_KEY],
                generation_config=generation_config,
                training_step=0,  # warmup is training step 0
                dataset_index=prompt_idx,
                is_eval=False,
            )
        )

    try:
        logger.info("Running warmup batch...")

        # Collect results for all warmup prompts
        for _ in range(args.num_unique_prompts_rollout):
            inference_results_Q.get()  # Discard warmup results
        logger.info("Warmup batch completed")
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

        # Start overall timing for main benchmark
        main_benchmark_start_time = time.perf_counter()

        # Process remaining batches with timing
        for batch_idx in range(1, num_batches):
            # Quick health check!
            [future.result() for future in [submission_future, generation_future] if future.done()]

            # Collect results for all prompts in this batch
            batch_results = []
            batch_start_time = time.perf_counter()
            for _ in range(args.num_unique_prompts_rollout):
                batch_results.append(inference_results_Q.get())

            completion_time = time.perf_counter()
            batch_generation_time = completion_time - batch_start_time

            # Debug logging to understand the response structure
            if batch_idx == 1:  # Only log for first main batch
                logger.info(f"[DEBUG] Batch {batch_idx} structure:")
                logger.info(f"  - Number of results collected: {len(batch_results)}")
                logger.info(f"  - Expected prompts: {args.num_unique_prompts_rollout}")
                logger.info(f"  - Expected samples per prompt: {args.num_samples_per_prompt_rollout}")
                for i, result in enumerate(batch_results[:2]):  # Log first 2 results
                    logger.info(f"  - Result {i}:")
                    logger.info(f"    - Number of responses: {len(result.responses)}")
                    logger.info(f"    - Response lengths: {[len(r) for r in result.responses]}")
                    logger.info(f"    - Number of finish_reasons: {len(result.finish_reasons)}")

            # Aggregate metrics from all individual results
            total_new_tokens = 0
            all_response_lengths = []
            all_finish_reasons = []

            for i, result in enumerate(batch_results):
                # Each result has n=num_samples_per_prompt_rollout responses
                result_tokens = sum(len(response) for response in result.responses)
                total_new_tokens += result_tokens
                all_response_lengths.extend([len(response) for response in result.responses])
                all_finish_reasons.extend(result.finish_reasons)

                # Extra debug for first batch
                if batch_idx == 1 and i < 2:
                    logger.info(f"  - Result {i} tokens: {result_tokens} from {len(result.responses)} responses")

            tokens_per_second = total_new_tokens / batch_generation_time if batch_generation_time > 0 else 0

            # Debug: Log expected vs actual tokens
            expected_total_responses = args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
            actual_total_responses = len(all_response_lengths)
            expected_tokens = expected_total_responses * args.response_length
            if batch_idx == 1:  # Only log for first main batch
                logger.info("[DEBUG] Token count analysis:")
                logger.info(
                    f"  - Expected responses: {expected_total_responses} ({args.num_unique_prompts_rollout} prompts × {args.num_samples_per_prompt_rollout} samples)"
                )
                logger.info(f"  - Actual responses: {actual_total_responses}")
                logger.info(f"  - Expected total tokens: {expected_tokens} (if all hit max)")
                logger.info(f"  - Actual total tokens: {total_new_tokens}")

            result_dict = {
                "tokens_per_second": tokens_per_second,
                "generation_time": batch_generation_time,
                "num_new_tokens": total_new_tokens,
                "finish_reasons": collections.Counter(all_finish_reasons),
                "response_lengths": all_response_lengths,
                "dataset_indices": [r.dataset_index for r in batch_results],
            }
            # Get prompt lengths for all results in this batch
            prompt_lengths = []
            response_lengths = all_response_lengths
            for r in batch_results:
                prompt_data = dataset[r.dataset_index]
                prompt = prompt_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
                # Only add one entry per unique prompt since flops() handles samples_per_prompt
                prompt_lengths.append(len(prompt))

            # Store expanded prompt lengths (one per response) for overall MFU calculation
            expanded_prompt_lengths = []
            for r in batch_results:
                prompt_data = dataset[r.dataset_index]
                prompt = prompt_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
                for _ in range(len(r.responses)):
                    expanded_prompt_lengths.append(len(prompt))
            result_dict["prompt_lengths"] = expanded_prompt_lengths

            # Calculate total FLOPs for all prompts and responses in the batch
            # prompt_lengths contains unique prompts, flops method handles samples_per_prompt
            model_flops = model_dims.flops(
                prompt_lengths, response_lengths, samples_per_prompt=args.num_samples_per_prompt_rollout
            )

            # MFU = (FLOPs / time) / peak_FLOPS * 100
            model_flops_per_second = model_flops / batch_generation_time
            result_dict["mfu"] = 100 * model_flops_per_second / device_flops

            save_completion_lengths([result_dict], timestamp, batch_idx)
            results.append(result_dict)
            logger.info(
                f"Batch {batch_idx}/{num_batches - 1}: "
                f"{result_dict['tokens_per_second']:.2f} new tokens/sec, "
                f"MFU: {result_dict['mfu']:.2f}%, "
                f"generation time: {batch_generation_time:.2f}s, "
                f"total new tokens: {total_new_tokens}"
            )

        # End overall timing for main benchmark
        main_benchmark_end_time = time.time()
        main_benchmark_total_time = main_benchmark_end_time - main_benchmark_start_time

        # Calculate total tokens generated in main benchmark
        total_main_tokens = sum(r["num_new_tokens"] for r in results)
        overall_tokens_per_second = (
            total_main_tokens / main_benchmark_total_time if main_benchmark_total_time > 0 else 0
        )

        # Debug: Log breakdown
        logger.info("[DEBUG] Token count breakdown:")
        logger.info(f"  - Number of main batches: {len(results)}")
        logger.info(f"  - Tokens per batch: {[r['num_new_tokens'] for r in results]}")
        logger.info(f"  - Total tokens across all batches: {total_main_tokens}")

        logger.info("\nOverall main benchmark performance:")
        logger.info(f"  Total wall-clock time: {main_benchmark_total_time:.2f}s")
        logger.info(f"  Total tokens generated: {total_main_tokens}")
        logger.info(f"  Overall tokens/second: {overall_tokens_per_second:.2f}")

        print_summary(results, main_benchmark_total_time, overall_tokens_per_second, args, model_config)
        save_benchmark_results_to_csv(
            results, main_benchmark_total_time, overall_tokens_per_second, args, model_config
        )
    finally:
        logger.info("Starting cleanup...")
        stop_event.set()

        # Wait for threads to finish with a timeout
        logger.info("Waiting for threads to complete...")
        try:
            # Give threads time to notice the stop event
            if submission_future:
                submission_future.result(timeout=5)
            if generation_future:
                generation_future.result(timeout=10)  # Give more time for generation thread
        except futures.TimeoutError:
            logger.warning("Threads did not complete within timeout, forcing shutdown")
        except Exception as e:
            logger.warning(f"Error waiting for threads: {e}")

        executor.shutdown(wait=True)
        logger.info("Threads cleaned up")


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
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
            elif key in ["response_lengths", "prompt_lengths"]:
                aggregated_results[key].extend(value)

    num_results = len(results)
    aggregated_results["avg_tokens_per_second"] = aggregated_results["total_tokens_per_second"] / num_results
    aggregated_results["avg_mfu"] = aggregated_results["total_mfu"] / num_results
    aggregated_results["avg_generation_time"] = aggregated_results["total_generation_time"] / num_results
    return aggregated_results


def print_summary(
    results: list[dict[str, Any]],
    total_time: float,
    overall_tokens_per_second: float,
    args: grpo_fast.Args,
    model_config: model_utils.ModelConfig,
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
    print(f"Total wall-clock time (main benchmark): {total_time:.2f}s")
    print(f"Total new tokens generated: {agg_results['total_num_new_tokens']}")
    print(f"Total time (main benchmark): {agg_results['total_generation_time']:.2f}s")
    print(f"Total new tokens generated: {agg_results['total_num_new_tokens']}")
    print(f"Overall tokens/second: {overall_tokens_per_second:.2f}")
    print("-" * 60)
    print(f"Average results over {len(results)} main benchmark batches:")
    print(f"Average tokens/second: {agg_results['avg_tokens_per_second']:.2f}")
    print(f"Average MFU: {agg_results['avg_mfu']:.2f}%")
    print(f"Average generation time per batch: {agg_results['avg_generation_time']:.2f}s")
    print(f"Average new tokens per sample: {avg_new_tokens_per_sample:.2f} tokens")

    max_length = np.max(agg_results["response_lengths"])
    mean_length = np.mean(agg_results["response_lengths"])
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

    # Free GPU memory after calculating FLOPs and before starting vLLM
    logger.info("Freeing GPU memory before starting vLLM...")
    free_all_gpu_memory()

    dataset = setup_dataset(args, tokenizer_config)
    vllm_engines, param_prompt_Q, inference_results_Q, actor_manager = setup_vllm_engines(args, model_config)

    # Create the timestamp here so we use it for both filenames.
    timestamp = int(time.time())
    save_config(args, tokenizer_config, model_config, timestamp)
    run_benchmark(dataset, vllm_engines, param_prompt_Q, inference_results_Q, args, model_config, timestamp)

    cleanup(vllm_engines, actor_manager)


if __name__ == "__main__":
    main()
