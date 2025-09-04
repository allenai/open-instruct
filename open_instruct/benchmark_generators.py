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
from open_instruct.actor_manager import ActorManager
from open_instruct.queue_types import PromptRequest

# For FLOPS, we assume bf16 and ignore sparsity.
# Memory bandwidth values are peak theoretical bandwidth.
GPU_SPECS = {
    "a100": {"flops": 312e12, "memory_size": 80e9, "memory_bandwidth": 1.6e12},  # 1.6 TB/s HBM2e
    "b200": {"flops": 2250e12, "memory_size": 192e9, "memory_bandwidth": 8e12},  # 8 TB/s HBM3e
    "h100": {"flops": 990e12, "memory_size": 80e9, "memory_bandwidth": 3.35e12},  # 3.35 TB/s HBM3
    "a6000": {"flops": 155e12, "memory_size": 48e9, "memory_bandwidth": 768e9},  # 768 GB/s GDDR6
    "l40s": {"flops": 362e12, "memory_size": 48e9, "memory_bandwidth": 864e9},  # 864 GB/s GDDR6
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
        "generation_time_percentage": (agg_results["total_generation_time"] / total_time) * 100,
        "total_tokens": agg_results["total_num_new_tokens"],
        "avg_tokens_per_second": agg_results["avg_tokens_per_second"],
        "avg_mfu": agg_results["avg_mfu"],
        "avg_mbu": agg_results["avg_mbu"],
        "avg_generation_time_per_batch": agg_results["avg_generation_time"],
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

    def weight_memory_bytes(self, num_tokens: int, dtype_bytes: int = 2) -> int:
        """Memory bytes for reading model weights for a given number of tokens.

        Args:
            num_tokens: Number of tokens to process
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total bytes for weight reads across all layers
        """
        num_kv = self.num_kv_heads if self.num_kv_heads is not None else self.num_attn_heads
        head_dim = self.hidden_size // self.num_attn_heads
        hidden_kv = num_kv * head_dim

        # Per-layer weight params (Q, K, V, O, MLP up, MLP down)
        w_q = self.hidden_size * self.hidden_size
        w_k = self.hidden_size * hidden_kv
        w_v = self.hidden_size * hidden_kv
        w_o = self.hidden_size * self.hidden_size
        w_up = self.hidden_size * self.intermediate_size
        w_dn = self.intermediate_size * self.hidden_size

        per_layer_weight_bytes = (w_q + w_k + w_v + w_o + w_up + w_dn) * dtype_bytes
        return self.num_layers * num_tokens * per_layer_weight_bytes

    def kv_cache_write_bytes(self, num_tokens: int, dtype_bytes: int = 2) -> int:
        """Memory bytes for writing KV cache for a given number of tokens.

        Args:
            num_tokens: Number of tokens being cached
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total bytes for KV cache writes across all layers
        """
        num_kv = self.num_kv_heads if self.num_kv_heads is not None else self.num_attn_heads
        head_dim = self.hidden_size // self.num_attn_heads

        # 2x for K and V
        kv_write_bytes_per_token = 2 * num_kv * head_dim * dtype_bytes
        return self.num_layers * num_tokens * kv_write_bytes_per_token

    def kv_cache_read_bytes(
        self,
        prompt_lengths: Sequence[int],
        response_lengths: Sequence[int],
        samples_per_prompt: int = 1,
        dtype_bytes: int = 2,
    ) -> int:
        """Memory bytes for reading KV cache during decode.

        For each new token generated, we read all previous tokens' KV cache.
        When generating multiple samples per prompt, the prompt KV cache is shared.

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total bytes for KV cache reads during decode
        """
        assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
            f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
        )

        num_kv = self.num_kv_heads if self.num_kv_heads is not None else self.num_attn_heads
        head_dim = self.hidden_size // self.num_attn_heads

        # For batched sampling with shared prompt KV cache:
        # - Prompt KV is read once per new token position across ALL samples (not per sample)
        # - Each sample has its own KV for generated tokens
        kv_read_terms = 0
        response_idx = 0

        for P in prompt_lengths:
            # For this prompt, collect all response lengths
            prompt_responses = []
            for _ in range(samples_per_prompt):
                prompt_responses.append(response_lengths[response_idx])
                response_idx += 1

            # Prompt KV reads: In synchronized batch generation with vLLM n>1,
            # the prompt KV cache is stored once but each sample reads it independently.
            # At each decoding position, each sample reads the prompt KV cache.
            # Number of positions = max response length (all generate synchronously)
            max_response_length = max(prompt_responses) if prompt_responses else 0
            # Each of the samples_per_prompt samples reads prompt KV at each position
            kv_read_terms += max_response_length * samples_per_prompt * P

            # Per-sample generated KV reads: Each sample reads its own previously generated tokens
            for R in prompt_responses:
                # Each token in this sample reads its previously generated tokens
                # sum_{i=0}^{R-1} i = R*(R-1)/2
                kv_read_terms += R * (R - 1) // 2

        # 2x for K and V
        kv_bytes_per_token = 2 * num_kv * head_dim * dtype_bytes
        return self.num_layers * kv_bytes_per_token * kv_read_terms

    def prefill_memory_bytes(self, prompt_lengths: Sequence[int], dtype_bytes: int = 2) -> int:
        """Memory bytes for prefill phase.

        During prefill:
        - Read weights once for the entire batch (batched matmul)
        - Write KV cache for each token

        Args:
            prompt_lengths: List of prompt lengths
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total memory bytes for prefill
        """
        # In batched prefill, weights are read once for the entire operation,
        # not once per token. We process all prompts in a single batch.
        num_prefill_batches = len(prompt_lengths)  # Each prompt is a "batch"
        weight_bytes = self.weight_memory_bytes(num_prefill_batches, dtype_bytes)

        # KV cache is written for every token
        total_prefill_tokens = sum(prompt_lengths)
        kv_write_bytes = self.kv_cache_write_bytes(total_prefill_tokens, dtype_bytes)
        return weight_bytes + kv_write_bytes

    def decode_memory_bytes(
        self,
        prompt_lengths: Sequence[int],
        response_lengths: Sequence[int],
        samples_per_prompt: int = 1,
        dtype_bytes: int = 2,
    ) -> int:
        """Memory bytes for decode/generation phase.

        During decode:
        - Read weights for each new token position (shared across samples in batch)
        - Write KV cache for each new token
        - Read all previous KV cache for attention

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total memory bytes for decode
        """
        # In synchronized batch generation, weights are read once per position,
        # not once per token. With multiple samples per prompt generating in parallel,
        # we only need to read weights for the number of unique positions.
        unique_positions = 0
        response_idx = 0
        for _ in prompt_lengths:
            # Get response lengths for this prompt's samples
            prompt_responses = response_lengths[response_idx : response_idx + samples_per_prompt]
            response_idx += samples_per_prompt
            # In synchronized generation, all samples generate the same number of positions
            # (up to the max length among them)
            unique_positions += max(prompt_responses) if prompt_responses else 0

        weight_bytes = self.weight_memory_bytes(unique_positions, dtype_bytes)

        # KV writes happen for all tokens (each sample writes its own KV)
        total_decode_tokens = sum(response_lengths)
        kv_write_bytes = self.kv_cache_write_bytes(total_decode_tokens, dtype_bytes)

        kv_read_bytes = self.kv_cache_read_bytes(prompt_lengths, response_lengths, samples_per_prompt, dtype_bytes)
        return weight_bytes + kv_write_bytes + kv_read_bytes

    def memory_bytes(
        self,
        prompt_lengths: Sequence[int],
        response_lengths: Optional[Sequence[int]] = None,
        samples_per_prompt: int = 1,
        dtype_bytes: int = 2,
    ) -> int:
        """Approximate total HBM bytes moved for prefill + decode.

        Returns an integer number of bytes. Divide by elapsed seconds to get B/s;
        compare against peak bandwidth to get utilization.

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total memory bytes moved

        Assumptions:
          - Weights are read once per token per layer (Q,K,V,O + MLP up/down)
          - KV cache: write K/V for every token; during decode, read all past K/V per new token
          - When batching samples, prompt KV cache is shared across samples
          - Embedding and LM head reads are ignored (usually dominated by matmul weight traffic)
        """
        total = self.prefill_memory_bytes(prompt_lengths, dtype_bytes)

        if response_lengths is not None:
            assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
                f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
            )

            # Pass original prompt_lengths with samples_per_prompt to correctly handle shared KV cache
            total += self.decode_memory_bytes(prompt_lengths, response_lengths, samples_per_prompt, dtype_bytes)

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

    queues_to_monitor = {"Param Prompt Queue": param_prompt_Q, "Inference Results Queue": inference_results_Q}
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args)

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

        # Create list of dataset indices for this batch
        dataset_indices = list(range(start_idx, end_idx))

        param_prompt_Q.put(
            PromptRequest(
                prompts=prompts,
                dataset_index=dataset_indices,
                generation_config=generation_config,
                start_time=time.perf_counter(),
            )
        )
    logger.info(f"[Submission Thread] All {num_batches} batches submitted")


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
    executor = futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="benchmark")

    generation_future = executor.submit(generate_thread, vllm_engines, stop_event)

    results = []
    device_name = get_device_name(torch.cuda.get_device_name(0))
    device_flops = GPU_SPECS[device_name]["flops"]
    device_memory_bandwidth = GPU_SPECS[device_name]["memory_bandwidth"]

    # Submit warmup batch first
    logger.info("Submitting warmup batch...")
    warmup_start_idx = 0
    warmup_end_idx = min(args.num_unique_prompts_rollout, len(dataset))
    warmup_data = dataset[warmup_start_idx:warmup_end_idx]
    warmup_prompts = warmup_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    warmup_dataset_indices = list(range(warmup_start_idx, warmup_end_idx))
    param_prompt_Q.put(
        PromptRequest(
            prompts=warmup_prompts,
            dataset_index=warmup_dataset_indices,
            generation_config=generation_config,
            start_time=time.perf_counter(),
        )
    )
    model_dims = load_model_dims(model_config.model_name_or_path)

    try:
        logger.info("Running warmup batch...")

        warmup_result = inference_results_Q.get()
        logger.info(f"Warmup batch completed with {len(warmup_result.responses)} responses")
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
            [future.result() for future in [submission_future, generation_future] if future.done()]
            result = inference_results_Q.get()

            completion_time = time.perf_counter()
            # Calculate generation time from when the request was enqueued
            batch_generation_time = completion_time - result.start_time if result.start_time else 0

            new_tokens = sum(len(response) for response in result.responses)
            tokens_per_second = new_tokens / batch_generation_time if batch_generation_time > 0 else 0

            result_dict = {
                "tokens_per_second": tokens_per_second,
                "generation_time": batch_generation_time,
                "num_new_tokens": new_tokens,
                "finish_reasons": collections.Counter(result.finish_reasons),
                "response_lengths": [len(response) for response in result.responses],
                "dataset_indices": result.dataset_index,
            }
            # Get prompt lengths using dataset indices from the result
            prompt_data = dataset[result.dataset_index]
            prompts = prompt_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
            prompt_lengths = [len(prompt) for prompt in prompts]
            response_lengths = [len(response) for response in result.responses]

            # Calculate total FLOPs for all prompts and responses in the batch
            # No need to expand prompt_lengths - the flops method now handles samples_per_prompt
            model_flops = model_dims.flops(
                prompt_lengths, response_lengths, samples_per_prompt=args.num_samples_per_prompt_rollout
            )

            # MFU = (FLOPs / time) / peak_FLOPS * 100
            model_flops_per_second = model_flops / batch_generation_time if batch_generation_time > 0 else 0
            result_dict["mfu"] = 100 * model_flops_per_second / device_flops

            # Calculate total memory bytes for all prompts and responses in the batch
            model_memory_bytes = model_dims.memory_bytes(
                prompt_lengths, response_lengths, samples_per_prompt=args.num_samples_per_prompt_rollout
            )

            # MBU = (Memory bytes / time) / peak_bandwidth * 100
            model_bytes_per_second = model_memory_bytes / batch_generation_time if batch_generation_time > 0 else 0
            result_dict["mbu"] = 100 * model_bytes_per_second / device_memory_bandwidth

            save_completion_lengths([result_dict], timestamp, batch_idx)
            results.append(result_dict)
            logger.info(
                f"Batch {batch_idx}/{num_batches - 1}: "
                f"{result_dict['tokens_per_second']:.2f} new tokens/sec, "
                f"MFU: {result_dict['mfu']:.2f}%, "
                f"MBU: {result_dict['mbu']:.2f}%, "
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


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate total and aggregated metrics from results."""
    aggregated_results = {
        "total_mfu": 0.0,
        "total_mbu": 0.0,
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
            elif key == "mbu":
                aggregated_results["total_mbu"] += value
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
    aggregated_results["avg_tokens_per_second"] = (
        aggregated_results["total_num_new_tokens"] / aggregated_results["total_generation_time"]
        if aggregated_results["total_generation_time"] > 0
        else 0
    )
    aggregated_results["avg_mfu"] = aggregated_results["total_mfu"] / num_results
    aggregated_results["avg_mbu"] = aggregated_results["total_mbu"] / num_results
    aggregated_results["avg_generation_time"] = aggregated_results["total_generation_time"] / num_results
    return aggregated_results


def print_summary(
    results: list[dict[str, Any]], total_time: float, args: grpo_fast.Args, model_config: model_utils.ModelConfig
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
    print(f"Total new tokens generated: {agg_results['total_num_new_tokens']}")
    print("-" * 60)
    print(f"Average results over {len(results)} main benchmark batches:")
    print(f"Average tokens/second: {agg_results['avg_tokens_per_second']:.2f}")
    print(f"Average MFU: {agg_results['avg_mfu']:.2f}%")
    print(f"Average MBU: {agg_results['avg_mbu']:.2f}%")
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
    print(f"GPU memory bandwidth: {gpu_specs['memory_bandwidth'] / 1e12:.2f} TB/s")

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
