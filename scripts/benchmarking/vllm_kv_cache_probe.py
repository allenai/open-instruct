"""Probe vLLM's actual KV-cache max concurrency ("max inference batch size").

For each (tensor_parallel_size, response_length) in the sweep, boots a real
vLLM engine and reports the max concurrency computed from the KV cache blocks
vLLM *actually* allocated after its memory-profiling pass. This is the same
computation as `open_instruct.vllm_utils.LLMRayActor.get_kv_cache_info`, so it
reflects real weight / activation / CUDA-graph memory and hybrid (Mamba +
attention) KV-cache layouts that an offline formula cannot model.

Run once on a GPU node with `--num_gpus` GPUs (default 8). The driver sweeps the
configs by spawning one isolated subprocess per config (a fresh process per
engine, since vLLM cannot cleanly switch tensor-parallel size in-process). Each
config uses `tensor_parallel_size` GPUs; per-engine concurrency is
data-parallel-independent, so the node total is per_engine * (num_gpus // tp).

max_model_len matches the benchmark: max_prompt_token_length + response_length.

Example (single launch, sweeps all six configs):
    uv run python scripts/benchmarking/vllm_kv_cache_probe.py \
        --model_name_or_path Qwen/Qwen3.6-35B-A3B \
        --tensor_parallel_sizes 2,4,8 --response_lengths 16384,32768
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import vllm
from vllm.v1 import kv_cache_interface
from vllm.v1.core import kv_cache_utils

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def run_worker(args) -> None:
    max_model_len = args.max_prompt_token_length + args.response_length
    llm = vllm.LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    vllm_config = llm.llm_engine.vllm_config
    kv_cache_specs = llm.collective_rpc("get_kv_cache_spec")
    kv_cache_groups = kv_cache_utils.get_kv_cache_groups(vllm_config, kv_cache_specs[0])
    kv_cache_config = kv_cache_interface.KVCacheConfig(
        num_blocks=vllm_config.cache_config.num_gpu_blocks, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
    )
    per_engine = int(kv_cache_utils.get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config))
    result = {
        "per_engine": per_engine,
        "num_gpu_blocks": vllm_config.cache_config.num_gpu_blocks,
        "max_model_len": max_model_len,
    }
    with open(args.result_path, "w") as f:
        json.dump(result, f)


def run_driver(args) -> None:
    tps = [int(x) for x in args.tensor_parallel_sizes.split(",")]
    response_lengths = [int(x) for x in args.response_lengths.split(",")]
    rows = []
    for response_length in response_lengths:
        for tp in tps:
            dp = args.num_gpus // tp
            logger.info(f"Probing tp={tp} dp={dp} response_length={response_length}")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(tp))}
            result_path = os.path.join(tempfile.gettempdir(), f"kv_probe_tp{tp}_rl{response_length}.json")
            if os.path.exists(result_path):
                os.remove(result_path)
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--model_name_or_path",
                args.model_name_or_path,
                "--tensor_parallel_size",
                str(tp),
                "--max_prompt_token_length",
                str(args.max_prompt_token_length),
                "--response_length",
                str(response_length),
                "--gpu_memory_utilization",
                str(args.gpu_memory_utilization),
                "--result_path",
                result_path,
            ]
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0 or not os.path.exists(result_path):
                logger.error(f"tp={tp} response_length={response_length} failed (returncode={proc.returncode})")
                rows.append((tp, dp, response_length, None, None, None))
                continue
            with open(result_path) as f:
                result = json.load(f)
            rows.append(
                (tp, dp, response_length, result["num_gpu_blocks"], result["per_engine"], result["per_engine"] * dp)
            )

    logger.info(f"Results for {args.model_name_or_path}:")
    logger.info(f"{'tp':>3} {'dp':>3} {'resp_len':>9} {'gpu_blocks':>11} {'per_engine':>11} {'total':>8}")
    for tp, dp, response_length, blocks, per_engine, total in rows:
        per_engine_s = "FAILED" if per_engine is None else str(per_engine)
        total_s = "-" if total is None else str(total)
        blocks_s = "-" if blocks is None else str(blocks)
        logger.info(f"{tp:>3} {dp:>3} {response_length:>9} {blocks_s:>11} {per_engine_s:>11} {total_s:>8}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--max_prompt_token_length", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--worker", action="store_true", help="Internal: run a single config and print its result.")
    parser.add_argument("--tensor_parallel_size", type=int, help="Worker mode: single tp.")
    parser.add_argument("--response_length", type=int, help="Worker mode: single response length.")
    parser.add_argument("--result_path", help="Worker mode: file path to write the JSON result to.")
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Driver mode: GPUs on the node (sets dp = num_gpus // tp)."
    )
    parser.add_argument(
        "--tensor_parallel_sizes", default="2,4,8", help="Driver mode: comma-separated tp values to sweep."
    )
    parser.add_argument(
        "--response_lengths", default="16384,32768", help="Driver mode: comma-separated response lengths."
    )
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
