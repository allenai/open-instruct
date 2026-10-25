"""Diagnostic-only forced-token decode; never use its timings as throughput.

The identity logit processor installs a token-selection hook after model logits
and raw logprobs are computed. Position copies synchronize the GPU with the CPU.
Keep these runs separate from ordinary sampled throughput measurements.
"""

import asyncio
import base64
import hashlib
import json
import math
from pathlib import Path

import httpx
import numpy as np
import torch
from sglang.srt.layers import sampler
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

from open_instruct.miles.rollout import benchmark_metrics


class ReplayIdentityProcessor(CustomLogitProcessor):
    def __init__(self):
        if not hasattr(sampler.Sampler, "_oi_benchmark_original_sampling"):
            sampler.Sampler._oi_benchmark_original_sampling = sampler.Sampler._sample_from_probs
            sampler.Sampler._sample_from_probs = select_frozen_tokens
        sampler.SGLANG_RETURN_ORIGINAL_LOGPROB = True

    def __call__(self, logits, custom_param_list=None):
        return logits


def select_frozen_tokens(self, probs, sampling_info, positions, simple_sampling_case):
    parameters = sampling_info.custom_params
    if parameters is None:
        return self._oi_benchmark_original_sampling(probs, sampling_info, positions, simple_sampling_case)
    if not simple_sampling_case or sampling_info.is_all_greedy:
        raise ValueError("Benchmark replay requires temperature one and no truncation")
    selected = []
    for params, position in zip(parameters, positions.cpu().tolist(), strict=True):
        offset = position + 1 - params["benchmark_prompt_length"]
        trajectory = params["benchmark_output_ids"]
        # Intermediate prefill and one overlap step can sample discarded tokens.
        selected.append(trajectory[offset] if 0 <= offset < len(trajectory) else 0)
    return torch.tensor(selected, dtype=torch.int32, device=probs.device)


async def run(args, settings, root):
    reference = Path(settings["replay_reference"])
    current = json.loads((root / "run-spec.json").read_text())
    raw, records = benchmark_metrics.reference_records(reference, current["model"], args.rollout_max_context_len)
    reference_trajectories = len(records)
    records = benchmark_metrics.replicate_trajectories(records, settings["replay_copies"])
    processor = ReplayIdentityProcessor.to_str()
    capacity = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with httpx.AsyncClient(
        timeout=settings["collection_timeout"], limits=httpx.Limits(max_connections=capacity)
    ) as client:
        for repeat in range(settings["replay_repeats"]):
            route_root = root / "benchmark-replay-routes" / str(repeat)
            route_root.mkdir(parents=True)

            async def generate(record, route_root=route_root):
                prompt_length = record["prompt_tokens"]
                output_ids = record["tokens"][prompt_length:]
                payload = {
                    "input_ids": record["tokens"][:prompt_length],
                    "sampling_params": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_new_tokens": len(output_ids),
                        "ignore_eos": True,
                        "custom_params": {
                            "benchmark_prompt_length": prompt_length,
                            "benchmark_output_ids": output_ids,
                        },
                    },
                    "custom_logit_processor": processor,
                    "return_logprob": True,
                    "return_routed_experts": args.use_rollout_routing_replay,
                }
                response = await client.post(url, json=payload)
                response.raise_for_status()
                meta = response.json()["meta_info"]
                probabilities = meta["output_token_logprobs"]
                if [row[1] for row in probabilities] != output_ids or not all(
                    math.isfinite(row[0]) for row in probabilities
                ):
                    raise ValueError("Diagnostic replay changed the frozen tokens or returned invalid logprobs")
                routes = (
                    base64.b64decode(meta["routed_experts"], validate=True)
                    if args.use_rollout_routing_replay
                    else None
                )
                if routes is not None and len(routes) != record["route_bytes"]:
                    raise ValueError("Diagnostic replay returned misaligned routing payload")
                route_path = None
                if routes is not None:
                    route_path = route_root / f"{record['index']:08d}.npz"
                    matrix = np.frombuffer(routes, dtype=np.int32).reshape(record["route_shape"])
                    await asyncio.to_thread(np.savez_compressed, route_path, routes=matrix)
                return {
                    "index": record["index"],
                    "source_index": record["source_index"],
                    "copy_index": record["copy_index"],
                    "group_index": record["group_index"],
                    "prompt_tokens": prompt_length,
                    "response_tokens": len(output_ids),
                    "logprobs": [row[0] for row in probabilities],
                    "route_sha256": hashlib.sha256(routes).hexdigest() if routes is not None else None,
                    "route_path": str(route_path.relative_to(root)) if route_path is not None else None,
                }

            output = await asyncio.gather(*(generate(record) for record in records))
            (root / f"benchmark-replay-{repeat}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in output))
    return {
        "diagnostic_only": True,
        "reference": str(reference),
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "trajectories": len(records),
        "reference_trajectories": reference_trajectories,
        "copies": settings["replay_copies"],
        "repeats": settings["replay_repeats"],
        "response_tokens_per_repeat": sum(r["response_tokens"] for r in records),
        "scope": "Forced-token cached decode with raw model logprobs; timings are not throughput.",
    }
