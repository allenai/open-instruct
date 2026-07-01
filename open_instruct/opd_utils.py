# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for online on-policy distillation."""

import dataclasses
import time
from typing import Any

import ray
import torch
import vllm
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from open_instruct import logger_utils, utils, vllm_utils
from open_instruct.distillkit.vllm_logprobs import extract_response_topk_from_prompt_logprobs

logger = logger_utils.setup_logger(__name__)


@dataclasses.dataclass
class OPDTeacherScoringResult:
    teacher_topk_token_ids: list[torch.Tensor]
    teacher_topk_logprobs: list[torch.Tensor]
    metrics: dict[str, float]


def build_teacher_sampling_params(topk: int) -> vllm.SamplingParams:
    return vllm.SamplingParams(
        top_k=-1, prompt_logprobs=topk, flat_logprobs=True, max_tokens=1, detokenize=False, skip_special_tokens=False
    )


class OPDTeacherScorerRayActor:
    """Ray actor that scores student rollouts under a fixed teacher model."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        model_revision: str | None,
        tokenizer_revision: str | None,
        tensor_parallel_size: int,
        enforce_eager: bool,
        dtype: str,
        seed: int,
        enable_prefix_caching: bool,
        max_model_len: int,
        gpu_memory_utilization: float,
        topk: int,
        distributed_executor_backend: str,
        trust_remote_code: bool,
        attention_backend: str | None,
    ):
        llm_kwargs: dict[str, Any] = {
            "model": model_name_or_path,
            "tokenizer": tokenizer_name_or_path,
            "revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_logprobs": topk,
            "logprobs_mode": "raw_logprobs",
            "max_model_len": max_model_len,
            "dtype": dtype,
            "seed": seed,
            "enforce_eager": enforce_eager,
            "enable_prefix_caching": enable_prefix_caching,
            "distributed_executor_backend": distributed_executor_backend,
        }
        if attention_backend is not None:
            llm_kwargs["attention_backend"] = attention_backend
        self.llm = vllm.LLM(**llm_kwargs)
        self.sampling_params = build_teacher_sampling_params(topk)
        self.topk = topk

    def score(self, queries: list[list[int]], responses: list[list[int]]) -> OPDTeacherScoringResult:
        if len(queries) != len(responses):
            raise ValueError(f"queries length {len(queries)} != responses length {len(responses)}")

        start_time = time.perf_counter()
        prompts = [
            vllm.TokensPrompt(prompt_token_ids=list(query) + list(response))
            for query, response in zip(queries, responses)
        ]
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)

        teacher_topk_token_ids = []
        teacher_topk_logprobs = []
        for i, (req_out, query, response) in enumerate(zip(outputs, queries, responses)):
            if req_out.prompt_logprobs is None:
                raise ValueError(f"Teacher vLLM output {i} is missing prompt_logprobs")
            token_ids, logprobs = extract_response_topk_from_prompt_logprobs(
                req_out.prompt_logprobs, prompt_len=len(query), response_len=len(response), k=self.topk
            )
            teacher_topk_token_ids.append(token_ids)
            teacher_topk_logprobs.append(logprobs)

        elapsed = time.perf_counter() - start_time
        total_response_tokens = sum(len(response) for response in responses)
        metrics = {
            "time/opd_teacher_scoring": elapsed,
            "opd/teacher_tokens_per_second": total_response_tokens / elapsed if elapsed > 0 else 0.0,
        }
        return OPDTeacherScoringResult(
            teacher_topk_token_ids=teacher_topk_token_ids, teacher_topk_logprobs=teacher_topk_logprobs, metrics=metrics
        )

    def ready(self) -> bool:
        return True


def create_teacher_scorers(
    *,
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    tokenizer_name_or_path: str,
    tokenizer_revision: str | None,
    model_name_or_path: str,
    model_revision: str | None,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
    topk: int,
    dtype: str,
    trust_remote_code: bool,
    attention_backend: str | None,
    pg: PlacementGroup | None = None,
) -> list[ray.actor.ActorHandle]:
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "mp"
    use_existing_pg = pg is not None
    num_gpus = tensor_parallel_size

    if pg is None:
        bundles = [{"GPU": tensor_parallel_size, "CPU": tensor_parallel_size} for _ in range(num_engines)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    bundle_indices_list = vllm_utils.get_bundle_indices_list(pg)
    teacher_scorers = []
    for i in range(num_engines):
        bundle_indices = (
            bundle_indices_list[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]
            if use_existing_pg
            else [bundle_indices_list[i]]
        )
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0],
        )
        teacher_scorers.append(
            ray.remote(OPDTeacherScorerRayActor)
            .options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env=ray.runtime_env.RuntimeEnv(
                    env_vars={
                        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                        "TORCH_CUDA_ARCH_LIST": vllm_utils.get_cuda_arch_list(),
                        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                    }
                ),
            )
            .remote(
                model_name_or_path=model_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                dtype=dtype,
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                topk=topk,
                distributed_executor_backend=distributed_executor_backend,
                trust_remote_code=trust_remote_code,
                attention_backend=attention_backend,
            )
        )

    utils.ray_get_with_progress(
        [scorer.ready.remote() for scorer in teacher_scorers], "Initializing OPD teacher scorers"
    )
    logger.info("Initialized %d OPD teacher scorer(s)", len(teacher_scorers))
    return teacher_scorers
