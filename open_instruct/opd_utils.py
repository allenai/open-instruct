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
from ray.util.placement_group import PlacementGroup

from open_instruct import logger_utils, vllm_utils
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
        self.vocab_size = self.llm.llm_engine.model_config.get_vocab_size()

    def score(self, queries: list[list[int]], responses: list[list[int]]) -> OPDTeacherScoringResult:
        if len(queries) != len(responses):
            raise ValueError(f"queries length {len(queries)} != responses length {len(responses)}")

        start_time = time.perf_counter()
        prompts = []
        for i, (query, response) in enumerate(zip(queries, responses)):
            prompt_token_ids = list(query) + list(response)
            bad_token_id = next((t for t in prompt_token_ids if t < 0 or t >= self.vocab_size), None)
            if bad_token_id is not None:
                raise ValueError(
                    f"Request {i} contains token id {bad_token_id}, outside the teacher vocab "
                    f"(size {self.vocab_size}). Student rollouts may only be scored by a teacher whose "
                    "vocab covers every sampled token; a student-only added token (e.g. a pad or chat "
                    "special token) likely reached the teacher."
                )
            prompts.append(vllm.TokensPrompt(prompt_token_ids=prompt_token_ids))
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
    distributed_executor_backend = vllm_utils.get_vllm_distributed_executor_backend(tensor_parallel_size)

    def remote_kwargs(engine_idx: int, bundle_indices: list[int]) -> dict[str, Any]:
        return dict(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            dtype=dtype,
            seed=seed + engine_idx,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            topk=topk,
            distributed_executor_backend=distributed_executor_backend,
            trust_remote_code=trust_remote_code,
            attention_backend=attention_backend,
        )

    teacher_scorers = vllm_utils.create_vllm_ray_actors(
        OPDTeacherScorerRayActor,
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        num_gpus=tensor_parallel_size,
        pg=pg,
        remote_kwargs_fn=remote_kwargs,
        ready_message="Initializing OPD teacher scorers",
    )
    logger.info("Initialized %d OPD teacher scorer(s)", len(teacher_scorers))
    return teacher_scorers
