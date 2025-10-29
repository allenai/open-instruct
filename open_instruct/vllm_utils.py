# Taken and modified from https://github.com/huggingface/trl
# Copyright 2024 The AllenAI Team. All rights reserved.
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

"""This file is copied from https://github.com/OpenRLHF/OpenRLHF"""

import asyncio
import os
import queue
import sys
import threading
import time
import types
from collections import defaultdict
from collections.abc import Awaitable
from concurrent import futures
from datetime import timedelta
from typing import Any

import ray
import torch
import torch.distributed
import vllm
from ray.util import queue as ray_queue
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    ProcessGroup,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.v1.core import kv_cache_utils

from open_instruct import logger_utils
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.tool_utils.tools import MaxCallsExceededTool, Tool
from open_instruct.utils import ModelDims, ray_get_with_progress

logger = logger_utils.setup_logger(__name__)

NUM_PREFETCH_WORKERS = 2
NUM_TOOL_WORKERS = 20
DRAIN_ACTIVE_TASKS_SLEEP_S = 1
SHOULD_STOP_TIMEOUT_S = 0.1


def assert_threaded_actor(instance):
    """Assert that an instance's class is suitable for use in a threaded (non-async) Ray actor.

    This function performs two checks:
      1. The class must not define any `async def` methods
         (including async generators, staticmethods, or classmethods).
      2. There must not be a running asyncio event loop in the current thread.

    Args:
        instance: The instance whose class to inspect.

    Raises:
        AssertionError: If the class defines one or more async methods, or a running asyncio event loop is detected.
    """
    try:
        loop = asyncio.get_running_loop()
        raise AssertionError(
            f"{instance.__class__.__name__} must run in a threaded Ray actor (no running event loop). "
            f"Detected RUNNING loop={loop!r} on thread='{threading.current_thread().name}'. "
            f"Python={sys.version.split()[0]}."
        )
    except RuntimeError:
        return


def _truncate_tool_output_tokens(
    tool_output_token_ids: list[int],
    current_prompt_token_ids: list[int],
    accumulated_tokens: list[int],
    max_model_len: int,
    max_tokens: int,
    current_mask_len: int,
) -> tuple[list[int], int, list[int]]:
    prompt_and_tool_output = current_prompt_token_ids + accumulated_tokens + tool_output_token_ids
    excess = len(prompt_and_tool_output) - max_model_len
    if excess > 0:
        tool_output_token_ids = tool_output_token_ids[:-excess]

    remaining = max_tokens - current_mask_len
    if remaining <= 0:
        return [], excess, prompt_and_tool_output
    elif len(tool_output_token_ids) > remaining:
        return tool_output_token_ids[:remaining], excess, prompt_and_tool_output

    return tool_output_token_ids, excess, prompt_and_tool_output


async def process_request_async(
    actor: "LLMRayActor",
    sub_request_id: str,
    base_request_id: str,
    prompt: vllm.TokensPrompt,
    sampling_params: vllm.SamplingParams,
):
    """Process a single async request with tool support, awaiting tools inline."""
    accumulated_tokens = []
    accumulated_logprobs = []
    masks = []
    num_calls = 0
    timeout = False
    tool_error = ""
    tool_output = ""
    tool_runtime = 0.0
    tool_called = False

    current_prompt = prompt
    current_prompt_token_ids = actor.request_metadata[base_request_id]["prompt_token_ids"]
    current_sampling_params = sampling_params.clone()
    final_prompt_token_ids = None
    iteration = 0

    while True:
        iteration_request_id = f"{sub_request_id}_iter{iteration}"
        outputs = [
            o
            async for o in actor.llm_engine.generate(current_prompt, current_sampling_params, iteration_request_id)
            if o.finished
        ]
        assert len(outputs) == 1, f"Expected exactly 1 output, got {len(outputs)} for request {iteration_request_id}"
        request_output = outputs[0]
        iteration += 1
        output = request_output.outputs[0]

        if final_prompt_token_ids is None:
            final_prompt_token_ids = request_output.prompt_token_ids

        accumulated_tokens.extend(output.token_ids)
        accumulated_logprobs.extend(output.logprobs)
        masks.extend([1] * len(output.token_ids))

        if not actor.tools or not actor.max_tool_calls:
            break

        triggered_tool, stop_str = get_triggered_tool(
            output.text, actor.tools, actor.max_tool_calls, num_calls, sampling_params
        )
        if triggered_tool is None:
            break

        assert actor.executor is not None, f"executor is None for request {sub_request_id}"

        loop = asyncio.get_running_loop()
        tool_result = await loop.run_in_executor(actor.executor, triggered_tool, output.text)

        tool_called = True
        num_calls += 1
        timeout = timeout or tool_result.timeout
        tool_error += "" if tool_result.error is None else tool_result.error
        tool_output += tool_result.output
        tool_runtime += tool_result.runtime

        tool_output_token_ids = actor.llm_engine.tokenizer.encode(
            "<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False
        )

        tool_output_token_ids, excess, prompt_and_tool_output = _truncate_tool_output_tokens(
            tool_output_token_ids,
            current_prompt_token_ids,
            accumulated_tokens,
            actor.llm_engine.model_config.max_model_len,
            sampling_params.max_tokens,
            len(masks),
        )

        accumulated_tokens.extend(tool_output_token_ids)
        accumulated_logprobs.extend(
            [{token_id: types.SimpleNamespace(logprob=0.0)} for token_id in tool_output_token_ids]
        )
        masks.extend([0] * len(tool_output_token_ids))

        new_sample_tokens = sampling_params.max_tokens - len(masks)
        if excess > 0 or new_sample_tokens <= 0:
            break

        current_prompt = vllm.TokensPrompt(prompt_token_ids=prompt_and_tool_output, cache_salt=base_request_id)
        current_prompt_token_ids = prompt_and_tool_output
        final_prompt_token_ids = prompt_and_tool_output
        current_sampling_params = sampling_params.clone()
        current_sampling_params.max_tokens = new_sample_tokens

    complete_output = vllm.CompletionOutput(
        index=split_request_id(sub_request_id)["request_index"],
        text="",
        token_ids=accumulated_tokens,
        cumulative_logprob=output.cumulative_logprob,
        logprobs=accumulated_logprobs,
        finish_reason=output.finish_reason,
        stop_reason=output.stop_reason,
    )

    if actor.tools:
        complete_output.mask = masks
        complete_output.num_calls = num_calls
        complete_output.timeout = timeout
        complete_output.tool_error = tool_error
        complete_output.tool_output = tool_output
        complete_output.tool_runtime = tool_runtime
        complete_output.tool_called = tool_called

    actor.active_tasks.pop(sub_request_id, None)

    actor.completion_queue.put(
        {
            "base_request_id": base_request_id,
            "expected_n": actor.request_metadata[base_request_id]["original_sampling_params"].n,
            "request_output": vllm.RequestOutput(
                request_id=sub_request_id,
                prompt=request_output.prompt,
                prompt_token_ids=final_prompt_token_ids,
                prompt_logprobs=request_output.prompt_logprobs,
                outputs=[complete_output],
                finished=True,
            ),
            "tools": actor.tools,
        }
    )


# Edited from: https://github.com/OpenRLHF/OpenRLHF/pull/971/files
# Turns out Ray doesnt necessarily place bundles together,
# so this function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
# avoids unnecessary communication for TP>1 with vllm.
def get_bundle_indices_list(placement_group: ray.util.placement_group) -> list[int]:
    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = defaultdict(list)
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles[node_id].append(bundle)

    flattened_bundle_indices = []
    for node_id, bundles in node_id_to_bundles.items():
        flattened_bundle_indices.extend(bundles)
    return flattened_bundle_indices


def make_request_id(request: PromptRequest) -> str:
    """Generate a unique tracking key for a request."""
    prefix = "eval" if request.is_eval else "train"
    return f"{prefix}_{request.epoch_number}_{request.training_step}_{request.dataset_index}"


def split_request_id(full_request_id: str) -> dict:
    """Split request ID into base ID and request index.

    >>> split_request_id("train_0_1_43039_0")
    {'base_id': 'train_0_1_43039', 'request_index': 0}
    >>> split_request_id("eval_0_5_12345_2")
    {'base_id': 'eval_0_5_12345', 'request_index': 2}
    """
    parts = full_request_id.split("_")
    return {"base_id": "_".join(parts[:-1]), "request_index": int(parts[-1])}


def get_triggered_tool(
    output_text: str,
    tools: dict[str, Tool],
    max_tool_calls: dict[str, int],
    num_calls: int,
    sampling_params: vllm.SamplingParams,
) -> tuple[Tool | None, str | None]:
    """Check if any tool was triggered and return the tool and stop_str if found.

    Args:
        output_text: The generated text to check for tool triggers
        tools: Dictionary mapping stop strings to Tool instances
        max_tool_calls: Dictionary mapping stop strings to their call limits
        num_calls: Current number of tool calls for this request
        sampling_params: Sampling parameters containing stop strings

    Returns:
        Tuple of (tool, stop_str) if a tool was triggered, (None, None) otherwise.
    """
    for stop_str in sampling_params.stop:
        if stop_str in tools and output_text.endswith(stop_str):
            if num_calls < max_tool_calls.get(stop_str, 0):
                return tools[stop_str], stop_str
            else:
                return MaxCallsExceededTool(start_str="<tool>", end_str="</tool>"), stop_str
    return None, None


def process_completed_request(request_id, outs, current_time, tools, request_metadata):
    """Process a completed request with all its samples and return the result.

    Args:
        request_id: The base request ID
        outs: List of vllm.RequestOutput objects for all sub-requests
        current_time: Current timestamp for performance metrics
        tools: Dictionary of available tools (may be None or empty)
        request_metadata: Dictionary containing metadata for all requests

    Returns:
        Tuple of (result, is_eval) where result is a GenerationResult and is_eval is a boolean
    """
    final_output = vllm.RequestOutput(
        request_id=request_id,
        prompt=outs[0].prompt,
        prompt_token_ids=outs[0].prompt_token_ids,
        prompt_logprobs=outs[0].prompt_logprobs,
        outputs=[completion for out in outs for completion in out.outputs],
        finished=outs[0].finished,
    )

    total_generation_tokens = sum(len(completion.token_ids) for out in outs for completion in out.outputs)
    metadata = request_metadata[request_id]  # Don't pop yet, _poll_tool_futures might need it

    # Process the vLLM RequestOutput into GenerationResult format
    response_ids = [list(out.token_ids) for out in final_output.outputs]
    finish_reasons = [out.finish_reason for out in final_output.outputs]
    use_tools = bool(tools)

    logprobs = []
    for idx, out in enumerate(final_output.outputs):
        assert len(out.token_ids) == len(out.logprobs), (
            f"vLLM CompletionOutput {idx}: token_ids length ({len(out.token_ids)}) "
            f"!= logprobs length ({len(out.logprobs)})"
        )
        logprobs.append(
            [logprob_dict[token_id].logprob for token_id, logprob_dict in zip(out.token_ids, out.logprobs)]
        )

    # Extract attributes based on whether tools are used
    if use_tools:
        # Extract tool-specific attributes from outputs
        masks = [getattr(out, "mask", [1] * len(out.token_ids)) for out in final_output.outputs]
        num_calls = [getattr(out, "num_calls", 0) for out in final_output.outputs]
        timeouts = [getattr(out, "timeout", False) for out in final_output.outputs]
        tool_errors = [getattr(out, "tool_error", "") for out in final_output.outputs]
        tool_outputs = [getattr(out, "tool_output", "") for out in final_output.outputs]
        tool_runtimes = [getattr(out, "tool_runtime", 0.0) for out in final_output.outputs]
        tool_calleds = [getattr(out, "tool_called", False) for out in final_output.outputs]
    else:
        # Use default values when tools are not used
        masks = [[1] * len(resp) for resp in response_ids]
        num_calls = [0] * len(response_ids)
        timeouts = [False] * len(response_ids)
        tool_errors = [""] * len(response_ids)
        tool_outputs = [""] * len(response_ids)
        tool_runtimes = [0.0] * len(response_ids)
        tool_calleds = [False] * len(response_ids)

    result = GenerationResult(
        responses=response_ids,
        finish_reasons=finish_reasons,
        masks=masks,
        request_info=RequestInfo(
            num_calls=num_calls,
            timeouts=timeouts,
            tool_errors=tool_errors,
            tool_outputs=tool_outputs,
            tool_runtimes=tool_runtimes,
            tool_calleds=tool_calleds,
        ),
        dataset_index=metadata["dataset_index"],
        epoch_number=metadata["epoch_number"],
        token_statistics=TokenStatistics(
            num_prompt_tokens=len(metadata["prompt_token_ids"]),
            num_response_tokens=total_generation_tokens,
            generation_time=current_time - metadata["start_time"],
        ),
        start_time=metadata["start_time"],
        logprobs=logprobs,
    )
    return result, metadata["is_eval"]


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str | None = None,
    pg_options: Any | None = None,
) -> ProcessGroup:
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend) if backend else Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def _prefetch_worker(actor: "LLMRayActor") -> None:
    while True:
        if actor._should_stop() or len(actor.active_tasks) >= actor.inference_batch_size:
            time.sleep(DRAIN_ACTIVE_TASKS_SLEEP_S)
            continue

        request = actor.prompt_queue.get()
        add_request(actor, request)


def add_request(actor: "LLMRayActor", request: PromptRequest) -> None:
    request_id = make_request_id(request)

    sampling_params = request.generation_config.clone()
    sampling_params.n = 1  # Use n=1 for tool processing

    actor.request_metadata[request_id] = {
        "is_eval": request.is_eval,
        "dataset_index": request.dataset_index,
        "epoch_number": request.epoch_number,
        "training_step": request.training_step,
        "sampling_params": sampling_params,
        "original_sampling_params": request.generation_config,
        "prompt_token_ids": list(request.prompt),
        "start_time": time.perf_counter(),
    }

    tokens_prompt = vllm.TokensPrompt(prompt_token_ids=request.prompt, cache_salt=request_id)

    for j in range(request.generation_config.n):
        sub_sampling_params = sampling_params.clone()
        if request.generation_config.seed is not None:
            sub_sampling_params.seed = request.generation_config.seed + j
        sub_request_id = f"{request_id}_{j}"
        actor.active_tasks[sub_request_id] = asyncio.run_coroutine_threadsafe(
            process_request_async(actor, sub_request_id, request_id, tokens_prompt, sub_sampling_params), actor.loop
        )


class LLMRayActor:
    """Ray actor for LLM generation with optional tool support."""

    def __init__(
        self,
        *args,
        tools: dict[str, Tool] | None = None,
        max_tool_calls: dict[str, int] | None = None,
        bundle_indices: list[int] | None = None,
        prompt_queue: ray_queue.Queue,
        results_queue: ray_queue.Queue,
        eval_results_queue: ray_queue.Queue,
        actor_manager: ray.actor.ActorHandle,
        inference_batch_size: int | None,
        inflight_updates: bool,
        **kwargs,
    ):
        assert_threaded_actor(self)
        self._init_config(tools, max_tool_calls, inference_batch_size, inflight_updates)
        self._init_queues(prompt_queue, results_queue, eval_results_queue, actor_manager)
        self._init_executor()

        noset_visible_devices = kwargs.pop("noset_visible_devices")
        distributed_executor_backend = kwargs.get("distributed_executor_backend")
        self._setup_gpu_visibility(noset_visible_devices, distributed_executor_backend)
        self._setup_and_start_async_engine(args, bundle_indices, kwargs)

    def _init_config(
        self,
        tools: dict[str, Tool] | None,
        max_tool_calls: dict[str, int] | None,
        inference_batch_size: int | None,
        inflight_updates: bool,
    ) -> None:
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.inference_batch_size = inference_batch_size
        self.inflight_updates = inflight_updates
        self.request_metadata = {}
        self.active_tasks = {}
        self.request_outputs = {}

    def _init_queues(self, prompt_queue, results_queue, eval_results_queue, actor_manager) -> None:
        self.completion_queue = queue.Queue()
        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.actor_manager = actor_manager

        # For caching should_stop status.
        self._last_should_stop_update = float("-inf")
        self._should_stop_value = False

    def _init_executor(self) -> None:
        max_workers = NUM_PREFETCH_WORKERS + (NUM_TOOL_WORKERS if self.tools else 0)
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._prefetch_future = self.executor.submit(_prefetch_worker, self)
        self._process_future = self.executor.submit(self.process_from_queue)

    def _setup_gpu_visibility(self, noset_visible_devices: bool, distributed_executor_backend: str) -> None:
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        if distributed_executor_backend == "ray":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    def _setup_and_start_async_engine(self, args, bundle_indices, kwargs) -> None:
        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            logger.debug(f"creating LLM with bundle_indices={bundle_indices}")

        engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        engine_args.disable_log_stats = True
        engine_args.disable_cascade_attn = True

        init_complete = threading.Event()
        self.loop = None
        self.llm_engine = None

        async def _init_engine():
            running_loop = asyncio.get_running_loop()
            assert running_loop == self.loop, f"Loop mismatch! running={running_loop}, actor.loop={self.loop}"
            return vllm.AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=False)

        def _run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.llm_engine = self.loop.run_until_complete(_init_engine())
            init_complete.set()
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=_run_loop, daemon=True)
        self.loop_thread.start()
        init_complete.wait()

    def get_model_dims(self):
        """Get only the model dimensions without loading weights."""
        return ModelDims.from_vllm_config(self.llm_engine.vllm_config)

    def _should_stop(self) -> bool:
        if (time.perf_counter() - self._last_should_stop_update) > SHOULD_STOP_TIMEOUT_S:
            should_stop_ref = self.actor_manager.should_stop.remote()
            ready_refs, _ = ray.wait([should_stop_ref], timeout=SHOULD_STOP_TIMEOUT_S)
            if ready_refs:
                self._should_stop_value = ray.get(ready_refs[0])
                self._last_should_stop_update = time.perf_counter()
            else:
                ray.cancel(should_stop_ref)
        return self._should_stop_value

    def _accumulate_sub_request(self, sub_request: dict) -> None:
        base_request_id = sub_request["base_request_id"]
        expected_n = sub_request["expected_n"]

        if base_request_id not in self.request_outputs:
            self.request_outputs[base_request_id] = {
                "outputs": [],
                "expected_n": expected_n,
                "tools": sub_request["tools"],
            }

        self.request_outputs[base_request_id]["outputs"].append(sub_request["request_output"])

        is_complete = len(self.request_outputs[base_request_id]["outputs"]) == expected_n
        if is_complete:
            self._finalize_completed_request(base_request_id)

    def _finalize_completed_request(self, base_request_id: str) -> None:
        outputs = self.request_outputs[base_request_id]["outputs"]
        ordered_outs = sorted(outputs, key=lambda x: split_request_id(x.request_id)["request_index"])

        current_time = time.perf_counter()
        result, is_eval = process_completed_request(
            base_request_id,
            ordered_outs,
            current_time,
            self.request_outputs[base_request_id]["tools"],
            self.request_metadata,
        )

        self.request_outputs.pop(base_request_id)
        self.request_metadata.pop(base_request_id, None)

        results_queue = self.eval_results_queue if is_eval else self.results_queue
        results_queue.put(result)

    def process_from_queue(self) -> None:
        while True:
            sub_request = self.completion_queue.get()
            self._accumulate_sub_request(sub_request)

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
        use_ray: bool = False,
        timeout_minutes: int = 120,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self.llm_engine.collective_rpc(
                "init_process_group",
                args=(
                    master_address,
                    master_port,
                    rank_offset,
                    world_size,
                    group_name,
                    backend,
                    use_ray,
                    timeout_minutes,
                ),
            ),
            self.loop,
        )
        return future.result(timeout=timeout_minutes * 60)

    def _run_async(self, coro: Awaitable[Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def _prepare_weight_update(self, name: str, dtype: str) -> None:
        # Wait for all active requests to complete.
        while not self.inflight_updates and len(self.active_tasks) > 0:
            self.check_background_threads()
            time.sleep(DRAIN_ACTIVE_TASKS_SLEEP_S)

        expected_dtype = str(self.llm_engine.model_config.dtype)
        assert dtype == expected_dtype, f"Mismatched dtype for {name}: received {dtype!r}, expected {expected_dtype!r}"

    def update_weight(self, name: str, dtype: str, shape: tuple[int, ...], empty_cache: bool = False) -> None:
        self._prepare_weight_update(name, dtype)
        return self._run_async(self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache)))

    def update_weight_cuda_ipc(
        self, name: str, dtype: str, shape: tuple[int, ...], ipc_handles: list[Any], empty_cache: bool = False
    ) -> None:
        self._prepare_weight_update(name, dtype)
        return self._run_async(
            self.llm_engine.collective_rpc(
                "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
            )
        )

    def reset_prefix_cache(self) -> None:
        return self._run_async(self.llm_engine.reset_prefix_cache())

    def ready(self) -> bool:
        return True

    def check_background_threads(self) -> None:
        if self._prefetch_future.done():
            self._prefetch_future.result()
        if self._process_future.done():
            self._process_future.result()
        active_tasks = list(self.active_tasks.items())
        for task_id, task in active_tasks:
            if task.done():
                task.result()
        if not self.loop_thread.is_alive():
            raise RuntimeError(
                "vLLM engine loop thread has died. Check logs for errors in EngineCore or async engine."
            )

    def get_kv_cache_info(self) -> int:
        """Get KV cache max concurrency from the vLLM engine."""
        kv_cache_specs = self._run_async(self.llm_engine.collective_rpc("get_kv_cache_spec"))

        vllm_config = self.llm_engine.vllm_config
        gpu_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = int(gpu_memory_utilization * total_gpu_memory)

        kv_cache_groups = kv_cache_utils.get_kv_cache_groups(vllm_config, kv_cache_specs[0])

        kv_cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, kv_cache_specs[0], available_memory
        )

        max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)

        return int(max_concurrency)


def get_cuda_arch_list() -> str:
    """Get CUDA compute capabilities and format them for TORCH_CUDA_ARCH_LIST."""
    if not torch.cuda.is_available():
        return ""

    cuda_capabilities = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_capabilities.append(f"{major}.{minor}")

    # Remove duplicates and sort
    cuda_capabilities = sorted(set(cuda_capabilities))
    cuda_arch_list = ";".join(cuda_capabilities)
    logger.info(
        f"Detected CUDA compute capabilities: {cuda_capabilities}, setting TORCH_CUDA_ARCH_LIST={cuda_arch_list}"
    )
    return cuda_arch_list


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    tokenizer_name_or_path: str,
    pretrain: str,
    revision: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
    vllm_gpu_memory_utilization: float = 0.9,
    single_gpu_mode: bool = False,
    pg: PlacementGroup | None = None,
    tools: dict[str, Tool] | None = None,
    max_tool_calls: list[int] = [5],
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
    inference_batch_size: int | None = None,
    use_fp8_kv_cache=False,
    inflight_updates: bool = False,
) -> list[LLMRayActor]:
    # Convert max_tool_calls to a dict mapping tool end strings to their limits
    if tools:
        assert len(max_tool_calls) == 1 or len(max_tool_calls) == len(tools), (
            "max_tool_calls must have length 1 (applies to all tools) or same length as tools (per-tool limit)"
        )
        # tool key is the end_str
        if len(max_tool_calls) == 1:
            max_tool_calls_dict = {end_str: max_tool_calls[0] for end_str in tools}
        else:
            max_tool_calls_dict = {
                end_str: limit for end_str, limit in zip(tools.keys(), max_tool_calls, strict=False)
            }
    else:
        max_tool_calls_dict = {}

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1 and single_gpu_mode:
        # every worker will use 0.5 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.5

    logger.info(f"num_gpus: {num_gpus}")

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # ensure we use bundles on the same node where possible if tp>1.
    bundle_indices_list = get_bundle_indices_list(pg)

    for i in range(num_engines):
        bundle_indices = None
        bundle_indices = bundle_indices_list[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0],
        )

        vllm_engines.append(
            ray.remote(LLMRayActor)
            .options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env=ray.runtime_env.RuntimeEnv(
                    env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TORCH_CUDA_ARCH_LIST": get_cuda_arch_list()}
                ),
            )
            .remote(
                model=pretrain,
                revision=revision,
                tokenizer=tokenizer_name_or_path,
                tokenizer_revision=revision,
                worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                dtype="bfloat16",
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                noset_visible_devices=ray_noset_visible_devices(),
                prompt_queue=prompt_queue,
                results_queue=results_queue,
                eval_results_queue=eval_results_queue,
                actor_manager=actor_manager,
                tools=tools,
                max_tool_calls=max_tool_calls_dict,
                inference_batch_size=inference_batch_size,
                inflight_updates=inflight_updates,
                kv_cache_dtype="auto" if not use_fp8_kv_cache else "fp8",
                calculate_kv_scales=use_fp8_kv_cache,
            )
        )

    ray_get_with_progress([engine.ready.remote() for engine in vllm_engines], "Initializing vLLM engines", timeout=300)

    return vllm_engines
