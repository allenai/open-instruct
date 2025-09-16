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
import dataclasses
import os
import time
from collections import defaultdict
from concurrent import futures
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import torch
import torch.distributed
import vllm
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.v1 import kv_cache_interface
from vllm.v1.core import kv_cache_utils

from open_instruct import logger_utils
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.tool_utils.tool_vllm import MaxCallsExceededTool, Tool
from open_instruct.utils import ray_get_with_progress

logger = logger_utils.setup_logger(__name__)


# Edited from: https://github.com/OpenRLHF/OpenRLHF/pull/971/files
# Turns out Ray doesnt necessarily place bundles together,
# so this function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
# avoids unnecessary communication for TP>1 with vllm.
def get_bundle_indices_list(placement_group: ray.util.placement_group) -> List[int]:
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
    return f"{prefix}_{request.training_step}_{request.dataset_index}"


def _extract_base_request_id(full_request_id: str) -> str:
    """Extract base request ID by removing the sample suffix.

    >>> _extract_base_request_id("train_1_43039_0")
    'train_1_43039'
    >>> _extract_base_request_id("eval_5_12345_2")
    'eval_5_12345'
    """
    return "_".join(full_request_id.split("_")[:-1])


def get_triggered_tool(
    output_text: str,
    tools: Dict[str, Tool],
    max_tool_calls: Dict[str, int],
    num_calls: int,
    sampling_params: vllm.SamplingParams,
) -> Optional[Tuple[Tool, str]]:
    """Check if any tool was triggered and return the tool and stop_str if found.

    Returns:
        Tuple of (tool, stop_str) if a tool was triggered, None otherwise.
    """
    for stop_str in sampling_params.stop:
        if stop_str in tools and output_text.endswith(stop_str):
            # Determine which tool to use
            if num_calls < max_tool_calls.get(stop_str, 0):
                return tools[stop_str], stop_str
            else:
                return MaxCallsExceededTool(start_str="<tool>", end_str="</tool>"), stop_str
    return None


def process_completed_request(request_id, outs, tracking, current_time, tools, request_metadata):
    """Process a completed request with all its samples and return the result.

    Args:
        request_id: The base request ID
        outs: List of vllm.RequestOutput objects for all sub-requests
        tracking: Dictionary containing tool tracking information
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
        training_step=metadata["training_step"],
        token_statistics=TokenStatistics(
            num_prompt_tokens=metadata["prompt_tokens"],
            num_response_tokens=total_generation_tokens,
            generation_time=current_time - metadata["start_time"],
        ),
        start_time=metadata["start_time"],
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
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

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


class LLMRayActor:
    """Ray actor for LLM generation with optional tool support."""

    def __init__(
        self,
        *args,
        tools: Optional[Dict[str, Tool]] = None,
        max_tool_calls: Optional[Dict[str, int]] = None,
        bundle_indices: list = None,
        prompt_queue=None,
        results_queue=None,
        eval_results_queue=None,
        actor_manager=None,
        inference_batch_size: Optional[int] = None,
        inflight_updates: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        self.logger = logger_utils.setup_logger(__name__)
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.inference_batch_size = inference_batch_size
        self.inflight_updates = inflight_updates
        self.verbose = verbose
        self.request_metadata = {}

        if self.tools:
            self.executor = futures.ThreadPoolExecutor(max_workers=20)
        else:
            self.executor = None

        noset_visible_devices = kwargs.pop("noset_visible_devices")
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            if self.verbose:
                self.logger.info(f"creating LLM with bundle_indices={bundle_indices}")

        self.engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        # Log stats causes a crash in the engine at assert outputs.scheduler_stats is not None when we call step() and there is nothing to step.
        self.engine_args.disable_log_stats = True

        # Cascade attention has known performance issues: https://github.com/vllm-project/vllm/issues/17652
        self.engine_args.disable_cascade_attn = True

        self.llm_engine = None

        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.actor_manager = actor_manager

        # For caching should_stop status.
        self._last_should_stop_update = float("-inf")
        self._should_stop_value = False
        self._should_stop_timeout_s = 5

        # Async tracking
        self.active_tasks = {}  # Track active async tasks
        self.request_outputs = {}
        self.request_outputs_lock = asyncio.Lock()  # Lock for thread-safe access

        # Create prefetch task
        self.prefetch_task = asyncio.create_task(self._prefetch_requests())

    def _should_stop(self) -> bool:
        if (time.perf_counter() - self._last_should_stop_update) > self._should_stop_timeout_s:
            should_stop_ref = self.actor_manager.should_stop.remote()
            ready_refs, _ = ray.wait([should_stop_ref], timeout=0.1)
            if ready_refs:
                self._should_stop_value = ray.get(ready_refs[0])
                self._last_should_stop_update = time.perf_counter()
            else:
                ray.cancel(should_stop_ref)
        return self._should_stop_value

    async def _prefetch_requests(self):
        """Prefetches requests from queue."""
        while True:
            # Check if we need more requests
            current_unfinished = len(self.active_tasks)
            if current_unfinished >= self.inference_batch_size:
                await asyncio.sleep(1)
                continue

            await self._add_request(await self.prompt_queue.get_async())

    async def _add_request(self, request: PromptRequest):
        """Add a request to the async LLM engine."""
        request_id = make_request_id(request)
        sampling_params = request.generation_config.clone()
        sampling_params.n = 1  # Use n=1 for tool processing
        self.request_metadata[request_id] = {
            "is_eval": request.is_eval,
            "dataset_index": request.dataset_index,
            "training_step": request.training_step,
            "sampling_params": sampling_params,
            "original_sampling_params": request.generation_config,
            "prompt_tokens": len(request.prompt),
            "start_time": time.perf_counter(),
        }

        tokens_prompt = vllm.TokensPrompt(prompt_token_ids=request.prompt, cache_salt=request_id)
        for j in range(request.generation_config.n):
            sub_sampling_params = sampling_params.clone()  # Already has n=1
            if request.generation_config.seed is not None:
                sub_sampling_params.seed = request.generation_config.seed + j
            sub_request_id = f"{request_id}_{j}"

            # Create a task to process this sub-request with its ID as the name
            task = asyncio.create_task(
                self._process_request(sub_request_id, request_id, tokens_prompt, sub_sampling_params, j),
                name=sub_request_id,
            )
            self.active_tasks[sub_request_id] = task

    def _insert_result_to_queue(self, result, is_eval: bool):
        """Insert result into the appropriate queue with blocking put."""
        results_queue = self.eval_results_queue if is_eval else self.results_queue
        results_queue.put(result)

    def _should_exit(self) -> bool:
        """Determine if the processing loop should exit.

        Returns:
            bool: True if the loop should exit, False otherwise.
        """
        # Check stop condition first
        stop_requested = self._should_stop()

        # Case 1: inflight_updates enabled and stop requested - exit immediately
        if self.inflight_updates and stop_requested:
            return True

        # Check for pending work
        active_tasks = len(self.active_tasks)

        # Case 2: stop requested and no pending work - exit
        if stop_requested and active_tasks == 0:
            return True

        # Case 3: no work left at all - exit
        if active_tasks == 0:
            return True

        # Otherwise, continue processing
        return False

    async def _ensure_engine_initialized(self):
        """Ensure the AsyncLLMEngine is initialized."""
        if self.llm_engine is None:
            self.llm_engine = vllm.AsyncLLMEngine.from_engine_args(self.engine_args, start_engine_loop=False)

    async def generate_one_completion(
        self, request_id: str, prompt: vllm.TokensPrompt, sampling_params: vllm.SamplingParams
    ) -> vllm.RequestOutput:
        """Generate a single completion from the async engine.

        Wraps the async generator to return a single RequestOutput.
        """
        generator = self.llm_engine.add_request(request_id, prompt, sampling_params)
        outputs = [output async for output in generator if output.finished]
        assert len(outputs) == 1, f"Expected exactly 1 output, got {len(outputs)} for request {request_id}"
        return outputs[0]

    async def _process_request(
        self,
        sub_request_id: str,
        base_request_id: str,
        prompt: vllm.TokensPrompt,
        sampling_params: vllm.SamplingParams,
        index: int,
    ):
        """Process a single sub-request from start to finish, including tool handling.

        Args:
            sub_request_id: The sub-request ID (e.g., "train_1_43039_2")
            base_request_id: The base request ID (e.g., "train_1_43039")
            prompt: The prompt tokens
            sampling_params: The sampling parameters
            index: The index of this sub-request (for n>1 sampling)
        """
        # Local tracking for this request
        request_output = None
        masks = []
        num_calls = 0
        timeout = False
        tool_error = ""
        tool_output_str = ""
        tool_runtime = 0.0
        tool_called = False

        current_prompt = prompt
        current_sampling_params = sampling_params

        while True:
            # Generate completion
            output = await self.generate_one_completion(sub_request_id, current_prompt, current_sampling_params)

            # Fix the index field
            assert len(output.outputs) == 1, f"{len(output.outputs)=}"
            output.outputs[0] = dataclasses.replace(output.outputs[0], index=index)

            # Initialize or extend request_output
            if request_output is None:
                request_output = output
            else:
                # Extend the token_ids in the existing CompletionOutput
                request_output.outputs[0].token_ids.extend(output.outputs[0].token_ids)

            masks.extend([1] * len(output.outputs[0].token_ids))

            # Check for tool calls - break early if no tools
            if not self.tools:
                break

            # Check if any tool was triggered
            tool_info = get_triggered_tool(
                output.outputs[0].text, self.tools, self.max_tool_calls, num_calls, current_sampling_params
            )
            if tool_info is None:
                break  # No tool triggered - request is complete

            tool, stop_str = tool_info

            tool_result = await asyncio.to_thread(tool, output.outputs[0].text)

            # Update tracking
            num_calls += 1
            timeout = tool_result.timeout
            tool_error += "" if tool_result.error is None else tool_result.error
            tool_output_str += tool_result.output
            tool_runtime += tool_result.runtime
            tool_called = True

            # Prepare tool output tokens
            tokenizer = self.llm_engine.engine.tokenizer
            tool_output_token_ids = tokenizer.encode(
                "<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False
            )

            # Check context length
            prompt_and_tool_output_token = (
                output.prompt_token_ids + request_output.outputs[0].token_ids + tool_output_token_ids
            )
            excess = len(prompt_and_tool_output_token) - self.llm_engine.model_config.max_model_len
            if excess > 0:
                tool_output_token_ids = tool_output_token_ids[:-excess]
                can_continue = False
            else:
                can_continue = True

            # Check max_tokens limit
            remaining = current_sampling_params.max_tokens - len(masks)
            if remaining <= 0:
                tool_output_token_ids = []
                can_continue = False
            elif len(tool_output_token_ids) > remaining:
                tool_output_token_ids = tool_output_token_ids[:remaining]
                can_continue = False

            # Add tool output to concatenated result
            request_output.outputs[0].token_ids.extend(tool_output_token_ids)
            masks.extend([0] * len(tool_output_token_ids))

            # Check if we can continue
            new_sample_tokens = current_sampling_params.max_tokens - len(masks)
            if not can_continue or new_sample_tokens <= 0:
                break

            # Prepare for next iteration
            current_prompt = vllm.TokensPrompt(prompt_token_ids=prompt_and_tool_output_token)
            current_sampling_params = current_sampling_params.clone()
            current_sampling_params.max_tokens = new_sample_tokens
            # Continue the while loop with new prompt

        # Attach tool metadata if tools are enabled
        complete_output = request_output.outputs[0]
        if self.tools:
            setattr(complete_output, "mask", masks)
            setattr(complete_output, "num_calls", num_calls)
            setattr(complete_output, "timeout", timeout)
            setattr(complete_output, "tool_error", tool_error)
            setattr(complete_output, "tool_output", tool_output_str)
            setattr(complete_output, "tool_runtime", tool_runtime)
            setattr(complete_output, "tool_called", tool_called)

        # Add to request_outputs with lock
        async with self.request_outputs_lock:
            if base_request_id not in self.request_outputs:
                self.request_outputs[base_request_id] = vllm.RequestOutput(
                    request_id=base_request_id,
                    prompt=request_output.prompt,
                    prompt_token_ids=request_output.prompt_token_ids,
                    prompt_logprobs=request_output.prompt_logprobs,
                    outputs=[],
                    finished=True,
                )

            self.request_outputs[base_request_id].outputs.append(complete_output)

    async def _check_and_process_completed_requests(self, base_request_ids: List[str]):
        """Check request_outputs for completed requests and process them.

        Args:
            base_request_ids: List of specific base request IDs to check.
        """
        if not base_request_ids:
            return 0

        processed_count = 0
        current_time = time.perf_counter()

        # Acquire lock once for entire operation
        async with self.request_outputs_lock:
            for base_request_id in base_request_ids:
                # Skip if not in outputs or metadata
                if base_request_id not in self.request_outputs or base_request_id not in self.request_metadata:
                    continue

                request_output = self.request_outputs[base_request_id]
                expected_n = self.request_metadata[base_request_id]["original_sampling_params"].n

                # Check if this request has all N outputs
                if len(request_output.outputs) != expected_n:
                    continue

                # Build ordered outputs
                ordered_outs = []
                for j in range(expected_n):
                    matching_output = None
                    for comp_output in request_output.outputs:
                        if hasattr(comp_output, "index") and comp_output.index == j:
                            matching_output = comp_output
                            break

                    ordered_outs.append(
                        vllm.RequestOutput(
                            request_id=f"{base_request_id}_{j}",
                            prompt=request_output.prompt,
                            prompt_token_ids=request_output.prompt_token_ids,
                            prompt_logprobs=request_output.prompt_logprobs,
                            outputs=[matching_output],
                            finished=True,
                        )
                    )

                # Remove from request_outputs
                self.request_outputs.pop(base_request_id)

                # Process and insert result (still within lock for consistency)
                result, is_eval = process_completed_request(
                    base_request_id, ordered_outs, {}, current_time, self.tools, self.request_metadata
                )
                self._insert_result_to_queue(result, is_eval=is_eval)

                # Clean up metadata
                self.request_metadata.pop(base_request_id, None)

                processed_count += 1

        return processed_count

    async def process_from_queue(self, timeout: float = 60.0):
        """Run generation loop using AsyncLLMEngine.

        Runs continuously until should_stop is set, periodically adding new requests
        and yielding control to allow weight synchronization.

        Returns:
            int: Number of requests processed
        """
        # Ensure engine is initialized
        await self._ensure_engine_initialized()

        # Start the AsyncLLMEngine background loop
        self.llm_engine.start_background_loop()

        iteration_count = 0
        total_processed = 0

        try:
            while not self._should_exit():
                iteration_count += 1

                # Health check for prefetch task
                if self.prefetch_task.done():
                    self.prefetch_task.result()  # This will raise if the task failed

                # Wait for any task to complete or timeout
                done, pending = await asyncio.wait(
                    self.active_tasks.values() if self.active_tasks else [],
                    timeout=timeout / 1000,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed tasks and collect their base request IDs
                completed_base_request_ids = set()
                for task in done:
                    await task  # Get result or raise exception

                    # Remove the completed task using its name
                    task_name = task.get_name()
                    if task_name in self.active_tasks:
                        self.active_tasks.pop(task_name, None)
                        # Extract base request ID from task name (e.g., "train_1_43039_2" -> "train_1_43039")
                        base_request_id = _extract_base_request_id(task_name)
                        completed_base_request_ids.add(base_request_id)

                # Only check the requests that just had tasks complete
                if completed_base_request_ids:
                    processed_count = await self._check_and_process_completed_requests(
                        list(completed_base_request_ids)
                    )
                    total_processed += processed_count

                if self.verbose and iteration_count % 100 == 0:
                    active_tasks = len(self.active_tasks)
                    self.logger.info(f"process_from_queue iteration {iteration_count}: active_tasks={active_tasks}")

        finally:
            # Wait for all active tasks to complete only if inflight_updates is False
            if not self.inflight_updates:
                await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

            # Process any remaining completed requests only if inflight_updates is False
            if not self.inflight_updates:
                # Check all requests by passing all base request IDs
                all_base_request_ids = list(self.request_outputs.keys())
                processed_count = await self._check_and_process_completed_requests(all_base_request_ids)
                total_processed += processed_count

            # Stop the AsyncLLMEngine background loop
            self.llm_engine.shutdown_background_loop()

            # Count total processed
            return total_processed

    async def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        use_ray=False,
        timeout_minutes=120,
    ):
        await self._ensure_engine_initialized()
        return await self.llm_engine.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray, timeout_minutes),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        await self._ensure_engine_initialized()
        return await self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        await self._ensure_engine_initialized()
        return await self.llm_engine.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    async def reset_prefix_cache(self):
        await self._ensure_engine_initialized()
        await self.llm_engine.reset_prefix_cache()

    async def sleep(self, level=1):
        await self._ensure_engine_initialized()
        await self.llm_engine.sleep(level=level)

    async def wake_up(self, tags: Optional[list[str]] = None):
        await self._ensure_engine_initialized()
        await self.llm_engine.wake_up(tags)

    async def ready(self):
        await self._ensure_engine_initialized()
        return True

    async def get_kv_cache_info(self):
        """Get KV cache max concurrency from the vLLM engine."""
        await self._ensure_engine_initialized()
        # AsyncLLMEngine wraps the underlying LLMEngine
        engine = self.llm_engine.engine
        # For UniProcExecutor, access through driver_worker
        kv_cache_specs = engine.model_executor.driver_worker.get_kv_cache_specs()
        kv_cache_spec = kv_cache_specs[0]
        # Group layers by their attention type (type_id) to handle models
        # with sliding attention in some layers but not others
        type_groups = defaultdict(list)
        for layer_name, layer_spec in kv_cache_spec.items():
            type_groups[layer_spec.type_id].append(layer_name)

        grouped_layer_names = list(type_groups.values())

        page_size = kv_cache_utils.get_uniform_page_size(kv_cache_spec)

        vllm_config = engine.vllm_config
        gpu_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = int(gpu_memory_utilization * total_gpu_memory)

        num_blocks = kv_cache_utils.get_num_blocks(vllm_config, len(kv_cache_spec), available_memory, page_size)

        per_layer_size = page_size * num_blocks
        kv_cache_tensors = [
            kv_cache_interface.KVCacheTensor(size=per_layer_size, shared_by=[layer_name])
            for layer_name in kv_cache_spec
        ]

        kv_cache_config = kv_cache_interface.KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_utils.create_kv_cache_group_specs(kv_cache_spec, grouped_layer_names),
        )
        max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config(engine.vllm_config, kv_cache_config)

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
    pg: Optional[ray.util.placement_group] = None,
    vllm_enable_sleep=False,
    tools: Optional[Dict[str, Tool]] = None,
    max_tool_calls: List[int] = [5],
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
    inference_batch_size: Optional[int] = None,
    use_fp8_kv_cache=False,
    inflight_updates: bool = False,
    verbose: bool = False,
) -> list[LLMRayActor]:
    # Convert max_tool_calls to a dict mapping tool end strings to their limits
    if tools:
        assert len(max_tool_calls) == 1 or len(max_tool_calls) == len(tools), (
            "max_tool_calls must have length 1 (applies to all tools) or same length as tools (per-tool limit)"
        )
        # tool key is the end_str
        if len(max_tool_calls) == 1:
            max_tool_calls_dict = {end_str: max_tool_calls[0] for end_str in tools.keys()}
        else:
            max_tool_calls_dict = {end_str: limit for end_str, limit in zip(tools.keys(), max_tool_calls)}
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
                # VLLM v1 multiprocessing is required due to https://github.com/vllm-project/vllm/issues/15349
                runtime_env=ray.runtime_env.RuntimeEnv(
                    env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TORCH_CUDA_ARCH_LIST": get_cuda_arch_list()}
                ),
            )
            .remote(
                model=pretrain,
                revision=revision,
                tokenizer=tokenizer_name_or_path,
                tokenizer_revision=revision,
                trust_remote_code=True,
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
                enable_sleep_mode=vllm_enable_sleep,
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
                verbose=verbose,
            )
        )

    # Verify engines initialized successfully
    try:
        ray_get_with_progress(
            [engine.ready.remote() for engine in vllm_engines], "Initializing vLLM engines", timeout=300
        )
    except TimeoutError as e:
        logger.error(f"vLLM engines failed to initialize: {e}")
        # Kill partially initialized actors before raising
        for engine in vllm_engines:
            ray.kill(engine)
        raise RuntimeError(f"vLLM engine initialization timed out: {e}")

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)


if __name__ == "__main__":
    num_engines = 1
    tensor_parallel_size = 1
    world_size = num_engines * tensor_parallel_size + 1
    vllm_engines = create_vllm_engines(
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        pretrain="facebook/opt-125m",
        revision="main",
        seed=42,
        enable_prefix_caching=False,
        max_model_len=1024,
    )
    llm = vllm_engines[0]
    from vllm.utils import get_ip, get_open_port

    master_address = get_ip()
    master_port = get_open_port()
    backend = "gloo"

    refs = [
        engine.init_process_group.remote(
            master_address, master_port, i * tensor_parallel_size + 1, world_size, "openrlhf", backend=backend
        )
        for i, engine in enumerate(vllm_engines)
    ]
    model_update_group = init_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name="openrlhf",
    )
    ray.get(refs)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    logger.info(f"output: {output}")
