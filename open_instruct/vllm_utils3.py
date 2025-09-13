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

import copy
import dataclasses
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

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


class ConcurrentDict:
    """Thread-safe dictionary for tracking request processing."""

    def __init__(self):
        self._dict = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        """Set a key-value pair atomically."""
        with self._lock:
            assert key not in self._dict, (
                f"Key {key} already exists in ConcurrentDict. This indicates duplicate entries."
            )
            self._dict[key] = value

    def get(self, key):
        """Get a value by key."""
        with self._lock:
            return self._dict.get(key)

    def contains(self, key):
        """Check if key exists."""
        with self._lock:
            return key in self._dict

    def assert_exists(self, key, error_msg):
        """Assert that a key exists with a custom error message."""
        with self._lock:
            assert key in self._dict, error_msg

    def assert_not_exists(self, key, error_msg):
        """Assert that a key does not exist with a custom error message."""
        with self._lock:
            assert key not in self._dict, error_msg

    def update_field(self, key, field, value):
        """Update a specific field in a dict value atomically."""
        with self._lock:
            assert key in self._dict, f"Key {key} not found in ConcurrentDict"
            assert field in self._dict[key], f"Field {field} not found in {key}"
            self._dict[key][field] = value

    def assert_and_update(self, key, field, expected_value, new_value, error_msg):
        """Assert current value and update atomically."""
        with self._lock:
            assert key in self._dict, f"Key {key} not found in ConcurrentDict"
            assert self._dict[key][field] == expected_value, error_msg
            self._dict[key][field] = new_value

    def delete(self, key):
        """Delete a key atomically."""
        with self._lock:
            if key in self._dict:
                del self._dict[key]

    def keys(self):
        """Get list of keys (snapshot)."""
        with self._lock:
            return list(self._dict.keys())


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


def _init_tool_tracking():
    """Initialize tracking variables for tool mode."""
    return {
        "num_calls": defaultdict(int),
        "timeout": defaultdict(bool),
        "tool_error": defaultdict(str),
        "tool_output": defaultdict(str),
        "tool_runtime": defaultdict(float),
        "tool_called": defaultdict(bool),
        "concat_outputs": {},
        "masks": defaultdict(list),
        "pending_tool_futures": {},
    }


def _handle_output(output, tools, tracking, sampling_params, max_tool_calls, executor):
    """
    Handle a finished output. Returns the output if it should be added to results,
    or None if it's being held for tool processing.

    This is a free function to keep the processing logic separate from the actor state.
    """
    if not tools:
        return output

    assert len(output.outputs) <= 1, f"{len(output.outputs)=}"  # In tool mode, sampling_params.n == 1
    o = output.outputs[0]

    # Update concatenated outputs
    if output.request_id in tracking["concat_outputs"]:
        # When updating an existing entry, we extend the token_ids of the first (and only) output
        # The entry should already have been validated to have exactly 1 output when first stored
        assert len(tracking["concat_outputs"][output.request_id].outputs) == 1, (
            f"Expected concat_outputs to have exactly 1 output for {output.request_id}, "
            f"but found {len(tracking['concat_outputs'][output.request_id].outputs)}. "
            f"This indicates multiple outputs were incorrectly stored."
        )
        tracking["concat_outputs"][output.request_id].outputs[0].token_ids.extend(o.token_ids)
    else:
        # Assert we're not processing the same request_id multiple times
        assert output.request_id not in tracking["concat_outputs"], (
            f"Request {output.request_id} is being processed multiple times. "
            f"It already exists in concat_outputs with {len(tracking['concat_outputs'].get(output.request_id, {}).get('outputs', []))} outputs."
        )

        # Assert when initially storing that we only have one output
        assert len(output.outputs) == 1, (
            f"Expected exactly 1 output when initially storing in concat_outputs for {output.request_id}, "
            f"but got {len(output.outputs)}. This may indicate tool processing created multiple outputs."
        )

        # Create a deep copy of the output to prevent reference issues
        # We need to copy the output object and its outputs list to avoid shared references
        copied_outputs = [copy.deepcopy(o) for o in output.outputs]
        assert len(copied_outputs) == 1, f"Copied outputs list has {len(copied_outputs)} items, expected 1"

        output_copy = vllm.RequestOutput(
            request_id=output.request_id,
            prompt=output.prompt,
            prompt_token_ids=output.prompt_token_ids,
            prompt_logprobs=output.prompt_logprobs,
            outputs=copied_outputs,
            finished=output.finished,
        )

        # Assert the copy has exactly 1 output
        assert len(output_copy.outputs) == 1, (
            f"After creating copy, expected 1 output but got {len(output_copy.outputs)} for {output.request_id}. "
            f"This indicates an issue with the deepcopy or RequestOutput constructor."
        )

        # Note: vllm.RequestOutput uses the exact list object we pass, not a copy
        # This is expected behavior, so we set copied_outputs to None after this to avoid reuse
        copied_outputs = None

        logger.info(f"Storing {output.request_id} in concat_outputs with {len(output_copy.outputs)} output(s)")
        tracking["concat_outputs"][output.request_id] = output_copy

        # Final verification
        assert len(tracking["concat_outputs"][output.request_id].outputs) == 1, (
            f"After storing, concat_outputs[{output.request_id}] has {len(tracking['concat_outputs'][output.request_id].outputs)} outputs, "
            f"expected 1. This indicates the outputs list was modified after assignment."
        )

    tracking["masks"][output.request_id].extend([1] * len(o.token_ids))

    # Check for tool calls
    for stop_str in sampling_params.stop:
        if stop_str in tools and o.text.endswith(stop_str):
            if tracking["num_calls"][output.request_id] < max_tool_calls.get(stop_str, 0):
                tool = tools[stop_str]
            else:
                tool = MaxCallsExceededTool(start_str="<tool>", end_str="</tool>")

            future = executor.submit(tool, o.text)
            tracking["pending_tool_futures"][output.request_id] = (future, o, output)

            return None  # Output is being held for tool processing

    return output


def _process_outputs(
    outputs: List[vllm.RequestOutput],
    dataset_index: Optional[int] = None,
    training_step: Optional[int] = None,
    token_statistics: Optional[TokenStatistics] = None,
    start_time: Optional[float] = None,
) -> "GenerationResult":
    """Process vLLM RequestOutputs into GenerationResult format."""
    # Validate dataset_index type
    assert isinstance(dataset_index, (int, type(None))), (
        f"dataset_index must be an integer or None, got {type(dataset_index)}: {dataset_index}"
    )
    response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
    finish_reasons = [out.finish_reason for output in outputs for out in output.outputs]

    masks = [[1] * len(resp) for resp in response_ids]
    num_calls = [0] * len(response_ids)
    timeouts = [0] * len(response_ids)
    tool_errors = [""] * len(response_ids)
    tool_outputs = [""] * len(response_ids)
    tool_runtimes = [0] * len(response_ids)
    tool_calleds = [False] * len(response_ids)

    request_info = RequestInfo(
        num_calls=num_calls,
        timeouts=timeouts,
        tool_errors=tool_errors,
        tool_outputs=tool_outputs,
        tool_runtimes=tool_runtimes,
        tool_calleds=tool_calleds,
    )

    result = GenerationResult(
        responses=response_ids,
        finish_reasons=finish_reasons,
        masks=masks,
        request_info=request_info,
        dataset_index=dataset_index,
        training_step=training_step,
        token_statistics=token_statistics,
        start_time=start_time,
    )

    return result


def _process_outputs_with_tools(
    outputs: List[vllm.RequestOutput],
    dataset_index: Optional[int] = None,
    training_step: Optional[int] = None,
    token_statistics: Optional[TokenStatistics] = None,
    start_time: Optional[float] = None,
) -> "GenerationResult":
    """Process vLLM RequestOutputs into GenerationResult format with tool information."""
    # Validate dataset_index type
    assert isinstance(dataset_index, (int, type(None))), (
        f"dataset_index must be an integer or None, got {type(dataset_index)}: {dataset_index}"
    )
    response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
    finish_reasons = [out.finish_reason for output in outputs for out in output.outputs]

    masks = [out.mask for output in outputs for out in output.outputs]

    num_calls = [out.num_calls for output in outputs for out in output.outputs]
    timeouts = [out.timeout for output in outputs for out in output.outputs]
    tool_errors = [out.tool_error for output in outputs for out in output.outputs]
    tool_outputs = [out.tool_output for output in outputs for out in output.outputs]
    tool_runtimes = [out.tool_runtime for output in outputs for out in output.outputs]
    tool_calleds = [out.tool_called for output in outputs for out in output.outputs]

    request_info = RequestInfo(
        num_calls=num_calls,
        timeouts=timeouts,
        tool_errors=tool_errors,
        tool_outputs=tool_outputs,
        tool_runtimes=tool_runtimes,
        tool_calleds=tool_calleds,
    )

    result = GenerationResult(
        responses=response_ids,
        finish_reasons=finish_reasons,
        masks=masks,
        request_info=request_info,
        dataset_index=dataset_index,
        training_step=training_step,
        token_statistics=token_statistics,
        start_time=start_time,
    )

    return result


def _finalize_outputs(
    output,
    tracking,
    dataset_index,
    training_step,
    tools,
    original_sampling_params,
    token_statistics=None,
    start_time=None,
):
    """Prepare final outputs based on whether tools were used."""
    # Validate dataset_index type
    assert isinstance(dataset_index, (int, type(None))), (
        f"dataset_index must be an integer or None, got {type(dataset_index)}: {dataset_index}"
    )

    if not tools:
        return _process_outputs(
            [output],
            dataset_index=dataset_index,
            training_step=training_step,
            token_statistics=token_statistics,
            start_time=start_time,
        )

    # Tool mode: add metadata and merge completions
    # Store the original request_id before output gets overwritten
    output_request_id = output.request_id

    # Only set attributes for sub-requests belonging to the current request
    for req_id in tracking["masks"]:
        # Skip if this sub-request doesn't belong to the current base request
        if _extract_base_request_id(req_id) != output_request_id:
            continue

        assert req_id in tracking["concat_outputs"], f"req_id {req_id} not in concat_outputs!"
        output_obj = tracking["concat_outputs"][req_id].outputs[0]
        setattr(output_obj, "mask", tracking["masks"][req_id])
        setattr(output_obj, "num_calls", tracking["num_calls"][req_id])
        setattr(output_obj, "timeout", tracking["timeout"][req_id])
        setattr(output_obj, "tool_error", tracking["tool_error"][req_id])
        setattr(output_obj, "tool_output", tracking["tool_output"][req_id])
        setattr(output_obj, "tool_runtime", tracking["tool_runtime"][req_id])
        setattr(output_obj, "tool_called", tracking["tool_called"][req_id])

    # Merge n completions into the same outputs
    # Filter tracking data to only include the current request
    relevant_outputs = {
        k: v for k, v in tracking["concat_outputs"].items() if _extract_base_request_id(k) == output_request_id
    }

    # Validate we have the expected number of outputs for this request
    expected_samples = len([k for k in relevant_outputs.keys() if "_".join(k.split("_")[:-1]) == output_request_id])
    if expected_samples == 0:
        raise ValueError(f"No outputs found in tracking['concat_outputs'] for request {output_request_id}")

    # Collect outputs from all sub-requests without modifying the tracking dictionary
    merged_outputs = {}
    for req_id, output_obj in relevant_outputs.items():
        real_req_id = _extract_base_request_id(req_id)

        # Each sub-request should have exactly one output
        assert len(output_obj.outputs) == 1, (
            f"Expected exactly 1 output per sub-request, got {len(output_obj.outputs)} for {req_id}. "
            f"This indicates tool processing created multiple outputs instead of one final result."
        )

        if real_req_id not in merged_outputs:
            # Initialize with lists to collect outputs from all sub-requests
            merged_outputs[real_req_id] = {
                "base_output": output_obj,  # Keep first sub-request as template for metadata
                "all_outputs": [],  # Collect all outputs here
            }

        # Add this sub-request's output to the collection (create a copy to avoid modifying original)
        merged_outputs[real_req_id]["all_outputs"].append(copy.deepcopy(output_obj.outputs[0]))

        logger.info(
            f"Collecting output for {real_req_id} from sub-request {req_id}: "
            f"total collected so far: {len(merged_outputs[real_req_id]['all_outputs'])}"
        )

    # Create new RequestOutput objects with merged outputs (without modifying originals)
    final_merged = {}
    for real_req_id, merge_data in merged_outputs.items():
        base_output = merge_data["base_output"]
        all_outputs = merge_data["all_outputs"]

        # Create a new RequestOutput with all collected outputs
        merged_output = vllm.RequestOutput(
            request_id=real_req_id,
            prompt=base_output.prompt,
            prompt_token_ids=base_output.prompt_token_ids,
            prompt_logprobs=base_output.prompt_logprobs,
            outputs=all_outputs,  # Use the collected outputs
            finished=base_output.finished,
        )

        final_merged[real_req_id] = merged_output

        logger.info(
            f"Created merged output for {real_req_id} with {len(merged_output.outputs)} outputs "
            f"(without modifying original tracking entries)"
        )

    # Since we're only processing one request at a time, final_merged should have exactly one entry
    assert len(final_merged) == 1, (
        f"Expected exactly 1 merged output for request {output_request_id}, but got {len(final_merged)}. "
        f"Keys in final_merged: {list(final_merged.keys())}"
    )

    # Get the single merged output (no sorting needed since there's only one)
    final_outputs = list(final_merged.values())

    # Additional assertion: verify we have exactly one output object
    assert len(final_outputs) == 1, (
        f"Expected exactly 1 output object in final_outputs, but got {len(final_outputs)}. "
        f"This indicates an issue with the merging logic."
    )

    # Validate total response count before processing
    total_responses = sum(len(output.outputs) for output in final_outputs)
    logger.info(
        f"_finalize_outputs: {len(relevant_outputs)} sub-requests merged into {len(final_outputs)} outputs with {total_responses} total responses"
    )

    # Add detailed logging before processing to help debug
    logger.info(
        f"Before _process_outputs_with_tools: final_outputs has {len(final_outputs)} items, "
        f"with output counts: {[len(out.outputs) for out in final_outputs]}"
    )

    # Assert each output in final_outputs has the expected structure
    for idx, output in enumerate(final_outputs):
        total_outputs = len(output.outputs)
        logger.info(f"final_outputs[{idx}] (id={output.request_id}) has {total_outputs} outputs")

    result = _process_outputs_with_tools(
        final_outputs,
        dataset_index=dataset_index,
        training_step=training_step,
        token_statistics=token_statistics,
        start_time=start_time,
    )

    # Add final validation to catch the specific error early
    expected_response_count = original_sampling_params.n
    actual_response_count = len(result.responses)
    if actual_response_count != expected_response_count:
        raise AssertionError(
            f"Response count mismatch in _finalize_outputs: expected {expected_response_count} responses "
            f"but got {actual_response_count}. This indicates incorrect merging of tool-processed outputs."
        )

    return result


def _process_completed_request(request_id, outs, tracking, current_time, tools, request_metadata):
    """Process a completed request with all its samples and return the result.

    This is a free function that processes completed requests independently of the actor state.

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
        outputs=[],  # Will be set below based on whether tools are enabled
        finished=outs[0].finished,
    )

    # When tools are enabled, _finalize_outputs will handle merging from tracking
    # When tools are disabled, we flatten all outputs here
    if not tools:
        final_output.outputs = [completion for out in outs for completion in out.outputs]

    total_generation_tokens = sum(len(completion.token_ids) for out in outs for completion in out.outputs)
    metadata = request_metadata[request_id]  # Don't pop yet, _poll_tool_futures might need it

    # Validate dataset_index from metadata
    assert isinstance(metadata["dataset_index"], int), (
        f"metadata['dataset_index'] must be an integer, got {type(metadata['dataset_index'])}: {metadata['dataset_index']} "
        f"for request_id={request_id}"
    )

    result = _finalize_outputs(
        final_output,
        tracking,
        metadata["dataset_index"],
        metadata["training_step"],
        tools,
        original_sampling_params=metadata["original_sampling_params"],
        token_statistics=TokenStatistics(
            num_prompt_tokens=metadata["prompt_tokens"],
            num_response_tokens=total_generation_tokens,
            generation_time=current_time - metadata["start_time"],
        ),
        start_time=metadata["start_time"],
    )
    return result, metadata["is_eval"]


def _extract_base_request_id(full_request_id: str) -> str:
    """Extract base request ID by removing the sample suffix.

    >>> _extract_base_request_id("train_1_43039_0")
    'train_1_43039'
    >>> _extract_base_request_id("eval_5_12345_2")
    'eval_5_12345'
    """
    return "_".join(full_request_id.split("_")[:-1])


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


def add_request(
    request: PromptRequest,
    llm_engine: vllm.LLMEngine,
    tools,
    request_metadata: dict,
    vllm_active_requests: dict = None,
) -> int:
    """Add a request to the LLM engine."""
    prefix = "eval" if request.is_eval else "train"
    request_id = f"{prefix}_{request.training_step}_{request.dataset_index}"
    sampling_params = request.generation_config.clone()
    sampling_params.n = 1  # Use n=1 for tool processing
    request_metadata[request_id] = {
        "is_eval": request.is_eval,
        "dataset_index": request.dataset_index,
        "training_step": request.training_step,
        "sampling_params": sampling_params,
        "original_sampling_params": request.generation_config,
        "prompt_tokens": len(request.prompt),
        "start_time": time.perf_counter(),
    }

    tokens_prompt = vllm.TokensPrompt(prompt_token_ids=request.prompt, cache_salt=request_id)
    n = request.generation_config.n
    for j in range(n):
        sub_sampling_params = sampling_params.clone()  # Already has n=1
        if request.generation_config.seed is not None:
            sub_sampling_params.seed = request.generation_config.seed + j
        sub_request_id = f"{request_id}_{j}"
        llm_engine.add_request(sub_request_id, tokens_prompt, sub_sampling_params)
        # Track this request as active in vLLM
        if vllm_active_requests is not None:
            vllm_active_requests[sub_request_id] = request.dataset_index

    # Assert all n sub-requests are tracked
    if vllm_active_requests is not None:
        for j in range(n):
            sub_request_id = f"{request_id}_{j}"
            assert sub_request_id in vllm_active_requests, (
                f"Sub-request {sub_request_id} missing from vllm_active_requests after adding. "
                f"Expected all {n} sub-requests for {request_id} to be tracked."
            )

    return n


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
        verbose: bool = False,
        **kwargs,
    ):
        self.logger = logger_utils.setup_logger(__name__)
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.inference_batch_size = inference_batch_size
        self.verbose = verbose
        self.request_metadata = {}
        self.request_tracking = ConcurrentDict()  # Thread-safe tracking
        self.vllm_active_requests = {}  # Track all requests currently in vLLM: request_id -> dataset_index

        if self.tools:
            self.executor = ThreadPoolExecutor(max_workers=20)
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
            self.logger.info(f"creating LLM with bundle_indices={bundle_indices}")

        # for some reason log stats causes a crash in the engine at assert outputs.scheduler_stats is not None
        self.llm_engine = vllm.LLMEngine.from_engine_args(vllm.EngineArgs(*args, **kwargs, disable_log_stats=True))

        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.actor_manager = actor_manager

        # For caching should_stop status.
        self._last_should_stop_update = float("-inf")
        self._should_stop_value = False
        self._should_stop_timeout_s = 5

        self._prefetch_cv = threading.Condition()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

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

    def _prefetch_worker(self):
        """Background worker that prefetches requests until we have enough buffered."""
        while True:
            with self._prefetch_cv:
                should_stop = self._should_stop()
                if should_stop and self.verbose:
                    try:
                        queue_size = self.prompt_queue.qsize()
                    except Exception:
                        queue_size = "unknown"
                    self.logger.info(f"Prefetch worker: should_stop=True, waiting. main_queue_size={queue_size}")
                if should_stop:
                    self._prefetch_cv.wait(timeout=0.1)
                    continue

                current_unfinished = self.llm_engine.get_num_unfinished_requests()
                if current_unfinished >= self.inference_batch_size:
                    self._prefetch_cv.wait(timeout=0.1)  # shorter wait is OK
                    continue
            try:
                request = self.prompt_queue.get(timeout=0.1)

                # Track that we pulled this dataset_index from the prompt queue
                dataset_index = request.dataset_index
                training_step = request.training_step
                # Create unique tracking key combining training_step and dataset_index
                tracking_key = f"{training_step}_{dataset_index}"

                # ALWAYS log this for debugging
                self.logger.info(
                    f"[_prefetch_worker] Pulling dataset_index={dataset_index} from prompt queue, "
                    f"is_eval={request.is_eval}, training_step={training_step}, tracking_key={tracking_key}"
                )

                self.request_tracking.assert_not_exists(
                    tracking_key,
                    f"Tracking key {tracking_key} already in request_tracking. "
                    f"This indicates duplicate dataset indices in prompt queue.",
                )
                self.request_tracking.set(
                    tracking_key,
                    {
                        "pulled": True,
                        "inserted": False,
                        "dataset_index": dataset_index,
                        "training_step": training_step,
                    },
                )

                self.logger.info(
                    f"[_prefetch_worker] Successfully tracked dataset_index={dataset_index}, "
                    f"current tracking keys: {self.request_tracking.keys()}"
                )

                num_added = add_request(
                    request,
                    self.llm_engine,
                    self.tools,
                    request_metadata=self.request_metadata,
                    vllm_active_requests=self.vllm_active_requests,
                )

                # Validate sub-request counts after adding
                prefix = "eval" if request.is_eval else "train"
                request_id = f"{prefix}_{request.training_step}_{request.dataset_index}"
                self._validate_single_request_counts(request_id, f"After adding request {request_id}")

                if self.verbose and num_added > 0:
                    self.logger.info(
                        f"Prefetch worker: added {num_added} requests directly to engine, "
                        f"current_unfinished={self.llm_engine.get_num_unfinished_requests()}"
                    )

                with self._prefetch_cv:
                    self._prefetch_cv.notify_all()

            except queue.Empty:
                continue

    def _insert_result_to_queue(self, result, is_eval: bool):
        """Insert result into the appropriate queue with blocking put."""
        # Validate that this dataset_index was pulled from prompt queue
        dataset_index = result.dataset_index
        if dataset_index is not None:  # dataset_index can be None for combined results
            # Build tracking key from training_step and dataset_index
            training_step = result.training_step
            tracking_key = f"{training_step}_{dataset_index}"

            # Log detailed tracking info
            self.logger.info(
                f"[_insert_result_to_queue] Attempting to insert dataset_index={dataset_index}, "
                f"training_step={training_step}, tracking_key={tracking_key}, "
                f"is_eval={is_eval}, "
                f"tracking contains: {self.request_tracking.keys()}, "
                f"active vLLM requests: {list(self.vllm_active_requests.values())}, "
                f"result type: {type(result)}, "
                f"has outputs: {hasattr(result, 'outputs')}"
            )

            # Check if this tracking_key was ever tracked
            # Note: We still check dataset_index in vllm_active_requests since that uses dataset_index alone
            if dataset_index in self.vllm_active_requests.values() and not self.request_tracking.contains(
                tracking_key
            ):
                self.logger.error(
                    f"CRITICAL: dataset_index {dataset_index} is in vllm_active_requests but tracking_key {tracking_key} NOT in request_tracking! "
                    f"This indicates tracking was lost or deleted prematurely."
                )

            # Since we no longer insert tool continuation results,
            # every result should have been tracked
            self.request_tracking.assert_exists(
                tracking_key,
                f"Tracking key {tracking_key} was never pulled from prompt queue. "
                f"Available keys: {self.request_tracking.keys()}",
            )
            self.request_tracking.assert_and_update(
                tracking_key,
                "inserted",
                False,
                True,
                f"Tracking key {tracking_key} has already been inserted into results queue. "
                f"This indicates duplicate result insertion.",
            )

        results_queue = self.eval_results_queue if is_eval else self.results_queue
        results_queue.put(result)

    def process_from_queue(self, timeout: float = 60.0):
        """Run generation loop using LLMEngine directly, with optional tool support.

        Runs continuously until should_stop is set, periodically adding new requests
        and yielding control to allow weight synchronization.

        Returns:
            int: Number of requests processed
        """

        # Initialize tracking and outputs if they don't exist (for backwards compatibility)
        if not hasattr(self, "tracking"):
            self.tracking = _init_tool_tracking()
        if not hasattr(self, "request_outputs"):
            self.request_outputs = {}

        # Use persistent instance variables for tracking and outputs
        # This ensures state is maintained across multiple calls
        total_processed = 0
        iteration_count = 0

        exit_reason = "unknown"
        while True:
            iteration_count += 1

            # Check exit conditions - only exit if should_stop AND no pending work
            if self._should_stop():
                pending_tools = len(self.tracking["pending_tool_futures"])
                unfinished = self.llm_engine.get_num_unfinished_requests()

                if pending_tools == 0 and unfinished == 0:
                    exit_reason = "should_stop requested (all work complete)"
                    break
                elif self.verbose and iteration_count % 100 == 0:
                    self.logger.info(f"Delaying stop: unfinished={unfinished}, pending_tools={pending_tools}")

            # Return every 10k iterations to allow weight synchronization
            if iteration_count % 10000 == 0:
                exit_reason = "10k iteration limit reached"
                if self.verbose:
                    self.logger.info(f"process_from_queue exiting: {exit_reason}")
                return total_processed

            # Process tool futures (just updates tracking, doesn't return results to insert)
            self._poll_tool_futures(self.tracking, self.llm_engine.tokenizer)
            current_time = time.time()
            # Tool outputs are now accumulated in tracking and will be inserted
            # only when the final request completes

            # Process engine steps - ONLY if there are unfinished requests
            if self.llm_engine.has_unfinished_requests():
                all_outputs = list(self.llm_engine.step())

                # Assert that all outputs from step() are finished
                unfinished_outputs = [o for o in all_outputs if not o.finished]
                assert not unfinished_outputs, (
                    f"vLLM step() returned {len(unfinished_outputs)} unfinished outputs. "
                    f"This violates our assumption that we only need to process finished outputs. "
                    f"Unfinished request IDs: {[o.request_id for o in unfinished_outputs]}"
                )

                step_outputs = all_outputs

                for output in step_outputs:
                    # We always set n=1 for each sub-request in add_request()
                    assert len(output.outputs) == 1, (
                        f"vLLM returned {len(output.outputs)} outputs for request {output.request_id}, "
                        f"but we always set n=1 for sub-requests. This indicates vLLM is not respecting "
                        f"the n parameter in sampling_params."
                    )

                    # Remove from vllm_active_requests since this request is finished
                    if output.request_id not in self.vllm_active_requests:
                        raise RuntimeError(
                            f"Sub-request {output.request_id} completed but was not in vllm_active_requests. "
                            f"This indicates a critical tracking bug. Active requests: {list(self.vllm_active_requests.keys())}"
                        )
                    del self.vllm_active_requests[output.request_id]
                    if self.verbose:
                        self.logger.info(f"Removed {output.request_id} from vllm_active_requests")

                    base_req_id = _extract_base_request_id(output.request_id)

                    # Check if metadata exists for this request
                    if base_req_id not in self.request_metadata:
                        raise RuntimeError(
                            f"Critical bug: Missing metadata for request {base_req_id}. "
                            f"This indicates metadata was cleaned up prematurely while sub-requests were still processing. "
                            f"All available metadata keys: {list(self.request_metadata.keys())}"
                        )

                    # Assert: check sample count before processing
                    expected_samples = self.request_metadata[base_req_id]["original_sampling_params"].n
                    current_samples_before = sum(
                        1 for k in self.tracking["concat_outputs"].keys() if _extract_base_request_id(k) == base_req_id
                    )

                    result = _handle_output(
                        output,
                        self.tools,
                        self.tracking,
                        self.request_metadata[base_req_id]["sampling_params"],
                        self.max_tool_calls,
                        self.executor,
                    )

                    # Assert: check sample count after processing
                    current_samples_after = sum(
                        1 for k in self.tracking["concat_outputs"].keys() if _extract_base_request_id(k) == base_req_id
                    )
                    assert current_samples_after <= expected_samples, (
                        f"Too many samples for {base_req_id}: expected ≤{expected_samples}, got {current_samples_after} (was {current_samples_before} before). Keys: {[k for k in self.tracking['concat_outputs'].keys() if _extract_base_request_id(k) == base_req_id]}"
                    )
                    # Result is None when we do more tool processing.
                    if result is not None:
                        # Sub-request is done (no more tool calls)

                        # Get the CompletionOutput to add
                        assert len(output.outputs) == 1, f"{len(output.outputs)=} != 1"
                        if output.request_id in self.tracking["concat_outputs"]:
                            complete_output = self.tracking["concat_outputs"][output.request_id].outputs[0]
                        else:
                            complete_output = result.outputs[0]

                        # Use helper to finalize the sub-request
                        processed = self._finalize_sub_request(
                            output.request_id, output, complete_output, current_time
                        )
                        total_processed += processed

                        # Validate after complete transition to request_outputs
                        self._validate_single_request_counts(base_req_id, f"After finalizing {output.request_id}")
                    else:
                        # Sub-request went to tools, validate the transition
                        self._validate_single_request_counts(
                            base_req_id, f"After sending {output.request_id} to tools"
                        )

            # Validate all requests after processing is complete
            self._validate_sub_request_counts("After processing all outputs")

            # If no work to do, break to yield control back to generate_thread,
            # allowing weight_sync_thread to acquire the LLMRayActor lock
            final_unfinished = self.llm_engine.get_num_unfinished_requests()
            pending_tools = len(self.tracking["pending_tool_futures"])
            if self.verbose and iteration_count % 100 == 0:
                self.logger.info(
                    f"process_from_queue iteration {iteration_count}: unfinished={final_unfinished}, pending_tools={pending_tools}"
                )
            if final_unfinished == 0 and pending_tools > 0:
                # Sleep for 1 second to let pending tools complete.
                time.sleep(1)
            if final_unfinished + pending_tools == 0:
                # CRITICAL INVARIANT CHECKS before exiting
                # When we think we're done, verify that ALL requests are actually complete

                # 1. Check metadata for incomplete requests
                incomplete_metadata = []
                for req_id, metadata in self.request_metadata.items():
                    expected_n = metadata["original_sampling_params"].n
                    if req_id not in self.request_outputs:
                        # This request has metadata but no outputs yet
                        incomplete_metadata.append(
                            f"{req_id}: no outputs in request_outputs (expecting {expected_n} samples)"
                        )
                    else:
                        num_outputs = len(self.request_outputs[req_id].outputs)
                        if num_outputs < expected_n:
                            incomplete_metadata.append(f"{req_id}: only {num_outputs}/{expected_n} outputs collected")

                # 2. Check for pending outputs not yet processed
                pending_in_outputs = []
                for req_id, request_output in self.request_outputs.items():
                    if request_output.outputs:  # Has outputs waiting
                        if req_id in self.request_metadata:
                            expected_n = self.request_metadata[req_id]["original_sampling_params"].n
                            if len(request_output.outputs) >= expected_n:
                                # We have enough outputs but haven't processed them
                                pending_in_outputs.append(
                                    f"{req_id}: {len(request_output.outputs)}/{expected_n} outputs ready but not inserted"
                                )

                # 3. Check tracking for orphaned sub-requests
                orphaned_tracking = []
                if self.tools and "concat_outputs" in self.tracking:
                    for sub_id in self.tracking["concat_outputs"]:
                        base_id = _extract_base_request_id(sub_id)
                        # Check if base request exists in metadata
                        if base_id not in self.request_metadata:
                            # This sub-request has no parent metadata - might be already processed
                            # Only flag as orphaned if it wasn't already inserted
                            if base_id not in self.request_outputs or not self.request_outputs[base_id]:
                                # No pending outputs for this base, so truly orphaned
                                orphaned_tracking.append(f"{sub_id} (base: {base_id})")

                # 4. Check if any sub-requests are still active in vLLM
                active_sub_requests = []
                for req_id in self.request_metadata.keys():
                    expected_n = self.request_metadata[req_id]["original_sampling_params"].n
                    for j in range(expected_n):
                        sub_id = f"{req_id}_{j}"
                        if sub_id in self.vllm_active_requests:
                            active_sub_requests.append(sub_id)

                # If we have incomplete requests or active sub-requests, don't exit yet
                if incomplete_metadata or pending_in_outputs or orphaned_tracking or active_sub_requests:
                    if self.verbose or active_sub_requests:
                        self.logger.info(
                            f"Detected incomplete state - continuing processing:\\n"
                            f"  Incomplete metadata: {len(incomplete_metadata)} requests\\n"
                            f"  Pending outputs: {len(pending_in_outputs)} requests\\n"
                            f"  Orphaned tracking: {len(orphaned_tracking)} sub-requests\\n"
                            f"  Active sub-requests in vLLM: {active_sub_requests[:5]}{'...' if len(active_sub_requests) > 5 else ''}"
                        )

                    # If we only have incomplete metadata/outputs but no active work, we have a real problem
                    if not active_sub_requests and final_unfinished == 0 and pending_tools == 0:
                        # Give it one more second in case outputs are in flight
                        time.sleep(1.0)

                        # Re-check after sleep
                        final_unfinished = self.llm_engine.get_num_unfinished_requests()
                        pending_tools = len(self.tracking["pending_tool_futures"])
                        active_sub_requests = [
                            f"{req_id}_{j}"
                            for req_id in self.request_metadata.keys()
                            for j in range(self.request_metadata[req_id]["original_sampling_params"].n)
                            if f"{req_id}_{j}" in self.vllm_active_requests
                        ]

                        if final_unfinished == 0 and pending_tools == 0 and not active_sub_requests:
                            # Still no work after waiting - this is a real problem
                            # Use comprehensive validation to identify the exact issue
                            try:
                                self._validate_sub_request_counts("At exit with incomplete state")
                            except AssertionError as e:
                                # Validation will provide detailed error message about what's wrong
                                error_msg = (
                                    f"\\n=== CRITICAL: INCOMPLETE STATE AT EXIT ===\\n"
                                    f"vLLM unfinished: {final_unfinished}\\n"
                                    f"Pending tools: {pending_tools}\\n"
                                    f"Active sub-requests: {active_sub_requests or 'None'}\\n"
                                    f"Incomplete metadata: {incomplete_metadata or 'None'}\\n"
                                    f"Ready but not inserted: {pending_in_outputs or 'None'}\\n"
                                    f"Orphaned tracking: {orphaned_tracking or 'None'}\\n"
                                    f"Request metadata keys: {list(self.request_metadata.keys())}\\n"
                                    f"Request outputs keys: {list(self.request_outputs.keys())}\\n"
                                    f"\\nValidation error: {str(e)}"
                                )
                                self.logger.error(error_msg)

                                # Re-raise with full context
                                assert False, (
                                    f"Attempting to exit process_from_queue with incomplete requests. "
                                    f"This would cause the training loop to hang waiting for results that will never come.\\n"
                                    f"{error_msg}"
                                )

                    # Continue processing - don't exit yet
                    continue

                exit_reason = "no work remaining"
                break

        if self.verbose:
            self.logger.info(f"process_from_queue exiting: {exit_reason}")
        return total_processed

    def _maybe_process_and_insert(
        self,
        request_id: str,
        request_outputs: Dict[str, List[vllm.RequestOutput]],
        tracking: Dict[str, Any],
        current_time: float,
    ) -> int:
        """Check if we have N requests for request_id, process them, and insert results in queue.

        Returns:
            int: Number of requests processed (0 or 1).
        """
        expected_n = self.request_metadata[request_id]["original_sampling_params"].n

        # ---- Readiness check and canonicalization of sub-requests ----
        # Build a canonical map: sub_id -> chosen RequestOutput
        # Prefer tool-merged results from tracking["concat_outputs"]; otherwise fallback to engine outputs.
        def _suffix_index(sub_req_id: str) -> int:
            # sub_req_id looks like "<base>_<j>"; take the last underscore
            try:
                return int(sub_req_id.rsplit("_", 1)[1])
            except Exception:
                return -1

        canonical: Dict[str, vllm.RequestOutput] = {}

        if self.tools:
            # 1) Only include outputs that are actually complete (no pending tool futures)
            for sub_req_id, output_obj in list(tracking["concat_outputs"].items()):
                if _extract_base_request_id(sub_req_id) != request_id:
                    continue
                # Check if this output has pending tool calls
                has_pending = sub_req_id in tracking.get("pending_tool_futures", {})
                logger.info(
                    f"Checking {sub_req_id} in concat_outputs: has {len(output_obj.outputs)} outputs, "
                    f"has_pending_tool_future={has_pending}"
                )

                # CRITICAL: Only consider this output ready if it has no pending tool futures
                if has_pending:
                    logger.info(f"Skipping {sub_req_id} - has pending tool future")
                    continue

                # Assert that tool-merged outputs have exactly one output
                assert len(output_obj.outputs) == 1, (
                    f"Tool-merged output for {sub_req_id} has {len(output_obj.outputs)} outputs, "
                    f"expected exactly 1. This indicates incorrect tool processing or merging. "
                    f"Output IDs: {[o.index for o in output_obj.outputs] if hasattr(output_obj.outputs[0], 'index') else 'no index attr'}"
                )
                canonical[sub_req_id] = output_obj

        # 2) Check outputs already collected in request_outputs
        # With new structure, request_outputs[request_id] is a single RequestOutput with multiple CompletionOutputs
        if request_id in request_outputs and request_outputs[request_id].outputs:
            # Each CompletionOutput should have an index field indicating which sub-request it came from
            for comp_output in request_outputs[request_id].outputs:
                if hasattr(comp_output, "index"):
                    sub_id = f"{request_id}_{comp_output.index}"
                    # Create a RequestOutput wrapper for this CompletionOutput to match canonical structure
                    if sub_id not in canonical:
                        # CRITICAL: Check if this sub-request has a pending tool future
                        # This mirrors the check in Section 1 to ensure consistency
                        if sub_id in tracking.get("pending_tool_futures", {}):
                            logger.info(f"Skipping {sub_id} from request_outputs - has pending tool future")
                            continue

                        # Create a wrapper RequestOutput for consistency
                        wrapper = vllm.RequestOutput(
                            request_id=sub_id,
                            prompt=request_outputs[request_id].prompt,
                            prompt_token_ids=request_outputs[request_id].prompt_token_ids,
                            prompt_logprobs=request_outputs[request_id].prompt_logprobs,
                            outputs=[comp_output],
                            finished=True,
                        )
                        canonical[sub_id] = wrapper

        # If we don't have all expected sub-requests yet, wait.
        # We expect sub-ids exactly: f"{request_id}_{j}" for j in range(expected_n)
        needed_ids = [f"{request_id}_{j}" for j in range(expected_n)]
        available = [sid for sid in needed_ids if sid in canonical]

        # Log the readiness check
        self.logger.info(
            f"[_maybe_process_and_insert] Checking {request_id}: "
            f"expected_n={expected_n}, available={len(available)}, "
            f"canonical_keys={list(canonical.keys())}, "
            f"needed={needed_ids}"
        )

        if len(available) < expected_n:
            self.logger.info(f"[_maybe_process_and_insert] Not ready - only {len(available)}/{expected_n} available")
            return 0

        # CRITICAL: Check if any sub-requests still have active vLLM continuations
        # This can happen when tool calls complete and add continuations back to vLLM
        active_sub_requests = [sub_id for sub_id in needed_ids if sub_id in self.vllm_active_requests]
        if active_sub_requests:
            logger.info(
                f"[_maybe_process_and_insert] Cannot process {request_id} yet - "
                f"sub-requests still active in vLLM: {active_sub_requests}"
            )
            return 0

        # Build ordered outs (0..n-1), ensuring one per sub-request.
        ordered_outs: List[vllm.RequestOutput] = []
        for j in range(expected_n):
            sub_id = f"{request_id}_{j}"
            out = canonical.get(sub_id)
            if out is None:
                # Should not happen due to check above; be conservative.
                return 0
            ordered_outs.append(out)

        # Ensure tracking stubs exist for every sub-request (tools enabled case).
        if self.tools:
            for out in ordered_outs:
                sub_id = out.request_id
                if sub_id not in tracking["concat_outputs"]:
                    # Create a stub so _finalize_outputs can read masks/metadata uniformly.
                    # Add assertion to detect if we're getting multiple outputs per sub-request
                    assert len(out.outputs) == 1, (
                        f"Expected exactly 1 output per sub-request when creating stub, "
                        f"but got {len(out.outputs)} outputs for {sub_id}. "
                        f"This may indicate vLLM is generating multiple samples per sub-request "
                        f"(e.g., best-of-n sampling)."
                    )

                    # Create deep copies to avoid reference issues
                    copied_outputs = [copy.deepcopy(o) for o in out.outputs]
                    assert len(copied_outputs) == 1, (
                        f"Copied outputs list for stub has {len(copied_outputs)} items, expected 1"
                    )

                    stub = vllm.RequestOutput(
                        request_id=sub_id,
                        prompt=out.prompt,
                        prompt_token_ids=out.prompt_token_ids,
                        prompt_logprobs=out.prompt_logprobs,
                        outputs=copied_outputs,
                        finished=True,
                    )

                    # Assert the stub has exactly 1 output
                    assert len(stub.outputs) == 1, (
                        f"After creating stub, expected 1 output but got {len(stub.outputs)} for {sub_id}. "
                        f"This indicates an issue with the list copy or RequestOutput constructor."
                    )

                    # Note: vllm.RequestOutput uses the exact list object we pass, not a copy
                    # This is expected behavior, so we set copied_outputs to None after this to avoid reuse
                    copied_outputs = None

                    logger.info(f"Creating stub for {sub_id} in concat_outputs with {len(stub.outputs)} output(s)")
                    tracking["concat_outputs"][sub_id] = stub

                    # Final verification
                    assert len(tracking["concat_outputs"][sub_id].outputs) == 1, (
                        f"After storing stub, concat_outputs[{sub_id}] has {len(tracking['concat_outputs'][sub_id].outputs)} outputs, "
                        f"expected 1. This indicates the outputs list was modified after assignment."
                    )
                    token_count = len(stub.outputs[0].token_ids) if stub.outputs else 0
                    tracking["masks"][sub_id] = [1] * token_count  # 1 = model tokens, 0 = tool tokens
                    tracking["num_calls"][sub_id] = 0
                    tracking["timeout"][sub_id] = False
                    tracking["tool_error"][sub_id] = ""
                    tracking["tool_output"][sub_id] = ""
                    tracking["tool_runtime"][sub_id] = 0.0
                    tracking["tool_called"][sub_id] = False

        # At this point we're ready to finalize exactly n samples.
        outs = ordered_outs
        # Do NOT mutate tracking here; cleanup happens after enqueue.

        # Remove the base entry from request_outputs to prevent growth.
        request_outputs.pop(request_id, None)
        result, is_eval = _process_completed_request(
            request_id, outs, tracking, current_time, self.tools, self.request_metadata
        )

        # Validate dataset_index consistency before inserting
        expected_dataset_index = self.request_metadata[request_id]["dataset_index"]
        actual_dataset_index = result.dataset_index

        self.logger.info(
            f"[_maybe_process_and_insert] Processing completed request {request_id}, "
            f"expected_dataset_index={expected_dataset_index}, "
            f"actual_dataset_index={actual_dataset_index}, "
            f"is_eval={is_eval}"
        )

        assert expected_dataset_index == actual_dataset_index, (
            f"Dataset index mismatch: expected {expected_dataset_index} from metadata, "
            f"but got {actual_dataset_index} in result for request_id {request_id}"
        )

        self._insert_result_to_queue(result, is_eval=is_eval)

        # Clean up request tracking for this dataset_index
        # Build tracking key from training_step and dataset_index
        # IMPORTANT: Get training_step BEFORE cleanup_request_data removes metadata
        training_step = self.request_metadata[request_id]["training_step"]
        tracking_key = f"{training_step}_{expected_dataset_index}"

        # Clean up metadata and tracking for this request after enqueuing
        self._cleanup_request_data(request_id, tracking)

        if self.request_tracking.contains(tracking_key):
            # Verify it was properly inserted before cleaning up
            tracking_entry = self.request_tracking.get(tracking_key)
            assert tracking_entry["inserted"], f"Tracking key {tracking_key} was not marked as inserted before cleanup"

            # Safety check: Ensure no active vLLM requests or pending tool futures for this dataset_index
            active_requests_for_dataset = [
                req_id for req_id, ds_idx in self.vllm_active_requests.items() if ds_idx == expected_dataset_index
            ]

            has_pending_tools = self._has_pending_tool_futures_for_request(request_id, tracking)

            # Get the IDs of pending tool futures for this request for better debugging
            pending_tool_ids = (
                [
                    req_id
                    for req_id in tracking["pending_tool_futures"]
                    if _extract_base_request_id(req_id) == request_id
                ]
                if has_pending_tools
                else []
            )

            if active_requests_for_dataset or has_pending_tools:
                raise ValueError(
                    f"CRITICAL: Attempting to clean up tracking_key {tracking_key} but found:\n"
                    f"  - Active vLLM requests: {active_requests_for_dataset}\n"
                    f"  - Has pending tool futures: {has_pending_tools}\n"
                    f"  - Pending tool future IDs: {pending_tool_ids}\n"
                    f"  - Request ID: {request_id}\n"
                    f"  - All active vLLM requests: {list(self.vllm_active_requests.keys())}\n"
                    f"This indicates requests are being cleaned up while still being processed!"
                )

            self.logger.info(
                f"[Cleanup] Deleting tracking for tracking_key={tracking_key}, "
                f"request_id={request_id}, "
                f"remaining tracking keys before delete: {self.request_tracking.keys()}"
            )
            self.request_tracking.delete(tracking_key)
            self.logger.info(
                f"[Cleanup] Successfully deleted tracking for tracking_key={tracking_key}, "
                f"remaining tracking keys after delete: {self.request_tracking.keys()}"
            )

        if self.verbose:
            self.logger.info(f"Completed and inserted request {request_id} with {expected_n} samples (eval={is_eval})")
        return 1

    def _has_pending_tool_futures_for_request(self, request_id: str, tracking: Dict[str, Any]) -> bool:
        """Check if there are any pending tool futures for a given base request ID."""
        if not self.tools or not tracking["pending_tool_futures"]:
            return False

        # Check if any pending tool futures belong to this base request
        for req_id in tracking["pending_tool_futures"]:
            if _extract_base_request_id(req_id) == request_id:
                return True
        return False

    def _has_pending_engine_requests_for_base_id(self, base_request_id: str) -> bool:
        """Check if there are any unfinished sub-requests in the vLLM engine for a given base request ID."""
        if not self.llm_engine.has_unfinished_requests():
            return False

        # Get all unfinished request IDs from the engine
        try:
            # vLLM engines have a scheduler that tracks unfinished requests
            # We need to check if any unfinished request IDs belong to our base request
            unfinished_request_ids = []

            # Try to get unfinished request IDs from scheduler
            if hasattr(self.llm_engine, "scheduler"):
                if hasattr(self.llm_engine.scheduler, "waiting") and self.llm_engine.scheduler.waiting:
                    unfinished_request_ids.extend([req.request_id for req in self.llm_engine.scheduler.waiting])
                if hasattr(self.llm_engine.scheduler, "running") and self.llm_engine.scheduler.running:
                    unfinished_request_ids.extend([req.request_id for req in self.llm_engine.scheduler.running])
                if hasattr(self.llm_engine.scheduler, "swapped") and self.llm_engine.scheduler.swapped:
                    unfinished_request_ids.extend([req.request_id for req in self.llm_engine.scheduler.swapped])

            # Check if any unfinished request belongs to our base request ID
            for req_id in unfinished_request_ids:
                if _extract_base_request_id(req_id) == base_request_id:
                    return True

        except Exception as e:
            # If we can't check the scheduler state, be conservative and assume there might be pending requests
            self.logger.warning(f"Could not check engine state for base_request_id {base_request_id}: {e}")
            return True

        return False

    def _validate_single_request_counts(self, base_request_id: str, context: str):
        """Validate that all sub-requests for a specific request are accounted for."""
        if base_request_id not in self.request_metadata:
            # Request already cleaned up, skip validation
            return

        expected = self.request_metadata[base_request_id]["original_sampling_params"].n

        # Count sub-requests for this specific request
        vllm_count = sum(1 for k in self.vllm_active_requests.keys() if k.startswith(base_request_id + "_"))
        tools_count = sum(
            1 for k in self.tracking["pending_tool_futures"].keys() if k.startswith(base_request_id + "_")
        )
        if base_request_id in self.request_outputs:
            outputs_count = len(self.request_outputs[base_request_id].outputs)
        else:
            outputs_count = 0

        total = vllm_count + tools_count + outputs_count

        if total != expected:
            # Get sub-request IDs in outputs based on their index
            output_indices = []
            if base_request_id in self.request_outputs:
                for comp_output in self.request_outputs[base_request_id].outputs:
                    if hasattr(comp_output, "index"):
                        output_indices.append(f"{base_request_id}_{comp_output.index}")
                    else:
                        output_indices.append("missing_index")

            error_msg = (
                f"[{context}] Validation failed for {base_request_id}:\n"
                f"  Expected: {expected}\n"
                f"  Found: {total}\n"
                f"  - vLLM active: {vllm_count} -> {[k for k in self.vllm_active_requests.keys() if k.startswith(base_request_id + '_')]}\n"
                f"  - Pending tools: {tools_count} -> {[k for k in self.tracking['pending_tool_futures'].keys() if k.startswith(base_request_id + '_')]}\n"
                f"  - Request outputs: {outputs_count} -> {output_indices}"
            )
            raise RuntimeError(f"Sub-request tracking inconsistency: {error_msg}")

    def _validate_sub_request_counts(self, context: str):
        """Validate that all sub-requests are accounted for."""
        # Group by base request ID - using defaultdict for cleaner code
        request_accounting = defaultdict(lambda: {"vllm": 0, "pending_tools": 0, "outputs": 0, "expected": 0})

        # Count vLLM active
        for sub_id in self.vllm_active_requests:
            base_id = _extract_base_request_id(sub_id)
            request_accounting[base_id]["vllm"] += 1

        # Count pending tools
        for sub_id in self.tracking["pending_tool_futures"]:
            base_id = _extract_base_request_id(sub_id)
            request_accounting[base_id]["pending_tools"] += 1

        # Count in request_outputs
        for base_id, request_output in self.request_outputs.items():
            request_accounting[base_id]["outputs"] += len(request_output.outputs)

        # Validate - crash if metadata is missing (that's a bug)
        errors = []
        for base_id in request_accounting:
            # If base_id is not in request_metadata, that's a critical error
            request_accounting[base_id]["expected"] = self.request_metadata[base_id]["original_sampling_params"].n

            counts = request_accounting[base_id]
            total = counts["vllm"] + counts["pending_tools"] + counts["outputs"]
            if total != counts["expected"]:
                # Add detailed logging before failing
                self.logger.error(f"[{context}] Validation failed for {base_id}:")
                self.logger.error(f"  Expected: {counts['expected']}")
                self.logger.error(f"  Found: {total}")
                self.logger.error(
                    f"  - vLLM active: {counts['vllm']} -> {[k for k in self.vllm_active_requests.keys() if k.startswith(base_id)]}"
                )
                self.logger.error(
                    f"  - Pending tools: {counts['pending_tools']} -> {[k for k in self.tracking['pending_tool_futures'].keys() if k.startswith(base_id)]}"
                )
                self.logger.error(f"  - Request outputs: {counts['outputs']}")

                errors.append(
                    f"{base_id}: expected {counts['expected']}, got {total} "
                    f"(vllm={counts['vllm']}, tools={counts['pending_tools']}, "
                    f"outputs={counts['outputs']})"
                )

        if errors:
            self.logger.error(f"[{context}] Sub-request count validation failed:\n" + "\n".join(errors))
            assert False, f"Sub-request tracking inconsistency at {context}"

    def _cleanup_request_data(self, request_id: str, tracking: Dict[str, Any]):
        """Clean up metadata and tracking data for a completed request."""
        # Check if there are still pending tool futures for this request
        if self._has_pending_tool_futures_for_request(request_id, tracking):
            # Don't clean up metadata yet - tool futures still need it
            return

        # Check if there are still unfinished sub-requests in the engine for this base request
        if self._has_pending_engine_requests_for_base_id(request_id):
            # Don't clean up metadata yet - engine step processing still needs it
            return

        # Remove request metadata only after both conditions are met:
        # 1. No pending tool futures for this request
        # 2. No unfinished sub-requests in the engine for this base request
        self.request_metadata.pop(request_id, None)

        # Clean up tracking data for all sub-requests of this request
        if self.tools:
            # Find all sub-request IDs that belong to this base request
            sub_request_ids = [
                k for k in tracking["concat_outputs"].keys() if _extract_base_request_id(k) == request_id
            ]

            for sub_req_id in sub_request_ids:
                # Clean up tracking dictionaries
                tracking["concat_outputs"].pop(sub_req_id, None)
                tracking["masks"].pop(sub_req_id, None)
                tracking["num_calls"].pop(sub_req_id, None)
                tracking["timeout"].pop(sub_req_id, None)
                tracking["tool_error"].pop(sub_req_id, None)
                tracking["tool_output"].pop(sub_req_id, None)
                tracking["tool_runtime"].pop(sub_req_id, None)
                tracking["tool_called"].pop(sub_req_id, None)
                # Note: pending_tool_futures should already be cleaned by _poll_tool_futures

    def _finalize_sub_request(self, sub_request_id, request_output_for_prompts, complete_output, current_time):
        """
        Finalize a completed sub-request by moving it to request_outputs and processing if ready.

        Args:
            sub_request_id: The sub-request ID (e.g., "train_1_43039_2")
            request_output_for_prompts: RequestOutput containing prompt info
            complete_output: The CompletionOutput to add
            current_time: Current timestamp for processing

        Returns:
            Number of processed requests (0 or 1)
        """
        base_request_id = _extract_base_request_id(sub_request_id)

        # Initialize request_outputs entry if needed
        if base_request_id not in self.request_outputs:
            self.request_outputs[base_request_id] = vllm.RequestOutput(
                request_id=base_request_id,
                prompt=request_output_for_prompts.prompt,
                prompt_token_ids=request_output_for_prompts.prompt_token_ids,
                prompt_logprobs=request_output_for_prompts.prompt_logprobs,
                outputs=[],
                finished=True,
            )

        # Extract the sub-request index from the sub_request_id and set it on the CompletionOutput
        # This is needed when tools are disabled and n>1 to properly identify which sub-request
        # each output belongs to
        if not self.tools and "_" in sub_request_id:
            # Extract index from sub_request_id like "train_1_43039_2" -> 2
            parts = sub_request_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                # Create new CompletionOutput with corrected index
                complete_output = dataclasses.replace(complete_output, index=int(parts[1]))

        # Add the completion output
        self.request_outputs[base_request_id].outputs.append(complete_output)

        # Try to process and insert if we have all expected outputs
        processed = self._maybe_process_and_insert(base_request_id, self.request_outputs, self.tracking, current_time)

        return processed

    def _poll_tool_futures(self, tracking, tokenizer):
        """Poll and handle completed tool executions."""
        if not self.tools or not tracking["pending_tool_futures"]:
            return []

        dict_keys_to_delete = []
        completed_outputs = []

        for req_id, (future, last_o, last_output) in list(tracking["pending_tool_futures"].items()):
            if not future.done():
                continue

            # Tool future is done, process it
            tool_result = future.result()  # Get the tool result

            # Get sampling params from request metadata for this request
            # Extract the base request ID by removing the sub-request suffix
            base_req_id = _extract_base_request_id(req_id)
            sampling_params = self.request_metadata[base_req_id]["sampling_params"]

            last_prompt_token_ids = last_output.prompt_token_ids
            last_token_ids = last_o.token_ids
            tool_output_token_ids = tokenizer.encode(
                "<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False
            )
            tracking["timeout"][req_id] = tool_result.timeout
            tracking["tool_error"][req_id] += "" if tool_result.error is None else tool_result.error
            tracking["tool_output"][req_id] += tool_result.output
            tracking["tool_runtime"][req_id] += tool_result.runtime
            tracking["tool_called"][req_id] = True

            # Edge case 1: clip against model context length
            prompt_and_tool_output_token = last_prompt_token_ids + last_token_ids + tool_output_token_ids
            tracking["num_calls"][req_id] += 1
            excess = len(prompt_and_tool_output_token) - self.llm_engine.model_config.max_model_len
            if excess > 0:
                tool_output_token_ids = tool_output_token_ids[:-excess]
                can_make_new_request = False
            else:
                can_make_new_request = True

            # Edge case 2: clip against per-request max_tokens
            remaining = sampling_params.max_tokens - len(tracking["masks"][req_id])
            if remaining <= 0:
                tool_output_token_ids = []
            elif len(tool_output_token_ids) > remaining:
                tool_output_token_ids = tool_output_token_ids[:remaining]

            tracking["concat_outputs"][req_id].outputs[0].token_ids.extend(tool_output_token_ids)
            tracking["masks"][req_id].extend([0] * len(tool_output_token_ids))
            new_sample_tokens = sampling_params.max_tokens - len(tracking["masks"][req_id])
            can_make_new_request = can_make_new_request and new_sample_tokens > 0

            if can_make_new_request:
                new_sampling_params = sampling_params.clone()
                new_sampling_params.max_tokens = new_sample_tokens

                try:
                    self.llm_engine.add_request(
                        req_id, vllm.TokensPrompt(prompt_token_ids=prompt_and_tool_output_token), new_sampling_params
                    )
                    # Track tool continuation request as active
                    base_req_id = _extract_base_request_id(req_id)
                    if base_req_id in self.request_metadata:
                        self.vllm_active_requests[req_id] = self.request_metadata[base_req_id]["dataset_index"]

                except Exception as e:
                    # Match original ToolUseLLM behavior - just log and continue
                    self.logger.error(f"[_poll_tool_futures] Error adding request {req_id}: {e}")
            else:
                # Can't make a new request (hit limits), finalize this sub-request
                base_req_id = _extract_base_request_id(req_id)

                # Log the state before finalizing
                other_pending = [
                    other_id
                    for other_id in tracking["pending_tool_futures"]
                    if _extract_base_request_id(other_id) == base_req_id and other_id != req_id
                ]
                self.logger.info(
                    f"[_poll_tool_futures] Finalizing {req_id} (can't continue). "
                    f"Other pending tools for {base_req_id}: {other_pending}"
                )

                # Remove from pending_tool_futures BEFORE finalization to ensure consistent state
                # This prevents the cleanup logic from seeing this as a pending tool future
                tracking["pending_tool_futures"].pop(req_id, None)

                complete_output = tracking["concat_outputs"][req_id].outputs[0]
                current_time = time.time()
                self._finalize_sub_request(req_id, last_output, complete_output, current_time)
                # Don't add to dict_keys_to_delete since we already removed it
                continue
            dict_keys_to_delete.append(req_id)

        # Remove the futures we just processed; do NOT clean up metadata here.
        for req_id in dict_keys_to_delete:
            tracking["pending_tool_futures"].pop(req_id, None)

        # Now validate after ALL removals are complete
        for req_id in dict_keys_to_delete:
            base_req_id = _extract_base_request_id(req_id)
            # Check if this was re-added to vLLM or finalized
            if req_id in self.vllm_active_requests:
                context = f"After moving {req_id} from tools back to vLLM"
            else:
                context = f"After finalizing {req_id} from tools"
            self._validate_single_request_counts(base_req_id, context)

        return completed_outputs

    def init_process_group(
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
        return self.llm_engine.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray, timeout_minutes),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm_engine.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    def reset_prefix_cache(self):
        self.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm_engine.sleep(level=level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.llm_engine.wake_up(tags)

    def ready(self):
        return True

    def get_kv_cache_info(self):
        """Get KV cache max concurrency from the vLLM engine."""
        kv_cache_specs = self.llm_engine.model_executor.get_kv_cache_specs()
        kv_cache_spec = kv_cache_specs[0]
        # Group layers by their attention type (type_id) to handle models
        # with sliding attention in some layers but not others
        type_groups = defaultdict(list)
        for layer_name, layer_spec in kv_cache_spec.items():
            type_groups[layer_spec.type_id].append(layer_name)

        grouped_layer_names = list(type_groups.values())

        page_size = kv_cache_utils.get_uniform_page_size(kv_cache_spec)

        vllm_config = self.llm_engine.vllm_config
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
        max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config(
            self.llm_engine.vllm_config, kv_cache_config
        )

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
