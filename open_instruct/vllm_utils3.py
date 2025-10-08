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

import dataclasses
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
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
from open_instruct.tool_utils.tools import MaxCallsExceededTool, Tool
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
        tracking["concat_outputs"][output.request_id].outputs[0].token_ids.extend(o.token_ids)
    else:
        tracking["concat_outputs"][output.request_id] = output

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
        training_step=metadata["training_step"],
        token_statistics=TokenStatistics(
            num_prompt_tokens=metadata["prompt_tokens"],
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
    tools: Dict[str, Tool],
    request_metadata: dict,
    vllm_active_requests: dict,
) -> int:
    """Add a request to the LLM engine."""
    request_id = make_request_id(request)
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
    for j in range(request.generation_config.n):
        sub_sampling_params = sampling_params.clone()  # Already has n=1
        if request.generation_config.seed is not None:
            sub_sampling_params.seed = request.generation_config.seed + j
        sub_request_id = f"{request_id}_{j}"
        llm_engine.add_request(sub_request_id, tokens_prompt, sub_sampling_params)
        vllm_active_requests.add(sub_request_id)

    return request.generation_config.n


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
        self.vllm_active_requests = set()  # Track all requests currently in vLLM

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

        engine_args = vllm.EngineArgs(*args, **kwargs)
        # Log stats causes a crash in the engine at assert outputs.scheduler_stats is not None when we call step() and there is nothing to step.
        engine_args.disable_log_stats = True

        # Cascade attention has known performance issues: https://github.com/vllm-project/vllm/issues/17652
        engine_args.disable_cascade_attn = True

        self.llm_engine = vllm.LLMEngine.from_engine_args(engine_args)

        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.actor_manager = actor_manager

        # For caching should_stop status.
        self._last_should_stop_update = float("-inf")
        self._should_stop_value = False
        self._should_stop_timeout_s = 5

        # Initialize instance variables before starting threads
        self.tracking = _init_tool_tracking()
        self.request_outputs = {}
        self._threads_started = threading.Event()

        # Start background threads
        self._executor = futures.ThreadPoolExecutor(max_workers=2)
        self._prefetch_future = self._executor.submit(self._prefetch_worker)
        self._process_future = self._executor.submit(self._process_from_queue)

    def get_model_dims_dict(self):
        """Get only the model dimensions as a simple dict without loading weights."""
        model_config = self.llm_engine.model_config
        parallel_config = self.llm_engine.vllm_config.parallel_config

        # Extract only the necessary dimensions as simple Python types
        hidden_size = model_config.get_hidden_size()
        intermediate_size = getattr(model_config.hf_text_config, "intermediate_size", 4 * hidden_size)

        return {
            "num_layers": model_config.get_num_layers(parallel_config),
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "vocab_size": model_config.get_vocab_size(),
            "num_attn_heads": model_config.get_num_attention_heads(parallel_config),
            "num_kv_heads": model_config.get_num_kv_heads(parallel_config),
        }

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

    def _prefetch_worker(self, sleep_length_s: int = 1):
        """Background worker that prefetches requests until we have enough buffered."""
        self._threads_started.set()
        while True:
            if not self.inflight_updates and self._should_stop():
                time.sleep(sleep_length_s)
                continue
            current_unfinished = self.llm_engine.get_num_unfinished_requests()
            if current_unfinished >= self.inference_batch_size:
                time.sleep(sleep_length_s)
                continue
            try:
                request = self.prompt_queue.get(timeout=0.1)
                add_request(
                    request,
                    self.llm_engine,
                    self.tools,
                    request_metadata=self.request_metadata,
                    vllm_active_requests=self.vllm_active_requests,
                )
            except queue.Empty:
                continue

    def _insert_result_to_queue(self, result, is_eval: bool):
        """Insert result into the appropriate queue with blocking put."""
        results_queue = self.eval_results_queue if is_eval else self.results_queue
        results_queue.put(result)

    def _process_from_queue(self, timeout: float = 60.0):
        """Run generation loop using LLMEngine directly, with optional tool support.

        Runs continuously in a background thread, processing requests from the engine.

        Returns:
            int: Number of requests processed
        """
        total_processed = 0
        iteration_count = 0

        while True:
            iteration_count += 1

            # Health check: ensure prefetch worker is alive. This will raise if it has crashed.
            if self._prefetch_future.done():
                self._prefetch_future.result()

            self._poll_tool_futures(self.tracking, self.llm_engine.tokenizer)
            current_time = time.perf_counter()
            if self.llm_engine.has_unfinished_requests():
                for output in [o for o in self.llm_engine.step() if o.finished]:
                    # Fix the index field for all sub-requests
                    # When we have n>1, we create sub-requests with IDs like
                    # train_3_12_0, train_3_12_1, etc. But vLLM creates CompletionOutputs with index=0
                    # for all of them (since each sub-request has n=1). We need to fix this.
                    # Extract the actual index from the sub-request ID
                    parts = output.request_id.rsplit("_", 1)
                    assert len(parts) == 2 and parts[1].isdigit(), (
                        f"Wrong request id format ({output.request_id}), should be request_id _ sub_request_index"
                    )

                    # Fix the index on the CompletionOutput
                    correct_index = int(parts[1])
                    output.outputs = [dataclasses.replace(o, index=correct_index) for o in output.outputs]
                    base_req_id = _extract_base_request_id(output.request_id)
                    result = _handle_output(
                        output,
                        self.tools,
                        self.tracking,
                        self.request_metadata[base_req_id]["sampling_params"],
                        self.max_tool_calls,
                        self.executor,
                    )

                    # Result is None when we do more tool processing.
                    if result is None:
                        # Request went to tools - remove from vllm_active_requests since it's no longer in vLLM
                        self.vllm_active_requests.discard(output.request_id)
                    else:
                        # Sub-request is done (no more tool calls)
                        if output.request_id in self.tracking["concat_outputs"]:
                            complete_output = self.tracking["concat_outputs"][output.request_id].outputs[0]
                        else:
                            complete_output = result.outputs[0]

                        # Remove from vllm_active_requests BEFORE calling _finalize_sub_request
                        # to avoid deadlock in _maybe_process_and_insert
                        self.vllm_active_requests.discard(output.request_id)
                        total_processed += self._finalize_sub_request(
                            output.request_id, output, complete_output, current_time
                        )
            if self.llm_engine.get_num_unfinished_requests() == 0:
                time.sleep(1)

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

        # Check if we have the base request in request_outputs
        if request_id not in request_outputs:
            return 0

        available_outputs = request_outputs[request_id].outputs
        if len(available_outputs) < expected_n:
            return 0

        needed_ids = [f"{request_id}_{j}" for j in range(expected_n)]
        active_sub_requests = [sub_id for sub_id in needed_ids if sub_id in self.vllm_active_requests]
        if active_sub_requests:
            return 0

        has_pending_tools = any(sub_id in tracking.get("pending_tool_futures", {}) for sub_id in needed_ids)
        if has_pending_tools:
            return 0

        # At this point we have all outputs ready. Build ordered outputs for processing.
        # First organize available_outputs into a dictionary for O(1) lookup
        outputs_by_index = {o.index: o for o in available_outputs if hasattr(o, "index")}

        # Verify we have all required outputs before proceeding
        if len(outputs_by_index) != expected_n or any(j not in outputs_by_index for j in range(expected_n)):
            self.logger.warning(
                f"Incomplete or malformed outputs for {request_id}. "
                f"Expected {expected_n} samples, got indices {sorted(outputs_by_index.keys())}. Skipping."
            )
            return 0

        ordered_outs: List[vllm.RequestOutput] = []
        for j in range(expected_n):
            # Create a RequestOutput wrapper for each CompletionOutput
            ordered_outs.append(
                vllm.RequestOutput(
                    request_id=f"{request_id}_{j}",
                    prompt=request_outputs[request_id].prompt,
                    prompt_token_ids=request_outputs[request_id].prompt_token_ids,
                    prompt_logprobs=request_outputs[request_id].prompt_logprobs,
                    outputs=[outputs_by_index[j]],
                    finished=True,
                )
            )

        # Remove the base entry from request_outputs to prevent growth.
        request_outputs.pop(request_id, None)
        result, is_eval = process_completed_request(
            request_id, ordered_outs, tracking, current_time, self.tools, self.request_metadata
        )
        self._insert_result_to_queue(result, is_eval=is_eval)
        self._cleanup_request_data(request_id, tracking)
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

    def _has_active_sub_requests_for_base_id(self, base_request_id: str) -> bool:
        """Check if there are any active sub-requests in vLLM for a given base request ID."""
        # Check if any active request IDs belong to our base request
        for req_id in self.vllm_active_requests:
            if _extract_base_request_id(req_id) == base_request_id:
                return True
        return False

    def _cleanup_request_data(self, request_id: str, tracking: Dict[str, Any]):
        """Clean up metadata and tracking data for a completed request."""
        # Check if there are still pending tool futures for this request
        if self._has_pending_tool_futures_for_request(request_id, tracking):
            # Don't clean up metadata yet - tool futures still need it
            return

        # Check if there are still active sub-requests in vLLM for this base request
        if self._has_active_sub_requests_for_base_id(request_id):
            # Don't clean up metadata yet - active requests still need it
            return

        # Remove request metadata only after both conditions are met:
        # 1. No pending tool futures for this request
        # 2. No active sub-requests in vLLM for this base request
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

        # Extract the sub-request index from the sub_request_id and set it on the CompletionOutput
        # This is needed to properly identify which sub-request each output belongs to.
        # MUST be done BEFORE adding to request_outputs so that
        # _maybe_process_and_insert can find the index field when checking completeness.
        if "_" in sub_request_id:
            # Extract index from sub_request_id like "train_1_43039_2" -> 2
            parts = sub_request_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                # Create new CompletionOutput with corrected index
                complete_output = dataclasses.replace(complete_output, index=int(parts[1]))

        # If tools are enabled, attach tool metadata to the output
        if self.tools:
            # Set tool metadata attributes on the output
            setattr(
                complete_output,
                "mask",
                self.tracking["masks"].get(sub_request_id, [1] * len(complete_output.token_ids)),
            )
            setattr(complete_output, "num_calls", self.tracking["num_calls"].get(sub_request_id, 0))
            setattr(complete_output, "timeout", self.tracking["timeout"].get(sub_request_id, False))
            setattr(complete_output, "tool_error", self.tracking["tool_error"].get(sub_request_id, ""))
            setattr(complete_output, "tool_output", self.tracking["tool_output"].get(sub_request_id, ""))
            setattr(complete_output, "tool_runtime", self.tracking["tool_runtime"].get(sub_request_id, 0.0))
            setattr(complete_output, "tool_called", self.tracking["tool_called"].get(sub_request_id, False))

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

        # Add the completion output (with index field already set if needed)
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
                        self.vllm_active_requests.add(req_id)

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
                current_time = time.perf_counter()
                self._finalize_sub_request(req_id, last_output, complete_output, current_time)
                # Don't add to dict_keys_to_delete since we already removed it
                continue
            dict_keys_to_delete.append(req_id)

        # Remove the futures we just processed; do NOT clean up metadata here.
        for req_id in dict_keys_to_delete:
            tracking["pending_tool_futures"].pop(req_id, None)

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

    def _prepare_weight_update(self, name: str, dtype: str) -> None:
        # First, drain all the requests when appropriate:
        while not self.inflight_updates:
            pending_tools = len(self.tracking["pending_tool_futures"])
            unfinished = self.llm_engine.get_num_unfinished_requests()

            if pending_tools == 0 and unfinished == 0:
                break

            time.sleep(sleep_s)
        # Then, check that the dtypes match.
        expected_dtype = str(self.llm_engine.model_config.dtype)
        assert dtype == expected_dtype, f"Mismatched dtype for {name}: received {dtype!r}, expected {expected_dtype!r}"

    def update_weight(self, name: str, dtype: str, shape: Tuple[int, ...], empty_cache: bool = False) -> None:
        self._prepare_weight_update(name, dtype)
        return self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(
        self, name: str, dtype: str, shape: Tuple[int, ...], ipc_handles: List[Any], empty_cache: bool = False
    ) -> None:
        self._prepare_weight_update(name, dtype)
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
        self._threads_started.wait(timeout=30)
        return True

    def check_background_threads(self):
        if self._prefetch_future.done():
            self._prefetch_future.result()
        if self._process_future.done():
            self._process_future.result()

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

    ray_get_with_progress([engine.ready.remote() for engine in vllm_engines], "Initializing vLLM engines", timeout=300)

    return vllm_engines
