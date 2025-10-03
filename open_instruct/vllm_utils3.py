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
import inspect
import logging
import os
import queue
import sys
import threading
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

from open_instruct import logger_utils
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.tool_utils.tools import MaxCallsExceededTool, Tool
from open_instruct.utils import ray_get_with_progress

logger = logger_utils.setup_logger(__name__)


def assert_threaded_actor(instance):
    """
    Assert that an instance's class is suitable for use in a threaded (non-async) Ray actor.

    This function performs two checks:
      1. The class must not define any `async def` methods
         (including async generators, staticmethods, or classmethods).
      2. There must not be a running asyncio event loop in the current thread.

    Args:
        instance: The instance whose class to inspect.

    Raises:
        AssertionError: If the class defines one or more async methods, or a running asyncio event loop is detected.
        RuntimeError: If an unexpected error occurs while checking for the event loop.
    """
    cls = instance.__class__
    cls_name = cls.__name__

    # --- Check for async methods defined directly on the class ---
    async_methods = []
    for name, obj in vars(cls).items():
        # unwrap @staticmethod / @classmethod to underlying function
        if isinstance(obj, (staticmethod, classmethod)):
            func = obj.__func__
        elif inspect.isfunction(obj):
            func = obj
        else:
            continue  # not a function we care about

        # catch both coroutine functions and async generators
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            async_methods.append(name)

    if async_methods:
        async_methods.sort()
        raise AssertionError(
            f"{cls_name} must not define async methods for threaded actor mode. "
            f"Found: {async_methods}. "
            f"Fix: convert these to sync methods and run async work in a background loop/thread."
        )

    # --- Check that no event loop is running in this thread ---
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        # Expected in threaded (non-async) contexts
        if "no running event loop" in str(e).lower():
            return
        # Other RuntimeError is unexpected—surface it
        raise
    else:
        # If we got a loop, we’re in an async actor (not allowed here)
        raise AssertionError(
            f"{cls_name} must run in a threaded Ray actor (no running event loop). "
            f"Detected RUNNING loop={loop!r} on thread='{threading.current_thread().name}'. "
            f"Python={sys.version.split()[0]}."
        )


async def generate_one_completion(
    llm_engine: vllm.AsyncLLMEngine, request_id: str, prompt: vllm.TokensPrompt, sampling_params: vllm.SamplingParams
) -> vllm.RequestOutput:
    """Generate a single completion from the async engine."""
    try:
        logger.debug(f"[generate_one_completion] Adding request {request_id} to engine")
        generator = llm_engine.generate(prompt, sampling_params, request_id)
        logger.debug(f"[generate_one_completion] Got generator for {request_id}, starting iteration")

        outputs = []
        async for output in generator:
            outputs.append(output)
            if output.finished:
                logger.debug(f"[generate_one_completion] Request {request_id} finished")

        assert len(outputs) == 1, f"Expected exactly 1 output, got {len(outputs)} for request {request_id}"
        return outputs[0]
    except Exception as e:
        logger.error(f"[generate_one_completion] FAILED for {request_id}: {type(e).__name__}: {e}", exc_info=True)
        raise


async def process_request_async(
    llm_engine: vllm.AsyncLLMEngine,
    sub_request_id: str,
    base_request_id: str,
    prompt: vllm.TokensPrompt,
    sampling_params: vllm.SamplingParams,
    completion_queue: queue.Queue,
    request_metadata: dict,
    active_tasks: dict,
    tools: Optional[Dict[str, Tool]] = None,
):
    """Process a single async request and push to completion queue when ready."""
    # Generate completion
    request_output = await generate_one_completion(llm_engine, sub_request_id, prompt, sampling_params)

    # Process the output
    complete_output = request_output.outputs[0]

    # Extract the j index from sub_request_id (format: base_id_j)
    j = int(sub_request_id.split("_")[-1])
    # Use dataclasses.replace to create a new CompletionOutput with the correct index
    complete_output = dataclasses.replace(complete_output, index=j)

    # Get expected_n from metadata
    expected_n = request_metadata[base_request_id]["original_sampling_params"].n

    # Clean up this sub-request from active_tasks
    active_tasks.pop(sub_request_id, None)

    # Create sub-request result with all needed metadata
    sub_request_result = {
        "base_request_id": base_request_id,
        "sub_request_id": sub_request_id,
        "j": j,
        "expected_n": expected_n,
        "request_output": vllm.RequestOutput(
            request_id=sub_request_id,
            prompt=request_output.prompt,
            prompt_token_ids=request_output.prompt_token_ids,
            prompt_logprobs=request_output.prompt_logprobs,
            outputs=[complete_output],
            finished=True,
        ),
        "tools": tools,
    }

    # Push sub-request to completion queue
    completion_queue.put(sub_request_result)


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
        assert_threaded_actor(self)

        self.logger = logger_utils.setup_logger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        self.logger.info("✓ Running in threaded Ray actor")
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.inference_batch_size = inference_batch_size
        self.inflight_updates = inflight_updates
        self.verbose = verbose
        self.request_metadata = {}
        self.completion_queue = queue.Queue()  # Thread-safe queue for completed GenerationResults
        self._future_check_interval_s = 5
        self._last_future_check = time.monotonic()

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
        logger.info("[LLMRayActor.__init__] Initialized with queues:")
        logger.info(f"  - prompt_queue: {prompt_queue}")
        logger.info(f"  - results_queue: {results_queue}")
        logger.info(f"  - eval_results_queue: {eval_results_queue}")

        # For caching should_stop status.
        self._last_should_stop_update = float("-inf")
        self._should_stop_value = False
        self._should_stop_timeout_s = 5
        self._inflight_ref = None

        self._executor = futures.ThreadPoolExecutor(max_workers=1)

        # Async tracking for request accumulation
        self.active_tasks = {}  # Track active async tasks
        self.request_outputs = {}  # Accumulate outputs until all N samples complete

        # Initialize async components with our own event loop
        self.init_complete = threading.Event()
        self.loop = None  # Will be set in thread
        self.llm_engine = None  # Will be set in thread

        # Start the async loop thread
        self.loop_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.loop_thread.start()

        # Wait for engine initialization
        if not self.init_complete.wait(timeout=120):
            raise RuntimeError("Failed to initialize AsyncLLMEngine within 120 seconds")

        self._prefetch_future = self._executor.submit(self._prefetch_worker)

    async def _init_engine_async(self):
        logger.info("Starting AsyncLLMEngine initialization...")
        self.llm_engine = vllm.AsyncLLMEngine.from_engine_args(self.engine_args, start_engine_loop=False)
        logger.info("AsyncLLMEngine created successfully")

    def _run_async_loop(self):
        """Run the async event loop in a dedicated thread."""
        # Create and set our event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Create engine from within running loop context
        # This ensures AsyncLLM.__init__ sees a running loop and starts output handler
        self.loop.run_until_complete(self._init_engine_async())

        # Signal init complete
        self.init_complete.set()

        # Keep loop running for async operations
        self.loop.run_forever()

    def _prefetch_worker(self, sleep_length_s: int = 1):
        """Background worker that prefetches requests until we have enough buffered."""
        while True:
            if self._should_stop():
                time.sleep(sleep_length_s)
                continue

            self._check_active_tasks()

            current_unfinished = len(self.active_tasks)
            if current_unfinished >= self.inference_batch_size:
                time.sleep(sleep_length_s)
                continue

            try:
                request = self.prompt_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if request is None:
                continue

            if self._should_stop():
                self.prompt_queue.put(request)
                time.sleep(sleep_length_s)
                continue

            self._add_request_sync(request)

    def get_model_dims_dict(self):
        """Get only the model dimensions as a simple dict without loading weights."""
        # In vLLM v1, use direct attributes
        model_config = self.llm_engine.model_config
        parallel_config = self.llm_engine.vllm_config.parallel_config

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

    def _check_async_loop_alive(self):
        """Check if the async event loop thread is still alive and raise if not."""
        if not self.loop_thread.is_alive():
            raise RuntimeError(
                "Async event loop thread has died. This likely means AsyncLLM initialization failed. "
                "Check earlier logs for the root cause exception."
            )

    def _should_exit(self) -> bool:
        """Determine if the processing loop should exit.

        Returns:
            bool: True if the loop should exit, False otherwise.
        """
        # Check stop condition first (cheapest check)
        stop_requested = self._should_stop()

        # Case 1: inflight_updates enabled and stop requested - exit immediately
        if self.inflight_updates and stop_requested:
            return True

        # Now check for pending work (only if needed)
        if stop_requested:
            # Need to check if we have pending work
            active_tasks = len(self.active_tasks)
            has_incomplete = len(self.request_outputs) > 0

            # Case 2: stop requested and no pending work - exit
            if active_tasks == 0 and not has_incomplete:
                return True
            # Otherwise, we have pending work and should continue

        return False

    def _add_request_sync(self, request: PromptRequest):
        """Add a request by spawning async tasks."""
        self._check_async_loop_alive()

        request_id = make_request_id(request)
        logger.info(
            f"[_add_request_sync] 🎯 Adding request: request_id={request_id}, is_eval={request.is_eval}, n={request.generation_config.n}"
        )
        logger.info(
            f"[_add_request_sync] Loop thread alive: {self.loop_thread.is_alive()}, loop: {self.loop}, loop_running: {self.loop.is_running()}"
        )

        sampling_params = request.generation_config.clone()
        sampling_params.n = 1  # Use n=1 for sub-requests

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
            sub_sampling_params = sampling_params.clone()
            if request.generation_config.seed is not None:
                sub_sampling_params.seed = request.generation_config.seed + j
            sub_request_id = f"{request_id}_{j}"

            # Spawn async task in our event loop
            future = asyncio.run_coroutine_threadsafe(
                process_request_async(
                    self.llm_engine,
                    sub_request_id,
                    request_id,
                    tokens_prompt,
                    sub_sampling_params,
                    self.completion_queue,
                    self.request_metadata,
                    self.active_tasks,
                    self.tools,
                ),
                self.loop,
            )
            # Track the future
            self.active_tasks[sub_request_id] = future
            logger.info(
                f"[_add_request_sync] Scheduled task for {sub_request_id}, future={future}, done={future.done()}"
            )

    def _insert_result_to_queue(self, result, is_eval: bool):
        """Insert result into the appropriate queue with blocking put."""
        queue_type = "eval" if is_eval else "train"
        results_queue = self.eval_results_queue if is_eval else self.results_queue
        logger.info(
            f"[_insert_result_to_queue] 📤 Attempting to put result into {queue_type} queue: {result.dataset_index if hasattr(result, 'dataset_index') else 'unknown'}"
        )
        logger.info(f"[_insert_result_to_queue] Queue object: {results_queue}")
        logger.info(f"[_insert_result_to_queue] Result type: {type(result)}")
        try:
            results_queue.put(result)
            logger.info(f"[_insert_result_to_queue] ✅ Successfully put result into {queue_type} queue")
        except Exception as e:
            logger.error(f"[_insert_result_to_queue] ❌ Failed to put result into {queue_type} queue: {e}")
            raise

    def process_from_queue(self, timeout: float = 60.0):
        """Run generation loop pulling from completion queue.

        Returns:
            int: Number of requests processed
        """
        self._check_async_loop_alive()
        total_processed = 0

        while not self._should_exit():
            if self._prefetch_future.done():
                self._prefetch_future.result()

            try:
                self._check_active_tasks()
                sub_request = self.completion_queue.get(timeout=1.0)

                # Check if it's a sub-request (dict) or already processed result (tuple)
                if isinstance(sub_request, dict):
                    # This is a sub-request that needs aggregation
                    base_request_id = sub_request["base_request_id"]
                    expected_n = sub_request["expected_n"]

                    # Initialize accumulator if needed
                    if base_request_id not in self.request_outputs:
                        self.request_outputs[base_request_id] = {
                            "outputs": [],
                            "expected_n": expected_n,
                            "tools": sub_request["tools"],
                        }

                    # Add this sub-request output
                    self.request_outputs[base_request_id]["outputs"].append(sub_request["request_output"])

                    logger.info(
                        f"[process_from_queue] Accumulated {len(self.request_outputs[base_request_id]['outputs'])}/{expected_n} for {base_request_id}"
                    )

                    # Check if all sub-requests are complete
                    if len(self.request_outputs[base_request_id]["outputs"]) == expected_n:
                        logger.info(
                            f"[process_from_queue] All sub-requests complete for {base_request_id}, aggregating"
                        )

                        # Sort outputs by j index (extracted from request_id)
                        outputs = self.request_outputs[base_request_id]["outputs"]
                        ordered_outs = sorted(outputs, key=lambda x: int(x.request_id.split("_")[-1]))

                        # Process the completed request
                        current_time = time.perf_counter()
                        result, is_eval = process_completed_request(
                            base_request_id,
                            ordered_outs,
                            {},  # tracking dict (empty for now)
                            current_time,
                            self.request_outputs[base_request_id]["tools"],
                            self.request_metadata,
                        )

                        # Clean up
                        self.request_outputs.pop(base_request_id)
                        self.request_metadata.pop(base_request_id, None)

                        # Insert result to appropriate queue
                        self._insert_result_to_queue(result, is_eval=is_eval)
                        total_processed += 1
                else:
                    # Legacy path if something directly puts a tuple result
                    result, is_eval = sub_request
                    self._insert_result_to_queue(result, is_eval=is_eval)
                    total_processed += 1

                # Log memory stats after processing
                self.logger.info(
                    f"[Memory Stats] Dicts: metadata={len(self.request_metadata)}, "
                    f"outputs={len(self.request_outputs)}, tasks={len(self.active_tasks)}"
                )

            except queue.Empty:
                pass
        return total_processed

    def _check_active_tasks(self):
        """Crash the actor immediately if any async task failed."""
        now = time.monotonic()
        if (now - self._last_future_check) < self._future_check_interval_s:
            return

        for request_id, future in list(self.active_tasks.items()):
            if future.cancelled():
                raise RuntimeError(f"Async generation future for {request_id} was unexpectedly cancelled")

            if future.done():
                future.result()

        self._last_future_check = now

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
        future = asyncio.run_coroutine_threadsafe(
            self.llm_engine.engine_core.collective_rpc_async(
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
        return future.result()

    def update_weight(self, name, dtype, shape, empty_cache=False):
        future = asyncio.run_coroutine_threadsafe(
            self.llm_engine.engine_core.collective_rpc_async("update_weight", args=(name, dtype, shape, empty_cache)),
            self.loop,
        )
        return future.result()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        future = asyncio.run_coroutine_threadsafe(
            self.llm_engine.engine_core.collective_rpc_async(
                "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
            ),
            self.loop,
        )
        return future.result()

    def reset_prefix_cache(self):
        future = asyncio.run_coroutine_threadsafe(self.llm_engine.reset_prefix_cache(), self.loop)
        return future.result()

    def ready(self):
        # Engine and prefetch are already initialized in __init__
        return True

    def get_kv_cache_info(self):
        """Get KV cache max concurrency from the vLLM engine."""
        # In vLLM v1, AsyncLLM has direct attributes model_config and vllm_config
        cache_config = self.llm_engine.vllm_config.cache_config
        model_config = self.llm_engine.model_config

        # Use the same calculation as vLLM's executor_base.py
        # Reference: https://github.com/vllm-project/vllm/blob/b6553be1bc75f046b00046a4ad7576364d03c835/vllm/executor/executor_base.py#L119-L120
        retries = 5
        for attempt in range(retries):
            num_gpu_blocks = cache_config.num_gpu_blocks
            block_size = cache_config.block_size
            max_model_len = model_config.max_model_len

            if num_gpu_blocks is not None and num_gpu_blocks != 0:
                # Calculate max concurrency using vLLM's formula
                max_concurrency = (num_gpu_blocks * block_size) / max_model_len
                logger.info(f"Calculated max_concurrency: {max_concurrency}")
                return int(max_concurrency)

            # Not initialized yet; wait a bit
            time.sleep(0.2)

        logger.warning("num_gpu_blocks not initialized after retries, returning default value 1")
        return 1


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
                runtime_env=ray.runtime_env.RuntimeEnv(env_vars={"TORCH_CUDA_ARCH_LIST": get_cuda_arch_list()}),
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
