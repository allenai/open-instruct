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
import logging
import os
import queue
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
from tqdm import tqdm

from open_instruct.queue_types import GenerationResult, RequestInfo
from open_instruct.tool_utils.tool_vllm import MaxCallsExceededTool, Tool
from open_instruct.utils import ray_get_with_progress

logger = logging.getLogger(__name__)


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

    assert len(output.outputs) <= 1  # In tool mode, sampling_params.n == 1
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


def _process_outputs(
    outputs: List[vllm.RequestOutput], dataset_index: Optional[List[int]] = None
) -> "GenerationResult":
    """Process vLLM RequestOutputs into GenerationResult format."""
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
    )

    return result


def _process_outputs_with_tools(
    outputs: List[vllm.RequestOutput], dataset_index: Optional[List[int]] = None
) -> "GenerationResult":
    """Process vLLM RequestOutputs into GenerationResult format with tool information."""
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
    )

    return result


def _finalize_outputs(outputs, tracking, dataset_index, tools):
    """Prepare final outputs based on whether tools were used."""
    if not tools:
        outputs.sort(key=lambda x: int(x.request_id.split("_")[-1]))
        return _process_outputs(outputs, dataset_index=dataset_index)

    # Tool mode: add metadata and merge completions
    for req_id in tracking["masks"]:
        assert req_id in tracking["concat_outputs"], f"req_id {req_id} not in concat_outputs!"
        output = tracking["concat_outputs"][req_id].outputs[0]
        setattr(output, "mask", tracking["masks"][req_id])
        setattr(output, "num_calls", tracking["num_calls"][req_id])
        setattr(output, "timeout", tracking["timeout"][req_id])
        setattr(output, "tool_error", tracking["tool_error"][req_id])
        setattr(output, "tool_output", tracking["tool_output"][req_id])
        setattr(output, "tool_runtime", tracking["tool_runtime"][req_id])
        setattr(output, "tool_called", tracking["tool_called"][req_id])

    # Merge n completions into the same outputs
    merged_outputs = {}
    for req_id in tracking["concat_outputs"]:
        real_req_id, _ = req_id.split("-")
        if real_req_id not in merged_outputs:
            merged_outputs[real_req_id] = tracking["concat_outputs"][req_id]
        else:
            merged_outputs[real_req_id].outputs.append(tracking["concat_outputs"][req_id].outputs[0])

    final_outputs = sorted(
        merged_outputs.values(), key=lambda x: (int(x.request_id.split("-")[0]), int(x.request_id.split("-")[1]))
    )

    return _process_outputs_with_tools(final_outputs, dataset_index=dataset_index)


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


@ray.remote
class ActorManager:
    """Centralized manager for controlling evaluation and weight updates across all LLMRayActors."""

    def __init__(self):
        self._should_stop = False

    def set_should_stop(self, should_stop: bool):
        """Set whether actors should stop processing."""
        self._should_stop = should_stop

    def should_stop(self) -> bool:
        """Check if actors should stop processing."""
        return self._should_stop


def add_request(
    prompt_queue: queue.Queue, llm_engine: vllm.LLMEngine, tools, timeout: float = 60.0, request_metadata: dict = None
):
    """Get a request from the queue and add it to the LLM engine."""
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(f"[add_request] Attempting to get request from queue with timeout={timeout}")
    try:
        request = prompt_queue.get(timeout=timeout)
        logger.info(
            f"[add_request] Got request from queue: training_step={request.training_step}, "
            f"dataset_index={request.dataset_index}, is_eval={request.is_eval}"
        )
    except queue.Empty:
        logger.debug(f"[add_request] Queue empty after timeout={timeout}")
        return

    prompt = request.prompt  # Single prompt from PromptRequest
    sampling_params = request.generation_config
    training_step = request.training_step if request.training_step is not None else 0
    dataset_index = request.dataset_index if request.dataset_index is not None else 0

    logger.info(f"[add_request] Processing prompt with dataset_index={dataset_index}, n={sampling_params.n}")

    # Store metadata if provided
    # Add prefix to prevent collision between train and eval requests with same dataset_index
    prefix = "eval" if request.is_eval else "train"
    base_request_id = f"{prefix}_{training_step}_{dataset_index}"
    if request_metadata is not None:
        request_metadata[base_request_id] = {
            "is_eval": request.is_eval,
            "dataset_index": dataset_index,
            "training_step": training_step,
        }

    if tools:
        # Need n=1 for individual tool tracking
        original_n = sampling_params.n
        sampling_params = copy.deepcopy(sampling_params)
        sampling_params.n = 1
        # Create individual requests for each sample when using tools
        for j in range(original_n):
            request_id = f"{base_request_id}-{j}"
            tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt)
            llm_engine.add_request(request_id, tokens_prompt, sampling_params)
    else:
        # Standard request format for non-tool mode
        request_id = base_request_id
        tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt)
        llm_engine.add_request(request_id, tokens_prompt, sampling_params)


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
        **kwargs,
    ):
        import logging

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("[LLMRayActor.__init__] Starting initialization")

        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.logger.info(f"[LLMRayActor.__init__] Tools configured: {len(self.tools)} tools")

        if self.tools:
            self.executor = ThreadPoolExecutor(max_workers=20)
        else:
            self.executor = None

        self.logger.info("[LLMRayActor.__init__] Processing noset_visible_devices")
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
            print(f"creating LLM with bundle_indices={bundle_indices}")

        self.logger.info("[LLMRayActor.__init__] About to create LLM engine from args")
        self.llm_engine = vllm.LLMEngine.from_engine_args(vllm.EngineArgs(*args, **kwargs))
        self.logger.info("[LLMRayActor.__init__] LLM engine created successfully")

        self.logger.info("[LLMRayActor.__init__] Setting up queues and other attributes")
        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.logger = logging.getLogger(__name__)
        self.actor_manager = actor_manager
        if inference_batch_size is None:
            raise ValueError("inference_batch_size must be specified.")
        self.inference_batch_size = inference_batch_size
        self.logger.info(f"[LLMRayActor.__init__] Initialization complete, batch_size={self.inference_batch_size}")

    def process_from_queue(self, timeout: float = 60.0):
        """Run generation loop using LLMEngine directly, with optional tool support.

        Returns:
            int: Number of requests processed.
        """
        self.logger.info(
            f"[LLMRayActor.process_from_queue] === STARTING === timeout={timeout}, batch_size={self.inference_batch_size}"
        )
        self.logger.info(
            f"[LLMRayActor.process_from_queue] Queue types: prompt_queue={type(self.prompt_queue)}, results_queue={type(self.results_queue)}"
        )

        num_processed = 0
        iteration = 0
        request_metadata = {}  # Track request metadata by ID

        # Initialize tracking for tools if needed
        self.logger.info(f"[LLMRayActor.process_from_queue] Initializing tool tracking, tools={len(self.tools)} tools")
        if self.tools:
            tracking = _init_tool_tracking()
            sampling_params = None  # Will be set from first request
            tokenizer = self.llm_engine.tokenizer
        else:
            tracking = None
            sampling_params = None
            tokenizer = None
        self.logger.info("[LLMRayActor.process_from_queue] Tool tracking initialized")

        # Pre-fill with initial requests
        self.logger.info(
            f"[LLMRayActor.process_from_queue] === PRE-FILLING === with {self.inference_batch_size} initial requests"
        )
        prefill_pbar = tqdm(
            range(self.inference_batch_size),
            desc="[vLLM] Pre-filling requests",
            total=self.inference_batch_size,
            leave=False,
        )
        for i in prefill_pbar:
            self.logger.info(
                f"[LLMRayActor.process_from_queue] Pre-fill: Calling add_request {i + 1}/{self.inference_batch_size}"
            )
            add_request(
                self.prompt_queue, self.llm_engine, self.tools, timeout=timeout, request_metadata=request_metadata
            )
            self.logger.info(f"[LLMRayActor.process_from_queue] Pre-fill: add_request {i + 1} returned")
            prefill_pbar.set_postfix({"loaded": i + 1})
        prefill_pbar.close()
        self.logger.info("[LLMRayActor.process_from_queue] === PRE-FILLING COMPLETE ===")

        self.logger.info("[LLMRayActor.process_from_queue] === MAIN LOOP STARTING ===")
        while True:
            iteration += 1
            if iteration % 100 == 0:
                self.logger.info(f"[LLMRayActor.process_from_queue] >>> ITERATION {iteration} START")

            # Non-blocking check for should_stop using ray.wait
            self.logger.debug(
                f"[LLMRayActor.process_from_queue] Iteration {iteration}: Checking actor_manager.should_stop"
            )
            should_stop_ref = self.actor_manager.should_stop.remote()
            ready_refs, _ = ray.wait([should_stop_ref], timeout=0.1)
            if ready_refs and ray.get(ready_refs[0]):
                self.logger.info(
                    "[LLMRayActor.process_from_queue] Actor manager signaled to stop. Exiting generation loop."
                )
                return num_processed
            self.logger.debug(
                f"[LLMRayActor.process_from_queue] Iteration {iteration}: should_stop check complete, continuing"
            )

            # Process a step and get completed outputs
            if iteration % 100 == 0:
                self.logger.info(f"[LLMRayActor.process_from_queue] Iteration {iteration}: About to call _step_engine")
            outputs = self._step_engine(tracking, sampling_params, tokenizer)
            if iteration % 100 == 0:
                self.logger.info(
                    f"[LLMRayActor.process_from_queue] Iteration {iteration}: _step_engine returned {len(outputs)} outputs"
                )

            # Process completed outputs
            for output in outputs:
                # Get request metadata
                base_request_id = output.request_id.split("-")[0] if "-" in output.request_id else output.request_id
                is_eval = request_metadata[base_request_id]["is_eval"]
                dataset_index = request_metadata[base_request_id]["dataset_index"]

                if not self.tools:
                    # Process complete output with all N completions
                    num_processed += 1
                    result = _process_outputs(
                        [output], dataset_index=[dataset_index] if dataset_index is not None else None
                    )

                    self.logger.info(
                        f"[vLLM] Processed prompt #{num_processed}, dataset_index={dataset_index}, "
                        f"n_completions={len(output.outputs)}, is_eval={is_eval}"
                    )

                    # Put result in appropriate queue
                    try:
                        queue_to_use = self.eval_results_queue if is_eval else self.results_queue
                        queue_to_use.put(result, timeout=10)
                        self.logger.info(f"[vLLM] Result for dataset_index={dataset_index} put in queue")
                    except queue.Full:
                        self.logger.warning("Results queue is full, discarding result.")
                else:
                    # Tool mode - process as before
                    num_processed += 1
                    result = _finalize_outputs([output], tracking, None, self.tools)

                    self.logger.info(
                        f"[vLLM] Processed prompt #{num_processed} with tools, dataset_index={dataset_index}, is_eval={is_eval}"
                    )

                    # Put result in appropriate queue
                    try:
                        queue_to_use = self.eval_results_queue if is_eval else self.results_queue
                        queue_to_use.put(result, timeout=10)
                        self.logger.info(f"[vLLM] Result for dataset_index={dataset_index} put in queue")
                    except queue.Full:
                        self.logger.warning("Results queue is full, discarding result.")

                # Clean up metadata for completed request
                if base_request_id in request_metadata:
                    del request_metadata[base_request_id]

                # Add new request to replace completed one
                add_request(
                    self.prompt_queue, self.llm_engine, self.tools, timeout=0.1, request_metadata=request_metadata
                )

            # Check termination condition
            pending_count = len(tracking["pending_tool_futures"]) if tracking else 0
            if not self.llm_engine.has_unfinished_requests() and pending_count == 0:
                self.logger.info(
                    f"[LLMRayActor] Terminating after {iteration} iterations, processed {num_processed} requests"
                )
                break

        return num_processed

    def _step_engine(self, tracking, sampling_params, tokenizer):
        """Unified processing for both tool and non-tool generation.

        Returns:
            List of completed outputs.
        """
        if tracking and tracking.get("pending_tool_futures"):
            self._poll_tool_futures(tracking, sampling_params, tokenizer)

        # Process engine steps - ONLY if there are unfinished requests (matching ToolUseLLM)
        outputs = []
        if self.llm_engine.has_unfinished_requests():
            step_outputs = list(self.llm_engine.step())
            if step_outputs:
                outputs_per_request = [len(o.outputs) for o in step_outputs]
                self.logger.info(
                    f"[LLMRayActor._step_engine] engine.step() returned {len(step_outputs)} step_outputs, outputs per request: {outputs_per_request}"
                )

            for output in step_outputs:
                if output.finished:
                    result = _handle_output(
                        output, self.tools, tracking, sampling_params, self.max_tool_calls, self.executor
                    )
                    if result is not None:
                        outputs.append(result)
                        self.logger.info(f"[LLMRayActor] Added output {output.request_id} to results")

        return outputs

    def _poll_tool_futures(self, tracking, sampling_params, tokenizer):
        """Poll and handle completed tool executions."""
        if not self.tools or not tracking["pending_tool_futures"]:
            return

        dict_keys_to_delete = []

        for req_id, (future, last_o, last_output) in tracking["pending_tool_futures"].items():
            if not future.done():
                continue

            # Tool future is done, process it
            tool_result = future.result()  # Get the tool result

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
                new_sampling_params = copy.deepcopy(sampling_params)
                new_sampling_params.max_tokens = new_sample_tokens

                try:
                    self.llm_engine.add_request(
                        req_id, vllm.TokensPrompt(prompt_token_ids=prompt_and_tool_output_token), new_sampling_params
                    )
                except Exception as e:
                    # Match original ToolUseLLM behavior - just log and continue
                    self.logger.error(f"[_poll_tool_futures] Error adding request {req_id}: {e}")

            dict_keys_to_delete.append(req_id)

        for req_id in dict_keys_to_delete:
            if req_id in tracking["pending_tool_futures"]:
                del tracking["pending_tool_futures"][req_id]

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
    print(f"Detected CUDA compute capabilities: {cuda_capabilities}, setting TORCH_CUDA_ARCH_LIST={cuda_arch_list}")
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
    inference_batch_size: Optional[int] = None,
    pg: Optional[ray.util.placement_group] = None,
    vllm_enable_sleep=False,
    tools: Optional[Dict[str, Tool]] = None,
    max_tool_calls: List[int] = [5],
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
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

    print(f"num_gpus: {num_gpus}")

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size,
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
    print(f"output: {output}")
