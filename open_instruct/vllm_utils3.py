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
import logging
import os
import queue
import re
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import ray
import requests
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

from open_instruct.utils import ray_get_with_progress

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ToolOutput:
    output: str
    called: bool
    error: str
    timeout: bool
    runtime: float
    start_str: str = "<output>\n"
    end_str: str = "\n</output>"


class Tool:
    def __init__(self, start_str: str, end_str: str):
        self.start_str = start_str
        self.end_str = end_str

    def __call__(self, prompt: str) -> ToolOutput:
        raise NotImplementedError("Subclasses must implement this method")


class MaxCallsExceededTool(Tool):
    def __call__(self, prompt: str) -> ToolOutput:
        return ToolOutput(output="Max tool calls exceeded.", called=False, error="", timeout=False, runtime=0)


class PythonCodeTool(Tool):
    """Tool for executing Python code via an API endpoint."""

    def __init__(self, api_endpoint: str, *args, **kwargs):
        self.api_endpoint = api_endpoint
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        re_str = r"(?s)(?<!`)<tool>\s*(.*?)\s*</tool>"
        re_str = re_str.replace("<tool>", "<code>").replace("</tool>", "</code>")

        code_blocks = re.findall(re_str, prompt, re.DOTALL)
        all_outputs = []
        timeout = False
        error = ""
        if len(code_blocks) == 0:
            return ToolOutput(output="", called=False, error="", timeout=False, runtime=0)

        # Only execute the last code block
        code = code_blocks[-1]

        # Define timeout in seconds
        timeout_seconds = 3
        start_time = time.time()
        try:
            # Call the FastAPI endpoint to execute the code with client-side timeout
            response = requests.post(
                self.api_endpoint, json={"code": code, "timeout": timeout_seconds}, timeout=timeout_seconds
            )

            # Parse the response
            result = response.json()

            # Process the API response
            output = result["output"]
            error = result.get("error") or ""

            all_outputs.append(output)
            if len(error) > 0:
                all_outputs.append("\n" + error)

        except requests.Timeout:
            # Handle client-side timeout specifically
            all_outputs.append(f"Timeout after {timeout_seconds} seconds")
            timeout = True

        except Exception as e:
            # Capture any other exceptions that occur during the API call
            error_message = f"Error calling API: {str(e)}\n"
            error_traceback = traceback.format_exc()
            all_outputs.append(error_message + error_traceback)

        # Return all captured outputs as a single string
        return ToolOutput(
            output="\n".join(all_outputs), called=True, error=error, timeout=timeout, runtime=time.time() - start_time
        )


@dataclasses.dataclass
class RequestInfo:
    """Container for tool usage information."""

    num_calls: List[int]
    timeouts: List[int]
    tool_errors: List[str]
    tool_outputs: List[str]
    tool_runtimes: List[float]
    tool_calleds: List[bool]


@dataclasses.dataclass
class GenerationResult:
    """Container for generation results from vLLM."""

    responses: List[List[int]]
    finish_reasons: List[str]
    masks: List[List[int]]
    request_info: RequestInfo
    dataset_index: Optional[List[int]] = None


@dataclasses.dataclass
class PromptRequest:
    """Container for prompt requests to vLLM."""

    prompts: List[List[int]]
    sampling_params: vllm.SamplingParams
    training_step: Optional[int] = None
    dataset_index: Optional[List[int]] = None
    is_eval: bool = False


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
        **kwargs,
    ):
        # Store tool-related parameters
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}

        # Initialize tool executor if tools are provided
        if self.tools:
            self.executor = ThreadPoolExecutor(max_workers=20)
            self.pending_tool_futures = {}
        else:
            self.executor = None
            self.pending_tool_futures = {}

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

        # Create EngineArgs and initialize LLMEngine
        engine_args = vllm.EngineArgs(*args, **kwargs)
        self.llm_engine = vllm.LLMEngine.from_engine_args(engine_args)
        self.llm = None  # Set llm to None when using engine directly

        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.logger = logging.getLogger(__name__)
        self.actor_manager = actor_manager

    def process_from_queue(self, timeout: float = 60.0):
        """Run generation loop using LLMEngine directly, with optional tool support."""
        while True:
            if ray.get(self.actor_manager.should_stop.remote()):
                self.logger.info("[LLMRayActor] Actor manager signaled to stop. Exiting generation loop.")
                return
            try:
                request = self.prompt_queue.get(timeout=timeout)
            except queue.Empty:
                self.logger.warning("[LLMRayActor] No request in the queue to process. Continuing.")
                continue

            # Process the request based on whether tools are available
            if self.tools:
                # Use tool-aware generation
                result = self._process_with_tools(request)
            else:
                # Use standard generation
                result = self._process_without_tools(request)

            try:
                if request.is_eval:
                    self.eval_results_queue.put(result, timeout=10)
                else:
                    self.results_queue.put(result, timeout=10)
            except queue.Full:
                self.logger.warning("Results queue is full, discarding result.")

    def _process_without_tools(self, request):
        """Standard generation without tool support."""
        prompts = request.prompts

        # Add requests to the engine
        for i, prompt in enumerate(prompts):
            request_id = f"batch_{request.training_step}_{i}"
            tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt)
            self.llm_engine.add_request(request_id, tokens_prompt, request.sampling_params)

        # Run the engine event loop until all requests are finished
        outputs = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)

        # Sort outputs by prompt index (i)
        outputs.sort(key=lambda x: int(x.request_id.split("_")[-1]))

        return self._process_outputs(outputs, dataset_index=request.dataset_index)

    def _process_with_tools(self, request):
        """Tool-aware generation using the ToolUseLLM approach."""
        prompts = request.prompts
        sampling_params = request.sampling_params

        # Override sampling params to have n=1 for individual processing
        single_n_sampling_params = copy.deepcopy(sampling_params)
        single_n_sampling_params.n = 1
        single_n_sampling_params.output_kind = vllm.sampling_params.RequestOutputKind.FINAL_ONLY

        # Add requests to the engine (handling n > 1 by creating multiple requests)
        for i, prompt in enumerate(prompts):
            for j in range(sampling_params.n):
                request_id = f"{i}-{j}"
                tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt)
                self.llm_engine.add_request(request_id, tokens_prompt, single_n_sampling_params)

        # Initialize tracking variables
        num_calls = defaultdict(int)
        timeout = defaultdict(bool)
        tool_error = defaultdict(str)
        tool_output = defaultdict(str)
        tool_runtime = defaultdict(float)
        tool_called = defaultdict(bool)
        concat_outputs = {}
        masks = defaultdict(list)

        # Get tokenizer for encoding tool outputs
        tokenizer = self.llm_engine.get_tokenizer()

        # Run the engine with tool support
        outputs = []
        while True:
            # Poll pending tool futures
            dict_keys_to_delete = []
            for req_id, (future, last_o, last_output) in self.pending_tool_futures.items():
                if future.done():
                    tool_result = future.result()
                    last_prompt_token_ids = last_output.prompt_token_ids
                    last_token_ids = last_o.token_ids
                    tool_output_token_ids = tokenizer.encode(
                        "<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False
                    )
                    timeout[req_id] = tool_result.timeout
                    tool_error[req_id] += "" if tool_result.error is None else tool_result.error
                    tool_output[req_id] += tool_result.output
                    tool_runtime[req_id] += tool_result.runtime
                    tool_called[req_id] = True

                    # Edge case 1: clip against model context length
                    prompt_and_tool_output_token = last_prompt_token_ids + last_token_ids + tool_output_token_ids
                    num_calls[req_id] += 1
                    excess = len(prompt_and_tool_output_token) - self.llm_engine.model_config.max_model_len
                    if excess > 0:
                        tool_output_token_ids = tool_output_token_ids[:-excess]
                        can_make_new_request = False
                    else:
                        can_make_new_request = True

                    # Edge case 2: clip against per-request max_tokens
                    remaining = single_n_sampling_params.max_tokens - len(masks[req_id])
                    if remaining <= 0:
                        tool_output_token_ids = []
                    elif len(tool_output_token_ids) > remaining:
                        tool_output_token_ids = tool_output_token_ids[:remaining]

                    concat_outputs[req_id].outputs[0].token_ids.extend(tool_output_token_ids)
                    masks[req_id].extend([0] * len(tool_output_token_ids))
                    new_sample_tokens = single_n_sampling_params.max_tokens - len(masks[req_id])
                    can_make_new_request = can_make_new_request and new_sample_tokens > 0

                    if can_make_new_request:
                        try:
                            new_sampling_params = copy.deepcopy(single_n_sampling_params)
                            new_sampling_params.max_tokens = new_sample_tokens
                            self.llm_engine.add_request(
                                req_id,
                                vllm.TokensPrompt(prompt_token_ids=prompt_and_tool_output_token),
                                new_sampling_params,
                            )
                        except Exception as e:
                            self.logger.error(f"Error adding request: {e}")

                    dict_keys_to_delete.append(req_id)

            for req_id in dict_keys_to_delete:
                del self.pending_tool_futures[req_id]

            # Process engine steps
            if self.llm_engine.has_unfinished_requests():
                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        assert len(output.outputs) <= 1  # because sampling_params.n == 1
                        o = output.outputs[0]
                        output_processed = False

                        if output.request_id not in concat_outputs:
                            concat_outputs[output.request_id] = output
                        else:
                            concat_outputs[output.request_id].outputs[0].token_ids.extend(o.token_ids)

                        masks[output.request_id].extend([1] * len(o.token_ids))

                        # Check if output ends with a tool stop string
                        for stop_str in single_n_sampling_params.stop:
                            if (
                                o.text.endswith(stop_str)
                                and stop_str in self.tools
                                and num_calls[output.request_id] < self.max_tool_calls.get(stop_str, 0)
                            ):
                                # Schedule tool call asynchronously
                                tool = self.tools[stop_str]
                                future = self.executor.submit(tool, o.text)
                                self.pending_tool_futures[output.request_id] = (future, o, output)
                                output_processed = True
                                break
                            elif (
                                o.text.endswith(stop_str)
                                and stop_str in self.tools
                                and num_calls[output.request_id] >= self.max_tool_calls.get(stop_str, 0)
                            ):
                                # Tool call limit exceeded
                                tool = MaxCallsExceededTool(start_str="<tool>", end_str="</tool>")
                                future = self.executor.submit(tool, o.text)
                                self.pending_tool_futures[output.request_id] = (future, o, output)
                                output_processed = True
                                break

                        if not output_processed:
                            outputs.append(output)

            if not self.llm_engine.has_unfinished_requests() and len(self.pending_tool_futures) == 0:
                break

        # Add tool metadata to outputs
        for req_id in masks:
            assert req_id in concat_outputs
            setattr(concat_outputs[req_id].outputs[0], "mask", masks[req_id])
            setattr(concat_outputs[req_id].outputs[0], "num_calls", num_calls[req_id])
            setattr(concat_outputs[req_id].outputs[0], "timeout", timeout[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_error", tool_error[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_output", tool_output[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_runtime", tool_runtime[req_id])
            setattr(concat_outputs[req_id].outputs[0], "tool_called", tool_called[req_id])

        # Merge n completions into the same outputs
        merged_outputs = {}
        for req_id in concat_outputs:
            real_req_id, _ = req_id.split("-")
            if real_req_id not in merged_outputs:
                merged_outputs[real_req_id] = concat_outputs[req_id]
            else:
                merged_outputs[real_req_id].outputs.append(concat_outputs[req_id].outputs[0])

        final_outputs = sorted(
            merged_outputs.values(), key=lambda x: (int(x.request_id.split("-")[0]), int(x.request_id.split("-")[1]))
        )

        return self._process_outputs_with_tools(final_outputs, dataset_index=request.dataset_index)

    def _process_outputs(
        self,
        outputs: List[Any],  # List of vllm.RequestOutput objects
        dataset_index: Optional[List[int]] = None,
    ) -> GenerationResult:
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
        self,
        outputs: List[Any],  # List of vllm.RequestOutput objects
        dataset_index: Optional[List[int]] = None,
    ) -> GenerationResult:
        """Process vLLM RequestOutputs into GenerationResult format with tool information."""
        response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
        finish_reasons = [out.finish_reason for output in outputs for out in output.outputs]

        # Extract tool-specific attributes from outputs
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

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray=False
    ):
        return self.llm_engine.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
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
    pg: Optional[ray.util.placement_group] = None,
    vllm_enable_sleep=False,
    tools: Optional[Dict[str, Tool]] = None,
    max_tool_calls: List[int] = [5],
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
) -> list:
    import vllm

    assert vllm.__version__ >= "0.8.1", "OpenRLHF only supports vllm >= 0.8.1"

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

        # Always use LLMRayActor with optional tool parameters
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
