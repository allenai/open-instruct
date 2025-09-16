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

import os
import queue
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


def _process_outputs(
    output: vllm.RequestOutput,
    dataset_index: int,
    training_step: int,
    token_statistics: TokenStatistics,
    start_time: float,
    use_tools: bool = False,
) -> "GenerationResult":
    """Process a vLLM RequestOutput into GenerationResult format.

    Args:
        output: The vLLM request output to process
        dataset_index: Index in the dataset
        training_step: Current training step
        token_statistics: Token usage statistics
        start_time: Request start time
        use_tools: Whether tools are being used

    Returns:
        GenerationResult containing processed outputs
    """
    response_ids = [list(out.token_ids) for out in output.outputs]
    finish_reasons = [out.finish_reason for out in output.outputs]

    # Extract attributes based on whether tools are used
    if use_tools:
        # Extract tool-specific attributes from outputs
        masks = [getattr(out, "mask", [1] * len(out.token_ids)) for out in output.outputs]
        num_calls = [getattr(out, "num_calls", 0) for out in output.outputs]
        timeouts = [getattr(out, "timeout", False) for out in output.outputs]
        tool_errors = [getattr(out, "tool_error", "") for out in output.outputs]
        tool_outputs = [getattr(out, "tool_output", "") for out in output.outputs]
        tool_runtimes = [getattr(out, "tool_runtime", 0.0) for out in output.outputs]
        tool_calleds = [getattr(out, "tool_called", False) for out in output.outputs]
    else:
        # Use default values when tools are not used
        masks = [[1] * len(resp) for resp in response_ids]
        num_calls = [0] * len(response_ids)
        timeouts = [False] * len(response_ids)
        tool_errors = [""] * len(response_ids)
        tool_outputs = [""] * len(response_ids)
        tool_runtimes = [0.0] * len(response_ids)
        tool_calleds = [False] * len(response_ids)

    return GenerationResult(
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
        dataset_index=dataset_index,
        training_step=training_step,
        token_statistics=token_statistics,
        start_time=start_time,
    )


def _finalize_outputs(
    output: vllm.RequestOutput,
    tracking: Dict[str, Any],
    dataset_index: int,
    training_step: int,
    tools: Optional[Dict[str, Tool]],
    original_sampling_params: vllm.SamplingParams,
    token_statistics: TokenStatistics,
    start_time: float,
) -> GenerationResult:
    """Prepare final outputs based on whether tools were used."""

    if not tools:
        return _process_outputs(
            output,
            dataset_index=dataset_index,
            training_step=training_step,
            token_statistics=token_statistics,
            start_time=start_time,
            use_tools=False,
        )

    # Collect all sub-request outputs for this request
    all_outputs = []
    for j in range(original_sampling_params.n):
        sub_req_id = f"{output.request_id}_{j}"

        # Set tool metadata attributes on the output
        sub_output = tracking["concat_outputs"][sub_req_id].outputs[0]
        setattr(sub_output, "mask", tracking["masks"][sub_req_id])
        setattr(sub_output, "num_calls", tracking["num_calls"][sub_req_id])
        setattr(sub_output, "timeout", tracking["timeout"][sub_req_id])
        setattr(sub_output, "tool_error", tracking["tool_error"][sub_req_id])
        setattr(sub_output, "tool_output", tracking["tool_output"][sub_req_id])
        setattr(sub_output, "tool_runtime", tracking["tool_runtime"][sub_req_id])
        setattr(sub_output, "tool_called", tracking["tool_called"][sub_req_id])

        all_outputs.append(sub_output)

    # Create merged output with all sub-request outputs
    merged_output = vllm.RequestOutput(
        request_id=output.request_id,
        prompt=output.prompt,
        prompt_token_ids=output.prompt_token_ids,
        prompt_logprobs=output.prompt_logprobs,
        outputs=all_outputs,
        finished=output.finished,
    )

    return _process_outputs(
        merged_output,
        dataset_index=dataset_index,
        training_step=training_step,
        token_statistics=token_statistics,
        start_time=start_time,
        use_tools=True,
    )


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


def add_request(request: PromptRequest, llm_engine: vllm.LLMEngine, tools, request_metadata: dict):
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
        llm_engine.add_request(f"{request_id}_{j}", tokens_prompt, sub_sampling_params)
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
        verbose: bool = False,
        **kwargs,
    ):
        self.logger = logger_utils.setup_logger(__name__)
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.inference_batch_size = inference_batch_size
        self.verbose = verbose
        self.request_metadata = {}

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

    def _insert_result_to_queue(self, result, is_eval: bool):
        """Insert result into the appropriate queue with blocking put."""
        results_queue = self.eval_results_queue if is_eval else self.results_queue
        results_queue.put(result)

    def fill_engine(self, timeout: float):
        """Fill the LLM engine queue with requests until inference_batch_size is reached.

        Args:
            timeout: Timeout for getting requests from the queue

        Returns:
            int: Number of requests added to the engine
        """
        num_added = 0
        if self._should_stop():
            return num_added
        num_to_add = self.inference_batch_size - self.llm_engine.get_num_unfinished_requests()
        while num_added < num_to_add:
            try:
                request = self.prompt_queue.get(timeout=timeout)
                num_added += add_request(request, self.llm_engine, self.tools, request_metadata=self.request_metadata)
            except queue.Empty:
                break
        return num_added

    def process_from_queue(self, timeout: float = 60.0):
        """Run generation loop using LLMEngine directly, with optional tool support.

        Returns:
            int: Number of requests processed
        """
        num_processed = self.fill_engine(timeout=timeout)

        if num_processed == 0:
            return num_processed

        tracking = _init_tool_tracking()
        outputs = []
        while True:
            outputs.extend(self._poll_tool_futures(tracking, self.llm_engine.tokenizer))

            # Process engine steps - ONLY if there are unfinished requests (matching ToolUseLLM)
            if self.llm_engine.has_unfinished_requests():
                step_outputs = [o for o in self.llm_engine.step() if o.finished]
                for output in step_outputs:
                    base_req_id = _extract_base_request_id(output.request_id)
                    result = _handle_output(
                        output,
                        self.tools,
                        tracking,
                        self.request_metadata[base_req_id]["sampling_params"],
                        self.max_tool_calls,
                        self.executor,
                    )
                    # Result is None when we do more tool processing.
                    if result is not None:
                        outputs.append(result)

            if self.llm_engine.get_num_unfinished_requests() + len(tracking["pending_tool_futures"]) == 0:
                break

        end_time = time.time()
        request_outputs = defaultdict(list)
        for output in outputs:
            request_id = _extract_base_request_id(output.request_id)
            request_outputs[request_id].append(output)
        for request_id, outs in request_outputs.items():
            final_output = vllm.RequestOutput(
                request_id=request_id,
                prompt=outs[0].prompt,
                prompt_token_ids=outs[0].prompt_token_ids,
                prompt_logprobs=outs[0].prompt_logprobs,
                outputs=[completion for out in outs for completion in out.outputs],
                finished=outs[0].finished,
            )
            total_generation_tokens = sum(len(completion.token_ids) for out in outs for completion in out.outputs)
            metadata = self.request_metadata.pop(request_id)

            result = _finalize_outputs(
                final_output,
                tracking,
                metadata["dataset_index"],
                metadata["training_step"],
                self.tools,
                metadata["original_sampling_params"],
                TokenStatistics(
                    num_prompt_tokens=metadata["prompt_tokens"],
                    num_response_tokens=total_generation_tokens,
                    generation_time=end_time - metadata["start_time"],
                ),
                metadata["start_time"],
            )
            self._insert_result_to_queue(result, is_eval=metadata["is_eval"])

        return len(request_outputs)

    def _poll_tool_futures(self, tracking, tokenizer):
        """Poll and handle completed tool executions."""
        if not self.tools or not tracking["pending_tool_futures"]:
            return []

        dict_keys_to_delete = []
        completed_outputs = []

        for req_id, (future, last_o, last_output) in tracking["pending_tool_futures"].items():
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
                except Exception as e:
                    # Match original ToolUseLLM behavior - just log and continue
                    self.logger.error(f"[_poll_tool_futures] Error adding request {req_id}: {e}")
            else:
                # If we can't make a new request, this tool execution is complete
                completed_outputs.append(tracking["concat_outputs"][req_id])

            dict_keys_to_delete.append(req_id)

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
