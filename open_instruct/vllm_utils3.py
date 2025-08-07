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
import logging
import os
import queue
from datetime import timedelta
from typing import Any, List, Optional, Union

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

from open_instruct.utils import ray_get_with_progress

logger = logging.getLogger(__name__)


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


@ray.remote
class LLMRayActor:
    def __init__(
        self,
        *args,
        bundle_indices: list = None,
        tool_use: bool = False,
        prompt_queue=None,
        results_queue=None,
        eval_results_queue=None,
        actor_manager=None,
        inference_batch_size: int = 1,
        **kwargs,
    ):
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

        # Pop update_weights_inflight before passing kwargs to vLLM
        self.update_weights_inflight = kwargs.pop("update_weights_inflight", False)

        if tool_use:
            # TODO: Need to update ToolUseLLM to use LLMEngine as well
            from open_instruct.tool_utils.tool_vllm import ToolUseLLM

            self.llm = ToolUseLLM(*args, **kwargs)
            self.llm_engine = self.llm.llm_engine
        else:
            # Create EngineArgs and initialize LLMEngine
            engine_args = vllm.EngineArgs(*args, **kwargs)
            self.llm_engine = vllm.LLMEngine.from_engine_args(engine_args)
            self.llm = None  # Set llm to None when using engine directly

        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.tool_use = tool_use
        self.logger = logging.getLogger(__name__)
        self.actor_manager = actor_manager
        self.inference_batch_size = inference_batch_size

    def _tool_generation_loop(self, timeout: float = 60.0):
        while True:
            if ray.get(self.actor_manager.should_stop.remote()):
                self.logger.info("[LLMRayActor] Actor manager signaled to stop. Exiting generation loop.")
                return
            try:
                request = self.prompt_queue.get(timeout=timeout)
                outputs = self.llm.generate(
                    sampling_params=request.sampling_params, prompt_token_ids=request.prompts, use_tqdm=False
                )
                result = self._process_outputs(outputs, dataset_index=request.dataset_index)
                if request.is_eval:
                    self.eval_results_queue.put(result)
                else:
                    self.results_queue.put(result)
            except queue.Empty:
                pass
            except queue.Full:
                self.logger.warning("Results queue is full, discarding result.")

    def _maybe_add_requests(
        self,
        prompt_queue,
        llm_engine,
        pending_requests,
        active_request_count,
        request_counter,
        inference_batch_size,
        should_stop,
        timeout,
    ):
        """Pull requests off the queue and add them to the engine until we have inference_batch_size requests queued.

        Returns updated (active_request_count, request_counter).
        """
        while active_request_count < inference_batch_size and not should_stop:
            try:
                # Try to get a request with very short timeout to avoid blocking
                request = prompt_queue.get(timeout=0.001 if active_request_count > 0 else timeout)

                # Add each prompt in the request to the engine
                for prompt_idx, prompt in enumerate(request.prompts):
                    request_id = f"req_{request_counter}_{prompt_idx}"
                    tokens_prompt = vllm.TokensPrompt(prompt_token_ids=prompt)
                    llm_engine.add_request(request_id, tokens_prompt, request.sampling_params)

                    # Track this request
                    pending_requests[request_id] = (request, prompt_idx)
                    active_request_count += 1

                request_counter += 1

            except queue.Empty:
                # No more requests available
                break

        return active_request_count, request_counter

    def _run_generation_loop(self, timeout: float = 60.0):
        """Run generation loop using LLMEngine directly."""
        # Track pending requests and their metadata
        pending_requests = {}  # request_id -> (original_request, prompt_index_in_request)
        active_request_count = 0
        request_counter = 0

        while True:
            # Check if we should stop
            should_stop = ray.get(self.actor_manager.should_stop.remote())

            # Exit conditions
            if should_stop and self.update_weights_inflight:
                self.logger.info(
                    "[LLMRayActor] Actor manager signaled to stop (update_weights_inflight=True). Exiting immediately."
                )
                return

            if should_stop and not self.llm_engine.has_unfinished_requests():
                self.logger.info("[LLMRayActor] Actor manager signaled to stop. Exiting generation loop.")
                return

            # Pull requests off the queue and add them to the engine
            active_request_count, request_counter = self._maybe_add_requests(
                self.prompt_queue,
                self.llm_engine,
                pending_requests,
                active_request_count,
                request_counter,
                self.inference_batch_size,
                should_stop,
                timeout,
            )

            # Process each output
            for output in self.llm_engine.step():
                processed_count = self._process_single_output(output, pending_requests)
                active_request_count -= processed_count

    def _process_single_output(self, output, pending_requests):
        """Process a single vLLM RequestOutput and handle it completely.

        Returns the number of requests that were processed (0 or 1).
        """
        # Only process finished outputs
        if not output.finished:
            return 0

        request_id = output.request_id
        if request_id not in pending_requests:
            return 0

        original_request, prompt_idx = pending_requests[request_id]

        # Extract response data from the output
        response_ids = [list(out.token_ids) for out in output.outputs]
        finish_reasons = [out.finish_reason for out in output.outputs]

        if self.tool_use:
            masks = [out.mask for out in output.outputs]
            num_calls = [out.num_calls for out in output.outputs]
            timeouts = [out.timeout for out in output.outputs]
            tool_errors = [out.tool_error for out in output.outputs]
            tool_outputs = [out.tool_output for out in output.outputs]
            tool_runtimes = [out.tool_runtime for out in output.outputs]
            tool_calleds = [out.tool_called for out in output.outputs]
        else:
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

        # Get the appropriate dataset index for this prompt
        dataset_index = None
        if original_request.dataset_index:
            # If we have dataset indices, extract the one for this prompt
            if prompt_idx < len(original_request.dataset_index):
                dataset_index = [original_request.dataset_index[prompt_idx]]

        result = GenerationResult(
            responses=response_ids,
            finish_reasons=finish_reasons,
            masks=masks,
            request_info=request_info,
            dataset_index=dataset_index,
        )

        # Send result to appropriate queue
        try:
            if original_request.is_eval:
                self.eval_results_queue.put(result, timeout=1)
            else:
                self.results_queue.put(result, timeout=1)
        except queue.Full:
            self.logger.warning("Results queue is full, discarding result.")

        # Clean up the pending request
        del pending_requests[request_id]

        return 1

    def process_from_queue(self, timeout=0.1):
        """Process a single element from the queue."""
        if self.tool_use:
            self.run_tool_generation_loop(timeout)
        else:
            self._run_generation_loop(timeout)

    def _process_outputs(
        self,
        outputs: List[Any],  # List of vllm.RequestOutput objects
        dataset_index: Optional[List[int]] = None,
    ) -> GenerationResult:
        """Process vLLM RequestOutputs into GenerationResult format."""
        # Debug logging
        self.logger.info(
            f"[_process_outputs] Processing {len(outputs)} RequestOutputs, dataset_indices={dataset_index}"
        )
        for i, output in enumerate(outputs):
            self.logger.info(
                f"[_process_outputs] Output {i}: request_id={output.request_id}, num_completions={len(output.outputs)}"
            )

        # Process outputs
        response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
        finish_reasons = [out.finish_reason for output in outputs for out in output.outputs]

        if self.tool_use:
            masks = [out.mask for output in outputs for out in output.outputs]
            num_calls = [out.num_calls for output in outputs for out in output.outputs]
            timeouts = [out.timeout for output in outputs for out in output.outputs]
            tool_errors = [out.tool_error for output in outputs for out in output.outputs]
            tool_outputs = [out.tool_output for output in outputs for out in output.outputs]
            tool_runtimes = [out.tool_runtime for output in outputs for out in output.outputs]
            tool_calleds = [out.tool_called for output in outputs for out in output.outputs]
        else:
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

        # Debug logging
        self.logger.info(
            f"[_process_outputs] Returning GenerationResult with {len(result.responses)} responses "
            f"for {len(dataset_index) if dataset_index else 0} dataset indices"
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
    vllm_gpu_memory_utilization: float,
    single_gpu_mode: bool,
    inference_batch_size: int,
    pg: Optional[ray.util.placement_group] = None,
    vllm_enable_sleep=False,
    tools: Optional[List[Any]] = None,
    max_tool_calls: List[int] = [5],
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
    update_weights_inflight: bool = False,
) -> list[LLMRayActor]:
    import vllm

    assert vllm.__version__ >= "0.8.1", "OpenRLHF only supports vllm >= 0.8.1"

    # Convert max_tool_calls to a dict mapping tool end strings to their limits
    assert len(max_tool_calls) == 1 or len(max_tool_calls) == len(tools), (
        "max_tool_calls must have length 1 (applies to all tools) or same length as tools (per-tool limit)"
    )
    # tool key is the end_str
    if len(max_tool_calls) == 1:
        max_tool_calls_dict = {tool: max_tool_calls[0] for tool in tools.keys()} if tools else {}
    else:
        max_tool_calls_dict = {tool: limit for tool, limit in zip(tools.keys(), max_tool_calls)} if tools else {}

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

        additional_kwargs = {}
        tool_use = False
        if tools is not None and len(tools) > 0:
            tool_use = True
            additional_kwargs["tools"] = tools
            additional_kwargs["max_tool_calls"] = max_tool_calls_dict

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                # VLLM v1 multiprocessing is required due to https://github.com/vllm-project/vllm/issues/15349
                runtime_env=ray.runtime_env.RuntimeEnv(
                    env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TORCH_CUDA_ARCH_LIST": get_cuda_arch_list()}
                ),
            ).remote(
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
                tool_use=tool_use,
                prompt_queue=prompt_queue,
                results_queue=results_queue,
                eval_results_queue=eval_results_queue,
                actor_manager=actor_manager,
                inference_batch_size=inference_batch_size,
                update_weights_inflight=update_weights_inflight,
                **additional_kwargs,
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
