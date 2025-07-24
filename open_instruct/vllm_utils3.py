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
import sys
import time
from datetime import timedelta
from typing import Any, List, Optional, Union

import ray
import torch
import torch.distributed

logger = logging.getLogger(__name__)
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
    is_eval: bool = False
    dataset_index: Optional[List[int]] = None
    training_step: Optional[int] = None


@dataclasses.dataclass
class PromptRequest:
    """Container for prompt requests to vLLM."""

    prompts: List[List[int]]
    training_step: Optional[int] = None
    eval_prompts: Optional[List[List[int]]] = None
    dataset_index: Optional[List[int]] = None


@ray.remote
class WeightUpdater:
    """Manages weight updates between main thread and vLLM engines."""

    def __init__(self, num_engines: int):
        self.num_engines = num_engines
        self.weight_update_available = False
        self.weight_data_refs = {}  # name -> ray object ref
        self.update_count = 0
        self.current_update_id = 0
        self.logger = logging.getLogger(__name__)

    def signal_weights_available(self, weight_refs: dict):
        """Called by main thread to signal new weights are available."""
        self.logger.info(f"[WeightUpdater] Signaling weights available. Num params: {len(weight_refs)}")
        self.weight_update_available = True
        self.weight_data_refs = weight_refs
        self.update_count = 0
        self.current_update_id += 1
        return self.current_update_id

    def check_update_available(self):
        """Called by vLLM actors to check if update needed."""
        return self.weight_update_available, self.current_update_id

    def get_weight_refs(self):
        """Called by vLLM actors to get weight data references."""
        return self.weight_data_refs

    def confirm_update_complete(self, update_id: int):
        """Called by vLLM actors after updating weights."""
        if update_id == self.current_update_id:
            self.update_count += 1
            self.logger.info(f"[WeightUpdater] Update confirmed: {self.update_count}/{self.num_engines}")
            if self.update_count >= self.num_engines:
                self.weight_update_available = False
                self.logger.info("[WeightUpdater] All engines updated")
                # Clean up weight refs to free memory
                self.weight_data_refs.clear()
        return self.update_count

    def wait_for_all_updates(self, update_id: int, timeout: float = 300):
        """Called by main thread to wait for all engines to complete update."""
        from tqdm import tqdm

        start_time = time.time()
        pbar = tqdm(
            total=self.num_engines,
            initial=self.update_count,
            desc="Waiting for vLLM engines to update weights",
            bar_format="{l_bar}{bar}{r_bar}\n",
        )

        last_count = self.update_count
        while True:
            current_count = self.update_count
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count

            if current_count >= self.num_engines:
                pbar.close()
                self.logger.info(f"[WeightUpdater] All updates complete for update_id={update_id}")
                return True

            if time.time() - start_time > timeout:
                pbar.close()
                self.logger.error(
                    f"[WeightUpdater] Timeout waiting for updates. Got {self.update_count}/{self.num_engines}"
                )
                return False

            time.sleep(0.1)  # Poll every 100ms

    def get_update_status(self):
        """Get current update status for debugging."""
        return {
            "update_available": self.weight_update_available,
            "current_update_id": self.current_update_id,
            "update_count": self.update_count,
            "num_engines": self.num_engines,
        }

    def ready(self):
        """Check if the WeightUpdater is ready."""
        return True


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


def _setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # otherwise INFO is filtered
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s (%(process)d) %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.handlers.clear()  # workers may be reused; avoid duplicates
    logger.addHandler(handler)
    logger.propagate = False  # don’t send to root twice
    return logger


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
        weight_updater=None,
        **kwargs,
    ):
        # We have to call this here as we need to initialize the logger within
        # ray.
        self.logger = _setup_logger(name="LLMRayActor")
        self.logger.info("Starting to initialize LLMRayActor.")
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

        if tool_use:
            from open_instruct.tool_utils.tool_vllm import ToolUseLLM

            self.llm = ToolUseLLM(*args, **kwargs)
        else:
            from vllm import LLM

            self.llm = LLM(*args, **kwargs)
        self.logger.info("Initialized LLM.")

        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.tool_use = tool_use
        self.weight_updater = weight_updater
        self.last_update_id = 0

        # Fail fast if weight_updater is not provided
        if self.weight_updater is None:
            raise ValueError("weight_updater is required for LLMRayActor")

        self.logger.info(f"Queue IDs - prompt_queue: {id(prompt_queue)}, results_queue: {id(results_queue)}")
        self.logger.info(f"Weight updater: {weight_updater}")
        self.logger.info("Done initialize LLMRayActor.")

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def _perform_weight_update(self, update_id: int):
        """Pull and apply weight updates from the WeightUpdater."""
        self.logger.info(f"[vLLM] Starting weight update for update_id={update_id}")

        try:
            # Get weight references from WeightUpdater
            weight_refs = ray.get(self.weight_updater.get_weight_refs.remote())
            self.logger.info(f"[vLLM] Got {len(weight_refs)} weight references")

            # Apply each weight update
            for name, weight_ref in weight_refs.items():
                # Get the weight data from Ray object store
                weight_data = ray.get(weight_ref)
                dtype, shape, empty_cache = weight_data["dtype"], weight_data["shape"], weight_data["empty_cache"]

                # Apply the update through collective RPC
                self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

            # Confirm update complete
            ray.get(self.weight_updater.confirm_update_complete.remote(update_id))
            self.last_update_id = update_id
            self.logger.info(f"[vLLM] Weight update complete for update_id={update_id}")

        except Exception as e:
            self.logger.error(f"[vLLM] Error during weight update: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

    def process_from_queue(
        self,
        sampling_params,
        eval_sampling_params=None,
        eval_freq=None,
        num_training_steps=None,
        resume_training_step=1,
    ):
        """Process prompts from the queue and put results in the results queue."""
        try:
            self.logger.info("[vLLM] Starting process_from_queue:")
            self.logger.info(f"  - num_training_steps: {num_training_steps}")
            self.logger.info(f"  - resume_training_step: {resume_training_step}")
            self.logger.info(f"  - will process steps: {resume_training_step} to {num_training_steps}")
            self.logger.info(f"  - prompt_queue: {self.prompt_queue} (id: {id(self.prompt_queue)})")
            self.logger.info(f"  - results_queue: {self.results_queue}")
            self.logger.info(f"  - eval_results_queue: {self.eval_results_queue}")

            # Test queue access
            try:
                queue_size = self.prompt_queue.qsize()
                self.logger.info(f"[vLLM] Initial prompt_queue size: {queue_size}")
            except Exception as e:
                self.logger.error(f"[vLLM] ERROR accessing prompt_queue: {e}")
                raise

            if self.prompt_queue is None:
                error_msg = "[vLLM] ERROR: prompt_queue is None!"
                self.logger.error(error_msg)
                return {"error": error_msg}
            if self.results_queue is None:
                error_msg = "[vLLM] ERROR: results_queue is None!"
                self.logger.error(error_msg)
                return {"error": error_msg}

            # Check if we have any steps to process
            if resume_training_step > num_training_steps:
                self.logger.info(
                    f"[vLLM] No steps to process: resume_training_step({resume_training_step}) > num_training_steps({num_training_steps})"
                )
                return {
                    "status": "no_steps_to_process",
                    "resume_training_step": resume_training_step,
                    "num_training_steps": num_training_steps,
                }

            # Process requests until we get a None (stop signal)
            self.logger.info("[vLLM] Starting to process requests from queue")
            while True:
                try:
                    # Use a short timeout to periodically yield control back to Ray
                    # This allows the actor to process other remote calls like update_weight
                    request = self.prompt_queue.get(timeout=0.01)  # 10ms timeout for faster response
                    self.logger.info("[vLLM] Successfully got request from queue")
                except Exception:
                    # Timeout is expected - check for weight updates during this time
                    # We already verified weight_updater is not None in __init__
                    update_available, update_id = ray.get(self.weight_updater.check_update_available.remote())
                    if update_available and update_id > self.last_update_id:
                        self.logger.info(f"[vLLM] Weight update available: update_id={update_id}")
                        self._perform_weight_update(update_id)
                    continue

                if request is None:
                    self.logger.info("[vLLM] Received None (stop signal), breaking")
                    break

                self.logger.info("[vLLM] Got request:")
                self.logger.info(f"  - request.training_step: {request.training_step}")
                self.logger.info(f"  - num_prompts: {len(request.prompts)}")
                self.logger.info(
                    f"  - dataset_indices: {request.dataset_index[:5]}..."
                    if len(request.dataset_index) > 5
                    else f"  - dataset_indices: {request.dataset_index}"
                )

                # Process training prompts
                self.logger.info(f"[vLLM] Processing {len(request.prompts)} prompts")
                result = self._generate_batch(
                    request.prompts, sampling_params, request.dataset_index, request.training_step
                )

                self.logger.info(f"[vLLM] Putting result in results_queue for training_step={request.training_step}")
                self.results_queue.put(result)
                self.logger.info("[vLLM] Successfully put result")

                # Handle evaluation if needed
                if (
                    request.eval_prompts is not None
                    and eval_sampling_params is not None
                    and (request.training_step - 1) % eval_freq == 0
                ):
                    eval_result = self._generate_batch(
                        request.eval_prompts, eval_sampling_params, request.dataset_index, request.training_step
                    )
                    eval_result.is_eval = True
                    # Put eval results in separate queue if available
                    if self.eval_results_queue is not None:
                        self.eval_results_queue.put(eval_result)
                    else:
                        self.results_queue.put(eval_result)
                    self.logger.info("[vLLM] Successfully put eval result")
        except Exception as e:
            self.logger.error(f"[vLLM] ERROR in process_from_queue: {e}")
            self.logger.error(f"[vLLM] Exception type: {type(e).__name__}")
            import traceback

            tb = traceback.format_exc()
            self.logger.error(f"[vLLM] Traceback:\n{tb}")

            # Return error information instead of None
            return {"error": str(e), "exception_type": type(e).__name__, "traceback": tb}

    def _generate_batch(
        self,
        prompts: List[List[int]],
        sampling_params,
        dataset_index: Optional[List[int]] = None,
        training_step: Optional[int] = None,
    ) -> GenerationResult:
        """Generate responses for a batch of prompts."""
        self.logger.info(f"[vLLM] Starting generation for {len(prompts)} prompts")
        outputs = self.llm.generate(sampling_params=sampling_params, prompt_token_ids=prompts, use_tqdm=False)
        self.logger.info(f"[vLLM] Generation complete, got {len(outputs)} outputs")

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

        return GenerationResult(
            responses=response_ids,
            finish_reasons=finish_reasons,
            masks=masks,
            request_info=request_info,
            dataset_index=dataset_index,
            training_step=training_step,
        )

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray=False
    ):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.logger.info(f"[vLLM] update_weight called for {name}")
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def ready(self):
        return True

    def ping(self):
        """Simple health check method."""
        return {"status": "alive", "timestamp": time.time()}


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
    tools: Optional[List[Any]] = None,
    max_tool_calls: List[int] = [5],
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
) -> tuple[list[LLMRayActor], WeightUpdater]:
    import vllm

    assert vllm.__version__ >= "0.8.1", "OpenRLHF only supports vllm >= 0.8.1"

    # Create WeightUpdater for coordinating weight updates
    weight_updater = WeightUpdater.remote(num_engines)
    ray.get(weight_updater.ready.remote())  # Ensure it's initialized
    logger.info(f"Created WeightUpdater for {num_engines} engines")

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
        # VLLM v1 multiprocessing is required due to https://github.com/vllm-project/vllm/issues/15349
        env_vars = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}

        # Pass TORCH_CUDA_ARCH_LIST if it's set in the main process
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            env_vars["TORCH_CUDA_ARCH_LIST"] = os.environ["TORCH_CUDA_ARCH_LIST"]
        else:
            torch_env_vars = [env_var for env_var in os.environ if env_var.startswith("TORCH_")]
            print(f"{torch_env_vars=}")

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env=ray.runtime_env.RuntimeEnv(env_vars=env_vars),
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
                weight_updater=weight_updater,
                **additional_kwargs,
            )
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep", rank_0_only=False)

    return vllm_engines, weight_updater


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
