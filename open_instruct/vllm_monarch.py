"""
Monarch-based vLLM engine actors for distributed inference.

This module provides Monarch actor implementations for vLLM inference engines,
replacing the Ray-based actors in vllm_utils.py for use with Monarch-based training.
"""

import asyncio
import dataclasses
import logging
import socket
import threading
import time
from datetime import timedelta

import datasets
import openai
import torch
import torch.distributed as dist
import uvicorn
import vllm
from monarch.actor import Actor, current_rank, endpoint
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.v1.core import kv_cache_utils

from open_instruct.data_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.dataset_transformation import GROUND_TRUTHS_KEY, RAW_PROMPT_KEY, VERIFIER_SOURCE_KEY
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.tool_utils.tools import Tool
from open_instruct.vllm_utils import (
    CompletionOutput,
    RequestOutput,
    SamplingConfig,
    _create_server_args,
    get_triggered_tool,
    make_request_id,
    split_request_id,
    truncate_tool_output_tokens,
)

logger = logging.getLogger(__name__)

INFERENCE_INIT_TIMEOUT_S = 600
DRAIN_ACTIVE_TASKS_SLEEP_S = 1


class VLLMMonarchEngine(Actor):
    """Monarch actor for vLLM inference engine.

    This is a Monarch-compatible version of LLMRayActor that uses
    Monarch's @endpoint decorator instead of Ray's remote methods.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
        seed: int = 0,
        tools: dict[str, Tool] | None = None,
        max_tool_calls: dict[str, int] | None = None,
        mask_tool_use: bool = True,
        inflight_updates: bool = False,
        reward_config: RewardConfig | None = None,
        train_dataset: datasets.Dataset | None = None,
        eval_dataset: datasets.Dataset | None = None,
    ):
        self.rank = current_rank().rank
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size

        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.mask_tool_use = mask_tool_use
        self.inflight_updates = inflight_updates
        self.reward_config = reward_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_fn = reward_config.build() if reward_config else None

        self.llm_engine = None
        self.loop = None
        self.server_port = None
        self.client: openai.AsyncOpenAI | None = None
        self._model_update_group = None

        self._prompt_queue: asyncio.Queue[PromptRequest] = asyncio.Queue()
        self._results_queue: asyncio.Queue[GenerationResult] = asyncio.Queue()
        self._eval_results_queue: asyncio.Queue[GenerationResult] = asyncio.Queue()

        self.request_metadata: dict[str, dict] = {}
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.request_outputs: dict[str, dict] = {}

        self._setup_engine(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
        )

    def _setup_engine(
        self,
        model: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        enable_prefix_caching: bool,
        seed: int,
    ):
        """Initialize the vLLM async engine."""
        engine_args = vllm.AsyncEngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
            disable_log_stats=True,
        )

        init_complete = threading.Event()

        async def _init_engine_and_server():
            running_loop = asyncio.get_running_loop()
            assert running_loop == self.loop

            engine_client = vllm.AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=False)

            args = _create_server_args(engine_client.vllm_config.model_config.model)
            app = build_app(args)
            await init_app_state(engine_client, engine_client.vllm_config, app.state, args)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            self.server_port = sock.getsockname()[1]

            logger.info(f"Starting vLLM OpenAI API server on port {self.server_port}")

            config = uvicorn.Config(app, host="127.0.0.1", port=self.server_port, log_level="warning")
            asyncio.create_task(uvicorn.Server(config).serve(sockets=[sock]))
            await asyncio.sleep(0.1)

            return engine_client

        def _run_loop():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.llm_engine = self.loop.run_until_complete(_init_engine_and_server())
            finally:
                init_complete.set()
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=_run_loop, daemon=True)
        self.loop_thread.start()

        if init_complete.wait(timeout=INFERENCE_INIT_TIMEOUT_S):
            if self.llm_engine is None:
                raise RuntimeError("vLLM engine initialization failed")
            self._init_openai_client()
            return
        raise RuntimeError("vLLM engine timed out during initialization")

    def _init_openai_client(self):
        """Initialize OpenAI-compatible client."""
        self.client = openai.AsyncOpenAI(
            base_url=f"http://127.0.0.1:{self.server_port}/v1", api_key="EMPTY", timeout=3600
        )
        self.model_name = self.llm_engine.vllm_config.model_config.model

    def _run_async(self, coro):
        """Run async function in the event loop."""
        if self.loop is None:
            raise RuntimeError("Event loop not initialized")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    @endpoint
    async def get_kv_cache_info(self) -> int:
        """Return the maximum number of concurrent requests based on KV cache."""
        gpu_cache_config = self.llm_engine.cache_config
        return gpu_cache_config.num_gpu_blocks

    @endpoint
    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout_minutes: int = 120,
    ) -> None:
        """Initialize a process group for weight synchronization."""
        logger.info(
            f"init_process_group: master={master_address}:{master_port}, "
            f"rank={rank}, world_size={world_size}, group={group_name}"
        )

        self._model_update_group = dist.new_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
            timeout=timedelta(minutes=timeout_minutes),
        )

    @endpoint
    async def update_weight(self, name: str, dtype: str, shape: tuple[int, ...], empty_cache: bool = False) -> None:
        """Update a model weight via broadcast from training rank."""
        expected_dtype = str(self.llm_engine.model_config.dtype)
        if dtype != expected_dtype:
            raise ValueError(f"Mismatched dtype for {name}: received {dtype}, expected {expected_dtype}")

        async def _update():
            await self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

        self._run_async(_update())

    @endpoint
    async def put_prompt(self, request: PromptRequest) -> None:
        """Submit a prompt request for generation. Called by DataPreparationActor."""
        await self._prompt_queue.put(request)

    @endpoint
    async def get_result(self) -> GenerationResult:
        """Get a completed generation result. Called by DataPreparationActor."""
        return await self._results_queue.get()

    @endpoint
    async def get_eval_result(self) -> GenerationResult:
        """Get a completed evaluation result."""
        return await self._eval_results_queue.get()

    @endpoint
    async def start_generation_loop(self) -> None:
        """Start the background generation loop."""
        asyncio.create_task(self._generation_loop())

    @endpoint
    async def get_queue_sizes(self) -> dict[str, int]:
        """Get current queue sizes for monitoring."""
        return {
            "prompt_queue": self._prompt_queue.qsize(),
            "results_queue": self._results_queue.qsize(),
            "eval_results_queue": self._eval_results_queue.qsize(),
            "active_tasks": len(self.active_tasks),
        }

    async def _generation_loop(self) -> None:
        """Background loop that processes prompts from the queue."""
        logger.info(f"VLLMMonarchEngine rank {self.rank} starting generation loop")
        while True:
            request = await self._prompt_queue.get()
            await self._add_request(request)

    async def _add_request(self, request: PromptRequest) -> None:
        """Add a prompt request for processing."""
        request_id = make_request_id(request)
        sampling_params = dataclasses.replace(request.generation_config, n=1)

        self.request_metadata[request_id] = {
            "is_eval": request.is_eval,
            "dataset_index": request.dataset_index,
            "prompt_id": request.prompt_id,
            "sampling_params": sampling_params,
            "original_sampling_params": request.generation_config,
            "prompt_token_ids": list(request.prompt),
            "start_time": time.perf_counter(),
        }

        for j in range(request.generation_config.n):
            seed = request.generation_config.seed + j if request.generation_config.seed is not None else None
            sub_sampling_params = dataclasses.replace(sampling_params, seed=seed)
            sub_request_id = f"{request_id}_{j}"
            task = asyncio.create_task(self._process_request(sub_request_id, sub_sampling_params))
            self.active_tasks[sub_request_id] = task

    async def _process_request(self, sub_request_id: str, sampling_params: SamplingConfig) -> None:
        """Process a single request with tool support."""
        response_tokens = []
        response_logprobs = []
        response_masks = []
        cumulative_logprob = 0.0
        num_calls = 0
        timeout = False
        tool_error = ""
        tool_output = ""
        tool_runtime = 0.0
        tool_called = False

        base_request_id = split_request_id(sub_request_id)["base_id"]
        original_prompt = self.request_metadata[base_request_id]["prompt_token_ids"]
        current_prompt = list(original_prompt)
        max_model_len = self.llm_engine.model_config.max_model_len
        current_max_tokens = sampling_params.max_tokens

        while True:
            current_sampling_params = dataclasses.replace(sampling_params, max_tokens=current_max_tokens)
            api_response = await self.client.completions.create(
                model=self.model_name,
                prompt=current_prompt,
                extra_body={
                    "return_token_ids": True,
                    "cache_salt": base_request_id,
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False,
                },
                **dataclasses.asdict(current_sampling_params),
            )

            output = api_response.choices[0]
            model_tokens = list(output.token_ids)

            response_tokens.extend(model_tokens)
            current_prompt.extend(model_tokens)

            assert output.logprobs and output.logprobs.token_logprobs, "logprobs must be available"
            for logprob in output.logprobs.token_logprobs:
                response_logprobs.append(logprob)
                cumulative_logprob += logprob

            response_masks.extend([1] * len(model_tokens))

            if not self.tools or not self.max_tool_calls:
                break

            triggered_tool, stop_str = get_triggered_tool(
                output.text, self.tools, self.max_tool_calls, num_calls, sampling_params
            )
            if triggered_tool is None:
                break

            loop = asyncio.get_running_loop()
            tool_result = await loop.run_in_executor(None, triggered_tool, output.text)

            tool_called = True
            num_calls += 1
            timeout = timeout or tool_result.timeout
            tool_error += "" if tool_result.error is None else tool_result.error
            tool_output += tool_result.output
            tool_runtime += tool_result.runtime

            tool_tokens = self.llm_engine.tokenizer.encode(
                "<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False
            )

            tool_tokens, excess = truncate_tool_output_tokens(
                tool_tokens,
                current_prompt_len=len(current_prompt),
                current_response_len=len(response_masks),
                max_model_len=max_model_len,
                max_tokens=sampling_params.max_tokens,
            )

            response_tokens.extend(tool_tokens)
            response_logprobs.extend([0.0] * len(tool_tokens))
            response_masks.extend([0 if self.mask_tool_use else 1] * len(tool_tokens))
            current_prompt.extend(tool_tokens)

            current_max_tokens = sampling_params.max_tokens - len(response_masks)
            if excess > 0 or current_max_tokens <= 0:
                break

        if output.finish_reason == "stop" and len(response_tokens) == 0:
            eos_token_id = self.llm_engine.tokenizer.eos_token_id
            response_tokens.append(eos_token_id)
            response_masks.append(1)
            response_logprobs.append(float("nan"))

        complete_output = CompletionOutput(
            index=split_request_id(sub_request_id)["request_index"],
            token_ids=response_tokens,
            cumulative_logprob=cumulative_logprob,
            logprobs=response_logprobs,
            finish_reason=output.finish_reason,
        )
        if self.tools:
            complete_output.mask = response_masks
            complete_output.num_calls = num_calls
            complete_output.timeout = timeout
            complete_output.tool_error = tool_error
            complete_output.tool_output = tool_output
            complete_output.tool_runtime = tool_runtime
            complete_output.tool_called = tool_called

        self.active_tasks.pop(sub_request_id, None)
        await self._accumulate_completion(base_request_id, sub_request_id, complete_output)

    async def _accumulate_completion(
        self, base_request_id: str, sub_request_id: str, complete_output: CompletionOutput
    ) -> None:
        """Accumulate completions and finalize when all are done."""
        expected_n = self.request_metadata[base_request_id]["original_sampling_params"].n

        if base_request_id not in self.request_outputs:
            self.request_outputs[base_request_id] = {"outputs": [], "expected_n": expected_n}

        request_output = RequestOutput(
            request_id=sub_request_id,
            prompt_token_ids=self.request_metadata[base_request_id]["prompt_token_ids"],
            outputs=[complete_output],
        )
        self.request_outputs[base_request_id]["outputs"].append(request_output)

        if len(self.request_outputs[base_request_id]["outputs"]) == expected_n:
            await self._finalize_completed_request(base_request_id)

    async def _finalize_completed_request(self, base_request_id: str) -> None:
        """Finalize a completed request and push to results queue."""
        outputs = self.request_outputs[base_request_id]["outputs"]
        ordered_outs = sorted(outputs, key=lambda x: split_request_id(x.request_id)["request_index"])

        current_time = time.perf_counter()
        metadata = self.request_metadata[base_request_id]

        final_output = RequestOutput(
            request_id=base_request_id,
            prompt_token_ids=ordered_outs[0].prompt_token_ids,
            outputs=[completion for out in ordered_outs for completion in out.outputs],
        )

        total_generation_tokens = sum(len(completion.token_ids) for out in ordered_outs for completion in out.outputs)

        response_ids = [list(out.token_ids) for out in final_output.outputs]
        finish_reasons = [out.finish_reason for out in final_output.outputs]
        use_tools = bool(self.tools)

        logprobs = []
        for out in final_output.outputs:
            logprobs.append(out.logprobs)

        if use_tools:
            masks = [getattr(out, "mask", [1] * len(out.token_ids)) for out in final_output.outputs]
            num_calls = [getattr(out, "num_calls", 0) for out in final_output.outputs]
            timeouts = [getattr(out, "timeout", False) for out in final_output.outputs]
            tool_errors = [getattr(out, "tool_error", "") for out in final_output.outputs]
            tool_outputs = [getattr(out, "tool_output", "") for out in final_output.outputs]
            tool_runtimes = [getattr(out, "tool_runtime", 0.0) for out in final_output.outputs]
            tool_calleds = [getattr(out, "tool_called", False) for out in final_output.outputs]
        else:
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
            prompt_id=metadata["prompt_id"],
            token_statistics=TokenStatistics(
                num_prompt_tokens=len(metadata["prompt_token_ids"]),
                num_response_tokens=total_generation_tokens,
                generation_time=current_time - metadata["start_time"],
            ),
            start_time=metadata["start_time"],
            logprobs=logprobs,
        )

        self.request_outputs.pop(base_request_id)
        self.request_metadata.pop(base_request_id, None)

        is_eval = metadata["is_eval"]
        dataset = self.eval_dataset if is_eval else self.train_dataset
        if self.reward_fn and dataset is not None:
            result.reward_scores, result.reward_metrics = await self._compute_rewards(result, dataset)

        results_queue = self._eval_results_queue if is_eval else self._results_queue
        await results_queue.put(result)

    async def _compute_rewards(self, result: GenerationResult, dataset: datasets.Dataset) -> tuple[list[float], dict]:
        """Compute rewards for the generation result."""
        example = dataset[result.dataset_index]
        decoded_responses = self.llm_engine.tokenizer.batch_decode(result.responses, skip_special_tokens=True)

        k = len(result.responses)
        k_ground_truths = [example[GROUND_TRUTHS_KEY]] * k
        k_datasets = [example[VERIFIER_SOURCE_KEY]] * k
        k_raw_queries = [example[RAW_PROMPT_KEY]] * k

        scores, metrics = await self.reward_fn(
            result.responses,
            decoded_responses,
            k_ground_truths,
            k_datasets,
            result.finish_reasons,
            result.request_info,
            k_raw_queries,
        )
        return scores, metrics

    def get_kv_cache_max_concurrency(self) -> int:
        """Get KV cache max concurrency from the vLLM engine."""
        kv_cache_specs = self._run_async(self.llm_engine.collective_rpc("get_kv_cache_spec"))

        vllm_config = self.llm_engine.vllm_config
        gpu_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = int(gpu_memory_utilization * total_gpu_memory)

        kv_cache_groups = kv_cache_utils.get_kv_cache_groups(vllm_config, kv_cache_specs[0])
        kv_cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )
        max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)

        return int(max_concurrency)

    @endpoint
    async def ready(self) -> bool:
        """Check if the engine is ready."""
        return self.llm_engine is not None

    @endpoint
    async def shutdown(self) -> None:
        """Shutdown the vLLM engine gracefully."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        logger.info(f"VLLMMonarchEngine rank {self.rank} shutdown complete")
