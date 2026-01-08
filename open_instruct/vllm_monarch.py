"""
Monarch-based vLLM engine actors for distributed inference.

This module provides Monarch actor implementations for vLLM inference engines,
replacing the Ray-based actors in vllm_utils.py for use with Monarch-based training.
"""

import asyncio
import logging
import socket
import threading
from datetime import timedelta
from typing import Any

import torch.distributed as dist
import uvicorn
import vllm
from monarch.actor import Actor, current_rank, endpoint
from vllm.entrypoints.openai.api_server import build_app, init_app_state

from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.tool_utils.tools import Tool
from open_instruct.vllm_utils import _create_server_args

logger = logging.getLogger(__name__)

INFERENCE_INIT_TIMEOUT_S = 600


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
        train_dataset: Any = None,
        eval_dataset: Any = None,
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
        self.client = None
        self._model_update_group = None

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
        import openai

        self.client = openai.OpenAI(base_url=f"http://127.0.0.1:{self.server_port}/v1", api_key="dummy")

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
    async def generate(
        self,
        prompts: list[str],
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        n: int = 1,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate completions for the given prompts.

        Returns a list of completion results, each containing:
        - text: The generated text
        - logprobs: Token log probabilities (if requested)
        - finish_reason: Why generation stopped
        """
        results = []

        for prompt in prompts:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                stop=stop,
                seed=seed,
                logprobs=1,
            )

            for choice in response.choices:
                results.append(
                    {"text": choice.text, "logprobs": choice.logprobs, "finish_reason": choice.finish_reason}
                )

        return results

    @endpoint
    async def shutdown(self) -> None:
        """Shutdown the vLLM engine gracefully."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        logger.info(f"VLLMMonarchEngine rank {self.rank} shutdown complete")
