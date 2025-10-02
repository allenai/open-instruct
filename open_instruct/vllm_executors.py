import asyncio
from typing import Optional

from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest


class PauseAwareUniExecutor(UniProcExecutor):
    """Uni executor with a cooperative pause lock for weight sync."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pause_lock: Optional[asyncio.Lock] = None

    async def _ensure_lock(self) -> asyncio.Lock:
        if self._pause_lock is None:
            self._pause_lock = asyncio.Lock()
        return self._pause_lock

    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> list[SamplerOutput]:
        lock = await self._ensure_lock()
        async with lock:
            return await super().execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        lock = await self._ensure_lock()
        async with lock:
            await super().stop_remote_worker_execution_loop_async()

    async def stop_remote_worker_execution_loop_no_lock(self) -> None:
        await super().stop_remote_worker_execution_loop_async()

    async def acquire_pause_lock(self) -> None:
        lock = await self._ensure_lock()
        await lock.acquire()

    async def release_pause_lock(self) -> None:
        lock = await self._ensure_lock()
        if lock.locked():
            lock.release()


class PauseAwareMPExecutor(MultiprocessingDistributedExecutor):
    """Multiprocessing executor with cooperative pause lock for TP > 1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pause_lock: Optional[asyncio.Lock] = None

    async def _ensure_lock(self) -> asyncio.Lock:
        if self._pause_lock is None:
            self._pause_lock = asyncio.Lock()
        return self._pause_lock

    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> list[SamplerOutput]:
        lock = await self._ensure_lock()
        async with lock:
            return await super().execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        lock = await self._ensure_lock()
        async with lock:
            await super().stop_remote_worker_execution_loop_async()

    async def stop_remote_worker_execution_loop_no_lock(self) -> None:
        await super().stop_remote_worker_execution_loop_async()

    async def acquire_pause_lock(self) -> None:
        lock = await self._ensure_lock()
        await lock.acquire()

    async def release_pause_lock(self) -> None:
        lock = await self._ensure_lock()
        if lock.locked():
            lock.release()
