import asyncio
from typing import Optional

from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest


class LockingExecutorMixin:
    """Serializes execute_model_async and weight updates via a shared asyncio lock."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock: Optional[asyncio.Lock] = asyncio.Lock()

    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> list[SamplerOutput]:
        async with self._lock:
            return await super().execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        async with self._lock:
            await super().stop_remote_worker_execution_loop_async()

    async def stop_remote_worker_execution_loop_no_lock(self) -> None:
        await super().stop_remote_worker_execution_loop_async()

    async def acquire_lock(self) -> None:
        await self._lock.acquire()

    async def release_lock(self) -> None:
        if self._lock.locked():
            self._lock.release()


class PauseAwareUniExecutor(LockingExecutorMixin, UniProcExecutor):
    pass


class PauseAwareMPExecutor(LockingExecutorMixin, MultiprocessingDistributedExecutor):
    pass
