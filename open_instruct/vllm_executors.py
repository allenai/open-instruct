import asyncio
from typing import Optional

from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest


class PauseExecutorMixin:
    """Mixin that enables cooperative pause/resume semantics around decode steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pause_lock: Optional[asyncio.Lock] = None
        self._pause_lock_owned_by_pause: bool = False
        self._pause_requested: bool = False
        self._pause_ack_event: Optional[asyncio.Event] = None
        self._pause_resume_event: Optional[asyncio.Event] = None

    async def _ensure_sync_primitives(self) -> None:
        if self._pause_lock is None:
            self._pause_lock = asyncio.Lock()
        if self._pause_ack_event is None:
            self._pause_ack_event = asyncio.Event()
        if self._pause_resume_event is None:
            self._pause_resume_event = asyncio.Event()

    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> list[SamplerOutput]:
        await self._ensure_sync_primitives()
        lock = self._pause_lock
        ack_event = self._pause_ack_event
        resume_event = self._pause_resume_event

        await lock.acquire()
        lock_acquired = True
        try:
            if self._pause_requested:
                await self.stop_remote_worker_execution_loop_no_lock()
                self._pause_requested = False
                ack_event.set()

                lock.release()
                lock_acquired = False

                await resume_event.wait()
                resume_event.clear()

                await lock.acquire()
                lock_acquired = True

            result = await super().execute_model_async(execute_model_req)

            if self._pause_requested:
                await self.stop_remote_worker_execution_loop_no_lock()
                self._pause_requested = False
                ack_event.set()

                lock.release()
                lock_acquired = False

                await resume_event.wait()
                resume_event.clear()

                await lock.acquire()
                lock_acquired = True

            return result
        finally:
            if lock_acquired and lock.locked():
                lock.release()

    async def stop_remote_worker_execution_loop_async(self) -> None:
        await self._ensure_sync_primitives()
        lock = self._pause_lock
        await lock.acquire()
        try:
            await self.stop_remote_worker_execution_loop_no_lock()
        finally:
            if lock.locked():
                lock.release()

    async def stop_remote_worker_execution_loop_no_lock(self) -> None:
        await super().stop_remote_worker_execution_loop_async()

    async def request_pause(self) -> None:
        await self._ensure_sync_primitives()
        lock = self._pause_lock
        ack_event = self._pause_ack_event
        resume_event = self._pause_resume_event

        ack_event.clear()
        resume_event.clear()

        if not lock.locked():
            await lock.acquire()
            await self.stop_remote_worker_execution_loop_no_lock()
            self._pause_requested = False
            self._pause_lock_owned_by_pause = True
            ack_event.set()
            return

        self._pause_requested = True
        await ack_event.wait()
        ack_event.clear()

    async def acquire_pause_lock(self) -> None:
        await self._ensure_sync_primitives()
        lock = self._pause_lock
        if self._pause_lock_owned_by_pause:
            self._pause_lock_owned_by_pause = False
            return
        if self._pause_lock_owned_by_pause:
            return
        await lock.acquire()

    async def release_pause_lock(self) -> None:
        await self._ensure_sync_primitives()
        resume_event = self._pause_resume_event
        self._pause_lock_owned_by_pause = False
        resume_event.set()
        lock = self._pause_lock
        if lock.locked():
            lock.release()


class PauseAwareUniExecutor(PauseExecutorMixin, UniProcExecutor):
    pass


class PauseAwareMPExecutor(PauseExecutorMixin, MultiprocessingDistributedExecutor):
    pass
