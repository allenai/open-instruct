"""Pause queued generation without cancelling samples or occupying serving slots."""

import asyncio


class GenerationAdmission:
    """A semaphore whose queued callers can be parked while active calls finish.

    Pause/acquire bookkeeping is atomic within the rollout actor's event loop.
    Reward verification runs after leaving this context in the pinned MILES path.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("Generation admission capacity must be positive")
        self.capacity = capacity
        self._semaphore = asyncio.Semaphore(capacity)
        self._open = asyncio.Event()
        self._open.set()
        self._idle = asyncio.Event()
        self._idle.set()
        self.active = 0
        self.waiting = 0
        self._tasks = set()
        self._stopped = False

    @property
    def paused(self):
        return not self._open.is_set()

    def pause(self):
        self._open.clear()

    def resume(self):
        if self._stopped:
            raise RuntimeError("Cannot reopen stopped generation admission")
        self._open.set()

    async def wait_idle(self):
        await self._idle.wait()

    async def acquire(self):
        task = asyncio.current_task()
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self.waiting += 1
        try:
            while True:
                await self._open.wait()
                if self._stopped:
                    raise asyncio.CancelledError
                await self._semaphore.acquire()
                # Pause may have happened while this caller waited for a slot.
                # Give it back before parking so evaluation cannot be starved.
                if not self._open.is_set() or self._stopped:
                    self._semaphore.release()
                    continue
                self.active += 1
                self._idle.clear()
                return True
        finally:
            self.waiting -= 1

    def release(self):
        if self.active < 1:
            raise RuntimeError("Generation admission released without an active call")
        self.active -= 1
        self._semaphore.release()
        if not self.active:
            self._idle.set()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        self.release()

    async def abort(self):
        """Retire a failed evaluation, including queued calls and reward tasks."""
        self._stopped = True
        self._open.set()
        tasks = [task for task in self._tasks if task is not asyncio.current_task() and not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
