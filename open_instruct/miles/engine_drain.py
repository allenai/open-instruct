"""Single-event-loop ownership for rolling publication (no CUDA or Ray dependencies).

Admission reserves a whole prompt group before any sibling can wait on a client
semaphore. A reservation lasts through the HTTP response, not reward evaluation.
Only the controller can reopen an engine after a complete publication handshake.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Assignment:
    engine: str
    incarnation: str
    version: int
    group: int
    requests: tuple[str, ...]
    attempt: int = 0


@dataclass
class Engine:
    identity: str
    incarnation: str
    version: int
    state: str = "serving"
    target: int | None = None
    requests: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class WeightSnapshot:
    """Owned immutable bucket references, never views of optimizer parameters."""

    version: int
    buckets: tuple[Any, ...]
    nbytes: int
    capture_seconds: float


class EngineDrain:
    def __init__(
        self,
        engines: list[Engine],
        deliver: Callable[[str, WeightSnapshot], Awaitable[int]],
        *,
        max_lag: int,
        capacity: int = 2,
        drain_timeout: float = 180,
        update_timeout: float = 180,
        event: Callable[[dict], None] | None = None,
    ):
        if not engines or len({e.identity for e in engines}) != len(engines):
            raise ValueError("engine drain requires distinct engines")
        if max_lag < 1 or capacity < 1 or min(drain_timeout, update_timeout) <= 0:
            raise ValueError("engine drain requires positive lag, snapshot capacity and deadlines")
        if any(e.version < 0 or e.state != "serving" or e.requests for e in engines):
            raise ValueError("engines must have acknowledged initial weights before admission")
        self.engines = {e.identity: e for e in engines}
        self.deliver = deliver
        self.max_lag, self.capacity = max_lag, capacity
        self.drain_timeout, self.update_timeout = drain_timeout, update_timeout
        self.step = max(e.version for e in engines)
        self.snapshot_ready = self.step
        self._event = event or (lambda record: None)
        self._changed = asyncio.Event()
        self._tails: dict[str, asyncio.Task] = {}
        self._snapshots: dict[int, tuple[WeightSnapshot, set[str]]] = {}
        self._failure: BaseException | None = None
        self._groups: dict[int, Assignment] = {}
        self._paused = False
        self._closed = False
        self._admission_sequence = 0

    def emit(self, kind: str, **fields):
        self._event({"event": kind, "time": time.time(), "monotonic": time.monotonic(), **fields})

    def check(self):
        if self._failure is not None:
            raise RuntimeError(
                "rolling publication failed; admission is closed; restart from a committed checkpoint"
            ) from self._failure

    def set_step(self, step: int):
        self.check()
        if type(step) is not int or step < self.step:
            raise ValueError("consuming optimizer step must not go backwards")
        self.step = step
        self._changed.set()

    async def reserve(self, group: int, count: int) -> Assignment:
        if count < 1:
            raise ValueError("a prompt group must contain requests")
        async with asyncio.timeout(self.drain_timeout + self.update_timeout):
            while True:
                self.check()
                if self._closed:
                    raise RuntimeError("engine admission is closed for shutdown")
                if group in self._groups:
                    raise ValueError(f"prompt group {group} already owns an engine reservation")
                # Reserve one step of lag headroom for unfinished work. The consumer
                # independently checks against its actual optimizer step at dequeue.
                eligible = [
                    e for e in self.engines.values() if e.state == "serving" and self.step - e.version < self.max_lag
                ]
                if eligible and not self._paused:
                    engine = min(eligible, key=lambda e: (len(e.requests), -e.version, e.identity))
                    attempt = self._admission_sequence
                    self._admission_sequence += 1
                    requests = tuple(f"{engine.incarnation}:{group}:{attempt}:{i}" for i in range(count))
                    assignment = Assignment(
                        engine.identity, engine.incarnation, engine.version, group, requests, attempt
                    )
                    engine.requests.update(requests)
                    self._groups[group] = assignment
                    self.emit("group_reserved", **vars(assignment), consuming_step=self.step)
                    return assignment
                self._changed.clear()
                await self._changed.wait()

    def decoded(self, assignment: Assignment, request: str, *, version: int, tokens: int):
        self.check()
        engine = self.engines[assignment.engine]
        if (
            self._groups.get(assignment.group) != assignment
            or engine.incarnation != assignment.incarnation
            or request not in engine.requests
            or version != assignment.version
            or engine.version != assignment.version
        ):
            error = ValueError("response ownership/version mismatch; engine must not be reused")
            self.fail(engine.identity, error)
            raise error
        engine.requests.remove(request)
        self.emit(
            "decode_finished",
            engine=engine.identity,
            request=request,
            group=assignment.group,
            attempt=assignment.attempt,
            assigned_version=assignment.version,
            executed_version=version,
            tokens=tokens,
        )
        self._changed.set()

    def graded(self, assignment: Assignment):
        engine = self.engines[assignment.engine]
        if any(r in engine.requests for r in assignment.requests):
            raise RuntimeError("cannot retire group ownership before all responses are terminal")
        if self._groups.pop(assignment.group, None) != assignment:
            raise RuntimeError("group ownership was lost or retired twice")
        self.emit(
            "group_graded",
            engine=engine.identity,
            group=assignment.group,
            attempt=assignment.attempt,
            version=assignment.version,
        )

    def fail(self, identity: str, error: BaseException):
        self.engines[identity].state = "unavailable"
        self._failure = self._failure or error
        self.emit("engine_unavailable", engine=identity, error=str(error))
        self._changed.set()

    async def wait_capacity(self):
        while len(self._snapshots) >= self.capacity:
            self.check()
            self._changed.clear()
            await self._changed.wait()
        self.check()

    def publish(self, snapshot: WeightSnapshot):
        self.check()
        if self._closed or len(self._snapshots) >= self.capacity:
            raise RuntimeError("snapshot capacity exhausted or publisher closed; await capacity before capture")
        if snapshot.version <= self.snapshot_ready or snapshot.version > self.step:
            raise ValueError("snapshot version must advance and may not exceed the optimizer step")
        self.snapshot_ready = snapshot.version
        self._snapshots[snapshot.version] = (snapshot, set(self.engines))
        self.emit(
            "snapshot_ready",
            version=snapshot.version,
            bytes=snapshot.nbytes,
            capture_seconds=snapshot.capture_seconds,
            retained_snapshots=len(self._snapshots),
        )
        for identity, engine in self.engines.items():
            engine.target = snapshot.version
            previous = self._tails.get(identity)
            # Close admission synchronously, before returning to request tasks.
            if engine.state == "serving":
                engine.state = "draining"
            self._tails[identity] = asyncio.create_task(self._update(identity, snapshot, previous))

    async def _update(self, identity, snapshot, previous):
        engine = self.engines[identity]
        try:
            if previous is not None:
                await previous
            self.check()
            engine.state = "draining"
            self.emit(
                "drain_started",
                engine=identity,
                version=engine.version,
                target=snapshot.version,
                outstanding_requests=len(engine.requests),
            )
            async with asyncio.timeout(self.drain_timeout):
                while engine.requests:
                    self._changed.clear()
                    await self._changed.wait()
                    self.check()
            self.emit("drain_finished", engine=identity, version=engine.version, target=snapshot.version)
            engine.state = "updating"
            self.emit("update_started", engine=identity, target=snapshot.version)
            version = await asyncio.wait_for(self.deliver(identity, snapshot), self.update_timeout)
            if version != snapshot.version:
                raise ValueError(f"engine {identity} acknowledged {version}, expected {snapshot.version}")
            engine.version = version
            # A queued newer snapshot retains closed admission. Never strand
            # siblings between updates: all siblings were reserved together.
            engine.state = "serving" if engine.target == version else "draining"
            self.emit(
                "engine_reopened" if engine.state == "serving" else "update_finished", engine=identity, version=version
            )
        except BaseException as error:
            self.fail(identity, error)
        finally:
            _, readers = self._snapshots[snapshot.version]
            readers.remove(identity)
            if not readers:
                del self._snapshots[snapshot.version]
                self.emit("snapshot_released", version=snapshot.version)
            self._changed.set()

    async def barrier(self):
        # Snapshot of tails is safe: only the driver submits versions, and it
        # waits here before submitting another, evaluating, or checkpointing.
        await asyncio.gather(*self._tails.values())
        self.check()

    async def pause(self):
        self._paused = True
        await self.barrier()

    def resume(self):
        self.check()
        self._paused = False
        self._changed.set()

    async def close(self):
        self._closed = True
        self._changed.set()
        await self.barrier()

    def status(self):
        self.check()
        return {
            "optimizer_step": self.step,
            "snapshot_ready": self.snapshot_ready,
            "fleet_converged": min(e.version for e in self.engines.values()),
            "retained_snapshot_bytes": sum(s.nbytes for s, _ in self._snapshots.values()),
            "groups_in_flight": len(self._groups),
            "engines": {
                key: {"state": e.state, "version": e.version, "target": e.target, "requests": len(e.requests)}
                for key, e in self.engines.items()
            },
        }
