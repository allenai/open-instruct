# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Lifecycle-safe adapter for MILES' fully asynchronous rollout producer."""

import asyncio
import time
from contextlib import suppress
from typing import Any

import httpx
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.http_utils import post

from open_instruct import logger_utils
from open_instruct.miles import async_capacity, pipeline_observer
from open_instruct.miles.errors import GenerationInterrupted

logger = logger_utils.setup_logger(__name__)

_QUIESCE_LOG_INTERVAL_SECONDS = 30.0
# Consecutive group requests that may lose their HTTP transport (connection
# reset, closed stream) before the producer treats the router as dead.
TRANSPORT_FAILURE_BUDGET = 8
# A producer join that outlives the health-check budget gets one retry after the
# engines abort their in-flight requests, which is what releases a generation
# task still waiting on a response.
_PUBLICATION_JOIN_RETRY_SECONDS = 180.0


class ManagedFullyAsyncRolloutFn(FullyAsyncRolloutFn):
    """Track, quiesce, and close all background generation tasks.

    MILES' producer intentionally outlives individual rollout calls, but the
    pinned implementation has no disposal protocol and keeps its active child
    tasks in a local variable. This adapter makes ownership explicit so the
    rollout manager can wait for terminal request outcomes before Ray teardown.
    """

    def __init__(self, input: Any) -> None:
        """Construct the MILES producer and its lifecycle signals."""
        super().__init__(input)
        capacity = async_capacity.report(
            vars(self.args), getattr(getattr(self.args, "olmo_core", None), "max_policy_lag", None)
        )
        logger.info("Async pipeline capacity: %s", capacity)
        for warning in capacity["warnings"]:
            logger.warning("Async capacity: %s", warning)
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._producer_idle = asyncio.Event()
        self._producer_idle.set()
        self._stop_requested = asyncio.Event()
        self._stopping = False
        self._shutdown_complete = False
        self._producing_groups: dict[int, Any] = {}
        # Group tasks keyed for failure bookkeeping; a transport failure on one
        # request discards its whole group and requeues the pristine prompts.
        self._task_groups: dict[asyncio.Task[Any], int] = {}
        self._transport_requeues = 0
        self._consecutive_transport_failures = 0
        self._ready_completion_counts: dict[int, dict[str, int]] = {}
        self._shutdown_unqueued_counts = dict(groups=0, samples=0, response_tokens=0)
        self._completed_put_wait_seconds = 0.0
        self._completed_put_started = None
        self._draining_groups: list[Any] | None = None
        self._interrupted_groups: list[int] = []
        self._publication_paused = False
        if getattr(self.args, "async_unused_samples_handler", None) == "drop":
            self._handle_unused = lambda group: self.data_source.acknowledge_groups([group])

    def _interrupted(self) -> bool:
        return bool(
            getattr(self.args, "_olmo_rollout_generation_interrupted", False)
            or getattr(self.args, "_olmo_rollout_pool_exhausted", False)
        )

    async def __call__(self, input: Any) -> Any:
        """Commit prompt consumption only after a complete training batch drains."""
        if input.evaluation:
            return await super().__call__(input)

        if self._publication_paused or self._interrupted():
            raise GenerationInterrupted("async producer requires recovery and current-weight publication")
        if self._worker is None and self._output is not None:
            self._worker = asyncio.create_task(self._worker_loop())
        self._draining_groups = []
        try:
            result = await super().__call__(input)
            # A partial dequeue is not consumed training data. Commit the entire
            # batch only after the upstream drain and filtering complete.
            self.data_source.acknowledge_groups(self._draining_groups)
            return result
        except BaseException:
            self._interrupted_groups.extend(g[0].group_index for g in self._draining_groups)
            raise
        finally:
            self._draining_groups = None

    async def _next_group(self, current_version: int | None) -> Any:
        # Own both outcomes: if the worker fails just as a dequeue succeeds,
        # that prompt must remain part of the interrupted batch's retry ledger.
        queue_get = asyncio.create_task(self._output.get(current_version=current_version))
        recorded = False
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker}, timeout=30, return_when=asyncio.FIRST_COMPLETED
                )
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async worker exited unexpectedly")
                if queue_get in done:
                    entry = queue_get.result()
                    if self._draining_groups is None:
                        self.data_source.acknowledge_groups([entry.prompt_group])
                    else:
                        self._draining_groups.append(entry.prompt_group)
                    recorded = True
                    return entry
                logger.warning("No completed async rollout groups for 30 seconds")
        finally:
            if not queue_get.done():
                queue_get.cancel()
            await asyncio.gather(queue_get, return_exceptions=True)
            if not recorded and not queue_get.cancelled() and queue_get.exception() is None:
                self._interrupted_groups.append(queue_get.result().prompt_group[0].group_index)

    def _submit_one_group(self) -> asyncio.Task:
        [group] = self.data_source.get_samples(1)
        self._producing_groups[group[0].group_index] = group
        self._scheduler.on_submit([group])
        task = asyncio.create_task(self._generate_group(group))
        self._task_groups[task] = group[0].group_index
        return task

    def _requeue_transport_failure(self, group_index: int | None, error: BaseException) -> bool:
        """Discard a group whose request lost its HTTP transport and regenerate it later.

        A lost connection (reset, closed stream, refused connect) leaves delivery
        unknown, so the partial group is never reused; its pristine prompts go
        back to the ledger exactly as after a preemption, and a fresh group is
        sampled under the current weights. Anything other than a transport error,
        or more than ``TRANSPORT_FAILURE_BUDGET`` failures in a row, still fails
        the run: a dead router is not a transient.
        """
        if not isinstance(error, httpx.TransportError) or group_index is None:
            return False
        self._consecutive_transport_failures += 1
        self._transport_requeues += 1
        self._producing_groups.pop(group_index, None)
        self.data_source.requeue_pending_groups([group_index])
        logger.warning(
            "Async group %s lost its HTTP transport (%s: %s); prompts requeued for regeneration "
            "(requeues=%d consecutive=%d/%d)",
            group_index,
            type(error).__name__,
            error,
            self._transport_requeues,
            self._consecutive_transport_failures,
            TRANSPORT_FAILURE_BUDGET,
        )
        if self._consecutive_transport_failures > TRANSPORT_FAILURE_BUDGET:
            raise RuntimeError(
                f"{self._consecutive_transport_failures} consecutive async requests lost their HTTP transport; "
                "the serving router is not reachable"
            ) from error
        return True

    async def _abort_engine_requests(self, timeout: float) -> None:
        """Ask every engine to abort its in-flight requests; unreachable engines are logged."""
        urls = await asyncio.wait_for(get_worker_urls(self.args), timeout)

        async def abort(url):
            try:
                await asyncio.wait_for(post(f"{url}/abort_request", {"abort_all": True}, max_retries=1), timeout)
            except (httpx.HTTPError, TimeoutError):
                logger.warning("Async publication abort could not reach worker=%s", url)

        await asyncio.gather(*(abort(url) for url in urls))

    async def _join_worker(self, timeout: float) -> list[Any]:
        assert self._worker is not None
        self._worker.cancel()
        return list(await asyncio.wait_for(asyncio.gather(self._worker, return_exceptions=True), timeout))

    async def prepare_publication(self) -> list[int]:
        """Cancel/join unfinished groups; keep completed buffered groups intact."""
        self._publication_paused = True
        self._producer_resumed.clear()
        self.state.aborted = True
        timeout = self.args.rollout_health_check_timeout
        aborted_engines = False
        if self._worker is not None:
            started = time.monotonic()
            try:
                results = await self._join_worker(timeout)
            except TimeoutError:
                logger.warning(
                    "Async producer join exceeded %.0fs with %d active group(s); aborting engine requests and retrying",
                    timeout,
                    len(self._active_tasks),
                )
                await self._abort_engine_requests(timeout)
                aborted_engines = True
                results = await self._join_worker(_PUBLICATION_JOIN_RETRY_SECONDS)
            logger.info("Async producer joined for publication in %.2fs", time.monotonic() - started)

            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, (asyncio.CancelledError, GenerationInterrupted)
                ):
                    raise result
            self._worker = None
        identities = list(dict.fromkeys([*self._producing_groups, *self._interrupted_groups]))
        if identities:
            if not aborted_engines:
                await self._abort_engine_requests(timeout)
            self.data_source.requeue_pending_groups(identities)
        self._producing_groups.clear()
        self._interrupted_groups.clear()

        self._scheduler = make_submission_scheduler(self.args, default="sample")
        self.state.reset()
        return identities

    async def finish_publication(self) -> None:
        """Resume the producer after recovered engines receive current weights."""
        if self._interrupted():
            raise RuntimeError("cannot resume async generation before engine recovery")
        self._publication_paused = False
        self._producer_resumed.set()
        if self._output is not None and self._worker is None and not self._stopping:
            self._worker = asyncio.create_task(self._worker_loop())

    async def _put_or_stop(self, item: Any) -> bool:
        """Put one completion unless final shutdown makes the buffer unreachable."""
        self._completed_put_started = time.monotonic()
        put_task = asyncio.create_task(self._output.put(item))
        stop_waiter = asyncio.create_task(self._stop_requested.wait())
        try:
            while True:
                if self._interrupted():
                    raise GenerationInterrupted("engine retired while async output buffer was full")
                done, _ = await asyncio.wait({put_task, stop_waiter}, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                if done:
                    break
            if put_task in done:
                put_task.result()
                return True
            put_task.cancel()
            with suppress(asyncio.CancelledError):
                await put_task
            return False
        finally:
            self._completed_put_wait_seconds += time.monotonic() - self._completed_put_started
            self._completed_put_started = None
            if not put_task.done():
                put_task.cancel()
            await asyncio.gather(put_task, return_exceptions=True)
            if not put_task.cancelled() and put_task.exception() is None:
                if put_task.result() is False:
                    self.data_source.acknowledge_groups([item.prompt_group])
                self._producing_groups.pop(item.prompt_group[0].group_index, None)
            stop_waiter.cancel()
            with suppress(asyncio.CancelledError):
                await stop_waiter

    async def _worker_loop(self) -> None:
        """Generate continuously while retaining ownership of every child task."""
        observers = [
            asyncio.create_task(pipeline_observer.observe(self)),
            asyncio.create_task(pipeline_observer.observe_engines(self, get_worker_urls)),
        ]
        active: set[asyncio.Task[Any]] = set()
        self._active_tasks = active
        self._producer_idle.clear()
        try:
            while True:
                if self._interrupted():
                    raise GenerationInterrupted("rollout engine retired during async production")
                if self._producer_resumed.is_set() and not self._stopping:
                    while self._scheduler.has_capacity(
                        pending_groups=len(active), group_budget=self._max_in_flight_groups()
                    ):
                        active.add(self._submit_one_group())
                    self._active_tasks = active

                if not active:
                    self._producer_idle.set()
                    if self._stopping:
                        return
                    await self._producer_resumed.wait()
                    self._producer_idle.clear()
                    continue

                try:
                    done, active = await asyncio.wait_for(self._scheduler.wait_for_progress(active), timeout=0.5)
                except TimeoutError:
                    continue
                self._active_tasks = active
                completions = []
                for task in done:
                    group_index = self._task_groups.pop(task, None)
                    error = None if task.cancelled() else task.exception()
                    if error is not None and self._requeue_transport_failure(group_index, error):
                        continue
                    completions.append(
                        self._collect_group_result(task, self._producing_groups[group_index])
                        if task.cancelled()
                        else task.result()
                    )
                    self._consecutive_transport_failures = 0
                self._ready_completion_counts.update(
                    (id(completion), pipeline_observer.completion_counts([completion])) for completion in completions
                )
                for completion in completions:
                    accepted = await self._put_or_stop(completion) if not self._stopping else False
                    counts = self._ready_completion_counts.pop(id(completion))
                    if not accepted:
                        for key, value in counts.items():
                            self._shutdown_unqueued_counts[key] += value
        finally:
            for observer in observers:
                observer.cancel()
            await asyncio.gather(*observers, return_exceptions=True)
            unfinished = [task for task in active if not task.done()]
            if unfinished:
                (logger.info if self._publication_paused or self._stopping else logger.warning)(
                    "Joining %d unfinished async generation task(s): publication_paused=%s stopping=%s",
                    len(unfinished),
                    self._publication_paused,
                    self._stopping,
                )
                for task in unfinished:
                    task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            self._task_groups.clear()
            self._active_tasks = set()
            self._producer_idle.set()

    async def _wait_until_idle(self) -> None:
        """Wait for terminal child outcomes, periodically reporting useful state."""
        assert self._worker is not None
        while not self._producer_idle.is_set():
            idle_waiter = asyncio.create_task(self._producer_idle.wait())
            try:
                done, _ = await asyncio.wait(
                    {idle_waiter, self._worker},
                    timeout=_QUIESCE_LOG_INTERVAL_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._worker in done:
                    self._worker.result()
                    if not self._producer_idle.is_set():
                        raise RuntimeError("fully-async producer exited before reaching an idle state")
                if not done:
                    logger.warning(
                        "Waiting for fully-async producer shutdown: active_groups=%d worker_done=%s",
                        len(self._active_tasks),
                        self._worker.done(),
                    )
            finally:
                idle_waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await idle_waiter

    async def shutdown(self) -> None:
        """Stop submissions, drain active requests, and join the producer task."""
        if self._shutdown_complete:
            return
        if self._worker is None:
            self._shutdown_complete = True
            logger.info("Fully-async producer shutdown skipped: worker was never started")
            return

        pipeline_observer.write_lifecycle(self, "shutdown_start")
        logger.info("Stopping fully-async producer: active_groups=%d", len(self._active_tasks))
        self._stopping = True
        self._stop_requested.set()
        self._producer_resumed.clear()
        await self._wait_until_idle()
        if not self._worker.done():
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
        else:
            self._worker.result()
        self._shutdown_complete = True
        pipeline_observer.write_lifecycle(self, "shutdown_complete")
        logger.info("Fully-async producer shutdown complete: active_groups=0")

    async def _call_eval(self, input: Any) -> Any:
        """Run MILES eval and stop permanently after the scheduled final eval."""
        output = await super()._call_eval(input)
        num_rollout = getattr(self.args, "num_rollout", None)
        if num_rollout is not None and input.rollout_id >= int(num_rollout) - 1:
            logger.info("Final evaluation completed; leaving fully-async producer stopped")
            await self.shutdown()
        return output
