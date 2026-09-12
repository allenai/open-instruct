"""MILES raw-JSON routing with independent health transport and owned shutdown."""

import asyncio
import importlib
from contextlib import suppress
from typing import Any, cast

import httpx
import setproctitle
import uvicorn
from fastapi import HTTPException, Request
from miles.router import router as native_router
from miles.utils import logging_utils

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


class ManagedMilesRouter(native_router.MilesRouter):
    """Keep long generation requests from consuming health-check connections.

    Quarantine is deliberately sticky. Only explicit worker registration can
    re-admit an incarnation; a successful health check cannot certify its weights.
    """

    def __init__(self, args, verbose=False):
        """Allocate separate generation and health transports."""
        self._retired = set()
        self._worker_epochs = {}
        self._health_task = None
        super().__init__(args, verbose=verbose)
        engines = max(1, args.rollout_num_gpus // args.rollout_num_gpus_per_engine)
        self.health_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max(4, 2 * engines)),
            timeout=httpx.Timeout(getattr(args, "rollout_health_check_timeout", 5.0)),
        )
        self.app.router.on_shutdown.append(self.close)

    def _setup_routes(self):
        self.app.post("/remove_worker")(self.remove_worker)
        super()._setup_routes()

    async def _start_background_health_check(self):
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        try:
            response = await self.health_client.get(f"{url}/health")
            return url, response.status_code == 200
        except httpx.HTTPError as error:
            logger.warning("Router health probe failed worker=%s error=%s", url, type(error).__name__)
            return url, False

    async def _health_check_loop(self):
        while True:
            await asyncio.sleep(self.args.rollout_health_check_interval)
            epochs = {
                u: self._worker_epochs.get(u, 0)
                for u in self.worker_request_counts
                if u not in self.dead_workers and u not in self._retired
            }
            results = await asyncio.gather(*(self._check_worker_health(url) for url in epochs))
            for url, healthy in results:
                if url in self._retired or self._worker_epochs.get(url, 0) != epochs[url]:
                    continue
                failures = 0 if healthy else self.worker_failure_counts.get(url, 0) + 1
                self.worker_failure_counts[url] = failures
                if failures >= self.args.miles_router_health_check_failure_threshold:
                    self.dead_workers.add(url)
                    logger.error("Router quarantined worker=%s consecutive_health_failures=%d", url, failures)

    async def add_worker(self, request: Request):
        """Register a new incarnation and clear its prior quarantine."""
        result = await super().add_worker(request)
        if not isinstance(result, dict):
            return result
        url = request.query_params.get("url") or request.query_params.get("worker_url")
        if not url:
            payload = await request.json()
            url = payload.get("url") or payload.get("worker_url")
        if url in self._retired and self.worker_request_counts.get(url, 0):
            raise HTTPException(409, "retired worker still has active requests; register a new endpoint")
        self._worker_epochs[url] = self._worker_epochs.get(url, 0) + 1
        self._retired.discard(url)
        self.dead_workers.discard(url)
        self.worker_failure_counts[url] = 0
        return result

    async def remove_worker(self, request: Request):
        """Retire an endpoint without invalidating in-flight request counters."""
        url = request.query_params.get("url") or request.query_params.get("worker_url")
        if not url:
            payload = await request.json()
            url = payload.get("url") or payload.get("worker_url")
        if not url:
            raise HTTPException(400, "worker_url is required")
        # Preserve counters until in-flight proxy requests execute their finally.
        self._retired.add(url)
        self.dead_workers.add(url)
        self._worker_epochs[url] = self._worker_epochs.get(url, 0) + 1
        return {"status": "success"}

    async def list_workers(self, request: Request):
        """List registered endpoints, including quarantined workers for aborts."""
        return {"urls": [u for u in self.worker_request_counts if u not in self._retired]}

    def _use_url(self):
        workers = [u for u in self.worker_request_counts if u not in self.dead_workers and u not in self._retired]
        if not workers:
            raise HTTPException(503, "No healthy rollout workers available", headers={"Retry-After": "1"})
        url = min(workers, key=self.worker_request_counts.get)
        self.worker_request_counts[url] += 1
        return url

    async def do_proxy(self, request, path, body=None, headers=None):
        """Surface engine transport loss as retryable unavailability."""
        try:
            return await super().do_proxy(request, path, body=body, headers=headers)
        except httpx.RequestError as error:
            logger.warning("Rollout proxy transport failed error=%s", type(error).__name__)
            raise HTTPException(503, "Rollout worker unavailable", headers={"Retry-After": "1"}) from error

    async def close(self):
        """Join the health task and close both connection pools."""
        if self._health_task is not None:
            self._health_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_task
            self._health_task = None
        await self.health_client.aclose()
        await self.client.aclose()


def run_managed_router(args):
    """Spawn target owned by olmo-miles; leave the pinned router library intact."""
    logging_utils.configure_logger_raw("miles_router")
    setproctitle.setproctitle("miles-router")
    router = ManagedMilesRouter(args)
    uvicorn.run(router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


def install():
    """Select the owned router before the rollout manager starts serving.

    Called from its data-source constructor, which runs before router startup
    for synchronous and asynchronous Core runs. The spawn target is importable.
    """
    manager = cast(Any, importlib.import_module("miles.ray.rollout.router_manager"))
    manager.run_miles_router = run_managed_router
