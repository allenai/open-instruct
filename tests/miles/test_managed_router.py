"""Real HTTP saturation, quarantine, and shutdown checks in the pinned runtime."""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
from miles.router import router as native_router

from open_instruct.miles import data_source, managed_router
from open_instruct.miles.managed_router import ManagedMilesRouter


def test_busy_generation_cannot_starve_health_and_shutdown_joins_probe(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    health = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            entered.set()
            release.wait(15)
            self.reply()

        def do_GET(self):
            health.append(self.path)
            self.reply()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def exercise():
        router = ManagedMilesRouter(
            SimpleNamespace(
                miles_router_max_connections=1,
                miles_router_timeout=10,
                rollout_num_gpus=1,
                rollout_num_gpus_per_engine=1,
                rollout_health_check_timeout=0.2,
                rollout_health_check_interval=0.01,
                miles_router_health_check_failure_threshold=2,
            )
        )
        url = f"http://127.0.0.1:{server.server_port}"
        router.worker_request_counts[url] = 0
        generation = asyncio.create_task(router.client.post(url + "/generate", json={}))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            errors = []
            original_get = router.client.get

            async def observed_get(*args, **kwargs):
                try:
                    return await original_get(*args, **kwargs)
                except httpx.PoolTimeout:
                    errors.append("PoolTimeout")
                    raise

            monkeypatch.setattr(router.client, "get", observed_get)
            # Actual pinned parent: the health request never reaches the server.
            assert await native_router.MilesRouter._check_worker_health(router, url) == (url, False)
            assert errors == ["PoolTimeout"] and health == [] and not generation.done()
            assert await router._check_worker_health(url) == (url, True)
            assert health == ["/health"] and not generation.done()
            await router._start_background_health_check()
            task = router._health_task
            await asyncio.sleep(0.1)
            assert len(health) > 2 and not router.dead_workers
            release.set()
            await generation
            await router.close()
            assert task.done() and task.cancelled()
            assert router.client.is_closed and router.health_client.is_closed
        finally:
            release.set()
            await asyncio.gather(generation, return_exceptions=True)
            await router.close()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_retired_worker_has_503_until_explicit_registration():
    async def exercise():
        router = ManagedMilesRouter(
            SimpleNamespace(
                miles_router_max_connections=1,
                miles_router_timeout=1,
                rollout_num_gpus=1,
                rollout_num_gpus_per_engine=1,
            )
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=router.app), base_url="http://router"
            ) as client:
                assert (await client.post("/generate", json={})).status_code == 503
                assert (await client.post("/add_worker", json={"url": "http://engine"})).status_code == 200
                router._use_url()
                assert (await client.post("/remove_worker", params={"url": "http://engine"})).status_code == 200
                assert (await client.get("/list_workers")).json() == {"urls": []}
                assert (await client.post("/add_worker", json={"url": "http://engine"})).status_code == 409
                router._finish_url("http://engine")
                assert (await client.post("/generate", json={})).status_code == 503
                assert (await client.post("/add_worker", json={"url": "http://engine"})).status_code == 200
                assert router._use_url() == "http://engine"
                router._finish_url("http://engine")
                assert router.worker_request_counts == {"http://engine": 0}
        finally:
            await router.close()

    asyncio.run(exercise())


def test_engine_transport_failure_is_retryable_and_releases_counter():
    async def exercise():
        router = ManagedMilesRouter(
            SimpleNamespace(
                miles_router_max_connections=1,
                miles_router_timeout=1,
                rollout_num_gpus=1,
                rollout_num_gpus_per_engine=1,
            )
        )

        async def disconnected(request):
            raise httpx.RemoteProtocolError("engine disappeared", request=request)

        await router.client.aclose()
        router.client = httpx.AsyncClient(transport=httpx.MockTransport(disconnected))
        router.worker_request_counts["http://engine"] = 0
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=router.app), base_url="http://router"
            ) as client:
                response = await client.post("/generate", json={})
                assert response.status_code == 503 and response.headers["Retry-After"] == "1"
                assert router.worker_request_counts == {"http://engine": 0}
        finally:
            await router.close()

    asyncio.run(exercise())


def test_data_source_selects_importable_router_before_server_start(monkeypatch):
    manager = managed_router.importlib.import_module("miles.ray.rollout.router_manager")
    original = manager.run_miles_router
    monkeypatch.setattr(manager, "run_miles_router", original)
    monkeypatch.setattr(data_source, "RolloutDataSourceWithBuffer", lambda args: object())
    data_source.DashboardDrainingRolloutDataSource(SimpleNamespace(use_miles_router=False))
    assert manager.run_miles_router is original
    data_source.DashboardDrainingRolloutDataSource(SimpleNamespace(use_miles_router=True))
    assert manager.run_miles_router is managed_router.run_managed_router
    managed_router.install()
    assert manager.run_miles_router is managed_router.run_managed_router
