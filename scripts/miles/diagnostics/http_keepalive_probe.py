"""CPU-only HTTP burst reproduction; synthetic 8 MiB route responses, no model.

Run in the qualified runtime image with 4 CPUs and 12 GiB memory. Three paired
trials differ only in Uvicorn idle keep-alive, five versus sixty seconds.
This isolates a possible transport failure mechanism, not GPU throughput.
"""

import asyncio
import base64
import json
import multiprocessing
import socket
import statistics
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

PAYLOAD = json.dumps(
    {"meta_info": {"routed_experts": base64.b64encode(b"routeids" * (6 * 1024 * 1024 // 8)).decode()}}
).encode()


def serve(port, raw, keepalive):
    app = FastAPI()

    @app.post("/generate")
    async def generate(req: Request):
        await req.body()
        await asyncio.sleep(0.1)
        return Response(PAYLOAD, media_type="application/json") if raw else JSONResponse(json.loads(PAYLOAD))

    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False, log_level="critical", timeout_keep_alive=keepalive)


async def exercise(port, n):
    errors = []
    lat = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=n), timeout=httpx.Timeout(60)) as client:

        async def one():
            t = time.perf_counter()
            try:
                response = await client.post(f"http://127.0.0.1:{port}/generate", json={"prompt": [1] * 1024})
                response.raise_for_status()
                value = response.json()
                assert len(value["meta_info"]["routed_experts"]) == 8 * 1024 * 1024
                lat.append(time.perf_counter() - t)
            except Exception as e:
                errors.append(type(e).__name__ + ":" + str(e))

        start = time.perf_counter()

        # Immediate reuse after a completion, as in a continuously fed producer.
        async def slot():
            for _ in range(3):
                await one()

        await asyncio.gather(*(slot() for _ in range(n)))
        return {
            "seconds": time.perf_counter() - start,
            "completed": len(lat),
            "errors": errors,
            "median_request_seconds": statistics.median(lat) if lat else None,
        }


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    for raw, keepalive in [(False, 5), (False, 60)] * 3:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        proc = multiprocessing.Process(target=serve, args=(port, raw, keepalive))
        proc.start()
        try:
            for _ in range(100):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.1)
            for n in (224,):
                row = asyncio.run(exercise(port, n))
                row.update(raw=raw, keepalive=keepalive, concurrency=n, wire_bytes=len(PAYLOAD))
                print(json.dumps(row), flush=True)
        finally:
            proc.terminate()
            proc.join(5)
