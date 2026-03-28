"""
BounceSim code execution API.

Launch locally with:
```
uv run uvicorn open_instruct.code_utils.ballsim_api:app --host 0.0.0.0 --port 2345
```
"""

import asyncio
import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_instruct import logger_utils
from open_instruct.code_utils.code_utils import decode_tests, get_successful_tests_fast

app = FastAPI()

logger = logger_utils.setup_logger(__name__)
_MAX_CONCURRENT_REQUESTS = max(1, int(os.environ.get("BALLSIM_API_MAX_CONCURRENCY", "8")))
_REQUEST_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)


class TestRequest(BaseModel):
    program: str
    tests: Any
    max_execution_time: float = 1.0


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/test_program")
async def test_program(request: TestRequest) -> dict[str, list[int] | list[float]]:
    try:
        decoded_tests = decode_tests(request.tests)
        async with _REQUEST_SEMAPHORE:
            # Run CPU/multiprocessing-heavy verification off the event loop so concurrent
            # HTTP requests are less likely to be dropped under load.
            results, runtimes = await asyncio.to_thread(
                get_successful_tests_fast, request.program, decoded_tests, request.max_execution_time
            )
        return {"results": results, "runtimes": runtimes}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
