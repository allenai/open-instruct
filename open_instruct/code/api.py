"""
can launch local server with:
```
uv run nohup uvicorn open_instruct.code.api:app --host 0.0.0.0 --port 1234 &
```

or launch the server in a docker container:
```
docker build -t code-api -f open_instruct/code/Dockerfile .
docker run -p 1234:1234 code-api
```

and then test with:
```
python open_instruct/code/api.py
```

or

curl -X GET http://localhost:1234/health
curl -X POST http://localhost:1234/test_program -H "Content-Type: application/json" -d '{"program": "def add(a, b): return a + b", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"], "max_execution_time": 1.0}'
"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .code_utils import get_successful_tests_fast

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRequest(BaseModel):
    program: str
    tests: List[str]
    max_execution_time: float = 1.0


@app.post("/test_program")
async def test_program(request: TestRequest):
    try:
        # logger.info("Executing tests for program: %s", request.program)
        results = get_successful_tests_fast(
            program=request.program, tests=request.tests, max_execution_time=request.max_execution_time
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import requests

    # API endpoint
    url = "http://localhost:1234/test_program"

    # Test data
    payload = {
        "program": """
def add(a, b):
    return a + b
""",
        "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"],  # This test will fail
        "max_execution_time": 1.0,
    }

    # Send POST request
    response = requests.post(url, json=payload)

    response_json = response.json()
    # Print results
    print("Status Code:", response.status_code)
    print("Response:", response_json)

    assert response_json["results"] == [1, 1, 0]
