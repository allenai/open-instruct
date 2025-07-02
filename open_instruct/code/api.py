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
import traceback
from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_instruct.code.code_utils import (
    get_successful_tests_fast,
    get_successful_tests_stdio,
)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRequest(BaseModel):
    program: str
    tests: List[Any]
    max_execution_time: float = 1.0


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/test_program")
async def test_program(request: TestRequest):
    try:
        # logger.info("Executing tests for program: %s", request.program)
        results = get_successful_tests_fast(
            program=request.program, tests=request.tests, max_execution_time=request.max_execution_time
        )
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test_program_stdio")
async def test_program_stdio(request: TestRequest):
    # run tests with the stdio format
    try:
        results = get_successful_tests_stdio(
            program=request.program, tests=request.tests, max_execution_time=request.max_execution_time
        )
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import requests

    stdio_url = "http://localhost:1234/test_program_stdio"

    program1 = """
X = int(input())
total_sum = 2025

count_X = 0
for i in range(1, 10):
    if X % i == 0:
        j = X // i
        if 1 <= j <= 9:
            count_X += 1

result = total_sum - (count_X * X)
print(result)
"""
    stdio_payload = {
        "program": program1,
        "tests": [
            {"input": "1", "output": "2024"},
            {"input": "11", "output": "2025"},
            {"input": "24", "output": "1929"},
            {"input": "1", "output": "1"},
            {"input": "1", "output": "2024"},
            {"input": "11", "output": "2025"},
            {"input": "24", "output": "1"},
        ],
        "max_execution_time": 6.0,
    }

    response = requests.post(stdio_url, json=stdio_payload)

    response_json = response.json()
    print("Status Code:", response.status_code)
    print("Response:", response_json)

    assert response_json["results"] == [1, 1, 1, 0, 1, 1, 0]

    # test multi-line inputs
    program2 = """
def is_prime(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num**0.5)+1, 2):
        if num % i == 0:
            return False
    return True

def generate_primes_up_to_nth(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

def main():
    import sys
    inputs = sys.stdin.read().split()
    for line in inputs:
        n = int(line)
        if n == 0:
            break
        primes = generate_primes_up_to_nth(n)
        print(sum(primes))

if __name__ == "__main__":
    main()
"""
    stdio_payload = {
        "program": program2,
        "tests": [
            {"input": "2\n9\n0\n", "output": "5\n100\n"},
            {"input": "1\n0\n", "output": "2\n"},
            {"input": "0\n", "output": ""},
            {"input": "10\n0\n", "output": "129\n"},
            {"input": "5\n3\n2\n0\n", "output": "28\n10\n5\n"},
            {"input": "10000\n0\n", "output": "496165411\n"},
            {"input": "10000\n0\n", "output": "goop\n"},
            {"input": "1\n2\n3\n4\n0\n", "output": "2\n5\n10\n17\n"},
            {"input": "3\n1\n0\n", "output": "10\n2\n"},
            {"input": "3\n1\n0\n", "output": "10\n3\n"},
        ],
        "max_execution_time": 6.0,
    }

    response = requests.post(stdio_url, json=stdio_payload)

    response_json = response.json()
    print("Status Code:", response.status_code)
    print("Response:", response_json)

    assert response_json["results"] == [1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
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
