"""
can launch the server with:

```
uv run uvicorn --host 0.0.0.0 open_instruct.code_api:app
```

and then test with:

```
python open_instruct/api.py
```
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from open_instruct.code_utils import get_successful_tests_fast

app = FastAPI()

class TestRequest(BaseModel):
    program: str
    tests: List[str]
    max_execution_time: float = 1.0

@app.post("/test_program")
async def test_program(request: TestRequest):
    try:
        results = get_successful_tests_fast(
            program=request.program,
            tests=request.tests,
            max_execution_time=request.max_execution_time
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
    url = "http://phobos-cs-aus-453.reviz.ai2.in:8000/test_program"

    # Test data
    payload = {
        "program": """
    def add(a, b):
        return a + b
    """,
        "tests": [
            "assert add(1, 2) == 3",
            "assert add(-1, 1) == 0",
            "assert add(0, 0) == 1"  # This test will fail
        ],
        "max_execution_time": 1.0
    }

    # Send POST request
    response = requests.post(url, json=payload)

    # Print results
    print("Status Code:", response.status_code)
    print("Response:", response.json()) 