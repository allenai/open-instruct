from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .code_utils import get_successful_tests_fast
import logging
import traceback
import os
import time
import uvicorn

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure log directory exists
log_dir = os.path.expanduser('~/../weka/oe-adapt-default/saurabhs/repos/open-instruct/open_instruct/code')
os.makedirs(log_dir, exist_ok=True)

class TestRequest(BaseModel):
    program: str
    tests: List[str]
    max_execution_time: float = 10.0

@app.post("/test_program")
async def test_program(request: TestRequest):
    start_time = time.time()
    try:
        logger.info("Executing tests for program: %s", request.program)
        results = get_successful_tests_fast(
            program=request.program,
            tests=request.tests,
            max_execution_time=request.max_execution_time
        )
        execution_time = time.time() - start_time
        logger.info(f"Test execution completed in {execution_time:.2f} seconds")
        return {"results": results}
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error after {execution_time:.2f} seconds: {str(e)}")
        logger.error(traceback.format_exc())
        # log to the file at '~/../weka/oe-adapt-default/saurabhs/repos/open_instruct/code/api.log'
        # write the entire traceback
        log_file = os.path.join(log_dir, 'api.log')
        with open(log_file, 'a') as f:
            f.write(f"Error after {execution_time:.2f} seconds: {e}\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*80 + "\n") 
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def test_program():
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

    response_json = response.json()
    # Print results
    print("Status Code:", response.status_code)
    print("Response:", response_json) 

    assert response_json['results'] == [1, 1, 0]


if __name__ == "__main__":
    uvicorn.run(
        "open_instruct.code.api:app",  # Use import string format
        host="0.0.0.0",
        port=8000,
        workers=32,
        log_level="debug",
        timeout_keep_alive=300,
        limit_concurrency=2000, 
        backlog=2048
    )
