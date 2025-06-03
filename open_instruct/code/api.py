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

For AppWorld tasks:
curl -X POST http://localhost:1234/execute_app_world -H "Content-Type: application/json" -d '{"program": "import apis\npasswords = apis.supervisor.show_account_passwords()\nspotify_password = next(p[\"password\"] for p in passwords if p[\"account_name\"] == \"spotify\")\napis.supervisor.complete_task(answer=spotify_password)", "dataset_name": "train", "max_execution_time": 30.0}'
"""

import logging
import os
import signal
import subprocess
import tempfile
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from appworld import AppWorld
from open_instruct.code.appworld_utils import resolve_task_id, execute_program_in_world
from open_instruct.code.code_utils import get_successful_tests_fast, full_undo_reliability_guard

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRequest(BaseModel):
    program: str
    tests: List[str]
    max_execution_time: float = 1.0

class ExecuteAppWorldRequest(BaseModel):
    program: str
    task_id: Optional[str] = None
    dataset_name: str = "train"  # train, dev, test_normal, test_challenge
    experiment_name: str = "api_execution"
    max_execution_time: float = 5.0
    max_interactions: int = 50
    tests: Optional[List[str]] = None  # Optional custom tests to run after execution

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

@app.post("/execute_app_world")
async def execute_app_world(request: ExecuteAppWorldRequest):
    """Execute an AppWorld task with the provided program."""
    # Ensure reliability guard is undone in case test_program endpoint was called before
    try:
        full_undo_reliability_guard()
    except Exception as guard_error:
        logger.warning(f"Could not undo reliability guard: {guard_error}")
    
    try:
        # Resolve task ID
        task_id = resolve_task_id(request.task_id, request.dataset_name)
        logger.info(f"Executing AppWorld program for task: {task_id}")
        
        # Initialize AppWorld environment and execute
        with AppWorld(
            task_id=task_id, 
            experiment_name=request.experiment_name,
            timeout_seconds=request.max_execution_time,
            max_interactions=request.max_interactions
        ) as world:
            result = execute_program_in_world(world, request.program, request.tests)
            return result
                
    except ValueError as ve:
        # Handle dataset/task ID resolution errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"AppWorld execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    
    # Test AppWorld endpoint
    print("\n" + "="*50)
    print("Testing AppWorld endpoint...")
    print("="*50)
    
    appworld_url = "http://localhost:1234/execute_app_world"
    
    # Simple AppWorld test program
    appworld_payload = {
        "program": """
# Get supervisor passwords
passwords = apis.supervisor.show_account_passwords()
print("Available accounts:", [p["account_name"] for p in passwords])

# Find spotify password
spotify_password = next(p["password"] for p in passwords if p["account_name"] == "spotify")
print("Spotify password found:", spotify_password)

# Complete the task with the answer
apis.supervisor.complete_task(answer=spotify_password)
""",
        "dataset_name": "train",
        "max_execution_time": 30.0
    }

    try:
        appworld_response = requests.post(appworld_url, json=appworld_payload)
        appworld_response_json = appworld_response.json()
        print(appworld_response_json)

        print("AppWorld Status Code:", appworld_response.status_code)
        print("AppWorld Response Keys:", list(appworld_response_json.keys()))
        print("Task Completed:", appworld_response_json.get('task_completed', False))
        print("Success:", appworld_response_json.get('success', False))
        
        if 'task_info' in appworld_response_json:
            print("Task ID:", appworld_response_json['task_info']['task_id'])
            print("Task Instruction:", appworld_response_json['task_info']['instruction'][:100] + "...")
            
    except Exception as e:
        print(f"AppWorld test failed: {e}")
        print("Note: AppWorld test requires AppWorld to be installed and configured properly.")