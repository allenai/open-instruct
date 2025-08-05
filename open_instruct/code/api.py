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
curl -X POST http://localhost:1234/test_program_stdio -H "Content-Type: application/json" -d '{"program": "import sys\nfor line in sys.stdin.read().splitlines():\n    print(int(line.strip()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}, {"input": "100\n", "output": "101\n"}], "max_execution_time": 1.0}'
curl -X POST http://localhost:1234/view_file -H "Content-Type: application/json" -d '{"repo_name": "joke2k/faker", "path": "faker/__init__.py", "view_range": [1, 20]}'
"""

import json
import logging
import os
import traceback
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_instruct.code.code_utils import (
    clone_repo,
    decode_tests,
    get_successful_tests_fast,
    get_successful_tests_stdio,
    view_file,
)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRequest(BaseModel):
    program: str
    tests: Any
    max_execution_time: float = 1.0


class ViewFileRequest(BaseModel):
    repo_name: str
    path: str
    view_range: Optional[List[int]] = None
    base_commit: Optional[str] = None


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/test_program")
async def test_program(request: TestRequest):
    try:
        # logger.info("Executing tests for program: %s", request.program)
        decoded_tests = decode_tests(request.tests)
        results, runtimes = get_successful_tests_fast(
            program=request.program, tests=decoded_tests, max_execution_time=request.max_execution_time
        )
        return {"results": results, "runtimes": runtimes}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test_program_stdio")
async def test_program_stdio(request: TestRequest):
    # run tests with the stdio format
    try:
        decoded_tests = decode_tests(request.tests)
        results, runtimes = get_successful_tests_stdio(
            program=request.program, tests=decoded_tests, max_execution_time=request.max_execution_time
        )
        return {"results": results, "runtimes": runtimes}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/view_file")
async def view_file_endpoint(request: ViewFileRequest):
    try:
        # Load the docker images config
        config_path = os.path.join(os.path.dirname(__file__), "docker_images.json")
        with open(config_path, "r") as f:
            docker_images = json.load(f)

        # Get repo info from docker_images.json
        if request.repo_name not in docker_images:
            raise HTTPException(status_code=404, detail=f"Repository {request.repo_name} not found in configuration")

        repo_info = docker_images[request.repo_name]

        # Override base_commit if provided in request
        if request.base_commit:
            repo_info["base_commit"] = request.base_commit

        # Set up the config for clone_repo
        config = {"call_paths": {"default": {"repo_domain": "github.com"}}, "home_dir": os.getcwd()}

        # Add repo name to repo_info
        repo_info["name"] = request.repo_name

        # Clone the repository and get the path
        repo_path = clone_repo(repo_info, config)

        # View the file
        file_content = "OBSERVATION:\n" + view_file(repo_path, request.path, request.view_range) + "\n"

        return {"content": file_content, "repo_path": repo_path}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
