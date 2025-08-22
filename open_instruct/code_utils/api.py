"""
can launch local server with:
```
uv run nohup uvicorn open_instruct.code_utils.api:app --host 0.0.0.0 --port 1234 &
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
import subprocess
import tempfile
import threading
import traceback
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_instruct.code_utils.code_utils import (
    clone_repo,
    decode_tests,
    get_successful_tests_fast,
    get_successful_tests_stdio,
    view_file,
)
from open_instruct import logger_utils
from open_instruct.code_utils.code_utils import decode_tests, get_successful_tests_fast, get_successful_tests_stdio

app = FastAPI()

logger = logger_utils.setup_logger(__name__)


class TestRequest(BaseModel):
    program: str
    tests: Any
    max_execution_time: float = 1.0


class ViewFileRequest(BaseModel):
    repo_name: str
    path: str
    view_range: Optional[List[int]] = None
    base_commit: Optional[str] = None
    patches: Optional[List[str]] = None


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

        # Acquire a per-repo lock to avoid patch races
        repo_lock = _get_repo_lock(request.repo_name)
        repo_lock.acquire()
        try:
            # Always hard reset to the base commit before applying patches
            base_commit = repo_info.get("base_commit")
            if request.base_commit:
                base_commit = request.base_commit
            if base_commit:
                _git_checkout(repo_path, base_commit)

            # Apply patches (if any), serve content, then revert
            if request.patches:
                _apply_patches(repo_path, request.patches)

            # View the file
            file_content = "OBSERVATION:\n" + view_file(repo_path, request.path, request.view_range) + "\n"

            # Perform cleanup BEFORE releasing the lock/returning, to guarantee no state leak
            try:
                _git_reset_hard(repo_path)
                if base_commit:
                    _git_checkout(repo_path, base_commit)
            except Exception as cleanup_err:
                logger.warning(f"Cleanup after view_file failed: {cleanup_err}")
            finally:
                repo_lock.release()

            return {"content": file_content, "repo_path": repo_path}
        except Exception:
            # Ensure the lock is released in case of unexpected exceptions
            try:
                repo_lock.release()
            except Exception:
                pass
            raise
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------------
# Internal helpers for safe patch application and cleanup
# --------------------------------------------------------------------------------------

_REPO_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _get_repo_lock(repo_name: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _REPO_LOCKS.get(repo_name)
        if lock is None:
            lock = threading.Lock()
            _REPO_LOCKS[repo_name] = lock
        return lock


def _run_git(repo_path: str, args: List[str]) -> None:
    completed = subprocess.run(["git", "-C", repo_path] + args, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"git command failed: git -C {repo_path} {' '.join(args)}\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
        )


def _git_checkout(repo_path: str, commit: str) -> None:
    # Force checkout to avoid local changes interference
    _run_git(repo_path, ["checkout", "-f", commit])


def _git_reset_hard(repo_path: str) -> None:
    _run_git(repo_path, ["reset", "--hard"])
    _run_git(repo_path, ["clean", "-fd"])


def _apply_patches(repo_path: str, patches: List[str]) -> None:
    for patch_text in patches:
        if not patch_text:
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tf:
            tf.write(patch_text)
            tf.flush()
            patch_file = tf.name
        # Try p0 then p1 for path stripping robustness
        errors: list[str] = []
        for strip in ("0", "1"):
            try:
                _run_git(repo_path, ["apply", "--whitespace=nowarn", "-p", strip, patch_file])
                break
            except Exception as e:
                errors.append(str(e))
        else:
            # If both attempts failed, surface a concise error
            raise RuntimeError(
                "Failed to apply patch with -p0 and -p1. Last error: " + (errors[-1] if errors else "unknown")
            )
