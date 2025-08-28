"""
can launch local server with:
```
uv run nohup uvicorn open_instruct.code_utils.api:app --host 0.0.0.0 --port 7872 &
```

or launch the server in a docker container:
```
docker build -t code-api -f open_instruct/code_utils/Dockerfile .
docker run -p 7872:7872 code-api
```
test with:

curl -X GET http://localhost:7872/health
curl -X POST http://localhost:7872/test_program -H "Content-Type: application/json" -d '{"program": "def add(a, b): return a + b", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"], "max_execution_time": 1.0}'
curl -X POST http://localhost:7872/test_program_stdio -H "Content-Type: application/json" -d '{"program": "import sys\nfor line in sys.stdin.read().splitlines():\n    print(int(line.strip()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}, {"input": "100\n", "output": "101\n"}], "max_execution_time": 1.0}'
curl -X POST http://localhost:7872/view_file -H "Content-Type: application/json" -d '{"repo_name": "john-kurkowski/tldextract", "path": "testbed/test.py", "view_range": [1, 8]}'
curl -X POST http://localhost:7872/edit_file -H "Content-Type: application/json" -d '{"repo_name": "john-kurkowski/tldextract", "command": "view", "path": "testbed"}'
curl -X POST http://localhost:7872/run_bash -H "Content-Type: application/json" -d '{"repo_name": "john-kurkowski/tldextract", "cmd": "ls -la", "cwd": "testbed"}'
"""

import json
import logging
import os
import subprocess
import traceback
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_instruct.code_utils.code_utils import decode_tests, get_successful_tests_fast, get_successful_tests_stdio
from open_instruct import logger_utils

app = FastAPI()

logger = logger_utils.setup_logger(__name__)

# In-memory edit history per repo_path and file path to support undo_edit
_EDIT_HISTORY: dict[str, dict[str, list[str]]] = {}


class TestRequest(BaseModel):
    program: str
    tests: Any
    max_execution_time: float = 1.0


class ViewFileRequest(BaseModel):
    path: str
    view_range: Optional[List[int]] = None
    repo_name: Optional[str] = None
    base_commit: Optional[str] = None
    patches: Optional[List[str]] = None


class RunBashRequest(BaseModel):
    cmd: str
    cwd: Optional[str] = None
    timeout_seconds: int = 60
    repo_name: Optional[str] = None
    base_commit: Optional[str] = None
    patches: Optional[List[str]] = None


class EditFileRequest(BaseModel):
    command: str
    path: str
    view_range: Optional[List[int]] = None
    file_text: Optional[str] = None
    old_str: Optional[str] = None
    new_str: Optional[str] = None
    insert_line: Optional[int] = None
    repo_name: Optional[str] = None
    base_commit: Optional[str] = None
    patches: Optional[List[str]] = None


@app.on_event("startup")
async def startup_initialization():
    """Initialize repos directory structure and clone repos from docker_images.json."""
    try:
        # Ensure base repos directory exists
        repos_dir = os.path.join(os.path.dirname(__file__), "repos")
        if not os.path.exists(repos_dir):
            logger.info(f"Startup: Creating {repos_dir}")
            os.makedirs(repos_dir, exist_ok=True)
        else:
            logger.info(f"Startup: {repos_dir} already exists")

        # Load docker_images.json to get all repo names and base commits
        docker_images_path = os.path.join(os.path.dirname(__file__), "docker_images.json")
        if os.path.exists(docker_images_path):
            with open(docker_images_path, "r") as f:
                docker_images = json.load(f)

            # Clone each repository if it doesn't exist
            for repo_name, repo_info in docker_images.items():
                repo_path = os.path.join(repos_dir, repo_name)
                testbed_path = os.path.join(repo_path, "testbed")

                # Check if testbed already has content (check for .git directory)
                git_dir = os.path.join(testbed_path, ".git")
                if os.path.exists(git_dir):
                    logger.debug(f"Startup: Repository already cloned for {repo_name}")
                    continue

                # Ensure repo directory exists
                if not os.path.exists(repo_path):
                    os.makedirs(repo_path, exist_ok=True)

                # Clone the repository into testbed directory
                github_url = f"https://github.com/{repo_name}.git"
                base_commit = repo_info.get("base_commit")

                logger.info(f"Startup: Cloning {repo_name} at commit {base_commit}")

                try:
                    # Clone the repository
                    clone_result = subprocess.run(
                        ["git", "clone", github_url, "testbed"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                    if clone_result.returncode == 0:
                        # Checkout the specific base commit if provided
                        if base_commit:
                            checkout_result = subprocess.run(
                                ["git", "checkout", base_commit],
                                cwd=testbed_path,
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                            if checkout_result.returncode == 0:
                                logger.info(f"Startup: Successfully cloned {repo_name} at {base_commit}")
                            else:
                                logger.warning(
                                    f"Startup: Cloned {repo_name} but failed to checkout {base_commit}: {checkout_result.stderr}"
                                )
                        else:
                            logger.info(f"Startup: Successfully cloned {repo_name}")
                    else:
                        logger.warning(f"Startup: Failed to clone {repo_name}: {clone_result.stderr}")
                        # Create empty testbed directory as fallback
                        if not os.path.exists(testbed_path):
                            os.makedirs(testbed_path, exist_ok=True)

                except subprocess.TimeoutExpired:
                    logger.warning(f"Startup: Timeout while cloning {repo_name}")
                    # Create empty testbed directory as fallback
                    if not os.path.exists(testbed_path):
                        os.makedirs(testbed_path, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Startup: Error cloning {repo_name}: {e}")
                    # Create empty testbed directory as fallback
                    if not os.path.exists(testbed_path):
                        os.makedirs(testbed_path, exist_ok=True)

        else:
            logger.warning(f"docker_images.json not found at {docker_images_path}")

    except Exception as e:
        logger.warning(f"Startup initialization failed: {e}")


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


def _get_history(repo_path: str) -> dict[str, list[str]]:
    if repo_path not in _EDIT_HISTORY:
        _EDIT_HISTORY[repo_path] = {}
    return _EDIT_HISTORY[repo_path]


def _read_file(abs_path: str) -> str:
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _write_file(abs_path: str, content: str) -> None:
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)


def _get_repo_root(repo_name: Optional[str] = None) -> str:
    """Get the absolute path to the repo directory."""
    base_dir = os.path.join(os.path.dirname(__file__), "repos")
    if repo_name:
        # Use the repo_name as-is (e.g., "john-kurkowski/tldextract")
        return os.path.join(base_dir, repo_name)
    # Default fallback to base repos directory
    return base_dir


def _normalize_path(path: str, repo_name: Optional[str] = None) -> tuple[str, str]:
    """
    Normalize a path for repository access.

    Args:
        path: Input path relative to the repository root
        repo_name: Repository name to determine the correct directory

    Returns:
        tuple: (actual_abs_path, display_path)
            - actual_abs_path: Real filesystem path within the repository
            - display_path: Path to show to user
    """
    # Get the repo root directory (e.g., repos/john-kurkowski/tldextract)
    repo_root = _get_repo_root(repo_name)

    # Ensure the repo directory exists if repo_name is provided
    if repo_name and not os.path.exists(repo_root):
        os.makedirs(repo_root, exist_ok=True)

    # Default to empty string for None or empty path
    if not path:
        path = ""

    # Handle different path formats
    if path.startswith("/"):
        # Remove leading slash for consistency
        full_path = path[1:]
    else:
        # Use path as-is
        full_path = path

    # Default to current directory if empty
    if not full_path:
        full_path = "."

    # Security check: prevent directory traversal
    if ".." in full_path:
        raise ValueError(f"Directory traversal not allowed: {path}")

    # Normalize the path to clean up any ./.. etc
    full_path = os.path.normpath(full_path)

    # Additional security check after normalization
    if full_path.startswith("..") or "/.." in full_path:
        raise ValueError(f"Path traversal not allowed after normalization: {path}")

    # Build the actual filesystem path
    actual_abs_path = os.path.join(repo_root, full_path)
    display_path = full_path

    # Final security check: ensure we're still within repo
    actual_abs_path = os.path.realpath(actual_abs_path)
    repo_root_real = os.path.realpath(repo_root)
    if not actual_abs_path.startswith(repo_root_real):
        raise ValueError(f"Path outside repo not allowed: {path}")

    return actual_abs_path, display_path


def _format_cat_n(content: str, file_descriptor: str, init_line: int = 1) -> str:
    content = content.expandtabs()
    numbered = [f"{i + init_line:6}\t{line}" for i, line in enumerate(content.split("\n"))]
    return f"Here's the result of running `cat -n` on {file_descriptor}:\n" + "\n".join(numbered) + "\n"


@app.post("/edit_file")
async def edit_file_endpoint(request: EditFileRequest):
    """Implements str_replace_editor commands for repository files."""
    try:
        # Normalize and secure the path
        abs_path, display_path = _normalize_path(request.path or "/testbed", request.repo_name)

        # Get history for the repo root
        repo_root = _get_repo_root(request.repo_name)
        hist = _get_history(repo_root)

        if request.command == "view":
            if os.path.isdir(abs_path):
                # Use find to get directory contents and convert paths to display format
                out = subprocess.run(
                    rf"find {abs_path} -maxdepth 2 -not -path '*/\.*'",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout = out.stdout or ""
                if (out.stderr or "").strip() == "":
                    # Convert real paths to display paths
                    repo_root = _get_repo_root(request.repo_name)
                    lines = []
                    for line in stdout.strip().split("\n"):
                        if line.strip():
                            # Convert absolute path to relative path
                            if line.startswith(repo_root):
                                rel_path = line[len(repo_root) :].lstrip("/")
                                lines.append(rel_path if rel_path else ".")

                    content = (
                        "OBSERVATION:\n"
                        f"Here's the files and directories up to 2 levels deep in {display_path}, excluding hidden items:\n"
                        + "\n".join(lines)
                        + ("\n" if lines and not "\n".join(lines).endswith("\n") else "")
                    )
                    return {"content": content}

            if not os.path.exists(abs_path):
                return {"content": f"OBSERVATION:\nPath not found: {display_path}\n"}
            if os.path.isdir(abs_path):
                return {"content": f"OBSERVATION:\n{display_path} is a directory. Try listing it instead.\n"}

            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

            if request.view_range:
                lines = raw.split("\n")
                start, end = request.view_range
                start = max(1, start)
                end = min(len(lines), end)
                raw = "\n".join(lines[start - 1 : end])
                content = _format_cat_n(raw, display_path, start)
            else:
                content = _format_cat_n(raw, display_path)

            return {"content": "OBSERVATION:\n" + content + "\n"}

        elif request.command == "create":
            if request.file_text is None:
                return {"content": "OBSERVATION:\nParameter `file_text` is required for command: create\n"}
            parent = os.path.dirname(abs_path)
            if not os.path.isdir(parent):
                # Try to create parent directory
                try:
                    os.makedirs(parent, exist_ok=True)
                except Exception as e:
                    return {
                        "content": f"OBSERVATION:\nThe parent directory does not exist and couldn't be created: {e}\n"
                    }
            _write_file(abs_path, request.file_text)
            hist.setdefault(abs_path, []).append("")
            return {"content": f"OBSERVATION:\nFile created successfully at: {display_path}\n"}

        elif request.command == "str_replace":
            if request.old_str is None:
                return {"content": "OBSERVATION:\nParameter `old_str` is required for command: str_replace\n"}
            if not os.path.exists(abs_path):
                return {"content": f"OBSERVATION:\nFile not found: {display_path}\n"}
            original = _read_file(abs_path).expandtabs()
            old_str = request.old_str.expandtabs()
            new_str = (request.new_str or "").expandtabs()
            occ = original.count(old_str)
            if occ == 0:
                return {
                    "content": f"OBSERVATION:\nNo replacement was performed, old_str `{old_str}` did not appear verbatim in {display_path}.\n"
                }
            if occ > 1:
                lines = [i + 1 for i, line in enumerate(original.split("\n")) if old_str in line]
                return {
                    "content": f"OBSERVATION:\nNo replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique\n"
                }
            if new_str == old_str:
                return {
                    "content": f"OBSERVATION:\nNo replacement was performed, old_str `{old_str}` is the same as new_str `{new_str}`.\n"
                }
            replaced = original.replace(old_str, new_str)
            _write_file(abs_path, replaced)
            hist.setdefault(abs_path, []).append(original)
            replacement_line = original.split(old_str)[0].count("\n")
            start_line = max(1, replacement_line - 4)
            end_line = min(replacement_line + 4 + new_str.count("\n"), len(replaced.splitlines()))
            snippet = "\n".join(replaced.split("\n")[start_line - 1 : end_line])
            msg = (
                f"The file {display_path} has been edited. "
                + _format_cat_n(snippet, f"a snippet of {display_path}", start_line)
                + "Review the changes and make sure they are as expected. Edit the file again if necessary."
            )
            return {"content": "OBSERVATION:\n" + msg}

        elif request.command == "insert":
            if request.insert_line is None or request.new_str is None:
                return {
                    "content": "OBSERVATION:\nParameter `insert_line` and `new_str` are required for command: insert\n"
                }
            if not os.path.exists(abs_path):
                return {"content": f"OBSERVATION:\nFile not found: {display_path}\n"}
            original = _read_file(abs_path).expandtabs()
            lines = original.split("\n")
            n = len(lines)
            if request.insert_line < 0 or request.insert_line > n:
                return {
                    "content": f"OBSERVATION:\nInvalid `insert_line` parameter: {request.insert_line}. It should be within the range of lines of the file: {[0, n]}\n"
                }
            new_lines = (
                lines[: request.insert_line] + request.new_str.expandtabs().split("\n") + lines[request.insert_line :]
            )
            new_text = "\n".join(new_lines)
            _write_file(abs_path, new_text)
            hist.setdefault(abs_path, []).append(original)
            snippet_lines = (
                lines[max(0, request.insert_line - 4) : request.insert_line]
                + request.new_str.expandtabs().split("\n")
                + lines[request.insert_line : request.insert_line + 4]
            )
            snippet = "\n".join(snippet_lines)
            msg = (
                f"The file {display_path} has been edited. "
                + _format_cat_n(snippet, "a snippet of the edited file", max(1, request.insert_line - 4 + 1))
                + "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
            )
            return {"content": "OBSERVATION:\n" + msg}

        elif request.command == "undo_edit":
            history = hist.get(abs_path, [])
            if not history:
                return {"content": f"OBSERVATION:\nNo edit history found for {display_path}.\n"}
            last_text = history.pop()
            _write_file(abs_path, last_text)
            msg = f"Last edit to {display_path} undone successfully. {_format_cat_n(last_text, display_path)}"
            return {"content": "OBSERVATION:\n" + msg}

        else:
            return {
                "content": f'OBSERVATION:\nUnrecognized command {request.command}. The allowed commands are: "view", "create", "str_replace", "insert", "undo_edit"\n'
            }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_bash")
async def run_bash_endpoint(request: RunBashRequest):
    try:
        # For bash commands, we run from the repo root, not testbed
        repo_root = _get_repo_root(request.repo_name)

        # If cwd is provided, it's relative to repo root
        if request.cwd:
            # Remove leading slash if present
            cwd_path = request.cwd[1:] if request.cwd.startswith("/") else request.cwd
            abs_cwd = os.path.join(repo_root, cwd_path)
            display_cwd = cwd_path
        else:
            # Default to repo root
            abs_cwd = repo_root
            display_cwd = "."

        # Ensure the directory exists
        if not os.path.exists(abs_cwd):
            os.makedirs(abs_cwd, exist_ok=True)

        try:
            completed = subprocess.run(
                ["/bin/bash", "-lc", request.cmd],
                cwd=abs_cwd,
                capture_output=True,
                text=True,
                timeout=max(1, int(request.timeout_seconds)),
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            rc = completed.returncode

            # Format command display
            if display_cwd == ".":
                cmd_prefix = f"$ {request.cmd}\n\n"
            else:
                cmd_prefix = f"$ cd {display_cwd} && {request.cmd}\n\n"

            content = (
                "OBSERVATION:\n"
                + cmd_prefix
                + (stdout if len(stdout) > 0 else "")
                + ("\n" if len(stdout) > 0 and len(stderr) > 0 else "")
                + (f"[stderr]\n{stderr}" if len(stderr) > 0 else "")
                + ("\n" if rc != 0 else "")
                + (f"[exit {rc}]" if rc != 0 else "")
            )
            return {"content": content, "returncode": rc}
        except subprocess.TimeoutExpired:
            return {
                "content": f"OBSERVATION:\nTimeout after {request.timeout_seconds} seconds while running: {request.cmd}\n",
                "timeout": True,
            }
    except HTTPException:
        raise
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
        # Normalize and secure the path
        abs_path, display_path = _normalize_path(request.path or "/testbed", request.repo_name)

        # Check if file exists
        if not os.path.exists(abs_path):
            return {"content": f"OBSERVATION:\nPath not found: {display_path}\n"}

        if os.path.isdir(abs_path):
            return {
                "content": f"OBSERVATION:\n{display_path} is a directory. Use the directory listing functionality instead.\n"
            }

        # Read the file
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        # Apply view range if specified
        if request.view_range:
            lines = raw.split("\n")
            start, end = request.view_range
            start = max(1, start)
            end = min(len(lines), end)
            raw = "\n".join(lines[start - 1 : end])
            content = _format_cat_n(raw, display_path, start)
        else:
            content = _format_cat_n(raw, display_path)

        return {"content": "OBSERVATION:\n" + content + "\n"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
