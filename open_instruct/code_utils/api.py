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
curl -X POST http://localhost:7872/view_file -H "Content-Type: application/json" -d '{"repo_name": "joke2k/faker", "path": "faker/__init__.py", "view_range": [1, 8]}'
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

# In-memory edit history per repo_path and file path to support undo_edit
_EDIT_HISTORY: dict[str, dict[str, list[str]]] = {}


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


class RunBashRequest(BaseModel):
    repo_name: str
    cmd: str
    cwd: Optional[str] = None
    base_commit: Optional[str] = None
    patches: Optional[List[str]] = None
    timeout_seconds: int = 60


class EditFileRequest(BaseModel):
    repo_name: str
    command: str
    path: str
    view_range: Optional[List[int]] = None
    file_text: Optional[str] = None
    old_str: Optional[str] = None
    new_str: Optional[str] = None
    insert_line: Optional[int] = None
    base_commit: Optional[str] = None
    patches: Optional[List[str]] = None


@app.on_event("startup")
async def startup_clone_all_repos():
    try:
        # Ensure /testbed exists
        REPO_DIR = os.path.join(os.path.dirname(__file__), "testbed")
        os.makedirs(REPO_DIR, exist_ok=True)
        # Load docker images config
        config_path = os.path.join(os.path.dirname(__file__), "docker_images.json")
        logger.info(f"Startup: loading docker images from {config_path}")
        with open(config_path, "r") as f:
            docker_images = json.load(f)
        config = {"call_paths": {"default": {"repo_domain": "github.com"}}, "home_dir": os.getcwd()}
        # Clone each repo if missing
        for repo_name, repo_info in docker_images.items():
            try:
                info = dict(repo_info)
                info["name"] = repo_name
                clone_repo(info, config)
                logger.info(f"Startup: ensured clone for {repo_name}")
            except Exception as e:
                logger.warning(f"Startup: failed to clone {repo_name}: {e}")
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


def _repo_file_abs(repo_path: str, rel_path: str) -> str:
    return os.path.join(repo_path, rel_path)


def _read_file(abs_path: str) -> str:
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _write_file(abs_path: str, content: str) -> None:
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)


def _format_cat_n(content: str, file_descriptor: str, init_line: int = 1) -> str:
    content = content.expandtabs()
    numbered = [f"{i + init_line:6}\t{line}" for i, line in enumerate(content.split("\n"))]
    return f"Here's the result of running `cat -n` on {file_descriptor}:\n" + "\n".join(numbered) + "\n"


@app.post("/edit_file")
async def edit_file_endpoint(request: EditFileRequest):
    """Implements str_replace_editor commands with matching formatting."""
    try:
        # Load docker images config
        config_path = os.path.join(os.path.dirname(__file__), "docker_images.json")
        with open(config_path, "r") as f:
            docker_images = json.load(f)
        # Graceful fallback: allow viewing /testbed even if repo isn't configured
        allow_testbed_view = False
        if request.repo_name not in docker_images:
            if request.command == "view" and (request.repo_name in {"testbed", "/testbed"}):
                allow_testbed_view = True
            else:
                raise HTTPException(status_code=404, detail=f"Repository {request.repo_name} not found in configuration")

        repo_info = docker_images.get(request.repo_name, {})
        if request.base_commit and repo_info is not None:
            repo_info["base_commit"] = request.base_commit

        config = {"call_paths": {"default": {"repo_domain": "github.com"}}, "home_dir": os.getcwd()}
        repo_path = "/testbed"
        if not allow_testbed_view:
            repo_info["name"] = request.repo_name
            repo_path = clone_repo(repo_info, config)

        # Normalize rel path
        rel_path = request.path or ""
        if rel_path.startswith("/testbed/"):
            rel_path = rel_path[9:]
        elif rel_path.startswith("testbed/"):
            rel_path = rel_path[8:]

        repo_lock = _get_repo_lock(request.repo_name)
        repo_lock.acquire()
        try:
            is_git_repo = not allow_testbed_view
            base_commit = repo_info.get("base_commit") if is_git_repo else None
            if request.base_commit:
                base_commit = request.base_commit
            if is_git_repo and base_commit:
                _git_checkout(repo_path, base_commit)
            if is_git_repo and request.patches:
                _apply_patches(repo_path, request.patches)

            hist = _get_history(repo_path)
            # Resolve absolute paths that point to /testbed
            if os.path.isabs(rel_path):
                abs_path = rel_path
            else:
                abs_path = _repo_file_abs(repo_path, rel_path)

            if request.command == "view":
                # Directory view support like str_replace_editor
                full_display_path = f"/testbed/{rel_path}" if rel_path else "/testbed"
                if os.path.isdir(abs_path):
                    out = subprocess.run(
                        rf"find {abs_path} -maxdepth 2 -not -path '*/\.*'",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    stdout = out.stdout or ""
                    if (out.stderr or "").strip() == "":
                        content = (
                            "OBSERVATION:\n"
                            f"Here's the files and directories up to 2 levels deep in {full_display_path}, excluding hidden items:\n"
                            + stdout
                            + ("\n" if not stdout.endswith("\n") else "")
                        )
                        return {"content": content}
                    # Fallthrough to file rendering on errors
                # If the request is targeting an absolute file under /testbed, read it directly
                if os.path.isabs(abs_path):
                    if not os.path.exists(abs_path):
                        return {"content": f"OBSERVATION:\nPath not found: {abs_path}\n"}
                    if os.path.isdir(abs_path):
                        return {"content": f"OBSERVATION:\n{abs_path} is a directory. Try listing it instead.\n"}
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        raw = f.read()
                    return {"content": "OBSERVATION:\n" + _format_cat_n(raw, abs_path) + "\n"}
                content = view_file(repo_path, rel_path, request.view_range)
                return {"content": "OBSERVATION:\n" + content + "\n"}

            elif request.command == "create":
                if request.file_text is None:
                    return {"content": "OBSERVATION:\nParameter `file_text` is required for command: create\n"}
                parent = os.path.dirname(abs_path)
                if not os.path.isdir(parent):
                    return {"content": f"OBSERVATION:\nThe parent directory {parent} does not exist. Please create it first.\n"}
                _write_file(abs_path, request.file_text)
                hist.setdefault(rel_path, []).append("")
                return {"content": f"OBSERVATION:\nFile created successfully at: /testbed/{rel_path}\n"}

            elif request.command == "str_replace":
                if request.old_str is None:
                    return {"content": "OBSERVATION:\nParameter `old_str` is required for command: str_replace\n"}
                original = _read_file(abs_path).expandtabs()
                old_str = request.old_str.expandtabs()
                new_str = (request.new_str or "").expandtabs()
                occ = original.count(old_str)
                if occ == 0:
                    return {"content": f"OBSERVATION:\nNo replacement was performed, old_str `{old_str}` did not appear verbatim in /testbed/{rel_path}.\n"}
                if occ > 1:
                    lines = [i + 1 for i, line in enumerate(original.split("\n")) if old_str in line]
                    return {"content": f"OBSERVATION:\nNo replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique\n"}
                if new_str == old_str:
                    return {"content": f"OBSERVATION:\nNo replacement was performed, old_str `{old_str}` is the same as new_str `{new_str}`.\n"}
                replaced = original.replace(old_str, new_str)
                _write_file(abs_path, replaced)
                hist.setdefault(rel_path, []).append(original)
                replacement_line = original.split(old_str)[0].count("\n")
                start_line = max(1, replacement_line - 4)
                end_line = min(replacement_line + 4 + new_str.count("\n"), len(replaced.splitlines()))
                snippet = "\n".join(replaced.split("\n")[start_line - 1 : end_line])
                msg = (
                    f"The file /testbed/{rel_path} has been edited. "
                    + _format_cat_n(snippet, f"a snippet of /testbed/{rel_path}", start_line)
                    + "Review the changes and make sure they are as expected. Edit the file again if necessary."
                )
                return {"content": "OBSERVATION:\n" + msg}

            elif request.command == "insert":
                if request.insert_line is None or request.new_str is None:
                    return {"content": "OBSERVATION:\nParameter `insert_line` and `new_str` are required for command: insert\n"}
                original = _read_file(abs_path).expandtabs()
                lines = original.split("\n")
                n = len(lines)
                if request.insert_line < 0 or request.insert_line > n:
                    return {"content": f"OBSERVATION:\nInvalid `insert_line` parameter: {request.insert_line}. It should be within the range of lines of the file: {[0, n]}\n"}
                new_lines = lines[: request.insert_line] + request.new_str.expandtabs().split("\n") + lines[request.insert_line :]
                new_text = "\n".join(new_lines)
                _write_file(abs_path, new_text)
                hist.setdefault(rel_path, []).append(original)
                snippet_lines = (
                    lines[max(0, request.insert_line - 4) : request.insert_line]
                    + request.new_str.expandtabs().split("\n")
                    + lines[request.insert_line : request.insert_line + 4]
                )
                snippet = "\n".join(snippet_lines)
                msg = (
                    f"The file /testbed/{rel_path} has been edited. "
                    + _format_cat_n(snippet, "a snippet of the edited file", max(1, request.insert_line - 4 + 1))
                    + "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
                )
                return {"content": "OBSERVATION:\n" + msg}

            elif request.command == "undo_edit":
                history = hist.get(rel_path, [])
                if not history:
                    return {"content": f"OBSERVATION:\nNo edit history found for /testbed/{rel_path}.\n"}
                last_text = history.pop()
                _write_file(abs_path, last_text)
                msg = f"Last edit to /testbed/{rel_path} undone successfully. {_format_cat_n(last_text, f'/testbed/{rel_path}')}"
                return {"content": "OBSERVATION:\n" + msg}

            else:
                return {"content": f"OBSERVATION:\nUnrecognized command {request.command}. The allowed commands are: \"view\", \"create\", \"str_replace\", \"insert\", \"undo_edit\"\n"}
        finally:
            try:
                if is_git_repo:
                    _git_reset_hard(repo_path)
                    if base_commit:
                        _git_checkout(repo_path, base_commit)
            except Exception as cleanup_err:
                logger.warning(f"Cleanup after edit_file failed: {cleanup_err}")
            finally:
                repo_lock.release()
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/run_bash")
async def run_bash_endpoint(request: RunBashRequest):
    try:
        # Load the docker images config
        config_path = os.path.join(os.path.dirname(__file__), "docker_images.json")
        logger.info(f"Loading docker images from {config_path}")
        with open(config_path, "r") as f:
            docker_images = json.load(f)

        # Get repo info from docker_images.json
        allow_testbed = request.repo_name in {"testbed", "/testbed"}
        if request.repo_name not in docker_images and not allow_testbed:
            raise HTTPException(status_code=404, detail=f"Repository {request.repo_name} not found in configuration")

        repo_info = docker_images.get(request.repo_name, {})

        # Override base_commit if provided in request
        if request.base_commit:
            repo_info["base_commit"] = request.base_commit

        config = {"call_paths": {"default": {"repo_domain": "github.com"}}, "home_dir": os.getcwd()}
        repo_path = "/testbed" if allow_testbed else None
        if not allow_testbed:
            repo_info["name"] = request.repo_name
            # Clone repo and get path
            repo_path = clone_repo(repo_info, config)

        # Normalize cwd
        cwd = request.cwd or "."
        if cwd.startswith("/testbed/"):
            cwd = cwd[9:]
        elif cwd.startswith("testbed/"):
            cwd = cwd[8:]

        # Acquire a per-repo lock to avoid concurrent patch/command races
        repo_lock = _get_repo_lock(request.repo_name)
        repo_lock.acquire()
        try:
            is_git_repo = not allow_testbed
            base_commit = repo_info.get("base_commit") if is_git_repo else None
            if request.base_commit:
                base_commit = request.base_commit
            if is_git_repo and base_commit:
                _git_checkout(repo_path, base_commit)
            if is_git_repo and request.patches:
                _apply_patches(repo_path, request.patches)

            # Execute the bash command
            full_cwd = os.path.join(repo_path, cwd)
            if not os.path.exists(full_cwd):
                raise HTTPException(status_code=400, detail=f"cwd does not exist: {cwd}")

            try:
                completed = subprocess.run(
                    ["/bin/bash", "-lc", request.cmd],
                    cwd=full_cwd,
                    capture_output=True,
                    text=True,
                    timeout=max(1, int(request.timeout_seconds)),
                )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                rc = completed.returncode
                content = (
                    "OBSERVATION:\n" +
                    (f"$ cd {cwd} && {request.cmd}\n\n" if cwd != "." else f"$ {request.cmd}\n\n") +
                    (stdout if len(stdout) > 0 else "") +
                    ("\n" if len(stdout) > 0 and len(stderr) > 0 else "") +
                    (f"[stderr]\n{stderr}" if len(stderr) > 0 else "") +
                    ("\n" if rc != 0 else "") +
                    (f"[exit {rc}]" if rc != 0 else "")
                )
                return {"content": content, "repo_path": repo_path, "returncode": rc}
            except subprocess.TimeoutExpired:
                return {
                    "content": f"OBSERVATION:\nTimeout after {request.timeout_seconds} seconds while running: {request.cmd}\n",
                    "repo_path": repo_path,
                    "timeout": True,
                }
            finally:
                try:
                    if is_git_repo:
                        _git_reset_hard(repo_path)
                        if base_commit:
                            _git_checkout(repo_path, base_commit)
                except Exception as cleanup_err:
                    logger.warning(f"Cleanup after run_bash failed: {cleanup_err}")
                finally:
                    repo_lock.release()
        except Exception:
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
        logger.info(f"Loading docker images from {config_path}")
        with open(config_path, "r") as f:
            docker_images = json.load(f)

        # Get repo info from docker_images.json
        allow_testbed = request.repo_name in {"testbed", "/testbed"}
        if request.repo_name not in docker_images and not allow_testbed:
            raise HTTPException(status_code=404, detail=f"Repository {request.repo_name} not found in configuration")

        repo_info = docker_images.get(request.repo_name, {})

        # Override base_commit if provided in request
        if request.base_commit:
            repo_info["base_commit"] = request.base_commit

        # Set up the config for clone_repo
        config = {"call_paths": {"default": {"repo_domain": "github.com"}}, "home_dir": os.getcwd()}

        # Add repo name to repo_info
        repo_path = "/testbed" if allow_testbed else None
        if not allow_testbed:
            repo_info["name"] = request.repo_name
            # Clone the repository and get the path
            repo_path = clone_repo(repo_info, config)

        # Acquire a per-repo lock to avoid patch races
        repo_lock = _get_repo_lock(request.repo_name)
        repo_lock.acquire()
        try:
            # Always hard reset to the base commit before applying patches
            is_git_repo = not allow_testbed
            base_commit = repo_info.get("base_commit") if is_git_repo else None
            if request.base_commit:
                base_commit = request.base_commit
            if is_git_repo and base_commit:
                _git_checkout(repo_path, base_commit)

            # Apply patches (if any), serve content, then revert
            if is_git_repo and request.patches:
                _apply_patches(repo_path, request.patches)

            # View the file
            # Support absolute /testbed paths directly
            rel_path = request.path or ""
            if rel_path.startswith("/testbed/"):
                rel_path = rel_path[9:]
            elif rel_path.startswith("testbed/"):
                rel_path = rel_path[8:]
            file_content = "OBSERVATION:\n" + view_file(repo_path, rel_path, request.view_range) + "\n"

            # Perform cleanup BEFORE releasing the lock/returning, to guarantee no state leak
            try:
                if is_git_repo:
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
