#!/usr/bin/env python3
"""
End-to-end test for CodeViewTool -> API path and repo context flow.

This script:
- Starts the FastAPI server from open_instruct.code_utils.api on a free port
- Builds a CodeViewTool pointing to that server
- Simulates grpo_fast-style tool_context (repo_name) injection
- Issues tool calls (bash and str_replace_editor view) that reference "/testbed"
  and verifies they resolve to the repo-specific testbed directory

Usage: python test_code_view_e2e.py
"""

import json
import socket
import subprocess
import sys
import time

import requests

from open_instruct.tool_utils.tool_vllm import CodeAgentTool


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout_s: float = 25.0):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        # Restore docker_images.json
        try:
            if bak_path.exists():
                docker_json_path.write_text(bak_path.read_text())
                bak_path.unlink(missing_ok=True)
            else:
                # Remove the test file if we created it
                docker_json_path.unlink(missing_ok=True)
        except Exception:
            pass
        time.sleep(0.3)
    return False


def main():
    port = _find_free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"Starting API on {base}")

    # Speed up API startup by temporarily disabling repo cloning on startup.
    # We do this by swapping docker_images.json with an empty mapping for the test run.
    import os
    from pathlib import Path
    api_dir = Path(__file__).parent / "open_instruct" / "code_utils"
    docker_json_path = api_dir / "docker_images.json"
    bak_path = api_dir / "docker_images.json.bak_test"
    try:
        if docker_json_path.exists():
            bak_path.write_text(docker_json_path.read_text())
        docker_json_path.write_text("{}\n")
    except Exception as e:
        print(f"Warning: could not adjust docker_images.json for test speed: {e}")

    # Start FastAPI server (uvicorn) as a subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "open_instruct.code_utils.api:app", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        ok = _wait_for_health(f"{base}/health", timeout_s=45.0)
        if not ok:
            raise SystemExit("API /health did not return 200 within timeout")

        # Initialize CodeViewTool without hardcoding repo_name so it can use tool_context
        tool = CodeAgentTool(
            start_str="<tool_call>",
            end_str="</tool_call>",
            api_endpoint=f"{base}/view_file",
            repo_name=None,
        )

        # Simulate grpo_fast tool_context passing with repo_name for deepdiff
        tool_ctx = {"repo_name": "seperman/deepdiff"}
        hidden_ctx = f"<!--tool_context:{json.dumps(tool_ctx)}-->"

        # 1) Test bash ls on /testbed (API should rewrite to repo-specific testbed)
        prompt_bash = (
            hidden_ctx
            + "\n<tool_call>\n"
            + json.dumps({
                "name": "bash",
                "arguments": {"command": "ls -la /testbed"},
            })
            + "\n</tool_call>"
        )
        print("\n=== Running bash ls on /testbed with repo_context ===")
        res_bash = tool(prompt_bash)
        print(res_bash.output)

        # 2) Test view on /testbed directory (should list contents up to 2 levels)
        prompt_view_dir = (
            hidden_ctx
            + "\n<tool_call>\n"
            + json.dumps({
                "name": "str_replace_editor",
                "arguments": {"command": "view", "path": "/testbed"},
            })
            + "\n</tool_call>"
        )
        print("\n=== Running view on /testbed directory ===")
        res_view_dir = tool(prompt_view_dir)
        print(res_view_dir.output)

        # 3) Test server-side repo inference by specifying owner/repo in path only
        prompt_view_infer = (
            "<tool_call>\n"
            + json.dumps({
                "name": "str_replace_editor",
                "arguments": {"command": "view", "path": "/seperman/deepdiff/testbed"},
            })
            + "\n</tool_call>"
        )
        print("\n=== Running view with server-side repo inference ===")
        res_view_infer = tool(prompt_view_infer)
        print(res_view_infer.output)

        # Simple sanity checks
        failure = False
        for label, out in (
            ("bash", res_bash.output),
            ("view_dir", res_view_dir.output),
            ("view_infer", res_view_infer.output),
        ):
            if "No such file or directory" in (out or ""):
                print(f"[FAIL] {label}: contains 'No such file or directory'")
                failure = True
        if failure:
            sys.exit(1)
        print("\nAll checks passed.")

    finally:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
        # Restore docker_images.json
        try:
            if bak_path.exists():
                docker_json_path.write_text(bak_path.read_text())
                bak_path.unlink(missing_ok=True)
            else:
                # Remove the test file if we created it
                docker_json_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()

