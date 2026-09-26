"""Exercise real code-verifier HTTP retries through a private loopback fault proxy."""

import argparse
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import requests

from open_instruct.miles.rewards import code_rewards


class ProxyState(TypedDict):
    statuses: list[int]
    attempts: list[int]
    forwarded: int


def exercise(upstream):
    """Keep all injected failures local; forward only healthy canary requests."""
    state: ProxyState = {"statuses": [], "attempts": [], "forwarded": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            status = state["statuses"].pop(0) if state["statuses"] else 200
            state["attempts"].append(status)
            if status == 200:
                try:
                    response = requests.post(
                        upstream + self.path, data=body, headers={"Content-Type": "application/json"}, timeout=30
                    )
                    status, body = response.status_code, response.content
                    state["forwarded"] += 1
                except requests.RequestException as error:
                    status, body = 502, str(error).encode()
            else:
                body = json.dumps({"injected_status": status}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if status == 429:
                self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    args = SimpleNamespace(
        code_api_url=f"http://127.0.0.1:{server.server_port}/test_program", code_pass_rate_reward_threshold=0.99
    )
    report = []
    retries = code_rewards.RETRY.total
    if not isinstance(retries, int):
        raise ValueError("Expected a bounded integer retry budget")
    try:
        for name, statuses, stdio, expected, expected_error in (
            ("function-transient-gateway", [503, 502, 504], False, 1.0, False),
            ("stdio-transient-gateway", [503, 502, 504], True, 1.0, False),
            ("exhausted-gateway", [503] * (retries + 1), True, None, True),
            ("healthy-after-exhaustion", [], True, 1.0, False),
            ("rate-limit-retry-after", [429], True, 1.0, False),
            ("sample-rejection", [500], True, 0.0, False),
        ):
            state.update(statuses=statuses.copy(), attempts=[], forwarded=0)
            started = time.monotonic()
            error_text, score, diagnostics = None, None, None
            program, tests = (
                ('print("2")', [{"input": "", "output": "2\n"}])
                if stdio
                else ("def add(a,b): return a+b", ["assert add(1,2)==3"])
            )
            try:
                score, diagnostics = asyncio.run(code_rewards.execute(args, program, tests, stdio=stdio))
            except RuntimeError as error:
                error_text = str(error)
            expected_attempts = statuses if expected_error or name == "sample-rejection" else statuses + [200]
            passed = (
                (error_text is not None) == expected_error
                and score == expected
                and state["attempts"] == expected_attempts
            )
            if not expected_error:
                passed = (
                    passed
                    and diagnostics is not None
                    and diagnostics["status"] == ("rejected" if name == "sample-rejection" else "ok")
                )
            row = {
                "name": name,
                "passed": passed,
                "attempts": state["attempts"].copy(),
                "forwarded": state["forwarded"],
                "seconds": time.monotonic() - started,
                "score": score,
                "diagnostics": diagnostics,
                "error": error_text,
            }
            report.append(row)
            print(json.dumps(row), flush=True)
            if not passed:
                raise AssertionError(f"Service recovery acceptance failed: {row}")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    return {
        "passed": True,
        "retry_budget": retries,
        "backoff_factor": code_rewards.RETRY.backoff_factor,
        "upstream": upstream,
        "cases": report,
        "limits": "Real HTTP retry/verifier boundary with local fault injection. Does not establish distributed trainer/engine replacement or in-loop recovery.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", help="Code service base URL without /test_program")
    parser.add_argument("--output", type=Path, default=Path("/output/services.json"))
    args = parser.parse_args()
    result = exercise(args.upstream.rstrip("/"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
