# Client-side code ported from code-execution-service revision 103ca279b35e8e81d118947f82b72c22676c6540.
"""Open Instruct registry adapter for complete, authenticated code grading.

Service failures propagate to the caller. This adapter never substitutes a zero
reward for an unavailable or incomplete grade.
"""

import ast
import asyncio
import dataclasses
import math
import os
import re
import threading
import time
from types import SimpleNamespace

import requests

from open_instruct.code_utils.execution_client import CodeExecutionClient, ServiceError, protocol

_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})([^\r\n]*)")
_PYTHON = {"python", "python3", "py"}
_END_TOKENS = re.compile(r"(?:(?:<\|endoftext\|>|<\|im_end\|>|<\|eot_id\|>)\s*)+\Z")


def extract_python_code(response):
    """Select the last complete Python fence, then an unlabelled fence, then raw code.

    Explicit other-language blocks are never Python submissions. An unfinished
    Python block is invalid rather than falling back to an earlier draft.
    """
    # Raw programs may contain Markdown fences inside string literals. Recognize
    # valid Python before applying Markdown extraction to a model's prose.
    try:
        ast.parse(response)
    except (SyntaxError, ValueError, RecursionError):
        pass
    else:
        return response
    # Rollouts may retain a decoded end token immediately after the closing fence.
    # Strip only terminal model delimiters, after preserving valid raw Python (which
    # can contain these exact strings). An actually unfinished fence stays invalid.
    response = _END_TOKENS.sub("", response)
    python, unlabelled = [], []
    marker, language, lines, saw_fence = None, None, [], False
    for line in response.splitlines(keepends=True):
        match = _FENCE.fullmatch(line.rstrip("\r\n"))
        if marker is None:
            if match:
                marker = match[1]
                info = match[2].strip().split(maxsplit=1)
                language = info[0].lower() if info else ""
                lines, saw_fence = [], True
            continue
        if match and match[1][0] == marker[0] and len(match[1]) >= len(marker) and not match[2].strip():
            code = "".join(lines).strip()
            if language in _PYTHON:
                python.append(code)
            elif not language:
                unlabelled.append(code)
            marker = None
        else:
            lines.append(line)
    if marker is not None and (language in _PYTHON or not language):
        return None
    if python:
        return python[-1]
    if unlabelled:
        return unlabelled[-1]
    return None if saw_fence else response


@dataclasses.dataclass
class ServiceConfig:
    api_url: str
    stdio: bool = False
    max_execution_time: float = 1.0
    pass_rate_reward_threshold: float = 0.99
    apply_perf_penalty: bool = False
    token_env: str = "CODE_EXECUTION_TOKEN"
    max_request_seconds: float = 20.0
    max_request_bytes: int = 5 << 20
    total_timeout: float = 300.0
    retries: int = 3
    expected_source_sha256: str | None = None
    fresh_process_per_test: bool = False


class CodeVerifier:
    get_config_class = staticmethod(lambda: ServiceConfig)

    def __init__(self, verifier_config):
        self.config = verifier_config
        threshold = self.config.pass_rate_reward_threshold
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("pass-rate threshold must be finite and between zero and one")
        seconds = self.config.max_execution_time
        if not math.isfinite(seconds) or not protocol.MIN_PER_TEST_SECONDS <= seconds <= protocol.MAX_PER_TEST_SECONDS:
            raise ValueError("per-test limit is outside the service's supported range")
        if not math.isfinite(self.config.total_timeout) or self.config.total_timeout <= 0:
            raise ValueError("complete-test timeout must be finite and positive")

    async def async_call(self, tokens, prediction, label, **kwargs):
        config = self.config
        deadline = time.monotonic() + config.total_timeout
        cancelled = threading.Event()
        if type(config.fresh_process_per_test) is not bool:
            raise ValueError("fresh_process_per_test must be a boolean")
        tests = protocol.decode_tests(label)
        mode = "stdio" if config.stdio else "assert"
        protocol.validate_tests(mode, tests)
        if not tests:
            raise ValueError("training code verifier requires at least one test")
        token = os.environ.get(config.token_env)
        if not token:
            raise ValueError(f"missing code-service bearer token in {config.token_env}")
        program = extract_python_code(prediction)
        if program is None or not program.strip():
            return SimpleNamespace(score=0.0, cost=0.0, diagnostics={"status": "completed", "outcome": "invalid_code"})

        def execute():
            if cancelled.is_set():
                raise ServiceError("code grading cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceError("complete-test request deadline exceeded while waiting for a worker")
            base = config.api_url.rstrip("/")
            for route in ("/test_program_stdio", "/test_program", "/v2/execute"):
                base = base.removesuffix(route)
            with requests.Session() as session:
                client = CodeExecutionClient(
                    base,
                    token=token,
                    session=session,
                    retries=config.retries,
                    max_request_seconds=config.max_request_seconds,
                    max_request_bytes=config.max_request_bytes,
                    total_timeout=remaining,
                    expected_source_sha256=config.expected_source_sha256,
                    cancel_event=cancelled,
                )
                return client.execute(
                    mode,
                    program,
                    tests,
                    per_test_seconds=config.max_execution_time,
                    fresh_process_per_test=config.fresh_process_per_test,
                )

        try:
            result = await asyncio.to_thread(execute)
        except asyncio.CancelledError:
            # requests cannot interrupt an in-flight HTTP call. Let that call drain
            # under its existing timeout, but prevent later batches and retries.
            cancelled.set()
            raise
        score = result.pass_fraction
        if score < config.pass_rate_reward_threshold:
            score = 0.0
        elif config.apply_perf_penalty:
            score = sum(
                max(0.0, 1.0 - outcome.runtime / config.max_execution_time) if outcome.passed else 0.0
                for outcome in result.outcomes
            ) / len(result.outcomes)
        return SimpleNamespace(
            score=score,
            cost=0.0,
            diagnostics={
                "status": "ok",
                "tests": len(tests),
                "request_ids": result.request_ids,
                "attempts": result.attempts,
                "wall_seconds": result.wall_seconds,
            },
        )
