# Client-side code ported from code-execution-service revision 103ca279b35e8e81d118947f82b72c22676c6540.
"""Client for the code execution service.

    client = CodeExecutionClient("https://<function-url>", token=os.environ["CODE_EXECUTION_TOKEN"])
    result = client.execute("assert", program, ["assert add(1, 2) == 3"], per_test_seconds=1.0)
    result.all_passed  # True/False
    result.outcomes    # [Outcome(status="passed", runtime=..., detail=None), ...]

Service faults (429, 5xx, transport errors, timeouts) raise ServiceError after retries so the caller
must mask the sample or abort training; they are never silently turned into
test failures. Other client errors (4xx) raise ClientError immediately.
"""

import base64
import dataclasses
import json
import math
import random
import time
import uuid
import zlib

import requests

from open_instruct.code_utils.execution_client import livecodebench, protocol

__all__ = ["ClientError", "CodeExecutionClient", "ExecuteResult", "Outcome", "ServiceError"]

RETRY_STATUSES = {429, 500, 502, 503, 504}


def _number(value, minimum=0):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= minimum
    except OverflowError:
        return False


def _validate_v1(data, count):
    if not isinstance(data, dict):
        raise ValueError("response must be an object")
    results, runtimes = data.get("results"), data.get("runtimes")
    if not isinstance(results, list) or not isinstance(runtimes, list):
        raise ValueError("results and runtimes must be lists")
    if len(results) != count or len(runtimes) != count:
        raise ValueError("response test count does not match request")
    for result, runtime in zip(results, runtimes, strict=True):
        if type(result) not in (int, float) or result not in (0, 1):
            raise ValueError("results must contain only numeric zero or one")
        if not _number(runtime, -1) or (runtime < 0 and runtime != -1) or (result == 1 and runtime < 0):
            raise ValueError("invalid test runtime")


def _validate_v2(data, count):
    if not isinstance(data, dict) or not isinstance(data.get("outcomes"), list):
        raise ValueError("response must contain outcomes")
    if len(data["outcomes"]) != count:
        raise ValueError("response test count does not match request")
    if not _number(data.get("wall_seconds")) or not isinstance(data.get("exit"), str):
        raise ValueError("invalid execution metadata")
    if data["exit"] in ("wall_timeout", "runner_error") or data.get("runner_error"):
        raise ValueError("execution was incomplete")
    for item in data["outcomes"]:
        if not isinstance(item, dict) or item.get("status") not in ("passed", "failed", "timeout", "error", "crashed"):
            raise ValueError("invalid or unexecuted test outcome")
        runtime, detail = item.get("runtime"), item.get("detail")
        if (runtime is not None and not _number(runtime)) or (item["status"] == "passed" and runtime is None):
            raise ValueError("invalid test runtime")
        if detail is not None and not isinstance(detail, str):
            raise ValueError("invalid test detail")


class ServiceError(RuntimeError):
    def __init__(self, message, status=None, attempts=0, request_id=None):
        super().__init__(message)
        self.status, self.attempts, self.request_id = status, attempts, request_id


class ClientError(ValueError):
    def __init__(self, message, status, request_id=None):
        super().__init__(message)
        self.status, self.request_id = status, request_id


@dataclasses.dataclass(frozen=True)
class Outcome:
    status: str
    runtime: float | None
    detail: str | None

    @property
    def passed(self):
        return self.status == "passed"


@dataclasses.dataclass(frozen=True)
class ExecuteResult:
    outcomes: list[Outcome]
    wall_seconds: float
    exit: str
    request_id: str
    attempts: int
    request_ids: tuple[str, ...] = ()

    @property
    def all_passed(self):
        return bool(self.outcomes) and all(outcome.passed for outcome in self.outcomes)

    @property
    def pass_fraction(self):
        return sum(outcome.passed for outcome in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0


class CodeExecutionClient:
    def __init__(
        self,
        base_url,
        token=None,
        *,
        timeout=30.0,
        retries=3,
        backoff=1.0,
        session=None,
        max_request_seconds=20.0,
        max_request_bytes=5 << 20,
        max_batch_tests=16,
        total_timeout=300.0,
        expected_source_sha256=None,
        cancel_event=None,
    ):
        for name, value in (
            ("timeout", timeout),
            ("total_timeout", total_timeout),
            ("max_request_seconds", max_request_seconds),
        ):
            if not _number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(retries) is not int or retries < 0 or not _number(backoff):
            raise ValueError("retries and backoff must be nonnegative")
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        if type(max_batch_tests) is not int or not 1 <= max_batch_tests <= protocol.MAX_TESTS:
            raise ValueError("invalid max_batch_tests")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = session or requests.Session()
        self.max_request_seconds = max_request_seconds
        self.max_request_bytes = max_request_bytes
        self.max_batch_tests = max_batch_tests
        self.total_timeout = total_timeout
        self.expected_source_sha256 = expected_source_sha256
        self.cancel_event = cancel_event

    def _check_cancelled(self, attempts=0, request_id=None):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ServiceError("code grading cancelled", attempts=attempts, request_id=request_id)

    def _safe_error(self, value):
        message = str(value)
        return message.replace(self.token, "<redacted>") if self.token else message

    def _headers(self, request_id):
        headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _post(self, path, payload, validate, deadline):
        request_id = uuid.uuid4().hex
        last_error = None
        for attempt in range(1, self.retries + 2):
            self._check_cancelled(attempt - 1, request_id)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceError(
                    "complete-test request deadline exceeded", attempts=attempt - 1, request_id=request_id
                )
            try:
                response = self.session.post(
                    self.base_url + path,
                    json=payload,
                    headers=self._headers(request_id),
                    timeout=min(self.timeout, remaining),
                )
            except requests.RequestException as exc:
                last_error = ServiceError(
                    f"{type(exc).__name__}: {self._safe_error(exc)}", attempts=attempt, request_id=request_id
                )
            else:
                self._check_cancelled(attempt, request_id)
                if response.status_code == 200:
                    try:
                        required = payload.get("required_comparisons", [])
                        if required and not set(required) <= set(
                            response.headers.get("X-Service-Comparisons", "").split(",")
                        ):
                            raise ValueError("execution service does not support the dataset's output comparison")
                        if (
                            self.expected_source_sha256
                            and response.headers.get("X-Service-Source-Sha256") != self.expected_source_sha256
                        ):
                            raise ValueError("execution service source fingerprint does not match the pinned revision")
                        data = response.json()
                        validate(data)
                    except (ValueError, TypeError, KeyError) as exc:
                        last_error = ServiceError(
                            f"invalid grading response: {exc}", response.status_code, attempt, request_id
                        )
                    else:
                        if time.monotonic() >= deadline:
                            raise ServiceError(
                                "complete-test request deadline exceeded", attempts=attempt, request_id=request_id
                            )
                        return data, request_id, attempt
                elif response.status_code == 429 or response.status_code >= 500 or response.status_code < 400:
                    last_error = ServiceError(
                        f"HTTP {response.status_code}: {self._safe_error(response.text)[:300]}",
                        response.status_code,
                        attempt,
                        request_id,
                    )
                else:
                    raise ClientError(
                        f"HTTP {response.status_code}: {self._safe_error(response.text)[:300]}",
                        response.status_code,
                        request_id,
                    )
            if attempt <= self.retries:
                self._check_cancelled(attempt, request_id)
                remaining = max(0, deadline - time.monotonic())
                # Full jitter prevents synchronized callers from retrying in lockstep.
                delay = random.uniform(0, min(self.backoff * 2 ** (attempt - 1), remaining))
                if self.cancel_event is None:
                    time.sleep(delay)
                elif self.cancel_event.wait(delay):
                    self._check_cancelled(attempt, request_id)
        # Nonnegative retries guarantee an attempt; every failed attempt sets an error.
        assert last_error is not None
        raise last_error

    def _batches(self, mode, program, tests, seconds, version, language="python", fresh_process_per_test=False):
        try:
            tests = protocol.decode_tests(tests)
            protocol.validate_tests(mode, tests)
            if not isinstance(program, str):
                raise protocol.BadRequest("program must be a string")
            if type(fresh_process_per_test) is not bool:
                raise protocol.BadRequest("fresh_process_per_test must be a boolean")
            seconds = protocol.clamp_per_test_seconds(seconds)
        except protocol.BadRequest as exc:
            raise ClientError(str(exc), 400) from exc
        # Reserve startup/cleanup and per-interpreter overhead below the gateway
        # deadline. Preflight every batch before sending any requests.
        count = min(self.max_batch_tests, int((self.max_request_seconds - 2) / (seconds + 0.1)))
        if count < 1:
            raise ClientError("per-test limit does not fit the configured request time budget", 400)

        def payload_for(batch):
            required = (
                sorted({test.get("comparison", "exact") for test in batch} - {"exact"}) if mode == "stdio" else []
            )
            capability = {"required_comparisons": required} if required else {}
            if version == "v1":
                return {
                    **capability,
                    "program": program,
                    "tests": batch,
                    "max_execution_time": seconds,
                    "fresh_process_per_test": fresh_process_per_test,
                }
            return {
                **capability,
                "language": language,
                "mode": mode,
                "program": program,
                "tests": batch,
                "limits": {"per_test_seconds": seconds},
                "fresh_process_per_test": fresh_process_per_test,
            }

        def fit(batch):
            payload = payload_for(batch)
            raw = json.dumps(batch, allow_nan=False).encode()
            if len(raw) > protocol.MAX_DECODED_TEST_BYTES:
                if len(batch) <= 1:
                    raise ClientError("one test exceeds decoded size limit", 413)
            else:
                if len(json.dumps(payload, allow_nan=False).encode()) <= self.max_request_bytes:
                    return [(payload, len(batch))]
                payload["tests"] = {"encoding": "zlib+json", "data": base64.b64encode(zlib.compress(raw)).decode()}
                if len(json.dumps(payload, allow_nan=False).encode()) <= self.max_request_bytes:
                    return [(payload, len(batch))]
            if len(batch) <= 1:
                raise ClientError("program and one test exceed request size limit even after compression", 413)
            middle = len(batch) // 2
            return fit(batch[:middle]) + fit(batch[middle:])

        batches = []
        for start in range(0, len(tests), count):
            batches.extend(fit(tests[start : start + count]))
        if not tests:
            batches = fit([])
        return batches

    def health(self):
        response = self.session.get(self.base_url + "/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def execute_livecodebench(self, program, inputs, outputs, *, fn_name=None, **kwargs):
        """Execute every LiveCodeBench input/output pair; optional fn_name selects call tests."""
        mode, tests = livecodebench.prepare_tests(inputs, outputs, fn_name)
        return self.execute(mode, program, tests, **kwargs)

    def execute(self, mode, program, tests, *, per_test_seconds=1.0, language="python", fresh_process_per_test=False):
        deadline = time.monotonic() + self.total_timeout
        batches = self._batches(mode, program, tests, per_test_seconds, "v2", language, fresh_process_per_test)
        outcomes, ids, attempts, wall, exit_kind = [], [], 0, 0.0, "ok"
        for payload, count in batches:
            data, request_id, used = self._post(
                "/v2/execute", payload, lambda data, count=count: _validate_v2(data, count), deadline
            )
            outcomes.extend(
                Outcome(item["status"], item.get("runtime"), item.get("detail")) for item in data["outcomes"]
            )
            ids.append(request_id)
            attempts += used
            wall += data["wall_seconds"]
            if exit_kind == "ok":
                exit_kind = data["exit"]
        return ExecuteResult(outcomes, wall, exit_kind, ids[-1], attempts, tuple(ids))

    # v1-compatible helpers, same shapes the previous service returned.
    def test_program(self, program, tests, max_execution_time=1.0, *, fresh_process_per_test=False):
        return self._v1("assert", program, tests, max_execution_time, fresh_process_per_test)

    def test_program_stdio(self, program, tests, max_execution_time=1.0, *, fresh_process_per_test=False):
        return self._v1("stdio", program, tests, max_execution_time, fresh_process_per_test)

    def _v1(self, mode, program, tests, seconds, fresh_process_per_test=False):
        deadline = time.monotonic() + self.total_timeout
        batches = self._batches(mode, program, tests, seconds, "v1", fresh_process_per_test=fresh_process_per_test)
        results, runtimes = [], []
        for payload, count in batches:
            data, _, _ = self._post(
                "/test_program" if mode == "assert" else "/test_program_stdio",
                payload,
                lambda data, count=count: _validate_v1(data, count),
                deadline,
            )
            results.extend(data["results"])
            runtimes.extend(data["runtimes"])
        return results, runtimes
