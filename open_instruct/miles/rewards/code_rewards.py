#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score generated Python programs through an external code-execution service.
The adapter extracts the submitted code, sends its tests to the service and turns
pass results into the existing Open Instruct reward, including the optional runtime
penalty. Bounded HTTP retries and per-response diagnostics distinguish failed tests
from service failures, with zero reward, an exception, or exclusion from training
according to the configured failure policy.

The hosted service runs Python older than 3.10, so programs whose annotations use
PEP 604 unions or builtin generics get ``from __future__ import annotations`` before
they are sent. The row checks at the end flag dataset rows that no program can pass.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import binascii
import builtins
import collections
import io
import json
import math
import os
import pickle
import re
import time
import zlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import requests

from open_instruct import logger_utils
from open_instruct.miles.infrastructure import infra_timeouts
from open_instruct.miles.rewards import service

logger = logger_utils.setup_logger(__name__)

# An opening fence, its info string (language tag) on the same line, then the body.
_CODE_BLOCK_PATTERN = re.compile(r"```[ \t]*([^\n`]*)\n(.*?)```", re.DOTALL)
_PYTHON_TAGS = frozenset({"python", "python3", "py", "py3"})
FAILURE_POLICIES = ("zero", "raise", "mask")
# AWS Lambda rejects synchronous request bodies over 6 MiB; stay a little below it so the
# API Gateway envelope also fits.
MAX_PAYLOAD_BYTES = 6_000_000
# Diagnostics key ``rewards._score`` turns into a training exclusion (``rewards.EXCLUDED_KEY``).
EXCLUDE_FROM_TRAINING = "exclude_from_training"
_BUILTIN_GENERICS = frozenset({"dict", "frozenset", "list", "set", "tuple", "type"})
_BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True, kw_only=True)
class CodeVerifierConfig:
    """Validated execution-service settings for one code verifier call."""

    api_url: str
    max_execution_time: float
    pass_rate_reward_threshold: float
    apply_perf_penalty: bool
    failure_policy: str = "zero"
    future_annotations: bool = True


def extract_python_code(model_output: str, function_name: str | None = None) -> str:
    """Return the fenced block to grade, or the complete response when no block qualifies.

    Blocks tagged as Python (``python``, ``py``, ``python3``, ``py3``, any case) take
    precedence over untagged blocks, and blocks tagged with another language (sample
    input, shell, C++) are never graded. When the tests call a known function, the last
    preferred block that defines it wins, so a trailing usage example is not graded in
    place of the solution; otherwise the last preferred block wins.
    """
    tagged, untagged = [], []
    for info, body in _CODE_BLOCK_PATTERN.findall(model_output):
        body = body.strip()
        if not body:
            continue
        language = info.split()[0].lower() if info.split() else ""
        if language in _PYTHON_TAGS:
            tagged.append(body)
        elif not language:
            untagged.append(body)
    blocks = tagged or untagged
    if not blocks:
        return model_output
    if function_name:
        definition = re.compile(
            rf"^[ \t]*(?:async[ \t]+def|def|class)[ \t]+{re.escape(function_name)}\b", re.MULTILINE
        )
        defining = [block for block in blocks if definition.search(block)]
        if defining:
            return defining[-1]
    return blocks[-1]


def function_name_from_tests(tests: Any) -> str | None:
    """The non-builtin function (or class) the assert tests call most often, if any.

    Only plain lists of test strings are inspected; compressed and stdio targets have
    no function name. Ties go to the name that appears first.
    """
    if not isinstance(tests, list):
        return None
    counts: collections.Counter[str] = collections.Counter()
    for test in tests:
        if not isinstance(test, str):
            continue
        try:
            tree = ast.parse(test)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in _BUILTIN_NAMES:
                counts[node.func.id] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _annotations(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = node.args
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            ):
                if argument is not None and argument.annotation is not None:
                    yield argument.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _needs_future_annotations(tree: ast.AST) -> bool:
    """Whether an annotation uses ``X | Y`` or ``list[int]``-style syntax older Pythons reject."""
    for annotation in _annotations(tree):
        for part in ast.walk(annotation):
            if isinstance(part, ast.BinOp) and isinstance(part.op, ast.BitOr):
                return True
            if (
                isinstance(part, ast.Subscript)
                and isinstance(part.value, ast.Name)
                and part.value.id in _BUILTIN_GENERICS
            ):
                return True
    return False


def add_future_annotations(program: str) -> str:
    """Add ``from __future__ import annotations`` when the program's annotations need it.

    The hosted service runs Python older than 3.10, where an annotation such as
    ``int | None`` (or ``list[int]`` before 3.9) raises at definition time. Postponed
    evaluation keeps annotations unevaluated, so those programs run unchanged. The import
    goes after the module docstring, or after a leading shebang/encoding comment.
    Runtime uses of the syntax (``isinstance(x, int | str)``, ``Alias = int | None``,
    ``typing.get_type_hints``) still fail; programs that do not parse are left alone.
    """
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError):
        return program
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
        ):
            return program
    if not _needs_future_annotations(tree):
        return program
    lines = program.splitlines(keepends=True)
    first = tree.body[0] if tree.body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        position = first.end_lineno
    else:
        position = 0
        while position < min(2, len(lines)) and lines[position].startswith("#"):
            position += 1
    if position and not lines[position - 1].endswith("\n"):
        lines[position - 1] += "\n"
    lines.insert(position, "from __future__ import annotations\n")
    return "".join(lines)


def code_verifier_config(args: Any, *, stdio: bool = False) -> CodeVerifierConfig:
    """Resolve code verifier settings from compatible args and environment."""
    url = getattr(args, "code_api_url", None) or os.environ.get("OI_MILES_CODE_API_URL")
    if not url:
        base = os.environ.get("CODE_API_URL", "http://localhost:1234").rstrip("/")
        url = f"{base}/test_program"
    if stdio:
        if not str(url).endswith("/test_program"):
            raise ValueError("code_stdio requires a code API URL ending in /test_program")
        url = str(url)[: -len("/test_program")] + "/test_program_stdio"
    max_execution_time = service.number_setting(
        args, "code_max_execution_time", "OI_MILES_CODE_MAX_EXECUTION_TIME", 1.0
    )
    threshold = service.number_setting(
        args, "code_pass_rate_reward_threshold", "OI_MILES_CODE_PASS_RATE_REWARD_THRESHOLD", 0.0
    )
    if max_execution_time <= 0:
        raise ValueError("OI_MILES_CODE_MAX_EXECUTION_TIME must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("OI_MILES_CODE_PASS_RATE_REWARD_THRESHOLD must be between zero and one")
    failure_policy = getattr(args, "code_failure_policy", None)
    if failure_policy is None:
        failure_policy = os.environ.get("OI_MILES_CODE_FAILURE_POLICY", "zero")
    if failure_policy not in FAILURE_POLICIES:
        raise ValueError("OI_MILES_CODE_FAILURE_POLICY must be 'zero', 'raise' or 'mask'")
    return CodeVerifierConfig(
        api_url=str(url),
        failure_policy=failure_policy,
        max_execution_time=max_execution_time,
        pass_rate_reward_threshold=threshold,
        apply_perf_penalty=service.bool_setting(
            args, "code_apply_perf_penalty", "OI_MILES_CODE_APPLY_PERF_PENALTY", False
        ),
        future_annotations=service.bool_setting(
            args, "code_future_annotations", "OI_MILES_CODE_FUTURE_ANNOTATIONS", True
        ),
    )


# The code service is an external API gateway that returns transient 5xx errors
# under load. urllib3 retries only idempotent methods by default, so the scoring
# POSTs must be allowed explicitly; the backoff grows 1, 2, 4 seconds. With the
# 30-second request timeout, a dead gateway costs at most about two minutes per
# sample before the configured failure policy applies. The earlier eight-retry
# budget took 517-530 seconds per failed sample and stalled whole prompt groups.
RETRY = requests.adapters.Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=[502, 503, 504],
    allowed_methods=frozenset({"GET", "POST"}),
    raise_on_status=False,
)
# Worst-case seconds one sample can spend inside the HTTP layer at the default
# 30-second timeout: (total + 1) attempts plus the exponential backoff sleeps.
RETRY_WORST_CASE_SECONDS = (RETRY.total + 1) * 30.0 + sum(
    min(RETRY.backoff_factor * 2**i, 120) for i in range(RETRY.total)
)


def _get_session() -> Any:
    return service.http_session(pool_size=100, retries=RETRY)


def _score_response(result: Any, *, config: CodeVerifierConfig) -> float:
    if not isinstance(result, dict):
        raise RuntimeError("code service response must be an object")
    passes = result.get("results")
    if not isinstance(passes, list):
        raise RuntimeError("code service response requires a results list")
    if any(isinstance(value, bool) is False and not isinstance(value, int | float) for value in passes):
        raise RuntimeError("code service results must be numeric")
    pass_rate = sum(float(value) for value in passes) / len(passes) if passes else 0.0
    score = 0.0 if pass_rate < config.pass_rate_reward_threshold else pass_rate
    if not config.apply_perf_penalty or score <= 0:
        return score
    runtimes = result.get("runtimes")
    if not isinstance(runtimes, list) or len(runtimes) != len(passes):
        raise RuntimeError("performance-penalized code rewards require aligned runtimes")
    penalized = [
        float(passed) * (config.max_execution_time - float(runtime)) / config.max_execution_time
        for passed, runtime in zip(passes, runtimes, strict=True)
    ]
    return sum(penalized) / len(penalized) if penalized else 0.0


async def code_score(args: Any, prediction: str, target: Any, *, stdio: bool = False) -> float:
    """Execute one completion against its tests and return its pass-rate reward."""
    score, _ = await execute(args, prediction, target, stdio=stdio)
    return score


async def execute(args: Any, prediction: str, target: Any, *, stdio: bool = False) -> tuple[float, dict]:
    """Score one completion and describe the service outcome for per-sample diagnostics."""
    config = code_verifier_config(args, stdio=stdio)
    program = extract_python_code(prediction, None if stdio else function_name_from_tests(target))
    if config.future_annotations:
        program = add_future_annotations(program)
    payload = {"program": program, "tests": target, "max_execution_time": config.max_execution_time}
    timeout = max(30.0, min(300.0, config.max_execution_time * 10))
    mask = config.failure_policy == "mask"

    diagnostics = {
        "status": "ok",
        "http_status": None,
        "program_chars": len(program),
        "tests": len(target) if isinstance(target, list) else -1,
        "stdio": stdio,
        "failure_policy": config.failure_policy,
    }
    if mask:
        # The gateway rejects bodies over its size limit whatever the program does, so the
        # sample carries no signal about the policy. Skip the request and exclude it.
        payload_bytes = len(json.dumps(payload, allow_nan=False))
        if payload_bytes > MAX_PAYLOAD_BYTES:
            diagnostics.update(
                status="rejected",
                rejection="payload_too_large",
                payload_bytes=payload_bytes,
                elapsed_seconds=0.0,
                **{EXCLUDE_FROM_TRAINING: True},
            )
            return 0.0, diagnostics

    def request() -> Any:
        try:
            response = infra_timeouts.request(
                _get_session(),
                "post",
                config.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            diagnostics["http_status"] = response.status_code
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            status = getattr(error.response, "status_code", None)
            if status is not None and (400 <= status < 500 and status != 429 or status == 500):
                # Preserve the existing rejection category for client errors
                # and HTTP 500 (which this harness can emit for program errors).
                # HTTP status alone does not establish the underlying cause.
                # Gateway and transport failures use the
                # configured failure policy after the HTTP retries finish.
                diagnostics["status"] = "rejected"
                if mask and status == 413:
                    # Payload too large: a grader limit, not a property of the program. HTTP
                    # 500 stays a zero: it may come from the harness running this program.
                    diagnostics.update(rejection="payload_too_large", **{EXCLUDE_FROM_TRAINING: True})
                logger.warning(
                    "code verifier rejected a sample with HTTP %s (program %d chars, %d tests); %s",
                    status,
                    len(program),
                    len(target) if isinstance(target, list) else -1,
                    "excluding it from training" if diagnostics.get(EXCLUDE_FROM_TRAINING) else "scoring it zero",
                )
                return {"results": []}
            raise

    started = time.monotonic()
    stage = "request"
    try:
        result = await asyncio.to_thread(request)
        stage = "response"
        score = _score_response(result, config=config)
        if not math.isfinite(score):
            raise RuntimeError(f"code verifier returned non-finite score {score!r}")
    except Exception as error:
        # Match original Open Instruct's continue-with-zero behavior, while
        # distinguishing missing/invalid grading from an actual failed test.
        # Config validation stays outside this try; cancellation is not caught.
        diagnostics.update(
            status="service_error", error_stage=stage, error_type=type(error).__name__, error=str(error)[:1000]
        )
        if config.failure_policy == "raise":
            raise RuntimeError(f"code verifier request failed for {config.api_url}: {error}") from error
        if mask:
            # No grade exists for this sample: leave it out of the loss and the group baseline
            # instead of teaching the policy that its program failed.
            diagnostics[EXCLUDE_FROM_TRAINING] = True
        logger.warning(
            "code verifier service failure after %.2fs; assigning zero reward and %s: %s",
            time.monotonic() - started,
            "excluding it from training" if mask else "continuing",
            diagnostics,
        )
        score = 0.0
    finally:
        diagnostics["elapsed_seconds"] = time.monotonic() - started
    return score, diagnostics


@dataclass
class ServiceConfig:
    api_url: str
    stdio: bool = False
    max_execution_time: float = 1.0
    pass_rate_reward_threshold: float = 0.99
    # "zero" (default), "raise", or "mask": exclude ungraded samples from training.
    failure_policy: str | None = None
    future_annotations: bool = True


class CodeVerifier:
    get_config_class = staticmethod(lambda: ServiceConfig)

    def __init__(self, verifier_config):
        self.config = verifier_config

    async def async_call(self, tokens, prediction, label, **kwargs):
        args = SimpleNamespace(
            code_api_url=self.config.api_url,
            code_failure_policy=self.config.failure_policy,
            code_max_execution_time=self.config.max_execution_time,
            code_pass_rate_reward_threshold=self.config.pass_rate_reward_threshold,
            code_future_annotations=self.config.future_annotations,
        )
        score, diagnostics = await execute(args, prediction, label, stdio=self.config.stdio)
        return SimpleNamespace(score=score, cost=0.0, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Dataset row checks, for data preparation. They never call the service.
# ---------------------------------------------------------------------------

# Problems a prompt rewrite can repair; every other problem means no program can pass.
FIXABLE_PROBLEMS = frozenset({"function_name_missing", "language_unspecified"})
# Room left in the request body for the program itself when estimating payload size.
_PROGRAM_ALLOWANCE_BYTES = 100_000
_MAX_DECODED_BYTES = 512 * 1024 * 1024


class _PlainUnpickler(pickle.Unpickler):
    """Loads only pickles of plain containers and scalars; never imports or calls anything."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def decode_tests_safely(target: Any) -> list | None:
    """The test list of a JSON or base64 zlib-pickled target, or None when it cannot be read.

    Mirrors the service's ``decode_tests`` without its arbitrary-code risk: pickles that
    reference any class or function are refused.
    """
    if isinstance(target, list):
        return target
    if not isinstance(target, str) or not target:
        return None
    try:
        tests = json.loads(target)
    except json.JSONDecodeError:
        try:
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(base64.b64decode(target, validate=True), _MAX_DECODED_BYTES)
            if decompressor.unconsumed_tail:
                return None
            tests = _PlainUnpickler(io.BytesIO(raw + decompressor.flush())).load()
        except (binascii.Error, zlib.error, pickle.UnpicklingError, EOFError, ValueError, TypeError):
            return None
    return tests if isinstance(tests, list) else None


def _mentions(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", text) is not None


def code_target_problems(name: str, target: Any, prompt: str) -> list[str]:
    """Reasons a ``code``/``code_stdio`` row cannot be graded fairly; empty when it can.

    - ``undecodable_tests``: the target is not a readable nonempty test list.
    - ``non_string_stdio_test``: a stdio test whose input or output is not a string (for
      example a list of lines); the service can never pass it.
    - ``non_string_test``: a function-call test that is not a string.
    - ``payload_too_large``: the tests alone leave no room under the gateway's body limit.
    - ``non_assert_test``: a function-call test without an ``assert``, which passes whenever
      the call does not raise (advisory).
    - ``function_name_missing``: the prompt never names the function the tests call
      (fixable with ``add_function_name_hint``).
    - ``language_unspecified``: a stdio prompt that never asks for Python (fixable with
      ``add_python_hint``).
    """
    tests = decode_tests_safely(target)
    if not tests:
        return ["undecodable_tests"]
    problems = []
    if name == "code_stdio":
        if not all(
            isinstance(test, dict) and isinstance(test.get("input"), str) and isinstance(test.get("output"), str)
            for test in tests
        ):
            problems.append("non_string_stdio_test")
    elif not all(isinstance(test, str) for test in tests):
        problems.append("non_string_test")
    if len(json.dumps(target)) + _PROGRAM_ALLOWANCE_BYTES > MAX_PAYLOAD_BYTES:
        problems.append("payload_too_large")
    if name == "code_stdio":
        if not re.search(r"python", prompt, re.IGNORECASE):
            problems.append("language_unspecified")
        return problems
    strings = [test for test in tests if isinstance(test, str)]
    if any(not _is_assert(test) for test in strings):
        problems.append("non_assert_test")
    function_name = function_name_from_tests(strings)
    if function_name and not _mentions(prompt, function_name):
        problems.append("function_name_missing")
    return problems


def _is_assert(test: str) -> bool:
    try:
        tree = ast.parse(test)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def code_row_problems(row: dict) -> list[str]:
    """``code_target_problems`` for a prepared Miles row (``metadata.verifiers`` and ``query``)."""
    metadata = row["metadata"]
    (spec,) = metadata["verifiers"]
    return code_target_problems(spec["name"], spec["target"], metadata["query"])


def add_function_name_hint(prompt: str, function_name: str) -> str:
    """The user prompt with a sentence naming the function (or class) the tests call."""
    kind = "class" if function_name[:1].isupper() else "function"
    return f"{prompt.rstrip()}\n\nYour solution must define a {kind} named `{function_name}`."


def add_python_hint(prompt: str) -> str:
    """The stdio user prompt with a sentence asking for a Python program."""
    return (
        f"{prompt.rstrip()}\n\nWrite your solution as a Python 3 program that reads from standard input "
        "and writes to standard output."
    )
