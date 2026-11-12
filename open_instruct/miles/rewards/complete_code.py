"""All-test Python rewards through the pinned, authenticated execution client.

Only the external service executes submitted code. A failed test scores zero;
missing results and infrastructure failures propagate rather than inventing a grade.
"""

import dataclasses
import math
import re
from functools import lru_cache
from urllib.parse import urlsplit

from open_instruct.code_utils.execution_client import livecodebench
from open_instruct.code_utils.execution_client import verifier as execution_verifier
from open_instruct.miles.rewards import deterministic_verifiers as common

_SUITE_LIMIT = 2000


@dataclasses.dataclass
class Config:
    api_url: str
    expected_source_sha256: str
    max_execution_time: float = 1.0
    total_timeout: float = 600.0


def validate_config(config):
    url = urlsplit(config.api_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Code service requires an HTTPS URL without credentials, query or fragment")
    if re.fullmatch(r"[0-9a-f]{64}", config.expected_source_sha256) is None:
        raise ValueError("Code service requires a qualified source SHA256")
    if not math.isfinite(config.max_execution_time) or not 0.05 <= config.max_execution_time <= 16:
        raise ValueError("Code execution time must be finite and between 0.05 and 16 seconds")
    if not math.isfinite(config.total_timeout) or config.total_timeout <= 0:
        raise ValueError("Code client timeout must be finite and positive")


@lru_cache(maxsize=4096)
def validate_assertion(test):
    try:
        compile(test, "<source-test>", "exec")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("Invalid Python syntax in source test") from error


def validate_target(target):
    if not isinstance(target, dict) or set(target) != {"mode", "tests"}:
        raise ValueError("Code target requires mode and tests")
    mode, tests = target["mode"], target["tests"]
    if mode not in {"assert", "stdio"} or not isinstance(tests, list) or not tests:
        raise ValueError("Code target requires a supported mode and nonempty test list")
    for test in tests:
        if mode == "assert":
            valid = isinstance(test, str) and bool(test.strip())
        else:
            valid = (
                isinstance(test, dict)
                and {"input", "output"} <= set(test) <= {"input", "output", "comparison"}
                and test.get("comparison", "exact") in ("exact", "nemo_v1")
                and isinstance(test["output"], str)
                and (
                    isinstance(test["input"], str)
                    or isinstance(test["input"], list)
                    and all(isinstance(line, str) for line in test["input"])
                )
            )
        if not valid:
            raise ValueError(f"Malformed {mode} code test")
        if mode == "assert":
            validate_assertion(test)


def livecodebench_target(unit_tests):
    if not isinstance(unit_tests, dict) or set(unit_tests) - {"inputs", "outputs", "fn_name"}:
        raise ValueError("Expected LiveCodeBench inputs, outputs and optional fn_name")
    inputs, outputs = unit_tests.get("inputs"), unit_tests.get("outputs")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise ValueError("LiveCodeBench requires nonempty, aligned inputs and outputs")
    all_tests = []
    for start in range(0, len(inputs), _SUITE_LIMIT):
        mode, tests = livecodebench.prepare_nemotron_tests(
            inputs[start : start + _SUITE_LIMIT], outputs[start : start + _SUITE_LIMIT], unit_tests.get("fn_name")
        )
        all_tests.extend(tests)
    target = {"mode": mode, "tests": all_tests}
    validate_target(target)
    return target


class CompleteCodeVerifier:
    get_config_class = staticmethod(lambda: Config)

    def __init__(self, verifier_config):
        validate_config(verifier_config)
        self.config = verifier_config

    async def async_call(self, tokens, prediction, label, **kwargs):
        validate_target(label)
        verifier = execution_verifier.CodeVerifier(
            execution_verifier.ServiceConfig(
                **dataclasses.asdict(self.config),
                stdio=label["mode"] == "stdio",
                pass_rate_reward_threshold=1.0,
                fresh_process_per_test=True,
            )
        )
        final = common.answer_text(prediction)
        score, partitions = 1.0, []
        # The client batches each partition within its gateway/payload limits.
        # Keep all tests, even beyond its per-suite limit and after a failed test.
        for start in range(0, len(label["tests"]), _SUITE_LIMIT):
            result = await verifier.async_call(tokens, final, label["tests"][start : start + _SUITE_LIMIT])
            if result.score not in (0.0, 1.0):
                raise RuntimeError("Complete-code service returned a nonbinary aggregate")
            score *= result.score
            partitions.append(result.diagnostics)
        # The pinned client raises infrastructure failures, but never trust a
        # returned zero without a completed-grade status. Preserve failures at
        # the boundary consumed by qualification and training omission checks.
        failed = any(
            not isinstance(partition, dict) or partition.get("status") not in {"ok", "completed"}
            for partition in partitions
        )
        return common.Result(
            score,
            diagnostics={
                "kind": "complete_code_v1",
                "status": "service_error" if failed else "ok",
                "mode": label["mode"],
                "tests": len(label["tests"]),
                "service_source_sha256": self.config.expected_source_sha256,
                "partitions": partitions,
            },
        )
