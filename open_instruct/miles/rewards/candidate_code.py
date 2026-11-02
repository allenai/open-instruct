"""Complete-test Python function rewards using the provisioned execution service."""

import asyncio
import math
import time
from dataclasses import dataclass

from open_instruct.miles.infrastructure import infra_timeouts
from open_instruct.miles.rewards import candidate_verifiers, code_rewards


def validate_target(tests):
    if not isinstance(tests, list) or not 1 <= len(tests) <= 256:
        raise ValueError("Expected 1..256 complete function tests")
    if any(not isinstance(test, str) or not test.strip() for test in tests):
        raise ValueError("Expected nonempty Python test strings")


@dataclass
class Config:
    api_url: str
    max_execution_time: float = 2.0


class CompleteCodeVerifier:
    get_config_class = staticmethod(lambda: Config)

    def __init__(self, verifier_config):
        if not isinstance(verifier_config.api_url, str) or not verifier_config.api_url.strip():
            raise ValueError("Complete-code verifier requires an API URL")
        if not math.isfinite(verifier_config.max_execution_time) or verifier_config.max_execution_time <= 0:
            raise ValueError("Complete-code execution time must be finite and positive")
        self.config = verifier_config

    async def async_call(self, tokens, prediction, label, **kwargs):
        validate_target(label)
        program = code_rewards.extract_python_code(candidate_verifiers.answer_text(prediction))

        def execute():
            values = []
            retries = 0
            started = time.monotonic()
            for start in range(0, len(label), 16):
                tests = label[start : start + 16]
                payload = {"program": program, "tests": tests, "max_execution_time": self.config.max_execution_time}
                for attempt in range(len(code_rewards.HTTP_500_RETRY_SLEEPS) + 1):
                    response = infra_timeouts.request(
                        code_rewards._get_session(), "post", self.config.api_url, json=payload, timeout=30
                    )
                    if response.status_code != 500 or attempt == len(code_rewards.HTTP_500_RETRY_SLEEPS):
                        break
                    time.sleep(code_rewards.HTTP_500_RETRY_SLEEPS[attempt])
                    retries += 1
                # An exhausted service error is unknown, not a failed test. Let
                # the registered reward wrapper capture the complete input.
                response.raise_for_status()
                body = response.json()
                result = body.get("results") if isinstance(body, dict) else None
                if not isinstance(result, list) or len(result) != len(tests):
                    raise RuntimeError("Execution service returned incomplete test results")
                if any(type(x) not in (bool, int) or x not in (0, 1) for x in result):
                    raise RuntimeError("Execution service returned nonbinary results")
                values.extend(result)
            return candidate_verifiers.Result(
                float(all(values)),
                diagnostics={
                    "kind": "complete_python_tests_v1",
                    "status": "ok",
                    "tests": len(values),
                    "passed": sum(values),
                    "http_500_retries": retries,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

        return await asyncio.to_thread(execute)
