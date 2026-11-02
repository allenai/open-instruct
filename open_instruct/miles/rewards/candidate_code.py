"""Complete-test Python function rewards using the provisioned execution service."""

import asyncio
from dataclasses import dataclass

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
        self.config = verifier_config

    async def async_call(self, tokens, prediction, label, **kwargs):
        validate_target(label)
        program = code_rewards.extract_python_code(candidate_verifiers.answer_text(prediction))

        def execute():
            values = []
            for start in range(0, len(label), 16):
                tests = label[start : start + 16]
                response = code_rewards._get_session().post(
                    self.config.api_url,
                    json={"program": program, "tests": tests, "max_execution_time": self.config.max_execution_time},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json().get("results")
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
                },
            )

        return await asyncio.to_thread(execute)
