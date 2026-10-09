"""OpenR1 grading preserves multi-answer equivalence and final-answer gates."""

import asyncio
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards

FACTORY = "open_instruct.miles.rewards.openr1_verifier.OpenR1MathVerifier"
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPEN_INSTRUCT_MATH_VERIFIER_PYTHONPATH") and importlib.util.find_spec("math_verify") is None,
    reason="Install runtime/miles/symbolic-verifier-requirements.txt in the verifier environment",
)


def test_openr1_equivalence_and_wrong_answer_controls_in_isolated_registry(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"openr1_math": {"factory": FACTORY}}))
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(registry), reward_final_answer_only=True))
    cases = [
        (r"\boxed{\frac{2}{4}}", r"\frac{1}{2}", 1),
        (r"\boxed{\frac{3}{4}}", r"\frac{1}{2}", 0),
        (r"\boxed{4001} and \boxed{8004}", r"4001 \text{ or } 8004", 1),
        (r"\boxed{8004}", r"4001 \text{ or } 8004", 0),
        (r"\boxed{4001} and \boxed{8005}", r"4001 \text{ or } 8004", 0),
        (r"\boxed{1}, \boxed{2}, \boxed{3}", "1,2", 0),
        (r"\boxed{A}", r"\textbf{(A)}", 1),
        (r"\boxed{B}", r"\textbf{(A)}", 0),
        (r"\boxed{\frac{46}{5}}", r"9\frac{1}{5}", 1),
        (r"\boxed{\frac{45}{5}}", r"9\frac{1}{5}", 0),
        (r"\boxed{y=-4x^2+4x-2}", r"y+1=-(2x-1)^2", 1),
        (r"\boxed{y=-4x^2+4x-1}", r"y+1=-(2x-1)^2", 0),
        ("", "42", 0),
        ("I cannot solve this.", "42", 0),
        (r"\boxed{43}", "42", 0),
    ]
    samples = [
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response="</think>" + prediction,
            prompt="Solve the problem",
            metadata={"verifiers": [{"name": "openr1_math", "target": label}]},
        )
        for prediction, label, _ in cases
    ]
    # A correct answer inside unfinished reasoning must not earn reward.
    samples.append(
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response=r"Reasoning: \boxed{42}",
            prompt="Solve the problem",
            metadata={"verifiers": [{"name": "openr1_math", "target": "42"}]},
        )
    )
    samples.append(
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response=r"\boxed{42}</think>\boxed{43}",
            prompt="Solve the problem",
            metadata={"verifiers": [{"name": "openr1_math", "target": "42"}]},
        )
    )
    children = set(rewards._CHILDREN)
    assert isinstance(
        rewards._resolve_verifier({"factory": FACTORY}, isolate_symbolic=True), rewards._IsolatedVerifier
    )
    assert asyncio.run(rewards.registered_reward(args, samples)) == [case[2] for case in cases] + [0, 0]
    assert children == rewards._CHILDREN
