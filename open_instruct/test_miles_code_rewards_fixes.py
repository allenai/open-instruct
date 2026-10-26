"""Code reward fixes: block choice, old-Python annotations, masked grader failures, row checks."""

import asyncio
import base64
import json
import pickle
import zlib
from types import SimpleNamespace

import pytest
import requests

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.megatron_grpo import grpo_hooks
from open_instruct.miles.rewards import code_rewards, rewards

FENCE = "```"


def fenced(tag, body):
    return f"{FENCE}{tag}\n{body}\n{FENCE}"


@pytest.mark.parametrize("tag", ["python", "py", "Python3", "py3", "python title=solution.py"])
def test_python_tags_are_stripped(tag):
    assert code_rewards.extract_python_code(fenced(tag, "print(1)")) == "print(1)"


def test_other_languages_and_sample_io_are_never_graded():
    response = "\n".join([fenced("python", "print(int(input()) * 2)"), fenced("", "3"), fenced("bash", "python a.py")])
    # Python-tagged blocks win over an untagged sample-output block that comes last.
    assert code_rewards.extract_python_code(response) == "print(int(input()) * 2)"
    assert code_rewards.extract_python_code(fenced("cpp", "int main() {}")).startswith(FENCE)
    assert code_rewards.extract_python_code("no fences at all") == "no fences at all"
    # Untagged blocks still count when nothing is tagged Python.
    assert code_rewards.extract_python_code(fenced("", "x = 1") + fenced("", "y = 2")) == "y = 2"


def test_the_block_defining_the_tested_function_beats_a_trailing_example():
    solution = "def add(a, b):\n    return a + b"
    example = "print(add(1, 2))"
    response = fenced("python", solution) + "\nUsage:\n" + fenced("python", example)
    assert code_rewards.extract_python_code(response) == example  # unchanged without a name
    assert code_rewards.extract_python_code(response, "add") == solution
    assert code_rewards.extract_python_code(response, "missing") == example
    assert code_rewards.extract_python_code(fenced("python", "class Solution:\n    pass"), "Solution").startswith(
        "class Solution"
    )


def test_function_name_comes_from_the_tests():
    tests = [
        "assert add(1, 2) == 3",
        "assert math.isclose(add(0.1, 0.2), 0.3)",
        "assert sorted(add([1], [0])) == [0, 1]",
    ]
    assert code_rewards.function_name_from_tests(tests) == "add"
    assert code_rewards.function_name_from_tests(["assert Solution().run(1) == 1"]) == "Solution"
    assert code_rewards.function_name_from_tests(["assert len([1]) == 1", "not python ("]) is None
    assert code_rewards.function_name_from_tests("compressed") is None


@pytest.mark.parametrize(
    "program",
    [
        "def f(x: int | None) -> int:\n    return 0\n",
        "def f(x: list[int]) -> dict[str, int]:\n    return {}\n",
        "x: tuple[int, ...] = (1,)\n",
        "class A:\n    def f(self, *args: int | str, **kw: set[int]):\n        pass\n",
    ],
)
def test_future_annotations_added_when_annotations_need_it(program):
    rewritten = code_rewards.add_future_annotations(program)
    assert rewritten == "from __future__ import annotations\n" + program
    compile(rewritten, "<program>", "exec")


def test_future_import_follows_docstring_and_shebang_and_is_idempotent():
    program = '#!/usr/bin/env python\n"""Doc."""\nimport sys\ndef f(x: int | None): pass'
    rewritten = code_rewards.add_future_annotations(program)
    assert rewritten.splitlines()[:4] == [
        "#!/usr/bin/env python",
        '"""Doc."""',
        "from __future__ import annotations",
        "import sys",
    ]
    assert code_rewards.add_future_annotations(rewritten) == rewritten
    compile(rewritten, "<program>", "exec")
    program = "# -*- coding: utf-8 -*-\ndef f(x: int | None): pass"
    assert code_rewards.add_future_annotations(program).splitlines()[1] == "from __future__ import annotations"
    docstring_only = '"""Doc."""'
    assert code_rewards.add_future_annotations(docstring_only + "\nx: list[int] = []") == (
        '"""Doc."""\nfrom __future__ import annotations\nx: list[int] = []'
    )


@pytest.mark.parametrize(
    "program",
    [
        "def f(x: int) -> List[int]:\n    return []\n",  # typing generics already work
        "x = isinstance(1, int | str)\n",  # runtime union: not an annotation
        "def f(:\n",  # does not parse
        "from __future__ import annotations\ndef f(x: int | None): pass\n",
    ],
)
def test_programs_that_do_not_need_the_import_are_unchanged(program):
    assert code_rewards.add_future_annotations(program) == program


class _Response:
    def __init__(self, status, body=None):
        self.status_code = status
        self.body = body or {"results": [1, 1]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self.body


class _Session:
    def __init__(self, status=200, error=None, results=(1, 1)):
        self.status, self.error, self.results, self.payloads = status, error, list(results), []

    def post(self, *args, json=None, **kwargs):
        self.payloads.append(json)
        if self.error:
            raise self.error
        return _Response(self.status, {"results": self.results})


def _execute(monkeypatch, session, policy, program="def f(): pass", tests=("assert f() is None",), **settings):
    monkeypatch.setattr(code_rewards, "_get_session", lambda: session)
    args = SimpleNamespace(
        code_api_url="https://svc/test_program",
        code_pass_rate_reward_threshold=0.99,
        code_failure_policy=policy,
        **settings,
    )
    return asyncio.run(code_rewards.execute(args, fenced("python", program), list(tests)))


@pytest.mark.parametrize(
    "error", [requests.ConnectionError("down"), requests.Timeout("slow"), None], ids=["conn", "timeout", "503"]
)
def test_mask_policy_excludes_service_failures(monkeypatch, error):
    session = _Session(503, error)
    score, diagnostics = _execute(monkeypatch, session, "mask")
    assert score == 0.0 and diagnostics["status"] == "service_error"
    assert diagnostics[code_rewards.EXCLUDE_FROM_TRAINING] is True
    # The default policy still scores the failure zero and keeps it in training.
    score, diagnostics = _execute(monkeypatch, _Session(503, error), None)
    assert score == 0.0 and code_rewards.EXCLUDE_FROM_TRAINING not in diagnostics


def test_mask_policy_excludes_payload_rejections_but_not_harness_errors(monkeypatch):
    _, diagnostics = _execute(monkeypatch, _Session(413), "mask")
    assert diagnostics["status"] == "rejected" and diagnostics["rejection"] == "payload_too_large"
    assert diagnostics[code_rewards.EXCLUDE_FROM_TRAINING] is True
    for status in (500, 400):
        score, diagnostics = _execute(monkeypatch, _Session(status), "mask")
        assert score == 0.0 and diagnostics["status"] == "rejected"
        assert code_rewards.EXCLUDE_FROM_TRAINING not in diagnostics


def test_mask_policy_skips_oversized_payloads_without_a_request(monkeypatch):
    session = _Session()
    huge = ["assert f() is None  # " + "x" * code_rewards.MAX_PAYLOAD_BYTES]
    score, diagnostics = _execute(monkeypatch, session, "mask", tests=huge)
    assert score == 0.0 and session.payloads == []
    assert diagnostics["rejection"] == "payload_too_large" and diagnostics[code_rewards.EXCLUDE_FROM_TRAINING]


def test_execute_sends_the_rewritten_solution_block(monkeypatch):
    session = _Session()
    response = fenced("python", "def f(x: int | None) -> None:\n    return None") + fenced("python", "print(f(1))")
    monkeypatch.setattr(code_rewards, "_get_session", lambda: session)
    args = SimpleNamespace(code_api_url="https://svc/test_program", code_failure_policy=None)
    score, _ = asyncio.run(code_rewards.execute(args, response, ["assert f(1) is None"]))
    assert score == 1.0
    assert session.payloads[0]["program"].startswith("from __future__ import annotations\ndef f(")
    args.code_future_annotations = False
    asyncio.run(code_rewards.execute(args, response, ["assert f(1) is None"]))
    assert session.payloads[1]["program"].startswith("def f(")


def test_config_accepts_mask_and_rejects_unknown_policies():
    config = code_rewards.code_verifier_config(SimpleNamespace(code_api_url="https://svc/test_program"))
    assert config.failure_policy == "zero" and config.future_annotations is True
    mask = SimpleNamespace(code_api_url="https://svc/test_program", code_failure_policy="mask")
    assert code_rewards.code_verifier_config(mask).failure_policy == "mask"
    with pytest.raises(ValueError, match="'mask'"):
        code_rewards.code_verifier_config(SimpleNamespace(code_api_url="x/test_program", code_failure_policy="skip"))


def _registry(tmp_path, policy):
    config = {"api_url": "https://svc/test_program", "failure_policy": policy}
    path = tmp_path / f"verifiers-{policy}.json"
    path.write_text(
        json.dumps({"code": {"factory": "open_instruct.miles.rewards.code_rewards.CodeVerifier", "config": config}})
    )
    return SimpleNamespace(olmo_core=CoreConfig(reward_config=str(path)))


def _sample(group, index, response="```python\ndef f(): pass\n```"):
    return SimpleNamespace(
        index=index,
        group_index=group,
        tokens=[1, 2, 3 + index],
        response_length=1,
        response=response,
        prompt="p",
        status=SimpleNamespace(value="completed"),
        metadata={"verifiers": [{"name": "code", "target": ["assert f() is None"]}]},
    )


def test_masked_samples_leave_the_megatron_grpo_baseline_and_loss(tmp_path, monkeypatch):
    args = _registry(tmp_path, "mask")
    samples = [_sample(0, 0), _sample(0, 1), _sample(1, 2), _sample(1, 3)]
    # The last sample is a graded failure: it stays in its group's baseline.
    for sample, session in zip(samples, [_Session(), _Session(503), _Session(), _Session(results=[0])], strict=True):
        monkeypatch.setattr(code_rewards, "_get_session", lambda session=session: session)
        sample.reward = asyncio.run(rewards.registered_reward(args, sample))
    assert [s.reward for s in samples] == [1.0, 0.0, 1.0, 0.0]
    assert rewards.excluded_from_training(samples[1]) == "code:service_error"
    assert all(rewards.excluded_from_training(s) is None for s in samples if s is not samples[1])

    monkeypatch.setenv(grpo_hooks.OUTPUT_ENV, str(tmp_path))
    for sample in samples:
        sample.remove_sample = False
    raw, centered = grpo_hooks.post_process(
        SimpleNamespace(n_samples_per_prompt=2, grpo_std_normalization=True), samples
    )
    assert raw == [1.0, 0.0, 1.0, 0.0]
    # Group 0 has one graded sample: its baseline is itself, and the ungraded one gets no advantage.
    assert centered[:2] == [0.0, 0.0]
    assert centered[2:] == pytest.approx([0.70710578, -0.70710578])
    assert [s.remove_sample for s in samples] == [False, True, False, False]

    # A later successful re-grade clears the exclusion.
    monkeypatch.setattr(code_rewards, "_get_session", lambda: _Session())
    assert asyncio.run(rewards.registered_reward(args, samples[1])) == 1.0
    assert rewards.excluded_from_training(samples[1]) is None


def _compressed(tests):
    return base64.b64encode(zlib.compress(pickle.dumps(tests))).decode()


def test_safe_decoding_reads_plain_pickles_and_refuses_objects():
    tests = [{"input": "1\n", "output": "2\n"}]
    assert code_rewards.decode_tests_safely(_compressed(tests)) == tests
    assert code_rewards.decode_tests_safely(json.dumps(tests)) == tests
    assert code_rewards.decode_tests_safely(_compressed([SimpleNamespace(a=1)])) is None
    assert code_rewards.decode_tests_safely("not base64!") is None
    assert code_rewards.decode_tests_safely("") is None


def test_row_problems():
    stdio_ok = [{"input": "1\n", "output": "2\n"}]
    stdio_lines = [{"input": ["1"], "output": ["2"]}]
    problems = code_rewards.code_target_problems
    assert problems("code_stdio", stdio_ok, "Double n. Use Python.") == []
    assert problems("code_stdio", _compressed(stdio_lines), "Use Python.") == ["non_string_stdio_test"]
    assert problems("code_stdio", stdio_ok, "Double n.") == ["language_unspecified"]
    assert problems("code", ["assert add(1, 2) == 3"], "Write add(a, b).") == []
    assert problems("code", ["assert add(1, 2) == 3"], "Write an adder; additional notes.") == [
        "function_name_missing"
    ]
    assert problems("code", ["add(1, 2)"], "Write add.") == ["non_assert_test"]
    assert problems("code", [["assert add(1, 2) == 3"]], "Write add.") == ["non_string_test"]
    assert problems("code", "garbage", "Write add.") == ["undecodable_tests"]
    big = ["assert add(1, 2) == 3  # " + "x" * code_rewards.MAX_PAYLOAD_BYTES]
    assert problems("code", big, "Write add.") == ["payload_too_large"]
    row = {"metadata": {"query": "Write add.", "verifiers": [{"name": "code", "target": ["assert add(1, 2) == 3"]}]}}
    assert code_rewards.code_row_problems(row) == []
    assert {"function_name_missing", "language_unspecified"} == code_rewards.FIXABLE_PROBLEMS


def test_prompt_hints():
    assert code_rewards.add_function_name_hint("Sum two numbers.\n", "add") == (
        "Sum two numbers.\n\nYour solution must define a function named `add`."
    )
    assert code_rewards.add_function_name_hint("Q", "Solution").endswith("a class named `Solution`.")
    hinted = code_rewards.add_python_hint("Read n and print 2n.")
    assert hinted.startswith("Read n and print 2n.\n\n") and "Python 3" in hinted
    assert code_rewards.code_target_problems("code_stdio", [{"input": "1", "output": "2"}], hinted) == []
