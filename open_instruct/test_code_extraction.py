"""Ignore other-language fences without imposing a new formatting/grading policy."""

import asyncio
import re
from unittest.mock import Mock

import pytest

from open_instruct import code_extraction, ground_truth_utils
from open_instruct.miles.rewards import code_rewards


@pytest.fixture(params=["legacy", "legacy_stdio", "miles", "miles_stdio"])
def adapter(request, monkeypatch):
    stdio = request.param.endswith("_stdio")
    response = Mock(status_code=200)
    response.json.return_value = {"results": [True], "runtimes": [0.01]}
    session = Mock()
    session.post.return_value = response
    if request.param.startswith("legacy"):
        url = "https://service.invalid/test_program_stdio" if stdio else "https://service.invalid/test_program"
        verifier = ground_truth_utils.CodeVerifier(ground_truth_utils.CodeVerifierConfig(url, 1.0, 1.0, False))
        monkeypatch.setattr(verifier, "_get_session", lambda: session)
    else:
        verifier = code_rewards.CodeVerifier(
            code_rewards.ServiceConfig(
                api_url="https://service.invalid/test_program", stdio=stdio, failure_policy="raise"
            )
        )
        monkeypatch.setattr(code_rewards, "_get_session", lambda: session)
    tests = [{"input": "", "output": "42"}] if stdio else ["assert f() == 42"]
    return verifier, session, tests, stdio


@pytest.mark.parametrize("language", ["bash", "text", "javascript", "cpp", "json", "sh", "c++", "c#", "BASH"])
def test_trailing_other_language_is_not_submitted(adapter, language):
    verifier, session, tests, stdio = adapter
    program = "def f(): return 42"
    prediction = f"```python\n{program}\n```\n```{language}\nexample\n```"
    result = asyncio.run(verifier.async_call([], prediction, tests))
    assert result.score == 1.0
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"] == {"program": program, "tests": tests, "max_execution_time": 1.0}
    route = "/test_program_stdio" if stdio else "/test_program"
    assert session.post.call_args.args[0].endswith(route)


_COMPATIBILITY_CASES = [
    "```python\nx = 1\n```\n```python\nx = 2\n```",
    "```python\nx = 1\n```\nCorrection:\n```\nx = 2\n```",
    "```python\nx = 1\n```\n```python\nx =",
    "```python\nx = 1\n```\n```\nx =",
    "1. Solution:\n    ```python\nx = 1\ny = 2\n    ```",
    "1. Solution:\n\t```python\nx = 1\n\t```",
    "```python\nx = 1```",
    "```python x = 1```",
    "```PYTHON x = 1```",
    "```x = 1```",
    "```python``` is the format.\n```python\nx = 1\n```",
    "```python\nx = 1\n```unexpected",
    "```python\nx = 1\n```<|im_end|>\n<|endoftext|>\n",
    "```python\nx = 1\n```<|eot_id|>",
    "```python\nx = 1\n```<|end_of_text|>",
    "```python\nx = 1\n<|endoftext|>",
    "x = 1<|endoftext|>",
    "x = 1\n",
    "~~~python\nx = 1\n~~~",  # Tilde fences are outside this compatibility fix.
    "```python\nx =\n```",  # Selected syntax errors still go to execution.
    "```python\n\n```",
    "",
]


@pytest.mark.parametrize("response", _COMPATIBILITY_CASES)
def test_existing_submission_behavior_is_preserved(adapter, response):
    matches = re.findall(r"```(?:python)?(.*?)```", response, re.DOTALL)
    expected = matches[-1].strip() if matches else response
    assert code_extraction.extract_python_code(response) == expected
    assert code_rewards.extract_python_code(response) == expected
    verifier, session, tests, _ = adapter
    result = asyncio.run(verifier.async_call([], response, tests))
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"]["program"] == expected
    assert result.score == 1.0  # Preserve service handling even for malformed/empty submissions.


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("```python3\nx = 1\n```", "x = 1"),
        ("```py\nx = 1\n```", "x = 1"),
        ("```PYTHON\nx = 1\n```", "x = 1"),
        ("```python\r\nx = 1\r\n```", "x = 1"),
        ("```\nx = 1\n```\n```bash\necho 1\n```", "x = 1"),
        ("```python\nx = 1\n```\n```bash\necho", "x = 1"),
        ("```bash\necho 1\n```", "```bash\necho 1\n```"),
    ],
)
def test_language_selection(response, expected):
    assert code_extraction.extract_python_code(response) == expected
    assert code_rewards.extract_python_code(response) == expected
    verifier = ground_truth_utils.CodeVerifier(
        ground_truth_utils.CodeVerifierConfig("https://service.invalid/test_program", 1.0, 1.0, False)
    )
    assert verifier.extract_python_code(response) == expected


@pytest.mark.parametrize(
    "tag", ["python3.11", "python3.10", "python2", "python2.7", "python-3.11", "PYTHON3.11", "py3.11"]
)
def test_versioned_python_submission_reaches_service(adapter, tag):
    verifier, session, tests, _ = adapter
    # Include an earlier draft: skipping the versioned block would silently
    # select that draft, not just send an invalid raw response.
    program = "def f(): return 42"
    response = f"```python\ndef f(): return 0\n```\n```{tag}\n{program}\n```"
    result = asyncio.run(verifier.async_call([], response, tests))
    assert result.score == 1.0
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"]["program"] == program
    assert code_extraction.extract_python_code(response) == program
