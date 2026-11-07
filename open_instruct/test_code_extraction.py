"""Code submissions must survive explanatory fences without grading abandoned drafts."""

import asyncio
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


@pytest.mark.parametrize("suffix", ["```bash\npython solution.py\n```", "```text\n42\n```", "```\n42\n```"])
def test_trailing_explanation_is_not_submitted(adapter, suffix):
    verifier, session, tests, stdio = adapter
    program = "def f(): return 42"
    result = asyncio.run(verifier.async_call([], f"```python\n{program}\n```\n{suffix}", tests))
    assert result.score == 1.0
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"] == {"program": program, "tests": tests, "max_execution_time": 1.0}
    route = "/test_program_stdio" if stdio else "/test_program"
    assert session.post.call_args.args[0].endswith(route)


@pytest.mark.parametrize(
    "response",
    [
        "```bash\npython solution.py\n```",
        "```python\ndef f(): return 42\n```\n```python\ndef f():",
        "```python\ndef f(): return 42\n```\n```\ndef f():",
        "```python\n\n```",
        "",
    ],
)
def test_invalid_submission_scores_zero_without_http(adapter, response):
    verifier, session, tests, _ = adapter
    result = asyncio.run(verifier.async_call([], response, tests))
    assert result.score == 0.0
    session.post.assert_not_called()
    if isinstance(verifier, code_rewards.CodeVerifier):
        assert result.diagnostics["status"] == "completed"
        assert result.diagnostics["outcome"] == "invalid_code"
        assert result.diagnostics["http_status"] is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("```python\nx = 1\n```\n```python\nx = 2\n```", "x = 2"),
        ("```python3\nx = 1\n```", "x = 1"),
        ("```py\nx = 1\n```", "x = 1"),
        ('```PYTHON title="solution.py"\nx = 1\n```', "x = 1"),
        ("```\nx = 1\n```\n```\nx = 2\n```", "x = 2"),
        ("```\nx = 1\n```\n```bash\necho 1\n```", "x = 1"),
        ("~~~python\nx = 1\n~~~", "x = 1"),
        ("````python\nx = '```'\n````", "x = '```'"),
        ("  ```python\nx = 1\n  ```", "x = 1"),
        ("```python\r\nx = 1\r\n```", "x = 1"),
        ("```python\nx = 1\n```<|im_end|>\n<|endoftext|>\n", "x = 1"),
        ("```python\nx = 1\n```<|eot_id|>", "x = 1"),
        ("x = 1<|endoftext|>", "x = 1"),
        ("x = 1\n", "x = 1\n"),
        ('x = """\n```bash\necho hi\n```\n"""', 'x = """\n```bash\necho hi\n```\n"""'),
        ('x = "<|endoftext|>"', 'x = "<|endoftext|>"'),
        ("```javascript\nx = 1\n```", None),
        ("```python\nx = 1\n```\n```python\nx =", None),
        ("```python\nx = 1\n~~~", None),
        ("```python\nx = 1\n```unexpected", None),
        ("```python\nx = 1\n<|endoftext|>", None),
        ("```python\nx = 1\n```\n```bash\necho", "x = 1"),
        ("```python\nx =\n```", "x ="),  # Syntax errors in selected code belong to execution.
    ],
)
def test_python_extraction_contract(response, expected):
    assert code_extraction.extract_python_code(response) == expected
    assert code_rewards.extract_python_code(response) == expected
    verifier = ground_truth_utils.CodeVerifier(
        ground_truth_utils.CodeVerifierConfig("https://service.invalid/test_program", 1.0, 1.0, False)
    )
    assert verifier.extract_python_code(response) == expected
