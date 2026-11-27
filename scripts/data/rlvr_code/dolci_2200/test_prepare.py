import hashlib
import json

import pytest
from prepare import replacement


def sample(output):
    target = json.dumps([{"input": ["001"], "output": output}])
    row = {"custom_id": "test-row", "id": "source", "dataset": ["code_stdio"], "ground_truth": [target]}
    entry = {
        "source_id": "source",
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "function_name": "",
        "tests": "1",
        "wrapped_outputs": "1",
    }
    return row, entry


@pytest.mark.parametrize(("lines", "text"), [([], ""), (["001"], "001"), ([" 01 ", "", "end", ""], " 01 \n\nend\n")])
def test_exact_text(lines, text):
    row, entry = sample(lines)
    (target,) = replacement(row, entry)["ground_truth"]
    assert json.loads(target) == [{"input": ["001"], "output": text}]
    assert json.loads(row["ground_truth"][0])[0]["output"] == lines


def test_quarantine_nested_return():
    row, entry = sample([[2]])
    assert replacement(row, dict(entry, function_name="array_diff")) is None
    with pytest.raises(ValueError):
        replacement(row, entry)


def test_reject_drift():
    row, entry = sample(["0"])
    with pytest.raises(ValueError, match="differs"):
        replacement(row, dict(entry, target_sha256="0" * 64))
