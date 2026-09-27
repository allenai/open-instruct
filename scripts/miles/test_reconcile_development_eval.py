import json
from types import SimpleNamespace

import pytest

reconcile_development_eval = pytest.importorskip(
    "reconcile_development_eval", reason="Requires the pinned olmo-eval environment", exc_type=ModuleNotFoundError
)


def fixture_source(tmp_path, count=4):
    source = tmp_path / "update-00000100-example"
    source.mkdir()
    item = dict(id="q1", task="development", samples=4, prompt="6 times 7?", label=["42"], verifier="math")
    (source / "selected-panel.jsonl").write_text(json.dumps(item) + "\n")
    rows = [
        dict(id="q1", task="development", replicate=i, text="</think> \\boxed{42}", finish={"type": "stop"}, tokens=10)
        for i in range(count)
    ]
    (source / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source


def test_partial_generation_is_not_published(tmp_path):
    source = fixture_source(tmp_path, count=3)
    assert reconcile_development_eval.read_development(source) is None


def test_duplicate_response_is_rejected(tmp_path):
    source = fixture_source(tmp_path)
    path = source / "generations.jsonl"
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile_development_eval.read_development(source)


def test_corrected_metrics_publication_and_retry(tmp_path):
    source = fixture_source(tmp_path)
    destination = tmp_path / "corrected"
    original = (source / "generations.jsonl").read_bytes()

    def fail(receipt, output):
        raise RuntimeError("temporary publication failure")

    with pytest.raises(RuntimeError):
        reconcile_development_eval.reconcile(source, destination, {}, SimpleNamespace(publish=fail))
    assert not (destination / "complete.json").exists()
    calls = []
    publisher = SimpleNamespace(publish=lambda receipt, output: calls.append(receipt))
    assert reconcile_development_eval.reconcile(source, destination, {}, publisher)
    assert calls[0]["update"] == 100
    task = json.loads((destination / "metrics.json").read_text())["tasks"][0]
    assert task["task"] == "development_gold_v2"
    assert task["metrics"]["pass_at_1"]["mean"] == task["metrics"]["pass_at_4"]["mean"] == 1
    assert not reconcile_development_eval.reconcile(source, destination, {}, publisher)
    assert len(calls) == 1
    assert (source / "generations.jsonl").read_bytes() == original


def test_publisher_uses_image_python_not_scoring_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        reconcile_development_eval.subprocess, "run", lambda command, **kw: calls.append((command, kw))
    )
    publisher = reconcile_development_eval.Publisher("/tmp/evaluate.py", "/usr/local/bin/python")
    publisher.publish({"update": 100}, tmp_path)
    command, options = calls[0]
    assert command == [
        "/usr/local/bin/python",
        "/tmp/evaluate.py",
        "publish",
        str(tmp_path / "publication-receipt.json"),
        "--results",
        str(tmp_path),
    ]
    assert options["check"] and options["timeout"] == 300
    assert json.loads((tmp_path / "publication-receipt.json").read_text()) == {"update": 100}
