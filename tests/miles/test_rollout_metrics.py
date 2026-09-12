"""Code-service outcomes recorded on samples aggregate into per-collection metrics."""

from types import SimpleNamespace

from open_instruct.miles import rollout_metrics


def _sample(diagnostics):
    return SimpleNamespace(metadata={"verifier_diagnostics": diagnostics} if diagnostics is not None else {})


def test_code_service_metrics_count_rejections_by_status():
    samples = [
        _sample({"code": {"status": "ok", "http_status": 200, "program_chars": 10, "tests": 3}}),
        _sample({"code_stdio": {"status": "rejected", "http_status": 413, "program_chars": 9000, "tests": 8}}),
        _sample({"code": {"status": "rejected", "http_status": 500, "program_chars": 40, "tests": 2}}),
        _sample({"general-quality": {"latency": 1.0}}),  # a judge record, no program: ignored
        _sample(None),
    ]
    metrics = rollout_metrics.code_service_metrics(samples)
    assert metrics["rollout/code_verifier/samples"] == 3
    assert metrics["rollout/code_verifier/rejected"] == 2
    assert abs(metrics["rollout/code_verifier/rejected_fraction"] - 2 / 3) < 1e-9
    assert metrics["rollout/code_verifier/rejected_413"] == 1
    assert metrics["rollout/code_verifier/rejected_500"] == 1


def test_no_code_samples_means_no_metrics():
    assert rollout_metrics.code_service_metrics([_sample({"math": {"latency": 0.1}})]) == {}
