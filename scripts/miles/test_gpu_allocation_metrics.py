import pytest
from scripts.miles import gpu_allocation_metrics


def job():
    return {
        "id": "attempt-1",
        "kind": "execution",
        "limits": {"gpus": ["gpu-a", "gpu-b"]},
        "status": {"started": "1970-01-01T00:00:10Z", "exited": "1970-01-01T00:01:00Z"},
    }


def test_extra_assigned_gpu_changes_allocation_rate_without_changing_inference_rate():
    result = gpu_allocation_metrics.summarize({"response_tokens_per_second": 100}, 1, (20, 50), [job()])
    assert result["inference_gpus"] == 1
    assert result["allocated_gpus"] == 2
    assert result["response_tokens_per_inference_gpu_second"] == 100
    assert result["response_tokens_per_allocated_gpu_second"] == 50


def test_unknown_allocation_is_not_inferred_from_inference_gpu_count():
    result = gpu_allocation_metrics.summarize({"response_tokens_per_second": 100}, 1, (20, 50))
    assert result["allocated_gpus"] is None
    assert result["response_tokens_per_allocated_gpu_second"] is None


def test_wrong_attempt_window_or_incomplete_allocation_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        gpu_allocation_metrics.summarize({"response_tokens_per_second": 100}, 1, (5, 50), job())
    with pytest.raises(ValueError, match="exactly one"):
        gpu_allocation_metrics.summarize({"response_tokens_per_second": 100}, 1, (20, 50), [job(), job()])
    with pytest.raises(ValueError, match="cannot cover"):
        gpu_allocation_metrics.summarize({"response_tokens_per_second": 100}, 3, (20, 50), job())
