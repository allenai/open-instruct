from types import SimpleNamespace

import pytest
from scripts.miles import evaluator_startup_budget


def test_readiness_arguments_and_crashes_preserved():
    calls = []

    def wait(init_queue, num_workers, timeout=900, **kwargs):
        calls.append((init_queue, num_workers, timeout, kwargs))
        if kwargs.get("workers") == "dead":
            raise RuntimeError("worker crashed")
        return {"worker": 1201}

    runner = SimpleNamespace(wait_for_init_times=wait)
    evaluator_startup_budget.configure(runner, 2400)
    assert runner.wait_for_init_times("queue", 1, workers="live", result_queue="results") == {"worker": 1201}
    assert calls == [("queue", 1, 2400, {"workers": "live", "result_queue": "results"})]
    with pytest.raises(RuntimeError, match="worker crashed"):
        runner.wait_for_init_times("queue", 1, workers="dead")


@pytest.mark.parametrize("timeout", [0, 900, 3601])
def test_reject_unbounded_or_unnecessary_budget(timeout):
    with pytest.raises(ValueError):
        evaluator_startup_budget.configure(None, timeout)


def test_reject_changed_runtime_contract():
    def wait(timeout=1800):
        pass

    with pytest.raises(RuntimeError, match="revalidate"):
        evaluator_startup_budget.configure(SimpleNamespace(wait_for_init_times=wait), 2400)
