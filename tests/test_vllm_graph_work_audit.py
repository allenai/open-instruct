"""CPU observations; no vLLM imports, CUDA execution or acceptance claims."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[1] / "open_instruct/vllm_graph_work_audit.py"
    spec = importlib.util.spec_from_file_location("vllm_graph_work_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_has_no_import_or_output(audit, monkeypatch):
    monkeypatch.delenv("OI_VLLM_GRAPH_WORK_AUDIT", raising=False)
    monkeypatch.setattr(audit.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    assert audit.install(tensor_parallel_size=7, multiprocessing=None) is None


@pytest.mark.parametrize("tp,mp", [(2, "0"), (1, "1"), (1, None)])
def test_unsupported_process_contract_rejected(audit, monkeypatch, tp, mp):
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    with pytest.raises(ValueError):
        audit.install(tensor_parallel_size=tp, multiprocessing=mp)


def test_exact_identity_rejects_changed_source(audit, tmp_path):
    path = tmp_path / "source.py"
    path.write_text("changed")
    with pytest.raises(ValueError, match="Unsupported"):
        audit.verify_sources({name: SimpleNamespace(__file__=str(path)) for name in audit.SOURCE_HASHES})


def test_executor_backend_and_missing_output_rejected(audit, monkeypatch):
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    with pytest.raises(ValueError, match="uni"):
        audit.install(tensor_parallel_size=1, multiprocessing="0", executor_backend="mp")
    monkeypatch.delenv("OI_VLLM_GRAPH_WORK_AUDIT_DIR", raising=False)
    with pytest.raises(ValueError, match="explicit"):
        audit.install(tensor_parallel_size=1, multiprocessing="0")


@pytest.mark.parametrize("counts,total", [({"a": 2}, 3), ({"a": -1}, -1), ({"a": True}, 1)])
def test_bad_scheduler_accounting_rejected(audit, counts, total):
    with pytest.raises(ValueError):
        audit.scheduled_work(SimpleNamespace(num_scheduled_tokens=counts, total_num_scheduled_tokens=total))


def test_branch_counts_scope_and_errors(audit, tmp_path):
    context = SimpleNamespace(batch_descriptor="x", cudagraph_runtime_mode="FULL")
    native = SimpleNamespace(
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: context,
        CUDAGraphMode=SimpleNamespace(NONE="NONE"),
    )
    events = []

    class Graph:
        runtime_mode = "FULL"

        def __init__(self):
            self.concrete_cudagraph_entries = {}

        def __call__(self, value):
            if value == "fail":
                raise RuntimeError("native")
            self.concrete_cudagraph_entries["x"] = SimpleNamespace(cudagraph=object())
            events.append(value)
            return value

    graph = Graph()

    class Runner:
        def execute_model(self, output, value="work"):
            if not output.total_num_scheduled_tokens:
                return self._dummy_run()
            graph(value)
            graph(value)
            self._dummy_run()
            return "result"

        def _dummy_run(self):
            graph("dummy")

        def profile_run(self):
            graph("profile")

    journal = audit.Journal(tmp_path, flush_every=2)
    audit.instrument(Runner, Graph, native, journal)
    runner = Runner()
    runner.profile_run()
    output = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1}, total_num_scheduled_tokens=2)
    assert runner.execute_model(output) == "result"
    assert not journal.path.exists()
    assert runner.execute_model(output) == "result"
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["scheduled_tokens"] == 2 and r["scheduled_requests"] == 2 for r in rows)
    assert all(sum(x["calls"] for x in r["wrapper_calls"]) == 2 for r in rows)
    assert all(r["excluded_dummy_profile_calls"] == 1 for r in rows)
    runner.execute_model(SimpleNamespace(num_scheduled_tokens={}, total_num_scheduled_tokens=0))
    assert len(journal.path.read_text().splitlines()) == 2
    with pytest.raises(RuntimeError, match="native"):
        runner.execute_model(output, "fail")
    last = json.loads(journal.path.read_text().splitlines()[-1])
    assert last["status"] == "error" and last["wrapper_calls"] == []
    # Context restored even after failure: profile cannot become busy work.
    runner.profile_run()
    assert len(journal.path.read_text().splitlines()) == 3


def test_capture_eager_mismatch_replay_are_separate(audit):
    context = SimpleNamespace(batch_descriptor="x", cudagraph_runtime_mode="FULL")
    native = SimpleNamespace(
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: context,
        CUDAGraphMode=SimpleNamespace(NONE="NONE"),
    )
    wrapper = SimpleNamespace(runtime_mode="FULL", concrete_cudagraph_entries={})
    assert audit.graph_branch(wrapper, native)[0] == "capture"
    wrapper.concrete_cudagraph_entries["x"] = SimpleNamespace(cudagraph=object())
    assert audit.graph_branch(wrapper, native)[0] == "replay-returned"
    context.cudagraph_runtime_mode = "PIECEWISE"
    assert audit.graph_branch(wrapper, native)[0] == "mode-mismatch"
    context.cudagraph_runtime_mode = "NONE"
    assert audit.graph_branch(wrapper, native)[0] == "eager"
    native.is_forward_context_available = lambda: False
    assert audit.graph_branch(wrapper, native)[0] == "no-context"


def test_journal_buffer_flush_and_absolute_path(audit, tmp_path):
    with pytest.raises(ValueError):
        audit.Journal("relative")
    with pytest.raises(ValueError):
        audit.Journal(tmp_path, 0)
    journal = audit.Journal(tmp_path, 32)
    for n in range(31):
        journal.append({"status": "returned", "number": n})
    assert not journal.path.exists()
    journal.flush()
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert len(rows) == 31 and not journal.pending
    assert [r["number"] for r in rows] == list(range(31))
