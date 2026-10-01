"""CPU observations; no vLLM imports, CUDA execution or acceptance claims."""

import ast
import asyncio
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


def test_boundary_flush_has_numbered_cutoff_without_claiming_later_work(audit, tmp_path, monkeypatch):
    journal = audit.Journal(tmp_path)
    journal.append({"status": "returned", "scheduled_tokens": 2})
    journal.append({"status": "returned", "scheduled_tokens": 4})
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    audit._INSTALLED = journal
    result = asyncio.run(audit.flush_boundary())
    assert result["busy_execution_count"] == 2 and result["prior_journal_records"] == 2
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [r["journal_record"] for r in rows] == [1, 2, 3]
    assert rows[-1]["status"] == "boundary-flushed" and not journal.pending
    journal.append({"status": "returned", "scheduled_tokens": 1})
    assert journal.busy_execution_count == 3 and len(journal.pending) == 1
    assert len(journal.path.read_text().splitlines()) == 3  # Later work is outside the cutoff.
    with pytest.raises(ValueError):
        journal.boundary("invented")


def test_enabled_boundary_without_installed_observer_fails(audit, monkeypatch):
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    with pytest.raises(RuntimeError, match="not installed"):
        asyncio.run(audit.flush_boundary())


def test_main_cutoffs_use_native_results_times_tuple_and_all_engines(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "open_instruct/grpo_fast.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "flush_graph_work_cutoffs")
    captured = []
    records = []
    native = SimpleNamespace(environ={"OI_VLLM_GRAPH_WORK_AUDIT": "1"})

    def wait(refs, **kwargs):
        captured.append((refs, kwargs))
        return refs, [0.1] * len(refs)

    namespace = {
        "os": native,
        "ray_get_with_progress": wait,
        "response_work_audit": SimpleNamespace(record=lambda *args: records.append(args)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    fn = namespace[node.name]
    engines = [
        SimpleNamespace(flush_graph_work_audit=SimpleNamespace(remote=lambda: {"status": "boundary-flushed"}))
        for _ in range(5)
    ]
    fn(SimpleNamespace(output_dir="output"), engines, 2)
    assert len(captured[0][0]) == 5 and captured[0][1]["timeout"] == 120
    assert len(records[0][2]["engine_cutoffs"]) == 5
    namespace["ray_get_with_progress"] = lambda *args, **kwargs: ([{"status": "disabled"}], [0.1])
    with pytest.raises(RuntimeError, match="Missing"):
        fn(SimpleNamespace(output_dir="output"), engines, 2)
    native.environ.clear()
    fn(SimpleNamespace(output_dir="output"), engines, 2)
    assert len(records) == 1
