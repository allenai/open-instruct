"""CPU observations; no vLLM imports, CUDA execution or acceptance claims."""

import ast
import asyncio
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

AUDIT_PATH = Path(__file__).resolve().parents[1] / "open_instruct/vllm_graph_work_audit.py"

# Minimal stand-ins for the guarded native modules, imported only in spawned children.
FAKE_VLLM = {
    "vllm/__init__.py": "",
    "vllm/envs.py": "VLLM_USE_V2_MODEL_RUNNER = False\n",
    "vllm/compilation/__init__.py": "",
    "vllm/compilation/cuda_graph.py": """from types import SimpleNamespace

CUDAGraphMode = SimpleNamespace(NONE="NONE")
_context = SimpleNamespace(batch_descriptor="x", cudagraph_runtime_mode="FULL")


def is_forward_context_available():
    return True


def get_forward_context():
    return _context


class CUDAGraphWrapper:
    runtime_mode = "FULL"

    def __init__(self):
        self.concrete_cudagraph_entries = {"x": SimpleNamespace(cudagraph=object())}

    def __call__(self):
        return None
""",
    "vllm/v1/__init__.py": "",
    "vllm/v1/worker/__init__.py": "",
    "vllm/v1/worker/gpu_model_runner.py": """from vllm.compilation.cuda_graph import CUDAGraphWrapper


class GPUModelRunner:
    def __init__(self):
        self.graph = CUDAGraphWrapper()

    def execute_model(self, scheduler_output):
        if scheduler_output.total_num_scheduled_tokens:
            self.graph()
            self.graph()
        return "output"

    def _dummy_run(self):
        self.graph()

    def profile_run(self):
        self.graph()
""",
}


def load_audit():
    spec = importlib.util.spec_from_file_location("vllm_graph_work_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audit():
    return load_audit()


def write_fake_vllm(root):
    for relative, source in FAKE_VLLM.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return {
        "vllm.compilation.cuda_graph": hashlib.sha256(
            (root / "vllm/compilation/cuda_graph.py").read_bytes()
        ).hexdigest(),
        "vllm.v1.worker.gpu_model_runner": hashlib.sha256(
            (root / "vllm/v1/worker/gpu_model_runner.py").read_bytes()
        ).hexdigest(),
    }


def engine_core_child(conn, fake_root, source_hashes, scenario):
    """Spawned stand-in for EngineCore: one uni TP1 worker answering collective_rpc by method name."""
    sys.path.insert(0, fake_root)
    audit = load_audit()
    audit.SOURCE_HASHES = source_hashes
    runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    runner = runner_module.GPUModelRunner()
    if scenario == "v2-runner":
        runner = type("GPUModelRunner", (), {})()
    elif scenario == "v2-env":
        importlib.import_module("vllm.envs").VLLM_USE_V2_MODEL_RUNNER = True

    class Worker(audit.GraphWorkAuditWorkerExtension):
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(world_size=1, tensor_parallel_size=1, distributed_executor_backend="uni")
        )
        model_runner = runner

    worker = Worker()
    while True:
        method, args = conn.recv()
        if method == "stop":
            return
        try:
            if method == "work":
                tokens = args[0]
                output = SimpleNamespace(
                    num_scheduled_tokens={"a": tokens} if tokens else {}, total_num_scheduled_tokens=tokens
                )
                result = worker.model_runner.execute_model(output)
            else:
                result = getattr(worker, method)(*args)
            conn.send(("ok", result))
        except Exception as error:
            conn.send(("error", f"{type(error).__name__}: {error}"))


class ChildEngine:
    """Actor-side engine client whose collective_rpc reaches the spawned child, like AsyncLLM."""

    def __init__(self, tmp_path, monkeypatch, scenario="v1"):
        monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
        monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT_DIR", str(tmp_path / "journals"))
        hashes = write_fake_vllm(tmp_path / "fake")
        context = multiprocessing.get_context("spawn")
        self.conn, child_conn = context.Pipe()
        self.process = context.Process(
            target=engine_core_child, args=(child_conn, str(tmp_path / "fake"), hashes, scenario), daemon=True
        )
        self.process.start()

    def call(self, method, *args):
        self.conn.send((method, args))
        if not self.conn.poll(60):
            raise TimeoutError(method)
        status, result = self.conn.recv()
        if status == "error":
            raise RuntimeError(result)
        return result

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return [self.call(method, *args)]

    def close(self):
        self.conn.send(("stop", ()))
        self.process.join(30)


@pytest.fixture
def child_engine(tmp_path, monkeypatch):
    engines = []

    def start(scenario="v1"):
        engines.append(ChildEngine(tmp_path, monkeypatch, scenario))
        return engines[-1]

    yield start
    for engine in engines:
        engine.close()


def test_disabled_has_no_import_or_extension(audit, monkeypatch):
    monkeypatch.delenv("OI_VLLM_GRAPH_WORK_AUDIT", raising=False)
    monkeypatch.setattr(audit.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    kwargs = {"tensor_parallel_size": 7}
    assert audit.configure_engine_kwargs(kwargs) is False
    assert kwargs == {"tensor_parallel_size": 7}


@pytest.mark.parametrize("tp,backend", [(2, "uni"), (1, "mp"), (1, None)])
def test_unsupported_engine_contract_rejected(audit, monkeypatch, tmp_path, tp, backend):
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="uni-executor TP1"):
        audit.configure_engine_kwargs({"tensor_parallel_size": tp, "distributed_executor_backend": backend})


def test_exact_identity_rejects_changed_source(audit, tmp_path):
    path = tmp_path / "source.py"
    path.write_text("changed")
    with pytest.raises(ValueError, match="Unsupported"):
        audit.verify_sources({name: SimpleNamespace(__file__=str(path)) for name in audit.SOURCE_HASHES})


def test_engine_kwargs_register_extension_and_reject_conflicts(audit, monkeypatch, tmp_path):
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    monkeypatch.delenv("OI_VLLM_GRAPH_WORK_AUDIT_DIR", raising=False)
    with pytest.raises(ValueError, match="explicit"):
        audit.configure_engine_kwargs({"distributed_executor_backend": "uni"})
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="existing vLLM worker extension"):
        audit.configure_engine_kwargs({"distributed_executor_backend": "uni", "worker_extension_cls": "other.Ext"})
    kwargs = {"tensor_parallel_size": 1, "distributed_executor_backend": "uni"}
    assert audit.configure_engine_kwargs(kwargs) is True
    assert kwargs["worker_extension_cls"] == audit.WORKER_EXTENSION
    assert audit.WORKER_EXTENSION.rsplit(".", 1)[1] == audit.GraphWorkAuditWorkerExtension.__name__


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


def test_boundary_flush_has_numbered_cutoff_without_claiming_later_work(audit, tmp_path):
    journal = audit.Journal(tmp_path)
    journal.append({"status": "returned", "scheduled_tokens": 2})
    journal.append({"status": "returned", "scheduled_tokens": 4})
    audit._INSTALLED = journal
    audit._ACTIVATION = {"process_id": os.getpid(), "actor_process_id": 7}
    with pytest.raises(RuntimeError, match="different actor"):
        audit.cutoff(8)
    result = audit.cutoff(7)
    assert result["busy_execution_count"] == 2 and result["prior_journal_records"] == 2
    assert result["process_id"] == os.getpid() and result["actor_process_id"] == 7
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [r["journal_record"] for r in rows] == [1, 2, 3]
    assert rows[-1]["status"] == "boundary-flushed" and not journal.pending
    journal.append({"status": "returned", "scheduled_tokens": 1})
    assert journal.busy_execution_count == 3 and len(journal.pending) == 1
    assert len(journal.path.read_text().splitlines()) == 3  # Later work is outside the cutoff.
    with pytest.raises(ValueError):
        journal.boundary("invented")


def test_enabled_cutoff_without_activated_observer_fails(audit, monkeypatch):
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    with pytest.raises(RuntimeError, match="not activated in this worker"):
        audit.cutoff(os.getppid())
    with pytest.raises(RuntimeError, match="not activated for this engine"):
        asyncio.run(audit.cutoff_engine(SimpleNamespace(), None))
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "0")
    assert asyncio.run(audit.cutoff_engine(SimpleNamespace(), None)) == {"status": "disabled"}


def test_parent_only_activation_rejected_before_native_imports(audit, monkeypatch, tmp_path):
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "1")
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    with pytest.raises(RuntimeError, match="not the actor process"):
        audit.activate(SimpleNamespace(), os.getpid())
    with pytest.raises(RuntimeError, match="direct child"):
        audit.activate(SimpleNamespace(), os.getppid() + 1)
    monkeypatch.setenv("OI_VLLM_GRAPH_WORK_AUDIT", "0")
    with pytest.raises(RuntimeError, match="did not reach"):
        audit.activate(SimpleNamespace(), os.getppid())
    assert audit._INSTALLED is None and audit._ACTIVATION is None


@pytest.mark.parametrize(
    "change",
    [
        {"status": "installed"},
        {"process_id": 100},
        {"process_id": "200"},
        {"parent_process_id": 101},
        {"actor_process_id": 101},
    ],
)
def test_activation_ack_must_come_from_spawned_child_of_actor(audit, change):
    ack = {"status": "activated", "process_id": 200, "parent_process_id": 100, "actor_process_id": 100}
    assert audit.validate_activation([dict(ack)], 100) == ack
    with pytest.raises(RuntimeError, match="spawned worker"):
        audit.validate_activation([ack | change], 100)
    for results in ([], [ack, ack], [None], (ack,)):
        with pytest.raises(RuntimeError, match="one graph audit"):
            audit.validate_activation(results, 100)


def test_cutoff_must_come_from_activated_process(audit):
    activation = {"process_id": 200, "actor_process_id": 100}
    report = {"status": "boundary-flushed", "process_id": 200}
    assert audit.validate_cutoff([report], activation) == report
    for bad in ({"status": "disabled", "process_id": 200}, {"status": "boundary-flushed", "process_id": 201}):
        with pytest.raises(RuntimeError, match="activated worker"):
            audit.validate_cutoff([bad], activation)


def test_spawned_worker_activation_journals_child_work_and_cutoff(audit, child_engine, tmp_path):
    engine = child_engine()
    child_pid = engine.process.pid
    with pytest.raises(RuntimeError, match="not the actor process"):
        engine.call(audit.ACTIVATE_RPC, child_pid)
    with pytest.raises(RuntimeError, match="direct child"):
        engine.call(audit.ACTIVATE_RPC, os.getpid() + 1)
    with pytest.raises(RuntimeError, match="not activated in this worker"):
        engine.call(audit.CUTOFF_RPC, os.getpid())
    engine.call("work", 3)  # Before activation: never journaled.

    activation = asyncio.run(audit.activate_engine(engine))
    assert activation["process_id"] == child_pid != os.getpid()
    assert activation["parent_process_id"] == activation["actor_process_id"] == os.getpid()
    assert activation["runner_class"] == "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
    assert asyncio.run(audit.activate_engine(engine)) == activation  # Idempotent for the same actor.
    assert audit._INSTALLED is None and audit._ACTIVATION is None  # The actor itself is not instrumented.

    for tokens in (5, 0, 2):
        assert engine.call("work", tokens) == "output"
    report = asyncio.run(audit.cutoff_engine(engine, activation))
    assert report["busy_execution_count"] == 2 and report["process_id"] == child_pid
    engine.call("work", 4)  # After the cutoff: outside its scope.

    journals = list((tmp_path / "journals").iterdir())
    assert [p.name for p in journals] == [Path(activation["journal_path"]).name]
    assert str(child_pid) in journals[0].name
    rows = [json.loads(line) for line in journals[0].read_text().splitlines()]
    assert [r["status"] for r in rows] == ["installed", "returned", "returned", "boundary-flushed"]
    assert all(r["process_id"] == child_pid for r in rows)
    assert rows[0]["actor_process_id"] == os.getpid()
    assert [r["scheduled_tokens"] for r in rows[1:3]] == [5, 2]
    assert all(
        r["wrapper_calls"]
        == [{"branch": "replay-returned", "calls": 2, "context_mode": "FULL", "wrapper_mode": "FULL"}]
        for r in rows[1:3]
    )


@pytest.mark.parametrize("scenario", ["v2-runner", "v2-env"])
def test_spawned_worker_rejects_non_v1_live_runner(audit, child_engine, tmp_path, scenario):
    engine = child_engine(scenario)
    with pytest.raises(RuntimeError, match="live V1 GPUModelRunner"):
        asyncio.run(audit.activate_engine(engine))
    assert not (tmp_path / "journals").exists()


def test_actor_activates_before_serving_and_cuts_off_through_worker_rpc():
    source = Path(__file__).resolve().parents[1] / "open_instruct/vllm_utils.py"
    text = source.read_text()
    assert "vllm_graph_work_audit.install(" not in text and "flush_boundary" not in text
    tree = ast.parse(text)
    setup = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_setup_and_start_async_engine"
    )

    def calls(node):  # In source order; ast.walk is breadth-first.
        found = sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
        return [ast.unparse(n.func) for n in found]

    order = calls(setup)
    assert order.index("vllm_graph_work_audit.configure_engine_kwargs") < order.index("vllm.AsyncEngineArgs")
    init = next(
        n for n in ast.walk(setup) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_init_engine_and_server"
    )
    order = calls(init)
    activate = order.index("vllm_graph_work_audit.activate_engine")
    assert order.index("vllm.AsyncLLMEngine.from_engine_args") < activate < order.index("build_app")
    flush = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "flush_graph_work_audit")
    assert "vllm_graph_work_audit.cutoff_engine(self.llm_engine, self.graph_work_activation)" in ast.unparse(flush)


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
