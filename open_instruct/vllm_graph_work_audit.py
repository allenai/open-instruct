"""Opt-in observations of busy work through one audited native vLLM image.

Scheduler tokens are scheduled model work, not generated or delivered responses.
Replay observations mean the native replay call returned, not device completion.
"""

import atexit
import functools
import hashlib
import importlib
import inspect
import json
import os
import socket
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCE_HASHES = {
    "vllm.compilation.cuda_graph": "0f98ae0ea90424eb697e9ea0bdc0bcbbbecdc8637f4a89bfbfcc2ba28b40476d",
    "vllm.v1.worker.gpu_model_runner": "3afc290d3df1be3df1b89b9b35942695f5896c7729f608d6c44580899212c301",
}
_INSTALLED = None


def verify_sources(modules):
    for name, digest in SOURCE_HASHES.items():
        path = Path(modules[name].__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unsupported vLLM graph audit source: {name}")


def scheduled_work(output):
    counts = output.num_scheduled_tokens
    total = output.total_num_scheduled_tokens
    if not isinstance(counts, dict) or type(total) is not int or total < 0:
        raise ValueError("Invalid scheduler work counters")
    if any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) != total:
        raise ValueError("Scheduler token counters disagree")
    return {"scheduled_tokens": total, "scheduled_requests": sum(n > 0 for n in counts.values())}


def graph_branch(wrapper, module):
    if not module.is_forward_context_available():
        return "no-context", "unknown", str(wrapper.runtime_mode)
    context = module.get_forward_context()
    mode = context.cudagraph_runtime_mode
    if mode == module.CUDAGraphMode.NONE:
        branch = "eager"
    elif mode != wrapper.runtime_mode:
        branch = "mode-mismatch"
    else:
        entry = wrapper.concrete_cudagraph_entries.get(context.batch_descriptor)
        branch = "capture" if entry is None or entry.cudagraph is None else "replay-returned"
    return branch, str(mode), str(wrapper.runtime_mode)


class Journal:
    """Append and fsync every 32 busy executions, on errors and normal exit.

    A hard process kill can lose at most 31 buffered executions. No per-wrapper
    writes or additional CUDA synchronization are introduced.
    """

    def __init__(self, directory, flush_every=32):
        directory = Path(directory)
        if not directory.is_absolute() or type(flush_every) is not int or flush_every < 1:
            raise ValueError("Graph journal requires absolute output and positive flush interval")
        directory.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        self.path = directory / f"graph-work-{host}-{os.getpid()}.jsonl"
        self.flush_every = flush_every
        self.pending = []
        self.lock = threading.Lock()
        self.record_count = 0
        self.busy_execution_count = 0

    def _append(self, payload):
        self.record_count += 1
        if "scheduled_tokens" in payload:
            self.busy_execution_count += 1
        line = (
            json.dumps(
                payload
                | {
                    "schema_version": 1,
                    "observed_utc": datetime.now(timezone.utc).isoformat(),
                    "process_id": os.getpid(),
                    "hostname": socket.gethostname(),
                    "beaker_job_id": os.environ.get("BEAKER_JOB_ID"),
                    "journal_record": self.record_count,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        self.pending.append(line)

    def append(self, payload):
        with self.lock:
            self._append(payload)
            if len(self.pending) >= self.flush_every or payload.get("status") == "error":
                self._flush()

    def _flush(self):
        if self.pending:
            with self.path.open("a") as stream:
                stream.writelines(self.pending)
                stream.flush()
                os.fsync(stream.fileno())
            self.pending.clear()

    def flush(self):
        with self.lock:
            self._flush()

    def boundary(self, name):
        """Persist a numbered cutoff; future in-flight work remains outside it."""
        if name != "post-final-model-save":
            raise ValueError("Unsupported graph journal boundary")
        with self.lock:
            report = {
                "status": "boundary-flushed",
                "boundary": name,
                "busy_execution_count": self.busy_execution_count,
                "prior_journal_records": self.record_count,
                "path": str(self.path),
                "scope": "Completed host execute_model calls through this cutoff; later/in-flight work not covered.",
            }
            self._append(report)
            self._flush()
            return report


def instrument(runner_class, graph_class, graph_module, journal):
    """Call audited native methods unchanged; retain wrapper branch observations."""
    local = threading.local()
    original_execute = runner_class.execute_model
    original_call = graph_class.__call__

    @functools.wraps(original_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        work = scheduled_work(scheduler_output)
        previous = getattr(local, "busy", None)
        busy = None
        if work["scheduled_tokens"] and work["scheduled_requests"]:
            busy = work | {"wrapper_calls": Counter(), "excluded_dummy_profile_calls": 0}
        local.busy = busy
        status = "returned"
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        except BaseException:
            status = "error"
            raise
        finally:
            local.busy = previous
            if busy is not None:
                calls = [
                    {"branch": key[0], "context_mode": key[1], "wrapper_mode": key[2], "calls": value}
                    for key, value in sorted(busy.pop("wrapper_calls").items())
                ]
                journal.append(busy | {"status": status, "wrapper_calls": calls})

    @functools.wraps(original_call)
    def call(self, *args, **kwargs):
        busy = getattr(local, "busy", None)
        branch = graph_branch(self, graph_module) if busy is not None else None
        result = original_call(self, *args, **kwargs)
        if busy is not None:
            busy["wrapper_calls"][branch] += 1
        return result

    runner_class.execute_model = execute
    graph_class.__call__ = call
    for name in ("_dummy_run", "profile_run"):
        original = getattr(runner_class, name)

        def suppress(method):
            @functools.wraps(method)
            def wrapped(self, *args, **kwargs):
                previous = getattr(local, "busy", None)
                if previous is not None:
                    previous["excluded_dummy_profile_calls"] += 1
                local.busy = None
                try:
                    return method(self, *args, **kwargs)
                finally:
                    local.busy = previous

            return wrapped

        setattr(runner_class, name, suppress(original))


def install(*, tensor_parallel_size, multiprocessing, executor_backend="uni"):
    """This integration is supported only for current in-process TP1 engines."""
    global _INSTALLED
    if os.environ.get("OI_VLLM_GRAPH_WORK_AUDIT", "0") != "1":
        return None
    if tensor_parallel_size != 1 or multiprocessing != "0" or executor_backend != "uni":
        raise ValueError("Graph audit requires in-process uni TP1 and VLLM_ENABLE_V1_MULTIPROCESSING=0")
    directory = os.environ.get("OI_VLLM_GRAPH_WORK_AUDIT_DIR")
    if not directory:
        raise ValueError("Graph audit output directory must be explicit")
    if _INSTALLED is not None:
        if _INSTALLED.path.parent != Path(directory):
            raise ValueError("Cannot change installed graph journal output")
        return _INSTALLED
    modules = {name: importlib.import_module(name) for name in SOURCE_HASHES}
    verify_sources(modules)
    runner = modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner
    graph_module = modules["vllm.compilation.cuda_graph"]
    for cls, name in (
        (runner, "execute_model"),
        (runner, "_dummy_run"),
        (runner, "profile_run"),
        (graph_module.CUDAGraphWrapper, "__call__"),
    ):
        # execute_model has the native torch.inference_mode decorator.
        function = inspect.unwrap(getattr(cls, name))
        if Path(inspect.getsourcefile(function)).resolve() != Path(modules[cls.__module__].__file__).resolve():
            raise ValueError("Native graph audit method already replaced")
    journal = Journal(directory)
    instrument(runner, graph_module.CUDAGraphWrapper, graph_module, journal)
    journal.append({"status": "installed", "source_hashes": SOURCE_HASHES, "buffer_limit_executions": 31})
    journal.flush()
    atexit.register(journal.flush)
    _INSTALLED = journal
    return journal


async def flush_boundary():
    """Run on the engine loop between model executions; no CUDA synchronization."""
    if os.environ.get("OI_VLLM_GRAPH_WORK_AUDIT", "0") != "1":
        return {"status": "disabled"}
    if _INSTALLED is None:
        raise RuntimeError("Graph observer was not installed in this engine process")
    return _INSTALLED.boundary("post-final-model-save")
