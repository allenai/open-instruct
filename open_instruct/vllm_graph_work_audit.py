"""Opt-in observations of busy work through one audited native vLLM image.

Scheduler tokens are scheduled model work, not generated or delivered responses.
Replay observations mean the native replay call returned, not device completion.

AsyncLLM always runs EngineCore, and with the uni executor its only worker, in a
spawned child of the actor regardless of VLLM_ENABLE_V1_MULTIPROCESSING. The observer
is therefore activated and cut off inside that child through a vLLM worker extension
and collective_rpc; the actor only validates the child's acknowledgments.
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
WORKER_EXTENSION = "open_instruct.vllm_graph_work_audit.GraphWorkAuditWorkerExtension"
ACTIVATE_RPC = "oi_graph_audit_activate"
CUTOFF_RPC = "oi_graph_audit_cutoff"
_INSTALLED = None
_ACTIVATION = None


def enabled():
    return os.environ.get("OI_VLLM_GRAPH_WORK_AUDIT", "0") == "1"


def output_directory():
    directory = os.environ.get("OI_VLLM_GRAPH_WORK_AUDIT_DIR")
    if not directory:
        raise ValueError("Graph audit output directory must be explicit")
    return directory


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


def configure_engine_kwargs(kwargs):
    """In the actor, before engine construction: require uni TP1 and register the worker extension."""
    if not enabled():
        return False
    if kwargs.get("tensor_parallel_size", 1) != 1 or kwargs.get("distributed_executor_backend") != "uni":
        raise ValueError("Graph audit requires a uni-executor TP1 engine")
    output_directory()
    if kwargs.get("worker_extension_cls") not in (None, "", WORKER_EXTENSION):
        raise ValueError("Graph audit cannot replace an existing vLLM worker extension")
    kwargs["worker_extension_cls"] = WORKER_EXTENSION
    return True


def activate(worker, actor_process_id):
    """In the EngineCore worker process: verify the live V1 runner and sources, then instrument."""
    global _INSTALLED, _ACTIVATION
    if not enabled():
        raise RuntimeError("Graph audit flag did not reach the vLLM worker process")
    if type(actor_process_id) is not int or os.getpid() == actor_process_id:
        raise RuntimeError("Graph observer must activate in the spawned worker, not the actor process")
    if os.getppid() != actor_process_id:
        raise RuntimeError("Graph observer worker is not a direct child of the requesting actor")
    directory = output_directory()
    if _ACTIVATION is not None:
        if _ACTIVATION["actor_process_id"] != actor_process_id:
            raise RuntimeError("Graph observer already activated for another actor")
        return _ACTIVATION
    parallel = worker.vllm_config.parallel_config
    if (parallel.world_size, parallel.tensor_parallel_size, parallel.distributed_executor_backend) != (1, 1, "uni"):
        raise ValueError("Graph audit requires a uni-executor TP1 worker")
    modules = {name: importlib.import_module(name) for name in SOURCE_HASHES}
    verify_sources(modules)
    runner = modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner
    if importlib.import_module("vllm.envs").VLLM_USE_V2_MODEL_RUNNER or type(worker.model_runner) is not runner:
        raise ValueError("Graph audit supports only the live V1 GPUModelRunner")
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
    activation = {
        "status": "activated",
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "actor_process_id": actor_process_id,
        "hostname": socket.gethostname(),
        "runner_class": f"{runner.__module__}.{runner.__qualname__}",
        "source_hashes": SOURCE_HASHES,
        "journal_path": str(journal.path),
    }
    journal.append(activation | {"status": "installed", "buffer_limit_executions": 31})
    journal.flush()
    atexit.register(journal.flush)
    _INSTALLED, _ACTIVATION = journal, activation
    return activation


def cutoff(actor_process_id):
    """In the activated worker, between model executions; no CUDA synchronization."""
    if _INSTALLED is None or _ACTIVATION is None:
        raise RuntimeError("Graph observer was not activated in this worker process")
    if os.getpid() != _ACTIVATION["process_id"] or actor_process_id != _ACTIVATION["actor_process_id"]:
        raise RuntimeError("Graph cutoff requested from a different actor or process")
    return _INSTALLED.boundary("post-final-model-save") | {
        "process_id": os.getpid(),
        "actor_process_id": actor_process_id,
    }


class GraphWorkAuditWorkerExtension:
    """Mixed into vLLM's worker class; collective_rpc runs these in the EngineCore process."""

    def oi_graph_audit_activate(self, actor_process_id):
        return activate(self, actor_process_id)

    def oi_graph_audit_cutoff(self, actor_process_id):
        return cutoff(actor_process_id)


def _single(results, kind):
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise RuntimeError(f"Expected one graph audit {kind} acknowledgment from the TP1 worker")
    return results[0]


def validate_activation(results, actor_process_id):
    ack = _single(results, "activation")
    process_id = ack.get("process_id")
    if (
        ack.get("status") != "activated"
        or type(process_id) is not int
        or process_id == actor_process_id
        or ack.get("parent_process_id") != actor_process_id
        or ack.get("actor_process_id") != actor_process_id
    ):
        raise RuntimeError("Graph observer activation was not acknowledged by a spawned worker of this actor")
    return ack


def validate_cutoff(results, activation):
    report = _single(results, "cutoff")
    if report.get("status") != "boundary-flushed" or report.get("process_id") != activation["process_id"]:
        raise RuntimeError("Graph cutoff did not come from the activated worker process")
    return report


async def activate_engine(engine_client):
    """In the actor, after engine construction and before serving requests."""
    actor_process_id = os.getpid()
    results = await engine_client.collective_rpc(ACTIVATE_RPC, args=(actor_process_id,))
    return validate_activation(results, actor_process_id)


async def cutoff_engine(engine_client, activation):
    """In the actor; the worker records the cutoff between its model executions."""
    if not enabled():
        return {"status": "disabled"}
    if activation is None:
        raise RuntimeError("Graph observer was not activated for this engine")
    results = await engine_client.collective_rpc(CUTOFF_RPC, args=(activation["actor_process_id"],))
    return validate_cutoff(results, activation)
