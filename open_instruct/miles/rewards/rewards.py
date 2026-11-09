"""Bridge MILES rollout samples to Open Instruct's registered verifier classes.
Each sample's named targets and weights determine a combined reward, while the
trusted registry controls which verifier implementations may be constructed. The
bridge attaches diagnostics and isolates supported math verifiers in worker
processes so a hung grading call can time out without blocking generation forever.

``core.reward_config`` points to a trusted JSON mapping of verifier names to
``{"factory": "package.Class", "config": {...}}``. Samples carry only names,
weights, and targets in ``metadata.verifiers``; data cannot choose code to import.
"""

import argparse
import asyncio
import atexit
import contextlib
import copy
import dataclasses
import functools
import importlib
import json
import math
import os
import signal
import sys
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

from open_instruct import logger_utils
from open_instruct.miles.rewards import general_judge, judge_registry

logger = logger_utils.setup_logger(__name__)

_MATH_FACTORY = "open_instruct.miles.rewards.math_verifier.MathVerifier"
_MATH_FACTORIES = {
    _MATH_FACTORY,
    "open_instruct.ground_truth_utils.MathVerifier",
    "open_instruct.ground_truth_utils.StrictMathVerifier",
}
_POOLS = weakref.WeakKeyDictionary()
_CHILDREN = set()
_FAILURE_CAPTURED = set()
_RESPONSE_PREFIX = b"MILES_VERIFIER_RESULT "


def _kill_remaining_children():
    for pid in tuple(_CHILDREN):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


atexit.register(_kill_remaining_children)


class MathVerifierTimeout(TimeoutError):
    """A symbolic grading request exceeded its bounded subprocess budget."""


class _MathProcessPool:
    """Persistent, bounded main-thread verifiers, never a fork of a Ray/CUDA worker.

    Consumers are asyncio tasks: event-loop shutdown cancels them and reaps their
    subprocesses. Cancelling a request or timing out kills its process, so a bad
    symbolic expression cannot leave a slot permanently occupied.
    """

    def __init__(self, workers=4, timeout=45.0):
        self.timeout = timeout
        self.queue = asyncio.Queue(maxsize=workers * 2)
        self.tasks = [asyncio.create_task(self._consume()) for _ in range(workers)]

    async def score(self, request):
        # Validate serializability before enqueueing; this is JSON IPC, not pickle.
        payload = json.dumps({"schema_version": 1, **request}, allow_nan=False).encode() + b"\n"
        future = asyncio.get_running_loop().create_future()
        try:
            await self.queue.put((payload, future))
            return await future
        finally:
            if not future.done():
                future.cancel()

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            future.cancel()
            self.queue.task_done()

    @staticmethod
    async def _start():
        environment = dict(os.environ)
        isolated = environment.get("OPEN_INSTRUCT_MATH_VERIFIER_PYTHONPATH")
        if isolated:
            environment["PYTHONPATH"] = isolated + os.pathsep + environment.get("PYTHONPATH", "")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "open_instruct.miles.rewards.rewards",
            "--math-worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )
        _CHILDREN.add(process.pid)
        return process

    @staticmethod
    async def _stop(process):
        if process is not None:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.wait()
            _CHILDREN.discard(process.pid)

    @staticmethod
    async def _exchange(process, payload):
        process.stdin.write(payload)
        await process.stdin.drain()
        while True:
            line = await process.stdout.readline()
            if not line:
                raise RuntimeError("Math verifier process exited without a result")
            if line.startswith(_RESPONSE_PREFIX):
                response = json.loads(line[len(_RESPONSE_PREFIX) :])
                break
        if response.get("schema_version") != 1:
            raise RuntimeError("Unsupported math verifier response schema")
        if "error" in response:
            raise RuntimeError("Math verifier failed: " + response["error"])
        return SimpleNamespace(**response["result"])

    async def _consume(self):
        process = None
        future = None
        exchange = None
        try:
            while True:
                payload, future = await self.queue.get()
                try:
                    if future.cancelled():
                        continue
                    if process is None:
                        process = await self._start()
                    exchange = asyncio.create_task(self._exchange(process, payload))
                    done, _ = await asyncio.wait(
                        (exchange, future), timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED
                    )
                    if exchange in done:
                        result = exchange.result()
                        if not future.done():
                            future.set_result(result)
                    else:
                        if not future.done():
                            future.set_exception(MathVerifierTimeout(f"Math verifier exceeded {self.timeout} seconds"))
                        exchange.cancel()
                        await asyncio.gather(exchange, return_exceptions=True)
                        await self._stop(process)
                        process = None
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                    await self._stop(process)
                    process = None
                finally:
                    if exchange is not None:
                        if not exchange.done():
                            exchange.cancel()
                        await asyncio.gather(exchange, return_exceptions=True)
                    if future is not None and not future.done():
                        future.cancel()
                    self.queue.task_done()
                    exchange = None
                    future = None
        finally:
            if exchange is not None:
                exchange.cancel()
                await asyncio.gather(exchange, return_exceptions=True)
            if future is not None and not future.done():
                future.cancel()
            await self._stop(process)


async def isolated_verifier_call(
    factory_spec, tokenized_prediction, prediction, label, query=None, rollout_state=None
):
    """Call an explicitly trusted verifier on a subprocess main thread.

    Symbolic math runs with main-thread SIGALRM timeouts. ANTLR 4.11 can live
    in a child-only dependency directory, preserving the rollout process's OmegaConf/ANTLR 4.9 installation.
    """
    loop = asyncio.get_running_loop()
    if loop not in _POOLS:
        pool = _MathProcessPool()
        _POOLS[loop] = pool
        loop_ref = weakref.ref(loop)

        def retire(_):
            current = loop_ref()
            if current is not None and all(task.done() for task in pool.tasks):
                _POOLS.pop(current, None)

        for task in pool.tasks:
            task.add_done_callback(retire)
    return await _POOLS[loop].score(
        dict(
            factory_spec=factory_spec,
            tokenized_prediction=tokenized_prediction,
            prediction=prediction,
            label=label,
            query=query,
            rollout_state=rollout_state,
        )
    )


class _IsolatedVerifier:
    def __init__(self, spec):
        self.spec = spec

    async def async_call(self, *args, **kwargs):
        start = time.monotonic()
        try:
            result = await isolated_verifier_call(self.spec, *args, **kwargs)
        except MathVerifierTimeout as error:
            logger.warning("Math verifier timeout; assigning zero reward: %s", error)
            return SimpleNamespace(
                score=0.0,
                cost=0.0,
                diagnostics={
                    "kind": "math",
                    "status": "timeout",
                    "error": str(error),
                    "failure_policy": "zero",
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
        result.diagnostics = {
            **(getattr(result, "diagnostics", None) or {}),
            "kind": "math",
            "status": "ok",
            "elapsed_seconds": time.monotonic() - start,
        }
        if symbolic_timeouts := result.diagnostics.get("symbolic_timeouts", 0):
            result.diagnostics["symbolic_timeout_rewarded"] = result.score > 0
            # Worker stderr is suppressed; surface the completed result here.
            logger.warning(
                "Math verifier completed with %d symbolic timeout(s); final score=%s", symbolic_timeouts, result.score
            )
        return result


def _resolve_verifier(spec: dict, *, isolate_symbolic=False):
    # Existing prepared MILES registries still name the shared legacy class.
    # Resolve that exact path to the MILES default without changing legacy callers
    # or mutating the stored registry. Do this in parent and subprocess alike.
    if spec["factory"] == "open_instruct.ground_truth_utils.MathVerifier":
        spec = {**spec, "factory": _MATH_FACTORY}
    module, _, symbol = spec["factory"].rpartition(".")
    factory = getattr(importlib.import_module(module), symbol)
    config = factory.get_config_class()(**spec.get("config", {}))
    if isolate_symbolic and any(f"{base.__module__}.{base.__name__}" in _MATH_FACTORIES for base in factory.__mro__):
        return _IsolatedVerifier(spec)
    return factory(verifier_config=config, **spec.get("kwargs", {}))


@functools.lru_cache(maxsize=32)
def _instantiate(spec_json):
    # Only the subprocess worker caches instances by specification. Registry names
    # retain separate instances even when two names use the same factory/config.
    return _resolve_verifier(json.loads(spec_json))


def _math_worker():
    # Libraries may print during import or grading; reserve stdout for JSON IPC.
    output = sys.stdout
    sys.stdout = sys.stderr
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.pop("schema_version") != 1:
                raise ValueError("Unsupported math verifier request schema")
            verifier = _instantiate(json.dumps(request.pop("factory_spec"), sort_keys=True))
            result = verifier(**request)
            response = {"schema_version": 1, "result": dataclasses.asdict(result)}
        except Exception as exc:
            response = {"schema_version": 1, "error": f"{type(exc).__name__}: {exc}"}
        output.write(_RESPONSE_PREFIX.decode() + json.dumps(response, allow_nan=False) + "\n")
        output.flush()


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


@functools.lru_cache(maxsize=8)
def _registry(path):
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or not document:
        raise ValueError("reward_config must map verifier names to factory/config objects")
    return {name: _resolve_verifier(spec, isolate_symbolic=True) for name, spec in document.items()}


@functools.lru_cache(maxsize=8)
def _masks_ungraded_path(path):
    module, _, symbol = path.rpartition(".")
    return bool(getattr(getattr(importlib.import_module(module), symbol), "masks_ungraded", False))


def masks_ungraded(args) -> bool:
    """Whether the configured reward post-process excludes ungraded responses itself.

    ``open_instruct.miles.rewards.exclusion`` hooks declare ``masks_ungraded``. For
    them an ungraded training response keeps reward 0 plus its ``reward_omission``
    marker, and the post-process masks it. Without such a hook the response has no
    reward and the pinned MILES collector drops its whole prompt group.
    """
    if getattr(args, "custom_convert_samples_to_train_data_path", None):
        return False  # Native conversion bypasses the reward post-process.
    path = getattr(args, "custom_reward_post_process_path", None)
    return bool(path) and _masks_ungraded_path(path)


def ungraded_count(samples) -> int:
    """Count consumed responses whose reward is a placeholder for a failed grade."""
    return sum(
        1
        for sample in samples
        if isinstance(getattr(sample, "metadata", None), dict) and "reward_omission" in sample.metadata
    )


def _capture_verifier_failure(args, sample, spec, response, error, *, capture_kind="exception"):
    """Capture one raised exception and one returned timeout per run/process.

    Normal rollout dumps contain already-scored batches and cannot diagnose a
    verifier that raises. Preserve the full text/target here without converting
    that error to a model reward or masking it if storage is unavailable. A returned
    timeout uses a separate slot so it cannot hide a later raised exception.
    """
    save = getattr(args, "save", None)
    key = (save, capture_kind)
    if not save or key in _FAILURE_CAPTURED:
        return
    # Bound capture attempts even if storage is failing during a burst of errors.
    _FAILURE_CAPTURED.add(key)
    try:
        root = Path(save) / "verifier_failures"
        root.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "capture_kind": capture_kind,
            "error": f"{type(error).__name__}: {error}",
            "verifier": spec,
            "prompt": sample.prompt,
            "response": sample.response,
            "scored_response": response,
            "group_index": getattr(sample, "group_index", None),
            "sample_index": getattr(sample, "index", None),
        }
        path = root / f"{os.getpid()}-{time.time_ns()}.json"
        with path.open("x") as stream:
            json.dump(record, stream, ensure_ascii=False, default=str)
        logger.error("Saved failed verifier input to %s", path)
    except Exception:
        logger.exception("Could not save failed verifier input; preserving original verifier exception")


async def _score(args, sample, *, omit_failures=False, failure_policy="raise"):
    registry = _registry(args.olmo_core.reward_config)
    metadata = sample.metadata
    if not isinstance(metadata, dict) or not isinstance(metadata.get("verifiers"), list) or not metadata["verifiers"]:
        raise ValueError("Each sample requires nonempty metadata.verifiers")
    metadata.pop("reward_omission", None)
    weighted_specs = []
    for spec in metadata["verifiers"]:
        if spec["name"] not in registry:
            raise ValueError(f"Verifier {spec['name']!r} is absent from the trusted reward registry")
        weighted_specs.append((spec, _finite(spec.get("weight", 1.0), "weight")))
    status = getattr(sample.status, "value", sample.status) if hasattr(sample, "status") else None
    reason = None
    response = sample.response
    if args.olmo_core.reward_zero_truncated and status == "truncated":
        reason = "truncated"
    elif args.olmo_core.reward_final_answer_only:
        _, separator, response = response.rpartition("</think>")
        if not separator or not response.strip():
            reason = "missing_final_answer"
    if reason:
        metadata["reward_gate"] = reason
        metadata["reward_components"] = [
            {"name": spec["name"], "score": 0.0, "weight": spec.get("weight", 1.0), "cost": 0.0}
            for spec in metadata["verifiers"]
        ]
        return 0.0
    components = []
    total = 0.0
    for spec, weight in weighted_specs:
        name = spec["name"]
        # response_length includes tool observations. Their tokens remain in the
        # trajectory, while the policy loss uses the separate MILES loss mask.
        tokens = sample.tokens[-sample.response_length :] if sample.response_length else []
        if args.olmo_core.reward_final_answer_only:
            # Text math verifiers do not use token IDs. Never pass the hidden
            # reasoning's tokens alongside an extracted final-answer string.
            tokens = []
        try:
            if judge_registry.bound(name):
                score = await general_judge.general_judge_score(
                    args, sample, name=name, target=copy.deepcopy(spec["target"]), prediction=response
                )
                diagnostics = metadata.get("verifier_diagnostics", {}).get(name, {})
                cost = 0.0
            else:
                # Verifiers may opt into final-answer handling; existing math/code
                # verifiers continue to grade the suffix through their normal method.
                verifier = registry[name]
                score_response = verifier.async_call
                if args.olmo_core.reward_final_answer_only:
                    score_response = getattr(verifier, "async_call_final_answer", score_response)
                result = await score_response(
                    tokens,
                    response,
                    # Verifiers may consume dictionary labels; preserve the original targets.
                    copy.deepcopy(spec["target"]),
                    query=metadata.get("query", sample.prompt),
                    rollout_state=metadata.get("rollout_state"),
                )
                diagnostics = getattr(result, "diagnostics", None)
                if isinstance(diagnostics, dict) and diagnostics.get("status") == "timeout":
                    error = TimeoutError(diagnostics.get("error") or "Verifier timed out without an error message")
                    _capture_verifier_failure(args, sample, spec, response, error, capture_kind="timeout")
                score, cost = result.score, result.cost
            if isinstance(diagnostics, dict) and diagnostics.get("status") in {"judge_error", "service_error"}:
                diagnostics = {**diagnostics, "scoring_failure_policy": failure_policy}
                metadata.setdefault("verifier_diagnostics", {})[name] = diagnostics
                if omit_failures or failure_policy == "raise":
                    raise RuntimeError("Verifier failed to grade the response; see verifier diagnostics")
                logger.warning(
                    "Verifier %s failed to grade the response; caller requested zero: %s", name, diagnostics
                )
                score = 0.0
            score = _finite(score, "verifier score")
            total = _finite(total + weight * score, "combined reward")
        except Exception as error:
            _capture_verifier_failure(args, sample, spec, response, error)
            if not omit_failures:
                raise
            metadata["reward_omission"] = {"verifier": name, "error": f"{type(error).__name__}: {error}"}
            metadata.pop("reward_components", None)
            if masks_ungraded(args):
                # The configured post-process gives this response zero advantage and
                # a zeroed loss mask and leaves it out of its group's baseline, so
                # the stored zero never acts as a grade. Its graded siblings train.
                logger.warning(
                    "Excluding ungraded response from prompt group %s after verifier %s failed: %s",
                    sample.group_index,
                    name,
                    metadata["reward_omission"]["error"],
                )
                return 0.0
            logger.warning(
                "Omitting prompt group %s after verifier %s failed: %s",
                sample.group_index,
                name,
                metadata["reward_omission"]["error"],
            )
            # Missing rewards are rejected before variance filtering and training
            # by the pinned MILES synchronous and async collection paths.
            return None
        components.append({"name": name, "score": score, "weight": weight, "cost": cost})
        # Verifiers without their own diagnostics still returned normally; record that
        # adapter-level fact so a missing status can mean only "unknown".
        if not isinstance(diagnostics, dict):
            diagnostics = {"kind": "adapter", "status": "completed"}
        metadata.setdefault("verifier_diagnostics", {})[name] = diagnostics
    total = _finite(total, "combined reward")
    metadata["reward_components"] = components
    return total


async def registered_reward(args, samples, *, failure_policy="raise", **kwargs):
    """Score with explicit handling of reported service/judge failures.

    The zero policy zeroes the failed verifier component and preserves diagnostics.
    Exceptions (including invalid targets/configuration) still propagate.
    """
    if failure_policy not in {"raise", "zero"}:
        raise ValueError("Reward failure_policy must be 'raise' or 'zero'")
    if not args.olmo_core.reward_config:
        raise ValueError("Set core.reward_config for the open-instruct verifier adapter")
    if isinstance(samples, list):
        return list(await asyncio.gather(*(_score(args, sample, failure_policy=failure_policy) for sample in samples)))
    return await _score(args, samples, failure_policy=failure_policy)


async def rollout_reward(args, samples, **kwargs):
    """Exclude ungraded training responses, or omit their groups without a masking hook.

    Reported failures receive zero in evaluation.

    MILES assigns group_index when drawing training prompts. Evaluation samples
    have no group_index and are not passed through the replacement/filter loop.
    Evaluation explicitly requests zero for reported service/judge failures.
    Registry, routing and unexpected evaluation exceptions remain fatal.
    """
    if not args.olmo_core.reward_config:
        raise ValueError("Set core.reward_config for the open-instruct verifier adapter")

    async def score(sample):
        training = getattr(sample, "group_index", None) is not None
        return await _score(args, sample, omit_failures=training, failure_policy="raise" if training else "zero")

    if isinstance(samples, list):
        return list(await asyncio.gather(*(score(sample) for sample in samples)))
    return await score(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated open-instruct math verifier worker")
    parser.add_argument("--math-worker", action="store_true", required=True)
    parser.parse_args()
    _math_worker()
