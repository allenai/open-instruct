"""MILES custom reward hook for open-instruct's existing verifier classes.

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
import weakref
from pathlib import Path
from types import SimpleNamespace

from open_instruct.miles import general_judge, judge_registry

_MATH_FACTORIES = {
    "open_instruct.ground_truth_utils.MathVerifier",
    "open_instruct.ground_truth_utils.StrictMathVerifier",
}
_POOLS = weakref.WeakKeyDictionary()
_CHILDREN = set()
_RESPONSE_PREFIX = b"MILES_VERIFIER_RESULT "


def _kill_remaining_children():
    for pid in tuple(_CHILDREN):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


atexit.register(_kill_remaining_children)


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
            "open_instruct.miles.rewards",
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
                            future.set_exception(TimeoutError(f"Math verifier exceeded {self.timeout} seconds"))
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

    Symbolic math keeps the original open-instruct/ANTLR grading semantics and
    SIGALRM timeout. ANTLR 4.11 can live in a child-only dependency directory,
    preserving the rollout process's OmegaConf/ANTLR 4.9 installation.
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
        return await isolated_verifier_call(self.spec, *args, **kwargs)


@functools.lru_cache(maxsize=32)
def _instantiate(spec_json):
    spec = json.loads(spec_json)
    module, _, symbol = spec["factory"].rpartition(".")
    factory = getattr(importlib.import_module(module), symbol)
    config = factory.get_config_class()(**spec.get("config", {}))
    return factory(verifier_config=config, **spec.get("kwargs", {}))


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
    result = {}
    for name, spec in document.items():
        module, _, symbol = spec["factory"].rpartition(".")
        factory = getattr(importlib.import_module(module), symbol)
        config = factory.get_config_class()(**spec.get("config", {}))
        symbolic = any(f"{base.__module__}.{base.__name__}" in _MATH_FACTORIES for base in factory.__mro__)
        result[name] = (
            _IsolatedVerifier(spec) if symbolic else factory(verifier_config=config, **spec.get("kwargs", {}))
        )
    return result


async def _score(args, sample):
    registry = _registry(args.olmo_core.reward_config)
    metadata = sample.metadata
    if not isinstance(metadata, dict) or not isinstance(metadata.get("verifiers"), list) or not metadata["verifiers"]:
        raise ValueError("Each sample requires nonempty metadata.verifiers")
    components = []
    total = 0.0
    for spec in metadata["verifiers"]:
        name = spec["name"]
        if name not in registry:
            raise ValueError(f"Verifier {name!r} is absent from the trusted reward registry")
        weight = _finite(spec.get("weight", 1.0), "weight")
        # response_length includes tool observations. Their tokens remain in the
        # trajectory, while the policy loss uses the separate MILES loss mask.
        tokens = sample.tokens[-sample.response_length :] if sample.response_length else []
        if judge_registry.bound(name):
            score = await general_judge.general_judge_score(
                args, sample, name=name, target=copy.deepcopy(spec["target"])
            )
            total += weight * score
            components.append({"name": name, "score": score, "weight": weight, "cost": 0.0})
            continue
        result = await registry[name].async_call(
            tokens,
            sample.response,
            # Verifiers may consume dictionary labels; preserve the original targets.
            copy.deepcopy(spec["target"]),
            query=metadata.get("query", sample.prompt),
            rollout_state=metadata.get("rollout_state"),
        )
        score = _finite(result.score, "verifier score")
        total += weight * score
        components.append({"name": name, "score": score, "weight": weight, "cost": result.cost})
    total = _finite(total, "combined reward")
    metadata["reward_components"] = components
    return total


async def registered_reward(args, samples, **kwargs):
    if not args.olmo_core.reward_config:
        raise ValueError("Set core.reward_config for the open-instruct verifier adapter")
    if isinstance(samples, list):
        return list(await asyncio.gather(*(_score(args, sample) for sample in samples)))
    return await _score(args, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated open-instruct math verifier worker")
    parser.add_argument("--math-worker", action="store_true", required=True)
    parser.parse_args()
    _math_worker()
