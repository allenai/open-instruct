"""Live H087 request accounting, including grades discarded by group filtering.

One CPU Ray actor owns the counters and the global 32-call client admission limit.
Sample diagnostics alone cannot measure arrivals or backlog at optimizer boundaries.
"""

import asyncio
import collections
import importlib
import time
import uuid

ACTOR = "open-instruct-if-gate-yes-metrics"
NAMESPACE = "open-instruct-rewards"
_LOCAL = None


def consumed_metrics(samples):
    records = [
        record
        for sample in samples
        for record in ((getattr(sample, "metadata", None) or {}).get("verifier_diagnostics") or {}).values()
        if isinstance(record, dict) and record.get("kind") == "general_judge"
    ]
    if not records:
        return {}
    skipped = sum(r.get("status") == "skipped" for r in records)
    graded = [r for r in records if r.get("status") != "skipped"]
    failures = sum(r.get("status") == "judge_error" for r in graded)
    metrics = {
        "rollout/general_judge/samples": len(graded),
        "rollout/general_judge/errors": failures,
        "rollout/general_judge/error_fraction": failures / len(graded) if graded else 0.0,
        "rollout/general_judge/skipped": skipped,
    }
    labels = collections.Counter(r.get("label") for r in graded if r.get("label") is not None)
    metrics.update({f"rollout/general_judge/{label}": n for label, n in labels.items()})
    latencies = [r["elapsed_seconds"] for r in graded if "elapsed_seconds" in r]
    if latencies:
        metrics["rollout/general_judge/latency_seconds_mean"] = sum(latencies) / len(latencies)
        metrics["rollout/general_judge/latency_seconds_max"] = max(latencies)
    return metrics


class JudgeMetrics:
    def __init__(self):
        self.condition = asyncio.Condition()
        self.arrived = self.completed = self.failed = self.skipped = self.attempts = 0
        self.queued = 0
        self.active = {}
        self.latency_sum = self.latency_max = self.service_sum = 0.0
        self.started_unix = time.time()

    async def begin(self, limit=32):
        started = time.monotonic()
        self.arrived += 1
        self.queued += 1
        async with self.condition:
            try:
                await self.condition.wait_for(lambda: len(self.active) < limit)
            except BaseException:
                self.completed += 1
                self.failed += 1
                raise
            finally:
                self.queued -= 1
            ticket = uuid.uuid4().hex
            self.active[ticket] = (started, time.monotonic())
            return ticket

    async def finish(self, ticket, success, attempts):
        async with self.condition:
            started, admitted = self.active.pop(ticket)
            elapsed = time.monotonic() - started
            self.completed += 1  # all terminal requests; failed is a subset
            self.failed += int(not success)
            self.attempts += attempts
            self.latency_sum += elapsed
            self.latency_max = max(self.latency_max, elapsed)
            self.service_sum += time.monotonic() - admitted
            self.condition.notify_all()

    async def skip(self):
        self.skipped += 1

    async def snapshot(self):
        return {
            "arrived": self.arrived,
            "completed": self.completed,
            "failed": self.failed,
            "queued": self.queued,
            "in_flight": len(self.active),
            "backlog": self.queued + len(self.active),
            "skipped": self.skipped,
            "attempts": self.attempts,
            "latency_seconds_sum": self.latency_sum,
            "latency_seconds_max": self.latency_max,
            "latency_seconds_mean": self.latency_sum / self.completed if self.completed else 0.0,
            "service_seconds_sum": self.service_sum,
            "started_unix": self.started_unix,
            "unix": time.time(),
        }


class _Remote:
    def __init__(self, actor):
        self.actor = actor

    async def begin(self, limit=32):
        return await self.actor.begin.remote(limit)

    async def finish(self, ticket, success, attempts):
        return await self.actor.finish.remote(ticket, success, attempts)

    async def skip(self):
        return await self.actor.skip.remote()

    async def snapshot(self):
        return await self.actor.snapshot.remote()


def start():
    """Driver calls once before any rollout workers start; no GPU resource claim."""
    ray = importlib.import_module("ray")
    return (
        ray.remote(JudgeMetrics).options(name=ACTOR, namespace=NAMESPACE, num_cpus=0, max_concurrency=100000).remote()
    )


def tracker():
    global _LOCAL
    # Qualification uses the identical client on one asyncio loop without Ray.
    # Training must find the driver's actor; silently local counters would lie.
    try:
        ray = importlib.import_module("ray")
    except ImportError:
        ray = None
    if ray is not None and ray.is_initialized():
        return _Remote(ray.get_actor(ACTOR, namespace=NAMESPACE))
    loop = asyncio.get_running_loop()
    if _LOCAL is None or _LOCAL[0] is not loop:
        _LOCAL = (loop, JudgeMetrics())
    return _LOCAL[1]
