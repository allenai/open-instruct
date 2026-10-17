"""Live H087 request accounting, including grades discarded by group filtering.

One CPU Ray actor owns the counters; admission stays local to each caller process.
Sample diagnostics alone cannot measure arrivals or backlog at optimizer boundaries.
"""

import asyncio
import collections
import importlib
import time

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

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
        self.arrived = self.completed = self.failed = self.skipped = self.attempts = 0
        self.queued = self.in_flight = 0
        self.latency_sum = self.latency_max = self.service_sum = 0.0
        self.started_unix = time.time()

    def arrive(self):
        self.arrived += 1
        self.queued += 1

    def admit(self):
        self.queued -= 1
        self.in_flight += 1

    def finish(self, success, attempts, elapsed, service=None):
        if service is None:  # cancelled before local admission
            self.queued -= 1
        else:
            self.in_flight -= 1
            self.service_sum += service
        self.completed += 1  # all terminal requests; failed is a subset
        self.failed += int(not success)
        self.attempts += attempts
        self.latency_sum += elapsed
        self.latency_max = max(self.latency_max, elapsed)

    def skip(self):
        self.skipped += 1

    async def snapshot(self):
        return {
            "arrived": self.arrived,
            "completed": self.completed,
            "failed": self.failed,
            "queued": self.queued,
            "in_flight": self.in_flight,
            "backlog": self.queued + self.in_flight,
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

    def _send(self, method, *args):
        # Never await counters: cancellation or a slow/dead actor must not hold
        # an admission slot or stall reward scoring.
        try:
            getattr(self.actor, method).remote(*args)
        except Exception:
            logger.warning("Could not send judge metrics event %s", method, exc_info=True)

    def arrive(self):
        self._send("arrive")

    def admit(self):
        self._send("admit")

    def finish(self, success, attempts, elapsed, service=None):
        self._send("finish", success, attempts, elapsed, service)

    def skip(self):
        self._send("skip")

    async def snapshot(self):
        return await self.actor.snapshot.remote()


def start():
    """Driver calls once before any rollout workers start; no GPU resource claim."""
    ray = importlib.import_module("ray")
    return ray.remote(JudgeMetrics).options(name=ACTOR, namespace=NAMESPACE, num_cpus=0, max_concurrency=1).remote()


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
