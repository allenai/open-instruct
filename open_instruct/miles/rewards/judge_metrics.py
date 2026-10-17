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
FAILURE_WINDOW_UPDATES = 50
FAILURE_MIN_REQUESTS = 1000


def check_failure_guard(counters):
    """Called by the driver after logging an awaited optimizer-boundary snapshot."""
    requests = counters["judge_requests_window"]
    failures = counters["terminal_failures_window"]
    updates = counters["judge_window_updates"]
    if requests < FAILURE_MIN_REQUESTS:
        return
    # Compare integers so exactly one failure per 200 requests never trips.
    if failures * 200 > requests:
        raise RuntimeError(
            f"H087 zero_guarded judge failure guard: {failures}/{requests} terminal failures "
            f"({counters['judge/terminal_failure_rate_window']:.3%}) over {updates} consecutive optimizer updates "
            "exceeds 0.5%; stopping training"
        )


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
        self.judge_requests = self.terminal_failures = 0
        self._last_requests = self._last_failures = 0
        self._updates = collections.deque(maxlen=FAILURE_WINDOW_UPDATES)
        self.requests_by_family = collections.Counter()
        self.failures_by_family = collections.Counter()
        self.requests_by_instruction = collections.Counter()
        self.failures_by_instruction = collections.Counter()

    def arrive(self):
        self.arrived += 1
        self.queued += 1

    def admit(self):
        self.queued -= 1
        self.in_flight += 1

    def finish(self, success, attempts, elapsed, service=None, terminal_failure=None, instruction_ids=()):
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
        # Cancelled requests and fatal context/programming errors retain their
        # existing backlog accounting but cannot dilute the guarded denominator.
        if terminal_failure is not None:
            self.judge_requests += 1
            self.terminal_failures += int(terminal_failure)
            # One logical request counts once for each distinct instruction and
            # family it carries, regardless of retries or repeated constraints.
            instructions = set(instruction_ids)
            families = {instruction.partition(":")[0] for instruction in instructions}
            self.requests_by_family.update(families)
            self.requests_by_instruction.update(instructions)
            if terminal_failure:
                self.failures_by_family.update(families)
                self.failures_by_instruction.update(instructions)

    def skip(self):
        self.skipped += 1

    async def optimizer_boundary(self):
        """Close one update, including requests from groups later filtered out."""
        self._updates.append((self.judge_requests - self._last_requests, self.terminal_failures - self._last_failures))
        self._last_requests, self._last_failures = self.judge_requests, self.terminal_failures
        return await self.snapshot()

    async def snapshot(self):
        requests = sum(update[0] for update in self._updates)
        failures = sum(update[1] for update in self._updates)
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
            "judge_requests": self.judge_requests,
            "judge_requests_window": requests,
            "terminal_failures_window": failures,
            "judge_window_updates": len(self._updates),
            "judge/terminal_failures": self.terminal_failures,
            "judge/terminal_failure_rate_window": failures / requests if requests else 0.0,
            **{
                f"judge/{kind}_by_{scope}/{key}": counts[key]
                for scope, requests_by_key, failures_by_key in (
                    ("family", self.requests_by_family, self.failures_by_family),
                    ("instruction", self.requests_by_instruction, self.failures_by_instruction),
                )
                for key in sorted(requests_by_key)
                for kind, counts in (("requests", requests_by_key), ("terminal_failures", failures_by_key))
            },
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

    def finish(self, success, attempts, elapsed, service=None, terminal_failure=None, instruction_ids=()):
        self._send("finish", success, attempts, elapsed, service, terminal_failure, instruction_ids)

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
