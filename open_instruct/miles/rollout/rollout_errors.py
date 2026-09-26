"""Recoverable generation failures and one sustained-error-rate stop rule."""

import asyncio
import collections
import json
import time
from pathlib import Path

import httpx
from miles.utils.tracking_utils import tracking

from open_instruct import logger_utils
from open_instruct.miles.errors import GenerationRequestTimeout

logger = logger_utils.setup_logger(__name__)

WINDOW_SECONDS = 300
FAILURE_FRACTION = 0.5
RETRY_DELAY_SECONDS = 1.0
REPORT_INTERVAL_SECONDS = 15
CATEGORIES = ("transport", "overload", "unavailable", "timeout")


def recovery_category(error: BaseException) -> str | None:
    """Only recognize generation failures with a known safe group-retry contract."""
    if isinstance(error, (GenerationRequestTimeout, httpx.TimeoutException)):
        return "timeout"
    if isinstance(error, httpx.TransportError):
        return "transport"
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    if error.request.method != "POST" or error.request.url.path != "/generate":
        return None
    if error.response.status_code == 429:
        return "overload"
    if error.response.status_code != 503:
        return None
    try:
        body = error.response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    # MILES uses detail; SGLang's /generate error envelope uses message.
    message = body.get("detail", body.get("message"))
    if not isinstance(message, str):
        return None
    return {
        "Rollout worker unavailable": "transport",
        "No healthy rollout workers available": "unavailable",
        "The request queue is full.": "overload",
        "The request is aborted by a higher priority request.": "overload",
        "Request waiting timeout reached.": "timeout",
        "Request running timeout reached.": "timeout",
    }.get(message)


class RolloutErrors:
    """Count terminal group attempts, excluding cancellation and reward filtering.

    Stop if the trailing five-minute failure fraction stays >= 50% for five
    minutes. Second buckets bound memory independently of request throughput.
    Empty windows reset the timer; a short burst cannot become fatal while idle.
    """

    def __init__(self):
        self.started = time.monotonic()
        self._buckets = collections.deque()
        self._bad_since = None
        self.successes = 0
        self.failures = collections.Counter()

    def record(self, successes=0, failures=()):
        now = int(time.monotonic())
        failed = len(failures)
        self.successes += successes
        self.failures.update(failures)
        if not successes and not failed:
            return
        if self._buckets and self._buckets[-1][0] == now:
            self._buckets[-1][1] += successes
            self._buckets[-1][2] += failed
        else:
            self._buckets.append([now, successes, failed])

    def metrics(self):
        now = time.monotonic()
        while self._buckets and self._buckets[0][0] <= int(now) - WINDOW_SECONDS:
            self._buckets.popleft()
        successes = sum(row[1] for row in self._buckets)
        failures = sum(row[2] for row in self._buckets)
        attempts = successes + failures
        fraction = failures / attempts if attempts else 0.0
        return {
            "errors/elapsed_seconds": now - self.started,
            "errors/completed_group_attempts": self.successes + self.failures.total(),
            "errors/requeued_groups": self.failures.total(),
            "errors/window_group_attempts": attempts,
            "errors/window_failed_groups": failures,
            "errors/failure_fraction_5m": fraction,
            "errors/high_failure_seconds": 0.0 if self._bad_since is None else now - self._bad_since,
            **{f"errors/{category}_groups": self.failures[category] for category in CATEGORIES},
        }

    def check(self):
        metrics = self.metrics()
        now = time.monotonic()
        if metrics["errors/failure_fraction_5m"] < FAILURE_FRACTION:
            self._bad_since = None
        elif self._bad_since is None:
            self._bad_since = now
        if self._bad_since is not None and now - self._bad_since >= WINDOW_SECONDS:
            raise RuntimeError(
                "Rollout failure rate stayed at or above 50% for 300 seconds: "
                f"{metrics['errors/window_failed_groups']}/{metrics['errors/window_group_attempts']} "
                "completed group attempts failed in the last five minutes"
            )

    def report(self, args, *, fatal=None):
        metrics = {**self.metrics(), "errors/fatal": int(fatal is not None)}
        # Diagnostics must neither mask the training error nor create one.
        try:
            if getattr(args, "save", None):
                path = Path(args.save) / "rollout_errors.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as stream:
                    stream.write(json.dumps({"time_unix": time.time(), "fatal": fatal, **metrics}) + "\n")
        except Exception:
            logger.exception("Could not retain rollout error metrics")
        try:
            tracking.log(args, metrics, step_key="errors/elapsed_seconds")
        except Exception:
            logger.exception("Could not publish rollout error metrics")


async def observe(producer):
    while True:
        producer._errors.report(producer.args)
        await asyncio.sleep(REPORT_INTERVAL_SECONDS)
