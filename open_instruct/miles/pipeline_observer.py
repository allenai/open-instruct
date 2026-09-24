"""Optional sampled queue occupancy; observations never dequeue or reset counters.

Semaphore occupancy includes requests in the router/server and response processing;
it is not GPU utilization. Private reads are for the pinned MILES/asyncio runtime.
Unsupported fields are null, not zero. The observer is owned by the producer task.
"""

import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path

import httpx

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def completion_counts(entries):
    """Count retained completed data without consuming the pinned buffer."""
    counts = {"groups": 0, "samples": 0, "response_tokens": 0}
    for entry in entries:
        if not hasattr(entry, "group"):
            return None
        counts["groups"] += 1
        for trajectory in entry.group:
            for sample in trajectory if isinstance(trajectory, list) else [trajectory]:
                if not isinstance(getattr(sample, "response_length", None), int):
                    return None
                counts["samples"] += 1
                counts["response_tokens"] += sample.response_length
    return counts


def write_lifecycle(producer, event):
    """Persist exact lifecycle boundaries separately from periodic samples."""
    if not getattr(producer.args, "save", None):
        return
    try:
        path = Path(producer.args.save) / "pipeline_lifecycle.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"event": event, **snapshot(producer)}) + "\n")
    except Exception:
        logger.exception("Pipeline lifecycle observation unavailable: %s", event)


def snapshot(producer):
    semaphore = getattr(producer.state, "generate_fn_semaphore", None)
    slots = getattr(semaphore, "_semaphore", semaphore)
    free = getattr(slots, "_value", None)
    waiters = getattr(slots, "_waiters", None)
    args = producer.args
    capacity = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    output = producer._output
    delegate = getattr(output, "_delegate", output)
    buffer = getattr(delegate, "_buffer", None)
    completed = completion_counts(buffer) if buffer is not None else None
    buffered_ids = {id(entry) for entry in buffer} if buffer is not None else set()
    ready = [
        counts
        for identity, counts in getattr(producer, "_ready_completion_counts", {}).items()
        if identity not in buffered_ids
    ]
    for task in producer._active_tasks:
        if task.done() and not task.cancelled() and task.exception() is None:
            ready.append(completion_counts([task.result()]))
    ready_counts = (
        {key: sum(row[key] for row in ready) for key in ("groups", "samples", "response_tokens")}
        if all(row is not None for row in ready)
        else None
    )
    return {
        "completed_queue": completed,
        "producer_ready": ready_counts,
        "shutdown_unqueued": getattr(producer, "_shutdown_unqueued_counts", None),
        "completed_put_wait_seconds": getattr(producer, "_completed_put_wait_seconds", None),
        "completed_put_current_wait_seconds": (
            time.monotonic() - producer._completed_put_started
            if getattr(producer, "_completed_put_started", None) is not None
            else 0.0
        ),
        "time_unix": time.time(),
        "producer_owned_groups": len(producer._producing_groups),
        **(producer._errors.metrics() if hasattr(producer, "_errors") else {}),
        "producer_active_group_tasks": sum(not task.done() for task in producer._active_tasks),
        "producer_admission_open": producer._producer_resumed.is_set(),
        "producer_unfinished_samples": getattr(producer._scheduler, "samples_in_flight", None),
        "completed_queue_groups": len(buffer) if buffer is not None else None,
        "completed_queue_capacity_groups": getattr(delegate, "_capacity", None),
        "generation_admission_paused": getattr(semaphore, "paused", None),
        "generation_admission_waiters": getattr(semaphore, "waiting", None),
        "generation_active_calls": getattr(semaphore, "active", None),
        "http_capacity_requests": capacity,
        "http_active_requests": capacity - free if free is not None else None,
        "http_waiting_requests": sum(not waiter.done() for waiter in waiters)
        if waiters is not None
        else 0
        if hasattr(slots, "_waiters")
        else None,
    }


async def observe(producer):
    interval = getattr(getattr(producer.args, "olmo_core", None), "pipeline_observation_interval", 0)
    if interval <= 0 or not getattr(producer.args, "save", None):
        return
    path = Path(producer.args.save) / "pipeline_occupancy.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", buffering=1) as stream:
            while True:
                stream.write(json.dumps(snapshot(producer)) + "\n")
                await asyncio.sleep(interval)
    except Exception:
        # Diagnostics must not terminate generation or change ownership.
        logger.exception("Pipeline occupancy observation stopped; remaining occupancy data is unavailable")


ENGINE_METRICS = frozenset(
    (
        "num_running_reqs",
        "num_queue_reqs",
        "token_usage",
        "full_token_usage",
        "mamba_usage",
        "gen_throughput",
        "cache_hit_rate",
        "utilization",
        "fwd_occupancy",
    )
)
METRIC = re.compile(r"^sglang:(\w+)(\{.*\})?\s+([^\s]+)")


def engine_metrics(text):
    """Retain series labels; never sum distinct ranks or reinterpret absent gauges as zero."""
    result = []
    for line in text.splitlines():
        match = METRIC.match(line)
        if match and match[1] in ENGINE_METRICS:
            value = float(match[3])
            result.append(dict(name=match[1], labels=match[2], value=value if math.isfinite(value) else None))
    return result


async def observe_engines(producer, get_urls):
    interval = getattr(getattr(producer.args, "olmo_core", None), "pipeline_observation_interval", 0)
    if interval <= 0 or not getattr(producer.args, "save", None):
        return
    path = Path(producer.args.save) / "engine_occupancy.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=2.0) as client:
            with path.open("a", buffering=1) as stream:
                urls = []

                async def collect(url):
                    record = dict(time_unix=time.time(), engine=url)
                    try:
                        response = await client.get(url + "/metrics")
                        response.raise_for_status()
                        record["series"] = engine_metrics(response.text)
                    except Exception as error:
                        record["error"] = str(error)
                    return record

                while True:
                    if not urls:
                        try:
                            urls = await asyncio.wait_for(get_urls(producer.args), timeout=2.0)
                        except Exception as error:
                            stream.write(json.dumps(dict(time_unix=time.time(), discovery_error=str(error))) + "\n")
                    for record in await asyncio.gather(*(collect(url) for url in urls)):
                        # A large fleet can exceed the result collector's per-file
                        # limit. Shard by endpoint without losing series labels.
                        identity = hashlib.sha256(record["engine"].encode()).hexdigest()[:12]
                        destination = path.with_name(f"engine_occupancy_{identity}.jsonl")
                        with destination.open("a") as output:
                            output.write(json.dumps(record) + "\n")
                    await asyncio.sleep(max(5.0, interval))
    except Exception:
        logger.exception("Engine occupancy observation stopped; remaining occupancy data is unavailable")
