"""Observe sibling timing without changing admission, routing or sample ownership.

Producer durations use one process's monotonic clock. Engine timestamps are wall
clocks supplied by SGLang; cross-engine skews assume synchronized host clocks.
Missing measurements remain absent rather than being treated as zero.
"""

import json
import math
import statistics
import time
import uuid
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

ENGINE_TIMESTAMPS = {
    "request_received_ts": "engine_received_unix",
    "forward_entry_time": "engine_first_forward_unix",
    "prefill_finished_time": "engine_first_prefill_finished_unix",
    "request_finished_ts": "engine_finished_unix",
}


def start_group(samples):
    """Start a fresh attempt before sibling tasks compete for HTTP admission."""
    attempt = uuid.uuid4().hex
    wall, monotonic = time.time(), time.monotonic()
    records = []
    for sample in samples:
        record = dict(
            group_attempt=attempt,
            group_index=sample.group_index,
            group_size=len(samples),
            sample_index=sample.index,
            group_created_unix=wall,
            group_created_monotonic=monotonic,
        )
        sample.metadata = {**(sample.metadata or {}), "sibling_timing": record}
        records.append(record)
    return records


def sample_record(sample):
    return (getattr(sample, "metadata", None) or {}).get("sibling_timing")


def admitted(sample):
    record = sample_record(sample)
    if record is not None:
        record["admitted_monotonic"] = time.monotonic()
        record["admitted_unix"] = time.time()


def response_received(sample, request_id, meta):
    record = sample_record(sample)
    if record is None:
        return
    record.update(
        request_id=request_id, response_received_monotonic=time.monotonic(), response_received_unix=time.time()
    )
    for source, destination in ENGINE_TIMESTAMPS.items():
        value = meta.get(source)
        if type(value) in (float, int) and math.isfinite(value) and value > 0:
            record[destination] = value
    # queue_time can describe a later retraction in this runtime, so do not use
    # it as the initial queue wait. Preserve it only as a raw engine observation.
    for key in ("queue_time", "e2e_latency", "num_retractions"):
        value = meta.get(key)
        if type(value) in (float, int) and math.isfinite(value) and value >= 0:
            record["engine_reported_" + key] = value


def finished(sample, outcome):
    record = sample_record(sample)
    if record is not None:
        record.update(
            call_finished_monotonic=time.monotonic(),
            call_finished_unix=time.time(),
            outcome=outcome,
            response_tokens=sample.response_length,
        )


def summarize(records):
    """Report coverage and first/last boundaries, only aggregate complete fields."""
    result = {"samples": len(records)}
    for label, key in (
        ("admission", "admitted_monotonic"),
        ("response_received", "response_received_monotonic"),
        ("call_finished", "call_finished_monotonic"),
        ("engine_received", "engine_received_unix"),
        ("engine_first_forward", "engine_first_forward_unix"),
        ("engine_first_prefill_finished", "engine_first_prefill_finished_unix"),
        ("engine_finished", "engine_finished_unix"),
    ):
        values = [r[key] for r in records if key in r]
        result[label + "_samples"] = len(values)
        if values and len(values) == len(records):
            result[label + "_first"] = min(values)
            result[label + "_last"] = max(values)
            result[label + "_skew_seconds"] = max(values) - min(values)
    for label, start, end in (
        ("admission_wait", "group_created_monotonic", "admitted_monotonic"),
        ("http", "admitted_monotonic", "response_received_monotonic"),
        ("engine_initial_wait", "engine_received_unix", "engine_first_forward_unix"),
        ("engine_execution", "engine_first_forward_unix", "engine_finished_unix"),
    ):
        durations = [r[end] - r[start] for r in records if start in r and end in r]
        valid = [d for d in durations if math.isfinite(d) and d >= 0]
        result[label + "_samples"] = len(valid)
        result[label + "_invalid_order_samples"] = len(durations) - len(valid)
        if valid and len(valid) == len(records):
            result[label + "_mean_seconds"] = statistics.mean(valid)
            result[label + "_max_seconds"] = max(valid)
    return result


def write_group(args, records, outcome):
    """Include failed/cancelled and later-discarded groups, independently of W&B consumption."""
    if not records or not getattr(args, "save", None):
        return
    try:
        row = dict(
            schema_version=1,
            group_attempt=records[0]["group_attempt"],
            group_index=records[0]["group_index"],
            outcome=outcome,
            group_finished_unix=time.time(),
            group_finished_monotonic=time.monotonic(),
            samples=records,
            summary=summarize(records),
        )
        # Shard the qualification artifacts to reduce per-file result size.
        path = Path(args.save) / f"sibling_timing_{records[0]['group_attempt'][0]}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    except Exception:
        logger.exception("Sibling timing observation unavailable")


def consumed_metrics(samples):
    """W&B summaries explicitly describe consumed groups; the JSONL covers all attempts."""
    groups = {}
    for sample in samples:
        record = sample_record(sample)
        if record is not None:
            groups.setdefault(record["group_attempt"], []).append(record)
    if not groups:
        return {}
    complete = [records for records in groups.values() if len(records) == records[0]["group_size"]]
    summaries = [summarize(records) for records in complete]
    prefix = "rollout/siblings/consumed/"
    metrics = {prefix + "groups": len(complete), prefix + "incomplete_groups": len(groups) - len(complete)}
    for key in sorted({key for summary in summaries for key in summary}):
        if not key.endswith("_seconds"):
            continue
        values = sorted(s[key] for s in summaries if key in s)
        if values:
            metrics[prefix + key + "/groups"] = len(values)
            metrics[prefix + key + "/mean"] = statistics.mean(values)
            metrics[prefix + key + "/p95"] = values[math.ceil(0.95 * len(values)) - 1]
            metrics[prefix + key + "/max"] = max(values)
    return metrics
