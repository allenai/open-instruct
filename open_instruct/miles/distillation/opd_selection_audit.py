"""Opt-in, per-attempt OPD selection evidence without prompts or token payloads.

Set OI_OPD_SELECTION_AUDIT to a JSONL path in a diagnostic run. Events describe
generation and delivery, not successful optimizer commits. Partial lengths on
aborted/cancelled attempts are lower bounds, never hypothetical final lengths.
"""

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from open_instruct.miles.distillation import opd_timing

_LOCK = threading.Lock()
_ATTEMPT_KEY = "oi_opd_selection_attempt"


def preflight_environment(environment):
    """Keep synthetic startup tests out of the actual run's capture artifacts."""
    return {
        key: value
        for key, value in environment.items()
        if key not in {"OI_OPD_SELECTION_AUDIT", "OI_OPD_EVAL_CAPTURE", opd_timing.ENV}
    }


async def generate(prompt_group, generate_group):
    """Observe the real generation callable and preserve its result/exception."""
    begin(prompt_group)
    opd_timing.start_group(prompt_group)
    try:
        result = await generate_group(prompt_group)
    except asyncio.CancelledError:
        record("generation_cancelled", prompt_group)
        raise
    except Exception as exc:
        record("generation_failed", prompt_group, error_type=type(exc).__name__)
        raise
    record("generation_returned", result.group)
    return result


def begin(group):
    if not os.environ.get("OI_OPD_SELECTION_AUDIT"):
        return
    attempt = uuid.uuid4().hex
    for sample in group:
        sample.metadata[_ATTEMPT_KEY] = attempt
    record("submitted", group)


def capture_eval(data, rollout_id):
    """Retain common-question comparisons locally instead of only a W&B table."""
    directory = os.environ.get("OI_OPD_EVAL_CAPTURE")
    if not directory:
        return
    destination = Path(directory) / f"eval-{rollout_id}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            for dataset, values in data.items():
                for sample, reward in zip(values["samples"], values["rewards"], strict=True):
                    row = {
                        "dataset": dataset,
                        "sample_index": sample.index,
                        "prompt": sample.prompt,
                        "response": sample.response,
                        "response_length": sample.response_length,
                        "status": sample.status.value,
                        "reward": reward,
                        "label": sample.label,
                        "tokens": sample.tokens,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def record(event, group, **context):
    path = os.environ.get("OI_OPD_SELECTION_AUDIT")
    if not path:
        return
    samples = [sample for item in group for sample in (item if isinstance(item, list) else [item])]
    if not samples:
        raise ValueError("Selection audit requires a nonempty group")
    first = samples[0]
    prompt = json.dumps(first.prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    row = {
        "schema_version": 1,
        "event": event,
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        "attempt_id": first.metadata.get(_ATTEMPT_KEY),
        "group_index": first.group_index,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "sample_indices": [sample.index for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "statuses": [sample.status.value for sample in samples],
        "weight_versions": [
            [
                int(span.version) if str(span.version).isdigit() else span.version
                for span in sample.all_weight_version_spans
            ]
            for sample in samples
        ],
        "weight_version_calls": [sample.to_dict()["weight_versions"] for sample in samples],
        **context,
    }
    destination = Path(path)
    with _LOCK:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
