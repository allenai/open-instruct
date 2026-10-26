"""Separate inference capacity from scheduler allocation in offline reports."""

import math
from datetime import datetime


def summarize(summary, inference_gpus, window, job=None):
    result = {
        "inference_gpus": inference_gpus,
        "response_tokens_per_inference_gpu_second": summary["response_tokens_per_second"] / inference_gpus,
        "allocated_gpus": None,
        "response_tokens_per_allocated_gpu_second": None,
        "legacy_summary_denominator": "Configured rollout_num_gpus, even when the field is named allocated_gpus",
        "scope": "Measured window only; startup, finalization and unused allocation time require the attempt ledger.",
    }
    if job is None:
        return result
    if isinstance(job, list):
        if len(job) != 1:
            raise ValueError("Supply metadata for exactly one Beaker execution attempt")
        job = job[0]
    if job.get("kind") != "execution" or not job.get("id"):
        raise ValueError("Expected Beaker execution job metadata")
    devices = job.get("limits", {}).get("gpus", [])
    if len(set(devices)) != len(devices) or len(devices) < inference_gpus:
        raise ValueError("Assigned GPU identities cannot cover the configured inference GPUs")
    status = job.get("status", {})
    if not status.get("started"):
        raise ValueError("Allocation metadata must describe a started attempt")
    started = datetime.fromisoformat(status["started"].replace("Z", "+00:00")).timestamp()
    stopped = status.get("exited") or status.get("finalized")
    stopped = datetime.fromisoformat(stopped.replace("Z", "+00:00")).timestamp() if stopped else math.inf
    if not started <= window[0] < window[1] <= stopped:
        raise ValueError("Measurement window is outside the supplied execution attempt")
    result.update(
        allocated_gpus=len(devices),
        response_tokens_per_allocated_gpu_second=summary["response_tokens_per_second"] / len(devices),
        beaker_job_id=job["id"],
        assigned_gpu_ids=devices,
        allocation_source="Beaker job limits.gpus; caller selects the producing attempt",
    )
    return result
