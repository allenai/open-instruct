"""Per-collection reward-service metrics from the samples' verifier diagnostics.

MILES calls this through ``custom_rollout_log_function_path`` before its own
rollout logging. Returning False keeps the native ``perf N`` record; the extra
record here makes service rejections visible in the log, in tracking, and to
the run analyzer instead of only as warnings.
"""

import collections
import json
from pathlib import Path

from miles.ray.rollout.metrics import compute_rollout_step
from miles.utils.tracking_utils import tracking

from open_instruct import logger_utils
from open_instruct.miles import sibling_timing

logger = logger_utils.setup_logger(__name__)


def code_service_metrics(samples) -> dict[str, float]:
    """Aggregate code-service outcomes recorded by ``code_rewards.execute``."""
    counts = collections.Counter()
    by_status = collections.Counter()
    service_errors = collections.Counter()
    for sample in samples:
        diagnostics = getattr(sample, "metadata", None)
        diagnostics = diagnostics.get("verifier_diagnostics") if isinstance(diagnostics, dict) else None
        if not isinstance(diagnostics, dict):
            continue
        for record in diagnostics.values():
            if not isinstance(record, dict) or "program_chars" not in record:
                continue
            counts["samples"] += 1
            if record.get("status") == "rejected":
                counts["rejected"] += 1
                by_status[record.get("http_status")] += 1
            if record.get("status") == "service_error":
                counts["service_errors"] += 1
                service_errors[record.get("http_status") or "transport_or_response"] += 1
    if not counts["samples"]:
        return {}
    metrics = {
        "rollout/code_verifier/samples": counts["samples"],
        "rollout/code_verifier/rejected": counts["rejected"],
        "rollout/code_verifier/rejected_fraction": counts["rejected"] / counts["samples"],
        "rollout/code_verifier/service_errors": counts["service_errors"],
        "rollout/code_verifier/service_error_fraction": counts["service_errors"] / counts["samples"],
    }
    for status, n in sorted(by_status.items(), key=lambda kv: str(kv[0])):
        metrics[f"rollout/code_verifier/rejected_{status}"] = n
    for status, n in sorted(service_errors.items(), key=lambda kv: str(kv[0])):
        metrics[f"rollout/code_verifier/service_error_{status}"] = n
    return metrics


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    metrics = code_service_metrics(samples)
    if metrics:
        logger.info("code_verifier %d: %s", rollout_id, metrics)
    timing_metrics = sibling_timing.consumed_metrics(samples)
    if timing_metrics:
        logger.info("sibling_timing %d: %s", rollout_id, timing_metrics)
        metrics.update(timing_metrics)
    if metrics:
        if isinstance(rollout_extra_metrics, dict):
            rollout_extra_metrics.update(metrics)
        else:
            tracking.log(
                args, {**metrics, "rollout/step": compute_rollout_step(args, rollout_id)}, step_key="rollout/step"
            )
    if getattr(args, "save", None):
        root = Path(args.save)
        root.mkdir(parents=True, exist_ok=True)
        lengths = [sample.response_length for sample in samples]
        record = {
            "rollout_id": rollout_id,
            "samples": len(samples),
            "response_tokens": sum(lengths),
            "response_lengths": lengths,
            "collection_wait_seconds": rollout_time,
            "mixed_responses": sum(len(set(sample.weight_versions or [])) > 1 for sample in samples),
            "queue_metrics": rollout_extra_metrics or {},
            "sibling_group_attempts": sorted(
                {
                    record["group_attempt"]
                    for sample in samples
                    if (record := sibling_timing.sample_record(sample)) is not None
                }
            ),
        }
        with (root / "rollout_flow.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    return False
