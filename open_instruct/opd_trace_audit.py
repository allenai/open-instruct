"""Audit an on-policy distillation (OPD) trainer trace on real data.

``grpo_fast.py`` writes one JSONL record per micro-batch when ``--save_traces`` is on
(``rl_utils.save_trainer_logprobs_to_disk``). With an OPD teacher the record also carries
the combined teacher logprobs and the advantages the trainer actually consumed, all shifted
the same way (``query_responses[:, 1:]``). This tool re-derives the pure-OPD advantage from
the dumped rollout and teacher logprobs and checks that the trainer used exactly that, so the
``[:, 1:]`` alignment of rollout logprobs, teacher logprobs and response mask is verified on
real data rather than on synthetic tensors. It mirrors ``open_instruct.miles.opd_audit``.

Checks per record:

* rollout, teacher, advantage and response-mask tensors share one shape;
* every response-token rollout and teacher logprob is finite (a NaN there would have become
  the ``INVALID_LOGPROB`` sentinel in the advantage);
* ``advantage == kl_coef * (teacher - rollout)`` on response tokens and ``0`` elsewhere
  (pure OPD; ``--adv_clip`` applies the same clamp the run used);
* the teacher signal is not identically zero.

Usage::

    python -m open_instruct.opd_trace_audit --trace_dir /weka/.../deletable_rollouts \\
        --run_name opd_trace_audit_qwen3 --step 1 --step 2 --kl_coef 1.0
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

REQUIRED_KEYS = ("vllm_logprobs", "teacher_logprobs", "advantages", "response_mask")


def trace_files(trace_dir: Path, run_name: str | None, step: int) -> list[Path]:
    """Trace files for ``step``; an empty ``run_name`` matches every run in ``trace_dir``."""
    prefix = f"{run_name}_" if run_name else "*_"
    return sorted(trace_dir.glob(f"{prefix}trainer_logprobs_step{step:06d}*.jsonl"))


def load_records(trace_dir: Path, run_name: str | None, step: int) -> list[dict]:
    files = trace_files(trace_dir, run_name, step)
    if not files:
        raise FileNotFoundError(f"No trace for run {run_name or '*'} step {step} under {trace_dir}")
    records = []
    for path in files:
        with open(path) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def _tensor(record: dict, key: str, dtype: torch.dtype) -> torch.Tensor:
    values = [math.nan if v is None else v for v in record[key]]
    return torch.tensor(values, dtype=dtype).reshape(record[f"{key}_shape"])


def audit_record(record: dict, kl_coef: float, adv_clip: float | None, atol: float) -> dict:
    """Return the per-record report; ``report["errors"]`` is empty when the record passes."""
    errors: list[str] = []
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        return {"sample_idx": record.get("sample_idx"), "errors": [f"record lacks {missing}"]}
    shapes = {key: tuple(record[f"{key}_shape"]) for key in REQUIRED_KEYS}
    mask = _tensor(record, "response_mask", torch.float32).bool()
    rollout = _tensor(record, "vllm_logprobs", torch.float32)
    teacher = _tensor(record, "teacher_logprobs", torch.float32)
    advantage = _tensor(record, "advantages", torch.float32)
    # Dumps written before the advantages were shifted carry them in the full query_response
    # frame ([B, T+1]); the trainer consumes advantages[:, 1:], so audit that view.
    advantages_shifted = False
    if advantage.shape[:-1] == mask.shape[:-1] and advantage.shape[-1] == mask.shape[-1] + 1:
        advantage = advantage[..., 1:]
        advantages_shifted = True
    shapes["advantages"] = tuple(advantage.shape)
    if len(set(shapes.values())) != 1:
        return {"sample_idx": record.get("sample_idx"), "errors": [f"shape mismatch {shapes}"]}

    response_tokens = int(mask.sum())
    if response_tokens == 0:
        errors.append("empty response mask")
    nonfinite_rollout = int((mask & ~torch.isfinite(rollout)).sum())
    nonfinite_teacher = int((mask & ~torch.isfinite(teacher)).sum())
    if nonfinite_rollout:
        errors.append(f"{nonfinite_rollout} non-finite rollout logprobs inside the response mask")
    if nonfinite_teacher:
        errors.append(f"{nonfinite_teacher} non-finite teacher logprobs inside the response mask")
    if not torch.isfinite(advantage).all():
        errors.append("non-finite advantages")

    expected = kl_coef * (teacher - rollout)
    if adv_clip is not None:
        expected = expected.clamp(-adv_clip, adv_clip)
    expected = torch.where(mask, expected, torch.zeros_like(expected))
    finite = torch.isfinite(expected) & torch.isfinite(advantage)
    diff = torch.where(finite, (advantage - expected).abs(), torch.zeros_like(advantage))
    max_error = float(diff.max()) if diff.numel() else 0.0
    off_mask_error = float(diff[~mask].max()) if (~mask).any() else 0.0
    if max_error > atol:
        errors.append(f"advantage differs from kl_coef*(teacher-rollout) by {max_error:.3e} (> {atol})")
    if off_mask_error > atol:
        errors.append(f"non-zero advantage {off_mask_error:.3e} outside the response mask")
    signal = float(expected[mask].abs().max()) if response_tokens else 0.0
    if response_tokens and signal == 0.0:
        errors.append("teacher signal is identically zero")

    report = {
        "sample_idx": record.get("sample_idx"),
        "response_tokens": response_tokens,
        "finite_rollout_logprobs_outside_mask": int((~mask & torch.isfinite(rollout)).sum()),
        "max_advantage_error": max_error,
        "max_abs_opd_signal": signal,
        "mean_reverse_kl": float((rollout - teacher)[mask].mean()) if response_tokens else 0.0,
        "advantages_shifted_in_audit": advantages_shifted,
        "errors": errors,
    }
    if "trainer_logprobs" in record:
        trainer = _tensor(record, "trainer_logprobs", torch.float32)
        gap = (trainer - rollout)[mask & torch.isfinite(trainer) & torch.isfinite(rollout)]
        report["mean_abs_trainer_rollout_gap"] = float(gap.abs().mean()) if gap.numel() else 0.0
    return report


def audit_trace(
    trace_dir: Path, run_name: str | None, steps: list[int], kl_coef: float, adv_clip: float | None, atol: float
) -> dict:
    summary = {"run_name": run_name, "kl_coef": kl_coef, "steps": {}, "failures": 0}
    for step in steps:
        reports = [audit_record(r, kl_coef, adv_clip, atol) for r in load_records(trace_dir, run_name, step)]
        failures = [r for r in reports if r["errors"]]
        summary["failures"] += len(failures)
        summary["steps"][step] = {
            "records": len(reports),
            "response_tokens": sum(r.get("response_tokens", 0) for r in reports),
            "max_advantage_error": max((r.get("max_advantage_error", 0.0) for r in reports), default=0.0),
            "max_abs_opd_signal": max((r.get("max_abs_opd_signal", 0.0) for r in reports), default=0.0),
            "records_with_unshifted_advantages": sum(r.get("advantages_shifted_in_audit", False) for r in reports),
            "failed_records": [{"sample_idx": r["sample_idx"], "errors": r["errors"]} for r in failures],
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace_dir", type=Path, required=True)
    parser.add_argument("--run_name", default="", help="run_name prefix of the trace files; empty matches every run")
    parser.add_argument("--step", type=int, action="append", required=True, help="training step; repeatable")
    parser.add_argument("--kl_coef", type=float, default=1.0)
    parser.add_argument("--adv_clip", type=float, default=None, help="the run's --opd_adv_clip, if any")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=None, help="write the JSON summary here as well")
    args = parser.parse_args(argv)

    summary = audit_trace(args.trace_dir, args.run_name, args.step, args.kl_coef, args.adv_clip, args.atol)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    if summary["failures"]:
        logger.error(f"OPD trace audit failed for {summary['failures']} record(s)")
        return 1
    logger.info("OPD trace audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
