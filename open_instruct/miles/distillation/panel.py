"""Compare fresh trajectories and fixed prefixes using matched scoring calls."""

import asyncio
import gzip
import json
import os
from pathlib import Path
from urllib import request

import torch

from open_instruct.miles.distillation import alignment, cancellation, config


def post(url, payload, timeout=600):
    req = request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


async def score(url, prefix, tokens, limit):
    ids = prefix + tokens
    async with limit:
        result = await asyncio.to_thread(
            post,
            url,
            {
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        )
    return alignment.extract_scores(result, ids, len(tokens))


def movement(initial, current, pressure):
    old, new, pressure = [torch.as_tensor(x, dtype=torch.float32) for x in (initial, current, pressure)]
    if old.shape != new.shape or old.shape != pressure.shape or not old.numel():
        raise ValueError("Fixed-text comparisons require the same nonempty token sequence")
    delta = new - old
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite fixed-text comparison")
    return {
        "tokens": old.numel(),
        "pressure_times_logprob_change_sum": (pressure * delta).sum().item(),
        "encouraged_logprob_change_sum": delta[pressure > 0].sum().item(),
        "encouraged_tokens": (pressure > 0).sum().item(),
        "discouraged_logprob_change_sum": delta[pressure < 0].sum().item(),
        "discouraged_tokens": (pressure < 0).sum().item(),
    }


async def observe(args, rollout_id, output, tokenizer, descriptor):
    """Score a predetermined panel; never select traces by their outcomes."""
    root = Path(args.save).parent / "diagnostics"
    root.mkdir(exist_ok=True)
    service = json.loads(os.environ[config.ENV])
    teacher = service["endpoint"] + "/generate"
    student = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    limit = asyncio.Semaphore(2)
    fixed_path = root / "fixed-traces.jsonl.gz"
    contexts = {r["metadata"]["prepared_sample_id"]: r for r in descriptor["rows"]}
    selected = []
    for dataset in output.data.values():
        counts = {}
        for sample in sorted(dataset["samples"], key=lambda s: s.index):
            identity = sample.metadata["prepared_sample_id"]
            if identity not in contexts or counts.get(identity, 0) >= 2:
                continue
            counts[identity] = counts.get(identity, 0) + 1
            selected.append(sample)
    if len(selected) != len(contexts) * 2:
        raise ValueError("Fixed question diagnostic has incomplete sample coverage")

    async def fresh(sample):
        row = contexts[sample.metadata["prepared_sample_id"]]
        tokens = sample.tokens[-sample.response_length :]
        prefix = sample.tokens[: -sample.response_length]
        full_prefix = tokenizer.encode(row["metadata"]["opd_teacher_prompt"], add_special_tokens=False)
        answer_prefix = tokenizer.encode(row["metadata"]["opd_answer_prompt"], add_special_tokens=False)
        bare, full, answer, frozen_bare = await asyncio.gather(
            score(student, prefix, tokens, limit),
            score(teacher, full_prefix, tokens, limit),
            score(teacher, answer_prefix, tokens, limit),
            score(teacher, prefix, tokens, limit),
        )
        pressure, metrics = cancellation.token_pressure(bare, full, answer)
        decode = sample.rollout_log_probs
        if decode is None or len(decode) != len(tokens):
            raise ValueError("Fresh-panel calibration requires generation log probabilities")
        behavior_pressure, behavior_metrics = cancellation.token_pressure(decode, full, answer)
        delta = torch.tensor(bare) - torch.tensor(decode)
        bare_gap = torch.tensor(frozen_bare) - torch.tensor(bare)
        record = {
            "sample_id": sample.metadata["prepared_sample_id"],
            "sample_index": sample.index,
            "rollout_id": rollout_id,
            "prefix": prefix,
            "tokens": tokens,
            "student_prefill": bare,
            "teacher_full": full,
            "teacher_answer": answer,
            "frozen_bare": frozen_bare,
            "student_decode": decode,
            "pressure": pressure.tolist(),
            "behavior_pressure": behavior_pressure.tolist(),
            "reward": sample.get_reward_value(args),
            "status": str(sample.status),
            "weight_versions": [call.to_dicts() for call in sample.weight_versions],
        }
        summary = {
            "kind": "fresh",
            "rollout_id": rollout_id,
            "sample_id": record["sample_id"],
            "reward": record["reward"],
            "status": record["status"],
            **metrics,
            "behavior_pressure": behavior_metrics,
            "full_minus_student_sum": (torch.tensor(full) - torch.tensor(bare)).sum().item(),
            "decode_prefill_abs_mean": delta.abs().mean().item(),
            "bare_teacher_student_prefill_abs_mean": bare_gap.abs().mean().item(),
        }
        return record, summary

    scored = await asyncio.gather(*(fresh(sample) for sample in selected))
    with gzip.open(root / f"fresh-{rollout_id:04d}.jsonl.gz", "wt") as stream:
        for record, _ in scored:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    summaries = [summary for _, summary in scored]
    if rollout_id == 0:
        if fixed_path.exists():
            raise ValueError("Refusing to overwrite the fixed baseline panel")
        with gzip.open(fixed_path, "wt") as stream:
            for record, _ in scored:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    else:
        with gzip.open(fixed_path, "rt") as stream:
            fixed = [json.loads(line) for line in stream]
        with gzip.open(root / f"fixed-{rollout_id:04d}.jsonl.gz", "wt") as stream:
            for record in fixed:
                current = await score(student, record["prefix"], record["tokens"], limit)
                summary = {
                    "kind": "fixed",
                    "rollout_id": rollout_id,
                    "sample_id": record["sample_id"],
                    **movement(record["student_prefill"], current, record["pressure"]),
                    "behavior_pressure_counterfactual": movement(
                        record["student_prefill"], current, record["behavior_pressure"]
                    ),
                }
                summaries.append(summary)
                stream.write(json.dumps({**summary, "student_prefill": current}, allow_nan=False) + "\n")
    with (root / "panel-metrics.jsonl").open("a") as stream:
        for summary in summaries:
            stream.write(json.dumps(summary, allow_nan=False) + "\n")
