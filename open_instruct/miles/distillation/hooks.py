"""Attach privileged-context teacher scores to fresh, bare-question rollouts."""

import asyncio
import functools
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from miles.rollout import on_policy_distillation
from open_instruct.miles.distillation import alignment, config
from open_instruct.miles.rewards import rewards


@functools.lru_cache(maxsize=4)
def resources(serialized, learner):
    service = json.loads(serialized)
    student = AutoTokenizer.from_pretrained(learner, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(service["source"], trust_remote_code=True)
    if (
        student.get_vocab() != teacher.get_vocab()
        or student.backend_tokenizer.to_str() != teacher.backend_tokenizer.to_str()
    ):
        raise ValueError("Context distillation requires identical student and teacher tokenizers")
    return service, teacher, asyncio.Semaphore(service["concurrency"])


async def reward(args, sample, **kwargs):
    service, tokenizer, limit = resources(os.environ[config.ENV], args.hf_checkpoint)
    if sample.response_length <= 0:
        raise ValueError("Distillation requires a nonempty student response")
    prompt = sample.metadata.get("opd_teacher_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Training sample lacks its frozen, success-derived teacher prompt")
    context = tokenizer.encode(prompt, add_special_tokens=False)
    response = sample.tokens[-sample.response_length :]
    ids = context + response
    if len(ids) > service["max_context_length"]:
        raise ValueError("Teacher context overflow; do not silently shorten the successful solution")
    requested = time.perf_counter()
    async with limit:
        admitted = time.perf_counter()
        result = await on_policy_distillation._post_json(
            service["endpoint"] + "/generate",
            {
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout_secs=service["request_timeout"],
        )
    completed = time.perf_counter()
    scores = alignment.extract_scores(result, ids, sample.response_length)
    sample.teacher_log_probs = torch.tensor(scores, dtype=torch.float32)
    task_reward = await rewards.registered_reward(args, sample, **kwargs)
    sample.train_metadata = dict(
        sample.train_metadata or {},
        opd_teacher_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        opd_teacher_version=0,
        opd_teacher_source=service["source"],
        opd_teacher_context_tokens=len(context),
        opd_teacher_queue_seconds=admitted - requested,
        opd_teacher_score_seconds=completed - admitted,
    )
    return task_reward


def post_process(args, samples, **kwargs):
    records = []
    for sample in samples:
        if sample.teacher_log_probs is None or len(sample.teacher_log_probs) != sample.response_length:
            raise ValueError("Teacher scores must cover every generated response token")
        if not bool(torch.isfinite(sample.teacher_log_probs).all()):
            raise ValueError("Teacher scores must be finite")
        records.append(
            {
                "sample_index": sample.index,
                "sample_id": sample.metadata.get("prepared_sample_id"),
                "response_length": sample.response_length,
                "status": str(sample.status),
                "task_reward": sample.get_reward_value(args),
                "teacher_log_probs": sample.teacher_log_probs.tolist(),
                "weight_versions": [call.to_dicts() for call in sample.weight_versions],
                **sample.train_metadata,
            }
        )
    if args.save:
        path = Path(args.save).parent / "teacher-scores.jsonl"
        with path.open("a") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        with path.with_name("teacher-metrics.jsonl").open("a") as stream:
            for record in records:
                stream.write(json.dumps({k: v for k, v in record.items() if k != "teacher_log_probs"}) + "\n")
    # Preserve outcome rewards for observation, while native OPD supplies the entire training signal.
    return [s.get_reward_value(args) for s in samples], [0.0] * len(samples)


def evaluate(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("Task evaluation callback is evaluation-only")
    reward_path, postprocess_path = args.custom_rm_path, args.custom_reward_post_process_path
    args.custom_rm_path = "open_instruct.miles.rewards.rewards.registered_reward"
    args.custom_reward_post_process_path = None
    try:
        generate = importlib.import_module("miles.rollout.sglang_rollout").generate_rollout
        return generate(args, rollout_id, data_source, evaluation=True)
    finally:
        args.custom_rm_path, args.custom_reward_post_process_path = reward_path, postprocess_path
