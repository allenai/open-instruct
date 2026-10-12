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
from open_instruct.miles.distillation import alignment, cancellation, config
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
        raise ValueError("Training sample lacks its privileged teacher prompt")
    response = sample.tokens[-sample.response_length :]
    objective = service.get("objective", "full")
    prompts = {"full": prompt}
    if objective == "cancel_answer_positive":
        answer_prompt = sample.metadata.get("opd_answer_prompt")
        if not isinstance(answer_prompt, str) or not answer_prompt:
            raise ValueError("Cancellation requires a separate answer-only teacher prompt")
        prompts["answer"] = answer_prompt
        if service.get("calibrate_initial"):
            versions = {span["version"] for call in sample.weight_versions for span in call.to_dicts()}
            if not versions:
                raise ValueError("Initial calibration requires recorded policy versions")
            if versions == {"0"}:
                prompts["bare_control"] = sample.tokens[: -sample.response_length]

    async def score_context(name, text):
        context = tokenizer.encode(text, add_special_tokens=False) if isinstance(text, str) else list(text)
        ids = context + response
        if len(ids) > service["max_context_length"]:
            raise ValueError("Teacher context overflow; do not silently shorten the reference")
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
        return (
            name,
            alignment.extract_scores(result, ids, sample.response_length),
            {
                "prompt_sha256": hashlib.sha256(
                    (text if isinstance(text, str) else json.dumps(text)).encode()
                ).hexdigest(),
                "context_tokens": len(context),
                "queue_seconds": admitted - requested,
                "score_seconds": completed - admitted,
            },
        )

    # Independent contexts share the bounded teacher admission semaphore.
    teacher_started = time.perf_counter()
    scored = await asyncio.gather(*(score_context(name, text) for name, text in prompts.items()))
    teacher_wall_seconds = time.perf_counter() - teacher_started
    values = {name: scores for name, scores, _ in scored}
    contexts = {name: metadata for name, _, metadata in scored}
    scores = values["full"]
    sample.teacher_log_probs = torch.tensor(scores, dtype=torch.float32)
    pressure_metadata = {}
    if objective == "cancel_answer_positive":
        behavior = sample.rollout_log_probs
        if behavior is None or len(behavior) != sample.response_length:
            raise ValueError("Cancellation requires generation-time student log probabilities")
        pressure, metrics = cancellation.token_pressure(
            behavior, scores, values["answer"], clip=service.get("pressure_clip")
        )
        # Native MILES subtracts beta * opd_reverse_kl. This field transports
        # negative shaped pressure here; it is not a KL divergence.
        sample.opd_reverse_kl = (-pressure).tolist()
        pressure_metadata = {
            "opd_pressure": metrics,
            "opd_answer_log_probs": values["answer"],
            "opd_behavior_log_probs": list(behavior),
            "opd_unscaled_pressure": pressure.tolist(),
        }
        if "bare_control" in values:
            gap = torch.tensor(values["bare_control"]) - torch.tensor(behavior)
            pressure_metadata.update(
                opd_initial_bare_log_probs=values["bare_control"],
                opd_initial_bare_gap_mean=gap.mean().item(),
                opd_initial_bare_gap_abs_mean=gap.abs().mean().item(),
                opd_initial_bare_gap_abs_max=gap.abs().max().item(),
            )
    task_reward = await rewards.registered_reward(args, sample, **kwargs)
    sample.train_metadata = dict(
        sample.train_metadata or {},
        opd_objective=objective,
        opd_contexts=contexts,
        opd_pressure_coef=service.get("kl_coef", getattr(args, "opd_kl_coef", 1.0)),
        opd_teacher_prompt_sha256=contexts["full"]["prompt_sha256"],
        opd_teacher_version=0,
        opd_teacher_source=service["source"],
        opd_teacher_context_tokens=contexts["full"]["context_tokens"],
        opd_teacher_queue_seconds=sum(c["queue_seconds"] for c in contexts.values()),
        opd_teacher_score_seconds=sum(c["score_seconds"] for c in contexts.values()),
        opd_teacher_wall_seconds=teacher_wall_seconds,
        **pressure_metadata,
    )
    return task_reward


def post_process(args, samples, **kwargs):
    records = []
    for sample in samples:
        if sample.teacher_log_probs is None or len(sample.teacher_log_probs) != sample.response_length:
            raise ValueError("Teacher scores must cover every generated response token")
        if not bool(torch.isfinite(sample.teacher_log_probs).all()):
            raise ValueError("Teacher scores must be finite")
        if sample.train_metadata.get("opd_objective") == "cancel_answer_positive":
            pressure = sample.opd_reverse_kl
            if pressure is None or len(pressure) != sample.response_length:
                raise ValueError("Cancellation pressure must cover every response token")
            if not bool(torch.isfinite(torch.as_tensor(pressure)).all()):
                raise ValueError("Cancellation pressure must be finite")
        records.append(
            {
                "sample_index": sample.index,
                "sample_id": sample.metadata.get("prepared_sample_id"),
                "response_length": sample.response_length,
                "response_token_ids": sample.tokens[-sample.response_length :],
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
                arrays = {
                    "teacher_log_probs",
                    "opd_answer_log_probs",
                    "opd_behavior_log_probs",
                    "opd_unscaled_pressure",
                    "opd_initial_bare_log_probs",
                    "response_token_ids",
                }
                stream.write(json.dumps({k: v for k, v in record.items() if k not in arrays}) + "\n")
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
        output = generate(args, rollout_id, data_source, evaluation=True)
        if args.save:
            path = Path(args.save).parent / "qualification-evaluation.jsonl"
            with path.open("a") as stream:
                for dataset, values in output.data.items():
                    for sample in values["samples"]:
                        stream.write(
                            json.dumps(
                                {
                                    "rollout_id": rollout_id,
                                    "dataset": dataset,
                                    "sample_index": sample.index,
                                    "sample_id": sample.metadata.get("prepared_sample_id"),
                                    "reward": sample.get_reward_value(args),
                                    "status": str(sample.status),
                                    "response_length": sample.response_length,
                                    "response": sample.response,
                                    "weight_versions": [call.to_dicts() for call in sample.weight_versions],
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
        return output
    finally:
        args.custom_rm_path, args.custom_reward_post_process_path = reward_path, postprocess_path
