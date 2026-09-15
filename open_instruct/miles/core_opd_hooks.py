"""Independent teacher scoring and task evaluation for synchronous Core OPD."""

import asyncio
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch
from miles.rollout import on_policy_distillation
from transformers import AutoTokenizer

from open_instruct.miles import core_opd, opd_alignment


@functools.lru_cache(maxsize=4)
def resources(serialized, learner):
    service = json.loads(serialized)
    student = AutoTokenizer.from_pretrained(learner, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(service["snapshot"], trust_remote_code=True)
    if service["alignment"] == "shared_token_ids" and (
        student.get_vocab() != teacher.get_vocab()
        or student.backend_tokenizer.to_str() != teacher.backend_tokenizer.to_str()
    ):
        raise ValueError("shared_token_ids requires identical tokenizers; choose exact_text_spans")

    def fingerprint(tok):
        return hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()

    service["student_tokenizer_sha256"] = fingerprint(student)
    service["teacher_tokenizer_sha256"] = fingerprint(teacher)
    return service, student, teacher, asyncio.Semaphore(service["concurrency"])


async def reward(args, sample, **kwargs):
    service, student, teacher, limit = resources(os.environ[core_opd.ENV], args.hf_checkpoint)
    if sample.response_length <= 0:
        raise ValueError("OPD requires a nonempty response")
    response_ids = sample.tokens[-sample.response_length :]
    if service["alignment"] == "shared_token_ids":
        ids = sample.tokens
        mapping = list(range(len(ids) - sample.response_length, len(ids)))
    else:
        messages = sample.metadata.get("opd_messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Cross-tokenizer OPD requires metadata.opd_messages from data preparation")
        context = teacher.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **service["chat_template_kwargs"]
        )
        ids, mapping = opd_alignment.align(student, response_ids, teacher, context)
    if len(ids) > service["max_context_length"]:
        raise ValueError("Teacher context overflow; increase teacher.max_context_length")
    async with limit:
        result = await on_policy_distillation._post_json(
            service["endpoint"] + "/generate",
            on_policy_distillation._score_payload(ids),
            timeout_secs=service["request_timeout"],
        )
    scores, mask = opd_alignment.extract_scores(result, ids, mapping)
    sample.teacher_log_probs = torch.tensor(scores, dtype=torch.float32)
    sample.train_metadata = dict(
        sample.train_metadata or {},
        opd_alignment_mask=mask,
        student_tokenizer_sha256=service["student_tokenizer_sha256"],
        teacher_tokenizer_sha256=service["teacher_tokenizer_sha256"],
    )
    # Native rollout statistics run before post_process and require scalar task
    # rewards, including when two siblings produce identical responses/scores.
    return 0.0


def post_process(args, samples, **kwargs):
    for sample in samples:
        if sample.teacher_log_probs is None or len(sample.teacher_log_probs) != sample.response_length:
            raise ValueError("Teacher response does not match the learner response length")
        if len(sample.train_metadata["opd_alignment_mask"]) != sample.response_length:
            raise ValueError("Teacher mask does not match the learner response length")
    if args.save and getattr(args.olmo_core, "diagnostic_interval", 0) > 0:
        path = Path(args.save).parent / "teacher-scores.jsonl"
        with path.open("a") as stream:
            for sample in samples:
                stream.write(
                    json.dumps(
                        {
                            "sample_index": sample.index,
                            "response_length": sample.response_length,
                            "teacher_log_probs": sample.teacher_log_probs.tolist(),
                            **sample.train_metadata,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
    return [0.0] * len(samples), [0.0] * len(samples)


def generate_rollout(*args, **kwargs):
    return importlib.import_module("miles.rollout.sglang_rollout").generate_rollout(*args, **kwargs)


def evaluate(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("Task evaluation callback is evaluation-only")
    reward_path, postprocess_path = args.custom_rm_path, args.custom_reward_post_process_path
    args.custom_rm_path = "open_instruct.miles.rewards.registered_reward"
    args.custom_reward_post_process_path = None
    try:
        return generate_rollout(args, rollout_id, data_source, evaluation=True)
    finally:
        args.custom_rm_path = reward_path
        args.custom_reward_post_process_path = postprocess_path
