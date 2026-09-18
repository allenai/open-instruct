"""Teacher score validation and task evaluation around upstream sampled-token OPD."""

import asyncio
import json
import math
import os
from pathlib import Path

import aiohttp
from miles.rollout import on_policy_distillation, sglang_rollout

from open_instruct.miles import eopd_math, rewards

_LIMIT = None
# The teacher's scoring response rides on the sample here (samples are deep-copied per group);
# `sample.reward` stays the numeric task reward (0.0 under pure OPD) because Miles's rollout
# metrics round and group by it. `post_process` turns the response into teacher_log_probs.
TEACHER_RESPONSE_KEY = "opd_teacher_response"


def _eopd():
    return eopd_math.Settings.from_environment()


async def _score(args, sample, settings, **kwargs):
    """Teacher scores for the sampled tokens, plus the teacher's top-k per position under EOPD."""
    if not settings.enabled:
        return await on_policy_distillation.reward_func(args, sample, **kwargs)
    payload = on_policy_distillation._score_payload(sample.tokens, top_k=settings.top_k)
    return await on_policy_distillation._post_json(
        on_policy_distillation._teacher_url_for_sample(args, sample),
        payload,
        timeout_secs=getattr(args, "sglang_router_request_timeout_secs", None),
    )


def _top_k(result, sample, settings):
    entries = on_policy_distillation._trim_input_field(
        result["meta_info"], "input_top_logprobs", sample.response_length
    )
    if len(entries) != sample.response_length:
        raise ValueError("Teacher top-k covers different positions than the response")
    return eopd_math.parse_top_entries(entries, settings.top_k)


async def reward(args, sample, **kwargs):
    global _LIMIT
    if _LIMIT is None:
        _LIMIT = asyncio.Semaphore(int(os.environ.get("OI_OPD_TEACHER_CONCURRENCY", "4")))
    settings = _eopd()
    async with _LIMIT:
        for attempt in range(3):
            try:
                result = await _score(args, sample, settings, **kwargs)
                entries = result["meta_info"]["input_token_logprobs"][1:][-sample.response_length :]
                if [entry[1] for entry in entries] != sample.tokens[-sample.response_length :]:
                    raise ValueError("Teacher scored different token positions or IDs")
                scores = on_policy_distillation._teacher_sampled_log_probs(result, sample.response_length)
                if len(scores) != sample.response_length or not all(math.isfinite(x) for x in scores.tolist()):
                    raise ValueError("Teacher scores are missing, misaligned or nonfinite")
                if settings.enabled:
                    _top_k(result, sample, settings)
                sample.metadata[TEACHER_RESPONSE_KEY] = result
                return 0.0
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt == 2:
                    raise
                await asyncio.sleep(attempt + 1)


def post_process(args, samples, **kwargs):
    """Upstream ``post_process_rewards`` for the sampled-token path, reading the response stored by ``reward``."""
    settings = _eopd()
    path = Path(os.environ["OI_OPD_OUTPUT"]) / "teacher-scores.jsonl"
    with path.open("a") as stream:
        for sample in samples:
            response = sample.metadata.pop(TEACHER_RESPONSE_KEY, None)
            if response is None:
                raise ValueError("Sample carries no teacher response; opd_hooks.reward must score every sample")
            sample.teacher_log_probs = on_policy_distillation._teacher_sampled_log_probs(
                response, sample.response_length
            )
            values = sample.teacher_log_probs.tolist()
            if len(values) != sample.response_length or not values or not all(map(math.isfinite, values)):
                raise ValueError("Invalid response-aligned teacher signal")
            record = {
                "sample_index": sample.index,
                "tokens": sample.tokens,
                "response_length": sample.response_length,
                "teacher_log_probs": values,
                "response": sample.response,
            }
            if settings.enabled:
                # The top-k rides to the trainer in train_metadata (the batch's `metadata` list);
                # the score log keeps only the derived gate so it stays small.
                ids, log_probs = _top_k(response, sample, settings)
                sample.train_metadata = {
                    **(sample.train_metadata or {}),
                    "eopd_topk_ids": ids.tolist(),
                    "eopd_topk_logprobs": log_probs.tolist(),
                }
                record["eopd_gate"] = eopd_math.gate(log_probs, settings.tau).int().tolist()
                record["eopd_proxy_entropy"] = eopd_math.proxy_entropy(log_probs).tolist()
                record["eopd_topk_mass"] = eopd_math.topk_mass(log_probs).tolist()
            stream.write(json.dumps(record) + "\n")
    # Pure on-policy distillation: the task reward is zero and the signal is the teacher KL.
    rewards = [0.0] * len(samples)
    return rewards, rewards


async def eval_reward(args, sample, **kwargs):
    """Score held-out samples with the verifiers named in their metadata (prepared verifiers.json)."""
    return await rewards.score(sample, os.environ["OI_OPD_REWARD_CONFIG"], args)


def evaluate(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("OPD task evaluation is evaluation-only")
    # Legacy GenerateState retains the first args object across calls. Mutate and
    # restore that same object so an initial evaluation cannot pin the task RM
    # as the training reward. The prototype uses the synchronous driver only.
    reward_path = args.custom_rm_path
    postprocess_path = args.custom_reward_post_process_path
    args.custom_rm_path = "open_instruct.miles.opd_hooks.eval_reward"
    args.custom_reward_post_process_path = None
    try:
        return sglang_rollout.generate_rollout(args, rollout_id, data_source, evaluation=True)
    finally:
        args.custom_rm_path = reward_path
        args.custom_reward_post_process_path = postprocess_path
