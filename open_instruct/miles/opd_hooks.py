"""Teacher score validation and task evaluation around upstream sampled-token OPD."""

import asyncio
import copy
import json
import math
import os
from pathlib import Path

import aiohttp
from miles.rollout import on_policy_distillation, sglang_rollout

from open_instruct.ground_truth_utils import GSM8KVerifier

_LIMIT = None


async def reward(args, sample, **kwargs):
    global _LIMIT
    if _LIMIT is None:
        _LIMIT = asyncio.Semaphore(int(os.environ.get("OI_OPD_TEACHER_CONCURRENCY", "4")))
    async with _LIMIT:
        for attempt in range(3):
            try:
                result = await on_policy_distillation.reward_func(args, sample, **kwargs)
                scores = on_policy_distillation._teacher_sampled_log_probs(result, sample.response_length)
                if len(scores) != sample.response_length or not all(math.isfinite(x) for x in scores.tolist()):
                    raise ValueError("Teacher scores are missing, misaligned or nonfinite")
                return result
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt == 2:
                    raise
                await asyncio.sleep(attempt + 1)


def post_process(args, samples, **kwargs):
    result = on_policy_distillation.post_process_rewards(args, samples, **kwargs)
    path = Path(os.environ["OI_OPD_OUTPUT"]) / "teacher-scores.jsonl"
    with path.open("a") as stream:
        for sample in samples:
            values = sample.teacher_log_probs.tolist()
            if len(values) != sample.response_length or not values or not all(map(math.isfinite, values)):
                raise ValueError("Invalid response-aligned teacher signal")
            stream.write(
                json.dumps(
                    {
                        "sample_index": sample.index,
                        "tokens": sample.tokens,
                        "response_length": sample.response_length,
                        "teacher_log_probs": values,
                        "response": sample.response,
                    }
                )
                + "\n"
            )
    return result


async def eval_reward(args, sample, **kwargs):
    return float(GSM8KVerifier()([], sample.response, sample.label).score)


def evaluate(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("OPD task evaluation is evaluation-only")
    eval_args = copy.copy(args)
    eval_args.custom_rm_path = "open_instruct.miles.opd_hooks.eval_reward"
    eval_args.custom_reward_post_process_path = None
    return sglang_rollout.generate_rollout(eval_args, rollout_id, data_source, evaluation=True)
