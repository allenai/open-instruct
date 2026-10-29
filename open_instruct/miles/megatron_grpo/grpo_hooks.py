"""Native Megatron GRPO hooks backed by the registered Open Instruct verifiers.

``reward`` is MILES' ``--custom-rm-path``: it scores through ``rewards.registered_reward`` (the
same registry, ``metadata.verifiers`` and weighted sum as the Core backend). ``post_process`` is
``--custom-reward-post-process-path``: it records the raw verifier rewards and centers them within
each complete prompt group, which MILES' GRPO estimator then broadcasts over response tokens.
Samples a verifier could not grade (``rewards.EXCLUDED_KEY``), and length-truncated samples when
``reward.exclude_truncated`` is set, get ``remove_sample``, a zero advantage, and no part in their
group's baseline.
Settings arrive as ``OI_GRPO_*`` environment from ``grpo_runtime``.
"""

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.megatron_grpo import domain_balance
from open_instruct.miles.rewards import rewards

OUTPUT_ENV = "OI_GRPO_OUTPUT"
REWARD_CONFIG_ENV = "OI_GRPO_REWARD_CONFIG"
ZERO_TRUNCATED_ENV = "OI_GRPO_REWARD_ZERO_TRUNCATED"
FINAL_ANSWER_ONLY_ENV = "OI_GRPO_REWARD_FINAL_ANSWER_ONLY"
EXCLUDE_TRUNCATED_ENV = "OI_GRPO_REWARD_EXCLUDE_TRUNCATED"
EVIDENCE = "verifier-rewards.jsonl"
REPLAYED = "verifier-rewards.replayed.jsonl"


def reward_environment(registry, reward_gates):
    """``OI_GRPO_*`` reward settings for the hooks, from the run file's ``[reward]`` section."""
    return {
        REWARD_CONFIG_ENV: str(registry),
        ZERO_TRUNCATED_ENV: "1" if reward_gates["zero_truncated"] else "0",
        FINAL_ANSWER_ONLY_ENV: "1" if reward_gates["final_answer_only"] else "0",
        EXCLUDE_TRUNCATED_ENV: "1" if reward_gates.get("exclude_truncated") else "0",
    }


def reconcile_evidence(root, rollouts, batch):
    """Keep exactly ``rollouts`` complete rollouts of verifier evidence; return how many records moved.

    ``post_process`` appends a rollout's records before that rollout's training is checkpointed, so a
    preempted run leaves records for rollouts the resumed trainer replays. The hook is not told the
    rollout ID, but it writes one fixed-size batch per rollout in rollout order, so the first
    ``rollouts * batch`` lines are the checkpointed rollouts. Later lines, including a partial write,
    move to ``REPLAYED`` for inspection and the replay appends their replacements.
    """
    path = Path(root) / EVIDENCE
    if not path.exists():
        if rollouts:
            raise ValueError("Verifier evidence is missing for checkpointed rollouts")
        return 0
    with path.open("r+b") as stream:
        for _ in range(rollouts * batch):
            if not stream.readline().endswith(b"\n"):
                raise ValueError("Verifier evidence is missing records for checkpointed rollouts")
        boundary = stream.tell()
        tail = stream.read()
        if tail:
            with (Path(root) / REPLAYED).open("ab") as replayed:
                replayed.write(tail if tail.endswith(b"\n") else tail + b"\n")
            stream.truncate(boundary)
    return len(tail.splitlines())


def scoring_args(args=None, environment=None):
    """A copy of ``args`` carrying the ``CoreConfig`` reward fields ``registered_reward`` reads."""
    environment = os.environ if environment is None else environment
    scoring = copy.copy(args) if args is not None else SimpleNamespace()
    scoring.olmo_core = CoreConfig(
        reward_config=environment[REWARD_CONFIG_ENV],
        reward_zero_truncated=environment.get(ZERO_TRUNCATED_ENV, "0") == "1",
        reward_final_answer_only=environment.get(FINAL_ANSWER_ONLY_ENV, "0") == "1",
    )
    return scoring


async def reward(args, sample, **kwargs):
    """Score one sample (``rm_hub.async_rm``) or a list (``rm_hub.batched_async_rm``)."""
    values = await rewards.registered_reward(scoring_args(args), sample)
    for value in values if isinstance(values, list) else [values]:
        if not math.isfinite(value):
            raise ValueError("Nonfinite verifier reward")
    return values


def _status(sample):
    status = getattr(sample, "status", None)
    return getattr(status, "value", status)


def post_process(args, samples, **kwargs):
    """Retain the verifier signal, then center it within prompt groups for native GRPO."""
    values = [float(sample.reward) for sample in samples]
    if not values or not all(map(math.isfinite, values)):
        raise ValueError("Missing or nonfinite verifier rewards")
    if len(samples) % args.n_samples_per_prompt:
        raise ValueError("Incomplete GRPO prompt group")
    path = Path(os.environ[OUTPUT_ENV]) / EVIDENCE
    with path.open("a") as stream:
        for sample, value in zip(samples, values, strict=True):
            if getattr(sample, "teacher_log_probs", None) is not None:
                raise ValueError("Teacher scores must be absent from verifier GRPO")
            stream.write(
                json.dumps(
                    {
                        "sample_index": sample.index,
                        "group_index": getattr(sample, "group_index", None),
                        "tokens": sample.tokens,
                        "response_length": sample.response_length,
                        "response": sample.response,
                        "status": _status(sample),
                        "reward": value,
                        "metadata": sample.metadata,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    groups = {}
    for index, sample in enumerate(samples):
        group = getattr(sample, "group_index", None)
        groups.setdefault(group if group is not None else index // args.n_samples_per_prompt, []).append(index)
    centered = [0.0] * len(samples)
    exclude_truncated = os.environ.get(EXCLUDE_TRUNCATED_ENV, "0") == "1"
    for indices in groups.values():
        if len(indices) != args.n_samples_per_prompt:
            raise ValueError("GRPO requires complete fixed-fanout prompt groups")
        prefixes = [samples[i].tokens[: len(samples[i].tokens) - samples[i].response_length] for i in indices]
        if any(prefix != prefixes[0] for prefix in prefixes):
            raise ValueError("GRPO prompt group contains different tokenized prompts")
        graded = []
        for i in indices:
            truncated = exclude_truncated and _status(samples[i]) == "truncated"
            if not truncated and rewards.excluded_from_training(samples[i]) is None:
                graded.append(i)
            else:
                samples[i].remove_sample = True
        if not graded:
            continue
        mean = sum(values[i] for i in graded) / len(graded)
        denominator = 1.0
        if args.grpo_std_normalization and len(graded) > 1:
            denominator = math.sqrt(sum((values[i] - mean) ** 2 for i in graded) / (len(graded) - 1)) + 1e-6
        for i in graded:
            centered[i] = (values[i] - mean) / denominator
    if domain_balance.enabled():
        domain_balance.attach(samples)
    return values, centered
