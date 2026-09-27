"""Uniform within-trace forced-exit groups, initially synchronous/barrier only."""

import asyncio
import copy
import json
import random
import re
import time
import uuid
from pathlib import Path

from miles.backends.core_utils.publication import policy_versions
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.generate_utils import generate_endpoint_utils
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.types import Sample

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

CLOSING_TEXT = "</think>\n\n"


def sequence_start(tokens, sequence):
    """Find a delimiter without decoding/re-encoding the parent prefix."""
    for index in range(len(tokens) - len(sequence) + 1):
        if tokens[index : index + len(sequence)] == sequence:
            return index
    return None


def thinking_end(tokens, tokenizer):
    # BPE can merge the tag with adjacent punctuation/newlines, so matching only
    # tokenizer.encode("</think>") misses naturally generated closing tags.
    pieces = [tokenizer.decode([token], skip_special_tokens=False) for token in tokens]
    close = "".join(pieces).find("</think>")
    if close < 0:
        return len(tokens)
    offset = 0
    for index, piece in enumerate(pieces):
        offset += len(piece)
        if offset > close:
            return index
    return len(tokens)


def uniform_positions(tokens, close_ids, count, max_prefix, tokenizer):
    """Spread cuts over actual paragraph boundaries, preserving original token IDs."""
    end = min(thinking_end(tokens, tokenizer), max_prefix)
    candidates, tail = [], ""
    for index, token in enumerate(tokens[: max(0, end)]):
        piece = tokenizer.decode([token], skip_special_tokens=False)
        tail = (tail + piece)[-64:]
        if "\n" in piece and re.search(r"\n[ \t\r]*\n[ \t\r]*$", tail):
            candidates.append(index + 1)
    if len(candidates) <= count:
        return candidates
    # Interior quantiles, without duplicated positions or fabricated boundaries.
    indices = [min(len(candidates) - 1, int(len(candidates) * (i + 1) / (count + 1))) for i in range(count)]
    return [candidates[index] for index in indices]


def cut_advantages(parent_reward, probes):
    """A separate per-parent comparison; never alter natural GRPO rewards."""
    outcomes = [r for probe in probes for r in probe["rewards"]]
    if not outcomes:
        return
    baseline = (parent_reward + sum(outcomes)) / (1 + len(outcomes))
    for probe in probes:
        probe["advantage"] = sum(probe["rewards"]) / len(probe["rewards"]) - baseline
        probe["cut_group_mean"] = baseline


def has_signal(group, args):
    rewards = [s.get_reward_value(args) for s in group]
    return min(rewards) != max(rewards)


def prepare_branch(parent, pristine, tokenizer, cut, close_ids, position, trial):
    sample = copy.deepcopy(pristine)
    prompt_length = len(parent.tokens) - parent.response_length
    prefix = parent.tokens[prompt_length : prompt_length + cut]
    sample.tokens = parent.tokens[:prompt_length] + prefix + close_ids
    sample.response_length = cut + len(close_ids)
    sample.response = tokenizer.decode(prefix + close_ids, skip_special_tokens=False)
    sample.loss_mask = [0] * sample.response_length
    # Historical prefix probabilities are retained but masked. Forced tokens have
    # intervention probability one, NOT the model's probability of those tokens.
    sample.rollout_log_probs = parent.rollout_log_probs[:cut] + [0.0] * len(close_ids)
    sample.weight_versions = []  # the new call records its current immutable barrier version
    sample.status = Sample.Status.PENDING
    sample.reward = None
    info = {
        "parent_index": parent.index,
        "parent_reward": parent.reward,
        "parent_status": parent.status.value,
        "prompt_id": parent.metadata.get("prepared_sample_id"),
        "cut": cut,
        "close_end": sample.response_length,
        "position": position,
        "trial": trial,
    }
    sample.metadata["forced_exit"] = info
    sample.train_metadata = {"forced_exit": info}
    return sample


async def gather_complete(coroutines):
    """An exception must not leave sibling requests running across publication."""
    tasks = [asyncio.create_task(c) for c in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class ForcedExitRollout(common.InferenceRolloutFn):
    """Natural GRPO groups; per-parent forced answers supply auxiliary labels only."""

    async def _parent(self, pristine, probe=True, parent_weight=1):
        state, args = self.state, self.state.args
        request = copy.deepcopy(pristine)
        if generate_endpoint_utils.policy_uses_routing_key(args):
            request.routing_key = str(uuid.uuid4())
        started = time.perf_counter()
        parent = await common.generate_and_rm(state, request, state.sampling_params.copy())
        parent.train_metadata = {"stopping_probes": [], "natural_seconds": time.perf_counter() - started}
        if parent.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
            raise RuntimeError("Forced-exit parent did not finish")
        if not probe:
            return parent
        # Match the natural answer boundary. BPE merges the final '>' with
        # trailing newlines; forcing a bare '>' creates a different, rare path.
        close_ids = state.tokenizer.encode(CLOSING_TEXT, add_special_tokens=False)
        if not close_ids:
            raise ValueError("The checkpoint tokenizer must encode a nonempty </think> delimiter")
        response_ids = parent.tokens[len(parent.tokens) - parent.response_length :]
        positions = uniform_positions(
            response_ids,
            close_ids,
            args.olmo_core.forced_exit_positions,
            args.rollout_max_response_len - len(close_ids) - args.olmo_core.forced_exit_answer_tokens,
            state.tokenizer,
        )

        async def branch(cut, position, trial):
            sample = prepare_branch(parent, pristine, state.tokenizer, cut, close_ids, position, trial)
            sample.index = -(
                1
                + parent.index * args.olmo_core.forced_exit_positions * args.olmo_core.forced_exit_trials
                + position * args.olmo_core.forced_exit_trials
                + trial
            )
            if generate_endpoint_utils.policy_uses_routing_key(args):
                sample.routing_key = str(uuid.uuid4())
            prefilled = sample.response_length
            params = state.sampling_params.copy()
            # single_turn subtracts the prefilled response from max_new_tokens.
            params["max_new_tokens"] = prefilled + args.olmo_core.forced_exit_answer_tokens
            started = time.perf_counter()
            sample = await common.generate_and_rm(state, sample, params)
            sample.metadata["probe_seconds"] = time.perf_counter() - started
            if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
                raise RuntimeError("Forced-exit branch did not finish")
            if sample.response_length == prefilled:
                raise RuntimeError("Forced-exit branch produced no sampled answer tokens")
            return sample

        branches = await gather_complete(
            [
                branch(cut, p, trial)
                for p, cut in enumerate(positions)
                for trial in range(args.olmo_core.forced_exit_trials)
            ]
        )
        versions = {v for s in [parent, *branches] for v in policy_versions.versions(s.weight_versions)}
        if len(versions) != 1:
            raise RuntimeError("Forced-exit group crossed a policy publication; barrier provenance is required")
        probes = []
        for position, cut in enumerate(positions):
            trials = branches[
                position * args.olmo_core.forced_exit_trials : (position + 1) * args.olmo_core.forced_exit_trials
            ]
            probes.append(
                {
                    "parent_index": parent.index,
                    "prompt_id": parent.metadata.get("prepared_sample_id"),
                    "cut": cut,
                    "close_ids": close_ids,
                    "position": position,
                    "thinking_length": thinking_end(response_ids, state.tokenizer),
                    "relative_position": cut / max(1, thinking_end(response_ids, state.tokenizer)),
                    "parent_weight": parent_weight,
                    "parent_final_answer": parent.response.rpartition("</think>")[2],
                    "probe_seconds": sum(s.metadata["probe_seconds"] for s in trials),
                    "parent_reward": parent.get_reward_value(args),
                    "parent_status": parent.status.value,
                    "rewards": [s.get_reward_value(args) for s in trials],
                    "truncated": [s.status == Sample.Status.TRUNCATED for s in trials],
                    "answer_lengths": [s.response_length - cut - len(close_ids) for s in trials],
                    "final_answers": [s.response.rpartition("</think>")[2] for s in trials],
                }
            )
        cut_advantages(parent.get_reward_value(args), probes)
        parent.train_metadata["stopping_probes"] = probes
        return parent

    async def _group(self, group):
        args = self.state.args
        count = min(args.olmo_core.forced_exit_parents, len(group))
        chosen = set(random.Random(args.rollout_seed + group[0].index).sample(range(len(group)), count))
        parents = await gather_complete(
            [self._parent(sample, index in chosen, len(group) / count) for index, sample in enumerate(group)]
        )
        versions = {v for sample in parents for v in policy_versions.versions(sample.weight_versions)}
        if len(versions) != 1:
            raise RuntimeError("Natural group crossed a policy publication")
        return parents

    async def _call_train(self, input):
        args = self.state.args
        accepted, attempted, rejected = [], 0, 0
        records = []
        while len(accepted) < args.rollout_batch_size:
            groups = self.data_source.get_samples(args.rollout_batch_size - len(accepted))
            completed = await gather_complete([self._group(group) for group in groups])
            for group in completed:
                attempted += 1
                keep = not args.olmo_core.filter_zero_std_groups or has_signal(group, args)
                records.append(
                    {
                        "group_index": group[0].group_index,
                        "prompt_id": group[0].metadata.get("prepared_sample_id"),
                        "natural_rewards": [s.get_reward_value(args) for s in group],
                        "accepted": keep,
                        "natural_signal": has_signal(group, args),
                        "stopping_signal": any(
                            p["advantage"] != 0 for s in group for p in s.train_metadata["stopping_probes"]
                        ),
                        "natural_seconds": sum(s.train_metadata["natural_seconds"] for s in group),
                        "parents": [
                            {
                                "index": s.index,
                                "length": s.response_length,
                                "status": s.status.value,
                                "probes": s.train_metadata["stopping_probes"],
                            }
                            for s in group
                        ],
                    }
                )
                if keep:
                    accepted.append(group)
                else:
                    rejected += 1
            logger.info(
                "Forced-exit collection %s: accepted=%s attempted=%s", input.rollout_id, len(accepted), attempted
            )
            if attempted >= args.rollout_batch_size * 100 and len(accepted) < args.rollout_batch_size:
                raise RuntimeError("Unable to fill a forced-exit collection with natural or auxiliary signal")
        if args.save:
            root = Path(args.save) / "forced-exits"
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{input.rollout_id}.json").write_text(json.dumps(records, allow_nan=False))
        probes = [p for r in records for parent in r["parents"] for p in parent["probes"]]
        trial_count = sum(len(p["rewards"]) for p in probes)
        self.state.reset()
        return RolloutFnTrainOutput(
            samples=accepted,
            metrics={
                "forced_exit/natural_signal_groups": sum(r["accepted"] and r["natural_signal"] for r in records),
                "forced_exit/stopping_only_groups": sum(
                    r["accepted"] and not r["natural_signal"] and r["stopping_signal"] for r in records
                ),
                "forced_exit/no_signal_groups": sum(
                    r["accepted"] and not r["natural_signal"] and not r["stopping_signal"] for r in records
                ),
                "forced_exit/natural_request_seconds": sum(r["natural_seconds"] for r in records),
                "forced_exit/probe_request_seconds": sum(p["probe_seconds"] for p in probes),
                "forced_exit/attempted_groups": attempted,
                "forced_exit/rejected_groups": rejected,
                "forced_exit/probed_positions": len(probes),
                "forced_exit/trials": trial_count,
                "forced_exit/truncation_rate": sum(sum(p["truncated"]) for p in probes) / max(1, trial_count),
                "forced_exit/exit_reward": sum(sum(p["rewards"]) for p in probes) / max(1, trial_count),
            },
        )
