"""Fresh stop/continue comparisons and masked conditional answer training."""

import dataclasses
import json
import random
import time
import uuid
from pathlib import Path

from miles.backends.core_utils.publication import policy_versions
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.generate_utils import generate_endpoint_utils
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.utils.types import Sample

from open_instruct import logger_utils
from open_instruct.miles.rollout import forced_exits, stopping_comparison

logger = logger_utils.setup_logger(__name__)


async def generate_continuation(input):
    """Mask closing for exactly one token; this constrained token is not trained."""
    sample = input.sample
    before = sample.response_length
    blocked = sample.metadata["comparative_continue_ids"]
    params = dict(input.sampling_params, max_new_tokens=before + 1, logit_bias={str(i): -1e9 for i in blocked})
    output = await input.state.generate_function(dataclasses.replace(input, sampling_params=params))
    sample = output.samples
    if sample.response_length != before + 1:
        raise RuntimeError("Continue intervention must generate exactly one token")
    if sample.tokens[-1] in blocked:
        raise RuntimeError("Closing-token suppression failed")
    sample.loss_mask = [0] * sample.response_length
    # EOS after the first token is a genuine terminal outcome, not a request to resume.
    if sample.status == Sample.Status.COMPLETED:
        return output
    sample.status = Sample.Status.PENDING
    return await input.state.generate_function(dataclasses.replace(input, sample=sample))


def branch_record(sample, advantage, denominator):
    routes = sample.rollout_routed_experts
    return {
        "tokens": sample.tokens,
        "loss_mask": sample.loss_mask,
        "log_probs": sample.rollout_log_probs,
        "advantage": advantage,
        "group_denominator": denominator,
        "routed_experts": routes.tolist() if routes is not None else None,
    }


class ComparativeExitRollout(forced_exits.ForcedExitRollout):
    """Keep natural groups unchanged; attach separately normalized auxiliary data."""

    def __init__(self, input):
        super().__init__(input)
        tokenizer = self.state.tokenizer
        self.close_ids = tokenizer.encode(forced_exits.CLOSING_TEXT, add_special_tokens=False)
        self.blocked_ids = sorted(
            token
            for token in set(tokenizer.get_vocab().values())
            if "</" in tokenizer.decode([token], skip_special_tokens=False)
        )
        if not self.close_ids or self.close_ids[0] not in self.blocked_ids:
            raise ValueError("Cannot identify the tokenizer's closing decision")

    async def _branch(self, parent, pristine, cut, action, trial, step, position):
        args, state = self.state.args, self.state
        sample = forced_exits.prepare_branch(
            parent, pristine, state.tokenizer, cut, self.close_ids if action == "stop" else [], position, trial
        )
        # Each intervention gets its own engine routing key; the barrier pins weights.
        if generate_endpoint_utils.policy_uses_routing_key(args):
            sample.routing_key = str(uuid.uuid4())
        sample.index = -(1 + parent.index * 1_000_000 + position * 10000 + (action == "continue") * 1000 + trial)
        if action == "continue":
            sample.metadata["comparative_continue_ids"] = self.blocked_ids
            sample.generate_function_path = "open_instruct.miles.rollout.comparative_exits.generate_continuation"
        before = sample.response_length
        started = time.perf_counter()
        # Both branches share the original total response cap, including the intervention.
        params = dict(state.sampling_params, max_new_tokens=args.rollout_max_response_len)
        # Deterministic distinct seeds preserve paired inputs across the two treatment arms.
        params["sampling_seed"] = (args.rollout_seed + step * 1000003 + abs(sample.index)) % (2**31)
        sample = await common.generate_and_rm(state, sample, params)
        if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
            raise RuntimeError("Comparative branch did not finish")
        active_start = before + (action == "continue")
        if sample.response_length < active_start:
            raise RuntimeError("Comparative branch ended before its intervention")
        sample.loss_mask = [0] * active_start + [1] * (sample.response_length - active_start)
        if len(sample.rollout_log_probs) != sample.response_length:
            raise RuntimeError("Branch behavior scores do not cover its response")
        sample.metadata["probe_seconds"] = time.perf_counter() - started
        return sample

    async def _score(self, parent, pristine, cut, step, position, *, audit=False):
        args, core = self.state.args, self.state.args.olmo_core
        n = core.forced_exit_initial_trials if step < core.forced_exit_initial_updates else core.forced_exit_trials
        samples = await forced_exits.gather_complete(
            [
                self._branch(parent, pristine, cut, action, i, step, position)
                for action in ("stop", "continue")
                for i in range(n)
            ]
        )
        if len({v for s in [parent, *samples] for v in policy_versions.versions(s.weight_versions)}) != 1:
            raise RuntimeError("Stop and continue samples crossed a policy publication")
        stop, continuation = samples[:n], samples[n:]
        rewards = [[s.get_reward_value(args) for s in group] for group in (stop, continuation)]
        lengths = [[s.response_length - cut for s in group] for group in (stop, continuation)]
        result = stopping_comparison.comparison(
            *rewards,
            *lengths,
            tie_bonus=core.forced_exit_tie_bonus,
            tie_min_accuracy=core.forced_exit_tie_min_accuracy,
        )
        training = []
        for group, outcomes in zip((stop, continuation), rewards, strict=True):
            training.extend(
                branch_record(s, a, 2 * n)
                for s, a in zip(group, stopping_comparison.centered_rewards(outcomes), strict=True)
            )
        prompt_length = len(parent.tokens) - parent.response_length
        end = forced_exits.thinking_end(parent.tokens[prompt_length:], self.state.tokenizer)
        return {
            **result,
            "cut": cut,
            "close_ids": self.close_ids,
            "guidance": "first_token",
            "rewards": rewards[0],
            "continue_rewards": rewards[1],
            "stop_lengths": lengths[0],
            "continue_lengths": lengths[1],
            "training_branches": training,
            "audit": audit,
            "parent_weight": args.n_samples_per_prompt,
            "parent_reward": parent.get_reward_value(args),
            "parent_status": parent.status.value,
            "relative_position": cut / max(1, end),
            "probe_seconds": sum(s.metadata["probe_seconds"] for s in samples),
            "generated_tokens": sum(sum(values) for values in lengths) - n * len(self.close_ids),
        }

    async def _probe_parent(self, parent, pristine, step, rng, *, audit=False):
        args, core = self.state.args, self.state.args.olmo_core
        tokens = parent.tokens[len(parent.tokens) - parent.response_length :]
        end = forced_exits.thinking_end(tokens, self.state.tokenizer)
        limit = args.rollout_max_response_len - len(self.close_ids) - 1
        if audit:
            if end == len(tokens) or not 0 < end <= limit:
                return
            probes = [await self._score(parent, pristine, end, step, 90, audit=True)]
            parent.train_metadata["stopping_probes"].extend(probes)
            return
        candidates = forced_exits.uniform_positions(
            tokens, core.forced_exit_screen_positions, limit, self.state.tokenizer
        )
        if not candidates:
            return
        screens = await forced_exits.gather_complete(
            [
                self._branch(parent, pristine, cut, "stop", trial, step, 20 + i)
                for i, cut in enumerate(candidates)
                for trial in range(core.forced_exit_screen_trials)
            ]
        )
        values = [
            sum(
                s.get_reward_value(args)
                for s in screens[i * core.forced_exit_screen_trials : (i + 1) * core.forced_exit_screen_trials]
            )
            for i in range(len(candidates))
        ]
        # Prefer the earliest best screen, but score it with independent seeds/samples.
        selected = [candidates[max(range(len(candidates)), key=lambda i: float(values[i]))]]
        others = [cut for cut in candidates if cut not in selected]
        if others:
            selected.append(rng.choice(others))
        probes = await forced_exits.gather_complete(
            [self._score(parent, pristine, cut, step, i) for i, cut in enumerate(selected)]
        )
        parent.train_metadata["stopping_probes"].extend(probes)
        parent.train_metadata["screen_tokens"] = sum(
            s.response_length - s.metadata["forced_exit"]["cut"] - len(self.close_ids) for s in screens
        )
        parent.train_metadata["screen_seconds"] = sum(s.metadata["probe_seconds"] for s in screens)

    def _controller(self, step):
        core = self.state.args.olmo_core
        root = Path(self.state.args.save) / "forced-exits-v2" if self.state.args.save else None
        rate = stopping_comparison.tapered_rate(
            step,
            core.forced_exit_parent_probability,
            core.forced_exit_halving_interval,
            core.forced_exit_floor_fraction,
        )
        audits = []
        if root and step > 0:
            previous = root / f"{step - 1}.json"
            if previous.exists():
                record = json.loads(previous.read_text())
                rate = record["next_rate"]
                audits = [a for a in record["audits"] if a["step"] >= step - core.forced_exit_audit_window]
        return root, rate, audits

    async def _call_train(self, input):
        args, core, step = self.state.args, self.state.args.olmo_core, input.rollout_id
        root, rate, audits = self._controller(step)
        rng = random.Random(args.rollout_seed + step * 104729)
        pristine_groups = self.data_source.get_samples(args.rollout_batch_size)
        groups = await forced_exits.gather_complete(
            [
                forced_exits.gather_complete([self._parent(s, probe=False) for s in pristine])
                for pristine in pristine_groups
            ]
        )
        chosen = [i for i in range(len(groups)) if rng.random() < rate]
        rng.shuffle(chosen)
        chosen = chosen[: core.forced_exit_max_parents_per_update]
        work = []
        for i in chosen:
            weights = stopping_comparison.parent_weights(
                [
                    forced_exits.thinking_end(s.tokens[len(s.tokens) - s.response_length :], self.state.tokenizer)
                    for s in groups[i]
                ],
                core.forced_exit_uniform_share,
                core.forced_exit_length_cap,
            )
            j = rng.choices(range(len(groups[i])), weights=weights)[0]
            work.append(
                self._probe_parent(groups[i][j], pristine_groups[i][j], step, random.Random(rng.getrandbits(64)))
            )
        await forced_exits.gather_complete(work)
        # Uniform audit acquisition is independent of parent length, reward and training rate.
        audit_round = step % core.forced_exit_audit_interval == 0
        if audit_round:
            by_task = {}
            for i, group in enumerate(groups):
                task = str(group[0].metadata.get("prepared_sample_id", "unknown")).split(":")[0]
                by_task.setdefault(task, []).append(i)
            for indices in by_task.values():
                i, j = rng.choice(indices), rng.randrange(args.n_samples_per_prompt)
                await self._probe_parent(groups[i][j], pristine_groups[i][j], step, rng, audit=True)
        records = []
        for group in groups:
            for parent in group:
                task = str(parent.metadata.get("prepared_sample_id", "unknown")).split(":")[0]
                for p in parent.train_metadata["stopping_probes"]:
                    record = {k: v for k, v in p.items() if k != "training_branches"}
                    records.append({**record, "parent_index": parent.index, "task": task})
                    if p["audit"]:
                        audits.append({"step": step, "task": task, "delta": p["accuracy_advantage"]})
        risky = any(
            stopping_comparison.risk_detected(
                [a["delta"] for a in audits if a["task"] == task],
                minimum_parents=core.forced_exit_risk_min_parents,
                margin=core.forced_exit_risk_margin,
            )
            for task in {a["task"] for a in audits}
        )
        next_rate = stopping_comparison.advance_rate(
            rate,
            step + 1,
            initial=core.forced_exit_parent_probability,
            interval=core.forced_exit_halving_interval,
            floor_fraction=core.forced_exit_floor_fraction,
            risk=risky,
        )
        if risky and not audit_round:
            # Existing risk suspends tapering; only a fresh audit doubles the rate.
            next_rate = rate
        if root:
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{step}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"rate": rate, "next_rate": next_rate, "audits": audits, "cuts": records}, allow_nan=False)
            )
            temporary.replace(path)
        parents = [p for group in groups for p in group]
        metrics = {
            "forced_exit/parent_probability": rate,
            "forced_exit/next_probability": next_rate,
            "forced_exit/risk": int(risky),
            "forced_exit/probed_positions": len(records),
            "forced_exit/tie_bonuses": sum(p["tie_bonus"] > 0 for p in records),
            "forced_exit/probe_tokens": sum(p["generated_tokens"] for p in records),
            "forced_exit/probe_seconds": sum(p["probe_seconds"] for p in records),
            "forced_exit/screen_tokens": sum(p.train_metadata.get("screen_tokens", 0) for p in parents),
            "forced_exit/screen_seconds": sum(p.train_metadata.get("screen_seconds", 0) for p in parents),
            "forced_exit/natural_seconds": sum(p.train_metadata["natural_seconds"] for p in parents),
        }
        self.state.reset()
        logger.info("Comparative stopping collection %s: %s", step, metrics)
        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
