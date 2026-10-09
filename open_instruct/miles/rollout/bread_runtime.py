"""Opt-in synchronous BREAD collection using the pinned MILES generation interface."""

import asyncio
import copy
import importlib
import json
import os
import time
from pathlib import Path

from miles.backends.core_utils.rollout.errors import GenerationInterrupted
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.filter_hub import base_types
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils import dumper_utils
from miles.utils.function_registry import load_function

from open_instruct import logger_utils
from open_instruct.miles.rollout import bread, bread_evaluation, bread_observation

logger = logger_utils.setup_logger(__name__)


class BreadRolloutFn(inference_rollout_common.InferenceRolloutFn):
    """Keep expert prefixes as prompt context and train only sampled continuations."""

    async def _call_eval(self, input):
        state = input.generate_state or self.state
        if not state.args.olmo_core.bread_eval_panel_size:
            return await super()._call_eval(input)
        native = importlib.import_module("miles.rollout.inference_rollout.inference_rollout_eval")
        base = importlib.import_module("miles.rollout.base_types")

        async def run(config):
            return await native.eval_rollout_single_dataset(
                bread_evaluation.evaluation_state(state), config, self.eval_prompt_dataset_cache
            )

        initial = (
            input.rollout_id == 0 and state.args.start_rollout_id == 0 and not getattr(self, "_bread_evaluated", False)
        )
        data, metrics = await bread_evaluation.evaluate(state.args, input, run, initial=initial)
        self._bread_evaluated = True
        return base.RolloutFnEvalOutput(data=data, metrics=metrics)

    async def _call_train(self, input):
        args = self.state.args
        await dumper_utils.configure_sglang(args)
        dynamic_filter = load_function(args.dynamic_sampling_filter_path)
        accepted = {}
        events = {}
        collection_started = time.monotonic()
        totals = {
            "bread/seen_groups": 0,
            "bread/zero_groups": 0,
            "bread/rescued_groups": 0,
            "bread/generated_tokens": 0,
            "bread/probe_groups": 0,
            "bread/dropped_groups": 0,
        }

        async def generate(group):
            submitted = time.time()
            for sample in group:
                sample.metadata = {**(sample.metadata or {}), "bread_request_submitted_unix": submitted}
            result = await inference_rollout_common.generate_and_rm_group(
                self.state, group, self.state.sampling_params.copy(), evaluation=False
            )
            if self.state.aborted or any(sample.status.value in ("aborted", "failed") for sample in result):
                raise GenerationInterrupted("BREAD generation interrupted; retry from pristine questions")
            totals["bread/generated_tokens"] += sum(sample.response_length for sample in result)
            return result

        context = min(
            args.olmo_core.packing_max_tokens or args.olmo_core.max_sequence_length,
            args.rollout_max_context_len or args.olmo_core.max_sequence_length,
        )

        def fits(prompt):
            length = len(self.state.tokenizer.encode(prompt, add_special_tokens=False))
            return length + args.rollout_max_response_len <= context

        async def process(sequence, group, queued):
            started = time.monotonic()
            started_unix = time.time()
            pristine = copy.deepcopy(group)
            initial = await generate(group)
            initial_finished = time.monotonic()
            scores = bread.rewards(initial, args.reward_key)
            search_started_unix = time.time()
            if scores is not None and not any(scores):
                self._record(
                    {
                        "event": "zero_group_ready",
                        "rollout_id": input.rollout_id,
                        "sequence": sequence,
                        "sample_id": (pristine[0].metadata or {}).get("prepared_sample_id"),
                        "weight_version": input.weight_version,
                        "ready_unix": search_started_unix,
                        "failure_kind": bread.failure_kind(initial),
                        "initial_responses": bread.responses(initial, args.reward_key),
                    }
                )
            result = await bread.rescue(
                pristine,
                initial,
                generate,
                reward_key=args.reward_key,
                max_prefixes=args.olmo_core.bread_max_prefixes,
                fits=fits,
            )
            totals["bread/seen_groups"] += 1
            scores = result.event["initial_rewards"]
            totals["bread/zero_groups"] += int(scores is not None and not any(scores))
            totals["bread/rescued_groups"] += int(result.event["outcome"] == "rescued")
            totals["bread/probe_groups"] += len(result.event["attempts"])
            final_scores = bread.rewards(result.group, args.reward_key)
            keep = final_scores is not None and 0 < sum(final_scores) < len(final_scores)
            if keep and dynamic_filter is not None:
                keep = bool(base_types.call_dynamic_filter(dynamic_filter, args, result.group).keep)
            event = {
                **result.event,
                "rollout_id": input.rollout_id,
                "weight_version": input.weight_version,
                "group_index": pristine[0].group_index,
                "sample_id": (pristine[0].metadata or {}).get("prepared_sample_id"),
                "initial_generated_tokens": sum(sample.response_length for sample in initial),
                "schema_version": 2,
                "sequence": sequence,
                "started_unix": started_unix,
                "completed_unix": time.time(),
                "search_started_unix": search_started_unix if scores is not None and not any(scores) else None,
                "collector_queue_seconds": started - queued,
                "elapsed_seconds": time.monotonic() - started,
                "initial_seconds": initial_finished - started,
                "search_seconds": time.monotonic() - initial_finished,
                "selected_responses": bread.responses(result.group, args.reward_key),
                "eligible": keep,
                "accepted": False,
            }
            events[sequence] = event
            self._record({**event, "event": "group_completed"})
            if not keep:
                totals["bread/dropped_groups"] += 1
            return result.group if keep else None

        remaining = args.olmo_core.bread_max_groups_per_batch
        pending = {}
        sequence = 0
        target = args.rollout_batch_size
        original_generate = getattr(self.state, "generate_function", None)
        if original_generate is not None:
            self.state.generate_function = bread_observation.timed_generator(original_generate)
        try:
            while pending or (remaining and len(accepted) < target):
                # Keep admission full until enough eligible groups are known. Drain
                # outstanding work and choose by input order, never completion speed.
                while remaining and len(pending) < args.olmo_core.bread_concurrent_groups and len(accepted) < target:
                    (group,) = self.data_source.get_samples(1)
                    task = asyncio.create_task(process(sequence, group, time.monotonic()))
                    pending[task] = sequence
                    sequence += 1
                    remaining -= 1
                if not pending:
                    break
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=pending.get):
                    identity = pending.pop(task)
                    group = task.result()
                    if group is not None:
                        accepted[identity] = group
                        # Retain at most one batch of token/routing tensors.
                        if len(accepted) > target:
                            del accepted[max(accepted)]
            if len(accepted) < target:
                raise RuntimeError("BREAD collection budget exhausted before obtaining enough mixed-reward groups")
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if original_generate is not None:
                self.state.generate_function = original_generate
        batch_ready = time.time()
        for identity, event in events.items():
            event["accepted"] = identity in accepted
            event["batch_ready_unix"] = batch_ready
            event["completion_to_batch_seconds"] = batch_ready - event["completed_unix"]
            event["arrival_to_batch_seconds"] = batch_ready - event["started_unix"]
            self._record({"event": "batch_admission", **event})
            if identity in accepted:
                for sample in accepted[identity]:
                    sample.metadata["bread_collection"] = {
                        "route": "bread" if event["outcome"] == "rescued" else "ordinary",
                        "failure_kind": event.get("failure_kind"),
                        "started_unix": event["started_unix"],
                        "completed_unix": event["completed_unix"],
                        "batch_ready_unix": batch_ready,
                        "weight_version": input.weight_version,
                    }
        totals.update(bread_observation.summarize(list(events.values())))
        totals["bread/collection_seconds"] = time.monotonic() - collection_started
        totals["bread/dropped_groups"] = len(events) - len(accepted)
        self.data_source.add_samples([])
        self.state.reset()
        groups = [accepted[key] for key in sorted(accepted)]
        return RolloutFnTrainOutput(samples=groups, metrics=totals)

    def _record(self, event):
        logger.info("BREAD group: %s", json.dumps(event, sort_keys=True))
        root = self.state.args.save
        if root:
            path = Path(root) / "bread" / "groups.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
