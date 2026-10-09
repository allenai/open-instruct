"""Opt-in synchronous BREAD collection using the pinned MILES generation interface."""

import copy
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
from open_instruct.miles.rollout import bread

logger = logger_utils.setup_logger(__name__)


class BreadRolloutFn(inference_rollout_common.InferenceRolloutFn):
    """Keep expert prefixes as prompt context and train only sampled continuations."""

    async def _call_train(self, input):
        args = self.state.args
        await dumper_utils.configure_sglang(args)
        dynamic_filter = load_function(args.dynamic_sampling_filter_path)
        accepted = []
        totals = {
            "bread/seen_groups": 0,
            "bread/zero_groups": 0,
            "bread/rescued_groups": 0,
            "bread/generated_tokens": 0,
            "bread/probe_groups": 0,
            "bread/dropped_groups": 0,
        }

        async def generate(group):
            result = await inference_rollout_common.generate_and_rm_group(
                self.state, group, self.state.sampling_params.copy(), evaluation=False
            )
            if self.state.aborted or any(sample.status.value in ("aborted", "failed") for sample in result):
                raise GenerationInterrupted("BREAD generation interrupted; retry from pristine questions")
            totals["bread/generated_tokens"] += sum(sample.response_length for sample in result)
            return result

        context = min(
            args.olmo_core.max_sequence_length, args.rollout_max_context_len or args.olmo_core.max_sequence_length
        )

        def fits(prompt):
            length = len(self.state.tokenizer.encode(prompt, add_special_tokens=False))
            return length + args.rollout_max_response_len <= context

        for _ in range(args.olmo_core.bread_max_groups_per_batch):
            started = time.monotonic()
            [group] = self.data_source.get_samples(1)
            pristine = copy.deepcopy(group)
            initial = await generate(group)
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
            # Unfinished or nonbinary groups cannot supply the intended curriculum signal.
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
                "elapsed_seconds": time.monotonic() - started,
                "accepted": keep,
            }
            self._record(event)
            if keep:
                accepted.append(result.group)
            else:
                totals["bread/dropped_groups"] += 1
            if len(accepted) == args.rollout_batch_size:
                break
        else:
            raise RuntimeError("BREAD collection budget exhausted before obtaining enough mixed-reward groups")
        self.data_source.add_samples([])
        self.state.reset()
        accepted.sort(key=lambda group: group[0].index)
        return RolloutFnTrainOutput(samples=accepted, metrics=totals)

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
