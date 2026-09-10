"""MILES async buffer with homogeneous groups and an optimizer-step lag budget."""

from copy import copy
from dataclasses import is_dataclass, replace

from miles.rollout.fully_async_data_buffer import DefaultDataBuffer, iter_samples

from open_instruct.miles.data import policy_versions


class HomogeneousPolicyDataBuffer:
    def __init__(self, input):
        args = copy(input.args)
        collection = args.rollout_batch_size * args.n_samples_per_prompt
        if collection % args.global_batch_size:
            raise ValueError("Rollout collection must contain complete optimizer batches")
        args.max_weight_staleness -= collection // args.global_batch_size - 1
        if args.max_weight_staleness < 0:
            raise ValueError("Policy lag budget cannot cover the last optimizer step")
        if is_dataclass(input):
            delegate_input = replace(input, args=args)
        else:
            delegate_input = copy(input)
            delegate_input.args = args
        self._delegate = DefaultDataBuffer(delegate_input)
        self._unused = input.unused_handler_fn
        self._samples = args.n_samples_per_prompt
        self._rejected = 0

    async def put(self, item):
        try:
            versions = policy_versions(
                {"weight_versions": [sample.weight_versions for sample in iter_samples(item.group)]}
            )
            valid = len(item.group) == self._samples and len(set(versions)) == 1
        except ValueError:
            valid = False
        if valid:
            await self._delegate.put(item)
        else:
            self._rejected += 1
            self._unused(item.prompt_group)

    async def get(self, **context):
        return await self._delegate.get(**context)

    def get_metrics(self):
        rejected, self._rejected = self._rejected, 0
        return {**self._delegate.get_metrics(), "rollout/fully_async/rejected_policy_groups": rejected}
