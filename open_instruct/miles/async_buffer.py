"""MILES async buffer with homogeneous groups and an optimizer-step lag budget."""

import time
from copy import copy
from dataclasses import is_dataclass, replace

from miles.rollout.fully_async_data_buffer import DefaultDataBuffer, iter_samples

from open_instruct.miles import policy_refresh
from open_instruct.miles.data import policy_versions
from open_instruct.miles.queue_metrics import QueueMetrics


class MeasuredDataBuffer(DefaultDataBuffer):
    def __init__(self, input):
        if not hasattr(DefaultDataBuffer, "on_dequeue"):
            raise RuntimeError("Completed-queue metrics require the pinned MILES runtime with on_dequeue support")
        super().__init__(input)
        self._queue_metrics = QueueMetrics()
        self._consumer_wait_seconds = 0.0

    def on_dequeue(self, entry, *, staleness, accepted):
        self._queue_metrics.record(
            [sample.response_length for sample in iter_samples(entry.group)], age=staleness, accepted=accepted
        )

    async def get(self, **context):
        started = time.monotonic()
        try:
            return await super().get(**context)
        finally:
            self._consumer_wait_seconds += time.monotonic() - started

    def get_metrics(self):
        waited, self._consumer_wait_seconds = self._consumer_wait_seconds, 0.0
        return {
            **super().get_metrics(),
            **self._queue_metrics.collect(),
            "rollout/fully_async/completed_queue/consumer_wait_seconds": waited,
        }


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
        self._delegate = MeasuredDataBuffer(delegate_input)
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

    async def reserve_drain_capacity(self, additional):
        """Allow only already-owned completions through a quiescent lifecycle barrier."""
        if type(additional) is not int or additional < 0:
            raise ValueError("drain capacity must be nonnegative")
        delegate = self._delegate
        async with delegate._cond:
            original = delegate._capacity
            delegate._capacity = max(original, len(delegate._buffer)) + additional
            delegate._cond.notify_all()
        return original

    async def restore_capacity(self, capacity):
        async with self._delegate._cond:
            self._delegate._capacity = capacity
            self._delegate._cond.notify_all()


class RefreshPolicyDataBuffer(HomogeneousPolicyDataBuffer):
    """Keep complete mixed-policy groups with validated token-level provenance.

    The delegate still uses the oldest behavior version for the lag limit and
    FIFO consumption. A refreshed suffix never conceals an old prefix.
    """

    async def put(self, item):
        samples = list(iter_samples(item.group))
        if len(item.group) != self._samples or len(samples) != self._samples:
            raise ValueError("Policy refresh requires complete single-turn prompt groups")
        if len({sample.group_index for sample in samples}) != 1:
            raise ValueError("Policy refresh cannot combine unrelated prompt groups")
        policy_refresh.validate_batch(
            {
                "metadata": [s.train_metadata for s in samples],
                "response_lengths": [s.response_length for s in samples],
                "weight_versions": [s.weight_versions for s in samples],
            }
        )
        await self._delegate.put(item)
