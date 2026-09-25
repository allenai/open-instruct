"""MILES async buffer with homogeneous groups and an optimizer-step lag budget."""

import time
from copy import copy
from dataclasses import is_dataclass, replace

from miles.rollout.filter_hub.base_types import call_dynamic_filter, iter_samples
from miles.rollout.filter_hub.common_filters import apply_missing_reward_filter
from miles.rollout.fully_async_data_buffer import DefaultDataBuffer
from miles.utils.types import Sample

from open_instruct.miles import inference_records, policy_refresh
from open_instruct.miles.data import policy_versions
from open_instruct.miles.queue_metrics import QueueMetrics


class MeasuredDataBuffer(DefaultDataBuffer):
    def __init__(self, input):
        if not hasattr(DefaultDataBuffer, "on_dequeue"):
            raise RuntimeError("Completed-queue metrics require the pinned MILES runtime with on_dequeue support")
        super().__init__(input)
        # Run the native filter here so the producer can distinguish an intentional
        # drop from an enqueued group and retire its checkpoint retry ledger entry.
        self._group_filter = self._dynamic_filter
        self._dynamic_filter = None
        self._queue_metrics = QueueMetrics()
        self._consumer_wait_seconds = 0.0
        core = getattr(self._args, "olmo_core", None)
        self._records = inference_records.Recorder(self._args) if getattr(core, "records_root", None) else None

    async def put(self, item):
        """Return False only for a dynamic-filter drop that the producer must acknowledge."""
        samples = list(iter_samples(item.group))
        # Preserve upstream abort/retry handling before reward filtering.
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            if self._records is not None:
                self._records.record_group(samples, decision="aborted")
        else:
            output = apply_missing_reward_filter(self._args, item.group)
            if output.keep:
                output = call_dynamic_filter(self._group_filter, self._args, item.group)
            if self._records is not None:
                self._records.record_group(
                    samples, decision="passed" if output.keep else "filtered", reason=output.reason
                )
            if not output.keep:
                self._metric_gatherer.on_dynamic_filter_drop(reason=output.reason)
                return False
        await super().put(item)
        return True

    def on_dequeue(self, entry, *, staleness, accepted):
        if self._records is not None:
            self._records.record_disposition(list(iter_samples(entry.group)), accepted=accepted, staleness=staleness)
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
            **(self._records.metrics() if self._records is not None else {}),
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
        self._rejected = self._incomplete = self._mixed = 0

    async def put(self, item):
        incomplete = len(item.group) != self._samples
        try:
            versions = policy_versions(
                {"weight_versions": [sample.weight_versions for sample in iter_samples(item.group)]}
            )
            mixed = len(set(versions)) != 1
        except ValueError:
            # Missing provenance cannot show that one behavior policy produced the group.
            mixed = True
        if not (incomplete or mixed):
            return await self._delegate.put(item)
        # A group can fail both checks, so the reason counters may sum past the total.
        self._rejected += 1
        self._incomplete += incomplete
        self._mixed += mixed
        self._unused(item.prompt_group)

    async def get(self, **context):
        return await self._delegate.get(**context)

    def get_metrics(self):
        rejected, incomplete, mixed = self._rejected, self._incomplete, self._mixed
        self._rejected = self._incomplete = self._mixed = 0
        return {
            **self._delegate.get_metrics(),
            "rollout/fully_async/rejected_policy_groups": rejected,
            "rollout/fully_async/rejected_incomplete_groups": incomplete,
            "rollout/fully_async/rejected_mixed_policy_groups": mixed,
        }

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
        return await self._delegate.put(item)
