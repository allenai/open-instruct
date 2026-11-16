"""Pool-aware queue adapter for the pinned Core asynchronous runtime."""

from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer, RefreshPolicyDataBuffer
from miles.rollout.filter_hub.common_filters import group_staleness

from open_instruct.miles.rollout.pool_batch_queue import PoolBatchQueue


class PoolBatchDataBuffer:
    def __init__(self, input):
        args, core = input.args, input.args.olmo_core
        if not args.fully_async or args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
            raise ValueError("Pool batches require fully async, one-update collections")
        buffer_type = RefreshPolicyDataBuffer if core.publication_mode == "refresh" else HomogeneousPolicyDataBuffer
        self.buffer = buffer_type(input)
        delegate = self.buffer._delegate
        size = args.rollout_batch_size
        # Pigeonhole bound: even if each pool has size-1 queued groups, the next
        # group completes a batch. Leave another batch for prepared reservations.
        minimum = len(core.pool_batch_values) * (size - 1) + 1 + size
        if delegate._capacity < minimum:
            raise ValueError(f"Pool batches require completed-buffer capacity >= {minimum} prompt groups")

        def stale(entry):
            age = group_staleness(entry.group, delegate._current_version)
            return age is not None and age > delegate._args.max_weight_staleness

        self.queue = PoolBatchQueue(
            size=size,
            fraction=core.pool_batch_fraction,
            field=core.pool_batch_field,
            values=core.pool_batch_values,
            seed=args.rollout_seed,
            version=lambda: delegate._current_version,
            stale=stale,
        )
        # Keep the runtime's get/put and disposition hooks intact. This adapter
        # deliberately depends on the queue interface of the pinned runtime.
        delegate._buffer = self.queue

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    async def release_prepared_capacity(self, count):
        # A cancelled reservation may have dequeued only part of a cohort. Its
        # prompts are requeued by the producer; remaining completed groups must
        # be eligible for a newly assembled full batch, never a partial cohort.
        self.queue.cohort.clear()
        await self.buffer.release_prepared_capacity(count)

    def get_metrics(self, trainer_model_id=None):
        return {
            **self.buffer.get_metrics(trainer_model_id),
            "rollout/pool_batches/formed": self.queue.batch_count,
            "rollout/pool_batches/homogeneous": self.queue.pool_batches,
            **{
                f"rollout/pool_batches/queued/{pool}": sum(self.queue.pool(entry) == pool for entry in self.queue)
                for pool in self.queue.values
            },
        }
