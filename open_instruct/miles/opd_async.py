"""Measured native Megatron async OPD with isolated evaluation and durable prompts."""

import copy
import time
from dataclasses import replace

from miles.rollout.base_types import RolloutFnEvalOutput
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets

from open_instruct.miles import async_rollout, data_source


class OPDAsyncDataSource(data_source.DashboardDrainingRolloutDataSource):
    """Keep prefetched batches in the restart ledger until their update is saved.

    The native async driver assembles batch N+1 before saving update N. Removing
    those prompts at assembly would silently skip them after preemption.
    """

    def __init__(self, args):
        super().__init__(args)
        self.consumer_rollout_id = args.start_rollout_id
        self._ack_by_rollout = {}

    def acknowledge_groups(self, groups):
        with self._cursor_lock:
            self._ack_by_rollout.setdefault(self.consumer_rollout_id, set()).update(g[0].group_index for g in groups)

    def save(self, rollout_id):
        with self._cursor_lock:
            for key in sorted(list(self._ack_by_rollout)):
                if key <= rollout_id:
                    groups = [self._pending_groups[index] for index in self._ack_by_rollout.pop(key)]
                    super().acknowledge_groups(groups)
            super().save(rollout_id)


class OPDAsyncRollout(async_rollout.ManagedFullyAsyncRolloutFn):
    """Use the managed producer without changing the sampled-token OPD objective."""

    async def __call__(self, input):
        if input.evaluation:
            return await super().__call__(input)
        self.data_source.consumer_rollout_id = input.rollout_id
        # Megatron's first publication is version 1, including on resume. Native
        # async prefetch assembles the NEXT batch before publishing this update;
        # enforce age against the version that will consume it, not that earlier
        # dequeue-time snapshot. This route requires exactly one update/rollout
        # and publication after every update (validated by OPDRunSpec).
        if not hasattr(self, "_first_rollout_id"):
            if input.weight_version != 1:
                raise ValueError("Async OPD expected initial Megatron publication version 1")
            self._first_rollout_id = input.rollout_id
        consumer_version = input.rollout_id - self._first_rollout_id + 1
        started = time.monotonic()
        result = await super().__call__(replace(input, weight_version=consumer_version))
        samples = [sample for group in result.samples for sample in group]
        if any(sample.oldest_weight_version is None for sample in samples):
            raise ValueError("Async OPD requires behavior weight versions for every sample")
        ages = [consumer_version - sample.oldest_weight_version for sample in samples]
        if min(ages) < 0 or max(ages) > self.args.max_weight_staleness:
            raise ValueError("Async OPD consumed samples outside its optimizer-step age budget")
        result.metrics = {
            **(result.metrics or {}),
            "rollout/fully_async/collection_seconds": time.monotonic() - started,
            "rollout/fully_async/consumer_version": consumer_version,
            "rollout/fully_async/consumed_age_mean": sum(ages) / len(ages),
            "rollout/fully_async/consumed_age_max": max(ages),
        }
        return result

    async def _call_eval(self, input):
        # In-flight training tasks retain self.state and its teacher hook. A
        # shallow state copy shares the engine semaphore, but owns separate args
        # so eval can use the math verifier without mutating training rewards.
        self._producer_resumed.clear()
        state = copy.copy(input.generate_state or self.state)
        state.args = copy.copy(state.args)
        state.args.custom_rm_path = "open_instruct.miles.opd_hooks.eval_reward"
        state.args.custom_reward_post_process_path = None
        final = input.rollout_id >= self.args.num_rollout - 1
        try:
            output = RolloutFnEvalOutput(data=await run_eval_datasets(state, self._eval_prompt_dataset_cache))
        finally:
            if final:
                await self.shutdown()
            else:
                self._producer_resumed.set()
        return output
