"""Selection tracing around the native measured queue, preserving its return contract."""

from miles.backends.core_utils.rollout import async_buffer
from miles.rollout.filter_hub.base_types import iter_samples

from open_instruct.miles.distillation import opd_selection_audit


class MeasuredDataBuffer(async_buffer.MeasuredDataBuffer):
    async def put(self, item):
        if any(sample.status.value == "aborted" for sample in iter_samples(item.group)):
            opd_selection_audit.record("aborted_group_rejected", item.group)
        return await super().put(item)

    def on_dequeue(self, entry, *, staleness, accepted):
        opd_selection_audit.record(
            "queue_selected" if accepted else "stale_group_rejected", entry.group, policy_age=staleness
        )
        return super().on_dequeue(entry, staleness=staleness, accepted=accepted)
