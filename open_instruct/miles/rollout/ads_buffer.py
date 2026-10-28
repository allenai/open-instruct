"""Feed scored groups to ADS before the unchanged native buffer/filter path."""

from miles.backends.core_utils.rollout import async_buffer
from miles.utils.types import Sample


class ADSRefreshBuffer(async_buffer.RefreshPolicyDataBuffer):
    def __init__(self, input):
        super().__init__(input)
        producer = getattr(input.unused_handler_fn, "__self__", None)
        self._source = getattr(producer, "data_source", None)
        if self._source is None or self._source._ads is None:
            raise ValueError("ADS buffer requires the managed retry producer and ADS data source")

    async def put(self, item):
        # Aborted or missing-reward work follows native retry/filter handling.
        # Its incomplete outcomes must never move the curriculum.
        if len(item.group) == self._samples and all(
            sample.status != Sample.Status.ABORTED and sample.reward is not None for sample in item.group
        ):
            self._source.observe_ads(item.group)
        return await super().put(item)

    def get_metrics(self, trainer_model_id=None):
        schedule = self._source._ads
        return {
            **super().get_metrics(trainer_model_id),
            "ads/feedback_groups": schedule.feedback_count,
            "ads/epoch": schedule.epoch,
            "ads/unique_prompts_scored": len(schedule.observed),
            **{f"ads/scored_groups_{key}": count for key, count in schedule.reward_groups.items()},
            "ads/pending_rounds": len(schedule.rounds),
            "ads/prob_max": max(schedule.probs.values()),
            "ads/prob_min": min(schedule.probs.values()),
            "ads/mean_rank_fraction": sum(
                sum(schedule.mini[c]) / len(schedule.mini[c]) / max(len(schedule.clusters[c]) - 1, 1)
                for c in schedule.clusters
            )
            / len(schedule.clusters),
        }
