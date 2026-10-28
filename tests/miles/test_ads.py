"""Curriculum direction, feedback completeness, and exact resume contracts."""

import asyncio
import copy
from types import SimpleNamespace

import pytest
from miles.backends.core_utils.rollout import async_buffer
from miles.utils.types import Sample

from open_instruct.miles.datasets import ads
from open_instruct.miles.rollout import ads_buffer
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource


def rows(clusters=4, size=20):
    return [
        {"prompt_id": f"{c}:{i}", "cluster_id": c, "difficulty": float(i)}
        for c in range(clusters)
        for i in range(size)
    ]


def test_complete_round_moves_easy_prompts_harder_and_preserves_cardinality():
    schedule = ads.Schedule(rows(), mini_size=2, active_clusters=2)
    draws = [schedule.take() for _ in range(4)]
    before = copy.deepcopy(schedule.mini)
    for identity, round_id, _ in draws[:-1]:
        schedule.observe(identity, round_id, 1)
    assert schedule.mini == before
    schedule.observe(draws[-1][0], draws[-1][1], 1)
    selected = {schedule.lookup[identity] for identity, _, _ in draws}
    assert len(selected) == 2
    for cid in selected:
        assert schedule.mini[cid] == [1, 2]
        assert schedule.rates[cid] == 1
    assert sum(schedule.probs.values()) == pytest.approx(1)
    assert min(schedule.probs[c] for c in selected) > min(schedule.probs.values())


def test_zero_rewards_are_feedback_and_move_toward_easier():
    schedule = ads.Schedule(rows(), mini_size=2, active_clusters=2)
    schedule.mini = {c: [5, 6] for c in schedule.clusters}
    draws = [schedule.take() for _ in range(4)]
    for identity, round_id, _ in draws:
        schedule.observe(identity, round_id, 0)
    assert schedule.feedback_count == 4
    for cid in {schedule.lookup[i] for i, _, _ in draws}:
        assert schedule.mini[cid] == [4, 5]


def test_partial_round_and_queued_selection_resume_exactly():
    original = ads.Schedule(rows(), mini_size=2, active_clusters=2)
    identity, round_id, _ = original.take()
    original.observe(identity, round_id, 0.5)
    restored = ads.Schedule(rows(), mini_size=2, active_clusters=2)
    restored.load_state_dict(original.state_dict())
    for _ in range(39):
        a, b = original.take(), restored.take()
        assert a == b
        original.observe(a[0], a[1], 0.75)
        restored.observe(b[0], b[1], 0.75)
    assert original.state_dict() == restored.state_dict()


def test_reject_duplicate_feedback_and_changed_resume_table():
    schedule = ads.Schedule(rows(), mini_size=2, active_clusters=2)
    identity, round_id, _ = schedule.take()
    schedule.observe(identity, round_id, 0)
    with pytest.raises(ValueError, match="Duplicate"):
        schedule.observe(identity, round_id, 0)
    altered = rows()
    altered[0]["difficulty"] = 1.5
    with pytest.raises(ValueError, match="changed"):
        ads.Schedule(altered, mini_size=2, active_clusters=2).load_state_dict(schedule.state_dict())


def test_late_async_round_cannot_move_new_epoch():
    schedule = ads.Schedule(rows(size=3), mini_size=2, active_clusters=2)
    pending = [[schedule.take() for _ in range(4)] for _ in range(4)]
    for batch in pending[:3]:
        for identity, round_id, _ in batch:
            schedule.observe(identity, round_id, 1)
    assert schedule.epoch == 1
    before = copy.deepcopy(schedule.mini)
    for identity, round_id, _ in pending[-1]:
        schedule.observe(identity, round_id, 1)
    assert schedule.mini == before
    assert schedule.feedback_count == 16


def test_small_clusters_and_collisions_keep_unique_candidates():
    schedule = ads.Schedule(rows(size=3), mini_size=8, active_clusters=4)
    draws = [schedule.take() for _ in range(12)]
    assert len({x[0] for x in draws}) == 12
    for index, (identity, round_id, _) in enumerate(draws):
        schedule.observe(identity, round_id, float(index % 2))
    assert all(len(p) == len(set(p)) == 3 for p in schedule.mini.values())


def test_data_source_resume_does_not_apply_completed_feedback_twice(tmp_path):
    def source(load=None):
        args = SimpleNamespace(
            rollout_global_dataset=False,
            use_miles_dashboard=False,
            fully_async=True,
            buffer_filter_path=None,
            n_samples_per_prompt=2,
            save=str(tmp_path),
            load=str(load) if load else None,
            rollout_shuffle=False,
            reward_key=None,
            start_rollout_id=1 if load else 0,
        )
        result = DashboardDrainingRolloutDataSource(args)
        args.rollout_global_dataset = True
        result._ads = ads.Schedule(rows(), mini_size=2, active_clusters=2)
        result._ads_samples = {
            r["prompt_id"]: Sample(
                prompt=r["prompt_id"], metadata={"prepared_sample_id": r["prompt_id"], "prompt_format": "bare"}
            )
            for r in rows()
        }
        # Runtime identities include format; align this synthetic table too.
        table = [{**r, "prompt_id": r["prompt_id"] + ":bare"} for r in rows()]
        result._ads = ads.Schedule(table, mini_size=2, active_clusters=2)
        result._ads_samples = {k + ":bare": v for k, v in result._ads_samples.items()}
        return result

    original = source()
    groups = original.get_samples(3)
    for sample in groups[0]:
        sample.reward = 1.0
    original.observe_ads(groups[0])
    original.save(0)
    restored = source(tmp_path)
    restored.load(0)
    recovered = restored.get_samples(3)
    for sample in recovered[0]:
        sample.reward = 0.0
    restored.observe_ads(recovered[0])
    assert restored._ads.feedback_count == 1
    assert original._ads.state_dict() == restored._ads.state_dict()
    assert [g[0].group_index for g in recovered] == [g[0].group_index for g in groups]
    assert [g[0].prompt for g in restored.get_samples(5)] == [g[0].prompt for g in original.get_samples(5)]


def test_buffer_observes_all_wrong_group_before_native_filter(monkeypatch):
    events = []
    buffer = ads_buffer.ADSRefreshBuffer.__new__(ads_buffer.ADSRefreshBuffer)
    buffer._samples = 2
    buffer._source = SimpleNamespace(observe_ads=lambda group: events.append("observe"))

    async def filtered(self, item):
        events.append("native_filter")
        return False

    monkeypatch.setattr(async_buffer.RefreshPolicyDataBuffer, "put", filtered)
    item = SimpleNamespace(group=[Sample(reward=0.0), Sample(reward=0.0)])
    assert asyncio.run(buffer.put(item)) is False
    assert events == ["observe", "native_filter"]
    events.clear()
    item.group[0].status = Sample.Status.ABORTED
    asyncio.run(buffer.put(item))
    assert events == ["native_filter"]


def test_missing_reward_retires_round_without_treating_failure_as_zero():
    schedule = ads.Schedule(rows(), mini_size=2, active_clusters=2)
    draws = [schedule.take() for _ in range(4)]
    schedule.observe(draws[0][0], draws[0][1], None)
    for identity, round_id, _ in draws[1:]:
        schedule.observe(identity, round_id, 0.5)
    assert not schedule.rounds
    assert schedule.discarded_count == 1
    assert schedule.feedback_count == 3
    assert all(rate == 0.5 for rate in schedule.rates.values())
    assert schedule.reward_groups == {"zero": 0, "one": 0, "mixed": 3}
