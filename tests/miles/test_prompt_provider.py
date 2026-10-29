"""Compare the provider-backed path with the pinned, unmodified MILES loader."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from miles.utils.types import Sample

from miles.rollout import data_source as miles_data_source
from open_instruct.miles.rollout import prompt_provider
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource
from open_instruct.miles.rollout.provider_source import ProviderRolloutDataSource


@pytest.fixture
def args(tmp_path, monkeypatch):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "".join(
            json.dumps({"input": f"prompt-{i}", "label": str(i), "metadata": {"nested": [i]}}) + "\n"
            for i in range(11)
        )
    )
    monkeypatch.setattr(miles_data_source, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(miles_data_source, "load_processor", lambda *a, **k: None)
    return SimpleNamespace(
        rollout_global_dataset=True,
        hf_checkpoint="unused",
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(path),
        rollout_max_prompt_len=None,
        input_key="input",
        multimodal_keys=None,
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=17,
        rollout_shuffle=True,
        n_samples_per_prompt=8,
        buffer_filter_path=None,
        save=str(tmp_path),
        load=None,
        start_rollout_id=0,
        fully_async=True,
        use_miles_dashboard=False,
    )


def payload(groups):
    return [[s.to_dict() for s in group] for group in groups]


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("k", [1, 8])
def test_original_order_payloads_ids_and_retry_buffer_are_preserved(args, shuffle, k):
    args.rollout_shuffle, args.n_samples_per_prompt = shuffle, k
    before = miles_data_source.RolloutDataSourceWithBuffer(args)
    after = ProviderRolloutDataSource(args)
    for n in [0, 1, 5, 5, 0, 1, 3, 8, 2]:
        assert payload(before.get_samples(n)) == payload(after.get_samples(n))
        assert (before.epoch_id, before.sample_offset) == (after.epoch_id, after.sample_offset)
    old_groups, new_groups = before.get_samples(2), after.get_samples(2)
    before.add_samples(old_groups)
    after.add_samples(new_groups)
    cursor = after.provider.state_dict()
    assert payload(before.get_samples(1)) == payload(after.get_samples(1))
    assert after.provider.state_dict() == cursor
    assert payload(before.get_samples(4)) == payload(after.get_samples(4))


def test_request_can_cross_multiple_epochs_and_copies_are_isolated(args):
    large, small = ProviderRolloutDataSource(args), ProviderRolloutDataSource(args)
    groups = large.get_samples(40)
    assert payload(groups) == payload([small.get_samples(1)[0] for _ in range(40)])
    groups[0][0].metadata["nested"].append("changed")
    assert "changed" not in groups[0][1].metadata["nested"]
    assert all("changed" not in s.metadata["nested"] for s in large.dataset.origin_samples)


@pytest.mark.parametrize("n", [-1, True, 1.5])
def test_invalid_request_does_not_consume_buffer_or_cursor(args, n):
    source = ProviderRolloutDataSource(args)
    source.add_samples(source.get_samples(1))
    before = copy.deepcopy(source.provider.state_dict())
    with pytest.raises(ValueError):
        source.get_samples(n)
    assert source.provider.state_dict() == before
    assert len(source.buffer) == 1


@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_preserves_cursor_and_pristine_pending_groups(args, fully_async, legacy):
    args.fully_async = fully_async
    source = DashboardDrainingRolloutDataSource(args)
    groups = source.get_samples(14)
    source.acknowledge_groups(groups[:-2])
    source.add_samples([groups[-1]])
    groups[-2][0].response = "partial output"
    source.save(0)
    path = miles_data_source.compute_global_dataset_state_path(args.save, rollout_id=0)
    state = torch.load(path, weights_only=True)
    if legacy:
        del state["prompt_provider"]
        torch.save(state, path)
    restored_args = copy.copy(args)
    restored_args.load, restored_args.start_rollout_id = args.save, 1
    restored = DashboardDrainingRolloutDataSource(restored_args)
    restored.load(0)
    pending = groups[-2:] if fully_async else groups[-1:]
    resumed = restored.get_samples(len(pending) + 3)
    assert [g[0].group_index for g in resumed[: len(pending)]] == [g[0].group_index for g in pending]
    assert all(not s.response and not s.tokens for g in resumed for s in g)
    source._delegate.buffer.clear()
    assert payload(resumed[len(pending) :]) == payload(source.get_samples(3))


class ReverseProvider:
    """A deliberately different selector with feedback and no epoch interface."""

    def __init__(self):
        self.cursor = 10
        self.outcomes = 0

    def next_samples(self, n):
        prompts = [Sample(prompt=f"custom-{self.cursor - i}") for i in range(n)]
        self.cursor -= n
        return prompts

    def feedback(self, records):
        self.outcomes += len(records)

    def state_dict(self):
        return {"kind": "reverse", "cursor": self.cursor, "outcomes": self.outcomes}

    def load_state_dict(self, state):
        assert state["kind"] == "reverse"
        self.cursor, self.outcomes = state["cursor"], state["outcomes"]


def test_alternate_provider_feedback_and_state_share_pending_checkpoint(args):
    source = DashboardDrainingRolloutDataSource(args, provider=ReverseProvider())
    a, b = source.get_samples(2)
    assert [a[0].prompt, b[0].prompt] == ["custom-10", "custom-9"]
    source.acknowledge_groups([a])
    source.feedback([{"kind": "group", "group_index": a[0].group_index}])
    source.save(0)
    args.load, args.start_rollout_id = args.save, 1
    restored = DashboardDrainingRolloutDataSource(args, provider=ReverseProvider())
    restored.load(0)
    assert restored._delegate.provider.state_dict() == source._delegate.provider.state_dict()
    replay, fresh = restored.get_samples(2)
    assert payload([replay]) == payload([b])
    assert fresh[0].prompt == "custom-8"
    # Switching implementations cannot silently consume a saved cursor.
    with pytest.raises(ValueError, match="configuration"):
        DashboardDrainingRolloutDataSource(args).load(0)


def test_shuffle_feedback_is_noop_and_changed_config_rejects_resume(args):
    source = DashboardDrainingRolloutDataSource(args)
    source.get_samples(1)
    state = source._delegate.provider.state_dict()
    source.feedback([{"kind": "group"}])
    assert source._delegate.provider.state_dict() == state
    source.save(0)
    args.load = args.save
    args.rollout_shuffle = False
    with pytest.raises(ValueError, match="configuration"):
        DashboardDrainingRolloutDataSource(args).load(0)


def test_blank_prompt_mode_and_empty_dataset(args):
    args.rollout_global_dataset = False
    assert payload(ProviderRolloutDataSource(args).get_samples(2)) == payload(
        miles_data_source.RolloutDataSourceWithBuffer(args).get_samples(2)
    )
    provider = prompt_provider.ShuffledPromptProvider([], shuffle=False)
    assert provider.next_samples(0) == []
    with pytest.raises(ValueError, match="empty dataset"):
        provider.next_samples(1)
