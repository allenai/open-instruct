from open_instruct.data_loader import StreamingDataLoaderConfig


def test_dataset_mixer_eval_list_defaults_to_none():
    cfg = StreamingDataLoaderConfig()
    assert cfg.dataset_mixer_eval_list is None


def test_rollout_count_targets_are_mutually_exclusive():
    try:
        StreamingDataLoaderConfig(num_response_tokens_rollout=16, num_response_completions_rollout=8)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected mutually exclusive rollout target validation to raise.")
