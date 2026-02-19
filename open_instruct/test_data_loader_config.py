from open_instruct.data_loader import StreamingDataLoaderConfig


def test_dataset_mixer_eval_list_defaults_to_none():
    cfg = StreamingDataLoaderConfig()
    assert cfg.dataset_mixer_eval_list is None
