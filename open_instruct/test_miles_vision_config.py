"""Image rollout selection configures the serving mode before GPU launch."""

import pytest

from open_instruct.miles.configuration import config
from open_instruct.miles.errors import InputError


def test_vision_hook_enables_multimodal_serving():
    run = config.RunConfig(config.CoreConfig(), {"custom_generate_function_path": config.VISION_GENERATOR})
    assert run.resolved_miles()["sglang_enable_multimodal"] is True
    assert "sglang_enable_multimodal" not in config.RunConfig(config.CoreConfig(), {}).resolved_miles()


@pytest.mark.parametrize(
    "core,miles",
    [
        ({"freeze_connector": False}, {}),
        ({"freeze_image_encoder": False}, {}),
        ({"publication_mode": "refresh"}, {}),
        ({"sequence_packing": True}, {}),
        ({}, {"sglang_enable_multimodal": False}),
        ({}, {"apply_chat_template": True}),
        ({}, {"multimodal_keys": {"image": "images"}}),
        ({}, {"partial_rollout": True}),
        ({}, {"use_rollout_routing_replay": True}),
        ({}, {"use_rollout_indexer_replay": True}),
    ],
)
def test_vision_rejects_unqualified_modes_before_launch(core, miles):
    run = config.RunConfig(
        core=config.CoreConfig(**core), miles={"custom_generate_function_path": config.VISION_GENERATOR, **miles}
    )
    with pytest.raises(InputError):
        run.resolved_miles()
