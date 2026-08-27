"""CPU tests for the multimodal SFT config builders (no weka, no GPU, no HF downloads)."""

import numpy as np
import pytest
import torch
from olmo_core.data.multimodal import MultimodalCollatorConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.vision import MultimodalLMConfig

from open_instruct import olmo_core_multimodal_utils as mm_utils

N_PATCHES = 729
PATCH_DIM = 588
POOL_SIZE = 4


def test_model_config_preset_applies_dropout():
    config = mm_utils.build_multimodal_model_config(
        mm_utils.MultimodalModelConfig(model_preset="molmo2_4B", residual_dropout=0.25)
    )
    assert isinstance(config, MultimodalLMConfig)
    assert config.lm.block.dropout == 0.25


def test_model_config_unknown_preset_raises():
    with pytest.raises(ValueError, match="no preset"):
        mm_utils.build_multimodal_model_config(mm_utils.MultimodalModelConfig(model_preset="bogus"))


def test_train_module_config_stage2_parity():
    training = mm_utils.MultimodalTrainingConfig()
    config = mm_utils.build_multimodal_train_module_config(training, world_size=8, gpus_per_node=8)

    assert config.rank_microbatch_size == 2 * 16384
    assert config.max_sequence_length == 16384
    assert config.response_logits_only is True
    assert config.z_loss_multiplier == 1e-4
    assert config.freeze_params is None

    overrides = {tuple(o.params): o.opts for o in config.optim.group_overrides}
    assert overrides[("connector.*",)]["lr"] == 5e-6
    assert overrides[("vision.*",)]["lr"] == 5e-6
    assert overrides[("connector.*",)]["scheduler_name"] == "connector"
    assert overrides[("vision.*",)]["scheduler_name"] == "vision"
    assert config.optim.lr == 1e-5
    assert set(config.scheduler.schedulers) == {"connector", "vision"}

    # Single node: plain FSDP, no shard-degree override (Stage2 behavior).
    assert config.dp_config.name == DataParallelType.fsdp


def test_train_module_config_multi_node_uses_hsdp():
    training = mm_utils.MultimodalTrainingConfig()
    config = mm_utils.build_multimodal_train_module_config(training, world_size=16, gpus_per_node=8)
    assert config.dp_config.name == DataParallelType.hsdp
    assert config.dp_config.shard_degree == 8


def test_train_module_config_freeze_params_passthrough():
    training = mm_utils.MultimodalTrainingConfig(freeze_params=["vision.*"])
    config = mm_utils.build_multimodal_train_module_config(training, world_size=1, gpus_per_node=1)
    assert config.freeze_params == ["vision.*"]


def test_collator_config_requires_pad_token():
    class NoPadTokenizer:
        pad_token_id = None
        name_or_path = "no-pad"

    with pytest.raises(ValueError, match="pad token"):
        mm_utils.build_multimodal_collator_config(NoPadTokenizer(), max_seq_length=64)


def _text_example(n: int) -> dict[str, np.ndarray]:
    """A zero-crop example in the vision-branch schema (the adapter contract, §5.3)."""
    return {
        "input_ids": np.arange(n, dtype=np.int64),
        "labels": np.arange(1, n + 1, dtype=np.int64),
        "loss_masks": np.full(n, 0.5, dtype=np.float32),
        "position_ids": np.arange(n, dtype=np.int64),
        "token_type_ids": np.zeros(n, dtype=np.int64),
        "images": np.zeros((0, N_PATCHES, PATCH_DIM), dtype=np.float32),
        "pooled_patches_idx": np.full((0, POOL_SIZE), -1, dtype=np.int64),
    }


def test_collator_pads_zero_crop_text_examples():
    """Guards the schema assumptions the open_instruct_sft adapter is built on."""
    collator = MultimodalCollatorConfig(pad_token_id=0, label_ignore_index=-100, pad_sequence_length=16).build()
    batch = collator([_text_example(5), _text_example(9)])

    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels"].shape == (2, 16)
    assert batch["loss_masks"].shape == (2, 16)
    # Right-padding: padded positions carry no loss.
    assert torch.all(batch["loss_masks"][0, 5:] == 0)
    assert torch.all(batch["loss_masks"][1, 9:] == 0)
    # An all-text batch still carries an images tensor (dummy crop) so FSDP ranks agree.
    assert "images" in batch
    assert batch["images"].shape[-2:] == (N_PATCHES, PATCH_DIM)


def test_global_batch_size_tokens():
    training = mm_utils.MultimodalTrainingConfig(max_seq_length=1024, global_batch_instances=4)
    assert mm_utils.global_batch_size_tokens(training) == 4096
