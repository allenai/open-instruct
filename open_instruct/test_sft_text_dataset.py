"""CPU tests for the open_instruct_sft text adapter (no weka, no GPU, no downloads).

The schema mapping under test is the ``MixtureSource`` contract every future adapter
must satisfy (docs/design/multimodal_sft.md §5.3).
"""

import math
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset
from olmo_core.data.multimodal import MultimodalCollatorConfig
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

from open_instruct import dataset_transformation, sft_mixture, sft_text_dataset

EOS = 99
MASK = dataset_transformation.MASKED_TOKEN_VALUE


def _adapter(rows: dict, **kwargs) -> sft_text_dataset.OpenInstructTextDataset:
    return sft_text_dataset.OpenInstructTextDataset(
        Dataset.from_dict(rows).with_format("numpy", columns=["input_ids", "labels"]), eos_token_id=EOS, **kwargs
    )


def test_adapter_registered_in_source_registry():
    assert (
        sft_mixture.SOURCE_REGISTRY[sft_mixture.OPEN_INSTRUCT_SFT_TYPE] is sft_text_dataset._build_open_instruct_source
    )


def test_shift_and_mask_against_hand_built_conversation():
    # A 6-token conversation: prompt [10, 11, 12] (masked), assistant [20, 21, EOS] (trained).
    # open-instruct convention: labels[i] is the target AT position i, -100 = masked.
    rows = {"input_ids": [[10, 11, 12, 20, 21, EOS]], "labels": [[MASK, MASK, MASK, 20, 21, EOS]]}
    example = _adapter(rows, loss_token_weighting="none")[0]

    # Vision schema: labels[i] = input_ids[i+1]; the final position's label is EOS with mask 0.
    np.testing.assert_array_equal(example["labels"], [11, 12, 20, 21, EOS, EOS])
    # Loss on positions whose NEXT token is trainable: positions 2, 3, 4 (predicting 20, 21, EOS).
    np.testing.assert_array_equal(example["loss_masks"], [0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(example["input_ids"], [10, 11, 12, 20, 21, EOS])
    np.testing.assert_array_equal(example["position_ids"], np.arange(6))
    np.testing.assert_array_equal(example["token_type_ids"], np.zeros(6))


def test_root_tokens_weighting():
    rows = {"input_ids": [[10, 11, 12, 20, 21, EOS]], "labels": [[MASK, MASK, MASK, 20, 21, EOS]]}
    example = _adapter(rows, loss_token_weighting="root_tokens")[0]
    expected = 2.0 / math.sqrt(3)  # 3 loss tokens
    np.testing.assert_allclose(example["loss_masks"][2:5], expected, rtol=1e-6)
    assert example["loss_masks"][0] == 0


def test_message_weight_scales_masks():
    rows = {"input_ids": [[10, 20, EOS]], "labels": [[MASK, 20, EOS]]}
    example = _adapter(rows, loss_token_weighting="none", message_weight=0.15)[0]
    np.testing.assert_allclose(example["loss_masks"], [0.15, 0.15, 0.0], rtol=1e-6)


def test_zero_crop_schema_shapes_and_dtypes():
    rows = {"input_ids": [[10, 20, EOS]], "labels": [[MASK, 20, EOS]]}
    example = _adapter(rows)[0]
    assert example["images"].shape == (0, N_PATCHES_SQ, PATCH_DIM)
    assert example["images"].dtype == np.float32
    assert example["pooled_patches_idx"].shape == (0, POOL_H * POOL_W)
    assert example["pooled_patches_idx"].dtype == np.int64
    assert example["loss_masks"].dtype == np.float32
    assert example["labels"].dtype == np.int64


def test_base_vocab_guard_rejects_extra_vocab_ids():
    rows = {"input_ids": [[10, 500, EOS]], "labels": [[MASK, 500, EOS]]}
    adapter = _adapter(rows, base_vocab_size=100)
    with pytest.raises(ValueError, match="base vocab"):
        adapter[0]


def test_adapter_examples_survive_the_collator():
    rows = {"input_ids": [[10, 11, 20, EOS]], "labels": [[MASK, MASK, 20, EOS]]}
    example = _adapter(rows, loss_token_weighting="root_tokens")[0]
    collator = MultimodalCollatorConfig(pad_token_id=0, label_ignore_index=-100, pad_sequence_length=8).build()
    batch = collator([example])
    assert batch["input_ids"].shape == (1, 8)
    assert float(batch["loss_masks"][0, 4:].sum()) == 0.0


def test_config_rejects_bad_weighting_and_empty_mixer():
    with pytest.raises(ValueError, match="loss_token_weighting"):
        sft_text_dataset.OpenInstructTextDatasetConfig(
            mixer_list=["x", "1.0"], max_seq_length=64, loss_token_weighting="bogus"
        )
    with pytest.raises(ValueError, match="mixer_list"):
        sft_text_dataset.OpenInstructTextDatasetConfig(mixer_list=[], max_seq_length=64)


def test_build_passes_open_instruct_args_through(monkeypatch):
    """The adapter must hand dataset_transformation the run tokenizer, memory-mapping, and the
    validated-recipe conventions (add_bos off, chat template fall-through)."""
    captured: dict = {}

    def fake_get_cached(**kwargs):
        captured.update(kwargs)
        return Dataset.from_dict({"input_ids": [[1, 2, EOS]], "labels": [[MASK, 2, EOS]]}), {}

    monkeypatch.setattr(dataset_transformation, "get_cached_dataset_tulu_with_statistics", fake_get_cached)
    tokenizer = mock.Mock()
    tokenizer.name_or_path = "allenai/olmo-3-tokenizer-instruct-dev"
    tokenizer.eos_token_id = EOS

    config = sft_text_dataset.OpenInstructTextDatasetConfig(
        mixer_list=["allenai/Dolci-Instruct-SFT", "1.0"], max_seq_length=16384, chat_template_name="olmo123"
    )
    adapter = config.build(tokenizer)

    assert len(adapter) == 1
    assert captured["dataset_mixer_list"] == ["allenai/Dolci-Instruct-SFT", "1.0"]
    assert captured["dataset_keep_in_memory"] is False
    assert captured["transform_fn_args"][0] == {"max_seq_length": 16384}
    tc = captured["tc"]
    assert tc.tokenizer_name_or_path == "allenai/olmo-3-tokenizer-instruct-dev"
    assert tc.chat_template_name == "olmo123"
    assert tc.add_bos is False


def test_factory_builds_from_spec_args(monkeypatch):
    built: dict = {}

    def fake_build(self, tokenizer):
        built["config"] = self
        return "adapter"

    monkeypatch.setattr(sft_text_dataset.OpenInstructTextDatasetConfig, "build", fake_build)
    spec = sft_mixture.SourceSpec(
        name="open_instruct_text",
        type=sft_mixture.OPEN_INSTRUCT_SFT_TYPE,
        group="nlp",
        args={"mixer_list": ["x", "1.0"], "base_vocab_size": 151936},
    )
    result = sft_text_dataset._build_open_instruct_source(spec, object(), seed=7, max_sequence_length=4096)
    assert result == "adapter"
    config = built["config"]
    assert config.max_seq_length == 4096
    assert config.dataset_config_seed == 7
    assert config.base_vocab_size == 151936
