"""GPU tests for the multimodal SFT training path (no weka, no downloads).

Uses a tiny ``MultimodalLM`` (2-layer ViT at CLIP geometry, 1M-param LM) and synthetic
examples in the §5.3 schema, so the tests exercise the real train module — weighted CE,
image-feature splicing, freezing — without touching real datasets or checkpoints.
"""

from unittest import mock

import numpy as np
import pytest
import torch
from olmo_core.config import DType
from olmo_core.data.multimodal import MixtureDataLoader, MultimodalCollatorConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.vision import MultimodalLMConfig, VisionEncoderConfig
from olmo_core.nn.vision.connector import VisionConnectorConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module import MultimodalTransformerTrainModuleConfig

VOCAB = 4096
IMAGE_PATCH_TOKEN_ID = 9
SEQ_LEN = 256
# CLIP ViT-L/14-336 geometry (defaults of VisionEncoderConfig): 24x24 = 576 patches.
N_PATCHES_SQ = 576
PATCH_DIM = 588
POOL_SIZE = 4
N_POOLED = N_PATCHES_SQ // POOL_SIZE  # 144 pooled features per crop

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def tiny_model_config() -> MultimodalLMConfig:
    lm = TransformerConfig.olmo2_1M(vocab_size=VOCAB)
    vision = VisionEncoderConfig(
        image_num_layers=2,
        image_emb_dim=64,
        image_num_heads=4,
        image_num_key_value_heads=4,
        image_head_dim=16,
        image_mlp_dim=128,
    )
    connector = VisionConnectorConfig.from_vision_encoder(vision, output_dim=lm.d_model)
    return MultimodalLMConfig(
        lm=lm, vision=vision, connector=connector, image_patch_token_id=IMAGE_PATCH_TOKEN_ID, vit_layers=(-1,)
    )


def _text_example(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(10, VOCAB, size=n).astype(np.int64)
    return {
        "input_ids": input_ids,
        "labels": np.roll(input_ids, -1),
        "loss_masks": np.full(n, 0.5, dtype=np.float32),
        "position_ids": np.arange(n, dtype=np.int64),
        "token_type_ids": np.zeros(n, dtype=np.int64),
        "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
        "pooled_patches_idx": np.full((0, POOL_SIZE), -1, dtype=np.int64),
    }


def _image_example(seed: int) -> dict[str, np.ndarray]:
    """One crop; the token stream carries exactly N_POOLED image-patch placeholders."""
    rng = np.random.default_rng(seed)
    n_text = 32
    input_ids = np.concatenate(
        [
            rng.integers(10, VOCAB, size=n_text),
            np.full(N_POOLED, IMAGE_PATCH_TOKEN_ID, dtype=np.int64),
            rng.integers(10, VOCAB, size=n_text),
        ]
    ).astype(np.int64)
    n = input_ids.shape[0]
    loss_masks = np.zeros(n, dtype=np.float32)
    loss_masks[-n_text:] = 1.0  # train on the trailing text only
    token_type_ids = np.zeros(n, dtype=np.int64)
    token_type_ids[n_text : n_text + N_POOLED] = 1
    return {
        "input_ids": input_ids,
        "labels": np.roll(input_ids, -1),
        "loss_masks": loss_masks,
        "position_ids": np.arange(n, dtype=np.int64),
        "token_type_ids": token_type_ids,
        "images": rng.normal(size=(1, N_PATCHES_SQ, PATCH_DIM)).astype(np.float32),
        "pooled_patches_idx": np.arange(N_PATCHES_SQ, dtype=np.int64).reshape(N_POOLED, POOL_SIZE),
    }


def _batch() -> dict[str, torch.Tensor]:
    collator = MultimodalCollatorConfig(pad_token_id=0, label_ignore_index=-100, pad_sequence_length=SEQ_LEN).build()
    return collator([_text_example(SEQ_LEN, seed=1), _image_example(seed=2)])


def _build_train_module(freeze_params: list[str] | None = None):
    model = tiny_model_config().build(init_device="cuda")
    train_module = _train_module_config(freeze_params).build(model)
    # train_batch reads trainer state (metrics, step counts); tests run without a Trainer.
    train_module._trainer = mock.Mock(global_step=1)
    return train_module


def _train_module_config(freeze_params: list[str] | None = None):
    return MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=2 * SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        optim=AdamWConfig(lr=1e-4),
        compile_model=False,
        compile_vision=False,
        compile_connector=False,
        vision_activation_checkpointing=False,
        connector_activation_checkpointing=False,
        autocast_precision=DType.bfloat16,
        freeze_params=freeze_params,
    )


@requires_gpu
def test_train_batch_mixed_modalities_produces_finite_loss():
    train_module = _build_train_module()
    train_module.train_batch(_batch())
    grads = [p.grad for p in train_module.model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "expected gradients after train_batch"
    assert all(torch.isfinite(g).all() for g in grads)


@requires_gpu
def test_freeze_params_keeps_vision_gradient_free():
    train_module = _build_train_module(freeze_params=["vision.*"])
    train_module.train_batch(_batch())
    vision_params = list(train_module.model.vision.parameters())
    assert vision_params
    assert all(not p.requires_grad for p in vision_params)
    assert any(p.grad is not None for p in train_module.model.connector.parameters())


class _StubDataset:
    def __init__(self, base_seed: int, length: int):
        self.base_seed = base_seed
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return _text_example(SEQ_LEN, seed=self.base_seed * 100_000 + index)


def _make_loader(work_dir) -> MixtureDataLoader:
    collator = MultimodalCollatorConfig(pad_token_id=0, label_ignore_index=-100, pad_sequence_length=SEQ_LEN).build()
    return MixtureDataLoader(
        [_StubDataset(1, 64), _StubDataset(2, 64)],
        [0.5, 0.5],
        collator,
        work_dir=str(work_dir),
        global_batch_size=2 * SEQ_LEN,
        seed=7,
        dataset_names=["a", "b"],
    )


def test_mixture_data_loader_resume_is_deterministic(tmp_path):
    """After restoring state_dict, the loader must replay the exact next batches."""
    loader = _make_loader(tmp_path / "run1")
    loader.reshuffle(epoch=1)
    it = iter(loader)
    consumed = [next(it) for _ in range(3)]
    del consumed
    state = loader.state_dict()
    expected = [next(it) for _ in range(2)]

    resumed = _make_loader(tmp_path / "run2")
    resumed.load_state_dict(state)
    resumed.reshuffle(epoch=1)
    resumed_it = iter(resumed)
    actual = [next(resumed_it) for _ in range(2)]

    for want, got in zip(expected, actual):
        torch.testing.assert_close(got["input_ids"], want["input_ids"])
        torch.testing.assert_close(got["loss_masks"], want["loss_masks"])
