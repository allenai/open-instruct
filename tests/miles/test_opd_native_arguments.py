"""Parse the real Megatron arguments before spending GPU time on an integration run."""

import importlib.util
import shlex
import sys
from pathlib import Path

import torch
from megatron.training import arguments as megatron_arguments
from miles.utils import arguments
from transformers import Qwen3_5Config

from open_instruct.miles.configuration import specs
from open_instruct.miles.distillation import opd_runtime


def test_native_megatron_parser_accepts_opd_profile(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = Qwen3_5Config(
        text_config={
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"},
        }
    )
    model = tmp_path / "model"
    config.save_pretrained(model)
    spec = specs.load(
        root / "tests/miles/fixtures/opd/qwen35-4b-tiny.toml",
        [
            'model.source="Qwen/Qwen3.5-2B"',
            'model.revision=""',
            'model.architecture="qwen3.5-2B"',
            "trainer.gpus=4",
            "inference.gpus=3",
        ],
    )
    profile = importlib.util.spec_from_file_location("profile", opd_runtime.PROFILES / "qwen3.5-2B.py")
    module = importlib.util.module_from_spec(profile)
    profile.loader.exec_module(module)
    prepared = {
        "model": str(model),
        "data": {
            "prompt_data": str(tmp_path / "prompts.jsonl"),
            "eval_prompt_data": ["math", str(tmp_path / "eval.jsonl")],
        },
    }
    argv = opd_runtime.native_arguments(
        spec, prepared, tmp_path / "checkpoint", "http://localhost:1/generate", shlex.split(module.model_args())
    )
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    # CPU validation supplies only the device generation queried by Megatron;
    # model construction and CUDA kernels remain part of the GPU gate.
    if not torch.cuda.is_available():
        monkeypatch.setattr(megatron_arguments, "get_device_arch_version", lambda: 10)
    parsed = arguments.parse_args()
    assert parsed.train_backend == "megatron"
    assert parsed.tensor_model_parallel_size == 2
    assert parsed.world_size == 4
    assert parsed.lr == 1e-6
    assert parsed.use_opd and parsed.opd_kl_coef == 1.0
