"""Real HF/native checkpoint preparation through the pinned CPU runtime."""

import json
from pathlib import Path

import pytest
import torch
from olmo_core import config as core_config
from olmo_core.data import tokenizer as core_tokenizer
from olmo_core.distributed import checkpoint
from olmo_core.nn import attention
from olmo_core.nn.moe.v2 import olmo3
from olmo_sglang import config as serving_config
from safetensors import torch as safetensors_torch
from scripts.miles import emo_smoke
from transformers import AutoConfig, AutoTokenizer

from open_instruct.miles.execution import emo, workflow
from open_instruct.test_miles_workflow import Spec


@pytest.mark.parametrize("source_format", ["hf", "olmo_core"])
def test_checkpoint_ingestion_preserves_weights_and_source(tmp_path, source_format):
    hf_source = tmp_path / "original-hf"
    emo_smoke.prepare(hf_source)
    original_hf = json.loads((hf_source / "config.json").read_text())
    expected = safetensors_torch.load_file(hf_source / "model.safetensors")
    source = hf_source
    if source_format == "olmo_core":
        model_config = olmo3.build_olmo3_moe_config_from_hf_config(
            emo.resolve_hf(original_hf, "full_pool"), attention_backend=attention.AttentionBackendName.torch
        )
        model = model_config.build(init_device="cpu")
        olmo3.load_olmo3_moe_hf_state(model, AutoConfig.from_pretrained(hf_source, local_files_only=True), expected)

        def restore_source_routing(config):
            if hasattr(config, "emo") and config.emo is not None:
                config.emo.full_pool = False
                config.emo.eval_document_expert_pool = None
                config.emo.source_config = None

        model_config.apply(restore_source_routing)
        source = tmp_path / "native" / "step0"
        checkpoint.save_model_and_optim_state(source / "model_and_optim", model)
        tokenizer = core_tokenizer.TokenizerConfig(
            vocab_size=256, eos_token_id=2, pad_token_id=0, bos_token_id=1, identifier=str(hf_source)
        )
        (source / "config.json").write_text(
            json.dumps({"model": model_config.as_config_dict(), "dataset": {"tokenizer": tokenizer.as_config_dict()}})
        )

    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    spec = Spec(tmp_path / "run", source)
    spec.model.update(format=source_format, emo_routing_mode="full_pool")
    prepared = Path(workflow.prepare_model(spec))
    hf = AutoConfig.from_pretrained(prepared, local_files_only=True)
    serving_config.validate_olmo3_moe_config(hf)
    assert hf.emo_routing_mode == "full_pool"
    assert hf.emo_eval_document_expert_pool == hf.n_routed_experts == 8
    assert hf.emo_source_config["emo_eval_document_expert_pool"] == (4 if source_format == "hf" else None)
    assert hf.emo_min_document_expert_pool == 2
    assert hf.emo_max_document_expert_pool == 4
    assert workflow.prepare_model(spec) == str(prepared)
    assert {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before

    actual = {}
    for shard in prepared.glob("*.safetensors"):
        actual.update(safetensors_torch.load_file(shard))
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    original_tokenizer = AutoTokenizer.from_pretrained(hf_source, local_files_only=True)
    prepared_tokenizer = AutoTokenizer.from_pretrained(prepared, local_files_only=True)
    assert prepared_tokenizer.encode("t4 t5 t6") == original_tokenizer.encode("t4 t5 t6")
    assert prepared_tokenizer.chat_template == original_tokenizer.chat_template

    # Reload the prepared config and weights into a fresh Core model without
    # requiring another explicit selection or sampling native document pools.
    restored = core_config.Config.from_dict(
        olmo3.build_olmo3_moe_config_from_hf_config(
            hf, attention_backend=attention.AttentionBackendName.torch
        ).as_config_dict()
    ).build(init_device="cpu")
    olmo3.load_olmo3_moe_hf_state(restored, hf, actual)
    router = restored.blocks["1"].routed_experts_router
    assert router.emo.full_pool
    assert not router.requires_segment_ids
