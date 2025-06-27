import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BambaConfig,
    BambaForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

# HACK for being able to load the collator without needing to install open-instruct
open_instruct_dir = Path(__file__).parent.parent.absolute()
sys.path.append(open_instruct_dir)
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening

MODEL_CLASSES = {"bamba": BambaForCausalLM, "llama": LlamaForCausalLM}
MODEL_CFGS = {
    "bamba": BambaConfig,
    "llama": LlamaConfig,
}
MODEL_KWARGS = {
    "bamba": dict(
        attention_dropout=0.0,
        attn_layer_indices=None,
        attn_rotary_emb=8,
        hidden_act="silu",
        hidden_size=32,
        initializer_range=0.02,
        intermediate_size=64,
        mamba_chunk_size=16,
        mamba_d_conv=4,
        mamba_d_state=16,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_n_heads=16,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        pad_token_id=0,
        vocab_size=99,
    ),
    "llama": dict(
        hidden_act="gelu",
        hidden_size=32,
        intermediate_size=64,
        is_training=True,
        max_position_embeddings=512,
        mlp_bias=False,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        vocab_size=99,
    ),
}


class TestPaddingFree:
    seqlen = 128
    batch_size = 2
    dtype = torch.bfloat16

    def get_fa2_model_and_cfg(self, model_name: str) -> nn.Module:
        model_cls = MODEL_CLASSES[model_name]
        model_cfg = MODEL_CFGS[model_name]
        model_kwargs = MODEL_KWARGS[model_name]
        cfg = model_cfg(
            **{
                **model_kwargs,
                "torch_dtype": self.dtype,
                "attn_implementation": "flash_attention_2",
            }
        )
        model = model_cls(cfg).to("cuda", dtype=self.dtype)
        return model, cfg

    @pytest.mark.parametrize("model_name", ["bamba", "llama"])
    def test_padding_free(self, model_name: str) -> None:
        torch.manual_seed(42)
        model, cfg = self.get_fa2_model_and_cfg(model_name)

        inputs = torch.randint(cfg.vocab_size, size=(self.batch_size, self.seqlen), device="cuda")
        # Non-padding-free batch:
        batch = {
            "input_ids": inputs,
            "labels": inputs,
            "attention_mask": torch.ones_like(inputs),
        }

        # Padding-free batch from the collator
        dataset = {idx: {"input_ids": example} for idx, example in enumerate(inputs)}
        collate_fn = TensorDataCollatorWithFlattening()
        train_dataloader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
        )
        pf_batch = next(iter(train_dataloader))
        assert batch["input_ids"].shape[0] == 2
        assert pf_batch["input_ids"].shape[0] == 1

        # Also create a batch with the pf style concatenation, but without the pf seq markers as a
        # control. Passing this through the model should give incorrect results.

        incorrect_pf_batch = {
            "input_ids": pf_batch["input_ids"],
            "labels": pf_batch["labels"],
            "attention_mask": torch.ones_like(pf_batch["input_ids"]),
        }

        with torch.no_grad():
            outputs = model(**batch)
            pf_outputs = model(**pf_batch)
            incorrect_pf_outputs = model(**incorrect_pf_batch)

        logits = outputs.logits.reshape(1, -1, outputs.logits.shape[-1])
        pf_logits = pf_outputs.logits
        incorrect_pf_logits = incorrect_pf_outputs.logits
        torch.testing.assert_close(pf_logits, logits)
        torch.testing.assert_close(outputs.loss, pf_outputs.loss)
        with pytest.raises(AssertionError, match="Mismatched elements:"):
            torch.testing.assert_close(pf_logits, incorrect_pf_logits)
