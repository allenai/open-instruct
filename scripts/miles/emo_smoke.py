"""Create a tiny EMO-bearing checkpoint for bounded, synthetic RL smoke tests."""

import argparse
import json
from pathlib import Path

import torch
from olmo_core.nn.hf import config as hf_config
from olmo_core.nn.moe.v2.hf import configuration_olmo3moe, modeling_olmo3moe
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


async def reward(args, sample, **kwargs):
    """Deterministic mixed rewards exercise policy gradients; no quality claim."""
    return float(sample.index % 2)


def prepare(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    torch.manual_seed(173)
    hf_config._register_olmo3moe_auto_classes()
    config = configuration_olmo3moe.Olmo3MoeConfig(
        vocab_size=256,
        hidden_size=128,
        attention_hidden_size=128,
        head_dim=64,
        dense_mlp_intermediate_size=256,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_head_qk_norm=True,
        layer_types=["full_attention", "full_attention"],
        dense_layers_indices=[0],
        normalize_expert_weights=1.0,
        max_position_embeddings=512,
        emo_min_document_expert_pool=2,
        emo_max_document_expert_pool=4,
        emo_eval_document_expert_pool=4,
        emo_eos_token_id=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    # Intentionally unresolved and restricted: workflow preparation must explicitly
    # select full_pool before either Core training or olmo-sglang can load it.
    model = modeling_olmo3moe.Olmo3MoeForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(root)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{f"t{i}": i for i in range(4, 256)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        chat_template="{{ bos_token }}{% for message in messages %} {{ message['content'] }}{% endfor %}",
    ).save_pretrained(root)
    (root / "smoke-fixture.json").write_text(
        json.dumps({"seed": 173, "purpose": "EMO routing mechanics, not learning quality"}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
