"""Offline CPU checks for DPO collation, response log probabilities, and gradients.

The GPU suite retains device, BF16, and runtime smoke coverage. These tests use
small, locally initialized models and an independent cross-entropy reference.
"""

import pytest
import torch
from olmo_core.nn.transformer import TransformerConfig
from tokenizers import Tokenizer, models
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from open_instruct import dpo_utils
from open_instruct.olmo_core_train_modules import DPOLMHead


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "word": 2}, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")


def test_collator_preserves_indices_and_masks(tokenizer):
    # Non-sequential indices catch accidental reindexing. Chosen/rejected lengths
    # differ both within and across pairs, exercising both rounds of padding.
    samples = [
        {
            "chosen_input_ids": [2, 3, 4, 5],
            "chosen_labels": [-100, -100, 4, 5],
            "chosen_attention_mask": [1, 1, 1, 1],
            "rejected_input_ids": [2, 6],
            "rejected_labels": [-100, 6],
            "rejected_attention_mask": [1, 1],
            "index": 9,
        },
        {
            "chosen_input_ids": [7, 8],
            "chosen_labels": [-100, 8],
            "chosen_attention_mask": [1, 1],
            "rejected_input_ids": [7, 9, 10],
            "rejected_labels": [-100, 9, 10],
            "rejected_attention_mask": [1, 1, 1],
            "index": 3,
        },
    ]
    batch = dpo_utils.DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, padding="longest")(samples)
    assert batch["index"].tolist() == [9, 3]
    for branch in ("chosen", "rejected"):
        for key, pad_value in (("input_ids", 0), ("labels", -100), ("attention_mask", 0)):
            field = f"{branch}_{key}"
            expected = [sample[field] + [pad_value] * (4 - len(sample[field])) for sample in samples]
            torch.testing.assert_close(batch[field], torch.tensor(expected))
            assert batch[field].device.type == "cpu"


@pytest.fixture(params=["hf", "olmo"])
def model_and_backend(request):
    # Avoid changing other tests' global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if request.param == "hf":
            config = LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                pad_token_id=0,
                attention_dropout=0.0,
            )
            config._attn_implementation = "eager"
            model = LlamaForCausalLM(config).float().cpu()
        else:
            model = TransformerConfig.llama_like(
                d_model=16, vocab_size=32, n_layers=1, n_heads=2, hidden_size_multiple_of=16, use_flash=False
            ).build(init_device="cpu")
            model.init_weights(device=torch.device("cpu"))
            model.lm_head.__class__ = DPOLMHead
    return model.eval(), request.param


@pytest.fixture
def batch():
    # Includes masked prompts, right padding, unequal branch lengths, different
    # response lengths, and an entirely masked example (zero score/gradient).
    return {
        "chosen_input_ids": torch.tensor([[2, 3, 4, 5, 0], [6, 7, 8, 0, 0], [9, 10, 0, 0, 0]]),
        "chosen_labels": torch.tensor([[-100, -100, 4, 5, -100], [-100, -100, 8, -100, -100], [-100] * 5]),
        "chosen_attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]),
        "rejected_input_ids": torch.tensor([[2, 11, 12, 0], [6, 13, 14, 15], [9, 16, 0, 0]]),
        "rejected_labels": torch.tensor([[-100, 11, 12, -100], [-100, -100, -100, 15], [-100] * 4]),
        "rejected_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]),
    }


def reference_logps(model, backend, batch, branch, average):
    """Score each unpadded example using CE, without production DPO helpers."""
    scores = []
    for ids, labels, mask in zip(
        batch[f"{branch}_input_ids"], batch[f"{branch}_labels"], batch[f"{branch}_attention_mask"], strict=True
    ):
        length = int(mask.sum())
        ids, labels = ids[:length].unsqueeze(0), labels[1:length]
        output = model(ids)
        logits = output.logits if backend == "hf" else output
        score = -F.cross_entropy(logits[0, :-1], labels, ignore_index=-100, reduction="sum")
        if average:
            score = score / (labels != -100).sum().clamp(min=1)
        scores.append(score)
    return torch.stack(scores)


@pytest.mark.parametrize("average", [False, True], ids=["sum", "mean"])
@pytest.mark.parametrize("mode", ["concatenated", "separate"])
def test_forward_scores_and_gradients_match_reference(model_and_backend, batch, average, mode):
    model, backend = model_and_backend
    suffix = "_olmo" if backend == "olmo" else ""
    forward = getattr(dpo_utils, f"{mode}_forward{suffix}")
    chosen, rejected, aux_loss = forward(model, batch, average_log_prob=average)
    assert aux_loss is None
    for scores in (chosen, rejected):
        assert scores.shape == (3,)
        assert scores.device.type == "cpu"
        assert torch.isfinite(scores).all()
        assert scores[-1].item() == 0.0

    # A preference loss exercises gradients through both score branches.
    loss = -F.logsigmoid(0.1 * (chosen - rejected)).mean()
    loss.backward()
    actual_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert actual_grads
    assert any(torch.count_nonzero(grad) for grad in actual_grads.values())
    model.zero_grad(set_to_none=True)

    expected_chosen = reference_logps(model, backend, batch, "chosen", average)
    expected_rejected = reference_logps(model, backend, batch, "rejected", average)
    torch.testing.assert_close(chosen, expected_chosen, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rejected, expected_rejected, atol=1e-5, rtol=1e-5)
    reference_loss = -F.logsigmoid(0.1 * (expected_chosen - expected_rejected)).mean()
    reference_loss.backward()
    expected_grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert actual_grads.keys() == expected_grads.keys()
    for name, grad in actual_grads.items():
        torch.testing.assert_close(
            grad, expected_grads[name], atol=1e-6, rtol=1e-4, msg=lambda msg, name=name: f"{name}: {msg}"
        )
