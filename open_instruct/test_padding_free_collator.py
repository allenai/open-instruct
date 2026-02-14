import unittest
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from parameterized import parameterized
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BambaConfig,
    BambaForCausalLM,
    DataCollatorForSeq2Seq,
    LlamaConfig,
    LlamaForCausalLM,
)

from open_instruct.dataset_processor import CHAT_TEMPLATES
from open_instruct.dataset_transformation import sft_tulu_tokenize_and_truncate_v1
from open_instruct.padding_free_collator import (
    TensorDataCollatorWithFlattening,
    TensorDataCollatorWithFlatteningDPO,
    concatenated_inputs,
    get_batch_logps,
)

try:
    import mamba_ssm  # noqa
    import causal_conv1d  # noqa

    mamba_and_causal_conv_available = True
except ImportError:
    mamba_and_causal_conv_available = False


MODEL_CLASSES = {"bamba": BambaForCausalLM, "llama": LlamaForCausalLM}
MODEL_CFGS = {"bamba": BambaConfig, "llama": LlamaConfig}
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
        num_hidden_layers=1,
        num_key_value_heads=2,
        pad_token_id=0,
    ),
    "llama": dict(
        hidden_act="gelu",
        hidden_size=32,
        intermediate_size=64,
        is_training=True,
        max_position_embeddings=512,
        mlp_bias=False,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=2,
    ),
}


def _get_fa2_model_and_cfg(model_name: str, vocab_size: int, dtype: torch.dtype) -> nn.Module:
    model_cls = MODEL_CLASSES[model_name]
    model_cfg = MODEL_CFGS[model_name]
    model_kwargs = MODEL_KWARGS[model_name]
    cfg = model_cfg(
        **{**model_kwargs, "dtype": dtype, "attn_implementation": "flash_attention_2", "vocab_size": vocab_size}
    )
    model = model_cls(cfg).to("cuda", dtype=dtype)
    return model, cfg


class TestPaddingFree(unittest.TestCase):
    seqlen = 128
    batch_size = 2
    dtype = torch.bfloat16

    @parameterized.expand([("bamba", "mean"), ("bamba", "sum"), ("llama", "mean"), ("llama", "sum")])
    @unittest.skipIf(not torch.cuda.is_available(), reason="Padding free tests require CUDA")
    def test_padding_free(self, model_name: str, loss_type: str) -> None:
        if model_name == "bamba" and not mamba_and_causal_conv_available:
            self.skipTest("bamba padding-free tests require mamba_ssm and causal_conv1d")
        torch.manual_seed(42)

        tokenizer = AutoTokenizer.from_pretrained("ibm-ai-platform/Bamba-9B-v2")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.chat_template = CHAT_TEMPLATES["tulu"]
        vocab_size = len(tokenizer)

        model, cfg = _get_fa2_model_and_cfg(model_name, vocab_size, self.dtype)
        model.initialize_weights()
        pf_model = deepcopy(model)

        data = {
            0: {
                "messages": [
                    {"role": "user", "content": "Why did the chicken cross the road?"},
                    {"role": "assistant", "content": "To get to the other side"},
                ]
            },
            1: {
                "messages": [
                    {"role": "user", "content": "What is one plus two?"},
                    {"role": "assistant", "content": "The answer is 3"},
                ]
            },
        }

        tok_data = {k: sft_tulu_tokenize_and_truncate_v1(v, tokenizer, max_seq_length=2**30) for k, v in data.items()}
        for v in tok_data.values():
            del v["messages"]

        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
        dataloader = DataLoader(tok_data, shuffle=False, collate_fn=collate_fn, batch_size=self.batch_size)

        pf_collate_fn = TensorDataCollatorWithFlattening()
        pf_dataloader = DataLoader(tok_data, shuffle=False, collate_fn=pf_collate_fn, batch_size=self.batch_size)

        batch = next(iter(dataloader))
        pf_batch = next(iter(pf_dataloader))
        for b in (batch, pf_batch):
            for k in b:
                if torch.is_tensor(b[k]):
                    b[k] = b[k].cuda()

        self.assertEqual(batch["input_ids"].shape[0], self.batch_size)
        self.assertEqual(pf_batch["input_ids"].shape[0], 1)

        incorrect_pf_batch = {
            "input_ids": pf_batch["input_ids"],
            "labels": pf_batch["labels"],
            "attention_mask": torch.ones_like(pf_batch["input_ids"]),
        }

        outputs = model(**batch)
        pf_outputs = pf_model(**pf_batch)
        with torch.no_grad():
            incorrect_pf_outputs = model(**incorrect_pf_batch)

        logits = outputs.logits.reshape(1, -1, outputs.logits.shape[-1])
        non_masked_logits = logits[:, batch["attention_mask"].flatten().bool()]
        pf_logits = pf_outputs.logits
        incorrect_pf_logits = incorrect_pf_outputs.logits
        torch.testing.assert_close(pf_logits, non_masked_logits)
        with self.assertRaisesRegex(AssertionError, "Mismatched elements:"):
            torch.testing.assert_close(pf_logits, incorrect_pf_logits)

        if loss_type == "mean":
            loss = outputs.loss
            pf_loss = pf_outputs.loss
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1).long(), reduce="sum")
            pf_loss = F.cross_entropy(
                pf_logits.view(-1, pf_logits.size(-1)), pf_batch["labels"].view(-1).long(), reduce="sum"
            )
        torch.testing.assert_close(loss, pf_loss)

        loss.backward()
        pf_loss.backward()

        grads = {n: p.grad for n, p in model.named_parameters()}
        pf_grads = {n: p.grad for n, p in pf_model.named_parameters()}
        for k, g in grads.items():
            torch.testing.assert_close(g, pf_grads[k])


def _make_dpo_features(
    num_samples: int, chosen_lengths: list[int], rejected_lengths: list[int], start_index: int = 0
) -> list[dict]:
    features = []
    for i in range(num_samples):
        chosen_len = chosen_lengths[i % len(chosen_lengths)]
        rejected_len = rejected_lengths[i % len(rejected_lengths)]
        features.append(
            {
                "chosen_input_ids": torch.ones(chosen_len, dtype=torch.long),
                "chosen_labels": torch.ones(chosen_len, dtype=torch.long),
                "rejected_input_ids": torch.ones(rejected_len, dtype=torch.long),
                "rejected_labels": torch.ones(rejected_len, dtype=torch.long),
                "index": start_index + i,
            }
        )
    return features


class TestDPOPackingIndices(unittest.TestCase):
    def _collate(self, max_seq_length, num_samples, chosen_lengths, rejected_lengths, start_index=0):
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)
        features = _make_dpo_features(num_samples, chosen_lengths, rejected_lengths, start_index)
        return collator(features)

    def _collate_and_get_logps(
        self,
        max_seq_length,
        num_samples,
        chosen_lengths,
        rejected_lengths,
        start_index=0,
        vocab_size=100,
        average_log_prob=False,
    ):
        batch = self._collate(max_seq_length, num_samples, chosen_lengths, rejected_lengths, start_index)
        concat_batch, bs = concatenated_inputs(batch)
        logits = torch.randn(1, concat_batch["concatenated_input_ids"].shape[1], vocab_size)
        logps = get_batch_logps(
            logits,
            concat_batch["concatenated_labels"],
            concat_batch["concatenated_cu_seq_lens_k"],
            average_log_prob=average_log_prob,
        )
        return batch, concat_batch, bs, logps

    @parameterized.expand([("no_truncation", 1000, 10, [10, 11, 12, 13]), ("with_padding", 500, 0, [0, 1, 2, 3])])
    def test_indices_preserved(self, name, max_seq_length, start_index, expected_indices):
        batch = self._collate(
            max_seq_length, num_samples=4, chosen_lengths=[50], rejected_lengths=[50], start_index=start_index
        )

        self.assertIn("index", batch)
        self.assertEqual(len(batch["index"]), 4)
        torch.testing.assert_close(batch["index"], torch.tensor(expected_indices))

    @parameterized.expand(
        [("no_truncation", 1000, 50, 50), ("after_truncation", 200, 100, 100), ("asymmetric", 250, 100, 50)]
    )
    def test_cu_seq_lens_matches_index_count(self, name, max_seq_length, chosen_len, rejected_len):
        batch = self._collate(
            max_seq_length, num_samples=4, chosen_lengths=[chosen_len], rejected_lengths=[rejected_len]
        )

        num_indices = len(batch["index"])
        self.assertEqual(len(batch["chosen_cu_seq_lens_k"]), num_indices + 1)
        self.assertEqual(len(batch["rejected_cu_seq_lens_k"]), num_indices + 1)

    def test_concatenated_inputs_returns_correct_bs(self):
        batch, concat_batch, bs, _ = self._collate_and_get_logps(
            max_seq_length=1000, num_samples=4, chosen_lengths=[50], rejected_lengths=[50]
        )

        self.assertEqual(bs, len(batch["index"]))
        self.assertIn("concatenated_cu_seq_lens_k", concat_batch)
        self.assertEqual(len(concat_batch["concatenated_cu_seq_lens_k"]), 2 * bs + 1)

    @parameterized.expand(
        [
            ("no_truncation", 1000, [100], [100], 4),
            ("slight_truncation", 300, [100], [100], 3),
            ("heavy_truncation", 150, [100], [100], 1),
            ("asymmetric", 250, [100], [50], 2),
        ]
    )
    def test_logps_count_matches_indices(
        self, name, max_seq_length, chosen_lengths, rejected_lengths, expected_min_indices
    ):
        batch, _, bs, logps = self._collate_and_get_logps(
            max_seq_length=max_seq_length,
            num_samples=4,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )

        num_indices = len(batch["index"])
        self.assertGreaterEqual(num_indices, expected_min_indices)
        self.assertEqual(len(logps), 2 * bs)

        chosen_logps = logps[:bs]
        rejected_logps = logps[bs:]
        self.assertEqual(len(chosen_logps), num_indices)
        self.assertEqual(len(rejected_logps), num_indices)

    @parameterized.expand([("no_truncation", 1000, []), ("with_truncation", 300, [3, 7, 11, 15])])
    def test_simulate_reference_cache(self, name, max_seq_length, expected_missing):
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)
        num_total_samples = 16
        all_features = _make_dpo_features(
            num_samples=num_total_samples, chosen_lengths=[100], rejected_lengths=[100], start_index=0
        )

        chosen_tensor = torch.full((num_total_samples,), float("-inf"))
        rejected_tensor = torch.full((num_total_samples,), float("-inf"))

        batch_size = 4
        for batch_start in range(0, num_total_samples, batch_size):
            batch_features = all_features[batch_start : batch_start + batch_size]
            batch = collator(batch_features)

            concat_batch, bs = concatenated_inputs(batch)
            logits = torch.randn(1, concat_batch["concatenated_input_ids"].shape[1], 100)
            logps = get_batch_logps(
                logits, concat_batch["concatenated_labels"], concat_batch["concatenated_cu_seq_lens_k"]
            )
            chosen_logps = logps[:bs]
            rejected_logps = logps[bs:]

            self.assertEqual(len(chosen_logps), len(batch["index"]))

            chosen_tensor[batch["index"]] = chosen_logps
            rejected_tensor[batch["index"]] = rejected_logps

        missing_chosen = torch.where(chosen_tensor == float("-inf"))[0].tolist()
        missing_rejected = torch.where(rejected_tensor == float("-inf"))[0].tolist()

        self.assertEqual(missing_chosen, expected_missing)
        self.assertEqual(missing_rejected, expected_missing)

    def test_prefilter_keeps_complete_sequences(self):
        batch = self._collate(max_seq_length=150, num_samples=4, chosen_lengths=[100], rejected_lengths=[100])

        self.assertEqual(len(batch["index"]), 1)
        self.assertEqual(batch["chosen_cu_seq_lens_k"][-1].item(), 100)
        self.assertEqual(batch["rejected_cu_seq_lens_k"][-1].item(), 100)

    def test_average_log_prob_all_masked_segment(self):
        vocab_size = 100
        seq_len = 20
        logits = torch.randn(1, seq_len, vocab_size)
        labels = torch.ones(1, seq_len, dtype=torch.long)
        labels[0, 5:10] = -100
        cu_seq_lens = torch.tensor([0, 5, 10, 20])

        result = get_batch_logps(logits, labels, cu_seq_lens, average_log_prob=True)

        self.assertEqual(result.shape[0], 3)
        self.assertFalse(torch.isnan(result).any(), f"Got NaN in result: {result}")

    def test_concatenated_cu_seq_lens_with_padding(self):
        batch, concat_batch, bs, _ = self._collate_and_get_logps(
            max_seq_length=500, num_samples=2, chosen_lengths=[50], rejected_lengths=[50]
        )

        self.assertEqual(bs, 2)
        self.assertEqual(batch["chosen_input_ids"].shape[-1], 500)
        self.assertEqual(batch["rejected_input_ids"].shape[-1], 500)
        self.assertEqual(concat_batch["concatenated_input_ids"].shape[-1], 1000)

        cu_seq_lens = concat_batch["concatenated_cu_seq_lens_k"]
        self.assertEqual(len(cu_seq_lens), 5)
        self.assertEqual(cu_seq_lens[0].item(), 0)
        self.assertEqual(cu_seq_lens[2].item(), 100)
        self.assertEqual(cu_seq_lens[3].item(), 550)
        self.assertEqual(cu_seq_lens[4].item(), 600)
