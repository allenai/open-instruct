"""Data collators for training, including distillation support."""

from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer


class DistillationDataCollator:
    """Collator for distillation batches with compressed teacher outputs."""

    def __init__(self, tokenizer: PreTrainedTokenizer, model=None):
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        standard_features = [
            {
                k: v
                for k, v in f.items()
                if k in {"input_ids", "labels", "attention_mask"}
            }
            for f in features
        ]
        batch = self.base_collator(standard_features)

        if "compressed_logprobs" not in features[0]:
            return batch

        max_seq_len = batch["input_ids"].shape[1]
        max_distill_len = max_seq_len - 1
        for key in ["compressed_logprobs", "bytepacked_indices"]:
            if key not in features[0]:
                continue
            padded = []
            for feature in features:
                tensor = (
                    feature[key]
                    if isinstance(feature[key], torch.Tensor)
                    else torch.tensor(feature[key])
                )
                pad_len = max_distill_len - tensor.shape[0]
                if pad_len > 0:
                    tensor = torch.cat(
                        [
                            tensor,
                            torch.zeros((pad_len, tensor.shape[1]), dtype=tensor.dtype),
                        ]
                    )
                padded.append(tensor)
            batch[key] = torch.stack(padded)
        return batch
