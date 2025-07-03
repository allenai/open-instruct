import warnings
from dataclasses import dataclass

import torch
from transformers import DefaultDataCollator


@dataclass
class TensorDataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator for padding-free training along the lines of https://huggingface.co/blog/packing-with-FA2

    Eliminates use of padding which is generically needed for per_gpu_batch_size > 1, thereby
    reducing memory costs and increasing throughput. Your model class must support padding-free
    training to use this collator correctly. Examples which support padding free include
    LlamaForCausalLM and BambaForCausalLM.

    The `input_ids` and `labels` from separate examples are concatenated together into a tensor of
    batch size 1, with additional information included in the batch to demarcate example boundaries.
    """

    def __init__(
        self,
        *args,
        return_flash_attn_kwargs=True,
        return_position_ids=True,
        return_seq_idx=True,
        separator_id=-100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_flash_attn_kwargs = return_flash_attn_kwargs
        self.return_position_ids = return_position_ids
        self.return_seq_idx = return_seq_idx
        self.separator_id = separator_id
        warnings.warn(
            "Using `TensorDataCollatorWithFlattening` will flatten the entire mini batch into a "
            "single long sequence. Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None, separator_id=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        if self.return_position_ids:
            pos_ids = []
        if self.return_seq_idx:
            seq_idx = []
        is_labels_provided = "labels" in features[0]
        ret = {"input_ids": [], "labels": []}
        separator = torch.tensor(
            [separator_id], dtype=features[0]["input_ids"].dtype, device=features[0]["input_ids"].device
        )
        for s_idx, item in enumerate(features):
            input_ids = item["input_ids"]
            ret["input_ids"].append(input_ids)
            if is_labels_provided:
                ret["labels"].append(separator)
                ret["labels"].append(item["labels"][1:])
            else:
                ret["labels"].append(separator)
                ret["labels"].append(input_ids[1:])
            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))
            if self.return_position_ids:
                pos_ids.append(torch.arange(input_ids.numel(), device=input_ids.device))
            if self.return_seq_idx:
                seq_idx.append(torch.full_like(input_ids, s_idx, dtype=torch.int32))

        if self.return_flash_attn_kwargs:
            ret["cu_seq_lens_q"] = ret["cu_seq_lens_k"] = torch.tensor(
                cu_seq_lens, dtype=torch.int32, device=features[0]["input_ids"].device
            )
            ret["max_length_q"] = ret["max_length_k"] = max_length
        if self.return_position_ids:
            ret["position_ids"] = torch.cat(pos_ids, dim=0)[None]
        if self.return_seq_idx:
            ret["seq_idx"] = torch.cat(seq_idx, dim=0)[None]
        ret["input_ids"] = torch.cat(ret["input_ids"], dim=0)[None]
        ret["labels"] = torch.cat(ret["labels"], dim=0)[None]
        return ret
