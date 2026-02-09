from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DefaultDataCollator

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

PAD_VALUES: dict[str, int] = {"labels": -100, "attention_mask": 0, "position_ids": 0, "seq_idx": -1}


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int | float) -> torch.Tensor:
    """Right-pad a tensor to a specified length along the last dimension."""
    if tensor.size(-1) >= length:
        return tensor
    return F.pad(tensor, (0, length - tensor.size(-1)), value=pad_value)


def _truncate_to_length(tensors: dict[str, torch.Tensor | None], length: int) -> dict[str, torch.Tensor | None]:
    return {k: (v[:, :length] if v is not None else None) for k, v in tensors.items()}


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

    return_flash_attn_kwargs: bool = True
    return_position_ids: bool = True
    return_seq_idx: bool = True
    separator_id: int = -100
    max_seq_length: int | None = None

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
        position_ids_tensor = None
        seq_idx_tensor = None
        if self.return_position_ids:
            position_ids_tensor = torch.cat(pos_ids, dim=0)[None]
        if self.return_seq_idx:
            seq_idx_tensor = torch.cat(seq_idx, dim=0)[None]
        input_ids_tensor = torch.cat(ret["input_ids"], dim=0)[None]
        labels_tensor = torch.cat(ret["labels"], dim=0)[None]

        tensors = {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "position_ids": position_ids_tensor,
            "seq_idx": seq_idx_tensor,
        }

        num_valid_seqs = len(features)
        wasted_tokens = 0
        sequences_dropped = 0
        if self.max_seq_length is not None:
            current_len = input_ids_tensor.shape[1]
            if current_len > self.max_seq_length:
                wasted_tokens = current_len - self.max_seq_length
                tensors = _truncate_to_length(tensors, self.max_seq_length)
                if self.return_flash_attn_kwargs:
                    cu_seq_lens_tensor = ret["cu_seq_lens_q"].clamp_max_(self.max_seq_length)
                    valid_mask = torch.cat(
                        [
                            torch.tensor([True], device=cu_seq_lens_tensor.device),
                            cu_seq_lens_tensor[1:] > cu_seq_lens_tensor[:-1],
                        ]
                    )
                    num_valid_seqs = valid_mask.sum().item() - 1
                    sequences_dropped = len(features) - num_valid_seqs
                    if sequences_dropped > 0:
                        logger.warning(
                            f"Truncation dropped {sequences_dropped} sequences "
                            f"(packed {len(features)} sequences into {current_len} tokens, "
                            f"truncated to {self.max_seq_length}, {num_valid_seqs} sequences remain)"
                        )
                        cu_seq_lens_tensor = cu_seq_lens_tensor[valid_mask]
                    ret["cu_seq_lens_q"] = ret["cu_seq_lens_k"] = cu_seq_lens_tensor
            elif current_len < self.max_seq_length:
                for key, tensor in tensors.items():
                    if tensor is not None:
                        tensors[key] = pad_to_length(tensor, self.max_seq_length, PAD_VALUES.get(key, 0))

        ret["_num_valid_seqs"] = num_valid_seqs
        ret["_wasted_tokens_from_truncation"] = wasted_tokens
        ret["_sequences_dropped"] = sequences_dropped
        for key, tensor in tensors.items():
            if tensor is not None:
                ret[key] = tensor
        return ret


@dataclass
class TensorDataCollatorWithFlatteningDPO(TensorDataCollatorWithFlattening):
    def __call__(self, features, return_tensors=None, separator_id=None):
        # Collate chosen and rejected separately, then combine with min(valid_seqs) alignment
        def filter_batch(match_string, features):
            return [{k.replace(match_string, ""): v for k, v in f.items() if match_string in k} for f in features]

        chosen_features = super().__call__(
            filter_batch("chosen_", features), return_tensors=return_tensors, separator_id=separator_id
        )
        rejected_features = super().__call__(
            filter_batch("rejected_", features), return_tensors=return_tensors, separator_id=separator_id
        )

        chosen_valid = chosen_features.pop("_num_valid_seqs")
        rejected_valid = rejected_features.pop("_num_valid_seqs")
        chosen_wasted = chosen_features.pop("_wasted_tokens_from_truncation")
        rejected_wasted = rejected_features.pop("_wasted_tokens_from_truncation")
        chosen_dropped = chosen_features.pop("_sequences_dropped")
        rejected_dropped = rejected_features.pop("_sequences_dropped")
        num_valid = min(chosen_valid, rejected_valid)

        if num_valid < chosen_valid and "cu_seq_lens_q" in chosen_features:
            chosen_features["cu_seq_lens_q"] = chosen_features["cu_seq_lens_q"][: num_valid + 1]
            chosen_features["cu_seq_lens_k"] = chosen_features["cu_seq_lens_k"][: num_valid + 1]
            boundary = chosen_features["cu_seq_lens_k"][-1].item()
            chosen_features["labels"][:, boundary:] = -100
        if num_valid < rejected_valid and "cu_seq_lens_q" in rejected_features:
            rejected_features["cu_seq_lens_q"] = rejected_features["cu_seq_lens_q"][: num_valid + 1]
            rejected_features["cu_seq_lens_k"] = rejected_features["cu_seq_lens_k"][: num_valid + 1]
            boundary = rejected_features["cu_seq_lens_k"][-1].item()
            rejected_features["labels"][:, boundary:] = -100

        total_sequences = len(features)
        sequences_dropped = max(chosen_dropped, rejected_dropped)
        result = {
            "_wasted_tokens_from_truncation": chosen_wasted + rejected_wasted,
            "_sequences_dropped": sequences_dropped,
            "_sequences_dropped_pct": 100.0 * sequences_dropped / total_sequences if total_sequences > 0 else 0.0,
        }
        for k in chosen_features:
            result["chosen_" + k] = chosen_features[k]
        for k in rejected_features:
            result["rejected_" + k] = rejected_features[k]
        if "index" in features[0]:
            all_indices = torch.tensor([f["index"] for f in features])
            result["index"] = all_indices[:num_valid]
        return result


def calculate_per_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = torch.full_like(labels, -100)
    shifted_labels[:, :-1] = labels[:, 1:]

    logits = logits.to(torch.float32)
    safe = shifted_labels.clamp(min=0)
    mask = (shifted_labels != -100).float()
    return torch.gather(logits.log_softmax(-1), 2, safe.unsqueeze(2)).squeeze(2) * mask


# - dpo concatenation  for padding free
def concatenated_inputs(
    batch: dict[str, list | torch.Tensor], tag: str = "concatenated_"
) -> tuple[dict[str, torch.Tensor], int]:
    chosen_features, rejected_features = {}, {}
    for k in batch:
        if k.startswith("chosen_"):
            chosen_features[k.replace("chosen_", "")] = batch[k]
        else:
            rejected_features[k.replace("rejected_", "")] = batch[k]

    # - need to return chosen
    ret = {f"{tag}input_ids": torch.cat([chosen_features["input_ids"], rejected_features["input_ids"]], dim=-1)}
    if "labels" in chosen_features:
        ret[f"{tag}labels"] = torch.cat([chosen_features["labels"], rejected_features["labels"]], dim=-1)

    if "cu_seq_lens_q" in chosen_features:
        chosen_input_len = chosen_features["input_ids"].shape[-1]
        ret[f"{tag}cu_seq_lens_q"] = torch.cat(
            [chosen_features["cu_seq_lens_q"], rejected_features["cu_seq_lens_q"][1:] + chosen_input_len]
        )
        ret[f"{tag}cu_seq_lens_k"] = torch.cat(
            [chosen_features["cu_seq_lens_k"], rejected_features["cu_seq_lens_k"][1:] + chosen_input_len]
        )
        ret[f"{tag}max_length_q"] = max(chosen_features["max_length_q"], rejected_features["max_length_q"])
        ret[f"{tag}max_length_k"] = max(chosen_features["max_length_k"], rejected_features["max_length_k"])

    if "position_ids" in chosen_features:
        ret[f"{tag}position_ids"] = torch.cat(
            [chosen_features["position_ids"], rejected_features["position_ids"]], dim=-1
        )

    if "seq_idx" in chosen_features:
        chosen_num_seqs = len(chosen_features["cu_seq_lens_k"]) - 1
        ret[f"{tag}seq_idx"] = torch.cat(
            [chosen_features["seq_idx"], rejected_features["seq_idx"] + chosen_num_seqs], dim=-1
        )

    return ret, len(chosen_features["cu_seq_lens_k"]) - 1


# for dpo - padding free
def get_batch_logps(
    per_token_logps: torch.Tensor, labels: torch.Tensor, cu_seq_lens: torch.Tensor, average_log_prob: bool = False
) -> torch.Tensor:
    per_token_logps = per_token_logps[:, :-1]
    loss_mask = labels[:, 1:] != -100

    cu_seq_lens = cu_seq_lens.clone() - 1
    cu_seq_lens[0] = 0

    num_seqs = cu_seq_lens.shape[0] - 1
    seq_len = per_token_logps.shape[1]
    positions = torch.arange(seq_len, device=cu_seq_lens.device)
    segment_ids = (positions.unsqueeze(0) >= cu_seq_lens[:-1].unsqueeze(1)).sum(0) - 1
    segment_ids = segment_ids.clamp(0, num_seqs - 1)

    masked_logps = (per_token_logps * loss_mask.float()).squeeze(0)
    mask_float = loss_mask.float().squeeze(0)

    segment_sums = torch.zeros(num_seqs, device=masked_logps.device, dtype=masked_logps.dtype)
    segment_sums.scatter_add_(0, segment_ids, masked_logps)

    if average_log_prob:
        segment_counts = torch.zeros(num_seqs, device=mask_float.device, dtype=mask_float.dtype)
        segment_counts.scatter_add_(0, segment_ids, mask_float)
        return segment_sums / segment_counts.clamp(min=1)
    return segment_sums
