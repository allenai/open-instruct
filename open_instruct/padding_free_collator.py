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

    return_flash_attn_kwargs: bool = True
    return_position_ids: bool = True
    return_seq_idx: bool = True
    separator_id: int = -100

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


@dataclass
class TensorDataCollatorWithFlatteningDPO(TensorDataCollatorWithFlattening):
    def __call__(self, features, return_tensors=None, separator_id=None):
        # call the original collator on chosen and rejected separately, then combine
        def filter_batch(match_string, features):
            return [{k.replace(match_string, ""): v for k, v in f.items() if match_string in k} for f in features]

        chosen_features = super().__call__(
            filter_batch("chosen_", features), return_tensors=return_tensors, separator_id=separator_id
        )
        rejected_features = super().__call__(
            filter_batch("rejected_", features), return_tensors=return_tensors, separator_id=separator_id
        )

        result = {}
        for k in chosen_features:
            result["chosen_" + k] = chosen_features[k]
        for k in rejected_features:
            result["rejected_" + k] = rejected_features[k]
        return result


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
        ret[f"{tag}cu_seq_lens_q"] = torch.cat(
            [
                chosen_features["cu_seq_lens_q"],
                rejected_features["cu_seq_lens_q"][1:] + chosen_features["cu_seq_lens_q"][-1],
            ]
        )
        ret[f"{tag}cu_seq_lens_k"] = torch.cat(
            [
                chosen_features["cu_seq_lens_k"],
                rejected_features["cu_seq_lens_k"][1:] + chosen_features["cu_seq_lens_k"][-1],
            ]
        )
        ret[f"{tag}max_length_q"] = max(chosen_features["max_length_q"], rejected_features["max_length_q"])
        ret[f"{tag}max_length_k"] = max(chosen_features["max_length_k"], rejected_features["max_length_k"])

    if "position_ids" in chosen_features:
        ret[f"{tag}position_ids"] = torch.cat(
            [chosen_features["position_ids"], rejected_features["position_ids"]], dim=-1
        )

    if "seq_idx" in chosen_features:
        ret[f"{tag}seq_idx"] = torch.cat(
            [chosen_features["seq_idx"], rejected_features["seq_idx"] + chosen_features["seq_idx"][0, -1]], dim=-1
        )

    return ret, len(chosen_features["cu_seq_lens_k"]) - 1


# for dpo - padding free
def get_batch_logps(
    logits: torch.Tensor, labels: torch.Tensor, cu_seq_lens: torch.Tensor, average_log_prob: bool = False
) -> torch.Tensor:
    assert logits.shape[:-1] == labels.shape

    # - we are going to get crossings at labels / logits
    #   cont batch boundaries, but we assume that the
    #   loss mask == True at those places
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    # there is a labels, logits shift operation above
    cu_seq_lens = cu_seq_lens.clone() - 1
    cu_seq_lens[0] = 0

    splits = cu_seq_lens.diff().tolist()
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    return torch.concat(
        [
            ((ps * mask).sum(-1) / mask.sum(-1) if average_log_prob else (ps * mask).sum(-1))
            for ps, mask in zip(torch.split(per_token_logps, splits, dim=-1), torch.split(loss_mask, splits, dim=-1))
        ]
    )
