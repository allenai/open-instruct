"""Compare natural routing on identical, serialized SFT probe inputs.

Each torch.save artifact contains input_ids, doc_lens, valid_mask (bool), and
routes: {layer_name: expert_ids}, with expert_ids shaped (*input_ids.shape, top_k).
Capture one forward per artifact, without route replay or backward recomputation.
Use rank-local inputs/routes when running distributed probes; compare every rank.
"""

import argparse
import json

import torch


def compare_routes(before: dict, after: dict) -> dict:
    for field in ("input_ids", "doc_lens", "valid_mask"):
        left, right = before[field], after[field]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"Probe {field} differs; token positions are not comparable")
    mask = before["valid_mask"]
    if mask.dtype != torch.bool or mask.shape != before["input_ids"].shape or not mask.any():
        raise ValueError("valid_mask must select at least one input token")
    if not before["routes"] or before["routes"].keys() != after["routes"].keys():
        raise ValueError("Missing or inconsistent router layers")
    report = {}
    for name, left in before["routes"].items():
        right = after["routes"][name]
        if left.shape != right.shape or left.shape[:-1] != mask.shape or left.shape[-1] == 0:
            raise ValueError(f"Invalid route shape for {name}")
        for values in (left, right):
            if values.dtype not in (torch.int32, torch.int64) or (values < 0).any():
                raise ValueError(f"Invalid expert IDs for {name}")
            ordered = values.sort(dim=-1).values
            if (ordered[..., 1:] == ordered[..., :-1]).any():
                raise ValueError(f"Repeated expert within a token for {name}")
        left, right = left[mask].long(), right[mask].long()
        left_set, right_set = left.sort(dim=-1).values, right.sort(dim=-1).values
        changed = (left_set != right_set).any(dim=-1)
        reordered = (left != right).any(dim=-1) & ~changed
        # Set overlap handles partially changed top-k selections, unlike rank equality.
        overlap = (left.unsqueeze(-1) == right.unsqueeze(-2)).any(dim=-1).sum().item()
        ids = torch.cat((left.flatten(), right.flatten())).unique(sorted=True)
        report[name] = {
            "tokens": left.shape[0],
            "top_k": left.shape[1],
            "changed_expert_set_count": changed.sum().item(),
            "changed_expert_set_fraction": changed.float().mean().item(),
            "order_only_change_count": reordered.sum().item(),
            "retained_expert_fraction": overlap / left.numel(),
            # Only observed experts are listed: absence does not establish a dead expert.
            "expert_assignment_counts_before": {str(i.item()): (left == i).sum().item() for i in ids},
            "expert_assignment_counts_after": {str(i.item()): (right == i).sum().item() for i in ids},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    before = torch.load(args.before, map_location="cpu", weights_only=True)
    after = torch.load(args.after, map_location="cpu", weights_only=True)
    print(json.dumps(compare_routes(before, after), indent=2))


if __name__ == "__main__":
    main()
