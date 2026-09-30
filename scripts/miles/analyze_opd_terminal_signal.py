"""Describe sampled OPD advantages by response length and terminal token on CPU.

These are raw detached advantages, not gradients or a counterfactual estimate of
stopping probability. Rank-local snapshots need not represent an entire update.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    ranks = []
    for values in zip(*pairs, strict=True):
        _, inverse, counts = torch.as_tensor(values, dtype=torch.float64).unique(
            sorted=True, return_inverse=True, return_counts=True
        )
        average_ranks = counts.cumsum(0).double() - (counts + 1).double() / 2
        ranks.append(average_ranks[inverse])
    value = torch.corrcoef(torch.stack(ranks))[0, 1]
    return value.item() if torch.isfinite(value) else None


def repeated_window_fraction(tokens, size=32):
    count = len(tokens) - size + 1
    if count <= 0:
        return None
    return 1 - len({tuple(tokens[i : i + size]) for i in range(count)}) / count


def stats(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not values.numel():
        return {"count": 0}
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "median": values.median().item(),
        "mean_absolute": values.abs().mean().item(),
        "negative_fraction": (values < 0).double().mean().item(),
    }


def summarize(data, eos_ids):
    rows, active_values, eos_values, ordinary_values = [], [], [], []
    max_error = 0.0
    for tokens, length, advantage, mask, teacher, student, truncated in zip(
        *(
            data[key]
            for key in (
                "tokens",
                "response_lengths",
                "advantages",
                "loss_masks",
                "teacher_log_probs",
                "rollout_log_probs",
                "truncated",
            )
        ),
        strict=True,
    ):
        advantage, teacher, student = (torch.as_tensor(x).double() for x in (advantage, teacher, student))
        mask = torch.as_tensor(mask).bool()
        response = torch.as_tensor(tokens)[-length:]
        if not (response.shape == advantage.shape == mask.shape == teacher.shape == student.shape):
            raise ValueError("Snapshot is not a full response-aligned slice")
        if not all(torch.isfinite(x).all() for x in (advantage, teacher, student)):
            raise ValueError("Nonfinite scoring data")
        max_error = max(max_error, (advantage - (teacher - student)).abs().max().item())
        is_eos = torch.zeros_like(mask)
        for token_id in eos_ids:
            is_eos |= response == token_id
        active = advantage[mask]
        active_values.append(active)
        eos_values.append(advantage[mask & is_eos])
        ordinary_values.append(advantage[mask & ~is_eos])
        rows.append(
            {
                "length": length,
                "truncated": bool(truncated),
                "active_tokens": active.numel(),
                "advantage_sum": active.sum().item(),
                "absolute_advantage_sum": active.abs().sum().item(),
                "mean_absolute_advantage": active.abs().mean().item() if active.numel() else None,
                "repeated_32gram_fraction": repeated_window_fraction(response.tolist()),
                "terminal_is_eos": bool(is_eos[-1]),
                "terminal_active": bool(mask[-1]),
                "terminal_advantage": advantage[-1].item(),
            }
        )
    if max_error > 1e-5:
        raise ValueError("Advantages do not match pure teacher-minus-rollout OPD")
    bins = {}
    for lower, upper in [(0, 4096), (4096, 8192), (8192, 16384), (16384, 10**9)]:
        selected = [row for row in rows if lower <= row["length"] < upper]
        count = sum(row["active_tokens"] for row in selected)
        bins[str(lower)] = {
            "responses": len(selected),
            "active_tokens": count,
            "token_mean_advantage": sum(row["advantage_sum"] for row in selected) / count if count else None,
            "token_mean_absolute_advantage": sum(row["absolute_advantage_sum"] for row in selected) / count
            if count
            else None,
        }
    return {
        "responses": len(rows),
        "mean_length": sum(row["length"] for row in rows) / len(rows),
        "truncated_responses": sum(row["truncated"] for row in rows),
        "max_advantage_identity_error": max_error,
        "all_active_tokens": stats(torch.cat(active_values)),
        "active_eos_tokens": stats(torch.cat(eos_values)),
        "active_non_eos_tokens": stats(torch.cat(ordinary_values)),
        "length_bins_lower_inclusive_upper_exclusive": bins,
        "exploratory_response_spearman_with_mean_absolute_advantage": {
            feature: spearman([row[feature] for row in rows], [row["mean_absolute_advantage"] for row in rows])
            for feature in ("length", "repeated_32gram_fraction")
        },
        "response_summaries": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--eos-id", type=int, action="append", required=True)
    args = parser.parse_args()
    result = {"scope": __doc__, "eos_ids": args.eos_id, "snapshots": []}
    for path in args.snapshots:
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved.get("cp_size", 1) != 1:
            raise ValueError("Context-parallel slices require explicit token alignment")
        result["snapshots"].append(
            {
                "source": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rollout_id": saved["rollout_id"],
                "rank": saved["rank"],
                **summarize(saved["rollout_data"], args.eos_id),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
