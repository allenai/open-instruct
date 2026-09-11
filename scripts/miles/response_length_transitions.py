"""Conditioned descriptive comparisons derived from the validated GSM8K100 audit."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.miles import analyze_response_lengths as analysis


def transitions(before, after):
    left = {row["id"]: row for row in before}
    right = {row["id"]: row for row in after}
    assert len(left) == len(before) and len(right) == len(after) and left.keys() == right.keys()
    groups = defaultdict(list)
    for key, first in left.items():
        last = right[key]
        pair = {"id": key, "before": first["response_tokens"], "after": last["response_tokens"]}
        groups[f"{'correct' if first['reward'] else 'wrong'}_to_{'correct' if last['reward'] else 'wrong'}"].append(
            pair
        )
        if first["reward"] and last["reward"] and not first["at_cap"] and not last["at_cap"]:
            groups["correct_both_and_uncapped_both"].append(pair)
        if not first["at_cap"] and not last["at_cap"]:
            groups["uncapped_both"].append(pair)
    return {
        name: {
            "count": len(rows),
            "before": analysis.lengths([row["before"] for row in rows]),
            "after": analysis.lengths([row["after"] for row in rows]),
            "paired_change": analysis.lengths([row["after"] - row["before"] for row in rows]),
            "pairs": rows,
        }
        for name, rows in groups.items()
    }


def extend(report):
    assert report["valid"]
    result = {
        "arms": {},
        "interpretation": "Conditioning and association only; not causal attribution or significance.",
    }
    for backend, arm in report["arms"].items():
        result["arms"][backend] = extra = {"training_windows": [], "evaluation_transitions": {}, "next_rollout": {}}
        for start, end in ((0, 20), (50, 60), (60, 70), (70, 80), (80, 100)):
            rows = [sample for entry in arm["training"][start:end] for sample in entry["samples"]]
            extra["training_windows"].append(
                {
                    "start_inclusive": start,
                    "end_exclusive": end,
                    "uncapped": analysis.lengths([row["response_tokens"] for row in rows if not row["at_cap"]]),
                    "correct_uncapped": analysis.lengths(
                        [row["response_tokens"] for row in rows if row["reward"] and not row["at_cap"]]
                    ),
                    "wrong_uncapped": analysis.lengths(
                        [row["response_tokens"] for row in rows if not row["reward"] and not row["at_cap"]]
                    ),
                }
            )
        evaluations = {row["step"]: row["samples"] for row in arm["evaluation"]}
        for before, after in ((0, 100), (60, 80), (80, 100), (60, 100)):
            extra["evaluation_transitions"][f"{before}_to_{after}"] = transitions(
                evaluations[before], evaluations[after]
            )
        for uniform in (False, True):
            pairs = [
                (first, last)
                for first, last in zip(arm["training"][:-1], arm["training"][1:], strict=True)
                if (first["zero_policy_advantage_sample_fraction"] == 1) == uniform
            ]
            extra["next_rollout"]["after_all_uniform_groups" if uniform else "after_any_mixed_group"] = {
                "count": len(pairs),
                "mean_next_length_change": mean(
                    last["length"]["mean"] - first["length"]["mean"] for first, last in pairs
                ),
                "mean_next_reward_change": mean(last["mean_reward"] - first["mean_reward"] for first, last in pairs),
                "warning": "Next rollout uses different questions and fresh stochastic samples; not an update-effect estimate.",
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(extend(json.loads(args.report.read_text())), indent=2))


if __name__ == "__main__":
    main()
