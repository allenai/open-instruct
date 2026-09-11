"""Read-only CPU analysis of the original 100-update paired GSM8K artifacts."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import torch

UPDATES = 100
EVAL_STEPS = (0, 20, 40, 60, 80, 100)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_digest(tokens):
    return hashlib.sha256((json.dumps(tokens, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def lengths(values):
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p90": percentile(values, 0.9),
    }


def summarize(rows, *, training):
    result = {
        "length": lengths([row["response_tokens"] for row in rows]),
        "correct_length": lengths([row["response_tokens"] for row in rows if row["reward"]]),
        "wrong_length": lengths([row["response_tokens"] for row in rows if not row["reward"]]),
        "mean_reward": mean(row["reward"] for row in rows),
        "cap_fraction": mean(row["at_cap"] for row in rows),
        "truncated_fraction": mean(row["truncated"] for row in rows),
    }
    if training:
        groups = defaultdict(list)
        for row in rows:
            groups[(row["rollout"], row["id"])].append(row["reward"])
        assert all(len(rewards) == 4 for rewards in groups.values()), "Wrong GRPO group size"
        zero_groups = sum(min(rewards) == max(rewards) for rewards in groups.values())
        result["zero_policy_advantage_group_fraction"] = zero_groups / len(groups)
        result["zero_policy_advantage_sample_fraction"] = zero_groups * 4 / len(rows)
        result["prompt_groups"] = len(groups)
    return result


def validate_sample(sample, prepared, proof, *, version, rollout, cap=4096):
    key = sample["metadata"]["prepared_sample_id"]
    assert key == prepared["metadata"]["prepared_sample_id"], "Prompt membership mismatch"
    assert sample["prompt"] == prepared["input"] and sample["label"] == prepared["label"], "Prompt/label changed"
    assert sample["metadata"].get("verifiers") == prepared["metadata"].get("verifiers"), "Verifier changed"
    versions = sample["weight_versions"]
    assert versions and all(str(value) == str(version) for value in versions), "Policy version mismatch"
    size = sample["response_length"]
    assert type(size) is int and 0 < size <= cap and size < len(sample["tokens"]), "Invalid response length"
    assert token_digest(sample["tokens"][:-size]) == proof["token_ids_sha256"], "Prompt token proof mismatch"
    probabilities = sample["rollout_log_probs"]
    assert len(probabilities) == size and all(math.isfinite(value) for value in probabilities), (
        "Invalid log probabilities"
    )
    assert sample["status"] in ("completed", "truncated") and not sample.get("remove_sample", False), "Failed sample"
    response = re.sub(r"(\d),(\d)", r"\1\2", sample["response"])
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response)
    extracted = numbers[-1] if numbers else response
    score = float(str(extracted).lower() == str(prepared["label"]).lower())
    assert isinstance(sample["reward"], int | float) and sample["reward"] == score, "Independent reward mismatch"
    return {
        "id": key,
        "rollout": rollout,
        "response_tokens": size,
        "reward": score,
        "at_cap": size == cap,
        "truncated": sample["status"] == "truncated",
        "response_sha256": hashlib.sha256(sample["response"].encode()).hexdigest(),
    }


def collect(root):
    torch.set_num_threads(1)
    preparation = json.loads((root / "preparation.json").read_text())
    for name, expected in preparation["files"].items():
        assert sha256(root / name) == expected, f"Preparation file changed: {name}"
    prepared = {
        partition: [json.loads(line) for line in (root / f"{partition}.jsonl").read_text().splitlines() if line]
        for partition in ("train", "eval")
    }
    assert len(prepared["train"]) == 400 and len(prepared["eval"]) == 128
    proofs = {row["prepared_sample_id"]: row for part in preparation["partitions"].values() for row in part["rows"]}
    result = {
        "schema_version": 1,
        "root": str(root),
        "preparation_sha256": sha256(root / "preparation.json"),
        "loader": "torch.load(map_location='cpu', weights_only=True); one CPU thread; no GPU operations",
        "arms": {},
    }
    for backend, directory, offset in (("core", "core/rollouts", 0), ("megatron", "megatron-r3/rollout_data", 1)):
        arm = {"training": [], "evaluation": [], "version_offset": offset}
        result["arms"][backend] = arm
        for training in (True, False):
            for step in range(UPDATES) if training else EVAL_STEPS:
                rollout = step if training or step == 0 else step - 1
                name = f"{rollout}.pt" if training else f"eval_{rollout}.pt"
                path = root / directory / name
                payload = torch.load(path, map_location="cpu", weights_only=True)
                assert payload["rollout_id"] == rollout, "Rollout filename mismatch"
                selected = prepared["train"][step * 4 : (step + 1) * 4] if training else prepared["eval"]
                expected = {row["metadata"]["prepared_sample_id"]: row for row in selected}
                counts = Counter(sample["metadata"]["prepared_sample_id"] for sample in payload["samples"])
                assert counts == Counter({key: 4 if training else 1 for key in expected}), (
                    "Wrong batch membership/count"
                )
                rows = [
                    validate_sample(
                        sample,
                        expected[sample["metadata"]["prepared_sample_id"]],
                        proofs[sample["metadata"]["prepared_sample_id"]],
                        version=step + offset,
                        rollout=rollout,
                    )
                    for sample in payload["samples"]
                ]
                entry = {"step": step, "file": name, "sha256": sha256(path), "samples": rows}
                entry.update(summarize(rows, training=training))
                arm["training" if training else "evaluation"].append(entry)
        all_rows = [row for entry in arm["training"] for row in entry["samples"]]
        arm["overall"] = summarize(all_rows, training=True)
        arm["zero_policy_advantage_updates"] = sum(
            entry["zero_policy_advantage_sample_fraction"] == 1 for entry in arm["training"]
        )
        arm["windows"] = []
        for start, end in ((0, 20), (20, 40), (40, 60), (50, 60), (60, 70), (70, 80), (60, 80), (80, 100)):
            rows = [row for entry in arm["training"][start:end] for row in entry["samples"]]
            arm["windows"].append({"start_inclusive": start, "end_exclusive": end, **summarize(rows, training=True)})
        assert len(arm["training"]) == 100 and len(arm["evaluation"]) == 6 and len(all_rows) == 1600
    result["valid"] = True
    result["interpretation"] = (
        "Descriptive one-run-per-backend evidence. Training rollout index i precedes optimizer update i+1. "
        "Equal binary rewards within a four-sample GRPO group imply zero group-centered policy advantages; "
        "this is reconstructed from rewards, not captured advantage tensors. Auxiliary losses and Adam momentum "
        "can still change parameters. Correct/wrong conditioning and changing training prompts prevent causal inference."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.root), indent=2))


if __name__ == "__main__":
    main()
