"""Audit the retained tiny multi-node/judge run without issuing any new model requests."""

import argparse
import collections
import hashlib
import importlib
import json
import math
from pathlib import Path

from scripts.miles import audit_workflow

from open_instruct.miles import general_judge, workflow

require = audit_workflow.require


def audit(root):
    root = Path(root)
    report = audit_workflow.audit(root, counters_only=True)
    plan = json.loads((root / "resolved-plan.json").read_text())
    miles = plan["miles"]
    attempts = [path for path in (root / "cluster").iterdir() if (path / "complete.json").is_file()]
    require(len(attempts) == 1, "Expected one completed cluster attempt")
    attempt = attempts[0]
    placements = [json.loads(p.read_text()) for p in sorted(attempt.glob("placement-*.json"))]
    require(len(placements) == 2 and len({p["address"] for p in placements}) == 2, "Expected distinct physical nodes")
    require(sum(len(p["ray_devices"]) for p in placements) == 3, "Wrong Ray policy GPU count")
    for placement in placements:
        judges = [device for devices in placement["judge_devices"].values() for device in devices]
        require(not set(judges) & set(placement["ray_devices"]), "Judge device overlaps policy pool")
    cleanup = [json.loads(p.read_text()) for p in attempt.glob("cleanup-*.json")]
    require(len(cleanup) == 2 and all(p.get("complete") for p in cleanup), "Missing replica cleanup")
    canaries = json.loads((attempt / "judge-canaries.json").read_text())
    require(len(canaries) == 4, "Both rubric controls must have completed")
    torch = importlib.import_module("torch")
    prepared = {
        split: {row["input"]: row for row in audit_workflow.read_jsonl(path)}
        for split, path in (
            ("train", root / "prepared/data/train.jsonl"),
            ("eval", Path(miles["eval_prompt_data"][1])),
        )
    }
    domains, bindings, generations, dumps = collections.Counter(), collections.Counter(), [], []
    for label, clock, split in ((0, 0, "train"), (1, 1, "train"), ("eval_0", 0, "eval"), ("eval_1", 2, "eval")):
        path = Path(miles["save_debug_rollout_data"].format(rollout_id=label))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        samples = payload["samples"]
        require(len(samples) == (16 if split == "train" else len(prepared[split])), "Unexpected sample count")
        groups = collections.Counter()
        for sample in samples:
            metadata = sample["metadata"]
            row = prepared[split][sample["prompt"]]
            require(sample["prompt"] == row["input"] and sample["label"] == row["label"], "Prompt/label changed")
            require(metadata["verifiers"] == row["metadata"]["verifiers"], "Verifier targets changed")
            length = sample["response_length"]
            prompt = sample["tokens"][:-length]
            digest = hashlib.sha256((json.dumps(prompt, sort_keys=True) + "\n").encode()).hexdigest()
            require(digest == row["metadata"]["run_prompt_token_ids_sha256"], "Tokenizer/template mismatch")
            require(0 < length <= miles["rollout_max_response_len"], "Invalid response length")
            versions = {int(v) for v in sample["weight_versions"]}
            require(
                len(versions) == 1 and 0 <= clock - next(iter(versions)) <= (1 if split == "train" else 0),
                "Policy version mismatch",
            )
            components = metadata["reward_components"]
            total = sum(c["score"] * c["weight"] for c in components)
            require(
                math.isfinite(total) and total == sample["reward"], "Reward components do not equal the trained reward"
            )
            for component in components:
                name = component["name"]
                if split == "train":
                    domains[name] += 1
                if name.startswith("general-"):
                    diagnostic = metadata["verifier_diagnostics"][name]
                    require(
                        diagnostic["context_checked"] and diagnostic["model"] == "Qwen/Qwen3-32B",
                        "Wrong judge contract",
                    )
                    require(diagnostic["binding"]["judge"] == "general", "Wrong named service")
                    require(
                        general_judge.parse_judge_response(diagnostic["raw_reply"])[1] == component["score"],
                        "Retained grade does not parse to its reward",
                    )
                    if split == "train":
                        bindings[name] += 1
                    generations.append(
                        {
                            "split": split,
                            "update": clock,
                            "binding": name,
                            "response": sample["response"],
                            "score": component["score"],
                            "diagnostics": diagnostic,
                        }
                    )
            groups[sample.get("group_index")] += 1
        if split == "train":
            require(len(groups) == 8 and set(groups.values()) == {2}, "Prompt groups have wrong multiplicity")
        dumps.append({"path": str(path), "sha256": audit_workflow.digest(path), "samples": len(samples)})
    require(set(bindings) == {"general-quality", "general-quality_ref"}, "No trained responses from one judge binding")
    require({"math", "ifeval", "code", "code_stdio"} <= set(domains), "Missing training-domain coverage")
    report.update(
        passed=True,
        qualification="tiny_multinode_named_judges",
        full_sample_audit=True,
        training_verifier_counts=dict(domains),
        training_judge_counts=dict(bindings),
        placements=placements,
        canaries=canaries,
        dump_files=dumps,
        judged_generations=generations,
    )
    report["limitations"] = [
        "Two EP2 updates, not EP8/long-context/learning or throughput qualification.",
        "Judge scores reparsed from retained replies; stochastic judges were not called again.",
        "Deterministic reward components checked for accounting; not independently re-executed here.",
        "No checkpoint/restart or injected distributed failure qualification.",
    ]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except Exception as error:
        workflow.write_json(args.report, {"passed": False, "error": str(error)})
        raise
    workflow.write_json(args.report, result)
    print(json.dumps({"passed": True, "report": str(args.report), "domains": result["training_verifier_counts"]}))
