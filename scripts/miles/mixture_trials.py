"""Run a frozen GSM8K/math/legacy-IF mixture through the existing MILES backend.

Prepare from hash-verified single-source preparation roots. Each of two updates
contains two independent prompts per source and four samples per prompt.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import ray
import torch
from miles.utils import arguments
from scripts.miles import datasource_trials
from transformers import AutoTokenizer

from open_instruct.ground_truth_utils import GSM8KVerifier
from open_instruct.miles import mixture
from open_instruct.miles.driver import train
from open_instruct.miles.rewards import registered_reward

BATCH_PROMPTS = mixture.BATCH_PROMPTS
REGISTRY = {
    "gsm8k": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"},
    "math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"},
    "ifeval_old": {"factory": "open_instruct.ground_truth_utils.IFEvalVerifierOld"},
}


def prepare(root, sources, hf, *, seed=17, local=False):
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    report = mixture.materialize(root, sources, hf, tokenizer, seed=seed, local=local)
    registry = mixture.encoded(REGISTRY)
    (root / "verifiers.json").write_bytes(registry)
    report["registry_sha256"] = mixture.digest(registry)
    (root / "preparation.json").write_bytes(mixture.encoded(report))
    return report


def verify_preparation(root):
    report, rows = mixture.verify(root)
    raw = (root / "verifiers.json").read_bytes()
    if mixture.digest(raw) != report["registry_sha256"] or json.loads(raw) != REGISTRY:
        raise ValueError("Trusted mixture verifier registry changed")
    return report, rows


def configuration(root):
    report = json.loads((root / "preparation.json").read_text())
    config = datasource_trials.configuration(root)
    config.miles.update(
        seed=report["seed"],
        rollout_seed=report["seed"],
        rollout_batch_size=BATCH_PROMPTS,
        global_batch_size=BATCH_PROMPTS * mixture.SAMPLES_PER_PROMPT,
        n_samples_per_prompt=mixture.SAMPLES_PER_PROMPT,
        num_rollout=mixture.UPDATES,
        eval_interval=mixture.UPDATES,
        eval_prompt_data=["mixture", str(root / "eval.jsonl")],
        n_samples_per_eval_prompt=1,
        rollout_max_prompt_len=480 if config.miles.get("colocate") else 2048,
    )
    return config


async def audit_samples(root, name, samples, expected_rows, *, version, cap):
    expected = {row["metadata"]["mixture"]["id"]: row for row in expected_rows}
    is_eval = name.startswith("eval")
    multiplicity = 1 if is_eval else mixture.SAMPLES_PER_PROMPT
    counts, group_ids = Counter(), defaultdict(set)
    rewards_by_source, rewards_by_prompt = defaultdict(list), defaultdict(list)
    at_cap = Counter()
    args = SimpleNamespace(olmo_core=SimpleNamespace(reward_config=str(root / "verifiers.json")))
    for sample in samples:
        marker = sample.get("metadata", {}).get("mixture", {})
        key = marker.get("id")
        if key not in expected:
            raise ValueError(f"{name}: unexpected source identity")
        row = expected[key]
        if (
            sample["prompt"] != row["input"]
            or sample["label"] != row["label"]
            or any(sample["metadata"].get(field) != value for field, value in row["metadata"].items())
        ):
            raise ValueError(f"{name}: changed source metadata, prompt, or target")
        length = sample["response_length"]
        if type(length) is not int or not 0 < length <= cap or length >= len(sample["tokens"]):
            raise ValueError(f"{name}: invalid response length")
        if mixture.digest(mixture.encoded(sample["tokens"][:-length])) != marker["token_ids_sha256"]:
            raise ValueError(f"{name}: prompt token IDs changed")
        scores = sample.get("rollout_log_probs")
        if scores is None or len(scores) != length or not all(math.isfinite(value) for value in scores):
            raise ValueError(f"{name}: invalid behavior log probabilities")
        if not sample.get("weight_versions") or {str(value) for value in sample["weight_versions"]} != {str(version)}:
            raise ValueError(f"{name}: missing or stale policy version")
        if sample.get("status") not in ("completed", "truncated") or sample.get("remove_sample", False):
            raise ValueError(f"{name}: unsuccessful response")
        source = marker["source"]
        verifier = mixture.SOURCES[source]
        direct = (
            GSM8KVerifier()([], sample["response"], row["label"]).score
            if source == "gsm8k"
            else await datasource_trials.direct_score(
                verifier, sample["response"], row["label"], query=row["metadata"].get("query")
            )
        )
        reconstructed = SimpleNamespace(**{**sample, "metadata": json.loads(json.dumps(row["metadata"]))})
        bridge = await registered_reward(args, reconstructed)
        if not math.isfinite(sample["reward"]) or direct != sample["reward"] or bridge != direct:
            raise ValueError(f"{name}: stored, direct, and bridge rewards disagree")
        counts[key] += 1
        if not is_eval:
            group_ids[key].add(sample.get("group_index"))
        rewards_by_prompt[key].append(direct)
        rewards_by_source[source].append(direct)
        at_cap[source] += length == cap
    if counts != Counter({key: multiplicity for key in expected}):
        raise ValueError(f"{name}: incorrect prompt membership or sample multiplicity")
    if not is_eval:
        for position, key in enumerate(expected):
            if group_ids[key] != {int(name) * BATCH_PROMPTS + position}:
                raise ValueError(f"{name}: prompt groups were merged, reordered, or reassigned")
    return {
        "samples": len(samples),
        "policy_version": version,
        "sources": {
            source: {
                "samples": len(rewards_by_source[source]),
                "mean_reward": sum(rewards_by_source[source]) / len(rewards_by_source[source]),
                "prompt_groups": sum(row["metadata"]["mixture"]["source"] == source for row in expected_rows),
                "mixed_reward_groups": sum(
                    min(values) != max(values)
                    for key, values in rewards_by_prompt.items()
                    if expected[key]["metadata"]["mixture"]["source"] == source
                ),
                "responses_at_token_cap": at_cap[source],
            }
            for source in mixture.SOURCES
        },
    }


async def audit_rollouts(root, rows, config):
    reports = {}
    for name, version in (("0", 0), ("1", 1), ("eval_0", 0), ("eval_1", 2)):
        is_eval = name.startswith("eval")
        selected = (
            rows["eval"] if is_eval else rows["train"][int(name) * BATCH_PROMPTS : (int(name) + 1) * BATCH_PROMPTS]
        )
        path = root / f"rollouts/{name}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("rollout_id") != int(name.removeprefix("eval_")):
            raise ValueError(f"{name}: rollout identity differs from filename")
        reports[name] = await audit_samples(
            root,
            name,
            payload["samples"],
            selected,
            version=version,
            cap=config.miles["eval_max_response_len" if is_eval else "rollout_max_response_len"],
        )
        reports[name]["sha256"] = mixture.digest(path.read_bytes())
    return reports


def audit(root):
    report, rows = verify_preparation(root)
    config = configuration(root)
    if json.loads((root / "arguments.json").read_text()) != config.arguments():
        raise ValueError("Mixture run arguments differ from the prepared protocol")
    results = asyncio.run(audit_rollouts(root, rows, config))
    records = [json.loads(line) for line in (root / "metrics/training_contract_rank0.jsonl").read_text().splitlines()]
    if [row["step"] for row in records if row.get("event") == "optimizer"] != [1, 2]:
        raise ValueError("Mixture requires two completed optimizer updates")
    publications = [json.loads(line) for line in (root / "metrics/publication.jsonl").read_text().splitlines()]
    if [row["version"] for row in publications] != [0, 1, 1, 2, 2] or [
        row.get("repeated_version") for row in publications
    ] != [False, False, True, False, True]:
        raise ValueError("Mixture publication/diagnostic sequence differs from the two-update protocol")
    result = {
        "passed": True,
        "preparation_sha256": mixture.digest((root / "preparation.json").read_bytes()),
        "results": results,
        "optimizer_steps": mixture.UPDATES,
        "source_order": report["source_order"],
        "interpretation": "Per-source verification and prompt-group integrity; no learning claim",
    }
    (root / "audit.json").write_bytes(mixture.encoded(result))
    return result


def run(root, *, validate_only=False):
    report, _ = verify_preparation(root)
    tokenizer = AutoTokenizer.from_pretrained(report["hf"], trust_remote_code=True)
    if mixture.digest(tokenizer.chat_template.encode()) != report["template_sha256"]:
        raise ValueError("Checkpoint chat template changed after mixture preparation")
    config = configuration(root)
    sys.argv = ["mixture-trial", *config.arguments()]
    args = arguments.parse_args()
    if (
        args.rollout_shuffle
        or args.apply_chat_template
        or not args.rollout_global_dataset
        or args.save_interval is not None
    ):
        raise ValueError("Mixture requires ordered completion prompts and no optimizer checkpoint saves")
    if validate_only:
        return {"validated": True, "prompts_per_update": BATCH_PROMPTS, "responses_per_update": args.global_batch_size}
    (root / "arguments.json").write_text(json.dumps(config.arguments(), indent=2) + "\n")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    ray.init(
        num_gpus=config.miles["num_gpus_per_node"], num_cpus=16, include_dashboard=False, object_store_memory=1024**3
    )
    try:
        asyncio.run(train(args))
    finally:
        ray.shutdown()
    return audit(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "validate", "run", "audit"])
    parser.add_argument("output", type=Path)
    for source in mixture.SOURCES:
        parser.add_argument(f"--{source}-root", type=Path)
    parser.add_argument("--hf", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        sources = {source: getattr(args, f"{source}_root") for source in mixture.SOURCES}
        if args.hf is None or any(root is None for root in sources.values()):
            parser.error("prepare requires --hf and all three --SOURCE-root arguments")
        result = prepare(args.output, sources, args.hf, seed=args.seed, local=args.local)
    elif args.command == "audit":
        result = audit(args.output)
    else:
        result = run(args.output, validate_only=args.command == "validate")
    print("MIXTURE_" + args.command.upper(), json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
