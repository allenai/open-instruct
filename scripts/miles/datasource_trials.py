"""Bounded datasource qualification with the real open-instruct reward adapter.

Examples (inside the pinned MILES image)::

    python -m scripts.miles.datasource_trials fixtures /output/verifier-fixtures
    python -m scripts.miles.datasource_trials prepare /weka/trial --task math --hf /weka/model-hf
    python -m scripts.miles.datasource_trials run /weka/trial

Preparation pins public datasets, strips reference answers, renders the supplied
checkpoint's chat template once, and reserves 16 of 24 prompts for evaluation.
The run uses the qualified SFT topology: two Core EP2 GPUs and one serving GPU.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import inspect
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import ray
import tomllib
import torch
from datasets import load_dataset
from miles.utils import arguments
from scripts.miles import sft_gsm8k
from transformers import AutoTokenizer

from open_instruct.ground_truth_utils import IFEvalVerifier, IFEvalVerifierOld, MathVerifier
from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.miles import rewards
from open_instruct.miles.driver import train
from open_instruct.miles.rewards import registered_reward

TASK_ROOT = Path(__file__).resolve().parents[2] / "configs/miles/tasks"
VERIFIERS = {"math": MathVerifier, "ifeval_old": IFEvalVerifierOld, "ifeval": IFEvalVerifier}


def task_spec(name):
    if name not in ("math", "ifeval"):
        raise ValueError(f"Unknown qualification task: {name}")
    with (TASK_ROOT / f"{name}.toml").open("rb") as stream:
        spec = tomllib.load(stream)
    if len(spec["revision"]) != 40 or any(c not in "0123456789abcdef" for c in spec["revision"]):
        raise ValueError("Dataset revision must be an immutable commit SHA")
    if spec["verifier"] not in VERIFIERS:
        raise ValueError("Task must select a known verifier")
    return spec


def normalize_target(value, verifier):
    if verifier == "math":
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Math target must be a nonempty answer string")
        return value
    if verifier == "ifeval_old":
        constraint = json.loads(value) if isinstance(value, str) else value
        if not isinstance(constraint, dict) or not isinstance(constraint.get("func_name"), str):
            raise ValueError(
                "Legacy IF target must contain func_name; modern instruction lists are a different format"
            )
        function = IF_FUNCTIONS_MAP.get(constraint["func_name"])
        if function is None:
            raise ValueError(f"Unknown legacy IF function: {constraint['func_name']}")
        kwargs = {key: value for key, value in constraint.items() if key != "func_name" and value is not None}
        try:
            inspect.signature(function).bind("response", **kwargs)
        except TypeError as error:
            raise ValueError(f"Invalid legacy IF arguments: {error}") from error
        # The legacy verifier pops func_name from dictionary labels. Serializing
        # here gives every invocation a fresh object, including repeated audits.
        return json.dumps(constraint, sort_keys=True)
    raise ValueError(f"Dataset preparation does not support {verifier}")


def prepare_rows(rows, tokenizer, spec, max_prompt_tokens=2048, select_bounded=False):
    prepared = []
    for index, row in enumerate(rows):
        if select_bounded and index >= 256:
            break
        messages = row["messages"]
        end = next((i for i, message in enumerate(messages) if message["role"] == "assistant"), len(messages))
        messages = messages[:end]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError(f"Source row {index} must end in a user prompt before its reference answer")
        target = normalize_target(row["ground_truth"], spec["verifier"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if tokens > max_prompt_tokens:
            if select_bounded:
                continue
            raise ValueError(f"Source row {index} has {tokens} prompt tokens; limit is {max_prompt_tokens}")
        prepared.append(
            dict(
                input=prompt,
                label=target,
                metadata={
                    "source_row": index,
                    "source_dataset": spec["dataset"],
                    "source_revision": spec["revision"],
                    "query": messages[-1]["content"],
                    "prompt_tokens": tokens,
                    "verifiers": [{"name": spec["verifier"], "target": target}],
                },
            )
        )
        if select_bounded and len(prepared) == 24:
            break
    if len(prepared) != 24 or len({row["input"] for row in prepared}) != 24:
        raise ValueError("Qualification requires exactly 24 unique prompts")
    return prepared


def write_registry(root, names):
    registry = {name: {"factory": f"open_instruct.ground_truth_utils.{VERIFIERS[name].__name__}"} for name in names}
    (root / "verifiers.json").write_text(json.dumps(registry, indent=2) + "\n")


def prepare(root, task, hf, source_jsonl=None, local=False):
    spec = task_spec(task)
    if root.exists():
        raise ValueError("Use a new output directory to preserve trial provenance")
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError("Supply an HF descriptor with an explicit chat template")
    if source_jsonl is None:
        rows = list(load_dataset(spec["dataset"], revision=spec["revision"], split="train[:256]", token=False))
        source = {"kind": "huggingface", "dataset": spec["dataset"], "revision": spec["revision"]}
    else:
        raw = source_jsonl.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        source = {"kind": "local_jsonl", "path": str(source_jsonl), "sha256": hashlib.sha256(raw).hexdigest()}
    prepared = prepare_rows(rows, tokenizer, spec, max_prompt_tokens=480 if local else 2048, select_bounded=True)
    if source_jsonl is not None:
        for row in prepared:
            row["metadata"].pop("source_dataset")
            row["metadata"].pop("source_revision")
            row["metadata"]["source_sha256"] = source["sha256"]
    root.mkdir(parents=True)
    hashes = {}
    for name, subset in (("train", prepared[:8]), ("eval", prepared[8:])):
        raw = "".join(json.dumps(row) + "\n" for row in subset)
        (root / f"{name}.jsonl").write_text(raw)
        hashes[name] = hashlib.sha256(raw.encode()).hexdigest()
    write_registry(root, [spec["verifier"]])
    report = dict(
        task=task,
        profile="local" if local else "sft",
        verifier=spec["verifier"],
        hf=str(hf.resolve()),
        source=source,
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        prepared_sha256=hashes,
        train_rows=[row["metadata"]["source_row"] for row in prepared[:8]],
        eval_rows=[row["metadata"]["source_row"] for row in prepared[8:]],
        selection="First 24 prompts within token budget, scanning at most 256 source rows",
        interpretation="Held out from these updates only; not a decontaminated benchmark or learning claim",
    )
    (root / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DATASOURCE_PREPARED", json.dumps(report), flush=True)
    return report


def configuration(root):
    report = json.loads((root / "preparation.json").read_text())
    config = sft_gsm8k.configuration(root)
    config = dataclasses.replace(config, core=dataclasses.replace(config.core, diagnostic_interval=1))
    config.miles.update(
        hf_checkpoint=report["hf"],
        eval_prompt_data=[report["task"], str(root / "eval.jsonl")],
        sglang_cuda_graph_backend_decode="full",
        sglang_cuda_graph_max_bs_decode=4,
        sglang_cuda_graph_backend_prefill="disabled",
    )
    config.miles.pop("sglang_disable_cuda_graph")
    if report.get("profile") == "local":
        # Small public models only. This qualifies the data/reward path, not
        # useful task accuracy or the full SFT model's memory requirements.
        config = dataclasses.replace(
            config,
            core=dataclasses.replace(
                config.core,
                expert_parallel_size=1,
                attention_backend="torch",
                max_sequence_length=512,
                activation_checkpointing=False,
            ),
        )
        for key in (
            "sglang_cuda_graph_backend_decode",
            "sglang_cuda_graph_max_bs_decode",
            "sglang_cuda_graph_backend_prefill",
        ):
            config.miles.pop(key)
        config.miles.update(
            actor_num_gpus_per_node=1,
            num_gpus_per_node=1,
            colocate=True,
            rollout_max_response_len=32,
            eval_max_response_len=32,
            rollout_max_context_len=512,
            sglang_context_length=512,
            sglang_max_total_tokens=4096,
            sglang_max_mamba_cache_size=16,
            sglang_mem_fraction_static=0.2,
            sglang_attention_backend="torch_native",
            sglang_disable_cuda_graph=True,
        )
    return config


def verify_preparation(root):
    report = json.loads((root / "preparation.json").read_text())
    for kind, digest in report["prepared_sha256"].items():
        if hashlib.sha256((root / f"{kind}.jsonl").read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared {kind} dataset changed after preparation")
    return report


async def direct_score(verifier_name, response, target, query=None):
    if verifier_name == "math":
        # Symbolic equivalence uses SIGALRM and requires a separate ANTLR
        # version from the trainer. Bypass the registry while retaining that
        # isolation for the independent audit invocation.
        result = await rewards.isolated_verifier_call(
            {"factory": "open_instruct.ground_truth_utils.MathVerifier"}, [], response, target, query=query
        )
        return result.score
    return VERIFIERS[verifier_name]()([], response, target, query=query).score


async def audit_samples(root, name, samples, rows, version, response_cap, verifier_name):
    expected = {row["input"]: row for row in rows}
    counts = Counter(sample["prompt"] for sample in samples)
    multiplicity = 1 if name.startswith("eval") else 4
    if len(samples) != 16 or set(counts) != set(expected) or any(n != multiplicity for n in counts.values()):
        raise ValueError(f"{name}: wrong prompt membership or sample multiplicity")
    args = SimpleNamespace(olmo_core=SimpleNamespace(reward_config=str(root / "verifiers.json")))
    rewards = []
    groups = defaultdict(list)
    for sample in samples:
        row = expected[sample["prompt"]]
        if sample["label"] != row["label"] or sample["metadata"]["verifiers"] != row["metadata"]["verifiers"]:
            raise ValueError(f"{name}: changed verifier target")
        if not sample["weight_versions"] or set(sample["weight_versions"]) != {str(version)}:
            raise ValueError(f"{name}: stale or missing policy version")
        logprobs = sample["rollout_log_probs"]
        if len(logprobs) != sample["response_length"] or not all(math.isfinite(p) for p in logprobs):
            raise ValueError(f"{name}: invalid response log probabilities")
        if not 0 < sample["response_length"] <= response_cap:
            raise ValueError(f"{name}: invalid response length")
        # Reconstruct reward inputs from the immutable prepared rows. Compare
        # both the actual async bridge and a direct verifier call to the dump.
        reconstructed = SimpleNamespace(
            prompt=sample["prompt"],
            response=sample["response"],
            response_length=sample["response_length"],
            tokens=sample["tokens"],
            metadata=json.loads(json.dumps(row["metadata"])),
        )
        score = await direct_score(verifier_name, sample["response"], row["label"], query=row["metadata"]["query"])
        bridge_score = await registered_reward(args, reconstructed)
        if not math.isfinite(sample["reward"]) or score != sample["reward"] or bridge_score != score:
            raise ValueError(f"{name}: reward differs from independent verification")
        rewards.append(score)
        groups[sample["prompt"]].append(score)
    return dict(
        samples=len(samples),
        mean_reward=sum(rewards) / len(rewards),
        fully_correct=sum(score == 1 for score in rewards),
        mixed_reward_groups=sum(min(group) != max(group) for group in groups.values()),
        responses_at_token_cap=sum(sample["response_length"] == response_cap for sample in samples),
        policy_version=version,
    )


async def audit_rollouts(root, preparation, rows, argv):
    reports = {}
    for name, version in (("0", 0), ("1", 1), ("eval_0", 0), ("eval_1", 2)):
        is_eval = name.startswith("eval")
        selected = rows["eval"] if is_eval else rows["train"][int(name) * 4 : (int(name) + 1) * 4]
        cap_flag = "--eval-max-response-len" if is_eval else "--rollout-max-response-len"
        cap = int(argv[argv.index(cap_flag) + 1])
        samples = torch.load(root / f"rollouts/{name}.pt", weights_only=False)["samples"]
        reports[name] = await audit_samples(root, name, samples, selected, version, cap, preparation["verifier"])
    return reports


def audit(root):
    preparation = verify_preparation(root)
    rows = {
        kind: [json.loads(line) for line in (root / f"{kind}.jsonl").read_text().splitlines()]
        for kind in ("train", "eval")
    }
    argv = json.loads((root / "arguments.json").read_text())
    reports = asyncio.run(audit_rollouts(root, preparation, rows, argv))
    publication = [json.loads(line) for line in (root / "metrics/publication.jsonl").read_text().splitlines()]
    core = json.loads(argv[argv.index("--olmo-core-config") + 1]) if "--olmo-core-config" in argv else {}
    interval = core.get("diagnostic_interval", 0)
    expected_versions = [0]
    diagnostic_publications = []
    for version in (1, 2):
        expected_versions.append(version)
        if interval and version % interval == 0:
            expected_versions.append(version)
            diagnostic_publications.append(len(expected_versions) - 1)
    if [entry["version"] for entry in publication] != expected_versions:
        raise ValueError(
            "Expected initial publication, two optimizer updates, and configured diagnostic republications"
        )
    expected_repeated = [False] + [a == b for a, b in zip(expected_versions, expected_versions[1:])]
    if interval and [entry.get("repeated_version") for entry in publication] != expected_repeated:
        raise ValueError("Diagnostic publications must mark repeated policy versions explicitly")
    report = dict(
        passed=True,
        task=preparation["task"],
        results=reports,
        publication=publication,
        optimizer_steps=2,
        diagnostic_republications=len(diagnostic_publications),
        diagnostic_republication_seconds=sum(
            publication[i].get("total_seconds", 0.0) for i in diagnostic_publications
        ),
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DATASOURCE_AUDIT_PASSED", json.dumps(report), flush=True)
    return report


def run(root, validate_only=False):
    verify_preparation(root)
    config = configuration(root)
    sys.argv = ["datasource-trial", *config.arguments()]
    args = arguments.parse_args()
    if args.save_interval is not None:
        raise ValueError("The bounded datasource trial must not save optimizer checkpoints")
    if validate_only:
        print("DATASOURCE_CONFIG_VALIDATED")
        return
    (root / "arguments.json").write_text(json.dumps(config.arguments(), indent=2) + "\n")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    ray.init(
        num_gpus=config.miles["num_gpus_per_node"], num_cpus=16, include_dashboard=False, object_store_memory=1024**3
    )
    try:
        asyncio.run(train(args))
    finally:
        ray.shutdown()
    audit(root)


def fixture_cases():
    modern = str(
        [
            {
                "instruction_id": ["keywords:existence", "change_case:english_lowercase"],
                "kwargs": [{"keywords": ["ocean"]}, {}],
            }
        ]
    )
    return [
        ("math", r"\frac{1}{2}", r"The answer is $\boxed{\frac{2}{4}}$.", 1.0),
        ("math", "2", r"$\boxed{3}$", 0.0),
        ("ifeval_old", json.dumps({"func_name": "validate_lowercase", "N": None}), "the ocean is blue.", 1.0),
        ("ifeval_old", json.dumps({"func_name": "validate_lowercase"}), "The ocean is blue.", 0.0),
        ("ifeval", modern, "<think>CAPITAL THINKING</think>\nthe ocean is blue.", 1.0),
        ("ifeval", modern, "the sky is blue.", 0.5),
        ("ifeval", modern, "THE SKY IS BLUE.", 0.0),
    ]


async def fixture_report(root):
    root.mkdir(parents=True, exist_ok=True)
    write_registry(root, VERIFIERS)
    args = SimpleNamespace(olmo_core=SimpleNamespace(reward_config=str(root / "verifiers.json")))
    results = []
    for name, target, response, expected in fixture_cases():
        sample = SimpleNamespace(
            prompt="Deterministic verifier fixture",
            tokens=[1, 2],
            response_length=1,
            response=response,
            metadata={"verifiers": [{"name": name, "target": target}]},
        )
        before = json.dumps(sample.metadata["verifiers"], sort_keys=True)
        actual = await registered_reward(args, sample)
        repeated = await registered_reward(args, sample)
        direct = await direct_score(name, response, target)
        if actual != expected or direct != expected or repeated != expected:
            raise ValueError(f"{name}: expected {expected}, bridge={actual}, direct={direct}, repeated={repeated}")
        if json.dumps(sample.metadata["verifiers"], sort_keys=True) != before:
            raise ValueError("Verifier mutated persistent targets")
        results.append(dict(verifier=name, expected=expected, bridge=actual, direct=direct, repeated=repeated))
    report = dict(passed=True, cases=results, interpretation="Verifier/adapter checks only; no model training")
    (root / "fixtures.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DATASOURCE_FIXTURES_PASSED", json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fixtures", "prepare", "validate", "run", "audit"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", choices=["math", "ifeval"])
    parser.add_argument("--hf", type=Path)
    parser.add_argument("--local", action="store_true", help="One-GPU tiny-model trial with a 32-token response cap")
    parser.add_argument(
        "--source-jsonl", type=Path, help="Offline input snapshot; records its hash instead of claiming HF provenance"
    )
    options = parser.parse_args()
    if options.command == "fixtures":
        asyncio.run(fixture_report(options.output))
    elif options.command == "prepare":
        if not options.task or not options.hf:
            parser.error("prepare requires --task and --hf")
        prepare(options.output, options.task, options.hf, options.source_jsonl, options.local)
    elif options.command == "audit":
        audit(options.output)
    else:
        run(options.output, validate_only=options.command == "validate")


if __name__ == "__main__":
    main()
