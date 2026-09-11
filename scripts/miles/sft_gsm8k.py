"""Bounded SFT GSM8K qualification using the native Core trainer and MILES."""

import argparse
import asyncio
import hashlib
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ray
import torch
from datasets import load_dataset
from miles.utils import arguments
from transformers import AutoTokenizer

from open_instruct.ground_truth_utils import GSM8KVerifier
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.driver import train

CHECKPOINT_NAME = "olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2"
CHECKPOINT_ROOT = Path("/weka/oe-training-default/robertb/olmo-miles/checkpoints")
HF_SOURCE = CHECKPOINT_ROOT / (CHECKPOINT_NAME + "-hf")
TEMPLATE_SOURCE = CHECKPOINT_ROOT / (CHECKPOINT_NAME + "-megatron/.olmo-miles/hf/chat_template.jinja")
TEMPLATE_SHA256 = "f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b"
DATASET = "ai2-adapt-dev/rlvr_gsm8k_zs"
DATASET_REVISION = "93ffaae6cd2acb8f821f6d4712651320a889b1b9"


def configuration(root):
    return RunConfig(
        CoreConfig(
            row_specialization="dynamic",
            expert_parallel_size=2,
            attention_backend="flash_4",
            max_sequence_length=6144,
            activation_checkpointing=True,
            max_train_rollout_logprob_abs_diff=0.05,
            reward_config=str(root / "verifiers.json"),
        ),
        dict(
            hf_checkpoint=str(root / "hf"),
            actor_num_gpus_per_node=2,
            num_gpus_per_node=3,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            offload_rollout=False,
            global_batch_size=16,
            rollout_batch_size=4,
            n_samples_per_prompt=4,
            num_rollout=2,
            prompt_data=str(root / "train.jsonl"),
            input_key="input",
            label_key="label",
            metadata_key="metadata",
            rollout_temperature=1.0,
            rollout_seed=17,
            rollout_max_response_len=4096,
            rollout_max_context_len=6144,
            eval_prompt_data=["gsm8k", str(root / "eval.jsonl")],
            eval_interval=2,
            eval_temperature=0.0,
            n_samples_per_eval_prompt=1,
            eval_max_response_len=4096,
            custom_rm_path="open_instruct.miles.rewards.registered_reward",
            sglang_context_length=6144,
            sglang_max_total_tokens=32768,
            sglang_max_running_requests=4,
            sglang_server_concurrency=4,
            sglang_mem_fraction_static=0.6,
            sglang_disable_radix_cache=True,
            sglang_max_mamba_cache_size=8,
            sglang_disable_cuda_graph=True,
            sglang_sampling_backend="pytorch",
            sglang_log_level="warning",
            check_weight_update_equal=True,
            update_weight_buffer_size=1024 * 1024 * 1024,
            save=str(root / "metrics"),
            save_debug_rollout_data=str(root / "rollouts/{rollout_id}.pt"),
            lr=1e-6,
            lr_decay_iters=2,
            seed=17,
        ),
    )


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE_SOURCE.read_bytes().removesuffix(b"\n")
    assert hashlib.sha256(template).hexdigest() == TEMPLATE_SHA256
    assert (HF_SOURCE / "config.json").is_file() and list(HF_SOURCE.glob("*.safetensors"))
    hf = root / "hf"
    hf.mkdir()
    for path in HF_SOURCE.iterdir():
        if path.is_file() and path.name != "chat_template.jinja":
            (hf / path.name).symlink_to(path)
    (hf / "chat_template.jinja").write_bytes(template)
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    # Render once with the exact olmo-miles template; MILES receives completion strings.
    tokenizer.chat_template = template.decode()
    rows = list(load_dataset(DATASET, revision=DATASET_REVISION, split="train[:24]"))
    prepared = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        end = next((i for i, message in enumerate(messages) if message["role"] == "assistant"), len(messages))
        messages = messages[:end]
        assert messages and messages[-1]["role"] == "user"
        target = row["ground_truth"]
        if isinstance(target, list):
            assert len(target) == 1
            target = target[0]
        if isinstance(target, dict):
            target = target["answer"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert len(tokenizer.encode(prompt, add_special_tokens=False)) <= 2048
        prepared.append(
            dict(
                input=prompt,
                label=target,
                metadata={
                    "source_row": index,
                    "query": messages[-1]["content"],
                    "verifiers": [{"name": "gsm8k", "target": target}],
                },
            )
        )
    assert len({row["input"] for row in prepared}) == 24
    for name, subset in (("train", prepared[:8]), ("eval", prepared[8:])):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in subset))
    (root / "verifiers.json").write_text(
        json.dumps({"gsm8k": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"}})
    )
    dtypes = Counter()
    headers = {}
    kda_parameters = {}
    for shard in sorted(HF_SOURCE.glob("*.safetensors")):
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            assert 0 < size < shard.stat().st_size - 8
            raw = stream.read(size)
        headers[shard.name] = hashlib.sha256(raw).hexdigest()
        for name, tensor in json.loads(raw).items():
            if name == "__metadata__":
                continue
            dtypes[tensor["dtype"]] += tensor["data_offsets"][1] - tensor["data_offsets"][0]
            if name.endswith((".A_log", ".dt_bias")):
                kda_parameters[name] = {key: tensor[key] for key in ("dtype", "shape")}
    report = dict(
        source=str(HF_SOURCE),
        source_header_sha256=headers,
        tensor_bytes_by_dtype=dict(dtypes),
        kda_parameters=kda_parameters,
        template_source=str(TEMPLATE_SOURCE),
        template_sha256=TEMPLATE_SHA256,
        dataset=DATASET,
        revision=DATASET_REVISION,
        train_rows=list(range(8)),
        heldout_rows=list(range(8, 24)),
        interpretation="Held out from this trial's updates; no claim of SFT decontamination",
        config=json.loads((HF_SOURCE / "config.json").read_text()),
    )
    (root / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SFT_GSM8K_PREPARED", json.dumps(report), flush=True)


def run(root, validate_only=False, decode_graphs=False):
    config = configuration(root)
    if decode_graphs:
        config.miles.pop("sglang_disable_cuda_graph")
        config.miles["sglang_cuda_graph_backend_decode"] = "full"
        config.miles["sglang_cuda_graph_max_bs_decode"] = 4
        config.miles["sglang_cuda_graph_backend_prefill"] = "disabled"
    sys.argv = ["sft-gsm8k", *config.arguments()]
    args = arguments.parse_args()
    assert args.save_interval is None, "This bounded trial does not save optimizer checkpoints"
    if validate_only:
        print("SFT_GSM8K_CONFIG_VALIDATED")
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


def audit(root):
    verifier = GSM8KVerifier()
    expected = {
        name: {
            row["input"]: row["label"]
            for row in (json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines())
        }
        for name in ("train", "eval")
    }
    arguments_used = json.loads((root / "arguments.json").read_text())
    caps = {
        kind: int(arguments_used[arguments_used.index(flag) + 1])
        for kind, flag in (("train", "--rollout-max-response-len"), ("eval", "--eval-max-response-len"))
    }
    reports = {}
    for name, count, version in (("0", 16, 0), ("1", 16, 1), ("eval_0", 16, 0), ("eval_1", 16, 2)):
        samples = torch.load(root / f"rollouts/{name}.pt", weights_only=False)["samples"]
        assert len(samples) == count
        groups = defaultdict(list)
        scores = []
        for sample in samples:
            labels = expected["eval" if name.startswith("eval") else "train"]
            assert sample["prompt"] in labels and sample["label"] == labels[sample["prompt"]]
            assert set(sample["weight_versions"]) == {str(version)}
            assert len(sample["rollout_log_probs"]) == sample["response_length"]
            assert torch.isfinite(torch.tensor(sample["rollout_log_probs"])).all()
            score = verifier([], sample["response"], sample["label"]).score
            assert score == sample["reward"]
            scores.append(score)
            groups[sample["prompt"]].append(score)
        if name.startswith("eval"):
            assert set(groups) == set(expected["eval"])
        else:
            assert len(groups) == 4 and all(len(group) == 4 for group in groups.values())
        reports[name] = dict(
            samples=count,
            correct=sum(scores),
            accuracy=sum(scores) / count,
            mixed_reward_groups=sum(min(group) != max(group) for group in groups.values()),
            responses_at_token_cap=sum(
                s["response_length"] >= caps["eval" if name.startswith("eval") else "train"] for s in samples
            ),
            policy_version=version,
        )
    publication = [json.loads(line) for line in (root / "metrics/publication.jsonl").read_text().splitlines()]
    assert [entry["version"] for entry in publication] == [0, 1, 2]
    report = dict(
        passed=True,
        optimizer_steps=2,
        results=reports,
        publication=publication,
        interpretation="Small correctness trial; 16-question changes do not establish learning quality",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SFT_GSM8K_AUDIT_PASSED", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "audit", "validate"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--decode-graphs", action="store_true")
    options = parser.parse_args()
    if options.command == "prepare":
        prepare(options.output)
    elif options.command == "audit":
        audit(options.output)
    else:
        run(options.output, validate_only=options.command == "validate", decode_graphs=options.decode_graphs)


if __name__ == "__main__":
    main()
