"""Audit all local GPU coverage draws and optimizer state before marking source readiness."""

import argparse
import asyncio
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from miles.utils.types import Sample
from olmo_core.config import DType
from olmo_core.distributed import checkpoint as core_checkpoint
from olmo_core.nn import attention
from olmo_core.nn.hf import config as hf_config_utils
from olmo_core.nn.moe.v2 import olmo3
from transformers import AutoModelForCausalLM

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards


async def audit(options):
    root, catalog_root = options.run, options.catalog
    prompt_path = catalog_root / "gpu-prompts.jsonl"
    prompts = [json.loads(line) for line in prompt_path.open()]
    args = SimpleNamespace(
        olmo_core=CoreConfig(reward_config=str(catalog_root / "gpu-verifiers.json"), reward_zero_truncated=True),
        reward_key=None,
    )
    counts, statuses, rows, timestamps = Counter(), Counter(), [], []
    for index, prompt in enumerate(prompts):
        path = root / f"rollouts/{index}.pt"
        data = torch.load(path, weights_only=False, map_location="cpu")
        timestamps.append(path.stat().st_mtime)
        assert data["rollout_id"] == index and len(data["samples"]) == 4
        for raw in data["samples"]:
            assert raw["prompt"] == prompt["input"] and raw["label"] == prompt["label"]
            assert raw["metadata"]["sample_id"] == prompt["metadata"]["sample_id"]
            assert raw["metadata"]["source_name"] == prompt["metadata"]["source_name"]
            assert raw["response_length"] > 0
            assert len(raw["rollout_log_probs"]) == raw["response_length"]
            assert all(math.isfinite(v) for v in raw["rollout_log_probs"])
            status = Sample.Status(raw["status"])
            sample = Sample(
                prompt=raw["prompt"],
                response=raw["response"],
                label=raw["label"],
                metadata=raw["metadata"],
                status=status,
                tokens=raw["tokens"],
                response_length=raw["response_length"],
            )
            independent = await rewards.registered_reward(args, sample)
            assert independent == raw["reward"]
            statuses[status.value] += 1
            counts[prompt["metadata"]["source_name"]] += 1
            spans = [span for call in raw["weight_versions"] for span in call]
            assert spans and {span["version"] for span in spans} == {str(index)}
            rows.append(
                {
                    "source": prompt["metadata"]["source_name"],
                    "id": raw["metadata"]["sample_id"],
                    "rollout": index,
                    "reward": independent,
                    "status": status.value,
                    "tokens": raw["response_length"],
                    "response": raw["response"],
                    "response_sha256": hashlib.sha256(raw["response"].encode()).hexdigest(),
                }
            )
    last = len(prompts) - 1
    complete_path = root / f"save/core/rollout_{last:07d}/complete.json"
    complete = json.loads(complete_path.read_text())
    assert complete["clock"]["completed_steps"] == len(prompts)
    hf_config_utils._register_olmo3moe_auto_classes()
    reference = AutoModelForCausalLM.from_pretrained(root / "hf", torch_dtype=torch.bfloat16)
    native = olmo3.build_olmo3_moe_config_from_hf_config(
        reference.config, dtype=DType.bfloat16, attention_backend=attention.AttentionBackendName.torch
    ).build(init_device="cpu")
    olmo3.load_olmo3_moe_hf_state(native, reference.config, reference.state_dict())
    initial = native.state_dict()
    values = core_checkpoint.load_keys(
        root / f"save/core/rollout_{last:07d}/model", [f"module.{name}.main" for name in initial]
    )
    differences = {
        name: (value.reshape(tensor.shape).float() - tensor.float()).abs().max().item()
        for (name, tensor), value in zip(initial.items(), values, strict=True)
    }
    assert all(math.isfinite(v) for v in differences.values()) and max(differences.values()) > 0
    assert all(count == 4 for count in counts.values())
    summary = {
        "status": "passed",
        "groups": len(prompts),
        "completions": len(rows),
        "named_sources": len(counts),
        "statuses": dict(statuses),
        "reward_counts": dict(Counter(str(r["reward"]) for r in rows)),
        "optimizer_steps": complete["clock"]["completed_steps"],
        "max_parameter_change": max(differences.values()),
        "parameters_changed": sum(v > 0 for v in differences.values()),
        "weights_checked": len(differences),
        "clock": complete["clock"],
        "all_weight_versions_match": True,
        "all_rewards_recomputed": True,
        "first_to_last_rollout_seconds": max(timestamps) - min(timestamps),
        "scope": "One actual prompt/source, four draws, 32 tokens; zero-variance filtering disabled. Auxiliary losses drive updates. Not a learning-signal experiment.",
    }
    (root / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "audited-completions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    # Publish a separate receipt; the input catalog itself remains immutable.
    (root / "source-gpu-receipts.json").write_text(
        json.dumps({name: {"completions": count, "status": "passed"} for name, count in counts.items()}, indent=2)
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    asyncio.run(audit(parser.parse_args()))
