"""Frozen-data adapter for a historical Open Instruct GSM8K comparison.

Run inside the original Olmo 3 image. The original trainer and verifiers remain
in that image; Adam beta2 is aligned and the initial evaluation is explicitly scheduled.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open

from open_instruct import dataset_transformation

ORIGINAL_TRAINER_SHA256 = "6367478b4c957cfe595745dd1549ad1884585fc7bc4b21a589cbe8ed46cee3b1"
PASSTHROUGH_TEMPLATE = "{{ messages[0]['content'] }}"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def convert_row(row, tokenizer):
    specs = row["metadata"]["verifiers"]
    if len(specs) != 1 or specs[0]["name"] != "gsm8k" or specs[0].get("weight", 1) != 1:
        raise ValueError("Original GSM8K control requires exactly one unit-weight GSM8K verifier")
    result = {
        "messages": [{"role": "user", "content": row["input"]}],
        "ground_truth": str(specs[0]["target"]),
        "dataset": "gsm8k",
        "prepared_sample_id": row["metadata"]["prepared_sample_id"],
    }
    transformed = dataset_transformation.rlvr_tokenize_v2(dict(result), tokenizer)
    ids = transformed[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    if sha(encoded(ids)) != row["metadata"]["run_prompt_token_ids_sha256"]:
        raise ValueError(f"Original tokenizer changed frozen prompt tokens: {result['prepared_sample_id']}")
    if not dataset_transformation.rlvr_filter_v1(
        transformed, tokenizer, max_prompt_token_length=2048, max_token_length=34816
    ):
        raise ValueError("Original filter would remove a frozen row")
    return result


def read_jsonl(path):
    # str.splitlines() also splits valid Unicode separators inside JSON strings.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def legacy_model(model, output):
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "olmo3" or config.get("architectures") != ["Olmo3ForCausalLM"]:
        raise ValueError("Historical alias requires a public Olmo3ForCausalLM checkpoint")
    shapes = {}
    for shard in model.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 -- safe_open is not an iterable mapping
                if name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
                    if name in shapes:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    shapes[name] = handle.get_slice(name).get_shape()
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    expected = {
        f"model.layers.{layer}.self_attn.{kind}_norm.weight": [heads * head_dim]
        for layer in range(config["num_hidden_layers"])
        for kind, heads in (("q", config["num_attention_heads"]), ("k", config["num_key_value_heads"]))
    }
    if shapes != expected:
        raise ValueError("Checkpoint Q/K norms do not match the historical global-normalization implementation")
    output.mkdir()
    links = {}
    for path in model.iterdir():
        if path.is_file() and path.name != "config.json":
            (output / path.name).symlink_to(path.resolve())
            links[path.name] = {"target": str(path.resolve()), "size": path.stat().st_size}
    config.update(model_type="olmo2-retrofit", architectures=["Olmo2RetrofitForCausalLM"])
    (output / "config.json").write_bytes(encoded(config))
    return {"source_config_sha256": sha((model / "config.json").read_bytes()), "norm_shapes": shapes, "links": links}


def restore_public_exports(output, source):
    restored = []
    for path in (output / "model").rglob("config.json"):
        config = json.loads(path.read_text())
        if config.get("model_type") != "olmo2-retrofit":
            continue
        if not list(path.parent.glob("*.safetensors")) and not list(path.parent.glob("pytorch_model*.bin")):
            continue
        config.update(model_type="olmo3", architectures=["Olmo3ForCausalLM"])
        path.write_bytes(encoded(config))
        for name in (
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ):
            if (source / name).is_file():
                shutil.copyfile(source / name, path.parent / name)
        restored.append(str(path.parent))
    if not restored:
        raise ValueError("Original trainer exited without a public HF model export")
    return restored


def prepare(model, source, output):
    if output.exists():
        raise ValueError("Prepared comparison directory already exists; verify or use a fresh path")
    manifest = json.loads((source / "manifest.json").read_text())
    for filename, expected in manifest["outputs"].items():
        if sha((source / filename).read_bytes()) != expected:
            raise ValueError(f"Core prepared artifact changed: {filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".original-gsm8k-", dir=output.parent))
    try:
        alias = legacy_model(model, staging / "legacy-model")
        token_root = staging / "tokenizer"
        token_root.mkdir()
        tokenizer = dataset_transformation.TokenizerConfig(tokenizer_name_or_path=str(model)).tokenizer
        original_template = tokenizer.chat_template
        tokenizer.chat_template = PASSTHROUGH_TEMPLATE
        tokenizer.save_pretrained(token_root)
        shutil.copyfile(model / "config.json", token_root / "config.json")
        # Check the exact loader that training uses, including its pad-token rules.
        tokenizer = dataset_transformation.TokenizerConfig(tokenizer_name_or_path=str(token_root)).tokenizer
        receipt = {
            "model": str(model),
            "source": str(source),
            "model_alias": alias,
            "splits": {},
            "original_chat_template_sha256": sha(original_template.encode()),
            "adapter_chat_template": PASSTHROUGH_TEMPLATE,
            "source_manifest_sha256": sha((source / "manifest.json").read_bytes()),
        }
        identities = []
        for split, expected_count in (("train", 6000), ("eval", 512)):
            rows = read_jsonl(source / f"{split}.jsonl")
            if len(rows) != expected_count:
                raise ValueError(f"Expected {expected_count} frozen {split} rows")
            converted = [convert_row(row, tokenizer) for row in rows]
            ids = {row["prepared_sample_id"] for row in converted}
            if len(ids) != len(converted):
                raise ValueError("Duplicate source identities")
            identities.append(ids)
            raw = b"".join(encoded(row) for row in converted)
            (staging / f"{split}.jsonl").write_bytes(raw)
            (staging / f"smoke-{split}.jsonl").write_bytes(
                b"".join(encoded(row) for row in converted[: 64 if split == "train" else 8])
            )
            receipt["splits"][split] = {"rows": len(converted), "sha256": sha(raw), "all_prompt_tokens_equal": True}
        if identities[0] & identities[1]:
            raise ValueError("Training and heldout identities overlap")
        receipt["files"] = {
            str(p.relative_to(staging)): sha(p.read_bytes())
            for p in staging.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        (staging / "preparation.json").write_bytes(encoded(receipt))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("ORIGINAL_BASELINE_PREPARATION_PASSED", json.dumps(receipt), flush=True)


def patch_trainer(text):
    changes = {
        "adam_alignment": {
            "before": "torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)",
            "after": "torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer, betas=(0.9, 0.95), eps=1e-8)",
        },
        "initial_evaluation": {
            "before": "            training_step % args.local_eval_every == 0\n",
            "after": "            (training_step % args.local_eval_every == 0 or (training_step == 1 and args.eval_on_step_0))\n",
        },
    }
    for name, change in changes.items():
        if text.count(change["before"]) != 1:
            raise ValueError(f"Cannot unambiguously apply original benchmark adjustment: {name}")
        text = text.replace(change["before"], change["after"])
    return text, changes


def train(model, prepared, output, *, smoke):
    receipt = json.loads((prepared / "preparation.json").read_text())
    if receipt["model"] != str(model):
        raise ValueError("Prepared model identity differs")
    for name, digest in receipt["files"].items():
        if sha((prepared / name).read_bytes()) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    for name, link in receipt["model_alias"]["links"].items():
        path = prepared / "legacy-model" / name
        if not path.is_symlink() or str(path.resolve()) != link["target"] or path.stat().st_size != link["size"]:
            raise ValueError(f"Original model alias changed: {name}")
    if output.exists():
        raise ValueError("Use a fresh original-framework run directory")
    output.mkdir(parents=True)
    trainer = Path("/stage/open_instruct/grpo_fast.py")
    original = trainer.read_bytes()
    if sha(original) != ORIGINAL_TRAINER_SHA256:
        raise ValueError("Original trainer source differs from the audited image")
    patched, changes = patch_trainer(original.decode())
    trainer.write_text(patched)
    steps = 3 if smoke else 200
    prefix = "smoke-" if smoke else ""
    options = {
        "exp_name": output.name,
        "model_name_or_path": str(prepared / "legacy-model"),
        "tokenizer_name_or_path": str(prepared / "tokenizer"),
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "bfloat16",
        "dataset_mixer_list": [str(prepared / f"{prefix}train.jsonl"), "1.0"],
        "dataset_mixer_eval_list": [str(prepared / f"{prefix}eval.jsonl"), "1.0"],
        "dataset_mixer_list_splits": "train",
        "dataset_mixer_eval_list_splits": "train",
        "max_token_length": 34816,
        "max_prompt_token_length": 2048,
        "response_length": 32768,
        "pack_length": 34816,
        "num_learners_per_node": 2,
        "vllm_num_engines": 4,
        "vllm_tensor_parallel_size": 1,
        "vllm_enforce_eager": True,
        "vllm_gpu_memory_utilization": 0.7,
        "vllm_enable_prefix_caching": False,
        "deepspeed_stage": 3,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "num_unique_prompts_rollout": 16,
        "num_samples_per_prompt_rollout": 4,
        "total_episodes": steps * 64,
        "num_mini_batches": 1,
        "num_epochs": 1,
        "learning_rate": 1e-6,
        "lr_scheduler_type": "constant",
        "warm_up_steps": 0,
        "weight_decay": 0.0,
        "beta": 0.0,
        "clip_lower": 0.2,
        "clip_higher": 0.28,
        "advantage_normalization_type": "centered",
        "verification_reward": 1.0,
        "temperature": 1.0,
        "seed": 17,
        "async_steps": 0,
        "inflight_updates": False,
        "eval_on_step_0": True,
        "local_eval_every": 1 if smoke else 50,
        "save_freq": steps,
        "checkpoint_state_freq": steps if smoke else 100,
        "checkpoint_state_dir": str(output / "checkpoints"),
        "output_dir": str(output / "model"),
        "push_to_hub": False,
        "try_auto_save_to_beaker": False,
        "try_launch_beaker_eval_jobs_on_weka": False,
        "save_traces": True,
        "with_tracking": True,
        "wandb_entity": "ai2-llm",
        "wandb_project_name": "olmo-rl-comparison",
        "backend_timeout": 120,
    }
    command = [sys.executable, str(trainer)]
    for key, value in options.items():
        command.append("--" + key)
        command.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    record = {
        "command": command,
        "model_alias": receipt["model_alias"],
        "original_source_sha256": sha(original),
        "patched_source_sha256": sha(trainer.read_bytes()),
        "source_adjustments": changes,
        "eval_policy_updates": sorted({0, *range((1 if smoke else 50) - 1, steps, 1 if smoke else 50)}),
        "remaining_differences": [
            "vLLM versus SGLang",
            "DeepSpeed/HF versus OLMo-core",
            "Original token-mean packed loss versus Core response reduction",
            "Original historical GSM8K verifier versus current verifier",
        ],
    }
    (output / "invocation.json").write_bytes(encoded(record))
    print("ORIGINAL_BASELINE_COMMAND", json.dumps(record), flush=True)
    subprocess.run(
        command, check=True, env={**os.environ, "WANDB_RUN_GROUP": "olmo3-sft-learning-confidence-20260914"}
    )

    exports = restore_public_exports(output, model)
    completion = {"completed_updates": steps, "public_exports": exports, "invocation_sha256": sha(encoded(record))}
    (output / "completion.json").write_bytes(encoded(completion))
    print("ORIGINAL_BASELINE_COMPLETED", json.dumps(completion), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "train"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.model, args.source, args.prepared)
    else:
        train(args.model, args.prepared, args.output, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
