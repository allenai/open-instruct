"""Freeze the historical light-SFT inputs and run its 200-update Core comparison."""

import argparse
import asyncio
import dataclasses
import importlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

from scripts.miles.prepare_gsm8k_parity import derive_rows, digest, json_bytes, write_immutable

HISTORY_PATH = Path(__file__).resolve().parents[2] / "configs/miles/reference/light-sft1000-gsm8k-historical.json"
ROOT = Path("/weka/oe-training-default/robertb/open-instruct/light-sft1000-gsm8k/20260911-v1")
COUNTS = {"train": 7473, "eval": 128}
CAMPAIGN = "gsm8k-light-sft1000-core-20260911-v1"


def history():
    return json.loads(HISTORY_PATH.read_text())


def checked_bytes(path, expected):
    raw = Path(path).read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"Historical artifact SHA256 differs: {path}")
    return raw


def validate_geometry(config):
    expected = history()["manifest"]["model"]["config_contract"]["architecture"]
    required = (
        "num_hidden_layers",
        "hidden_size",
        "latent_moe_dim",
        "n_routed_experts",
        "num_experts_per_tok",
        "layer_types",
        "vocab_size",
        "dense_layers_indices",
        "linear_num_key_heads",
        "linear_value_head_dim",
    )
    differences = {key: (expected[key], config.get(key)) for key in required if config.get(key) != expected[key]}
    if differences:
        raise ValueError(f"Not the historical 20-layer latent-KDA SFT1000 checkpoint: {differences}")


def checkpoint_descriptor(root, source):
    model = history()["manifest"]["model"]
    config_raw = checked_bytes(source / "config.json", model["hf_config_sha256"])
    validate_geometry(json.loads(config_raw))
    template = (source / "chat_template.jinja").read_bytes().removesuffix(b"\n")
    if digest(template) != model["chat_template_sha256"]:
        raise ValueError("Historical instruction template differs")
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise ValueError("Historical HF checkpoint has no safetensors shards")
    headers = {}
    router_tensors = {}
    dtype_counts = {}
    for shard in shards:
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size < min(100_000_000, shard.stat().st_size - 8):
                raise ValueError(f"Invalid safetensors header: {shard}")
            raw = stream.read(size)
            tensors = json.loads(raw)
            for name, tensor in tensors.items():
                if name != "__metadata__":
                    dtype_counts[tensor["dtype"]] = dtype_counts.get(tensor["dtype"], 0) + 1
                    if ".router." in name:
                        router_tensors[name] = {
                            "dtype": tensor["dtype"],
                            "shape": tensor["shape"],
                            "shard": shard.name,
                        }
                if name != "__metadata__" and tensor["data_offsets"][1] > shard.stat().st_size - 8 - size:
                    raise ValueError(f"Truncated tensor payload: {shard}:{name}")
        headers[shard.name] = {"header_sha256": digest(raw), "bytes": shard.stat().st_size}
    hf = root / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name != "chat_template.jinja":
            target = hf / item.name
            if target.is_symlink():
                if target.resolve() != item.resolve():
                    raise ValueError(f"Conflicting checkpoint descriptor: {target}")
            elif target.exists():
                raise ValueError(f"Descriptor must reference the original checkpoint: {target}")
            else:
                target.symlink_to(item)
    write_immutable(hf / "chat_template.jinja", template)
    torch = importlib.import_module("torch")
    safetensors = importlib.import_module("safetensors")
    for name, info in router_tensors.items():
        with safetensors.safe_open(source / info["shard"], framework="pt", device="cpu") as reader:
            original = reader.get_tensor(name)
        original_fp32 = original.float()
        delta = original_fp32 - original.to(torch.bfloat16).float()
        info["payload_sha256"] = digest(original.contiguous().view(torch.uint8).numpy().tobytes())
        info["bf16_rounding"] = {
            "changed_elements": int(delta.count_nonzero()),
            "elements": original.numel(),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / original_fp32.norm().clamp_min(1e-30)),
        }
    if len(router_tensors) != 19:
        raise ValueError(f"Expected 19 sparse-block router tensors, got {len(router_tensors)}")
    return {
        "source": str(source),
        "shards": headers,
        "weight_payload_hashes_verified": False,
        "source_tensor_dtype_counts": dtype_counts,
        "source_router_tensors": router_tensors,
        "source_config_dtype": json.loads(config_raw).get("torch_dtype", json.loads(config_raw).get("dtype")),
        "import_contract": {
            "core_hf_load_dtype": "bfloat16",
            "core_factory_dtype": "bfloat16",
            "sglang_router_storage": "ReplicatedLinear inherits engine model dtype; no FP32 override",
            "router_projection_compute": "FP32 F.linear(input.float(), weight.float()) in both current implementations",
            "non_bf16_source_router_import": "Source FP32 values, if present, are rounded by BF16 model import; not identical original-precision policy",
            "runtime_storage_validation": "Initial all-tensor publication equality gate remains enabled",
        },
    }


def normalize_answer(value):
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", value)
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", cleaned)
    return numbers[-1] if numbers else cleaned


def offline_rows(requests, tokenizer):
    rows, evidence = [], []
    for record in requests:
        request, doc = record["request"], record["doc"]
        prompt = f"Question: {doc['query']}\nAnswer:"
        expected = {
            "max_gen_toks": 512,
            "do_sample": False,
            "temperature": 0,
            "logprobs": 1,
            "num_samples": 1,
            "add_special_tokens": False,
        }
        if (
            request["context"] != prompt
            or request["generation_kwargs"] != expected
            or request["stop_sequences"] != ["Question:", "\n\n"]
            or request["provider_request"]["endpoint"] != "/completions"
            or record["label"] != normalize_answer(doc["short_answer"])
        ):
            raise ValueError("Historical full-test request semantics differ")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + 512 > 1024:
            raise ValueError("Historical full-test prompt cannot fit its original context budget")
        name = f"historical-full-test-{record['native_id']}"
        rows.append(
            {
                "id": name,
                "input": prompt,
                "label": record["label"],
                "metadata": {
                    "prepared_sample_id": name,
                    "query": doc["query"],
                    "native_id": record["native_id"],
                    "verifiers": [{"name": "gsm8k", "target": record["label"], "weight": 1.0}],
                },
            }
        )
        evidence.append(
            {
                "prepared_sample_id": name,
                "prompt_sha256": digest(prompt.encode()),
                "token_ids_sha256": digest(json_bytes(ids)),
                "prompt_tokens": len(ids),
            }
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate full-test question IDs")
    return rows, evidence


def prepare(root, historical_result):
    """Read original WEKA artifacts; only write this comparison's new directory."""
    root = Path(root)
    if (root / "preparation.json").exists():
        return verify(root)
    old = history()
    source = Path(old["manifest"]["model"]["hf_checkpoint"])
    descriptor = checkpoint_descriptor(root, source)
    write_immutable(root / "checkpoint-inventory.json", json_bytes(descriptor))
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        root / "hf", trust_remote_code=True
    )
    if digest(tokenizer.chat_template.encode()) != old["manifest"]["model"]["chat_template_sha256"]:
        raise ValueError("Tokenizer selected a different instruction template")
    native_manifest_path = Path(old["manifest"]["data"]["rl_manifest"]["path"])
    native_raw = checked_bytes(native_manifest_path, old["manifest"]["data"]["rl_manifest"]["sha256"])
    native_manifest = json.loads(native_raw)
    write_immutable(root / "baseline/rl-manifest.json", native_raw)
    partitions = {}
    for partition, count in COUNTS.items():
        artifact = native_manifest["artifacts"][partition]
        raw = checked_bytes(native_manifest_path.parent / artifact["path"], artifact["sha256"])
        if partition == "train" and digest(raw) != old["manifest"]["data"]["sha256"]:
            raise ValueError("Historical training selection differs")
        native = [json.loads(line) for line in raw.splitlines()]
        if len(native) != count or artifact["records"] != count:
            raise ValueError(f"Historical {partition} count differs")
        rows, proofs = derive_rows(native, tokenizer, max_prompt_tokens=1023)
        write_immutable(root / f"baseline/{partition}.jsonl", raw)
        write_immutable(root / f"{partition}.jsonl", b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        partitions[partition] = {"records": count, "rows": proofs}
    train_prompts = {row["prompt_sha256"] for row in partitions["train"]["rows"]}
    if train_prompts.intersection(row["prompt_sha256"] for row in partitions["eval"]["rows"]):
        raise ValueError("Historical train/test prompt overlap")
    full = old["full_test"]
    raw = checked_bytes(Path(historical_result) / full["requests_path"], full["requests_sha256"])
    requests = [json.loads(line) for line in raw.splitlines()]
    if len(requests) != 1319:
        raise ValueError("Historical full-test requests are incomplete")
    rows, proofs = offline_rows(requests, tokenizer)
    write_immutable(root / "offline/requests.jsonl", raw)
    write_immutable(root / "offline/prompts.jsonl", b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    write_immutable(root / "offline/token-proofs.json", json_bytes(proofs))
    write_immutable(root / "historical.json", HISTORY_PATH.read_bytes())
    write_immutable(
        root / "verifiers.json", json_bytes({"gsm8k": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"}})
    )
    names = [
        "train.jsonl",
        "eval.jsonl",
        "baseline/train.jsonl",
        "baseline/eval.jsonl",
        "baseline/rl-manifest.json",
        "hf/config.json",
        "hf/chat_template.jinja",
        "hf/tokenizer.json",
        "hf/tokenizer_config.json",
        "offline/requests.jsonl",
        "offline/prompts.jsonl",
        "offline/token-proofs.json",
        "historical.json",
        "verifiers.json",
        "checkpoint-inventory.json",
    ]
    report = {
        "schema_version": 1,
        "historical_experiment": old["historical_training_experiment"],
        "selection": "Exact historical prepared rows and order; no new selection or shuffle",
        "seed": 1,
        "descriptor": descriptor,
        "partitions": partitions,
        "files": {name: digest((root / name).read_bytes()) for name in names},
    }
    write_immutable(root / "preparation.json", json_bytes(report))
    return verify(root)


def verify(root):
    report = json.loads((root / "preparation.json").read_text())
    old = history()
    if report["historical_experiment"] != old["historical_training_experiment"]:
        raise ValueError("Preparation references another historical run")
    required = {
        "train.jsonl",
        "eval.jsonl",
        "baseline/train.jsonl",
        "baseline/eval.jsonl",
        "baseline/rl-manifest.json",
        "hf/config.json",
        "hf/chat_template.jinja",
        "hf/tokenizer.json",
        "hf/tokenizer_config.json",
        "offline/requests.jsonl",
        "offline/prompts.jsonl",
        "offline/token-proofs.json",
        "historical.json",
        "verifiers.json",
        "checkpoint-inventory.json",
    }
    if set(report["files"]) != required:
        raise ValueError("Incomplete preparation hash inventory")
    for name, expected in report["files"].items():
        checked_bytes(root / name, expected)
    checked_bytes(root / "historical.json", digest(HISTORY_PATH.read_bytes()))
    checked_bytes(root / "baseline/rl-manifest.json", old["manifest"]["data"]["rl_manifest"]["sha256"])
    checked_bytes(root / "hf/config.json", old["manifest"]["model"]["hf_config_sha256"])
    checked_bytes(root / "offline/requests.jsonl", old["full_test"]["requests_sha256"])
    for partition, count in COUNTS.items():
        rows = [json.loads(line) for line in (root / f"{partition}.jsonl").read_text().splitlines()]
        proofs = report["partitions"][partition]["rows"]
        if len(rows) != count or len(proofs) != count:
            raise ValueError("Prepared record count differs")
        for row, proof in zip(rows, proofs, strict=True):
            if (
                row["metadata"]["prepared_sample_id"] != proof["prepared_sample_id"]
                or digest(row["input"].encode()) != proof["prompt_sha256"]
            ):
                raise ValueError("Prepared prompt identity differs from token proof")
    validate_geometry(json.loads((root / "hf/config.json").read_text()))
    if checkpoint_descriptor(root, Path(report["descriptor"]["source"])) != report["descriptor"]:
        raise ValueError("Checkpoint descriptor or safetensors headers changed")
    return report


def configuration(root):
    base = importlib.import_module("scripts.miles.gsm8k_parity").configuration(
        root, updates=200, eval_interval=10, campaign=CAMPAIGN, save_interval=50
    )
    config = dataclasses.replace(base, core=dataclasses.replace(base.core, max_sequence_length=1024))
    config.miles.pop("sglang_disable_radix_cache", None)
    config.miles.update(
        num_gpus_per_node=4,
        rollout_num_gpus=2,
        global_batch_size=32,
        rollout_batch_size=8,
        rollout_seed=1,
        seed=1,
        rollout_max_response_len=512,
        rollout_max_prompt_len=1023,
        rollout_max_context_len=1024,
        eval_temperature=1.0,
        eval_max_response_len=512,
        sglang_context_length=8192,
        sglang_max_total_tokens=771285,
        sglang_max_running_requests=8,
        sglang_mem_fraction_static=0.18,
        sglang_max_mamba_cache_size=52,
        sglang_mamba_radix_cache_strategy="extra_buffer",
        sglang_sampling_backend="flashinfer",
        sglang_cuda_graph_max_bs_decode=8,
        sglang_chunked_prefill_size=16384,
        sglang_max_prefill_tokens=16384,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        router_policy="cache_aware",
        router_cache_threshold=0.8,
        router_balance_abs_threshold=4,
        router_balance_rel_threshold=1.5,
        eval_function_path="scripts.miles.light_sft_eval.HistoricalEvaluation",
    )
    return config


def effective_settings(args):
    expected = {
        "num_rollout": 200,
        "eval_interval": 10,
        "global_batch_size": 32,
        "rollout_batch_size": 8,
        "n_samples_per_prompt": 4,
        "seed": 1,
        "rollout_seed": 1,
        "rollout_shuffle": False,
        "rollout_max_context_len": 1024,
        "rollout_max_response_len": 512,
        "eval_temperature": 1.0,
        "eval_max_response_len": 512,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
        "grpo_std_normalization": False,
        "normalize_advantages": False,
        "calculate_per_token_loss": False,
        "use_rollout_logprobs": False,
        "use_rollout_routing_replay": False,
        "use_routing_replay": False,
        "fully_async": False,
        "sglang_sampling_backend": "flashinfer",
        "router_policy": "cache_aware",
        "router_cache_threshold": 0.8,
        "router_balance_abs_threshold": 4,
        "router_balance_rel_threshold": 1.5,
        "sglang_disable_radix_cache": False,
        "sglang_max_running_requests": 8,
        "sglang_server_concurrency": 4,
        "sglang_max_mamba_cache_size": 52,
        "sglang_mamba_radix_cache_strategy": "extra_buffer",
        "sglang_cuda_graph_backend_decode": "full",
        "sglang_cuda_graph_max_bs_decode": 8,
        "sglang_cuda_graph_backend_prefill": "disabled",
        "sglang_chunked_prefill_size": 16384,
        "sglang_max_prefill_tokens": 16384,
        "n_samples_per_eval_prompt": 1,
        "skip_eval_before_train": False,
        "sglang_context_length": 8192,
        "sglang_max_total_tokens": 771285,
        "sglang_mem_fraction_static": 0.18,
        "num_gpus_per_node": 4,
        "actor_num_gpus_per_node": 2,
        "rollout_num_gpus": 2,
        "save_interval": 50,
        "lr": 1e-6,
        "lr_decay_style": "constant",
        "lr_warmup_iters": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "clip_grad": 1.0,
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
    }
    actual = {key: getattr(args, key) for key in expected}
    differences = {key: (value, actual[key]) for key, value in expected.items() if actual[key] != value}
    if differences:
        raise ValueError(f"Light-SFT protocol differs: {differences}")
    return actual


def run(root, validate_only=False, local_hf=None):
    preparation = verify(root)
    config = configuration(root)
    if local_hf is not None:
        staging = json.loads((root / "local-staging.json").read_text())
        if staging["destination"] != str(local_hf) or not staging["verified_full_payload"]:
            raise ValueError("Missing verified local checkpoint staging")
        config.miles["hf_checkpoint"] = str(local_hf)
        config.miles["sglang_log_level"] = "info"
    sys.argv = ["light-sft1000-core", *config.arguments()]
    args = importlib.import_module("miles.utils.arguments").parse_args()
    effective = effective_settings(args)
    if validate_only:
        print("LIGHT_SFT_CONFIG_VALIDATED", json.dumps(effective), flush=True)
        return
    output = root / "core"
    output.mkdir()
    for name, value in (("arguments", config.arguments()), ("effective", effective), ("preparation", preparation)):
        (output / f"{name}.json").write_bytes(json_bytes(value))
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    ray = importlib.import_module("ray")
    driver = importlib.import_module("open_instruct.miles.driver")
    started = time.monotonic()
    ray.init(num_gpus=4, num_cpus=16, include_dashboard=False, object_store_memory=1024**3)
    try:
        asyncio.run(driver.train(args))
    finally:
        ray.shutdown()
    completed = importlib.import_module("scripts.miles.gsm8k_parity").check_completion(
        root, updates=200, eval_interval=10
    )
    for step in (0, 200):
        report = json.loads((output / f"offline-{step}.json").read_text())
        if report["count"] != 1319 or report["step"] != step:
            raise ValueError("Missing full-test before/after evaluation")
    completed.update(completed=True, elapsed_seconds=time.monotonic() - started)
    (output / "completion.json").write_bytes(json_bytes(completed))
    print("LIGHT_SFT_COMPLETED", json.dumps(completed), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validate", "run", "audit"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--local-hf", type=Path)
    parser.add_argument("--historical-result", type=Path, default=Path("/historical-offline"))
    args = parser.parse_args()
    if args.stage == "prepare":
        report = prepare(args.root, args.historical_result)
        print(
            "LIGHT_SFT_PREPARED",
            json.dumps(
                {
                    "root": str(args.root),
                    "records": {key: value["records"] for key, value in report["partitions"].items()},
                    "checkpoint": report["descriptor"],
                    "preparation_sha256": digest((args.root / "preparation.json").read_bytes()),
                }
            ),
            flush=True,
        )
    elif args.stage == "audit":
        audit = importlib.import_module("scripts.miles.analyze_gsm8k_parity").audit
        report = audit(
            args.root,
            "core",
            updates=200,
            eval_steps=tuple(range(0, 201, 10)),
            prompts_per_update=8,
            samples_per_prompt=4,
            response_cap=512,
        )
        (args.root / "core/audit.json").write_bytes(json_bytes(report))
        if not report["valid"]:
            raise ValueError("Light-SFT rollout audit failed")
    else:
        run(args.root, validate_only=args.stage == "validate", local_hf=args.local_hf)


if __name__ == "__main__":
    main()
