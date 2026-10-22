"""Fixture preparation and independent audits for the optional integration group."""

import argparse
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from miles.backends.core_utils import fla_compat
from olmo_core.distributed import checkpoint
from olmo_core.nn import attention
from olmo_core.nn.hf import config as hf_config_utils
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from safetensors import torch as safetensors_torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# This is a script entrypoint; share the existing application-owned lifecycle fixture.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_moe  # noqa: E402

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_record(module_name):
    module = importlib.import_module(module_name)
    path = Path(module.__file__).resolve()
    directory = path.parent
    files = sorted(directory.rglob("*.py"))
    content = hashlib.sha256()
    for file in files:
        content.update(str(file.relative_to(directory)).encode() + b"\0" + file.read_bytes())
    revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {
        "imported_path": str(path),
        "git_head": revision.stdout.strip() or None,
        "python_tree_sha256": content.hexdigest(),
    }


def prepare(root, miles_source, image_id):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Local integration requires exactly one visible CUDA GPU; this is a failure, not a skip")
    candidate = source_record("miles")
    if Path(candidate["imported_path"]).parent != miles_source / "miles":
        raise RuntimeError(f"Wrong Miles import: {candidate['imported_path']}; expected {miles_source / 'miles'}")
    if "sha256:" not in image_id:
        raise ValueError("Supply an immutable sha256 image ID/digest")
    spec = json.loads((HERE / "fixture.json").read_text())
    provenance = {
        "image_id_from_launcher": image_id,
        "sources": {
            name: source_record(name) for name in ("miles", "olmo_core", "olmo_sglang", "sglang", "open_instruct")
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "fixture_sha256": digest(HERE / "fixture.json"),
        "harness_sha256": {
            p.name: digest(p) for p in (HERE / "runner.py", HERE / "runtime.py", HERE.parent / "local_moe.py")
        },
    }
    write(root / "provenance.json", provenance)
    hf_config_utils._register_olmo3moe_auto_classes()
    tokenizer = Tokenizer(models.BPE(unk_token="<eos>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [row["input"] for row in spec["prompts"]],
        trainers.BpeTrainer(
            vocab_size=512, special_tokens=["<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<eos>", eos_token="<eos>", pad_token="<eos>")
    fast.add_tokens([f"<reserved-{i}>" for i in range(512 - len(fast))])
    assert len(fast) == spec["model"]["vocab_size"]
    torch.manual_seed(spec["seed"])
    model = AutoModelForCausalLM.from_config(Olmo3MoeConfig(**spec["model"])).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.zero_()
    fixture = root / "fixture"
    fixture.mkdir()
    model.save_pretrained(fixture / "hf")
    fast.save_pretrained(fixture / "hf")
    rows = [dict(row, metadata={"verifiers": [{"name": "gsm8k", "target": row["label"]}]}) for row in spec["prompts"]]
    (fixture / "prompts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(fixture / "verifiers.json", {"gsm8k": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"}})
    write(
        root / "fixture-hashes.json",
        {str(p.relative_to(fixture)): digest(p) for p in sorted(fixture.rglob("*")) if p.is_file()},
    )
    for name in ("control", "resumed"):
        shutil.copytree(fixture, root / name)


def audit(root):
    local_moe.audit(SimpleNamespace(output=root, updates=3, routing_replay=True))
    records = [json.loads(line) for line in (root / "save/training_contract_rank0.jsonl").read_text().splitlines()]
    steps = [r for r in records if r["event"] == "optimizer"]
    assert [r["step"] for r in steps] == [1, 2, 3]
    for step in steps:
        assert not step["optimizer_skipped"]
        for group in step["local_pre_optimizer_gradients"].values():
            assert math.isfinite(group["local_l2"]) and group["local_l2"] > 0
        for group in step["sampled_model_updates"].values():
            assert math.isfinite(group["sampled_update_l2"]) and group["sampled_update_l2"] > 0
    scores = [r for r in records if r["event"] == "scores"]
    assert {r["rollout_id"] for r in scores} == {0, 1, 2}
    assert all(math.isfinite(r["max_abs"]) and r["max_abs"] <= 0.05 for r in scores)
    for step in range(3):
        complete = json.loads((root / f"save/core/rollout_{step:07d}/complete.json").read_text())
        assert complete["clock"]["completed_steps"] == step + 1
        assert complete["clock"]["next_rollout_id"] == step + 1
        assert complete["cursor_sha256"] == digest(root / f"save/rollout/global_dataset_state_dict_{step}.pt")
    # Reload the final exported model, and compare every tensor with the saved native model.
    exported = AutoModelForCausalLM.from_pretrained(root / "export", torch_dtype=torch.bfloat16)
    initial = AutoModelForCausalLM.from_pretrained(root / "hf", torch_dtype=torch.bfloat16)
    native = (
        olmo3.build_olmo3_moe_config_from_hf_config(
            initial.config, dtype=local_moe.DType.bfloat16, attention_backend=attention.AttentionBackendName.torch
        )
        .build(init_device="cpu")
        .to(torch.bfloat16)
    )
    # OLMoDDPModel.apply_ddp materializes every parameter in BF16, including KDA
    # scalars whose standalone constructor defaults to FP32. Mirror that cast
    # before restoring optimizer master weights; this is not a tolerance change.
    keys = [f"module.{name}.main" for name in native.state_dict()]
    values = checkpoint.load_keys(root / "save/core/rollout_0000002/model", keys)
    native.load_state_dict(
        {
            name: value.reshape(target.shape).to(target.dtype)
            for (name, target), value in zip(native.state_dict().items(), values, strict=True)
        }
    )
    expected = olmo3.gather_olmo3_moe_hf_state(native, initial.config, cpu=True)
    saved = safetensors_torch.load_file(root / "export/model.safetensors")
    assert saved.keys() == expected.keys() == exported.state_dict().keys()
    for name, actual in saved.items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0, msg=name)
    for name, actual in exported.state_dict().items():
        # HF reload explicitly requests BF16; native KDA scalars may be saved in FP32.
        torch.testing.assert_close(actual, saved[name].to(actual.dtype), rtol=0, atol=0, msg=name)
    fla_compat.install_kda_triton_compat()
    exported.cuda().eval()
    tokens = torch.tensor([[10, 20, 30]], device="cuda")
    with torch.no_grad():
        assert torch.isfinite(exported(tokens, use_cache=False).logits).all()
    write(
        root / "integration-audit.json",
        {
            "passed": True,
            "updates": 3,
            "samples": 12,
            "export_exact": True,
            "max_logprob_error": max(r["max_abs"] for r in scores),
        },
    )


def compare_value(left, right, *, rtol, atol):
    if isinstance(left, torch.Tensor):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol, equal_nan=False)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_value(left[key], right[key], rtol=rtol, atol=atol)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            compare_value(a, b, rtol=rtol, atol=atol)
    else:
        assert left == right, (left, right)


def compare(root):
    control, resumed = root / "control", root / "resumed"
    spec = json.loads((HERE / "fixture.json").read_text())
    tolerance = spec["comparison"]
    errors = {}
    for step in range(3):
        left = torch.load(control / f"rollouts/{step}.pt", weights_only=False)
        right = torch.load(resumed / f"rollouts/{step}.pt", weights_only=False)
        assert len(left["samples"]) == len(right["samples"]) == 4
        # Match sample identities, not completion order from concurrent requests.
        for a, b in zip(
            sorted(left["samples"], key=lambda s: s["index"]),
            sorted(right["samples"], key=lambda s: s["index"]),
            strict=True,
        ):
            for key in ("index", "prompt", "tokens", "reward", "weight_versions", "rollout_routed_experts"):
                compare_value(a[key], b[key], rtol=0, atol=0)
    for step in (1, 2):
        path = f"save/core/rollout_{step:07d}"
        metadata = checkpoint.get_checkpoint_metadata(control / path / "model")
        keys = sorted(metadata.state_dict_metadata)
        assert keys == sorted(checkpoint.get_checkpoint_metadata(resumed / path / "model").state_dict_metadata)
        left = checkpoint.load_keys(control / path / "model", keys)
        right = checkpoint.load_keys(resumed / path / "model", keys)
        # Includes FP32 master weights, Adam moments and optimizer step state.
        for key, a, b in zip(keys, left, right, strict=True):
            limits = spec["optimizer_moments_comparison"] if key.endswith((".exp_avg", ".exp_avg_sq")) else tolerance
            compare_value(a, b, **limits)
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                errors[f"{step}/{key}"] = (a - b).abs().max().item()
        a = torch.load(control / path / "rank_0.pt", weights_only=False)
        b = torch.load(resumed / path / "rank_0.pt", weights_only=False)
        compare_value(a["scheduler"], b["scheduler"], rtol=0, atol=0)
        a = torch.load(control / f"save/rollout/global_dataset_state_dict_{step}.pt", weights_only=False)
        b = torch.load(resumed / f"save/rollout/global_dataset_state_dict_{step}.pt", weights_only=False)
        compare_value(a, b, rtol=0, atol=0)
    write(root / "comparison.json", {"passed": True, "tolerance": tolerance, "max_abs_errors": errors})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "compare"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--miles-source", type=Path)
    parser.add_argument("--image-id")
    options = parser.parse_args()
    if options.command == "prepare":
        prepare(options.root, options.miles_source, options.image_id)
    elif options.command == "audit":
        audit(options.root)
    else:
        compare(options.root)


if __name__ == "__main__":
    main()
