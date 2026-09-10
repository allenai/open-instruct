"""Local GSM8K lifecycle check using OLMo-core's saved local-4090-moe checkpoint.

Run inside the MILES/Core image. The source checkpoint is read-only; prepare
writes a BF16 HF export and a small task slice beneath the output directory.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import ray
import torch
from datasets import load_dataset
from miles.utils import arguments
from olmo_core.config import DType
from olmo_core.distributed import checkpoint as core_checkpoint
from olmo_core.nn import attention
from olmo_core.nn.hf import config as hf_config_utils
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from pyarrow import parquet
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct.ground_truth_utils import GSM8KVerifier
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.driver import train


def prepare(options):
    root = options.output
    root.mkdir(parents=True, exist_ok=True)
    document = json.loads((options.source / "config.json").read_text())
    source = document["model"]
    block = source["block"]
    mixer = block["sequence_mixer"]
    experts = block["routed_experts"]
    router = block["routed_experts_router"]
    # This fixture deliberately supports the saved homogeneous conventional MoE.
    assert source["name"] == "moe_fused_v2" and not block.get("latent_moe")
    assert not block.get("shared_experts") and not block.get("use_pre_norm")
    assert mixer["type"] == "attention" and not mixer.get("gate")
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
    hf_config_utils._register_olmo3moe_auto_classes()
    hf = Olmo3MoeConfig(
        vocab_size=source["vocab_size"],
        hidden_size=source["d_model"],
        attention_hidden_size=mixer["d_attn"],
        head_dim=mixer["head_dim"],
        num_hidden_layers=source["n_layers"],
        num_attention_heads=mixer["n_heads"],
        num_key_value_heads=mixer["n_kv_heads"],
        moe_intermediate_size=experts["hidden_size"],
        n_routed_experts=experts["num_experts"],
        num_experts_per_tok=router["top_k"],
        shared_expert_intermediate_size=None,
        dense_layers_indices=[],
        layer_types=["full_attention"] * source["n_layers"],
        use_head_qk_norm=mixer["use_head_qk_norm"],
        rope_theta=mixer["rope"]["theta"],
        rms_norm_eps=mixer["qk_norm"]["eps"],
        use_peri_ln=block["use_peri_norm"],
        embed_norm=source.get("embedding_norm") is not None,
        embed_scale=source["embed_scale"],
        gating_function=router["gating_function"],
        normalize_expert_weights=router["normalize_expert_weights"],
        restore_weight_scale=router["restore_weight_scale"],
        max_position_embeddings=512,
        sliding_window=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf,
        dtype=DType.bfloat16,
        attention_backend=attention.AttentionBackendName.torch,
        attention_type=attention.AttentionType.fused_v2,
    )
    native = config.build(init_device="cpu")
    targets = native.state_dict()
    # This older DDP checkpoint stores flat FP32 optimizer master parameters.
    keys = [f"module.{name}.main" for name in targets]
    checkpoint_path = options.source / "model_and_optim"
    metadata = core_checkpoint.get_checkpoint_metadata(checkpoint_path)
    assert set(keys) == {key for key in metadata.state_dict_metadata if key.endswith(".main")}
    values = core_checkpoint.load_keys(checkpoint_path, keys)
    state = {
        name: value.reshape(target.shape).to(target.dtype)
        for (name, target), value in zip(targets.items(), values, strict=True)
    }
    native.load_state_dict(state, strict=True)
    exported = olmo3.gather_olmo3_moe_hf_state(native, hf, cpu=True)
    reference = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16)
    reference.load_state_dict(exported, strict=True)
    hf_path = root / "hf"
    if hf_path.exists():
        raise FileExistsError(hf_path)
    reference.save_pretrained(hf_path)
    tokenizer.save_pretrained(hf_path)
    # Validate that the adapter's unfused factory preserves the exported weights.
    imported = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.bfloat16, attention_backend=attention.AttentionBackendName.torch
    ).build(init_device="cpu")
    olmo3.load_olmo3_moe_hf_state(imported, hf, exported)
    roundtrip = olmo3.gather_olmo3_moe_hf_state(imported, hf, cpu=True)
    for name, value in exported.items():
        torch.testing.assert_close(roundtrip[name], value, rtol=0, atol=0)
    del imported
    tokens = tokenizer("Question: What is six times seven? Answer:", return_tensors="pt").input_ids.cuda()
    native.cuda().eval()
    reference.cuda().eval()
    with torch.no_grad():
        native_logits = native(tokens).float()
        hf_logits = reference(tokens, use_cache=False).logits.float()
    cosine = torch.nn.functional.cosine_similarity(native_logits.flatten(), hf_logits.flatten(), dim=0).item()
    assert cosine > 0.999, cosine
    report = dict(
        source=str(options.source),
        source_config=document,
        parameters=sum(p.numel() for p in native.parameters()),
        source_dtype="float32",
        export_dtype="bfloat16",
        exact_bf16_roundtrip=True,
        native_hf_logit_cosine=cosine,
        native_hf_logit_max_abs=(native_logits - hf_logits).abs().max().item(),
    )
    (root / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    write_tasks(root, parquet.read_table(options.dataset).to_pylist()[:8])
    print(json.dumps({key: value for key, value in report.items() if key != "source_config"}, indent=2))


def write_tasks(root, rows):
    prepared = []
    for row in rows:
        messages = row["messages"]
        # The cached RLVR rows also contain reference assistant solutions.
        # Only the conversation prefix preceding the first assistant is a prompt.
        first_assistant = next(
            (i for i, message in enumerate(messages) if message["role"] == "assistant"), len(messages)
        )
        messages = messages[:first_assistant]
        assert messages and messages[-1]["role"] == "user"
        prompt = "\n".join(message["content"] for message in messages) + "\nAnswer:"
        target = row["ground_truth"]
        if isinstance(target, list):
            assert len(target) == 1
            target = target[0]
        if isinstance(target, dict):
            target = target["answer"]
        prepared.append(
            dict(input=prompt, label=target, metadata={"verifiers": [{"name": "gsm8k", "target": target}]})
        )
    (root / "prompts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prepared))
    (root / "verifiers.json").write_text(
        json.dumps({"gsm8k": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"}})
    )


def bootstrap(options):
    """Create fresh random weights and fetch public tasks; no private checkpoint input."""
    root = options.output
    root.mkdir(parents=True, exist_ok=True)
    hf_config_utils._register_olmo3moe_auto_classes()
    torch.manual_seed(17)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    hf = Olmo3MoeConfig(
        vocab_size=50304,
        hidden_size=128,
        attention_hidden_size=128,
        head_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=None,
        dense_layers_indices=[],
        layer_types=["full_attention", "full_attention"],
        use_head_qk_norm=True,
        use_peri_ln=True,
        max_position_embeddings=512,
        sliding_window=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16)
    if (root / "hf").exists():
        raise FileExistsError(root / "hf")
    model.save_pretrained(root / "hf")
    tokenizer.save_pretrained(root / "hf")
    dataset = load_dataset(
        "ai2-adapt-dev/rlvr_gsm8k_zs", revision="93ffaae6cd2acb8f821f6d4712651320a889b1b9", split="train[:8]"
    )
    write_tasks(root, list(dataset))
    (root / "preparation.json").write_text(
        json.dumps(
            dict(
                source="fresh random Olmo3MoeConfig, seed 17; public GPT2 tokenizer and RLVR GSM8K",
                parameters=sum(p.numel() for p in model.parameters()),
                export_dtype="bfloat16",
            ),
            indent=2,
        )
        + "\n"
    )


def hybrid(options):
    """Build a tiny KDA+latent policy using the already prepared public task slice."""
    root = options.output
    root.mkdir(parents=True, exist_ok=True)
    hf_config_utils._register_olmo3moe_auto_classes()
    hf = Olmo3MoeConfig.from_pretrained(options.fixture / "hf")
    hf.layer_types = ["linear_attention", "full_attention"]
    hf.dense_layers_indices = [0]
    hf.dense_mlp_intermediate_size = 256
    hf.dense_mlp_uses_shared_experts = True
    hf.shared_expert_intermediate_size = 128
    hf.latent_moe_dim = 64
    hf.use_rope = False
    hf.attention_gate_type = "elementwise"
    hf.linear_num_key_heads = 8
    hf.linear_num_value_heads = 8
    hf.linear_key_head_dim = 64
    hf.linear_value_head_dim = 64
    torch.manual_seed(17)
    model = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.zero_()
    if (root / "hf").exists():
        raise FileExistsError(root / "hf")
    model.save_pretrained(root / "hf")
    AutoTokenizer.from_pretrained(options.fixture / "hf").save_pretrained(root / "hf")
    for name in ("prompts.jsonl", "verifiers.json"):
        shutil.copyfile(options.fixture / name, root / name)
    (root / "preparation.json").write_text(
        json.dumps(
            dict(
                source="fresh random KDA + latent MoE; seed 17; public tokenizer/task slice",
                parameters=sum(p.numel() for p in model.parameters()),
                export_dtype="bfloat16",
            ),
            indent=2,
        )
        + "\n"
    )


def run(options):
    root = options.output
    if options.expert_parallel_size > 1 and not options.disaggregated:
        raise ValueError("The EP fixture requires disaggregated serving")
    total_gpus = options.expert_parallel_size + 1 if options.disaggregated else 1
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    config = RunConfig(
        CoreConfig(
            attention_backend="torch",
            expert_parallel_size=options.expert_parallel_size,
            stream_moe_export=not options.legacy_export,
            max_train_rollout_logprob_abs_diff=0.05,
            weight_sync_mode="per_tensor" if options.per_tensor else "flattened",
            max_sequence_length=512,
            activation_checkpointing=False,
            reward_config=str(root / "verifiers.json"),
            router_aux_loss_weight=0.01,
            router_z_loss_weight=0.001,
        ),
        dict(
            hf_checkpoint=str(root / "hf"),
            global_batch_size=4,
            rollout_batch_size=1,
            n_samples_per_prompt=4,
            num_rollout=3 if options.resume else 2,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            actor_num_gpus_per_node=options.expert_parallel_size,
            num_gpus_per_node=total_gpus,
            colocate=True,
            offload_rollout=False,
            prompt_data=str(root / "prompts.jsonl"),
            input_key="input",
            label_key="label",
            metadata_key="metadata",
            rollout_max_response_len=32,
            sglang_context_length=512,
            sglang_max_total_tokens=4096,
            sglang_max_running_requests=4,
            sglang_server_concurrency=4,
            sglang_mem_fraction_static=0.2,
            sglang_disable_cuda_graph=True,
            sglang_attention_backend="torch_native",
            sglang_sampling_backend="pytorch",
            custom_rm_path="open_instruct.miles.rewards.registered_reward",
            save=str(root / "save"),
            save_interval=1,
            save_debug_rollout_data=str(root / "rollouts/{rollout_id}.pt"),
            sglang_log_level="warning",
            lr=1e-5,
            lr_decay_iters=3,
        ),
    )
    hf = json.loads((root / "hf/config.json").read_text())
    if "linear_attention" in hf.get("layer_types", []):
        config.miles.update(sglang_disable_radix_cache=True, sglang_max_mamba_cache_size=16)
    if options.disaggregated:
        config.miles.pop("colocate")
    if not options.resume:
        config.miles["check_weight_update_equal"] = True
    if options.resume:
        config.miles.update(load=str(root / "save"), start_rollout_id=2)
    (root / ("resume-config.json" if options.resume else "run-config.json")).write_text(
        json.dumps(config.arguments(), indent=2)
    )
    sys.argv = ["local-moe", *config.arguments()]
    args = arguments.parse_args()
    if options.validate_only:
        print("LOCAL_MOE_CONFIG_VALIDATED")
        return
    ray.init(num_gpus=total_gpus, num_cpus=6, include_dashboard=False, object_store_memory=256 * 1024 * 1024)
    try:
        asyncio.run(train(args))
    finally:
        ray.shutdown()
    latest = json.loads((root / "save/core-latest.json").read_text())
    expected = 2 if options.resume else 1
    assert latest["rollout_id"] == expected
    complete = json.loads((root / f"save/core/rollout_{expected:07d}/complete.json").read_text())
    assert complete["clock"]["completed_steps"] == expected + 1
    print("LOCAL_MOE_GSM8K_LIFECYCLE_PASSED", json.dumps(complete["clock"]))


def audit(options):
    """Independently check task scores, policy versions, cursor and real updates."""
    root = options.output
    verifier = GSM8KVerifier()
    prompts = [json.loads(line) for line in (root / "prompts.jsonl").read_text().splitlines()]
    scores = []
    for rollout_id in range(3):
        data = torch.load(root / f"rollouts/{rollout_id}.pt", weights_only=False)
        assert data["rollout_id"] == rollout_id and len(data["samples"]) == 4
        for sample in data["samples"]:
            assert sample["prompt"] == prompts[rollout_id]["input"]
            assert sample["label"] == prompts[rollout_id]["label"]
            assert set(sample["weight_versions"]) == {str(rollout_id)}
            assert len(sample["rollout_log_probs"]) == sample["response_length"]
            assert torch.isfinite(torch.tensor(sample["rollout_log_probs"])).all()
            score = verifier([], sample["response"], sample["label"]).score
            assert score == sample["reward"]
            scores.append(score)
    hf_config_utils._register_olmo3moe_auto_classes()
    reference = AutoModelForCausalLM.from_pretrained(root / "hf", torch_dtype=torch.bfloat16)
    native = olmo3.build_olmo3_moe_config_from_hf_config(
        reference.config, dtype=DType.bfloat16, attention_backend=attention.AttentionBackendName.torch
    ).build(init_device="cpu")
    olmo3.load_olmo3_moe_hf_state(native, reference.config, reference.state_dict())
    initial = native.state_dict()
    keys = [f"module.{name}.main" for name in initial]
    changed = []
    for step in (1, 2):
        values = core_checkpoint.load_keys(root / f"save/core/rollout_{step:07d}/model", keys)
        differences = {
            name: (value.reshape(tensor.shape).float() - tensor.float()).abs().max().item()
            for (name, tensor), value in zip(initial.items(), values, strict=True)
        }
        assert all(torch.isfinite(torch.tensor(list(differences.values()))))
        assert any(value > 0 for value in differences.values())
        changed.append(differences)
    report = dict(
        verified_samples=len(scores),
        rewards=scores,
        rollout_versions=[0, 1, 2],
        prompt_cursor_continued=True,
        optimizer_steps=3,
        max_parameter_changes_from_initial=changed,
        interpretation="All-zero task rewards imply zero policy advantages; router auxiliary losses drive updates.",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "LOCAL_MOE_AUDIT_PASSED",
        json.dumps({k: v for k, v in report.items() if k != "max_parameter_changes_from_initial"}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "bootstrap", "hybrid", "run", "audit"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disaggregated", action="store_true")
    parser.add_argument("--expert-parallel-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--legacy-export", action="store_true")
    parser.add_argument("--per-tensor", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    if options.command == "prepare":
        prepare(options)
    elif options.command == "hybrid":
        hybrid(options)
    elif options.command == "bootstrap":
        bootstrap(options)
    elif options.command == "run":
        run(options)
    else:
        audit(options)


if __name__ == "__main__":
    main()
