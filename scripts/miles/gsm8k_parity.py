"""Run the Core arm of the bounded, shared-data GSM8K comparison."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import ray
from miles.utils import arguments
from scripts.miles.prepare_gsm8k_parity import verify_preparation

from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.driver import train

CAMPAIGN = "gsm8k-core-megatron-20260910-v1"
DEFAULT_ROOT = Path("/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1")
UPDATES = 100
EVAL_INTERVAL = 20


def configuration(root):
    output = root / "core"
    return RunConfig(
        CoreConfig(
            expert_parallel_size=2,
            attention_backend="flash_4",
            max_sequence_length=6144,
            activation_checkpointing=True,
            max_train_rollout_logprob_abs_diff=0.05,
            diagnostic_interval=0,
            router_aux_loss_weight=0.01,
            router_z_loss_weight=1e-5,
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
            num_rollout=UPDATES,
            prompt_data=str(root / "train.jsonl"),
            input_key="input",
            label_key="label",
            metadata_key="metadata",
            rollout_temperature=1.0,
            rollout_seed=17,
            rollout_max_response_len=4096,
            rollout_max_prompt_len=2048,
            rollout_max_context_len=6144,
            eval_prompt_data=["gsm8k", str(root / "eval.jsonl")],
            eval_interval=EVAL_INTERVAL,
            eval_temperature=0.0,
            n_samples_per_eval_prompt=1,
            eval_max_response_len=4096,
            custom_rm_path="open_instruct.miles.rewards.registered_reward",
            sglang_context_length=6144,
            sglang_attention_backend="triton",
            sglang_max_total_tokens=32768,
            sglang_max_running_requests=4,
            sglang_server_concurrency=4,
            sglang_mem_fraction_static=0.6,
            sglang_disable_radix_cache=True,
            sglang_max_mamba_cache_size=8,
            sglang_cuda_graph_backend_decode="full",
            sglang_cuda_graph_max_bs_decode=4,
            sglang_cuda_graph_backend_prefill="disabled",
            sglang_sampling_backend="pytorch",
            sglang_log_level="warning",
            check_weight_update_equal=True,
            update_weight_buffer_size=1024**3,
            save=str(output / "metrics"),
            save_debug_rollout_data=str(output / "rollouts/{rollout_id}.pt"),
            lr=1e-6,
            lr_decay_style="constant",
            lr_decay_iters=UPDATES,
            lr_warmup_iters=0,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            clip_grad=1.0,
            eps_clip=0.2,
            eps_clip_high=0.28,
            disable_grpo_std_normalization=True,
            seed=17,
            use_wandb=True,
            wandb_mode="online",
            wandb_team="ai2-llm",
            wandb_project="olmo-rl-comparison",
            wandb_group=CAMPAIGN + "-core",
            wandb_dir=str(output / "wandb"),
            disable_wandb_random_suffix=True,
            wandb_always_use_train_step=True,
        ),
    )


def effective_settings(args):
    """Record actual parser defaults too, so implicit differences are visible."""
    expected = {
        "num_rollout": UPDATES,
        "global_batch_size": 16,
        "rollout_batch_size": 4,
        "n_samples_per_prompt": 4,
        "eval_interval": EVAL_INTERVAL,
        "rollout_shuffle": False,
        "skip_eval_before_train": False,
        "grpo_std_normalization": False,
        "normalize_advantages": False,
        "calculate_per_token_loss": False,
        "use_rollout_logprobs": False,
        "skip_actor_forward_only": False,
        "use_rollout_routing_replay": False,
        "use_routing_replay": False,
        "fully_async": False,
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "lr": 1e-6,
        "lr_decay_style": "constant",
        "lr_warmup_iters": 0,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
        "clip_grad": 1.0,
        "save_interval": None,
    }
    actual = {name: getattr(args, name) for name in expected}
    differences = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
    if differences:
        raise ValueError(f"GSM8K comparison settings differ from the frozen protocol: {differences}")
    return actual


def check_completion(root):
    output = root / "core"
    records = [
        json.loads(line) for line in (output / "metrics/training_contract_rank0.jsonl").read_text().splitlines()
    ]
    steps = [row["step"] for row in records if row["event"] == "optimizer"]
    publications = [json.loads(line) for line in (output / "metrics/publication.jsonl").read_text().splitlines()]
    if steps != list(range(1, UPDATES + 1)) or [row["version"] for row in publications] != list(range(UPDATES + 1)):
        raise ValueError("Missing optimizer steps or weight publications")
    names = (
        [str(i) for i in range(UPDATES)]
        + ["eval_0"]
        + [f"eval_{i - 1}" for i in range(EVAL_INTERVAL, UPDATES + 1, EVAL_INTERVAL)]
    )
    for name in names:
        if not (output / f"rollouts/{name}.pt").is_file():
            raise ValueError(f"Missing retained rollout/evaluation: {name}")
    return {"optimizer_steps": len(steps), "publication_count": len(publications), "rollout_files": len(names)}


def run(root, validate_only=False):
    preparation = verify_preparation(root)
    config = configuration(root)
    sys.argv = ["gsm8k-parity-core", *config.arguments()]
    args = arguments.parse_args()
    effective = effective_settings(args)
    if validate_only:
        print("GSM8K_PARITY_CONFIG_VALIDATED", json.dumps(effective), flush=True)
        return
    output = root / "core"
    output.mkdir()  # Never overwrite a prior comparison arm.
    (output / "arguments.json").write_text(json.dumps(config.arguments(), indent=2) + "\n")
    (output / "effective.json").write_text(json.dumps(effective, indent=2) + "\n")
    (output / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    started = time.monotonic()
    ray.init(num_gpus=3, num_cpus=16, include_dashboard=False, object_store_memory=1024**3)
    try:
        asyncio.run(train(args))
    finally:
        ray.shutdown()
    report = {**check_completion(root), "elapsed_seconds": time.monotonic() - started, "completed": True}
    (output / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print("GSM8K_PARITY_CORE_COMPLETED", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=DEFAULT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run(args.root, args.validate_only)


if __name__ == "__main__":
    main()
