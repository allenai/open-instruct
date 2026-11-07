"""Bounded, real MILES lifecycle over one actual prompt from every qualified source.

This deliberately disables zero-variance filtering for random toy-model rewards.
Policy-learning efficacy is NOT measured; native reward/filter controls are separate.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import ray


def main(options):
    if getattr(options, "context_length", 8192) < 64 or getattr(options, "samples_per_prompt", 8) < 2:
        raise ValueError("Require context >= 64 and at least two completions")
    options.catalog = options.catalog.resolve()
    root = options.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    hf = root / "hf"
    shutil.copytree(options.fixture, hf)
    config = json.loads((hf / "config.json").read_text())
    context = getattr(options, "context_length", 8192)
    samples = getattr(options, "samples_per_prompt", 8)
    config["max_position_embeddings"] = context
    config["sliding_window"] = context
    (hf / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    prompts = [json.loads(line) for line in (options.catalog / "gpu-prompts.jsonl").open()]
    steps = len(prompts)
    if not 1 <= steps <= 128:
        raise ValueError("Coverage run must have 1..128 source representatives")
    spec = {
        "core": {
            "attention_backend": "torch",
            "max_sequence_length": context,
            "activation_checkpointing": False,
            "reward_config": str(options.catalog / "gpu-verifiers.json"),
            "reward_zero_truncated": True,
            "filter_zero_std_groups": False,
            "router_aux_loss_weight": 0.01,
            "router_z_loss_weight": 0.001,
            "max_train_rollout_logprob_abs_diff": 0.05,
        },
        "miles": {
            "hf_checkpoint": str(hf),
            "global_batch_size": samples,
            "rollout_batch_size": 1,
            "n_samples_per_prompt": samples,
            "num_rollout": steps,
            "rollout_num_gpus": 1,
            "rollout_num_gpus_per_engine": 1,
            "actor_num_gpus_per_node": 1,
            "num_gpus_per_node": 1,
            "colocate": True,
            "offload_rollout": False,
            "prompt_data": str(options.catalog / "gpu-prompts.jsonl"),
            "input_key": "input",
            "label_key": "label",
            "metadata_key": "metadata",
            "rollout_shuffle": False,
            "rollout_seed": 17,
            "rollout_max_response_len": 32,
            "sglang_context_length": context,
            "sglang_max_total_tokens": max(32768, context * 2),
            "sglang_max_running_requests": samples,
            "sglang_server_concurrency": samples,
            "sglang_mem_fraction_static": 0.2,
            "sglang_disable_cuda_graph": True,
            "sglang_disable_radix_cache": True,
            "sglang_attention_backend": "torch_native",
            "sglang_sampling_backend": "pytorch",
            "custom_rm_path": "open_instruct.miles.rewards.rewards.registered_reward",
            "save": str(root / "save"),
            "save_interval": steps,
            "save_debug_rollout_data": str(root / "rollouts/{rollout_id}.pt"),
            "rollout_sample_rate": 1.0,
            "sglang_log_level": "warning",
            "lr": 1e-5,
            "lr_decay_iters": steps,
            "check_weight_update_equal": True,
        },
    }
    path = root / "config.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    (root / "fixture-provenance.json").write_text(
        json.dumps(
            {
                "source": str(options.fixture),
                "config_change": f"{context} context for bounded actual prompts; unchanged weights",
                "weights": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in hf.glob("*.safetensors")},
                "purpose": "Mechanics only. Router auxiliary loss can update weights even when policy rewards are zero.",
                "groups": steps,
                "completions_per_group": samples,
                "max_new_tokens": 32,
            },
            indent=2,
        )
        + "\n"
    )
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    ray.init(num_gpus=1, num_cpus=8, include_dashboard=False, object_store_memory=256 * 1024 * 1024)
    try:
        sys.argv = ["open_instruct.miles", "train", str(path), "--debug"]
        runpy.run_module("open_instruct.miles", run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    main(parser.parse_args())
