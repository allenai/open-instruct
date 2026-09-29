"""CPU-safe native argument mapping; no teacher service or OPD signal."""

from pathlib import Path

from open_instruct.miles import options


def conversion_command(executable, miles_root, architecture, model, checkpoint, tensor_parallel):
    """Keep conversion on the same TP-only mesh as the GRPO trainer.

    The native converter overrides even explicit PP1 under world size > 1.
    The managed entry point guards that inference when tensor parallelism is used.
    """
    return [
        executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={tensor_parallel}",
        "-m",
        "open_instruct.miles.megatron_grpo_convert",
        "--native-script",
        str(Path(miles_root) / "tools/convert_hf_to_torch_dist.py"),
        *architecture,
        "--hf-checkpoint",
        str(model),
        "--save",
        str(checkpoint),
        "--tensor-model-parallel-size",
        str(tensor_parallel),
        "--pipeline-model-parallel-size",
        "1",
    ]


def native_arguments(spec, prepared, checkpoint, architecture):
    doc, root = spec.document, Path(spec.output["root"])
    inf, training, trainer, opt, tracking = (
        doc[k] for k in ("inference", "training", "trainer", "optimizer", "tracking")
    )
    values = {
        "train-backend": "megatron",
        "hf-checkpoint": prepared["model"],
        "ref-load": str(checkpoint),
        "save": str(root / "checkpoints"),
        "save-interval": training["save_interval"],
        "save-hf": str(root / "hf-{rollout_id}"),
        "num-rollout": training["num_rollouts"],
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": trainer["gpus"],
        "num-gpus-per-node": trainer["gpus"] + inf["gpus"],
        "rollout-num-gpus": inf["gpus"],
        "rollout-num-gpus-per-engine": inf["tensor_parallel_size"],
        "tensor-model-parallel-size": trainer["tensor_parallel_size"],
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "expert-model-parallel-size": 1,
        "expert-tensor-parallel-size": 1,
        "micro-batch-size": 1,
        "global-batch-size": inf["rollout_batch_size"] * inf["samples_per_prompt"],
        "rollout-batch-size": inf["rollout_batch_size"],
        "n-samples-per-prompt": inf["samples_per_prompt"],
        "prompt-data": prepared["data"]["prompt_data"],
        "input-key": "input",
        "label-key": "label",
        "metadata-key": "metadata",
        "rollout-max-response-len": inf["max_response_length"],
        "rollout-max-prompt-len": inf["max_prompt_length"],
        "rollout-temperature": inf["temperature"],
        "rollout-top-p": inf["top_p"],
        "rollout-seed": doc["data"]["seed"],
        "seed": doc["data"]["seed"],
        "sglang-context-length": inf["max_context_length"],
        "seq-length": inf["max_context_length"],
        "sglang-mem-fraction-static": 0.6,
        "sglang-max-running-requests": inf["max_running_requests"],
        "sglang-max-total-tokens": inf["max_context_length"] * inf["max_running_requests"],
        "sglang-attention-backend": "triton",
        "sglang-sampling-backend": "pytorch",
        "advantage-estimator": "grpo",
        "opd-kl-coef": 0.0,
        "loss-type": "policy_loss",
        "custom-rm-path": "open_instruct.miles.megatron_grpo_hooks.reward",
        "custom-reward-post-process-path": "open_instruct.miles.megatron_grpo_hooks.post_process",
        "eval-interval": training["eval_interval"] or training["num_rollouts"],
        "n-samples-per-eval-prompt": inf["eval_samples_per_prompt"],
        "eval-max-response-len": inf["eval_max_response_length"] or inf["max_response_length"],
        "eval-temperature": inf["eval_temperature"],
        "eval-top-p": inf["eval_top_p"],
        "optimizer": "adam",
        "lr": opt["learning_rate"],
        "lr-decay-style": opt["lr_decay_style"],
        "lr-warmup-iters": opt["lr_warmup_iters"],
        "min-lr": opt["min_lr"],
        "weight-decay": opt["weight_decay"],
        "adam-beta1": opt["adam_beta1"],
        "adam-beta2": opt["adam_beta2"],
        "adam-eps": opt["adam_eps"],
        "clip-grad": opt["clip_grad"],
        "eps-clip": doc["objective"]["eps_clip"],
        "eps-clip-high": doc["objective"]["eps_clip_high"],
        "eps-clip-c": doc["objective"]["eps_clip_c"],
        "kl-coef": 0.0,
        "kl-loss-coef": 0.0,
        "entropy-coef": 0.0,
        "update-weights-interval": 1,
        "attention-dropout": 0.0,
        "hidden-dropout": 0.0,
        "attention-backend": "flash",
        "qkv-format": "thd",
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        "megatron-to-hf-mode": "raw",
        "dump-details": str(root / "debug"),
        "wandb-project": tracking["wandb_project"],
        "wandb-group": spec.name,
        "wandb-dir": str(root / "wandb"),
    }
    if opt["lr_decay_style"] != "constant":
        values["lr-decay-iters"] = training["num_rollouts"]
    if training["keep_checkpoints"]:
        values["custom-megatron-post-save-hook-path"] = "open_instruct.miles.opd_retention.post_save"
    resume = training["resume"] and (root / "checkpoints/latest_checkpointed_iteration.txt").exists()
    if resume:
        values["load"] = str(root / "checkpoints")
    # Encode modeled options independently of the optional GPU runtime.
    args = list(architecture)
    for key, value in values.items():
        args += ["--" + key, str(value)]
    args += [
        "--rollout-shuffle",
        "--sequence-parallel",
        "--sglang-disable-flashinfer-autotune",
        "--sglang-disable-cuda-graph",
        "--sglang-disable-radix-cache",
        "--router-disable-circuit-breaker",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
    ]
    if not doc["objective"]["std_normalization"]:
        args.append("--disable-grpo-std-normalization")
    if training["loss_aggregation"] == "token":
        args.append("--calculate-per-token-loss")
    if resume:
        args.append("--use-checkpoint-opt-param-scheduler")
    if tracking["wandb_mode"] != "disabled":
        args += ["--use-wandb", "--wandb-mode", tracking["wandb_mode"]]
        if tracking["wandb_entity"]:
            args += ["--wandb-team", tracking["wandb_entity"]]
    args += ["--eval-prompt-data", *prepared["data"]["eval_prompt_data"]]
    # Apply only unowned serving/runtime controls; the spec rejects objective overrides.
    overrides = doc["miles"]
    kept, skip = [], False
    index = options.option_index()
    for token in args:
        if token.startswith("--"):
            record = index.get(token[2:].split("=", 1)[0].replace("-", "_"))
            replaced = record is not None and record["dest"] in overrides
            skip = replaced and "=" not in token
            if replaced:
                continue
        elif skip:
            continue
        kept.append(token)
    return kept + options.encode_options(overrides)
