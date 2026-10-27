"""CPU-safe native argument mapping for teacher-free Megatron verifier GRPO."""

from pathlib import Path

from open_instruct.miles.distillation import opd_config, opd_runtime
from open_instruct.miles.errors import InputError
from open_instruct.miles.megatron_grpo import grpo_config


def preflight_code(spec):
    """Parse the native arguments and assert the objective the run file selected reached Miles."""
    objective = spec.document["objective"]
    selected = (
        bool(spec.document["miles"].get("fully_async", False)),
        objective["use_rollout_logprobs"],
        "custom_loss" if objective["policy_loss"] == "dppo" else "policy_loss",
    )
    return (
        "from miles.utils.arguments import parse_args; "
        "from miles.rollout.data_source import RolloutDataSourceWithBuffer; a=parse_args(); "
        "assert not a.use_opd and a.opd_kl_coef == 0 and a.kl_coef == 0 and not a.use_kl_loss "
        "and not a.normalize_advantages and a.rewards_normalization and a.advantage_estimator == 'grpo'; "
        f"assert (a.fully_async, a.use_rollout_logprobs, a.loss_type) == {selected!r}; "
        "RolloutDataSourceWithBuffer(a)"
    )


def responses_per_rollout(spec):
    """Responses one rollout trains on: the global batch, and one verifier evidence record each."""
    inference = spec.document["inference"]
    return inference["rollout_batch_size"] * inference["samples_per_prompt"]


def checkpoint_marker(root):
    """The file in which Megatron records the latest checkpointed rollout ID."""
    return Path(root) / "checkpoints/latest_checkpointed_iteration.txt"


def checkpointed_rollouts(spec):
    """Rollouts a launch will not replay: the latest checkpoint's rollout ID plus one when resuming, else 0.

    MILES saves the Megatron checkpoint for rollout ``r`` as iteration ``r`` and restarts at
    ``start_rollout_id = r + 1``."""
    marker = checkpoint_marker(spec.output["root"])
    if not (spec.document["training"]["resume"] and marker.exists()):
        return 0
    try:
        return int(marker.read_text().strip()) + 1
    except ValueError as error:
        raise InputError(f"Resume checkpoint marker {marker} does not hold a rollout ID") from error


def native_arguments(spec, prepared, checkpoint, architecture):
    doc, root = spec.document, Path(spec.output["root"])
    inf, training, trainer, opt, tracking, objective = (
        doc[k] for k in ("inference", "training", "trainer", "optimizer", "tracking", "objective")
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
        "global-batch-size": responses_per_rollout(spec),
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
        "loss-type": "custom_loss" if objective["policy_loss"] == "dppo" else "policy_loss",
        "custom-rm-path": grpo_config.REWARD_HOOK,
        "custom-reward-post-process-path": grpo_config.POST_PROCESS_HOOK,
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
    if objective["policy_loss"] == "dppo":
        # DPPO replaces the clipped surrogate; the eps-clip values above are then unused.
        values["custom-loss-function-path"] = grpo_config.DPPO_LOSS
    if doc["miles"].get("fully_async", False):
        # The async OPD restart ledger and measured buffer; grpo_train installs the producer
        # (grpo_async.VerifierAsyncRollout) after native parsing.
        values["data-source-path"] = opd_config.ASYNC_SOURCE
        values["custom-async-data-buffer-path"] = opd_config.ASYNC_BUFFER
    if opt["lr_decay_style"] != "constant":
        values["lr-decay-iters"] = training["num_rollouts"]
    resume = checkpointed_rollouts(spec) > 0
    if resume:
        # Megatron restores weights, optimizer and RNG; MILES restores the data cursor. The
        # scheduler switch below stops MILES stepping the restored LR schedule a second time.
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
    if objective["use_rollout_logprobs"]:
        args.append("--use-rollout-logprobs")
    if resume:
        args.append("--use-checkpoint-opt-param-scheduler")
    if tracking["wandb_mode"] != "disabled":
        args += ["--use-wandb", "--wandb-mode", tracking["wandb_mode"]]
        if tracking["wandb_entity"]:
            args += ["--wandb-team", tracking["wandb_entity"]]
    args += ["--eval-prompt-data", *prepared["data"]["eval_prompt_data"]]
    # [miles] overrides replace only unowned serving/runtime controls; the spec rejects owned ones.
    return opd_runtime.with_native_overrides(args, doc["miles"])
