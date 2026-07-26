import enum
import itertools
import math
import os
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Literal

import numpy as np
import pandas as pd
import ray
import ray.util.queue as ray_queue
import torch
import torch.distributed as dist
import wandb
from datasets import Dataset

from open_instruct import data_loader as data_loader_lib
from open_instruct import data_types, logger_utils, model_utils, olmo_core_utils, utils
from open_instruct.rl_utils import masked_mean
from open_instruct.utils import (
    INVALID_LOGPROB,
    calibrate_checkpoint_state_dir,
    download_latest_checkpoint_from_gs,
    ensure_universal_checkpoint_exists,
    get_beaker_whoami,
)

logger = logger_utils.setup_logger(__name__)
TORCH_DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def compute_pass_at_k_metrics(correct_per_prompt: np.ndarray) -> dict[str, float]:
    """Average pass@1 plus unbiased pass@k (Chen et al.) for k in 1, 2, 4, ... <= n.

    ``correct_per_prompt`` is shape ``(num_prompts, num_completions)``; truthy entries mark correct
    completions.

    ``eval/pass_at_1`` is the average over prompts of ``c/n``, where ``c`` is the number of correct
    completions for that prompt and ``n`` is the number of samples (same as ``eval/pass_at_1_unbiased``).
    When ``n > 1``, ``eval/pass_at_{n}`` is the fraction with at least one correct completion.
    ``eval/pass_at_{k}_unbiased`` uses ``1 - C(n-c, k) / C(n, k)`` per prompt (averaged), when there
    are at least k incorrect completions; otherwise 1.0.
    """
    arr = np.asarray(correct_per_prompt, dtype=bool)
    if arr.ndim != 2 or arr.shape[1] < 1:
        return {}
    num_samples = int(arr.shape[1])
    c_arr = arr.sum(axis=1).astype(np.int64).reshape(-1)
    metrics: dict[str, float] = {"eval/pass_at_1": float((c_arr.astype(np.float64) / num_samples).mean())}
    if num_samples > 1:
        metrics[f"eval/pass_at_{num_samples}"] = float((c_arr > 0).mean())
    k_pow = 1
    while k_pow <= num_samples:
        estimates: list[float] = []
        for c in c_arr:
            c_int = int(c)
            wrong = num_samples - c_int
            if wrong >= k_pow:
                estimates.append(1.0 - math.comb(wrong, k_pow) / math.comb(num_samples, k_pow))
            else:
                estimates.append(1.0)
        metrics[f"eval/pass_at_{k_pow}_unbiased"] = float(np.mean(estimates))
        k_pow *= 2
    return metrics


class GRPOLossType(enum.StrEnum):
    dapo = "dapo"
    cispo = "cispo"
    dppo = "dppo"


PolicyRatioDenominator = Literal["old_policy", "rollout_policy"]
RolloutImportanceCorrection = Literal["none", "clipped"]
RhoMaskMetric = Literal["none", "ratio", "tv", "kl"]
RhoMaskSource = Literal["old_policy", "current_policy"]
RhoMaskLevel = Literal["token", "sequence"]
RhoMaskDirection = Literal["symmetric", "increase_only"]


@dataclass
class GRPOExperimentConfig(
    olmo_core_utils.ExperimentConfig,
    olmo_core_utils.TrainingConfig,
    olmo_core_utils.LoggingConfig,
    olmo_core_utils.CheckpointConfig,
):
    # Optimizer
    set_weight_decay_on_bias_and_norm: bool = True
    """Whether to set weight decay on bias and norm layers"""

    # Batch sizes
    total_episodes: int = 100000
    """The total number of episodes in the dataset"""
    world_size: int | None = None
    """RUNTIME VALUE: The number of processes (GPUs) to use for training ONLY"""
    num_training_steps: int | None = None
    """RUNTIME VALUE: The number of training_steps to train"""
    local_eval_every: int = 100
    """Run evaluation after this many training steps. This controls in-loop evals, which reuse the generation/reward verifier setup. Set to -1 to disable."""
    save_freq: int = 200
    """How many train steps to save the model"""
    backend_timeout: int = 120
    """Timeout for inference/training backends in minutes. Default is 2 hours (120 min)."""
    model_dtype: str = "bfloat16"
    """Model dtype for training. Supported values: 'bfloat16', 'float32'."""

    # Algorithm
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    clip_lower: float = 0.2
    """the lower clip range"""
    clip_higher: float = 0.272
    """the higher clip range. Sometimes we want this to be higher, see DAPO (https://arxiv.org/abs/2503.14476)"""
    policy_ratio_denominator: PolicyRatioDenominator = "old_policy"
    """Policy used in the denominator of the policy-gradient ratio.
    ``old_policy`` preserves PPO-style updates and retains old logprobs;
    ``rollout_policy`` uses vLLM logprobs directly and does not retain old logprobs."""
    rollout_importance_correction: RolloutImportanceCorrection = "clipped"
    """How to correct rollout samples to the old trainer policy.
    ``clipped`` applies the configured clamp to π_old / μ; ``none`` uses unit weights.
    Must be ``none`` when ``policy_ratio_denominator=rollout_policy`` because π_θ / μ
    is already the policy ratio."""
    rho_clamp_lower_bound: float = 0.0
    """Lower bound for clamping π_old / μ when rollout importance correction is enabled (0 disables)."""
    rho_clamp_upper_bound: float = 2.0
    """Upper bound for clamping π_old / μ when rollout importance correction is enabled (0 disables)."""
    rho_mask_lower_bound: float = 0.0
    """Drop tokens whose configured rho-mask statistic falls below this value (0 disables)."""
    rho_mask_upper_bound: float = 0.0
    """Drop tokens whose configured rho-mask statistic exceeds this value (0 disables)."""
    rho_mask_metric: RhoMaskMetric = "ratio"
    """Statistic used by the token-drop mask: none, probability ratio, binary TV, or binary KL."""
    rho_mask_source: RhoMaskSource = "old_policy"
    """Policy compared with the rollout policy when constructing the token-drop mask."""
    rho_mask_level: RhoMaskLevel = "token"
    """Whether the token-drop statistic is applied per token or averaged per response sequence."""
    rho_mask_direction: RhoMaskDirection = "symmetric"
    """Whether threshold violations always drop or only drop updates that increase divergence."""
    kl_estimator: Literal[0, 1, 2, 3] = 2
    """the KL estimator to use"""
    loss_denominator: str = "token"
    """Optional constant denominator for masked_mean; can be "token" or a float value.
    when "token", the loss is divided by the total number of tokens in the batch (standard LM training).
    when a float value, the loss is divided by this value (ideally, max tokens in batch, per Dr GRPO).
    """
    alpha: float = 0.6
    """The alpha value for doing polyak updates (ref_param = alpha * param + (1 - alpha) * ref_param)
    reference: [TR-DPO](https://huggingface.co/papers/2404.09656), but it's actually pretty commonly
    used. E.g., [TD3](https://arxiv.org/abs/1802.09477) uses https://github.com/vwxyzjn/cleanrl/blob/dcc289fc6f0bda492fa7360a155262cf826b12a5/cleanrl/td3_continuous_action.py#L269
    """
    ref_policy_update_freq: int | None = None
    """How many training steps to take before updating the reference policy."""
    load_ref_policy: bool = True
    """Whether to load and use a reference policy for KL penalty calculation."""
    mask_reference_kl_with_policy: bool = False
    """Whether the final policy update mask also removes the reference-KL term.
    False (default) preserves legacy behavior: policy-clipped or divergence-filtered
    response tokens still receive reference-KL gradient when their logprobs are valid.
    True couples the policy and reference-KL masks."""
    loss_fn: GRPOLossType = GRPOLossType.dapo
    """Which mask/cap to apply to the common ``-ρ · advantage · log π_θ`` objective."""
    record_entropy: bool = False
    """whether to record the entropy of the policy during training. Uses extra memory."""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    num_learners_per_node: list[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node (e.g., --num_learners_per_node 2 4 means 2 learner processes
    on the first node and 4 learner processes on the second node; each process will have 1 GPU)"""
    num_nodes: int = 1
    """Number of nodes for distributed training."""
    sequence_parallel_size: int = 1
    """sequence parallel size - how many GPUs we will parallelize sequences across during training.
    Useful for super-long context lengths."""
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    deepspeed_zpg: int = 8
    """the deepspeed zpg value. Higher values are more memory efficient but slower. Set to 1 to disable zpg, which uses less memory but is significantly slower. Ideally is set to the number of GPUs per node (usually 8, default)."""
    deepspeed_offload_param: bool = False
    """whether to offload parameters to CPU (reduces GPU memory usage)"""
    deepspeed_offload_optimizer: bool = False
    """whether to offload optimizer states to CPU (reduces GPU memory usage)"""
    deepspeed_checkpoint_load_universal: bool = False
    """DeepSpeed checkpoint.load_universal: load checkpoints across different parallel configs"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""
    fsdp_shard_degree: int | None = None
    """FSDP shard degree. None means auto-detect."""
    fsdp_num_replicas: int | None = None
    """Number of FSDP replicas. None means auto-detect."""
    enable_queue_dashboard: bool = True
    """whether to enable the ActorManager queue monitoring dashboard"""
    queue_dashboard_port: int | None = None
    """optional port for the dashboard server (if None, finds a free port automatically)"""

    # Experiment tracking
    verbose: bool = False
    """If toggled, debug output will be shown"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: str | None = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: str | None = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: str | None = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    checkpoint_state_freq: int = 200
    """How often to save the model checkpoint, optimizer states, and lr scheduler states (in steps)"""
    checkpoint_state_dir: str | None = None
    """Where to save the model checkpoint (if applicable)"""
    gs_checkpoint_state_dir: str | None = None
    """The actual `checkpoint_state_dir` to use (handling the case where gs_bucket_path is provided)"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    oe_eval_beaker_image: str | None = None
    """the docker image for evaluation for oe-eval"""
    oe_eval_gpu_multiplier: int | None = None
    """multiply the gpus used for each oe-eval task"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""
    eval_workspace: str = "ai2/tulu-3-results"
    """the workspace to launch evaluation jobs on"""
    send_slack_alerts: bool = False
    """Whether to send Slack alerts on training failures"""

    # Evaluation behavior
    eval_on_step_0: bool = False
    """Whether to run local evaluation at training step 0. Defaults to False."""
    eval_pass_at_k: int = 1
    """Number of completions per eval prompt for local pass@k metrics."""
    eval_top_p: float | None = None
    """Optional eval-only top_p override. If None, uses training top_p."""

    def __post_init__(self):
        if self.send_slack_alerts and not os.environ.get("SLACK_WEBHOOK_URL"):
            logger.warning(
                "--send_slack_alerts is set but SLACK_WEBHOOK_URL is not in the environment. Slack alerts will not be sent."
            )
        if self.loss_denominator != "token" and float(self.loss_denominator) <= 0:
            raise ValueError(
                f"loss_denominator must be a valid float greater than 0 if not 'token', got: {self.loss_denominator}"
            )
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq <= 0:
            raise ValueError("`checkpoint_state_freq` must be greater than 0 if `checkpoint_state_dir` is provided!")
        if self.save_freq != self.checkpoint_state_freq:
            logger.warning(
                "On the olmo-core training path, --save_freq is a no-op for periodic saves; "
                "olmo-core checkpoints are full training state and saved every "
                "--checkpoint_state_freq steps (got save_freq=%d, checkpoint_state_freq=%d).",
                self.save_freq,
                self.checkpoint_state_freq,
            )

        if self.gs_checkpoint_state_dir is not None and not self.gs_checkpoint_state_dir.startswith("gs://"):
            raise ValueError(f"`gs_checkpoint_state_dir` must start with 'gs://', got: {self.gs_checkpoint_state_dir}")
        if self.eval_on_step_0 and self.local_eval_every <= 0:
            raise ValueError(
                "`eval_on_step_0` requires `local_eval_every` > 0. "
                "Set `local_eval_every` to a positive value or disable `eval_on_step_0`."
            )
        if self.gs_bucket_path is not None and not self.gs_bucket_path.startswith("gs://"):
            raise ValueError(f"`gs_bucket_path` must start with 'gs://', got: {self.gs_bucket_path}")
        if self.sequence_parallel_size > 1 and self.deepspeed_stage != 3:
            raise ValueError("`sequence_parallel_size` > 1 requires `deepspeed_stage` to be 3!")

        total_learner_gpus = sum(self.num_learners_per_node)
        if self.fsdp_shard_degree is not None and self.fsdp_num_replicas is not None:
            expected = self.fsdp_shard_degree * self.fsdp_num_replicas
            if expected != total_learner_gpus:
                raise ValueError(
                    f"fsdp_shard_degree ({self.fsdp_shard_degree}) * fsdp_num_replicas ({self.fsdp_num_replicas}) "
                    f"= {expected}, but total learner GPUs = {total_learner_gpus} "
                    f"(from num_learners_per_node={self.num_learners_per_node}). These must match."
                )
        elif self.fsdp_shard_degree is not None:
            if total_learner_gpus % self.fsdp_shard_degree != 0:
                raise ValueError(
                    f"fsdp_shard_degree ({self.fsdp_shard_degree}) must evenly divide "
                    f"total learner GPUs ({total_learner_gpus})."
                )
        elif self.fsdp_num_replicas is not None:
            if total_learner_gpus % self.fsdp_num_replicas != 0:
                raise ValueError(
                    f"fsdp_num_replicas ({self.fsdp_num_replicas}) must evenly divide "
                    f"total learner GPUs ({total_learner_gpus})."
                )

        if self.gs_bucket_path is not None and self.gs_checkpoint_state_dir is None:
            if self.checkpoint_state_dir is None:
                raise ValueError("`checkpoint_state_dir` must be provided when using `gs_bucket_path`!")
            checkpoint_dir_name = self.checkpoint_state_dir.rstrip("/")
            beaker_users = get_beaker_whoami()
            if beaker_users is not None:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{beaker_users}/{checkpoint_dir_name}"
            else:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{checkpoint_dir_name}"
            if not checkpoint_dir_name.startswith("/filestore"):
                self.checkpoint_state_dir = f"/filestore{self.checkpoint_state_dir}"

        if self.checkpoint_state_dir is not None:
            if self.gs_checkpoint_state_dir is not None:
                download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
            if self.deepspeed_checkpoint_load_universal:
                ensure_universal_checkpoint_exists(self.checkpoint_state_dir)
        if not self.load_ref_policy and self.beta != 0.0:
            raise ValueError(
                "When load_ref_policy=False, beta must be 0.0. "
                f"Got beta={self.beta}. Set --beta 0.0 or --load_ref_policy to use KL penalty."
            )
        if self.eval_top_p is not None and not (0.0 < self.eval_top_p <= 1.0):
            raise ValueError(f"`eval_top_p` must be in (0, 1], got {self.eval_top_p}")
        valid_options = {
            "policy_ratio_denominator": ("old_policy", "rollout_policy"),
            "rollout_importance_correction": ("none", "clipped"),
            "rho_mask_metric": ("none", "ratio", "tv", "kl"),
            "rho_mask_source": ("old_policy", "current_policy"),
            "rho_mask_level": ("token", "sequence"),
            "rho_mask_direction": ("symmetric", "increase_only"),
        }
        for name, options in valid_options.items():
            value = getattr(self, name)
            if value not in options:
                raise ValueError(f"`{name}` must be one of {options}, got {value!r}.")
        if self.policy_ratio_denominator == "rollout_policy" and self.rollout_importance_correction != "none":
            raise ValueError(
                "`rollout_importance_correction` must be `none` when "
                "`policy_ratio_denominator=rollout_policy` because π_θ / μ is already the policy ratio."
            )
        if (
            self.rho_mask_metric != "none"
            and self.rho_mask_source == "old_policy"
            and self.policy_ratio_denominator != "old_policy"
        ):
            raise ValueError(
                "`rho_mask_source=old_policy` requires `policy_ratio_denominator=old_policy` so old logprobs exist."
            )
        if self.rollout_importance_correction == "clipped":
            if self.rho_clamp_lower_bound > 0.0 and self.rho_clamp_lower_bound >= 1.0:
                raise ValueError(
                    f"rho_clamp_lower_bound must satisfy 0 < lb < 1 when set, got {self.rho_clamp_lower_bound}."
                )
            if self.rho_clamp_upper_bound > 0.0 and self.rho_clamp_upper_bound <= 1.0:
                raise ValueError(f"rho_clamp_upper_bound must be > 1 when set, got {self.rho_clamp_upper_bound}.")
        if self.rho_mask_metric == "none":
            if self.rho_mask_lower_bound != 0.0 or self.rho_mask_upper_bound != 0.0:
                raise ValueError("rho mask bounds must both be 0 when `rho_mask_metric=none`.")
        elif self.rho_mask_metric == "ratio":
            if self.rho_mask_lower_bound > 0.0 and not (0.0 < self.rho_mask_lower_bound < 1.0):
                raise ValueError(
                    f"rho_mask_lower_bound must satisfy 0 < lb < 1 when set, got {self.rho_mask_lower_bound}."
                )
            if self.rho_mask_upper_bound > 0.0 and self.rho_mask_upper_bound <= 1.0:
                raise ValueError(f"rho_mask_upper_bound must be > 1 when set, got {self.rho_mask_upper_bound}.")
        else:
            if self.rho_mask_lower_bound != 0.0:
                raise ValueError(
                    f"`rho_mask_lower_bound` is not used with {self.rho_mask_metric} divergence masking; "
                    f"set it to 0 (got {self.rho_mask_lower_bound})."
                )
            if self.rho_mask_upper_bound < 0.0 or not math.isfinite(self.rho_mask_upper_bound):
                raise ValueError(
                    f"rho_mask_upper_bound must be finite and >= 0 for {self.rho_mask_metric}, "
                    f"got {self.rho_mask_upper_bound}."
                )
        if self.loss_fn == GRPOLossType.dppo:
            required = {
                "policy_ratio_denominator": (self.policy_ratio_denominator, "rollout_policy"),
                "rollout_importance_correction": (self.rollout_importance_correction, "none"),
                "rho_mask_source": (self.rho_mask_source, "current_policy"),
                "rho_mask_level": (self.rho_mask_level, "token"),
                "rho_mask_direction": (self.rho_mask_direction, "increase_only"),
            }
            invalid = [
                f"{name}={actual!r} (expected {expected!r})"
                for name, (actual, expected) in required.items()
                if actual != expected
            ]
            if invalid:
                raise ValueError("DPPO requires " + ", ".join(invalid) + ".")
            if self.rho_mask_metric not in ("tv", "kl"):
                raise ValueError(f"DPPO requires `rho_mask_metric` to be `tv` or `kl`, got {self.rho_mask_metric!r}.")
            if not math.isfinite(self.rho_mask_upper_bound) or self.rho_mask_upper_bound <= 0.0:
                raise ValueError(
                    "DPPO divergence masking requires `rho_mask_upper_bound` (the trust-region "
                    f"threshold δ) to be finite and > 0, got {self.rho_mask_upper_bound}."
                )


def mask_logprobs(vllm_logprobs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Set non-response positions to INVALID_LOGPROB and replace NaNs."""
    vllm_logprobs = torch.masked_fill(vllm_logprobs, ~response_mask, INVALID_LOGPROB)
    vllm_logprobs = torch.nan_to_num(vllm_logprobs, nan=INVALID_LOGPROB)
    return vllm_logprobs


def compute_vllm_local_debug_metrics(
    local_logprobs: torch.Tensor, vllm_logprobs: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, float]:
    """Compute debug metrics comparing vLLM logprobs against locally-recomputed logprobs."""
    with torch.no_grad():
        valid_mask = (
            response_mask.bool()
            & torch.isfinite(local_logprobs)
            & torch.isfinite(vllm_logprobs)
            & (local_logprobs <= 0)
            & (vllm_logprobs <= 0)
        )
        valid_count = valid_mask.sum()
        local_f = local_logprobs.detach().to(torch.float32)
        vllm_f = vllm_logprobs.detach().to(torch.float32)
        masked_diff = torch.where(valid_mask, (local_f - vllm_f).abs(), torch.zeros_like(local_f))
        mean_diff = masked_diff.sum() / valid_count if valid_count > 0 else torch.tensor(0.0)
        max_diff = masked_diff.max() if valid_count > 0 else torch.tensor(0.0)
        std_diff = masked_diff[valid_mask].std() if valid_count > 1 else torch.tensor(0.0)

        # These actions are already sampled from μ, so log μ(a|s) - log π(a|s)
        # is the Monte Carlo estimator of KL(μ || π). Multiplying by μ again
        # would double-weight the behavior-policy probability.
        masked_reverse_kl = torch.where(valid_mask, vllm_f - local_f, torch.zeros_like(local_f))
        mean_reverse_kl = masked_reverse_kl.sum() / valid_count if valid_count > 0 else torch.tensor(0.0)

    return {
        "debug/vllm_vs_local_logprob_diff_mean": float(mean_diff),
        "debug/vllm_vs_local_logprob_diff_max": float(max_diff),
        "debug/vllm_vs_local_logprob_diff_std": float(std_diff),
        "debug/vllm_local_reverse_kl": float(mean_reverse_kl),
    }


def _rho_drop_masks(
    rho: torch.Tensor, response_mask: torch.Tensor, lower: float, upper: float
) -> tuple[torch.Tensor, torch.Tensor]:
    dropped_low = (rho < lower) & response_mask if lower > 0.0 else torch.zeros_like(response_mask)
    dropped_high = (rho > upper) & response_mask if upper > 0.0 else torch.zeros_like(response_mask)
    return dropped_low, dropped_high


def _sequence_level_mean(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence masked mean, broadcast back to every token.

    Sequences are identified with rows of ``values`` (shape [B, T]); padding tokens
    are excluded from the count. Empty rows return 0.
    """
    valid = response_mask.float()
    seq_sum = (values * valid).sum(dim=-1, keepdim=True)
    seq_count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    seq_mean = seq_sum / seq_count
    return seq_mean.expand_as(values)


@dataclass
class RhoCorrection:
    """Detached policy-gradient weights and the final per-token update mask.

    ``ratio`` is the configured policy ratio (``π_θ / π_old`` or ``π_θ / μ``).
    ``rho`` is ``π_old / μ`` when old logprobs are retained, otherwise it is
    the direct ``π_θ / μ`` ratio. ``weights`` is the sole coefficient used by
    the score-function loss after the configured correction, cap, and masks.
    ``valid_mask`` excludes padding and invalid loss metadata before filtering.
    ``metrics`` maps wandb keys to per-token tensors that get reduced by
    ``masked_mean(., response_mask)`` at logging time.
    ``histogram_metrics`` maps wandb keys to flat 1D tensors of values
    (response tokens only); these bypass the scalar reduction and are
    concatenated across micro-batches and logged as wandb histograms.
    """

    weights: torch.Tensor
    mask: torch.Tensor
    valid_mask: torch.Tensor
    ratio: torch.Tensor
    rho: torch.Tensor
    clipfrac: torch.Tensor
    metrics: dict[str, torch.Tensor]
    histogram_metrics: dict[str, torch.Tensor] = field(default_factory=dict)


def compute_binary_divergence(
    behavior_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    divergence_type: Literal["tv", "kl"],
) -> torch.Tensor:
    """Per-token binary (Bernoulli) divergence between behavior and policy.

    Implements the binary approximation from Eqs. 13/14 of the DPPO paper
    (https://arxiv.org/abs/2602.04879): collapse the categorical distribution
    over the vocabulary into a Bernoulli over ``{sampled_token, all_others}``
    using only the per-token logprobs. This is a memory-cheap lower bound on
    the true policy divergence that requires no extra forward passes. It is the
    shared divergence measure behind both the ``vaco`` and ``dppo`` masking
    algorithms in :func:`compute_rho_correction`.

    Args:
        behavior_logprobs: log μ(a_t|s_t), the rollout (vLLM) policy.
        policy_logprobs:   log π(a_t|s_t), the current trainer policy.
        response_mask:     bool mask selecting valid response positions.
        divergence_type:   ``tv`` for total variation or ``kl`` for KL.

    Returns:
        Float tensor of the same shape as ``policy_logprobs``; entries outside
        ``response_mask`` are zeroed.
    """
    if behavior_logprobs.shape != policy_logprobs.shape or policy_logprobs.shape != response_mask.shape:
        raise ValueError(
            "behavior_logprobs, policy_logprobs, and response_mask must have matching shapes, got "
            f"{behavior_logprobs.shape}, {policy_logprobs.shape}, and {response_mask.shape}."
        )
    orig_dtype = policy_logprobs.dtype
    # Float64 avoids cancellation in 1 - p for sampled-token probabilities
    # close to one. Those values determine a hard DPPO keep/drop decision.
    behavior_logprobs_f = behavior_logprobs.detach().to(torch.float64).clamp(max=0.0)
    policy_logprobs_f = policy_logprobs.detach().to(torch.float64).clamp(max=0.0)
    eps = torch.finfo(torch.float64).eps
    mu = torch.exp(behavior_logprobs_f).clamp(eps, 1.0 - eps)
    pi = torch.exp(policy_logprobs_f).clamp(eps, 1.0 - eps)
    if divergence_type == "tv":
        divergence = (mu - pi).abs()
    elif divergence_type == "kl":
        divergence = mu * (mu.log() - pi.log()) + (1.0 - mu) * (torch.log1p(-mu) - torch.log1p(-pi))
        divergence = divergence.clamp_min(0.0)
    else:
        raise ValueError(f"Unknown binary divergence type: {divergence_type}. Expected `tv` or `kl`.")
    return torch.where(response_mask.bool(), divergence, torch.zeros_like(divergence)).to(orig_dtype)


@torch.no_grad()
def compute_rho_correction(
    vllm_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    config: GRPOExperimentConfig,
    old_logprobs: torch.Tensor | None = None,
) -> RhoCorrection:
    """Build the detached policy-gradient coefficient and structural update mask."""
    expected_shape = new_logprobs.shape
    tensors = [("vllm_logprobs", vllm_logprobs), ("response_mask", response_mask), ("advantages", advantages)]
    if old_logprobs is not None:
        tensors.append(("old_logprobs", old_logprobs))
    for name, tensor in tensors:
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tensor.shape} must match new_logprobs shape {expected_shape}.")
    if config.policy_ratio_denominator == "old_policy" and old_logprobs is None:
        raise ValueError("old_logprobs is required when `policy_ratio_denominator=old_policy`.")
    if config.policy_ratio_denominator == "rollout_policy" and old_logprobs is not None:
        raise ValueError("old_logprobs must be omitted when `policy_ratio_denominator=rollout_policy`.")

    new_logprobs_f = new_logprobs.detach().to(torch.float32)
    vllm_logprobs_f = vllm_logprobs.detach().to(torch.float32)
    advantages_f = advantages.detach().to(torch.float32)
    old_logprobs_f = old_logprobs.detach().to(torch.float32) if old_logprobs is not None else None
    valid_mask = (
        response_mask.bool()
        & torch.isfinite(new_logprobs_f)
        & torch.isfinite(vllm_logprobs_f)
        & torch.isfinite(advantages_f)
        & (new_logprobs_f <= 0)
        & (vllm_logprobs_f <= 0)
    )
    if old_logprobs_f is not None:
        valid_mask &= torch.isfinite(old_logprobs_f) & (old_logprobs_f <= 0)

    safe_new_logprobs = torch.where(valid_mask, new_logprobs_f, torch.zeros_like(new_logprobs_f))
    safe_vllm_logprobs = torch.where(valid_mask, vllm_logprobs_f, torch.zeros_like(vllm_logprobs_f))
    safe_old_logprobs = (
        torch.where(valid_mask, old_logprobs_f, torch.zeros_like(old_logprobs_f))
        if old_logprobs_f is not None
        else None
    )

    denominator_logprobs = safe_old_logprobs if config.policy_ratio_denominator == "old_policy" else safe_vllm_logprobs
    assert denominator_logprobs is not None
    policy_log_ratio = safe_new_logprobs - denominator_logprobs
    policy_ratio = torch.exp(policy_log_ratio)
    policy_ratio_is_finite = torch.isfinite(policy_ratio)

    log_rho = safe_old_logprobs - safe_vllm_logprobs if safe_old_logprobs is not None else policy_log_ratio
    rho = torch.exp(log_rho)
    rho_is_finite = torch.isfinite(rho)

    dropped_low = torch.zeros_like(valid_mask)
    dropped_high = torch.zeros_like(valid_mask)
    metrics: dict[str, torch.Tensor] = {}
    mask_log_ratio = torch.zeros_like(policy_log_ratio)
    if config.rho_mask_metric != "none":
        mask_policy_logprobs = safe_old_logprobs if config.rho_mask_source == "old_policy" else safe_new_logprobs
        assert mask_policy_logprobs is not None
        mask_log_ratio = mask_policy_logprobs - safe_vllm_logprobs
    if config.rho_mask_metric == "ratio":
        mask_statistic = torch.exp(mask_log_ratio)
        if config.rho_mask_level == "sequence":
            mask_statistic = torch.exp(_sequence_level_mean(mask_log_ratio, valid_mask))
        dropped_low, dropped_high = _rho_drop_masks(
            mask_statistic, valid_mask, config.rho_mask_lower_bound, config.rho_mask_upper_bound
        )
    elif config.rho_mask_metric in ("tv", "kl"):
        divergence = compute_binary_divergence(
            behavior_logprobs=safe_vllm_logprobs,
            policy_logprobs=mask_policy_logprobs,
            response_mask=valid_mask,
            divergence_type=config.rho_mask_metric,
        )
        if config.rho_mask_level == "sequence":
            divergence = _sequence_level_mean(divergence, valid_mask)
        dropped_low, dropped_high = _rho_drop_masks(
            divergence, valid_mask, config.rho_mask_lower_bound, config.rho_mask_upper_bound
        )
        metrics["val/rho_divergence"] = divergence.float()
    elif config.rho_mask_metric != "none":
        raise ValueError(f"Invalid rho mask metric: {config.rho_mask_metric}")

    if config.rho_mask_metric != "none" and config.rho_mask_direction == "increase_only":
        moving_away = torch.sign(mask_log_ratio) * advantages_f > 0
        dropped_low &= moving_away
        dropped_high &= moving_away

    correction_mask = valid_mask & ~dropped_low & ~dropped_high
    correction_weight = torch.ones_like(rho)
    rho_was_clamped = torch.zeros_like(valid_mask)
    if config.rollout_importance_correction == "clipped":
        correction_weight = rho
        if config.rho_clamp_lower_bound > 0.0:
            correction_weight = torch.clamp(correction_weight, min=config.rho_clamp_lower_bound)
        if config.rho_clamp_upper_bound > 0.0:
            correction_weight = torch.clamp(correction_weight, max=config.rho_clamp_upper_bound)
        rho_was_clamped = (correction_weight != rho) & valid_mask

    if config.loss_fn == GRPOLossType.dapo:
        algorithm_clipped = (
            ((advantages_f > 0) & (policy_ratio > 1.0 + config.clip_higher))
            | ((advantages_f < 0) & (policy_ratio < 1.0 - config.clip_lower))
        ) & correction_mask
        policy_weight = policy_ratio
        final_mask = correction_mask & ~algorithm_clipped
    elif config.loss_fn == GRPOLossType.cispo:
        algorithm_clipped = (policy_ratio > 1.0 + config.clip_higher) & correction_mask
        policy_weight = torch.clamp(policy_ratio, max=1.0 + config.clip_higher)
        final_mask = correction_mask
    elif config.loss_fn == GRPOLossType.dppo:
        algorithm_clipped = torch.zeros_like(valid_mask)
        policy_weight = policy_ratio
        final_mask = correction_mask
    else:
        raise ValueError(f"Invalid loss function: {config.loss_fn}")

    total_weight = policy_weight * correction_weight
    retained_overflow = final_mask & ~torch.isfinite(total_weight)
    retained_overflow_count = int(retained_overflow.sum().item())
    if retained_overflow_count:
        raise FloatingPointError(
            f"A policy-gradient ratio overflowed for {retained_overflow_count} retained response token(s)."
        )

    weights = torch.where(final_mask, total_weight, torch.zeros_like(total_weight))
    finite_rho_mask = valid_mask & rho_is_finite
    rho_hist = {"val/rho_hist": rho[finite_rho_mask]}
    metric_weight = correction_weight if config.policy_ratio_denominator == "old_policy" else policy_weight
    metrics |= {
        "val/rho_drop_frac": (dropped_low | dropped_high).float(),
        "val/rho_drop_low_frac": dropped_low.float(),
        "val/rho_drop_high_frac": dropped_high.float(),
        "val/rho_overflow_frac": (valid_mask & (~rho_is_finite | ~policy_ratio_is_finite)).float(),
        "val/rho_weight": torch.where(final_mask, metric_weight, torch.zeros_like(metric_weight)).float(),
        "val/rho_clipfrac": rho_was_clamped.float(),
    }
    return RhoCorrection(
        weights=weights,
        mask=final_mask,
        valid_mask=valid_mask,
        ratio=policy_ratio,
        rho=rho,
        clipfrac=algorithm_clipped.float(),
        metrics=metrics,
        histogram_metrics=rho_hist,
    )


def accumulate_rho_histograms(acc: dict[str, list[torch.Tensor]], correction: RhoCorrection) -> None:
    for key, values in correction.histogram_metrics.items():
        acc.setdefault(key, []).append(values.detach().cpu())


def finalize_rho_histograms(acc: dict[str, list[torch.Tensor]]) -> dict[str, np.ndarray]:
    finalized: dict[str, np.ndarray] = {}
    for key, chunks in acc.items():
        values = torch.cat(chunks)
        if values.numel() > 0:
            finalized[key] = values.numpy()
    return finalized


def resolve_old_logprobs(
    cache: list[torch.Tensor | None],
    sample_idx: int,
    epoch_idx: int,
    num_mini_batches: int,
    new_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed PPO denominator for one sample."""
    if num_mini_batches == 1 and epoch_idx == 0:
        cache[sample_idx] = new_logprobs.detach()
    old_logprobs = cache[sample_idx]
    if old_logprobs is None:
        raise RuntimeError(f"old logprobs were not initialized for sample {sample_idx}.")
    return old_logprobs


@dataclass
class GRPOLossOutput:
    """Per-token loss terms plus the intermediates the training loops log.

    ``pg_loss``, ``clipfrac``, and ``kl`` are per-token [B, T] tensors; the total loss is
    ``pg_loss + beta * kl`` reduced by ``masked_mean``. ``ratio`` is the
    configured policy ratio where representable and zero otherwise, ``rho``
    carries the detached coefficient and final update mask, and ``kl_mask`` is
    the exact structural mask applied to the reference-KL term.
    """

    pg_loss: torch.Tensor
    clipfrac: torch.Tensor
    kl: torch.Tensor
    ratio: torch.Tensor
    rho: RhoCorrection
    kl_mask: torch.Tensor


def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    vllm_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    response_mask: torch.Tensor,
    config: GRPOExperimentConfig,
    old_logprobs: torch.Tensor | None = None,
) -> GRPOLossOutput:
    """Compute ``-M · stopgrad(weight · advantage) · log π_θ`` plus optional reference KL.

    Every decision tensor is detached. ``torch.where`` removes excluded
    selected-token log probabilities from the autograd graph before either
    objective is evaluated, so a masked token receives exactly zero direct
    gradient even if its unmasked value is non-finite.
    """
    expected_shape = new_logprobs.shape
    for name, tensor in (
        ("vllm_logprobs", vllm_logprobs),
        ("advantages", advantages),
        ("response_mask", response_mask),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tensor.shape} must match new_logprobs shape {expected_shape}.")
    if ref_logprobs is not None and ref_logprobs.shape != expected_shape:
        raise ValueError(f"ref_logprobs shape {ref_logprobs.shape} must match new_logprobs shape {expected_shape}.")
    if old_logprobs is not None and old_logprobs.shape != expected_shape:
        raise ValueError(f"old_logprobs shape {old_logprobs.shape} must match new_logprobs shape {expected_shape}.")

    response_mask = response_mask.bool()
    invalid_new_logprobs = response_mask & (~torch.isfinite(new_logprobs.detach()) | (new_logprobs.detach() > 0))
    invalid_count = int(invalid_new_logprobs.sum().item())
    if invalid_count:
        raise FloatingPointError(
            f"new_logprobs contains {invalid_count} non-finite or positive value(s) at response positions."
        )

    rho = compute_rho_correction(
        vllm_logprobs=vllm_logprobs,
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        response_mask=response_mask,
        advantages=advantages,
        config=config,
    )
    safe_advantages = torch.where(
        torch.isfinite(advantages.detach()), advantages.detach(), torch.zeros_like(advantages.detach())
    )
    masked_new_logprobs = torch.where(rho.mask, new_logprobs, torch.zeros_like(new_logprobs))
    pg_loss = torch.where(
        rho.mask, -rho.weights * safe_advantages * masked_new_logprobs, torch.zeros_like(masked_new_logprobs)
    )

    finite_ratio_mask = rho.valid_mask & torch.isfinite(rho.ratio)
    ratio = torch.where(finite_ratio_mask, rho.ratio, torch.zeros_like(rho.ratio))

    if ref_logprobs is not None:
        detached_ref_logprobs = ref_logprobs.detach()
        kl_mask = response_mask & torch.isfinite(detached_ref_logprobs) & (detached_ref_logprobs <= 0)
        if config.mask_reference_kl_with_policy:
            kl_mask &= rho.mask
        if config.kl_estimator == 3:
            # Estimator 3 multiplies by π_θ/μ, so tokens whose direct ratio is
            # invalid or overflowed cannot safely contribute even when KL is
            # configured independently from the policy mask.
            kl_mask &= finite_ratio_mask
        kl_new_logprobs = torch.where(kl_mask, new_logprobs, torch.zeros_like(new_logprobs))
        safe_ref_logprobs = torch.where(kl_mask, detached_ref_logprobs, torch.zeros_like(detached_ref_logprobs))
        ref_logprobs_diff = (kl_new_logprobs.to(torch.float32) - safe_ref_logprobs.to(torch.float32)).clamp(
            -40.0, 40.0
        )
        kl_all = model_utils.estimate_kl(ref_logprobs_diff, ratio)
        kl = torch.where(kl_mask, kl_all[config.kl_estimator], torch.zeros_like(pg_loss))
    else:
        kl_mask = torch.zeros_like(rho.mask)
        kl = torch.zeros_like(pg_loss)

    return GRPOLossOutput(pg_loss=pg_loss, clipfrac=rho.clipfrac, kl=kl, ratio=ratio, rho=rho, kl_mask=kl_mask)


def log_policy_loss_configuration(config: GRPOExperimentConfig) -> None:
    """Log the configured policy-loss equations once at process startup."""
    denominator = "π_old" if config.policy_ratio_denominator == "old_policy" else "μ (rollout policy)"
    correction = "1"
    if config.rollout_importance_correction == "clipped":
        lower = config.rho_clamp_lower_bound
        upper = config.rho_clamp_upper_bound or "∞"
        correction = f"clip(π_old / μ, {lower}, {upper})"
    if config.rho_mask_metric == "none":
        mask = "disabled"
    else:
        source = "π_old" if config.rho_mask_source == "old_policy" else "π_θ"
        mask = (
            f"{config.rho_mask_metric}({source}, μ), {config.rho_mask_level}, "
            f"{config.rho_mask_direction}, bounds=({config.rho_mask_lower_bound}, "
            f"{config.rho_mask_upper_bound})"
        )
    logger.info(
        "Policy loss configuration:\n"
        f"  objective: {config.loss_fn}\n"
        f"  policy ratio: π_θ / {denominator}\n"
        f"  rollout correction: {correction}\n"
        f"  drop mask: {mask}\n"
        f"  reference KL mask: {'policy mask' if config.mask_reference_kl_with_policy else 'valid tokens'}"
    )


def forward_for_logprobs(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int,
    temperature: float,
    return_entropy: bool = False,
    pass_olmo_core_doc_lens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward pass to compute log probabilities."""
    extra_kwargs = {}
    if pass_olmo_core_doc_lens:
        assert attention_mask is not None
        doc_lens, max_doc_lens = olmo_core_utils.doc_lens_from_attention_mask(attention_mask)
        extra_kwargs = {"doc_lens": doc_lens, "max_doc_lens": max_doc_lens}
        attention_mask = None
    output = model(input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids, **extra_kwargs)
    logits = getattr(output, "logits", output)
    logits = logits / temperature
    # The logits at position i predict token i+1, so we align them with labels shifted by 1
    logits = logits[:, :-1]
    labels = query_responses[:, 1:].clone().to(logits.device)
    # Replace pad tokens with 0 to avoid index out of bounds errors in gather
    labels[labels == pad_token_id] = 0
    logprob_BT = model_utils.log_softmax_and_gather(logits, labels)

    # For now, entropy is just for monitoring, and we don't pass gradients through it.
    entropy = None
    if return_entropy:
        with torch.no_grad():
            entropy = model_utils.entropy_from_logits(logits)

    return logprob_BT, entropy


def compute_logprobs(
    model: torch.nn.Module,
    data_BT: data_types.CollatedBatchData,
    pad_token_id: int,
    temperature: float,
    use_grad: bool = False,
    batch_size: int | None = None,
    pass_olmo_core_doc_lens: bool = False,
) -> list[torch.Tensor]:
    """Compute log probabilities for all samples in batch."""
    logprobs_BT: list[torch.Tensor] = []
    num_samples = len(data_BT.query_responses)

    if batch_size is None:
        batch_size = 1

    context = torch.enable_grad() if use_grad else torch.no_grad()
    with context:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))

            query_responses = [data_BT.query_responses[i] for i in batch_indices]
            position_ids = [data_BT.position_ids[i] for i in batch_indices]
            shapes = [tuple(t.shape) for t in query_responses]

            if len(set(shapes)) != 1:
                for i in batch_indices:
                    single_logprobs, _ = forward_for_logprobs(
                        model,
                        data_BT.query_responses[i],
                        data_BT.attention_masks[i] if pass_olmo_core_doc_lens else None,
                        data_BT.position_ids[i],
                        pad_token_id,
                        temperature,
                        False,
                        pass_olmo_core_doc_lens=pass_olmo_core_doc_lens,
                    )

                    response_mask_BT = data_BT.response_masks[i]
                    single_logprobs = mask_logprobs(single_logprobs, response_mask_BT[:, 1:])
                    logprobs_BT.append(single_logprobs)
                continue

            batch_query_responses = torch.cat(query_responses, dim=0)
            batch_position_ids = torch.cat(position_ids, dim=0)
            batch_attention_mask = (
                torch.cat([data_BT.attention_masks[i] for i in batch_indices], dim=0)
                if pass_olmo_core_doc_lens
                else None
            )

            batch_logprobs, _ = forward_for_logprobs(
                model,
                batch_query_responses,
                batch_attention_mask,
                batch_position_ids,
                pad_token_id,
                temperature,
                False,
                pass_olmo_core_doc_lens=pass_olmo_core_doc_lens,
            )

            sample_sizes = [data_BT.query_responses[i].shape[0] for i in batch_indices]
            split_logprobs = torch.split(batch_logprobs, sample_sizes, dim=0)

            for i, logprob_BT in zip(batch_indices, split_logprobs):
                response_mask_BT = data_BT.response_masks[i]
                logprob_BT = mask_logprobs(logprob_BT, response_mask_BT[:, 1:])
                logprobs_BT.append(logprob_BT)

    return logprobs_BT


def calculate_token_counts(
    accumulation_steps: int,
    data_BT: data_types.CollatedBatchData,
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
) -> dict[int, float]:
    """Compute total token counts per accumulation group, all-reduced across DP ranks."""
    accumulation_counts: dict[int, float] = {}
    local_counts = [mask[:, 1:].sum().float() for mask in data_BT.response_masks]
    if not local_counts:
        return accumulation_counts

    counts_tensor = torch.stack(local_counts).to(device)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM, group=process_group)

    for i, count in enumerate(counts_tensor):
        group_idx = i // accumulation_steps
        key = int(group_idx * accumulation_steps)
        accumulation_counts[key] = accumulation_counts.get(key, 0.0) + count.item()

    return accumulation_counts


_SCALAR_LOSS_STAT_KEYS = [
    "loss/kl_avg",
    "loss/policy_avg",
    "loss/total_avg",
    "objective/kl0_avg",
    "objective/kl1_avg",
    "objective/kl2_avg",
    "objective/kl3_avg",
    "policy/clipfrac_avg",
    "val/ratio",
    "val/rho_clipfrac",
    "val/rho_divergence",
    "val/rho_weight",
    "val/rho_drop_frac",
    "val/rho_drop_low_frac",
    "val/rho_drop_high_frac",
    "val/rho_overflow_frac",
]


def create_loss_stats(num_samples: int, device: torch.device, record_entropy: bool = False) -> dict[str, torch.Tensor]:
    stats = {key: torch.zeros(num_samples, device=device) for key in _SCALAR_LOSS_STAT_KEYS}
    if record_entropy:
        stats |= {"policy/entropy_avg": torch.zeros(num_samples, device=device)}
    return stats


def populate_sample_loss_stats(
    loss_stats_B: dict[str, torch.Tensor],
    sample_idx: int,
    loss_output: GRPOLossOutput,
    loss: torch.Tensor,
    response_mask: torch.Tensor,
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropy: torch.Tensor | None,
    config: GRPOExperimentConfig,
) -> None:
    with torch.no_grad():
        valid_mask = loss_output.rho.valid_mask & response_mask.bool()
        if config.load_ref_policy and ref_logprobs is not None:
            safe_new_logprobs = torch.where(loss_output.kl_mask, new_logprobs, torch.zeros_like(new_logprobs))
            safe_ref_logprobs = torch.where(loss_output.kl_mask, ref_logprobs, torch.zeros_like(ref_logprobs))
            ref_logprobs_diff = (safe_new_logprobs - safe_ref_logprobs).clamp(-40.0, 40.0)
            kl_4BT = model_utils.estimate_kl(ref_logprobs_diff, loss_output.ratio)
            kl_4BT = torch.where(loss_output.kl_mask.unsqueeze(0), kl_4BT, torch.zeros_like(kl_4BT))
            kl_values = masked_mean(kl_4BT, response_mask).float()
            for j in range(4):
                loss_stats_B[f"objective/kl{j}_avg"][sample_idx] = kl_values[j]
            loss_stats_B["loss/kl_avg"][sample_idx] = kl_values[config.kl_estimator] * config.beta
        for key, value in loss_output.rho.metrics.items():
            loss_stats_B[key][sample_idx] = masked_mean(value, valid_mask)
        loss_stats_B["policy/clipfrac_avg"][sample_idx] = masked_mean(loss_output.clipfrac, valid_mask)
        loss_stats_B["loss/policy_avg"][sample_idx] = masked_mean(loss_output.pg_loss, valid_mask)
        loss_stats_B["loss/total_avg"][sample_idx] = loss
        finite_ratio_mask = valid_mask & torch.isfinite(loss_output.ratio)
        loss_stats_B["val/ratio"][sample_idx] = masked_mean(loss_output.ratio, finite_ratio_mask)
        if entropy is not None:
            entropy_mask = valid_mask & torch.isfinite(entropy)
            safe_entropy = torch.where(entropy_mask, entropy, torch.zeros_like(entropy))
            loss_stats_B["policy/entropy_avg"][sample_idx] = masked_mean(safe_entropy, entropy_mask).float()


def compute_metrics_from_loss_stats(
    loss_stats_B: dict[str, torch.Tensor], token_counts: torch.Tensor
) -> dict[str, float]:
    total_tokens = token_counts.sum()
    weights = token_counts / total_tokens if total_tokens > 0 else torch.zeros_like(token_counts)

    metrics: dict[str, float] = {}
    for key in loss_stats_B:
        metrics[key] = (loss_stats_B[key] * weights).sum().item()
    metrics["val/ratio_var"] = (weights * (loss_stats_B["val/ratio"] - metrics["val/ratio"]) ** 2).sum().item()
    return metrics


def perform_weight_sync(
    broadcast_refs: list[ray.ObjectRef],
    vllm_engines: list[ray.actor.ActorHandle],
    actor_manager: ray.actor.ActorHandle,
    *,
    progress: bool = False,
    inflight_updates: bool = False,
) -> tuple[dict[str, float], list]:
    """Pause actors, broadcast weights, await/skip inner engine RPCs, wake engines, resume actors.

    With `inflight_updates=False`, broadcast results are treated as
    list-of-lists of inner engine-update ObjectRefs which get flattened and
    awaited before waking. Pass `inflight_updates=True` to skip that inner
    await — either because `broadcast_refs` are already engine RPC refs, or
    because updates are intentionally left in flight.
    """
    start = time.perf_counter()
    ray.get(actor_manager.set_should_stop.remote(True))
    try:
        results, actor_sync_times = utils.ray_get_with_progress(
            broadcast_refs, desc="Broadcasting weights to vLLM engines", enable=progress
        )
        if not inflight_updates:
            utils.ray_get_with_progress(
                itertools.chain.from_iterable(results), desc="Waiting for vLLM engine update RPCs", enable=progress
            )
        utils.ray_get_with_progress(
            [e.wake_up.remote() for e in vllm_engines], desc="Waking up vLLM engines", enable=progress
        )
    finally:
        ray.get(actor_manager.set_should_stop.remote(False))
    sync_time_stats = {"time/weight_sync": time.perf_counter() - start}
    if actor_sync_times:
        sync_time_stats["time/weight_sync_mean"] = float(np.mean(actor_sync_times))
        sync_time_stats["time/weight_sync_min"] = float(np.min(actor_sync_times))
        sync_time_stats["time/weight_sync_max"] = float(np.max(actor_sync_times))
        sync_time_stats["time/weight_sync_median"] = float(np.median(actor_sync_times))
    return sync_time_stats, results


def maybe_evaluate(
    args: GRPOExperimentConfig,
    training_step: int,
    evaluation_inference_results_Q: ray_queue.Queue,
    tokenizer,
    episode,
    eval_dataset: Dataset,
    eval_generation_config,
    model_dims: utils.ModelDims,
    base_env_config: data_types.EnvConfig,
    max_possible_score: float,
    actor_manager=None,
) -> bool:
    """Optionally evaluate the model.

    Returns True if evaluation results were successfully collected, False otherwise.
    """
    if eval_dataset is None:
        return True

    try:
        is_final_step = training_step >= args.num_training_steps  # ty: ignore[unsupported-operator]
        num_eval_prompts = len(eval_dataset)
        if not is_final_step:
            queued_results = evaluation_inference_results_Q.qsize()
            if queued_results < num_eval_prompts:
                logger.info(
                    "[Main Thread] ⏳ Eval responses pending (%s/%s); deferring evaluation.",
                    queued_results,
                    num_eval_prompts,
                )
                return False

        timeout = 100 if is_final_step else 0.01

        eval_result, eval_batch, eval_reward_metrics, _ = data_loader_lib.accumulate_inference_batches(
            evaluation_inference_results_Q,
            eval_generation_config,
            num_prompts=num_eval_prompts,
            model_dims=model_dims,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            base_env_config=base_env_config,
            actor_manager=actor_manager,
            timeout=timeout,
            active_sampling=False,
            filter_zero_std_samples=False,
            replenish_prompts=False,
            max_possible_score=max_possible_score,
            training_step=training_step,
        )

        logger.info("[Main Thread] 📊 Evaluation responses received")

        eval_sequence_lengths = np.array([len(response) for response in eval_result.responses])
        eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_result.finish_reasons) / len(
            eval_result.finish_reasons
        )
        eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
        eval_pass_at_k_metrics: dict[str, float] = {}
        scores = np.array(eval_batch.scores)
        eval_k = eval_generation_config.n

        if scores.size and scores.size % eval_k == 0:
            scores_per_prompt = scores.reshape(-1, eval_k)
            correct_per_prompt = scores_per_prompt >= max_possible_score - 1e-8
            eval_pass_at_k_metrics.update(compute_pass_at_k_metrics(correct_per_prompt))
        else:
            logger.warning(
                "Eval scores size %s is not divisible by eval_k %s; skipping pass@k metrics.", scores.size, eval_k
            )
        eval_metrics: dict[str, Any] = {
            "eval/scores": scores.mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
            **eval_pass_at_k_metrics,
        }

        total_tokens = (
            eval_result.token_statistics.num_prompt_tokens + eval_result.token_statistics.num_response_tokens
        )
        eval_metrics["eval/actor_tokens_per_second"] = total_tokens / eval_result.token_statistics.generation_time

        model_utils.print_rich_single_line_metrics(eval_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(eval_batch.queries if eval_batch else [])
        table["response"] = eval_batch.decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]  # ty: ignore[not-iterable]
        table["scores"] = eval_batch.scores
        table["ground_truth"] = eval_batch.ground_truths if eval_batch else []
        if eval_batch.active_tools is not None:
            table["active_tools"] = [str(tools) if tools is not None else "all" for tools in eval_batch.active_tools]
        df = pd.DataFrame(table)

        if args.with_tracking:
            eval_metrics["sample_completions"] = wandb.Table(dataframe=df)
            wandb.log(eval_metrics, step=training_step)
        else:
            model_utils.print_rich_table(df.iloc[:1])
        del table
        return True
    except Empty:
        logger.warning("[Main Thread] 🙈 Evaluation responses not received")
        return False
