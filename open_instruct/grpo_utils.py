import enum
import itertools
import math
import os
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Literal, cast

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
    use_rho_correction: bool = True
    """Master switch for the train/infer ratio ρ = π^train_old / π^infer_old correction.
    When True, ρ is clamped to [rho_clamp_lower_bound, rho_clamp_upper_bound] and tokens
    whose ρ falls outside [rho_mask_lower_bound, rho_mask_upper_bound] have their
    per-token policy loss zeroed out. This unifies truncated importance sampling
    (https://fengyao.notion.site/off-policy-rl) and IcePop (https://arxiv.org/abs/2510.18855)."""
    rho_clamp_lower_bound: float = 0.0
    """Lower bound for clamping ρ before reweighting the policy loss (0 disables)."""
    rho_clamp_upper_bound: float = 2.0
    """Upper bound for clamping ρ before reweighting the policy loss (0 disables)."""
    rho_mask_lower_bound: float = 0.0
    """Tokens with ρ below this value are dropped (0 disables)."""
    rho_mask_upper_bound: float = 0.0
    """Tokens with ρ above this value are dropped (0 disables)."""
    rho_mask_sequence_level: bool = False
    """If True, apply the rho mask at the sequence level (DeepSeek-V3.2 style):
    compute the mean log-ratio (1/|o_i|) Σ_t log(π_old / π_θ) per response sequence,
    exponentiate to get a per-sequence ρ, and broadcast the keep/drop decision to every
    token in that sequence. If False (default), the mask is applied per token."""
    rho_mask_tv_divergence: bool = False
    """If True, applies the TV divergence masking from VACO (https://arxiv.org/abs/2603.01365)
    maintains same rho for correction but masks using bounds and sequence-level TV divergence abs(p_seq - 1)
    won't mask tokens that purport to decrease TV divergence: advantage * logprob_diff <= 0
    """
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
    loss_fn: GRPOLossType = GRPOLossType.dapo
    """Whether to use DAPO or CISPO loss function."""
    use_liger_grpo_loss: bool = False
    """Whether to use the tiled lm-head GRPO loss path, which avoids materializing the full
    vocabulary logits by recomputing the lm-head projection and loss tile-by-tile (DeepSpeed
    ``TiledFusedLogitsLoss`` pattern). High value for large-vocab / long-context memory.
    Supports ``loss_fn=dapo`` and ``loss_fn=cispo`` with the default KL estimator."""
    liger_grpo_loss_chunk_size: int = 8
    """Number of tiles (shards) to split the flattened tokens into when computing the tiled
    lm-head GRPO loss. Larger values reduce peak memory at the cost of more lm-head recomputes."""
    record_entropy: bool = False
    """whether to record the entropy of the policy during training. Uses extra memory."""
    use_vllm_logprobs: bool = False
    """whether to use vLLM's logprobs for training instead of calculating them via forward pass"""

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
        if self.use_vllm_logprobs and self.use_rho_correction:
            raise ValueError(
                "Cannot use both `use_vllm_logprobs` and `use_rho_correction`. "
                "use_vllm_logprobs sets old_logprobs to vLLM logprobs, making the ρ correction pointless."
            )
        if self.use_liger_grpo_loss:
            if self.loss_fn not in (GRPOLossType.dapo, GRPOLossType.cispo):
                raise ValueError("`use_liger_grpo_loss` currently only supports `loss_fn=dapo` or `loss_fn=cispo`.")
            if self.record_entropy:
                raise ValueError("`use_liger_grpo_loss` does not support `record_entropy=True`.")
            if self.kl_estimator != 2:
                raise ValueError("`use_liger_grpo_loss` uses the default KL estimator and requires `kl_estimator=2`.")
            if self.liger_grpo_loss_chunk_size < 1:
                raise ValueError(f"`liger_grpo_loss_chunk_size` must be >= 1 (got {self.liger_grpo_loss_chunk_size}).")
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
        if self.use_rho_correction:
            if self.rho_mask_lower_bound > 0.0 and not (0.0 < self.rho_mask_lower_bound < 1.0):
                raise ValueError(
                    f"rho_mask_lower_bound must satisfy 0 < lb < 1 when set, got {self.rho_mask_lower_bound}."
                )
            if self.rho_mask_upper_bound > 0.0 and self.rho_mask_upper_bound <= 1.0:
                raise ValueError(f"rho_mask_upper_bound must be > 1 when set, got {self.rho_mask_upper_bound}.")
            if self.rho_clamp_lower_bound > 0.0 and self.rho_clamp_lower_bound >= 1.0:
                raise ValueError(
                    f"rho_clamp_lower_bound must satisfy 0 < lb < 1 when set, got {self.rho_clamp_lower_bound}."
                )
            if self.rho_clamp_upper_bound > 0.0 and self.rho_clamp_upper_bound <= 1.0:
                raise ValueError(f"rho_clamp_upper_bound must be > 1 when set, got {self.rho_clamp_upper_bound}.")


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
        valid_mask = response_mask & ~torch.isnan(vllm_logprobs)
        valid_count = valid_mask.sum()
        diff = (local_logprobs - vllm_logprobs).abs()
        masked_diff = torch.masked_fill(diff, ~valid_mask, 0.0)
        mean_diff = masked_diff.sum() / valid_count if valid_count > 0 else torch.tensor(0.0)
        max_diff = masked_diff.max() if valid_count > 0 else torch.tensor(0.0)
        std_diff = masked_diff[valid_mask].std() if valid_count > 1 else torch.tensor(0.0)

        reverse_kl = torch.exp(vllm_logprobs) * (vllm_logprobs - local_logprobs)
        masked_reverse_kl = torch.masked_fill(reverse_kl, ~valid_mask, 0.0)
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
    """Per-token stop-gradient correction for the train/infer engine mismatch.

    ``weights`` is multiplied into the policy loss (all-ones disables the correction).
    ``metrics`` maps wandb keys to per-token tensors that get reduced by
    ``masked_mean(., response_mask)`` at logging time.
    ``histogram_metrics`` maps wandb keys to flat 1D tensors of values
    (response tokens only); these bypass the scalar reduction and are
    concatenated across micro-batches and logged as wandb histograms.
    """

    weights: torch.Tensor
    metrics: dict[str, torch.Tensor]
    histogram_metrics: dict[str, torch.Tensor] = field(default_factory=dict)


def compute_rho_correction(
    old_logprob: torch.Tensor,
    vllm_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    config: GRPOExperimentConfig,
) -> RhoCorrection:
    """Compute the unified ρ = π^train_old / π^infer_old correction (clamp + mask)."""
    logprob_diff = torch.where(
        response_mask, (old_logprob - vllm_logprobs).clamp(-10.0, 10.0), torch.zeros_like(old_logprob)
    )
    rho = torch.exp(logprob_diff)
    rho_hist = {"val/rho_hist": rho[response_mask].detach().float()}
    if not config.use_rho_correction:
        return RhoCorrection(weights=torch.ones_like(rho), metrics={}, histogram_metrics=rho_hist)

    rho_effective = (
        torch.exp(_sequence_level_mean(logprob_diff, response_mask)) if config.rho_mask_sequence_level else rho
    )

    if config.rho_mask_tv_divergence:
        # don't change rho_effective as it is our truncated importance sampling
        # calculate sequence-level TV divergence with abs(rho - 1)
        # filter if TV divergence > delta and advantage * logprob_diff > 0
        tv_divergence = torch.abs(rho - 1.0)
        tv_sequence_level = _sequence_level_mean(tv_divergence, response_mask)
        tv_dropped_low, tv_dropped_high = _rho_drop_masks(
            tv_sequence_level, response_mask, config.rho_mask_lower_bound, config.rho_mask_upper_bound
        )
        tokens_increase_tv = torch.sign(logprob_diff) * advantages > 0
        dropped_low = tv_dropped_low & tokens_increase_tv
        dropped_high = tv_dropped_high & tokens_increase_tv
    else:
        dropped_low, dropped_high = _rho_drop_masks(
            rho_effective, response_mask, config.rho_mask_lower_bound, config.rho_mask_upper_bound
        )

    in_range = response_mask & ~dropped_low & ~dropped_high

    rho_clamped = rho_effective
    if config.rho_clamp_lower_bound > 0.0:
        rho_clamped = torch.clamp(rho_clamped, min=config.rho_clamp_lower_bound)
    if config.rho_clamp_upper_bound > 0.0:
        rho_clamped = torch.clamp(rho_clamped, max=config.rho_clamp_upper_bound)

    weights = torch.where(in_range, rho_clamped, torch.zeros_like(rho_clamped))
    metrics = {
        "val/rho_drop_frac": (dropped_low | dropped_high).float(),
        "val/rho_drop_low_frac": dropped_low.float(),
        "val/rho_drop_high_frac": dropped_high.float(),
        "val/rho_weight": weights.float(),
        "val/rho_clipfrac": (rho_clamped != rho_effective).float(),
    }
    return RhoCorrection(weights=weights, metrics=metrics, histogram_metrics=rho_hist)


def accumulate_rho_histograms(acc: dict[str, list[torch.Tensor]], correction: RhoCorrection) -> None:
    for key, values in correction.histogram_metrics.items():
        acc.setdefault(key, []).append(values.detach().cpu())


def finalize_rho_histograms(acc: dict[str, list[torch.Tensor]]) -> dict[str, np.ndarray]:
    return {key: torch.cat(chunks).numpy() for key, chunks in acc.items()}


def resolve_old_logprob(
    old_logprobs_cache: list[torch.Tensor | None],
    sample_idx: int,
    epoch_idx: int,
    num_mini_batches: int,
    use_vllm_logprobs: bool,
    vllm_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Return the old (baseline) logprobs for a sample.

    With multiple mini-batches, old logprobs are pre-computed and cached.
    With a single mini-batch, they are lazily set on the first epoch from
    either vllm logprobs or the current policy's detached logprobs.
    """
    if num_mini_batches > 1:
        result = old_logprobs_cache[sample_idx]
    else:
        with torch.no_grad():
            if epoch_idx == 0:
                if use_vllm_logprobs:
                    old_logprobs_cache[sample_idx] = vllm_logprobs
                else:
                    old_logprobs_cache[sample_idx] = new_logprobs.detach()
            result = old_logprobs_cache[sample_idx]
    assert result is not None
    return result


def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    config: GRPOExperimentConfig,
    rho_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.loss_fn == GRPOLossType.dapo:
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_lower, 1.0 + config.clip_higher)
        clipfrac = (pg_losses2 > pg_losses).float()
        pg_loss = torch.max(pg_losses, pg_losses2)
    elif config.loss_fn == GRPOLossType.cispo:
        # cispo: directly clip ratio, no lower bound.
        # reinforce loss, so multiply by new logprobs
        clipfrac = (ratio > 1.0 + config.clip_higher).float()
        pg_loss = -advantages * torch.clamp(ratio.detach(), max=1.0 + config.clip_higher) * new_logprobs
    else:
        raise ValueError(f"Invalid loss function: {config.loss_fn}")

    pg_loss *= rho_weights
    clipfrac *= (rho_weights != 0).float()

    if ref_logprobs is not None:
        # We want the KL loss to backpropagate through the model.
        # We also clamp the KL loss to avoid numerical instability.
        # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
        ref_logprobs_diff = (new_logprobs - ref_logprobs).clamp(-40.0, 40.0)
        kl_all = model_utils.estimate_kl(ref_logprobs_diff, ratio)
        kl = kl_all[config.kl_estimator]
    else:
        kl = torch.zeros_like(pg_loss)

    return pg_loss, clipfrac, kl


class TiledGRPOLMHeadLoss(torch.autograd.Function):
    """Tiled DAPO/CISPO lm-head loss that avoids materializing full-vocabulary logits.

    This follows DeepSpeed's ``TiledFusedLogitsLoss`` pattern: the lm-head
    projection and scalar loss are recomputed per tile in ``forward`` and
    ``torch.autograd.backward`` is called per tile to accumulate lm-head grads.
    The custom ``backward`` then returns the precomputed hidden-state gradient
    so the outer DeepSpeed backward only traverses the backbone.

    The scalar loss it returns matches ``masked_mean(pg + beta * kl, mask, None,
    loss_denominator) * loss_scale`` from the non-tiled path, so a run can switch
    the flag on/off without changing the objective.
    """

    @staticmethod
    def forward(
        ctx,
        lm_head: torch.nn.Module,
        hidden_states: torch.Tensor,
        selected_token_ids: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        has_ref_logprobs: bool,
        temperature: float,
        beta: float,
        clip_lower: float,
        clip_higher: float,
        loss_fn: str,
        shards: int,
        loss_scale: torch.Tensor,
        loss_denom: torch.Tensor,
        compute_params: list[torch.nn.Parameter],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
        if selected_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("selected_token_ids must match hidden_states batch/sequence dimensions")
        if response_mask.shape != hidden_states.shape[:2]:
            raise ValueError("response_mask must match hidden_states batch/sequence dimensions")
        if shards < 1:
            raise ValueError(f"shards must be >= 1, got {shards}")
        loss_type = GRPOLossType(loss_fn)
        if loss_type not in (GRPOLossType.dapo, GRPOLossType.cispo):
            raise ValueError(f"`tiled_grpo_lm_head_loss` does not support loss_fn={loss_type}.")

        x_requires_grad = hidden_states.requires_grad
        batch_size, seq_len, hidden_size = hidden_states.shape
        x = hidden_states.detach().reshape(-1, hidden_size)
        x.requires_grad_(x_requires_grad)

        labels = selected_token_ids.reshape(-1)
        mask_2d = response_mask.to(dtype=torch.bool)
        mask = mask_2d.reshape(-1)
        advantages = advantages.reshape(-1)
        old_logprobs = old_logprobs.reshape(-1)
        if has_ref_logprobs:
            ref_logprobs = ref_logprobs.reshape(-1)

        num_tokens = x.shape[0]
        shards = min(shards, num_tokens)
        loss_denom = loss_denom.to(dtype=torch.float32).clamp_min(1.0)
        incoming_grad = (loss_scale.detach().to(dtype=torch.float32) / loss_denom).reshape(())

        x_grad = torch.zeros_like(x) if x_requires_grad else None
        x_shards = list(torch.chunk(x, chunks=shards, dim=0))
        label_shards = list(torch.chunk(labels, chunks=shards, dim=0))
        mask_shards = list(torch.chunk(mask, chunks=shards, dim=0))
        advantage_shards = list(torch.chunk(advantages, chunks=shards, dim=0))
        old_logprob_shards = list(torch.chunk(old_logprobs, chunks=shards, dim=0))
        ref_logprob_shards = list(torch.chunk(ref_logprobs, chunks=shards, dim=0)) if has_ref_logprobs else []

        total_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
        total_pg_sum = torch.zeros_like(total_loss_sum)
        total_kl_sum = torch.zeros(4, dtype=torch.float32, device=hidden_states.device)
        total_clip_sum = torch.zeros_like(total_loss_sum)
        total_ratio_sum = torch.zeros_like(total_loss_sum)
        metric_denom = mask_2d.to(dtype=torch.float32).sum().clamp_min(1.0)
        compute_params = [p for p in compute_params if p.requires_grad]

        for shard_idx, x_shard in enumerate(x_shards):
            if compute_params:
                # ZeRO-3 reduces a param's grad once ds_grad_is_ready flips True; only let the
                # last shard trigger the reduction so all tiles accumulate into the same buffer.
                grad_is_ready = shard_idx + 1 == len(x_shards)
                for param in compute_params:
                    # ds_grad_is_ready is injected by DeepSpeed ZeRO-3 at runtime.
                    param.ds_grad_is_ready = grad_is_ready  # ty: ignore[unresolved-attribute]

            shard_step = x_shard.shape[0]
            shard_offset = shard_idx * x_shards[0].shape[0]
            x_shard.requires_grad_(x_requires_grad)
            if x_grad is not None:
                x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            with torch.enable_grad():
                logits = lm_head(x_shard)
                if temperature != 1.0:
                    logits = logits / temperature
                shard_labels = label_shards[shard_idx]
                new_logprobs = torch.gather(logits, dim=-1, index=shard_labels.unsqueeze(-1)).squeeze(-1)
                new_logprobs = new_logprobs - torch.logsumexp(logits, dim=-1)

                ratio = torch.exp(new_logprobs - old_logprob_shards[shard_idx])
                if loss_type == GRPOLossType.dapo:
                    pg_losses = -advantage_shards[shard_idx] * ratio
                    pg_losses2 = -advantage_shards[shard_idx] * torch.clamp(ratio, 1.0 - clip_lower, 1.0 + clip_higher)
                    pg_loss = torch.max(pg_losses, pg_losses2)
                    shard_clip = (pg_losses2 > pg_losses).detach().float()
                else:  # cispo
                    clipped_ratio = torch.clamp(ratio.detach(), max=1.0 + clip_higher)
                    pg_loss = -advantage_shards[shard_idx] * clipped_ratio * new_logprobs
                    shard_clip = (ratio > 1.0 + clip_higher).detach().float()

                if has_ref_logprobs:
                    ref_diff = (new_logprobs - ref_logprob_shards[shard_idx]).clamp(-40.0, 40.0)
                    kl_all = model_utils.estimate_kl(ref_diff, ratio)
                    kl = kl_all[2]
                else:
                    kl = torch.zeros_like(pg_loss)

                shard_mask = mask_shards[shard_idx].to(dtype=pg_loss.dtype)
                per_token_loss = pg_loss + beta * kl
                loss_sum = (per_token_loss * shard_mask).sum()

            total_loss_sum = total_loss_sum + loss_sum.detach().float()
            total_pg_sum = total_pg_sum + (pg_loss.detach().float() * shard_mask.float()).sum()
            if has_ref_logprobs:
                total_kl_sum = total_kl_sum + (kl_all.detach().float() * shard_mask.float()).sum(dim=-1)
            total_clip_sum = total_clip_sum + (shard_clip * shard_mask.float()).sum()
            total_ratio_sum = total_ratio_sum + (ratio.detach().float() * shard_mask.float()).sum()
            torch.autograd.backward(loss_sum, incoming_grad.to(dtype=loss_sum.dtype))

        if compute_params:
            for param in compute_params:
                param.ds_grad_is_ready = True  # ty: ignore[unresolved-attribute]

        if x_grad is None:
            x_grad = torch.zeros_like(x)
        ctx.save_for_backward(x_grad.reshape(batch_size, seq_len, hidden_size).detach())

        loss = total_loss_sum / loss_denom * loss_scale.detach().to(dtype=total_loss_sum.dtype)
        pg_avg = total_pg_sum / metric_denom
        kl_avg = total_kl_sum / metric_denom
        clipfrac = total_clip_sum / metric_denom
        ratio_avg = total_ratio_sum / metric_denom
        return loss, pg_avg, kl_avg, clipfrac, ratio_avg

    @staticmethod
    def backward(ctx, *grads) -> tuple:
        (x_grad,) = ctx.saved_tensors
        grad = grads[0]
        if isinstance(grad, torch.Tensor):
            x_grad = x_grad * grad.to(dtype=x_grad.dtype)
        return (None, x_grad, *([None] * 15))


def tiled_grpo_lm_head_loss(
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    selected_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    temperature: float,
    beta: float,
    clip_lower: float,
    clip_higher: float,
    shards: int,
    loss_scale: torch.Tensor,
    loss_denom: torch.Tensor,
    loss_fn: str | GRPOLossType = GRPOLossType.dapo,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-efficient tiled GRPO lm-head loss. Returns ``(loss, pg_avg, kl_avg, clipfrac, ratio_avg)``."""
    has_ref_logprobs = ref_logprobs is not None
    if ref_logprobs is None:
        ref_logprobs = torch.empty(0, dtype=old_logprobs.dtype, device=old_logprobs.device)
    compute_params = list(lm_head.parameters(recurse=False))
    return TiledGRPOLMHeadLoss.apply(
        lm_head,
        hidden_states,
        selected_token_ids,
        response_mask,
        advantages,
        old_logprobs,
        ref_logprobs,
        has_ref_logprobs,
        temperature,
        beta,
        clip_lower,
        clip_higher,
        loss_fn,
        shards,
        loss_scale,
        loss_denom,
        compute_params,
    )


def _unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a (possibly DeepSpeed-wrapped) model down to the HF causal-LM module."""
    inner = model
    seen: set[int] = set()
    while hasattr(inner, "module") and id(inner) not in seen:
        seen.add(id(inner))
        inner = inner.module
    return cast(torch.nn.Module, inner)


def get_causal_lm_backbone_and_lm_head(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    causal_lm = _unwrap_causal_lm(model)
    base_model_prefix = getattr(causal_lm, "base_model_prefix", None)
    if isinstance(base_model_prefix, str) and hasattr(causal_lm, base_model_prefix):
        backbone = getattr(causal_lm, base_model_prefix)
    elif hasattr(causal_lm, "model"):
        backbone = causal_lm.model
    elif hasattr(causal_lm, "base_model"):
        backbone = causal_lm.base_model
    else:
        raise AttributeError(f"Could not find causal LM backbone for {type(causal_lm).__name__}.")
    lm_head = getattr(causal_lm, "lm_head", None)
    if lm_head is None:
        raise AttributeError(f"Could not find lm_head for {type(causal_lm).__name__}.")
    return cast(torch.nn.Module, backbone), cast(torch.nn.Module, lm_head)


def forward_for_liger_hidden_states(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Run only the causal-LM backbone (no lm-head) and return the shifted hidden states.

    ``attention_mask=None`` lets HF build the correct 3D intra-document mask from
    ``position_ids`` for packed sequences, matching ``forward_for_logprobs``.
    """
    backbone, _ = get_causal_lm_backbone_and_lm_head(model)
    output = backbone(input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids)
    hidden_states = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    return hidden_states[:, :-1]


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
    "val/rho_weight",
    "val/rho_drop_frac",
    "val/rho_drop_low_frac",
    "val/rho_drop_high_frac",
]


def create_loss_stats(num_samples: int, device: torch.device, record_entropy: bool = False) -> dict[str, torch.Tensor]:
    stats = {key: torch.zeros(num_samples, device=device) for key in _SCALAR_LOSS_STAT_KEYS}
    if record_entropy:
        stats |= {"policy/entropy_avg": torch.zeros(num_samples, device=device)}
    return stats


def populate_sample_loss_stats(
    loss_stats_B: dict[str, torch.Tensor],
    sample_idx: int,
    pg_loss: torch.Tensor,
    clipfrac: torch.Tensor,
    ratio: torch.Tensor,
    loss: torch.Tensor,
    response_mask: torch.Tensor,
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropy: torch.Tensor | None,
    config: GRPOExperimentConfig,
    rho_metrics: dict[str, torch.Tensor] | None = None,
) -> None:
    with torch.no_grad():
        if config.load_ref_policy and ref_logprobs is not None:
            ref_logprobs_diff = (new_logprobs - ref_logprobs).clamp(-40.0, 40.0)
            kl_4BT = model_utils.estimate_kl(ref_logprobs_diff, ratio)
            kl_values = masked_mean(kl_4BT, response_mask).float()
            for j in range(4):
                loss_stats_B[f"objective/kl{j}_avg"][sample_idx] = kl_values[j]
            loss_stats_B["loss/kl_avg"][sample_idx] = kl_values[config.kl_estimator] * config.beta
        if rho_metrics is not None:
            for key, value in rho_metrics.items():
                loss_stats_B[key][sample_idx] = masked_mean(value, response_mask)
        loss_stats_B["policy/clipfrac_avg"][sample_idx] = masked_mean(clipfrac, response_mask)
        loss_stats_B["loss/policy_avg"][sample_idx] = masked_mean(pg_loss, response_mask)
        loss_stats_B["loss/total_avg"][sample_idx] = loss
        loss_stats_B["val/ratio"][sample_idx] = masked_mean(ratio, response_mask)
        if entropy is not None:
            loss_stats_B["policy/entropy_avg"][sample_idx] = masked_mean(entropy, response_mask).float()


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
