import enum
import importlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist

from open_instruct import data_types, logger_utils, model_utils, olmo_core_utils
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
    tvpo = "tvpo"


class DPPODivergenceType(enum.StrEnum):
    tv = "tv"
    kl = "kl"


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
    lm_head_fp32: bool = False
    """Whether to keep the final LM head projection in fp32 for both HF training and vLLM rollout models."""

    # Algorithm
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    clip_lower: float = 0.2
    """the lower clip range"""
    clip_higher: float = 0.272
    """the higher clip range. Sometimes we want this to be higher, see DAPO (https://arxiv.org/abs/2503.14476)"""
    truncated_importance_sampling_ratio_cap: float = 2.0
    """The maximum cap for truncated importance sampling ratio (0 means disabled)"""
    tis_mask_lower: float = 0.0
    """Absolute lower bound for the trust-region mask on the π_θ/π_rollout ratio.

    When >0, tokens with ratio ≤ tis_mask_lower are multiplied by 0 in the pg loss.
    Set to 0 to disable the lower side of the mask.
    """
    tis_mask_upper: float = 0.0
    """Absolute upper bound for the trust-region mask on the π_θ/π_rollout ratio.

    When >0, tokens with ratio ≥ tis_mask_upper are multiplied by 0 in the pg loss.
    Set to 0 to disable the upper side of the mask.
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
    """Which policy-loss function to use: 'dapo', 'cispo', 'dppo'
    (https://arxiv.org/abs/2602.04879), or 'tvpo' (prompt-level TV trust region)."""
    dppo_divergence_type: DPPODivergenceType = DPPODivergenceType.tv
    """For DPPO: which divergence to use to define the trust region ('tv' or 'kl').

    Uses the binary (Bernoulli {sampled_token, all_others}) approximation from
    Eqs. 13/14 of the DPPO paper (https://arxiv.org/abs/2602.04879). Only used
    when ``loss_fn=dppo``.
    """
    dppo_divergence_threshold: float = 0.1
    """For DPPO: the trust-region threshold δ on the binary divergence.

    Tokens whose update would push them outside the trust region (per Eq. 12)
    are masked out of the policy gradient. Only used when ``loss_fn=dppo``.

    The DPPO paper (https://arxiv.org/abs/2602.04879) uses δ=0.15 (Sec 5) and
    δ=0.2 (scaling experiments, Appendix F) for binary TV, and δ=0.05 for
    binary KL. We default to 0.1 as a moderately tighter middle-ground; bump
    closer to 0.15–0.2 for paper-faithful runs.
    """
    tvpo_divergence_threshold: float = 0.02
    """For TVPO: prompt-level total-variation trust-region radius δ.

    A prompt is considered in budget when ``(1/2) * E_t |π_θ/μ_θ' − 1| ≤ δ``,
    where the inner expectation is the rollout-uniform mean across all rollouts
    of that prompt that appear in the same microbatch. When *every* prompt in
    the microbatch is in budget the mask is all-ones (full REINFORCE-IS step).
    Otherwise blocked tokens have their loss replaced by its detached value so
    no gradient flows but the loss value is preserved for logging. Only used
    when ``loss_fn=tvpo``.
    """
    tvpo_truncation_cap: float = 20.0
    """For TVPO: importance-weight truncation cap c used in the surrogate
    ``-A · clamp(r, max=c).detach() · log π_θ`` (REINFORCE-with-IS form).

    A large cap (default 20) is recommended because the trust region is enforced
    by the TVPO mask, not by this cap; the cap only protects against numerical
    blow-ups in extreme ratios. Only used when ``loss_fn=tvpo``.
    """
    record_entropy: bool = False
    """whether to record the entropy of the policy during training. Uses extra memory."""
    use_vllm_logprobs: bool = False
    """whether to use vLLM's logprobs for training instead of calculating them via forward pass"""
    use_liger_grpo_loss: bool = False
    """Whether to use Liger-Kernel's fused linear GRPO loss for the policy loss."""
    liger_grpo_loss_chunk_size: int = 2048
    """Batch chunk size passed to LigerFusedLinearGRPOLoss."""
    liger_grpo_loss_compiled: bool = False
    """Whether Liger should torch.compile the GRPO loss math. Disabled by default for Ray/ZeRO dynamic shapes."""

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

    def __post_init__(self):
        if self.send_slack_alerts and not os.environ.get("SLACK_WEBHOOK_URL"):
            logger.warning(
                "--send_slack_alerts is set but SLACK_WEBHOOK_URL is not in the environment. Slack alerts will not be sent."
            )
        if self.use_vllm_logprobs and self.truncated_importance_sampling_ratio_cap > 0.0:
            raise ValueError(
                "Cannot use both `use_vllm_logprobs` and `truncated_importance_sampling_ratio_cap`. "
                "use_vllm_logprobs sets old_logprobs to vLLM logprobs, making importance sampling pointless."
            )
        if self.loss_fn == GRPOLossType.dppo:
            if self.dppo_divergence_threshold <= 0.0:
                raise ValueError(
                    f"DPPO requires `dppo_divergence_threshold` > 0 (got {self.dppo_divergence_threshold})."
                )
            # DPPO's trust region must be anchored on the rollout policy μ_θ'
            # (Takeaway 2 in arXiv:2602.04879). Forcing `use_vllm_logprobs=True`
            # ensures the cached `old_logprob` (and therefore the importance
            # ratio used in the loss) is exactly μ_θ', without needing a
            # separate code path.
            if not self.use_vllm_logprobs:
                raise ValueError(
                    "DPPO requires `use_vllm_logprobs=True` so the importance ratio is "
                    "computed against the rollout policy μ_θ' (Takeaway 2 in arXiv:2602.04879). "
                    "Pass `--use_vllm_logprobs True --truncated_importance_sampling_ratio_cap 0` "
                    "alongside `--loss_fn dppo`."
                )
        if self.loss_fn == GRPOLossType.tvpo:
            if self.tvpo_divergence_threshold <= 0.0:
                raise ValueError(
                    f"TVPO requires `tvpo_divergence_threshold` > 0 (got {self.tvpo_divergence_threshold})."
                )
            if self.tvpo_truncation_cap <= 0.0:
                raise ValueError(f"TVPO requires `tvpo_truncation_cap` > 0 (got {self.tvpo_truncation_cap}).")
            # Like DPPO, TVPO's trust region is anchored on the rollout policy
            # μ_θ', so the cached `old_logprob` must be the vLLM logprobs.
            if not self.use_vllm_logprobs:
                raise ValueError(
                    "TVPO requires `use_vllm_logprobs=True` so the importance ratio is "
                    "computed against the rollout policy μ_θ'. "
                    "Pass `--use_vllm_logprobs True --truncated_importance_sampling_ratio_cap 0` "
                    "alongside `--loss_fn tvpo`."
                )
        if self.tis_mask_lower < 0.0 or self.tis_mask_upper < 0.0:
            raise ValueError(
                f"tis_mask_lower and tis_mask_upper must be ≥ 0 "
                f"(got {self.tis_mask_lower=}, {self.tis_mask_upper=}). Use 0 to disable."
            )
        if self.tis_mask_lower > 0.0 and self.tis_mask_upper > 0.0 and self.tis_mask_lower >= self.tis_mask_upper:
            raise ValueError(
                "tis_mask_lower must be less than tis_mask_upper when both mask bounds are enabled, "
                f"got {self.tis_mask_lower=} and {self.tis_mask_upper=}."
            )
        if self.loss_denominator != "token" and float(self.loss_denominator) <= 0:
            raise ValueError(
                f"loss_denominator must be a valid float greater than 0 if not 'token', got: {self.loss_denominator}"
            )
        if self.use_liger_grpo_loss:
            if self.loss_fn not in (GRPOLossType.dapo, GRPOLossType.cispo):
                raise ValueError(
                    "Liger GRPO loss currently only supports `loss_fn` values dapo and cispo in open-instruct "
                    f"(got {self.loss_fn})."
                )
            if self.load_ref_policy and self.beta != 0.0 and self.kl_estimator != 2:
                raise ValueError(
                    "Liger GRPO loss computes the k2/k3-style KL estimator used by `kl_estimator=2`; "
                    f"got kl_estimator={self.kl_estimator}."
                )
            if self.record_entropy:
                raise ValueError("Liger GRPO loss does not support `record_entropy`.")
            if self.liger_grpo_loss_chunk_size <= 0:
                raise ValueError(f"liger_grpo_loss_chunk_size must be > 0, got {self.liger_grpo_loss_chunk_size}.")
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq == -1:
            raise ValueError("`checkpoint_state_freq` must be greater than 0 if `checkpoint_state_dir` is provided!")

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


def mask_logprobs(vllm_logprobs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Set non-response positions to INVALID_LOGPROB and replace NaNs."""
    vllm_logprobs = torch.masked_fill(vllm_logprobs, ~response_mask, INVALID_LOGPROB)
    vllm_logprobs = torch.nan_to_num(vllm_logprobs, nan=INVALID_LOGPROB)
    return vllm_logprobs


def compute_tis_weights(
    old_logprob: torch.Tensor, vllm_logprobs: torch.Tensor, response_mask: torch.Tensor, cap: float
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute truncated importance sampling weights: clamp(π_old / π_vllm, max=cap).

    Returns (clamped, unclamped) tuple, both None when cap <= 0 (disabled).
    """
    if cap <= 0:
        return None, None
    unclamped = torch.ones_like(old_logprob)
    logprob_diff = old_logprob - vllm_logprobs
    logprob_diff = torch.where(response_mask, logprob_diff.clamp(-10.0, 10.0), torch.zeros_like(logprob_diff))
    unclamped = torch.where(response_mask, torch.exp(logprob_diff), unclamped)
    clamped = torch.clamp(unclamped, max=cap)
    return clamped, unclamped


def combine_tis_terms(*terms: torch.Tensor | None) -> torch.Tensor | None:
    """Element-wise product of non-None TIS tensors (weights and/or masks).

    Returns ``None`` if every term is ``None``. Used to combine the numeric-mismatch
    cap from :func:`compute_tis_weights` with the trust-region mask from
    :func:`compute_tis_mask` into a single ``tis_weights`` argument.
    """
    result: torch.Tensor | None = None
    for term in terms:
        if term is None:
            continue
        result = term if result is None else result * term
    return result


def compute_tis_mask(
    new_logprobs: torch.Tensor,
    vllm_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    lower_bound: float,
    upper_bound: float,
) -> torch.Tensor | None:
    """Binary {0, 1} trust-region gate on the π_θ/π_rollout ratio.

    Implements a two-sided absolute ratio gate: lower_bound < x < upper_bound. The ratio
    is computed from the current trainer logprobs and the rollout (vLLM) logprobs directly,
    independent of any cached π_old, so the gate matches the paper's r_t even under
    multi-epoch / cached old_logprob setups.

    Returns a float tensor the same shape as ``new_logprobs`` with 1.0 on in-range response
    positions and 0.0 elsewhere, or ``None`` when both bounds are disabled. Non-response
    positions and positions with NaN vllm logprobs are also set to 0.
    """
    if lower_bound <= 0.0 and upper_bound <= 0.0:
        return None
    with torch.no_grad():
        logprob_diff = (new_logprobs - vllm_logprobs).clamp(-10.0, 10.0)
        ratio = torch.exp(logprob_diff)
        lower = lower_bound if lower_bound > 0.0 else float("-inf")
        upper = upper_bound if upper_bound > 0.0 else float("inf")
        valid = response_mask & ~torch.isnan(vllm_logprobs)
        in_range = (ratio > lower) & (ratio < upper)
        return (valid & in_range).to(new_logprobs.dtype)


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


def compute_binary_divergence(
    behavior_logprobs: torch.Tensor, policy_logprobs: torch.Tensor, response_mask: torch.Tensor, divergence_type: str
) -> torch.Tensor:
    """Per-token binary (Bernoulli) divergence between behavior and policy.

    Implements the binary approximation from Eqs. 13/14 of the DPPO paper
    (https://arxiv.org/abs/2602.04879): collapse the categorical distribution
    over the vocabulary into a Bernoulli over ``{sampled_token, all_others}``
    using only the per-token logprobs. This is a memory-cheap lower bound on
    the true policy divergence that requires no extra forward passes.

    Args:
        behavior_logprobs: log μ(a_t|s_t), the rollout (vLLM) policy.
        policy_logprobs:   log π(a_t|s_t), the current trainer policy.
        response_mask:     bool mask selecting valid response positions.
        divergence_type:   ``"tv"`` for total variation or ``"kl"`` for KL.

    Returns:
        Float tensor of the same shape as ``policy_logprobs``; entries outside
        ``response_mask`` are zeroed.
    """
    eps = 1e-9
    # Real logprobs are <= 0; non-response sentinel positions can be > 0
    # (see ``mask_logprobs`` / INVALID_LOGPROB). Clamp so exp() stays in [eps, 1].
    mu = torch.exp(behavior_logprobs.clamp(min=-30.0, max=0.0))
    pi = torch.exp(policy_logprobs.clamp(min=-30.0, max=0.0))
    if divergence_type == DPPODivergenceType.tv:
        divergence = (mu - pi).abs()
    elif divergence_type == DPPODivergenceType.kl:
        mu_clip = mu.clamp(eps, 1.0 - eps)
        pi_clip = pi.clamp(eps, 1.0 - eps)
        divergence = mu_clip * (mu_clip.log() - pi_clip.log()) + (1.0 - mu_clip) * (
            (1.0 - mu_clip).log() - (1.0 - pi_clip).log()
        )
    else:
        raise ValueError(
            f"Unknown DPPO divergence type: {divergence_type}. Expected one of {list(DPPODivergenceType)}."
        )
    return torch.where(response_mask, divergence, torch.zeros_like(divergence))


def compute_dppo_mask(
    new_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    divergence_type: str,
    divergence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the DPPO trust-region mask M_t (Eq. 12).

    The mask zeros out updates that would both push the policy further away from
    the rollout (``r_t > 1`` for positive advantage, or ``r_t < 1`` for negative
    advantage) AND have already exceeded the divergence threshold ``δ``.
    Updates that move the ratio back towards 1 are never masked, preserving
    PPO's beneficial asymmetric structure.

    Args:
        new_logprobs:        log π_θ at the sampled tokens.
        behavior_logprobs:   log μ_θ' at the sampled tokens (from the rollout).
        advantages:          per-token advantages, same shape.
        ratio:               π_θ(y_t|s_t) / μ_θ'(y_t|s_t).
        response_mask:       bool mask selecting valid response positions.
        divergence_type:     ``"tv"`` or ``"kl"`` (passed to
            :func:`compute_binary_divergence`).
        divergence_threshold: scalar trust-region radius δ.

    Returns:
        ``(mask, divergence)`` where ``mask`` is a 0/1 float tensor and
        ``divergence`` is the per-token binary divergence (for logging).
    """
    with torch.no_grad():
        divergence = compute_binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=new_logprobs,
            response_mask=response_mask,
            divergence_type=divergence_type,
        )
        outside_region = divergence > divergence_threshold
        bad_high = (advantages > 0) & (ratio > 1.0) & outside_region
        bad_low = (advantages < 0) & (ratio < 1.0) & outside_region
        bad = bad_high | bad_low
        mask = (~bad & response_mask).to(new_logprobs.dtype)
    return mask, divergence


def compute_tvpo_mask(
    new_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    response_mask: torch.Tensor,
    divergence_threshold: float,
    rollout_ids: torch.Tensor | None = None,
    num_samples_per_prompt: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the TVPO trust-region mask.

    TVPO enforces a *prompt-level* trust region using the standard total-variation
    upper bound from importance sampling: ``TV(π_θ, μ_θ') ≤ (1/2) E_t [|π_θ/μ_θ' − 1|]``.
    For every prompt represented in the microbatch we average ``|r − 1| / 2`` over
    its rollout's response tokens, then average those per-rollout TVs (uniform per
    rollout) into one per-prompt TV.

    The mask is constructed in two stages:

    1. **Hierarchical short-circuit.** If every prompt's TV is at or below the
       threshold ``δ``, the mask is all-ones for response positions — TVPO
       behaves as a plain truncated-IS REINFORCE step.
    2. **Directional release valve.** If at least one prompt is over budget,
       a token is *kept* iff ``(prompt_tv ≤ δ) OR (A · sign(π − μ) ≤ 0)``. The
       second clause keeps tokens whose gradient direction would *reduce* the
       prompt's TV (push the policy back towards μ_θ'), even when over budget.

    The caller is responsible for using the mask via the ``policy_freeze_mask``
    argument of :func:`compute_grpo_loss`, which substitutes the detached loss
    value for blocked tokens (zero gradient, value preserved for logging).

    Args:
        new_logprobs:        log π_θ at the sampled tokens, shape ``[B, T]``.
        behavior_logprobs:   log μ_θ' at the sampled tokens (from the rollout),
            shape ``[B, T]``.
        advantages:          per-token advantages, shape ``[B, T]``.
        ratio:               π_θ / μ_θ' at the sampled tokens, shape ``[B, T]``.
        response_mask:       bool mask selecting valid response positions.
        divergence_threshold: scalar trust-region radius δ on TV.
        rollout_ids:         optional ``[B, T]`` int tensor giving the rollout id
            of each token. When ``None``, the entire microbatch is treated as a
            single prompt with a single rollout (no group aggregation), matching
            the per-sample fallback in the reference implementation.
        num_samples_per_prompt: number of rollouts per prompt. Prompt id is
            derived as ``rollout_id // num_samples_per_prompt``, matching the
            consecutive-grouped layout produced by the rollout pipeline.

    Returns:
        ``(mask, prompt_tv_per_token)`` where ``mask`` is a 0/1 float tensor of
        the same shape as ``new_logprobs`` (1 = keep, 0 = freeze gradient), and
        ``prompt_tv_per_token`` broadcasts the per-prompt TV back to every
        response token (zero on padding) for logging.
    """
    with torch.no_grad():
        per_token_tv_half = 0.5 * (ratio - 1.0).abs()
        per_token_tv_half = torch.where(response_mask, per_token_tv_half, torch.zeros_like(per_token_tv_half))

        prompt_tv_per_token = torch.zeros_like(per_token_tv_half)
        if rollout_ids is None:
            num_response = response_mask.sum().to(per_token_tv_half.dtype).clamp_min(1.0)
            sample_tv = per_token_tv_half.sum() / num_response
            prompt_tv_per_token = torch.where(
                response_mask, sample_tv.expand_as(per_token_tv_half), prompt_tv_per_token
            )
        else:
            valid_rollout_ids = rollout_ids[response_mask]
            if valid_rollout_ids.numel() > 0:
                unique_rollouts, token_to_rollout_local = torch.unique(valid_rollout_ids, return_inverse=True)
                num_rollouts = int(unique_rollouts.numel())
                ones_tok = torch.ones(
                    valid_rollout_ids.numel(), dtype=per_token_tv_half.dtype, device=per_token_tv_half.device
                )
                rollout_sum = torch.zeros(num_rollouts, dtype=per_token_tv_half.dtype, device=per_token_tv_half.device)
                rollout_sum.index_add_(0, token_to_rollout_local, per_token_tv_half[response_mask])
                rollout_count = torch.zeros_like(rollout_sum)
                rollout_count.index_add_(0, token_to_rollout_local, ones_tok)
                per_rollout_tv = rollout_sum / rollout_count.clamp_min(1.0)

                rollout_to_prompt_id = unique_rollouts.div(num_samples_per_prompt, rounding_mode="floor")
                unique_prompts, rollout_to_prompt_local = torch.unique(rollout_to_prompt_id, return_inverse=True)
                num_prompts = int(unique_prompts.numel())
                ones_roll = torch.ones_like(per_rollout_tv)
                prompt_sum = torch.zeros(num_prompts, dtype=per_rollout_tv.dtype, device=per_rollout_tv.device)
                prompt_sum.index_add_(0, rollout_to_prompt_local, per_rollout_tv)
                prompt_count = torch.zeros_like(prompt_sum)
                prompt_count.index_add_(0, rollout_to_prompt_local, ones_roll)
                per_prompt_tv = prompt_sum / prompt_count.clamp_min(1.0)

                token_prompt_local = rollout_to_prompt_local[token_to_rollout_local]
                prompt_tv_per_token[response_mask] = per_prompt_tv[token_prompt_local]

        valid_tv = prompt_tv_per_token[response_mask] if response_mask.any() else prompt_tv_per_token.new_zeros(())
        max_prompt_tv = valid_tv.max() if valid_tv.numel() > 0 else prompt_tv_per_token.new_zeros(())
        if bool((max_prompt_tv <= divergence_threshold).item()):
            mask = response_mask.to(new_logprobs.dtype)
        else:
            prob = torch.exp(new_logprobs.clamp(min=-30.0, max=0.0))
            old_prob = torch.exp(behavior_logprobs.clamp(min=-30.0, max=0.0))
            ref_grad = torch.sign(prob - old_prob)
            token_safe = (advantages * ref_grad) <= 0
            prompt_within_budget = prompt_tv_per_token <= divergence_threshold
            keep = (prompt_within_budget | token_safe) & response_mask
            mask = keep.to(new_logprobs.dtype)
    return mask, prompt_tv_per_token


def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    config: GRPOExperimentConfig,
    tis_weights: torch.Tensor | None = None,
    policy_freeze_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.loss_fn == GRPOLossType.dapo:
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_lower, 1.0 + config.clip_higher)
    elif config.loss_fn == GRPOLossType.cispo:
        # cispo: directly clip ratio, no lower bound.
        # reinforce loss, so multiply by new logprobs
        pg_losses = -advantages * torch.clamp(ratio.detach(), max=1.0 + config.clip_higher) * new_logprobs
        pg_losses2 = pg_losses
    elif config.loss_fn == GRPOLossType.dppo:
        # DPPO (https://arxiv.org/abs/2602.04879). The trust region is enforced
        # by a DPPO mask passed in via ``tis_weights`` (see
        # :func:`compute_dppo_mask`); the caller is responsible for computing
        # ``ratio`` as π_θ / μ_θ' (rollout/behavior policy) — see Takeaway 2.
        # Eq. 11: L_DPPO = E[Σ_t M_t · r_t · A_t]. No symmetric clipping.
        pg_losses = -advantages * ratio
        pg_losses2 = pg_losses
    elif config.loss_fn == GRPOLossType.tvpo:
        # TVPO: REINFORCE-with-IS surrogate -A · clamp(r, max=c).detach() · log π.
        # Trust region is enforced via ``policy_freeze_mask`` (built by
        # :func:`compute_tvpo_mask`), which substitutes the detached loss for
        # blocked tokens so they contribute no gradient. The truncation cap is
        # primarily a numerical safeguard since the mask, not the cap, defines
        # the trust region.
        truncated_ratio = torch.clamp(ratio, max=config.tvpo_truncation_cap).detach()
        pg_losses = -advantages * truncated_ratio * new_logprobs
        pg_losses2 = pg_losses
    else:
        raise ValueError(f"Invalid loss function: {config.loss_fn}")

    if policy_freeze_mask is not None:
        keep = policy_freeze_mask.bool()
        pg_losses = torch.where(keep, pg_losses, pg_losses.detach())
        pg_losses2 = torch.where(keep, pg_losses2, pg_losses2.detach())

    if tis_weights is not None:
        pg_losses = pg_losses * tis_weights
        pg_losses2 = pg_losses2 * tis_weights

    pg_loss_max = torch.max(pg_losses, pg_losses2)

    if ref_logprobs is not None:
        # We want the KL loss to backpropagate through the model.
        # We also clamp the KL loss to avoid numerical instability.
        # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
        ref_logprobs_diff = (new_logprobs - ref_logprobs).clamp(-40.0, 40.0)
        kl_all = model_utils.estimate_kl(ref_logprobs_diff, ratio)
        kl = kl_all[config.kl_estimator]
    else:
        kl = torch.zeros_like(pg_loss_max)

    return pg_losses, pg_losses2, pg_loss_max, kl


def _compute_packing_kwargs(position_ids: torch.Tensor, cp_context: object | None = None) -> dict:
    """Compute packing kwargs from position_ids for packed sequences.

    These are consumed by the Qwen3.5 GatedDeltaNet packing patch so that the
    causal conv1d and recurrent state respect sequence boundaries.

    Full-attention layers do NOT need explicit cu_seq_lens here — HF flash
    attention already detects packing from position_ids resets.  Passing
    cu_seq_lens_q/k would also break Ulysses sequence parallelism, which
    slices position_ids per rank.

    When *cp_context* is provided it must already be a fully-constructed
    :class:`fla.ops.cp.context.FLACPContext` (e.g. from
    :func:`build_fla_cp_context_for_sample`). Its rank-local ``cu_seqlens``
    take precedence inside the FLA chunk kernel.
    """
    batch_size, seq_len = position_ids.shape
    is_start = position_ids == 0
    seq_idx = (is_start.cumsum(dim=-1) - 1).to(torch.int32)
    assert batch_size == 1, f"cu_seqlens computation assumes batch_size=1, got {batch_size}"
    starts = torch.where(is_start[0])[0].to(torch.int32)
    cu_seqlens = torch.cat([starts, torch.tensor([seq_len], dtype=torch.int32, device=position_ids.device)])
    kwargs: dict = {"seq_idx": seq_idx, "cu_seqlens": cu_seqlens}
    if cp_context is not None:
        # The context already carries rank-local cu_seqlens as set by
        # ``build_cp_context``; do NOT overwrite.  FLA's chunk kernel takes
        # cu_seqlens from the context when cp_context is present.
        kwargs["cp_context"] = cp_context
    return kwargs


def build_fla_cp_context_for_sample(
    global_position_ids: torch.Tensor, sp_world_size: int, sp_group, conv_kernel_size: int | None, local_seq_len: int
):
    """Build an ``FLACPContext`` correctly for one packed sample under Ulysses SP.

    ``global_position_ids`` is the un-sharded ``position_ids`` for the sample
    (shape ``[1, L_real]``). Ulysses pads each row to
    ``L_pad = local_seq_len * sp_world_size`` and slices by ``local_seq_len``
    per rank. We reconstruct global cu_seqlens covering all ``L_pad`` tokens
    (padding is treated as a single trailing synthetic sub-sequence that
    doesn't cross any sub-sequence boundary) and hand it to
    :func:`fla.ops.cp.build_cp_context`, which computes:

    - ``cu_seqlens`` (rank-local varlen boundaries)
    - ``pre_num_ranks`` / ``post_num_ranks`` (how many neighbor ranks the
      first/last sub-sequence on this rank extends into)
    - ``is_first_rank`` / ``is_last_rank`` (whether this rank owns the
      start/end of the sub-sequence that spans its boundary)
    - ``pre_num_conv_tokens`` (used by ``causal_conv1d`` CP to right-overlap
      kernel receptive fields)

    The key property we get from this vs. the old static construction is:
    **sub-sequences that don't cross a rank boundary are correctly started
    from zero state**, instead of being incorrectly chained across ranks.
    """
    # Lazy import so non-Qwen3.5 paths don't require fla's cp module.
    build_cp_context = importlib.import_module("fla.ops.cp").build_cp_context

    assert global_position_ids.dim() == 2 and global_position_ids.shape[0] == 1, (
        f"expected [1, L] position_ids, got {tuple(global_position_ids.shape)}"
    )

    real_len = int(global_position_ids.shape[-1])
    padded_len = local_seq_len * sp_world_size
    is_start = (global_position_ids == 0)[0]
    starts = torch.where(is_start)[0].to(torch.int32)
    # cu_seqlens = [start_of_seq_0, start_of_seq_1, ..., real_len, padded_len]
    # The final [real_len, padded_len] pair lets FLA partition tokens even on
    # ranks whose chunk lies entirely in the right-padding region.
    boundaries = [starts]
    if real_len > 0:
        boundaries.append(torch.tensor([real_len], dtype=torch.int32, device=global_position_ids.device))
    if padded_len > real_len:
        boundaries.append(torch.tensor([padded_len], dtype=torch.int32, device=global_position_ids.device))
    global_cu = torch.cat(boundaries)
    # Drop duplicate boundaries introduced when sub-sequences end exactly at
    # real_len or when real_len == padded_len.
    global_cu = torch.unique_consecutive(global_cu)

    return build_cp_context(
        cu_seqlens=global_cu.to(torch.long),
        group=sp_group,
        conv1d_kernel_size=conv_kernel_size,
        cu_seqlens_cpu=global_cu.to(torch.long).cpu(),
    )


def forward_for_logprobs(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int,
    temperature: float,
    return_entropy: bool = False,
    cp_context: Any = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward pass to compute log probabilities."""
    # For packed sequences, pass attention_mask=None so HF flash attention uses position_ids
    # to isolate sub-sequences instead of treating the whole pack as one sequence.
    extra_kwargs: dict = {}
    if (position_ids.diff(dim=-1) < 0).any():
        attention_mask = None
        extra_kwargs = _compute_packing_kwargs(position_ids, cp_context=cp_context)
    elif cp_context is not None:
        # Under SP, this rank's chunk may have no local position resets (e.g. a single
        # long rollout spanning both ranks) but still needs cp_context for cross-rank
        # SSM state passing in linear-attention layers.
        extra_kwargs = {"cp_context": cp_context}
    output = model(input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids, **extra_kwargs)
    logits = getattr(output, "logits", output)
    # The logits at position i predict token i+1, so we align them with labels shifted by 1
    logits = logits[:, :-1]
    if temperature != 1.0:
        logits.div_(temperature)
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


def _compute_forward_extra_kwargs(
    position_ids: torch.Tensor, attention_mask: torch.Tensor | None, cp_context: Any = None
) -> tuple[torch.Tensor | None, dict]:
    extra_kwargs: dict = {}
    if (position_ids.diff(dim=-1) < 0).any():
        attention_mask = None
        extra_kwargs = _compute_packing_kwargs(position_ids, cp_context=cp_context)
    elif cp_context is not None:
        extra_kwargs = {"cp_context": cp_context}
    return attention_mask, extra_kwargs


def _unwrap_deepspeed_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def get_causal_lm_backbone_and_head(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Return the transformer backbone and output head for fused-loss paths."""
    unwrapped_model = _unwrap_deepspeed_model(model)
    language_model = getattr(unwrapped_model, "language_model", None)

    lm_head = getattr(unwrapped_model, "lm_head", None)
    if lm_head is None and language_model is not None:
        lm_head = getattr(language_model, "lm_head", None)
    if lm_head is None and hasattr(unwrapped_model, "get_output_embeddings"):
        lm_head = unwrapped_model.get_output_embeddings()
    if lm_head is None:
        raise ValueError(f"Could not find an lm_head for {type(unwrapped_model).__name__}")

    backbone = getattr(unwrapped_model, "model", None)
    if backbone is None and language_model is not None:
        backbone = getattr(language_model, "model", None)
    if backbone is None:
        base_model_prefix = getattr(unwrapped_model, "base_model_prefix", None)
        if base_model_prefix:
            backbone = getattr(unwrapped_model, base_model_prefix, None)
    if backbone is None:
        raise ValueError(f"Could not find a transformer backbone for {type(unwrapped_model).__name__}")

    return backbone, lm_head


@dataclass(frozen=True)
class LigerGRPOForwardOutput:
    """Backbone output and LM-head parameters shared by the Liger GRPO path."""

    hidden_states: torch.Tensor
    lm_head_weight: torch.Tensor
    selected_token_ids: torch.Tensor
    lm_head_bias: torch.Tensor | None = None

    @property
    def lm_head_tensors(self) -> list[torch.Tensor]:
        return [param for param in (self.lm_head_weight, self.lm_head_bias) if isinstance(param, torch.Tensor)]


def forward_for_chunked_lm_head_logprobs(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int,
    temperature: float,
    lm_head_chunk_size: int = 2048,
    cp_context: Any = None,
) -> torch.Tensor:
    """Compute selected-token logprobs by running the lm_head over sequence chunks.

    This avoids materializing a full ``[batch, seq, vocab]`` logits tensor while
    still calling the actual lm_head module, so DeepSpeed hooks and lm_head fp32
    patches remain active.
    """
    attention_mask, extra_kwargs = _compute_forward_extra_kwargs(position_ids, attention_mask, cp_context=cp_context)
    backbone, lm_head = get_causal_lm_backbone_and_head(model)
    output = backbone(
        input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids, **extra_kwargs
    )
    hidden_states = getattr(output, "last_hidden_state", output[0] if isinstance(output, tuple) else output)
    hidden_states = hidden_states[:, :-1, :]

    labels = query_responses[:, 1:].clone().to(hidden_states.device)
    labels[labels == pad_token_id] = 0

    logprobs: list[torch.Tensor] = []
    for start in range(0, hidden_states.shape[1], lm_head_chunk_size):
        end = min(start + lm_head_chunk_size, hidden_states.shape[1])
        logits = lm_head(hidden_states[:, start:end, :])
        if temperature != 1.0:
            logits = logits / temperature
        logprobs.append(model_utils.log_softmax_and_gather(logits, labels[:, start:end]))
    return torch.cat(logprobs, dim=1) if len(logprobs) > 1 else logprobs[0]


def forward_for_liger_grpo_loss(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int,
    lm_head_fp32: bool = False,
    cp_context: Any = None,
) -> LigerGRPOForwardOutput:
    """Forward the backbone only and prepare inputs for Liger's fused GRPO loss."""
    attention_mask, extra_kwargs = _compute_forward_extra_kwargs(position_ids, attention_mask, cp_context=cp_context)
    backbone, lm_head = get_causal_lm_backbone_and_head(model)
    output = backbone(
        input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids, **extra_kwargs
    )
    last_hidden_state = getattr(output, "last_hidden_state", output[0] if isinstance(output, tuple) else output)
    last_hidden_state = last_hidden_state[:, :-1, :]

    selected_token_ids = query_responses[:, 1:].clone().to(last_hidden_state.device)
    selected_token_ids[selected_token_ids == pad_token_id] = 0

    if lm_head_fp32:
        last_hidden_state = last_hidden_state.float()

    return LigerGRPOForwardOutput(
        hidden_states=last_hidden_state,
        lm_head_weight=lm_head.weight,
        selected_token_ids=selected_token_ids,
        lm_head_bias=getattr(lm_head, "bias", None),
    )


def selected_logprobs_for_liger_grpo(forward_output: LigerGRPOForwardOutput, temperature: float) -> torch.Tensor:
    """Compute detached selected-token logprobs from Liger GRPO forward inputs."""
    return selective_logprobs_from_lm_head(
        hidden_states=forward_output.hidden_states.detach(),
        lm_head_weight=forward_output.lm_head_weight.detach(),
        selected_token_ids=forward_output.selected_token_ids,
        bias=forward_output.lm_head_bias.detach() if forward_output.lm_head_bias is not None else None,
        temperature=temperature,
    )


def selective_logprobs_from_lm_head(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    selected_token_ids: torch.Tensor,
    temperature: float,
    bias: torch.Tensor | None = None,
    vocab_chunk_size: int = 4096,
    seq_chunk_size: int = 2048,
) -> torch.Tensor:
    """Compute selected-token logprobs without materializing full-vocab logits."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    hidden = hidden_states.reshape(batch_size * seq_len, hidden_size).contiguous()
    targets = selected_token_ids.reshape(batch_size * seq_len).contiguous()
    logprobs = torch.empty(targets.shape, device=hidden.device, dtype=torch.float32)
    inv_temperature = 1.0 / temperature
    vocab_size = lm_head_weight.shape[0]

    for seq_start in range(0, hidden.shape[0], seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, hidden.shape[0])
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]
        num_rows = seq_end - seq_start
        row_idx = torch.arange(num_rows, device=hidden.device)

        max_logits = torch.full((num_rows,), float("-inf"), device=hidden.device, dtype=torch.float32)
        sum_exp = torch.zeros((num_rows,), device=hidden.device, dtype=torch.float32)
        target_logits = torch.zeros((num_rows,), device=hidden.device, dtype=torch.float32)

        for vocab_start in range(0, vocab_size, vocab_chunk_size):
            vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
            weight_chunk = lm_head_weight[vocab_start:vocab_end]
            logits_chunk = (hidden_chunk @ weight_chunk.to(hidden_chunk.dtype).t()).float()
            if bias is not None:
                logits_chunk.add_(bias[vocab_start:vocab_end].float())
            logits_chunk.mul_(inv_temperature)

            chunk_max = logits_chunk.amax(dim=-1)
            new_max = torch.maximum(max_logits, chunk_max)
            sum_exp = sum_exp * torch.exp(max_logits - new_max) + torch.exp(logits_chunk - new_max.unsqueeze(-1)).sum(
                dim=-1
            )
            max_logits = new_max

            in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
            local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            target_logits += logits_chunk[row_idx, local_idx] * in_chunk

        logprobs[seq_start:seq_end] = target_logits - (max_logits + torch.log(sum_exp))

    return logprobs.reshape(batch_size, seq_len)


def compute_logprobs(
    model: torch.nn.Module,
    data_BT: data_types.CollatedBatchData,
    pad_token_id: int,
    temperature: float,
    use_grad: bool = False,
    batch_size: int | None = None,
    cp_context: Any = None,
    cp_contexts: list[Any] | None = None,
    use_chunked_lm_head: bool = False,
    lm_head_chunk_size: int = 2048,
) -> list[torch.Tensor]:
    """Compute log probabilities for all samples in batch.

    ``cp_contexts`` (if provided) is a per-sample list of FLA CP contexts
    — one entry per sample, aligned with ``data_BT.query_responses``. This is
    the correct path under Ulysses sequence parallelism for Qwen3.5 hybrid
    models: each sample's context is built by ``build_fla_cp_context_for_sample``
    from that sample's *global* (pre-split) cu_seqlens. The older
    ``cp_context`` scalar is kept for backward compatibility and is used as a
    fallback for every sample when ``cp_contexts`` is None.
    """
    logprobs_BT: list[torch.Tensor] = []
    num_samples = len(data_BT.query_responses)

    if batch_size is None:
        batch_size = 1

    def ctx_for(i: int) -> Any:
        if cp_contexts is not None:
            return cp_contexts[i]
        return cp_context

    context = torch.enable_grad() if use_grad else torch.no_grad()
    with context:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))

            query_responses = [data_BT.query_responses[i] for i in batch_indices]
            position_ids = [data_BT.position_ids[i] for i in batch_indices]
            shapes = [tuple(t.shape) for t in query_responses]

            # If samples can carry different cp_contexts, we cannot safely
            # concatenate them into a single forward — fall back to one
            # forward per sample (same pattern used when shapes differ).
            different_ctx = cp_contexts is not None and len({id(cp_contexts[i]) for i in batch_indices}) > 1

            if len(set(shapes)) != 1 or different_ctx:
                for i in batch_indices:
                    if use_chunked_lm_head:
                        single_logprobs = forward_for_chunked_lm_head_logprobs(
                            model,
                            data_BT.query_responses[i],
                            None,
                            data_BT.position_ids[i],
                            pad_token_id,
                            temperature,
                            lm_head_chunk_size=lm_head_chunk_size,
                            cp_context=ctx_for(i),
                        )
                    else:
                        single_logprobs, _ = forward_for_logprobs(
                            model,
                            data_BT.query_responses[i],
                            None,
                            data_BT.position_ids[i],
                            pad_token_id,
                            temperature,
                            False,
                            cp_context=ctx_for(i),
                        )

                    response_mask_BT = data_BT.response_masks[i]
                    single_logprobs = mask_logprobs(single_logprobs, response_mask_BT[:, 1:].bool())
                    logprobs_BT.append(single_logprobs)
                continue

            batch_query_responses = torch.cat(query_responses, dim=0)
            batch_position_ids = torch.cat(position_ids, dim=0)

            if use_chunked_lm_head:
                batch_logprobs = forward_for_chunked_lm_head_logprobs(
                    model,
                    batch_query_responses,
                    None,
                    batch_position_ids,
                    pad_token_id,
                    temperature,
                    lm_head_chunk_size=lm_head_chunk_size,
                    cp_context=ctx_for(batch_indices[0]),
                )
            else:
                batch_logprobs, _ = forward_for_logprobs(
                    model,
                    batch_query_responses,
                    None,
                    batch_position_ids,
                    pad_token_id,
                    temperature,
                    False,
                    cp_context=ctx_for(batch_indices[0]),
                )

            sample_sizes = [data_BT.query_responses[i].shape[0] for i in batch_indices]
            split_logprobs = torch.split(batch_logprobs, sample_sizes, dim=0)

            for i, logprob_BT in zip(batch_indices, split_logprobs):
                response_mask_BT = data_BT.response_masks[i]
                logprob_BT = mask_logprobs(logprob_BT, response_mask_BT[:, 1:].bool())
                logprobs_BT.append(logprob_BT)

    return logprobs_BT


def calculate_token_counts(
    accumulation_steps: int,
    data_BT: data_types.CollatedBatchData,
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
) -> dict[int, float]:
    """Compute total token counts per accumulation group, all-reduced across DP ranks.

    Copied from grpo_fast.py to share logic with olmo_core_train_modules.py.
    """
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
    "val/tis_clipfrac",
    "val/tis_ratio",
]


def create_loss_stats(num_samples: int, device: torch.device, record_entropy: bool = False) -> dict[str, torch.Tensor]:
    stats = {key: torch.zeros(num_samples, device=device) for key in _SCALAR_LOSS_STAT_KEYS}
    if record_entropy:
        stats |= {"policy/entropy_avg": torch.zeros(num_samples, device=device)}
    return stats


def populate_sample_loss_stats(
    loss_stats_B: dict[str, torch.Tensor],
    sample_idx: int,
    pg_losses: torch.Tensor,
    pg_losses2: torch.Tensor,
    pg_loss: torch.Tensor,
    ratio: torch.Tensor,
    loss: torch.Tensor,
    response_mask: torch.Tensor,
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropy: torch.Tensor | None,
    config: GRPOExperimentConfig,
    tis_clamped: torch.Tensor | None = None,
    tis_unclamped: torch.Tensor | None = None,
) -> None:
    with torch.no_grad():
        if config.load_ref_policy and ref_logprobs is not None:
            ref_logprobs_diff = (new_logprobs - ref_logprobs).clamp(-40.0, 40.0)
            kl_4BT = model_utils.estimate_kl(ref_logprobs_diff, ratio)
            kl_values = masked_mean(kl_4BT, response_mask).float()
            for j in range(4):
                loss_stats_B[f"objective/kl{j}_avg"][sample_idx] = kl_values[j]
            loss_stats_B["loss/kl_avg"][sample_idx] = kl_values[config.kl_estimator] * config.beta
        if tis_clamped is not None and tis_unclamped is not None:
            loss_stats_B["val/tis_ratio"][sample_idx] = masked_mean(tis_clamped.float(), response_mask)
            loss_stats_B["val/tis_clipfrac"][sample_idx] = masked_mean(
                (tis_clamped < tis_unclamped).float(), response_mask
            )
        loss_stats_B["policy/clipfrac_avg"][sample_idx] = masked_mean((pg_losses2 > pg_losses).float(), response_mask)
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
