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
    use_liger_grpo_loss: bool = False
    """Whether to use the tiled lm-head GRPO loss path. Supports DAPO, CISPO, DPPO, and TVPO."""
    liger_grpo_loss_chunk_size: int = 1
    """Number of lm-head loss tiles to use in the tiled GRPO loss path."""
    liger_grpo_loss_compile: bool = True
    """Deprecated; retained for backward-compatible configs."""

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
    sequence_tis_mask_log_ratio_threshold: float = 0.0
    """Sequence-level threshold δ for avg log(π_rollout / π_θ).

    When >0, whole response samples whose average log-ratio exceeds this threshold
    are multiplied by 0 in the pg loss.
    """
    sequence_tis_mask_negative_advantages_only: bool = True
    """Only apply the sequence-level ratio mask to samples with negative mean advantage."""
    kl_estimator: Literal[0, 1, 2, 3] = 2
    """the KL estimator to use"""
    loss_denominator: str = "token"
    """Optional denominator for the policy loss; can be "token", "sequence", or a float value.
    when "token", the loss is divided by the total number of tokens in the batch (standard LM training).
    when "sequence", each sequence's token losses are averaged first, then sequences are averaged equally.
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

    # PPO value model. When enabled, the trainer learns a scalar value function
    # and replaces group-relative advantages with GAE advantages.
    use_value_model: bool = False
    """Whether to train and use a scalar PPO value model alongside the policy."""
    value_model_name_or_path: str | None = None
    """Optional path or HF name for initializing the value model. Defaults to the policy model."""
    value_loss_coef: float = 0.5
    """Coefficient for the value loss."""
    value_learning_rate: float | None = None
    """Value model learning rate. Defaults to the policy learning rate."""
    vf_clip_range: float = 0.2
    """PPO-style value-function clipping range. Set to 0 to disable clipping."""
    gamma: float = 1.0
    """Discount factor for GAE."""
    gae_lambda: float = 1.0
    """Lambda for GAE."""
    value_num_mini_batches: int | None = None
    """Optional mini-batch count for the value model. Defaults to num_mini_batches."""
    whiten_advantages: bool = False
    """If True, whiten GAE advantages across response tokens before policy training."""

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
        if self.use_liger_grpo_loss:
            if self.loss_fn not in (GRPOLossType.dapo, GRPOLossType.cispo, GRPOLossType.dppo, GRPOLossType.tvpo):
                raise ValueError(
                    "`use_liger_grpo_loss` currently only supports `loss_fn=dapo`, `loss_fn=cispo`, "
                    "`loss_fn=dppo`, or `loss_fn=tvpo`."
                )
            if self.record_entropy:
                raise ValueError("`use_liger_grpo_loss` does not support `record_entropy=True`.")
            if self.loss_denominator not in ("token", "sequence"):
                raise ValueError("`use_liger_grpo_loss` currently requires `loss_denominator=token` or `sequence`.")
            if self.kl_estimator != 2:
                raise ValueError("`use_liger_grpo_loss` uses the default KL estimator and requires `kl_estimator=2`.")
            if self.liger_grpo_loss_chunk_size <= 0:
                raise ValueError("`liger_grpo_loss_chunk_size` must be greater than 0.")
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
        if self.use_value_model:
            if self.value_loss_coef < 0.0:
                raise ValueError(f"`value_loss_coef` must be >= 0, got {self.value_loss_coef}.")
            if not 0.0 <= self.gamma <= 1.0:
                raise ValueError(f"`gamma` must be in [0, 1], got {self.gamma}.")
            if not 0.0 <= self.gae_lambda <= 1.0:
                raise ValueError(f"`gae_lambda` must be in [0, 1], got {self.gae_lambda}.")
            if self.value_num_mini_batches is not None and self.value_num_mini_batches <= 0:
                raise ValueError("`value_num_mini_batches` must be greater than 0 when set.")
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
        if self.sequence_tis_mask_log_ratio_threshold < 0.0:
            raise ValueError(
                "sequence_tis_mask_log_ratio_threshold must be ≥ 0 "
                f"(got {self.sequence_tis_mask_log_ratio_threshold=}). Use 0 to disable."
            )
        if self.loss_denominator not in ("token", "sequence") and float(self.loss_denominator) <= 0:
            raise ValueError(
                "loss_denominator must be 'token', 'sequence', or a valid float greater than 0, "
                f"got: {self.loss_denominator}"
            )
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


def compute_sequence_tis_mask(
    new_logprobs: torch.Tensor,
    vllm_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    log_ratio_threshold: float,
    negative_advantages_only: bool = True,
    rollout_sample_ids: torch.Tensor | None = None,
    sequence_process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor | None:
    """Whole-sequence gate: mask when mean log(π_rollout / π_θ) exceeds δ."""
    if log_ratio_threshold <= 0.0:
        return None
    with torch.no_grad():
        if rollout_sample_ids is not None:
            sequence_ids = rollout_sample_ids
            valid = response_mask.bool() & (rollout_sample_ids >= 0)
        else:
            row_ids = torch.arange(response_mask.shape[0], device=response_mask.device, dtype=torch.long)
            sequence_ids = row_ids[:, None].expand_as(response_mask)
            valid = response_mask.bool()

        valid = valid & ~torch.isnan(new_logprobs) & ~torch.isnan(vllm_logprobs)
        has_valid = valid.any()
        local_max = (
            sequence_ids[valid].max() if has_valid else torch.tensor(-1, dtype=torch.long, device=response_mask.device)
        )
        if sequence_process_group is not None:
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=sequence_process_group)
        if local_max.item() < 0:
            return torch.zeros(response_mask.shape, dtype=new_logprobs.dtype, device=response_mask.device)

        counts = torch.zeros(int(local_max.item()) + 1, dtype=torch.float32, device=response_mask.device)

        log_ratio = (vllm_logprobs - new_logprobs).clamp(-10.0, 10.0)
        log_ratio = torch.where(valid, log_ratio, torch.zeros_like(log_ratio))
        advantage_values = torch.where(valid, advantages, torch.zeros_like(advantages))

        flat_ids = sequence_ids[valid].long()
        log_ratio_sums = torch.zeros_like(counts)
        advantage_sums = torch.zeros_like(counts)
        if flat_ids.numel() > 0:
            counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
            log_ratio_sums.scatter_add_(0, flat_ids, log_ratio[valid].float())
            advantage_sums.scatter_add_(0, flat_ids, advantage_values[valid].float())
        if sequence_process_group is not None:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=sequence_process_group)
            dist.all_reduce(log_ratio_sums, op=dist.ReduceOp.SUM, group=sequence_process_group)
            dist.all_reduce(advantage_sums, op=dist.ReduceOp.SUM, group=sequence_process_group)

        avg_log_ratio = log_ratio_sums / counts.clamp_min(1.0)
        mean_advantage = advantage_sums / counts.clamp_min(1.0)
        sequence_should_mask = avg_log_ratio > log_ratio_threshold
        if negative_advantages_only:
            sequence_should_mask = sequence_should_mask & (mean_advantage < 0.0)

        keep_by_sequence = (~sequence_should_mask).to(dtype=new_logprobs.dtype)
        mask = torch.zeros(response_mask.shape, dtype=new_logprobs.dtype, device=response_mask.device)
        mask[valid] = keep_by_sequence[flat_ids]
        return mask


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
    sequence_process_group: dist.ProcessGroup | None = None,
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
        sequence_process_group: optional sequence-parallel process group. When
            set, token sums/counts are all-reduced before computing rollout and
            prompt TV so the trust region is based on full sequences, not local
            SP shards.

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
            tv_sum = per_token_tv_half.sum()
            num_response = response_mask.sum().to(per_token_tv_half.dtype)
            if sequence_process_group is not None:
                dist.all_reduce(tv_sum, op=dist.ReduceOp.SUM, group=sequence_process_group)
                dist.all_reduce(num_response, op=dist.ReduceOp.SUM, group=sequence_process_group)
            sample_tv = tv_sum / num_response.clamp_min(1.0)
            prompt_tv_per_token = torch.where(
                response_mask, sample_tv.expand_as(per_token_tv_half), prompt_tv_per_token
            )
        else:
            valid = response_mask.bool()
            has_valid = bool(valid.any().item())
            local_max = rollout_ids[valid].max() if has_valid else torch.tensor(-1, device=rollout_ids.device)
            local_max = local_max.to(dtype=torch.long)
            if sequence_process_group is not None:
                dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=sequence_process_group)
            if local_max.item() >= 0:
                num_rollouts = int(local_max.item()) + 1
                rollout_sum = torch.zeros(num_rollouts, dtype=per_token_tv_half.dtype, device=per_token_tv_half.device)
                rollout_count = torch.zeros_like(rollout_sum)
                if has_valid:
                    flat_rollout_ids = rollout_ids[valid].long()
                    rollout_sum.index_add_(0, flat_rollout_ids, per_token_tv_half[valid])
                    rollout_count.index_add_(
                        0, flat_rollout_ids, torch.ones_like(flat_rollout_ids, dtype=rollout_count.dtype)
                    )
                if sequence_process_group is not None:
                    dist.all_reduce(rollout_sum, op=dist.ReduceOp.SUM, group=sequence_process_group)
                    dist.all_reduce(rollout_count, op=dist.ReduceOp.SUM, group=sequence_process_group)

                rollout_has_tokens = rollout_count > 0
                per_rollout_tv = rollout_sum / rollout_count.clamp_min(1.0)
                rollout_to_prompt_id = torch.arange(num_rollouts, device=rollout_ids.device, dtype=torch.long).div(
                    num_samples_per_prompt, rounding_mode="floor"
                )
                num_prompts = int(rollout_to_prompt_id.max().item()) + 1
                prompt_sum = torch.zeros(num_prompts, dtype=per_rollout_tv.dtype, device=per_rollout_tv.device)
                prompt_count = torch.zeros_like(prompt_sum)
                prompt_sum.index_add_(
                    0,
                    rollout_to_prompt_id,
                    torch.where(rollout_has_tokens, per_rollout_tv, torch.zeros_like(per_rollout_tv)),
                )
                prompt_count.index_add_(0, rollout_to_prompt_id, rollout_has_tokens.to(dtype=prompt_count.dtype))
                per_prompt_tv = prompt_sum / prompt_count.clamp_min(1.0)

                if has_valid:
                    token_prompt_id = rollout_ids[valid].long().div(num_samples_per_prompt, rounding_mode="floor")
                    prompt_tv_per_token[valid] = per_prompt_tv[token_prompt_id]

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


class TiledGRPOLMHeadLoss(torch.autograd.Function):
    """Tiled DAPO/CISPO/DPPO/TVPO lm-head loss that avoids materializing full logits.

    This follows DeepSpeed's ``TiledFusedLogitsLoss`` pattern: the lm-head
    projection and scalar loss are recomputed per tile in ``forward`` and
    ``torch.autograd.backward`` is called per tile to accumulate lm-head grads.
    The custom ``backward`` then returns the precomputed hidden-state gradient
    so the outer DeepSpeed backward only traverses the backbone.
    """

    @staticmethod
    def forward(
        ctx,
        lm_head: torch.nn.Module,
        hidden_states: torch.Tensor,
        selected_token_ids: torch.Tensor,
        response_mask: torch.Tensor,
        policy_mask: torch.Tensor,
        has_policy_mask: bool,
        policy_freeze_mask: torch.Tensor,
        has_policy_freeze_mask: bool,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        has_ref_logprobs: bool,
        temperature: float,
        beta: float,
        clip_lower: float,
        clip_higher: float,
        loss_fn: str,
        dppo_divergence_type: str,
        dppo_divergence_threshold: float,
        tvpo_truncation_cap: float,
        sequence_loss: bool,
        rollout_sample_ids: torch.Tensor,
        has_rollout_sample_ids: bool,
        sequence_process_group: dist.ProcessGroup | None,
        shards: int,
        loss_scale: torch.Tensor,
        compute_params: list[torch.nn.Parameter],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B, T, H], got {tuple(hidden_states.shape)}")
        if selected_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("selected_token_ids must match hidden_states batch/sequence dimensions")
        if response_mask.shape != hidden_states.shape[:2]:
            raise ValueError("response_mask must match hidden_states batch/sequence dimensions")
        if has_policy_mask and policy_mask.shape != hidden_states.shape[:2]:
            raise ValueError("policy_mask must match hidden_states batch/sequence dimensions")
        if has_policy_freeze_mask and policy_freeze_mask.shape != hidden_states.shape[:2]:
            raise ValueError("policy_freeze_mask must match hidden_states batch/sequence dimensions")
        if shards < 1:
            raise ValueError(f"shards must be >= 1, got {shards}")
        loss_type = GRPOLossType(loss_fn)

        x_requires_grad = hidden_states.requires_grad
        batch_size, seq_len, hidden_size = hidden_states.shape
        x = hidden_states.detach().reshape(-1, hidden_size)
        x.requires_grad_(x_requires_grad)

        labels = selected_token_ids.reshape(-1)
        mask_2d = response_mask.to(dtype=torch.bool)
        mask = mask_2d.reshape(-1)
        advantages = advantages.reshape(-1)
        old_logprobs = old_logprobs.reshape(-1)
        if has_policy_mask:
            policy_mask = policy_mask.reshape(-1)
        if has_policy_freeze_mask:
            policy_freeze_mask = policy_freeze_mask.reshape(-1)
        if has_ref_logprobs:
            ref_logprobs = ref_logprobs.reshape(-1)

        num_tokens = x.shape[0]
        shards = min(shards, num_tokens)
        if sequence_loss:
            rollout_ids = rollout_sample_ids if has_rollout_sample_ids else None
            loss_weights_2d, loss_denom = _sequence_loss_weights(mask_2d, rollout_ids, sequence_process_group)
        else:
            loss_weights_2d = mask_2d.to(dtype=torch.float32)
            loss_denom = loss_weights_2d.sum().clamp_min(1.0)
        loss_weights = loss_weights_2d.reshape(-1)
        metric_denom = mask_2d.to(dtype=torch.float32).sum()
        incoming_grad = (loss_scale.detach().to(dtype=torch.float32) / loss_denom).reshape(())

        x_grad = torch.zeros_like(x) if x_requires_grad else None
        x_shards = list(torch.chunk(x, chunks=shards, dim=0))
        label_shards = list(torch.chunk(labels, chunks=shards, dim=0))
        mask_shards = list(torch.chunk(mask, chunks=shards, dim=0))
        loss_weight_shards = list(torch.chunk(loss_weights, chunks=shards, dim=0))
        advantage_shards = list(torch.chunk(advantages, chunks=shards, dim=0))
        old_logprob_shards = list(torch.chunk(old_logprobs, chunks=shards, dim=0))
        policy_mask_shards = list(torch.chunk(policy_mask, chunks=shards, dim=0)) if has_policy_mask else []
        policy_freeze_mask_shards = (
            list(torch.chunk(policy_freeze_mask, chunks=shards, dim=0)) if has_policy_freeze_mask else []
        )
        ref_logprob_shards = list(torch.chunk(ref_logprobs, chunks=shards, dim=0)) if has_ref_logprobs else []

        total_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
        total_kl_sum = torch.zeros(4, dtype=torch.float32, device=hidden_states.device)
        total_clip_sum = torch.zeros_like(total_loss_sum)
        total_ratio_sum = torch.zeros_like(total_loss_sum)
        compute_params = [p for p in compute_params if p.requires_grad]

        for shard_idx, x_shard in enumerate(x_shards):
            if compute_params:
                grad_is_ready = shard_idx + 1 == len(x_shards)
                for param in compute_params:
                    param.ds_grad_is_ready = grad_is_ready

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
                elif loss_type == GRPOLossType.cispo:
                    clipped_ratio = torch.clamp(ratio.detach(), max=1.0 + clip_higher)
                    pg_losses = -advantage_shards[shard_idx] * clipped_ratio * new_logprobs
                    pg_losses2 = pg_losses
                elif loss_type == GRPOLossType.dppo:
                    pg_losses = -advantage_shards[shard_idx] * ratio
                    pg_losses2 = pg_losses
                    dppo_mask, _ = compute_dppo_mask(
                        new_logprobs=new_logprobs,
                        behavior_logprobs=old_logprob_shards[shard_idx],
                        advantages=advantage_shards[shard_idx],
                        ratio=ratio,
                        response_mask=mask_shards[shard_idx].bool(),
                        divergence_type=dppo_divergence_type,
                        divergence_threshold=dppo_divergence_threshold,
                    )
                    pg_losses = pg_losses * dppo_mask
                    pg_losses2 = pg_losses2 * dppo_mask
                elif loss_type == GRPOLossType.tvpo:
                    truncated_ratio = torch.clamp(ratio, max=tvpo_truncation_cap).detach()
                    pg_losses = -advantage_shards[shard_idx] * truncated_ratio * new_logprobs
                    pg_losses2 = pg_losses
                else:
                    raise ValueError(f"`tiled_grpo_lm_head_loss` does not support loss_fn={loss_type}.")
                if has_policy_freeze_mask:
                    keep = policy_freeze_mask_shards[shard_idx].bool()
                    pg_losses = torch.where(keep, pg_losses, pg_losses.detach())
                    pg_losses2 = torch.where(keep, pg_losses2, pg_losses2.detach())
                if has_policy_mask:
                    policy_weight = policy_mask_shards[shard_idx].to(dtype=pg_losses.dtype)
                    pg_losses = pg_losses * policy_weight
                    pg_losses2 = pg_losses2 * policy_weight
                pg_loss = torch.max(pg_losses, pg_losses2)

                if has_ref_logprobs:
                    ref_diff = (new_logprobs - ref_logprob_shards[shard_idx]).clamp(-40.0, 40.0)
                    kl_all = model_utils.estimate_kl(ref_diff, ratio)
                    kl = kl_all[2]
                else:
                    kl = torch.zeros_like(pg_loss)
                    kl_all = torch.zeros(4, *pg_loss.shape, dtype=pg_loss.dtype, device=pg_loss.device)

                shard_mask = mask_shards[shard_idx].to(dtype=pg_loss.dtype)
                per_token_loss = pg_loss + beta * kl
                loss_sum = (per_token_loss * loss_weight_shards[shard_idx].to(dtype=per_token_loss.dtype)).sum()

            total_loss_sum = total_loss_sum + loss_sum.detach().float()
            total_kl_sum = total_kl_sum + (kl_all.detach().float() * shard_mask.float()).sum(dim=-1)
            total_clip_sum = total_clip_sum + ((pg_losses2 > pg_losses).detach().float() * shard_mask.float()).sum()
            total_ratio_sum = total_ratio_sum + (ratio.detach().float() * shard_mask.float()).sum()
            torch.autograd.backward(loss_sum, incoming_grad.to(dtype=loss_sum.dtype))

        if compute_params:
            for param in compute_params:
                param.ds_grad_is_ready = True

        if sequence_process_group is not None:
            dist.all_reduce(total_kl_sum, op=dist.ReduceOp.SUM, group=sequence_process_group)
            dist.all_reduce(total_clip_sum, op=dist.ReduceOp.SUM, group=sequence_process_group)
            dist.all_reduce(total_ratio_sum, op=dist.ReduceOp.SUM, group=sequence_process_group)
            dist.all_reduce(metric_denom, op=dist.ReduceOp.SUM, group=sequence_process_group)
        metric_denom = metric_denom.clamp_min(1.0)

        if x_grad is None:
            x_grad = torch.zeros_like(x)
        ctx.save_for_backward(x_grad.reshape(batch_size, seq_len, hidden_size).detach())

        loss = total_loss_sum / loss_denom * loss_scale.detach().to(dtype=total_loss_sum.dtype)
        kl_avg = total_kl_sum / metric_denom
        clipfrac = total_clip_sum / metric_denom
        ratio_avg = total_ratio_sum / metric_denom
        return loss, kl_avg, clipfrac, ratio_avg

    @staticmethod
    def backward(ctx, *grads) -> tuple:
        (x_grad,) = ctx.saved_tensors
        grad = grads[0]
        if isinstance(grad, torch.Tensor):
            x_grad = x_grad * grad.to(dtype=x_grad.dtype)
        return (None, x_grad, *([None] * 25))


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
    loss_fn: str | GRPOLossType = GRPOLossType.dapo,
    dppo_divergence_type: str | DPPODivergenceType = DPPODivergenceType.tv,
    dppo_divergence_threshold: float = 0.1,
    tvpo_truncation_cap: float = 20.0,
    loss_denominator: str = "token",
    rollout_sample_ids: torch.Tensor | None = None,
    sequence_process_group: dist.ProcessGroup | None = None,
    policy_mask: torch.Tensor | None = None,
    policy_freeze_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    has_ref_logprobs = ref_logprobs is not None
    if ref_logprobs is None:
        ref_logprobs = torch.empty(0, dtype=old_logprobs.dtype, device=old_logprobs.device)
    has_policy_mask = policy_mask is not None
    if policy_mask is None:
        policy_mask = torch.empty(0, dtype=response_mask.dtype, device=response_mask.device)
    has_policy_freeze_mask = policy_freeze_mask is not None
    if policy_freeze_mask is None:
        policy_freeze_mask = torch.empty(0, dtype=response_mask.dtype, device=response_mask.device)
    has_rollout_sample_ids = rollout_sample_ids is not None
    if rollout_sample_ids is None:
        rollout_sample_ids = torch.empty(0, dtype=torch.long, device=old_logprobs.device)
    compute_params = list(lm_head.parameters(recurse=False))
    return TiledGRPOLMHeadLoss.apply(
        lm_head,
        hidden_states,
        selected_token_ids,
        response_mask,
        policy_mask,
        has_policy_mask,
        policy_freeze_mask,
        has_policy_freeze_mask,
        advantages,
        old_logprobs,
        ref_logprobs,
        has_ref_logprobs,
        temperature,
        beta,
        clip_lower,
        clip_higher,
        loss_fn,
        dppo_divergence_type,
        dppo_divergence_threshold,
        tvpo_truncation_cap,
        loss_denominator == "sequence",
        rollout_sample_ids,
        has_rollout_sample_ids,
        sequence_process_group,
        shards,
        loss_scale,
        compute_params,
    )


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

    Under Ulysses SP, a rank-local slice can start in the middle of a packed
    sub-sequence. Include a leading local boundary in that case so every packed
    call has well-formed rank-local metadata even when there is no local reset.
    """
    batch_size, seq_len = position_ids.shape
    is_start = position_ids == 0
    seq_idx = is_start.cumsum(dim=-1)
    seq_idx = (seq_idx - is_start[:, :1].to(seq_idx.dtype)).to(torch.int32)
    assert batch_size == 1, f"cu_seqlens computation assumes batch_size=1, got {batch_size}"
    starts = torch.where(is_start[0])[0].to(torch.int32)
    if starts.numel() == 0 or starts[0].item() != 0:
        starts = torch.cat([torch.zeros(1, dtype=torch.int32, device=position_ids.device), starts])
    cu_seqlens = torch.cat([starts, torch.tensor([seq_len], dtype=torch.int32, device=position_ids.device)])
    cu_seqlens = torch.unique_consecutive(cu_seqlens)
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
    attention_mask, extra_kwargs = _compute_forward_extra_kwargs(position_ids, attention_mask, cp_context=cp_context)
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


def _unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


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
    return backbone, lm_head


def patch_liger_grpo_lm_head_forward(
    lm_head: torch.nn.Module, liger_grpo_loss: torch.nn.Module, lm_head_fp32: bool = False
) -> bool:
    """Route explicit GRPO-loss calls through lm_head.forward so ZeRO hooks see normal module execution."""
    if getattr(lm_head, "_open_instruct_liger_grpo_patch", False):
        return False

    original_forward = lm_head.forward

    def patched_forward(
        hidden_states: torch.Tensor,
        *args,
        selected_token_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        old_per_token_logps: torch.Tensor | None = None,
        **kwargs,
    ):
        if selected_token_ids is None:
            return original_forward(hidden_states, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError("Liger GRPO lm_head forward only supports keyword GRPO inputs.")
        if attention_mask is None or advantages is None:
            raise RuntimeError("Liger GRPO lm_head forward requires attention_mask and advantages.")

        weight = lm_head.weight
        bias = getattr(lm_head, "bias", None)
        if lm_head_fp32:
            hidden_states = hidden_states.float()
            weight = weight.float()
            bias = bias.float() if isinstance(bias, torch.Tensor) else bias

        return liger_grpo_loss(
            _input=hidden_states,
            lin_weight=weight,
            selected_token_ids=selected_token_ids,
            attention_mask=attention_mask,
            advantages=advantages,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
        )

    lm_head.forward = patched_forward
    lm_head._open_instruct_liger_grpo_patch = True
    return True


def forward_for_liger_hidden_states(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cp_context: Any = None,
) -> torch.Tensor:
    attention_mask, extra_kwargs = _compute_forward_extra_kwargs(position_ids, attention_mask, cp_context=cp_context)
    backbone, _ = get_causal_lm_backbone_and_lm_head(model)
    output = backbone(
        input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids, **extra_kwargs
    )
    hidden_states = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    return hidden_states[:, :-1]


def _compute_forward_extra_kwargs(
    position_ids: torch.Tensor, attention_mask: torch.Tensor | None, cp_context: Any = None
) -> tuple[torch.Tensor | None, dict]:
    extra_kwargs: dict = {}
    if cp_context is not None or (position_ids.diff(dim=-1) < 0).any():
        attention_mask = None
        extra_kwargs = _compute_packing_kwargs(position_ids, cp_context=cp_context)
    return attention_mask, extra_kwargs


def compute_logprobs(
    model: torch.nn.Module,
    data_BT: data_types.CollatedBatchData,
    pad_token_id: int,
    temperature: float,
    use_grad: bool = False,
    batch_size: int | None = None,
    cp_context: Any = None,
    cp_contexts: list[Any] | None = None,
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


def _sequence_id_counts(
    response_mask: torch.Tensor,
    rollout_sample_ids: torch.Tensor | None = None,
    sequence_process_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = response_mask.bool()
    if rollout_sample_ids is not None:
        valid = valid & (rollout_sample_ids >= 0)
        sequence_ids = rollout_sample_ids
    else:
        row_ids = torch.arange(response_mask.shape[0], device=response_mask.device, dtype=torch.long)
        sequence_ids = row_ids[:, None].expand_as(response_mask)

    has_valid = valid.any()
    local_max = (
        sequence_ids[valid].max() if has_valid else torch.tensor(-1, dtype=torch.long, device=response_mask.device)
    )
    if sequence_process_group is not None:
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=sequence_process_group)
    if local_max.item() < 0:
        return torch.zeros(0, dtype=torch.float32, device=response_mask.device), valid

    counts = torch.zeros(int(local_max.item()) + 1, dtype=torch.float32, device=response_mask.device)
    if has_valid:
        flat_ids = sequence_ids[valid].long()
        counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
    if sequence_process_group is not None:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=sequence_process_group)

    return counts, valid


def _count_sequences(
    response_mask: torch.Tensor,
    rollout_sample_ids: torch.Tensor | None = None,
    sequence_process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    counts, _ = _sequence_id_counts(response_mask, rollout_sample_ids, sequence_process_group)
    return (counts > 0).to(dtype=torch.float32).sum()


def calculate_sequence_counts(
    accumulation_steps: int,
    data_BT: data_types.CollatedBatchData,
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
    sequence_process_group: dist.ProcessGroup | None = None,
    sequence_group_rank: int = 0,
) -> dict[int, float]:
    """Compute sequence counts per accumulation group, all-reduced across DP ranks."""
    accumulation_counts: dict[int, float] = {}
    local_counts = []
    for i, response_mask in enumerate(data_BT.response_masks):
        response_mask = response_mask[:, 1:].bool()
        rollout_sample_ids = data_BT.rollout_sample_ids[i][:, 1:] if data_BT.rollout_sample_ids is not None else None
        count = _count_sequences(response_mask, rollout_sample_ids, sequence_process_group)
        if sequence_process_group is not None and sequence_group_rank != 0:
            count = torch.zeros_like(count)
        local_counts.append(count)
    if not local_counts:
        return accumulation_counts

    counts_tensor = torch.stack(local_counts).to(device)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM, group=process_group)

    for i, count in enumerate(counts_tensor):
        group_idx = i // accumulation_steps
        key = int(group_idx * accumulation_steps)
        accumulation_counts[key] = accumulation_counts.get(key, 0.0) + count.item()

    return accumulation_counts


def sequence_weighted_mean(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    denominator: float,
    rollout_sample_ids: torch.Tensor | None = None,
    sequence_process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Average token losses within each sequence, then average sequences equally."""
    weights, _ = _sequence_loss_weights(response_mask, rollout_sample_ids, sequence_process_group)
    numerator = (values * weights.to(dtype=values.dtype)).sum()

    return numerator / denominator if denominator > 0 else torch.zeros_like(numerator)


def _sequence_loss_weights(
    response_mask: torch.Tensor,
    rollout_sample_ids: torch.Tensor | None = None,
    sequence_process_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts, valid = _sequence_id_counts(response_mask, rollout_sample_ids, sequence_process_group)
    weights = torch.zeros(response_mask.shape, dtype=torch.float32, device=response_mask.device)
    if counts.numel() == 0:
        return weights, torch.zeros((), dtype=torch.float32, device=response_mask.device)
    sequence_count = (counts > 0).to(dtype=torch.float32).sum()
    if not valid.any():
        return weights, sequence_count

    if rollout_sample_ids is not None:
        sequence_ids = rollout_sample_ids
    else:
        row_ids = torch.arange(response_mask.shape[0], device=response_mask.device, dtype=torch.long)
        sequence_ids = row_ids[:, None].expand_as(response_mask)

    flat_ids = sequence_ids[valid].long()
    weights[valid] = 1.0 / counts[flat_ids].clamp_min(1.0)
    return weights, sequence_count


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
