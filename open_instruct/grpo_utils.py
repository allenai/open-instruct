import enum
import math
import os
from dataclasses import dataclass, field
from typing import Literal

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
    truncated_importance_sampling_ratio_cap: float = 2.0
    """The maximum cap for truncated importance sampling ratio (0 means disabled)"""
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
        if self.loss_denominator != "token" and float(self.loss_denominator) <= 0:
            raise ValueError(
                f"loss_denominator must be a valid float greater than 0 if not 'token', got: {self.loss_denominator}"
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
    tis_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.loss_fn == GRPOLossType.dapo:
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_lower, 1.0 + config.clip_higher)
    elif config.loss_fn == GRPOLossType.cispo:
        # cispo: directly clip ratio, no lower bound.
        # reinforce loss, so multiply by new logprobs
        pg_losses = -advantages * torch.clamp(ratio.detach(), max=1.0 + config.clip_higher) * new_logprobs
        pg_losses2 = pg_losses
    else:
        raise ValueError(f"Invalid loss function: {config.loss_fn}")

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


def forward_for_logprobs(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    position_ids: torch.Tensor,
    pad_token_id: int,
    temperature: float,
    return_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward pass to compute log probabilities."""
    output = model(input_ids=query_responses, attention_mask=None, position_ids=position_ids)
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
            attention_masks = [data_BT.attention_masks[i] for i in batch_indices]
            position_ids = [data_BT.position_ids[i] for i in batch_indices]
            shapes = [tuple(t.shape) for t in query_responses]

            if len(set(shapes)) != 1:
                for i in batch_indices:
                    single_logprobs, _ = forward_for_logprobs(
                        model, data_BT.query_responses[i], data_BT.position_ids[i], pad_token_id, temperature, False
                    )

                    response_mask_BT = data_BT.response_masks[i]
                    single_logprobs = mask_logprobs(single_logprobs, response_mask_BT[:, 1:].bool())
                    logprobs_BT.append(single_logprobs)
                continue

            batch_query_responses = torch.cat(query_responses, dim=0)
            batch_attention_masks = torch.cat(attention_masks, dim=0)
            batch_position_ids = torch.cat(position_ids, dim=0)

            batch_logprobs, _ = forward_for_logprobs(
                model, batch_query_responses, batch_position_ids, pad_token_id, temperature, False
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
