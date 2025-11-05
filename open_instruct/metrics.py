import torch


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: int | None = None, denominator: float | None = None
) -> torch.Tensor:
    """Compute the mean of tensor values considering only masked (valid) positions.

    Args:
        values: Input tensor to compute mean over
        mask: Boolean or binary mask tensor (same shape as values) indicating valid positions
        axis: Axis along which to sum before computing mean, or None for all axes
        denominator: Optional fixed denominator to use instead of mask.sum(). Useful when
            the denominator should be consistent across batches.

    Returns:
        Masked mean of the input values as a scalar tensor
    """
    numerator = (values * mask).sum(axis=axis)
    denom = mask.sum(axis=axis) if denominator is None else denominator
    return (numerator / denom).mean()


class LossStatistics:
    """Accumulates training statistics across minibatches for GRPO training.

    Tracks KL divergence estimates, policy gradient losses, clipping statistics,
    and importance ratios across multiple minibatches. Provides methods to update
    statistics and convert accumulated values to a metrics dictionary.
    """

    def __init__(self, num_batches: int, record_entropy: bool = False):
        """Initialize loss statistics storage.

        Args:
            num_batches: Number of minibatches to track statistics for
            record_entropy: Whether to track policy entropy statistics
        """
        self.kl_stats = torch.zeros(4, num_batches)
        self.kl_loss_stats = torch.zeros(num_batches)
        self.pg_clipfrac_stats = torch.zeros(num_batches)
        self.pg_loss_stats = torch.zeros(num_batches)
        self.loss_stats = torch.zeros(num_batches)
        self.ratio_stats = torch.zeros(num_batches)
        self.entropy_stats = torch.zeros(num_batches) if record_entropy else None

    def update_kl_estimates(self, i, ref_logprobs_diff, ratio, mb_response_masks_bool, args):
        """Compute and store KL divergence estimates for a minibatch.

        Computes four different KL estimators (kl1-kl4) based on log probability
        differences between the current policy and reference policy.

        Args:
            i: Minibatch index
            ref_logprobs_diff: Log probability differences (new - ref) [batch, seq_len]
            ratio: Importance ratio exp(new_logprobs - old_logprobs) [batch, seq_len]
            mb_response_masks_bool: Boolean mask for valid response tokens [batch, seq_len]
            args: Training arguments containing kl_estimator, masked_mean settings

        Returns:
            KL divergence values for the selected estimator (shape: [batch, seq_len])
        """
        kl_values = torch.stack(
            [
                ref_logprobs_diff,
                ref_logprobs_diff**2 / 2,
                torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff,
                ratio * ref_logprobs_diff,
            ]
        )

        vmapped_fn = torch.vmap(
            lambda v: masked_mean(v, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator)
        )
        self.kl_stats[:, i] = vmapped_fn(kl_values).float()

        kl_idx = {"kl1": 0, "kl2": 1, "kl3": 2, "kl4": 3}[args.kl_estimator]
        return kl_values[kl_idx]

    def update_stats(
        self, i, mb_response_masks_bool, pg_losses, pg_losses2, pg_loss_max, ratio, loss, mb_entropy, args
    ):
        """Update all training statistics for a minibatch.

        Args:
            i: Minibatch index
            mb_response_masks_bool: Boolean mask for valid response tokens [batch, seq_len]
            pg_losses: Unclipped policy gradient losses [batch, seq_len]
            pg_losses2: Clipped policy gradient losses [batch, seq_len]
            pg_loss_max: Element-wise max of pg_losses and pg_losses2 [batch, seq_len]
            ratio: Importance ratio [batch, seq_len]
            loss: Total loss value (scalar)
            mb_entropy: Policy entropy [batch, seq_len]
            args: Training arguments containing beta, record_entropy, masked_mean settings
        """
        kl_idx = {"kl1": 0, "kl2": 1, "kl3": 2, "kl4": 3}[args.kl_estimator]
        self.kl_loss_stats[i] = self.kl_stats[kl_idx, i] * args.beta
        self.pg_clipfrac_stats[i] = masked_mean(
            (pg_losses2 > pg_losses).float(),
            mb_response_masks_bool,
            args.masked_mean_axis,
            args.masked_mean_denominator,
        )
        self.pg_loss_stats[i] = masked_mean(
            pg_loss_max, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
        )
        self.loss_stats[i] = loss
        self.ratio_stats[i] = masked_mean(
            ratio, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
        )
        if args.record_entropy and self.entropy_stats is not None:
            self.entropy_stats[i] = masked_mean(
                mb_entropy, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
            ).float()

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert accumulated statistics to a metrics dictionary.

        Returns:
            Dictionary mapping metric names to their averaged values across all minibatches
        """
        metrics = {
            "objective/kl_avg": self.kl_stats[0].mean(),
            "objective/kl2_avg": self.kl_stats[1].mean(),
            "objective/kl3_avg": self.kl_stats[2].mean(),
            "objective/kl4_avg": self.kl_stats[3].mean(),
            "loss/policy_avg": self.pg_loss_stats.mean(),
            "loss/kl_avg": self.kl_loss_stats.mean(),
            "loss/total_avg": self.loss_stats.mean(),
            "policy/clipfrac_avg": self.pg_clipfrac_stats.mean(),
            "val/ratio": self.ratio_stats.mean(),
            "val/ratio_var": self.ratio_stats.var(),
        }
        if self.entropy_stats is not None:
            metrics["policy/entropy_avg"] = self.entropy_stats.mean()
        return metrics


class MetricsTracker:
    """Preallocated tensor-based metrics storage for efficient distributed reduction.

    Stores all metrics in a single preallocated tensor to enable efficient all-reduce
    operations in distributed training. Maintains a mapping from metric names to
    tensor indices for fast access.
    """

    def __init__(self, max_metrics: int = 32, device: str = "cuda"):
        """Initialize metrics tracker.

        Args:
            max_metrics: Maximum number of unique metrics to track
            device: Device to allocate metrics tensor on (default: "cuda")
        """
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def add(self, name: str, value: torch.tensor):
        """Add or update a metric value.

        If the metric name is new, allocates a new index in the metrics tensor.
        If the metric already exists, updates its value at the existing index.

        Args:
            name: Metric name (e.g., "loss/policy_avg")
            value: Metric value (scalar tensor or convertible to tensor)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If max_metrics limit is exceeded
        """
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1

        self.metrics[self.names2idx[name]] = value
        return self

    def add_dict(self, metrics_dict: dict[str, torch.Tensor]):
        """Add multiple metrics from a dictionary.

        Args:
            metrics_dict: Dictionary mapping metric names to values

        Returns:
            Self for method chaining
        """
        for k, v in metrics_dict.items():
            self.add(k, v)
        return self

    def get_metrics_list(self) -> dict[str, float]:
        """Convert tracked metrics to a dictionary of Python floats.

        Returns:
            Dictionary mapping metric names to their float values
        """
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}
