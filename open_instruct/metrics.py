import torch

# TODO: Add docstrings to MetricsTracker, LossStatistics, masked_mean, and all methods
#       added in this refactoring branch (compare_vllm_logprobs_to_local,
#       maybe_apply_importance_sampling, calculate_loss)


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: int | None = None, denominator: float | None = None
) -> torch.Tensor:
    numerator = (values * mask).sum(axis=axis)
    denom = mask.sum(axis=axis) if denominator is None else denominator
    return (numerator / denom).mean()


class LossStatistics:
    def __init__(self, num_batches: int, record_entropy: bool = False):
        self.kl_stats = torch.zeros(4, num_batches)
        self.kl_loss_stats = torch.zeros(num_batches)
        self.pg_clipfrac_stats = torch.zeros(num_batches)
        self.pg_loss_stats = torch.zeros(num_batches)
        self.loss_stats = torch.zeros(num_batches)
        self.ratio_stats = torch.zeros(num_batches)
        self.entropy_stats = torch.zeros(num_batches) if record_entropy else None

    def update_kl_estimates(self, i, ref_logprobs_diff, ratio, mb_response_masks_bool, args):
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


class MetricsTracker:
    """A simple class to prellocate all metrics in an array
    so we can do only one allreduce operation to get the metrics mean"""

    def __init__(self, max_metrics: int = 32, device: str = "cuda"):
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def add(self, name: str, value: torch.tensor):
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1

        self.metrics[self.names2idx[name]] = value
        return self

    def get_metrics_list(self) -> dict[str, float]:
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}
