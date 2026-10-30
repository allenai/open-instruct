"""Bridge replaceable prompt selection to MILES' existing buffered group API."""

import copy
from pathlib import Path

import torch
from miles.utils.types import Sample

from miles.rollout.data_source import RolloutDataSourceWithBuffer, compute_global_dataset_state_path
from open_instruct.miles.rollout import prompt_provider, remind_provider


class ProviderRolloutDataSource(RolloutDataSourceWithBuffer):
    """Keep dataset loading and buffering; delegate fresh prompts to a provider.

    The outer Open Instruct wrapper owns locking and atomic checkpoints with the
    pending-group ledger. Dataset size remains available for MILES' run accounting;
    the provider interface itself does not require epochs or quotas.
    """

    def __init__(self, args, *, provider: prompt_provider.PromptProvider | None = None):
        super().__init__(args)
        core = getattr(args, "olmo_core", None)
        if provider is not None:
            self.provider = provider
        elif getattr(core, "prompt_provider", "shuffle") == "remind":
            self.provider = remind_provider.ReMindPromptProvider(
                lambda: self.dataset,
                shuffle=bool(args.rollout_shuffle),
                seed=args.rollout_seed,
                review_fraction=core.remind_review_fraction,
                enqueue_probability=core.remind_enqueue_probability,
            )
        else:
            self.provider = prompt_provider.ShuffledPromptProvider(
                lambda: self.dataset, shuffle=bool(getattr(args, "rollout_shuffle", False))
            )
        self._sync_legacy_cursor()

    def _sync_legacy_cursor(self):
        # These compatibility fields describe fresh-data traversal, not reviews.
        provider = (
            self.provider.base if isinstance(self.provider, remind_provider.ReMindPromptProvider) else self.provider
        )
        if isinstance(provider, prompt_provider.ShuffledPromptProvider):
            self.epoch_id = provider.epoch_id
            self.sample_offset = provider.sample_offset

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if type(num_samples) is not int or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        buffered = self.buffer.copy()
        counters = self.sample_group_index, self.sample_index
        provider_state = None
        try:
            groups = self._get_samples_from_buffer(num_samples)
            if len(groups) > num_samples:
                raise ValueError(f"buffer_filter_path returned {len(groups)} groups; requested {num_samples}")
            if len(groups) == num_samples:
                return groups
            provider_state = copy.deepcopy(self.provider.state_dict())
            prompts = self.provider.next_samples(num_samples - len(groups))
            if len(prompts) != num_samples - len(groups):
                raise ValueError("Prompt provider must return exactly the requested number of prompts")
            for prompt in prompts:
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt)
                    sample.group_index = self.sample_group_index
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                self.sample_group_index += 1
                groups.append(group)
        except Exception:
            # A delivery is all-or-nothing, including buffered retries, provider
            # selection state, and IDs assigned during response-group expansion.
            self.buffer[:] = buffered
            self.sample_group_index, self.sample_index = counters
            if provider_state is not None:
                self.provider.load_state_dict(provider_state)
            raise
        self._sync_legacy_cursor()
        return groups

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset or not self.args.load:
            return
        path = Path(compute_global_dataset_state_path(self.args.load, rollout_id=rollout_id))
        if not path.exists():
            return  # The outer wrapper distinguishes a fresh model from unsafe resume.
        state = torch.load(path, map_location="cpu", weights_only=True)
        provider_state = state.get("prompt_provider")
        if provider_state is None:
            if not isinstance(self.provider, prompt_provider.ShuffledPromptProvider):
                raise ValueError("A custom prompt provider cannot restore a legacy shuffle cursor")
            provider_state = self.provider.state_dict() | {
                "epoch_id": state.get("epoch_id", 0),
                "sample_offset": state.get("sample_offset", 0),
            }
        self.provider.load_state_dict(provider_state)
        for key in ("epoch_id", "sample_offset", "sample_group_index", "sample_index", "metadata"):
            if key in state:
                setattr(self, key, state[key])
        # Provider state is authoritative, including in older ReMind checkpoints
        # whose top-level cursor fields were left at zero.
        self._sync_legacy_cursor()
