"""Bridge replaceable prompt selection to MILES' existing buffered group API."""

import copy
from pathlib import Path

import torch
from miles.utils.types import Sample

from miles.rollout import data_source
from open_instruct.miles.rollout import prompt_provider


class ProviderRolloutDataSource(data_source.RolloutDataSourceWithBuffer):
    """Keep dataset loading and buffering; delegate fresh prompts to a provider.

    The outer Open Instruct wrapper owns locking and atomic checkpoints with the
    pending-group ledger. Dataset size remains available for MILES' run accounting;
    the provider interface itself does not require epochs or quotas.
    """

    def __init__(self, args, *, provider: prompt_provider.PromptProvider | None = None):
        super().__init__(args)
        self.provider = (
            provider
            if provider is not None
            else prompt_provider.ShuffledPromptProvider(
                self.dataset, shuffle=bool(getattr(args, "rollout_shuffle", False))
            )
        )

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if type(num_samples) is not int or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        groups = self._get_samples_from_buffer(num_samples)
        for prompt in self.provider.next_samples(num_samples - len(groups)):
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        if isinstance(self.provider, prompt_provider.ShuffledPromptProvider):
            # Preserve the legacy cursor fields for existing checkpoint readers.
            self.epoch_id = self.provider.epoch_id
            self.sample_offset = self.provider.sample_offset
        return groups

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset or not self.args.load:
            return
        path = Path(data_source.compute_global_dataset_state_path(self.args.load, rollout_id=rollout_id))
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
