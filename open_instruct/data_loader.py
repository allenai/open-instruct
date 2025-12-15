from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from datasets import Dataset
from olmo_core.data import data_loader


def to_device(batch: dict[str, Any], device: torch.device | None) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class HFDataLoader(data_loader.DataLoaderBase):
    """A DataLoader that wraps a HuggingFace Dataset for use with olmo_core's Trainer.

    This class implements the DataLoaderBase interface, providing iteration over
    a HuggingFace Dataset with support for sharding across distributed workers,
    shuffling, checkpointing, and optional collation.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
        work_dir: str,
        automatic_reshuffle: bool = False,
        collator: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
        fs_local_rank: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        if fs_local_rank is None:
            fs_local_rank = rank
        super().__init__(
            work_dir=work_dir,
            global_batch_size=batch_size,
            dp_world_size=world_size,
            dp_rank=rank,
            fs_local_rank=fs_local_rank,
        )

        self._full_dataset = dataset.map(lambda example, idx: example | {"dataset_index": idx}, with_indices=True)
        self.seed = seed
        self._batch_size = batch_size
        self._per_rank_batch_size = batch_size // world_size
        self._collator = collator if collator is not None else (lambda x: x)
        self._automatic_reshuffle = automatic_reshuffle
        self._excluded_indices: set[int] = set()
        self._epoch: int = 0
        self._current_iter: Iterator[dict[str, Any]] | None = None
        self._device = device

        self._reshard(epoch=0)

    def __next__(self) -> dict[str, Any]:
        if self._current_iter is None:
            self._current_iter = iter(self)
        try:
            return next(self._current_iter)
        except StopIteration:
            self._current_iter = None
            if self._automatic_reshuffle:
                self.reshuffle()
                if self.effective_size == 0:
                    raise RuntimeError("All dataset examples have been excluded. Cannot continue iteration.") from None
                self._current_iter = iter(self)
                return next(self._current_iter)
            self._epoch += 1
            self.batches_processed = 0
            raise

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        """Return an iterable over all batches in the epoch."""
        start_example = self.batches_processed * self._per_rank_batch_size
        batch_examples: list[dict[str, Any]] = []
        for i in range(start_example, self.effective_size):
            example = self.dataset[i]
            batch_examples.append(example | {"prompt_id": f"{self._epoch}_{example['dataset_index']}"})
            if len(batch_examples) == self._per_rank_batch_size:
                yield to_device(self._collator(batch_examples), self._device)
                batch_examples = []

    @property
    def total_batches(self) -> int:
        """Return the total number of batches in an epoch."""
        return self.effective_size // self._per_rank_batch_size

    def state_dict(self) -> dict[str, Any]:
        """Return a state dictionary for checkpointing."""
        return {
            "epoch": self._epoch,
            "batches_processed": self.batches_processed,
            "excluded_indices": list(self._excluded_indices),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a state dictionary to restore the data loader's state."""
        self._excluded_indices = set(state_dict.get("excluded_indices", []))
        # Set epoch to one less than target since reshuffle() increments it
        self._epoch = state_dict["epoch"] - 1
        self.reshuffle()
        assert self._epoch == state_dict["epoch"]
        self.batches_processed = state_dict["batches_processed"]
        self._current_iter = None

    def exclude_index(self, index: int) -> None:
        """Exclude a dataset index from future iterations.

        Args:
            index: The dataset_index to exclude.
        """
        self._excluded_indices.add(index)

    def reshuffle(self, epoch: int | None = None, **kwargs: Any) -> None:
        """Reshuffle and reshard the dataset for a new epoch.

        Args:
            epoch: The epoch number to use for shuffling seed. If None, increments internal counter.
            **kwargs: Additional keyword arguments (unused, for API compatibility).
        """
        self._epoch = self._epoch + 1 if epoch is None else epoch
        self.batches_processed = 0
        self._reshard(self._epoch)

    def _reshard(self, epoch: int) -> None:
        """Reshard the dataset for a given epoch."""
        shuffled = self._full_dataset.shuffle(seed=self.seed + epoch)
        filtered = shuffled.filter(lambda x: x["dataset_index"] not in self._excluded_indices)
        global_size = len(filtered)
        total_batches = global_size // self._batch_size
        self.effective_size = total_batches * self._per_rank_batch_size
        self.dataset = filtered.shard(num_shards=self.dp_world_size, index=self.dp_rank)

    def get_mock_batch(self) -> dict[str, Any]:
        """Return a batch with arbitrary data for dry-run testing.

        Used by the trainer to do a dry-run of the
        forward and backward pass before training officially starts.
        """
        num_examples = min(self._per_rank_batch_size, len(self.dataset))
        examples = [self.dataset[i] for i in range(num_examples)]
        return to_device(self._collator(examples), self._device)

    def global_num_tokens_in_batch(self, batch: dict[str, Any]) -> int | None:
        """Return the total number of tokens in the batch across all ranks."""
        num_tokens = 0
        for key, value in batch.items():
            if "input_ids" in key and hasattr(value, "numel"):
                num_tokens += value.numel()
        if num_tokens == 0:
            return None
        return num_tokens * self.dp_world_size
