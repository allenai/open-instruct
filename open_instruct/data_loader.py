from collections.abc import Iterable
from typing import Any

from datasets import Dataset
from olmo_core.data import data_loader


class HFDataLoader(data_loader.DataLoaderBase):
    """A DataLoader that wraps a HuggingFace Dataset for use with olmo_core's Trainer.

    This class implements the DataLoaderBase interface, providing iteration over
    a HuggingFace Dataset with support for sharding across distributed workers,
    shuffling, and checkpointing.
    """

    def __init__(
        self, dataset: Dataset, batch_size: int, seed: int, rank: int, world_size: int, work_dir: str
    ) -> None:
        """Initialize the HFDataLoader.

        Args:
            dataset: The HuggingFace Dataset to load data from.
            batch_size: The global batch size.
            seed: Random seed for shuffling.
            rank: The rank of the current process in the distributed setup.
            world_size: Total number of processes in the distributed setup.
            work_dir: Working directory for the data loader (required by DataLoaderBase).
        """
        super().__init__(
            work_dir=work_dir, global_batch_size=batch_size, dp_world_size=world_size, dp_rank=rank, fs_local_rank=0
        )

        dataset_with_indices = dataset.map(lambda example, idx: example | {"dataset_index": idx}, with_indices=True)
        self.dataset = dataset_with_indices.shard(num_shards=world_size, index=rank).shuffle(seed=seed)
        self.seed = seed
        self._batch_size = batch_size
        self.effective_size = len(self.dataset) - (len(self.dataset) % batch_size)

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        """Return an iterable over all batches in the epoch."""
        epoch = self._epoch or 0
        for i in range(self.batches_processed, self.effective_size):
            example = self.dataset[i]
            example["prompt_id"] = f"{epoch}_{example['dataset_index']}"
            yield example

    @property
    def total_batches(self) -> int:
        """Return the total number of batches in an epoch."""
        return self.effective_size

    def state_dict(self) -> dict[str, float]:
        """Return a state dictionary for checkpointing."""
        return {"epoch": self._epoch, "batches_processed": self.batches_processed}

    def load_state_dict(self, state: dict[str, float]) -> None:
        """Load a state dictionary to restore the data loader's state."""
        self._epoch = state["epoch"]
        self.batches_processed = state["batches_processed"]
        if self._epoch is not None:
            self.dataset = self.dataset.shuffle(seed=self.seed + self._epoch)
            self.effective_size = len(self.dataset) - (len(self.dataset) % self._batch_size)

    def reshuffle(self, epoch: int | None = None, **kwargs: Any) -> None:
        """Reshuffle the dataset for a new epoch.

        Args:
            epoch: The epoch number. If None, increments from the current epoch
                (or starts at 1 if no epoch has been set).
            **kwargs: Additional keyword arguments (unused, for API compatibility).
        """
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch
        self.batches_processed = 0
        self.dataset = self.dataset.shuffle(seed=self.seed + epoch)
        self.effective_size = len(self.dataset) - (len(self.dataset) % self._batch_size)

    def get_mock_batch(self) -> dict[str, Any]:
        """Return a batch with arbitrary data for dry-run testing.

        Used by the trainer to do a dry-run of the
        forward and backward pass before training officially starts.

        Returns:
            The first item from the dataset.
        """
        return self.dataset[0]
