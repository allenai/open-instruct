from collections.abc import Iterable, Iterator
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
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int,
        rank: int,
        world_size: int,
        work_dir: str,
        automatic_reshuffle: bool = False,
    ) -> None:
        """Initialize the HFDataLoader.

        Args:
            dataset: The HuggingFace Dataset to load data from.
            batch_size: The global batch size.
            seed: Random seed for shuffling.
            rank: The rank of the current process in the distributed setup.
            world_size: Total number of processes in the distributed setup.
            work_dir: Working directory for the data loader (required by DataLoaderBase).
            automatic_reshuffle: If True, automatically reshuffle at epoch boundaries.
        """
        super().__init__(
            work_dir=work_dir, global_batch_size=batch_size, dp_world_size=world_size, dp_rank=rank, fs_local_rank=0
        )

        dataset_with_indices = dataset.map(lambda example, idx: example | {"dataset_index": idx}, with_indices=True)
        self._original_dataset = dataset_with_indices.shard(num_shards=world_size, index=rank)
        self.dataset = self._original_dataset.shuffle(seed=seed)
        self.seed = seed
        self._batch_size = batch_size
        self.effective_size = len(self.dataset) - (len(self.dataset) % batch_size)
        self._automatic_reshuffle = automatic_reshuffle
        self._excluded_indices: set[int] = set()
        self._epoch: int = 0
        self._current_iter: Iterator[dict[str, Any]] | None = None

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
        for i in range(self.batches_processed, self.effective_size):
            example = self.dataset[i]
            yield example | {"prompt_id": f"{self._epoch}_{example['dataset_index']}"}

    @property
    def total_batches(self) -> int:
        """Return the total number of batches in an epoch."""
        return self.effective_size // self._batch_size

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
        """Reshuffle the dataset for a new epoch.

        Args:
            epoch: The epoch number (unused, for API compatibility).
            **kwargs: Additional keyword arguments (unused, for API compatibility).
        """
        self._epoch += 1
        self.batches_processed = 0
        shuffled = self._original_dataset.shuffle(seed=self.seed + self._epoch)
        # If this is slow, we can speed it up by making this a boolean mask.
        self.dataset = shuffled.filter(lambda x: x["dataset_index"] not in self._excluded_indices)
        self.effective_size = len(self.dataset) - (len(self.dataset) % self._batch_size)

    def get_mock_batch(self) -> dict[str, Any]:
        """Return a batch with arbitrary data for dry-run testing.

        Used by the trainer to do a dry-run of the
        forward and backward pass before training officially starts.

        Returns:
            The first item from the dataset.
        """
        return self.dataset[0]
