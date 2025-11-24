from collections.abc import Iterable, Iterator
from typing import Any

from datasets import Dataset
from olmo_core.data import data_loader


class HFDataLoader(data_loader.DataLoaderBase):
    """A DataLoader that wraps a HuggingFace Dataset for use with olmo_core's Trainer.

    This class implements the DataLoaderBase interface, providing iteration over
    a HuggingFace Dataset with support for sharding across distributed workers,
    shuffling, checkpointing, and excluding specific indices.
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
        self._exclude_set: set[int] = set()
        self.effective_size = len(self.dataset) - (len(self.dataset) % batch_size)
        self._current_iter: Iterator[dict[str, Any]] | None = None

    def next_item(self) -> dict[str, Any]:
        """Return the next item from the dataset and increment the batch counter.

        Note: This method is not part of the DataLoaderBase API. It provides a way to
        manually iterate through the dataset one item at a time while tracking progress.

        Returns:
            The next item from the dataset as a dictionary.

        Raises:
            StopIteration: When the dataset has been exhausted.
        """
        if self._current_iter is None:
            self._current_iter = self._iter_batches()
        item = next(self._current_iter)
        self.batches_processed += 1
        return item

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        """Return an iterable over all batches in the epoch.

        Part of the DataLoaderBase API. This method yields individual items from the
        dataset, starting from the current batch position.

        Returns:
            An iterable over dataset items for the current epoch.
        """
        for i in range(self.batches_processed, self.effective_size):
            yield self.dataset[i]

    @property
    def total_batches(self) -> int:
        """Return the total number of batches in an epoch.

        Part of the DataLoaderBase API.

        Returns:
            The total number of batches available in the dataset.
        """
        return self.effective_size

    @property
    def epoch_number(self) -> int:
        """Return the current epoch number, defaulting to 0 if not set.

        Note: This method is not part of the DataLoaderBase API. It differs from the
        base class's `epoch` property in that it returns 0 instead of raising an error
        when the epoch has not been set.

        Returns:
            The current epoch number, or 0 if no epoch has been set.
        """
        return self._epoch if self._epoch is not None else 0

    def exclude_index(self, index: int) -> None:
        """Mark a dataset index to be excluded from future iterations.

        Note: This method is not part of the DataLoaderBase API. Excluded indices
        will be filtered out when reshuffle() or load_state_dict() is called.

        Args:
            index: The dataset index to exclude.
        """
        self._exclude_set.add(index)

    def state_dict(self) -> dict[str, Any]:
        """Return a state dictionary for checkpointing.

        Part of the DataLoaderBase API.

        Returns:
            A dictionary containing the epoch, batches_processed, and exclude_set.
        """
        return {
            "epoch": self._epoch,
            "batches_processed": self.batches_processed,
            "exclude_set": list(self._exclude_set),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load a state dictionary to restore the data loader's state.

        Part of the DataLoaderBase API.

        Args:
            state: A state dictionary from state_dict().
        """
        self._epoch = state["epoch"]
        self.batches_processed = state["batches_processed"]
        self._exclude_set = set(state.get("exclude_set", []))
        if self._epoch is not None:
            self._apply_exclude_and_shuffle(self.seed + self._epoch)
        self._current_iter = None

    def _apply_exclude_and_shuffle(self, seed: int) -> None:
        """Apply exclusions and reshuffle the dataset.

        Note: This method is not part of the DataLoaderBase API. It is an internal
        helper method used by reshuffle() and load_state_dict().

        Args:
            seed: The random seed to use for shuffling.
        """
        if self._exclude_set:
            self.dataset = self.dataset.filter(lambda x: x["dataset_index"] not in self._exclude_set)
        self.dataset = self.dataset.shuffle(seed=seed)
        self.effective_size = len(self.dataset) - (len(self.dataset) % self._batch_size)

    def reshuffle(self, epoch: int | None = None, **kwargs: Any) -> None:
        """Reshuffle the dataset for a new epoch.

        Part of the DataLoaderBase API. Should be called before starting each epoch.

        Args:
            epoch: The epoch number. If None, increments from the current epoch
                (or starts at 1 if no epoch has been set).
            **kwargs: Additional keyword arguments (unused, for API compatibility).
        """
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch
        self.batches_processed = 0
        self._apply_exclude_and_shuffle(self.seed + epoch)
        self._current_iter = None

    def get_mock_batch(self) -> dict[str, Any]:
        """Return a batch with arbitrary data for dry-run testing.

        Part of the DataLoaderBase API. Used by the trainer to do a dry-run of the
        forward and backward pass before training officially starts.

        Returns:
            The first item from the dataset.
        """
        return self.dataset[0]
