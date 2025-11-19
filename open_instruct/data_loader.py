import tempfile
from collections.abc import Iterator
from typing import Any

from datasets import Dataset
from olmo_core.data import data_loader


class HFDataLoader(data_loader.DataLoaderBase):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        work_dir: str | None = None,
    ):
        if work_dir is None:
            work_dir = tempfile.gettempdir()

        super().__init__(
            work_dir=work_dir, global_batch_size=batch_size, dp_world_size=world_size, dp_rank=rank, fs_local_rank=0
        )

        dataset = dataset.map(lambda example, idx: example | {"dataset_index": idx}, with_indices=True)
        self.original_dataset = dataset.shard(num_shards=world_size, index=rank)
        self.seed = seed

        self.dataset = self.original_dataset.shuffle(seed=seed)
        self.effective_size = len(self.dataset) - (len(self.dataset) % batch_size)

    def _iter_batches(self) -> Iterator[dict[str, Any]]:
        for i in range(self.batches_processed, self.effective_size):
            yield self.dataset[i]

    @property
    def total_batches(self) -> int:
        return self.effective_size

    @property
    def epoch_number(self) -> int:
        return self._epoch if self._epoch is not None else 0

    def exclude_index(self, index: int) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "batches_processed": self.batches_processed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch = state["epoch"]
        self.batches_processed = state["batches_processed"]
        if self._epoch is not None:
            self.dataset = self.original_dataset.shuffle(seed=self.seed + self._epoch)

    def reshuffle(self, epoch: int | None = None, **kwargs):
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch
        self.batches_processed = 0
        self.dataset = self.original_dataset.shuffle(seed=self.seed + epoch)

    def get_mock_batch(self) -> dict[str, Any]:
        return self.dataset[0]
