from collections.abc import Iterator
from typing import Any

from datasets import Dataset
from olmo_core.data import data_loader


class HFDataLoader(data_loader.TextDataLoaderBase):
    def __init__(self, dataset: Dataset, batch_size: int, seed: int, rank: int = 0, world_size: int = 1):
        dataset = dataset.map(lambda _, idx: {"dataset_index": idx}, with_indices=True)
        self.original_dataset = dataset.shard(num_shards=world_size, index=rank)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch_number = 0
        self.index = 0

        self.dataset = self.original_dataset.shuffle(seed=seed)
        self.effective_size = len(self.dataset) - (len(self.dataset) % batch_size)

    def _iter_batches(self) -> Iterator[dict[str, Any]]:
        while self.index < self.effective_size:
            item = self.dataset[self.index]
            self.index += 1
            yield item

    @property
    def total_batches(self) -> int:
        return self.effective_size

    def state_dict(self) -> dict[str, Any]:
        return {"epoch_number": self.epoch_number, "index": self.index}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch_number = state["epoch_number"]
        self.index = state["index"]
        self.dataset = self.original_dataset.shuffle(seed=self.seed + self.epoch_number)

    def reshuffle(self):
        self.epoch_number += 1
        self.index = 0
        self.dataset = self.original_dataset.shuffle(seed=self.seed + self.epoch_number)

    def get_mock_batch(self) -> dict[str, Any]:
        return self.dataset[0]
