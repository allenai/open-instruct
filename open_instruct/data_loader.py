from typing import Any

import numpy as np
from datasets import Dataset
from olmo_core.data import data_loader


class HFDataLoader(data_loader.TextDataLoaderBase):
    def __init__(self, dataset: Dataset, batch_size: int, seed: int, rank: int = 0, world_size: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch_number = 0
        self.index = 0

        indices = np.arange(len(dataset))
        shard_size = len(indices) // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else len(indices)
        self.shard_indices = indices[start_idx:end_idx]

        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.shard_indices)

        self.effective_size = len(self.shard_indices) - (len(self.shard_indices) % batch_size)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        if self.index >= self.effective_size:
            raise StopIteration

        idx = self.shard_indices[self.index]
        self.index += 1

        return self.dataset[int(idx)] | {"dataset_index": int(idx)}

    def reshuffle(self):
        self.epoch_number += 1
        self.index = 0
        self.rng.shuffle(self.shard_indices)

    def reset(self):
        self.index = 0

    @property
    def total_batches(self) -> int:
        return self.effective_size

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch_number": self.epoch_number,
            "index": self.index,
            "shard_indices": self.shard_indices.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch_number = state["epoch_number"]
        self.index = state["index"]
        self.shard_indices = state["shard_indices"].copy()
        self.rng.bit_generator.state = state["rng_state"]

    def _iter_batches(self):
        raise NotImplementedError("Use __next__ instead")

    def get_mock_batch(self) -> dict[str, Any]:
        return self.dataset[0]
