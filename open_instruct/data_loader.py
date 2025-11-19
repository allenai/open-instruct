from collections.abc import Iterator
from typing import Any

from datasets import Dataset
from olmo_core.data import data_loader


class HFDataLoader(data_loader.DataLoaderBase):
    def __init__(self, dataset: Dataset, batch_size: int, seed: int, rank: int, world_size: int, work_dir: str):
        super().__init__(
            work_dir=work_dir, global_batch_size=batch_size, dp_world_size=world_size, dp_rank=rank, fs_local_rank=0
        )

        self._source = dataset.map(lambda example, idx: example | {"dataset_index": idx}, with_indices=True)
        self._rank = rank
        self._world_size = world_size
        self.seed = seed
        self._batch_size = batch_size
        self._exclude_list: list[int] = []

        self.dataset = self._source.shard(num_shards=world_size, index=rank).shuffle(seed=seed)
        self.effective_size = len(self.dataset) - (len(self.dataset) % batch_size)
        self._current_iter = None

    def next_item(self) -> dict[str, Any]:
        if self._current_iter is None:
            self._current_iter = self._iter_batches()
        item = next(self._current_iter)
        self.batches_processed += 1
        return item

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
        self._exclude_list.append(index)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "batches_processed": self.batches_processed, "exclude_list": self._exclude_list}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch = state["epoch"]
        self.batches_processed = state["batches_processed"]
        self._exclude_list = state.get("exclude_list", [])
        if self._epoch is not None:
            self._apply_exclude_and_shuffle(self.seed + self._epoch)
        self._current_iter = None

    def _apply_exclude_and_shuffle(self, seed: int) -> None:
        sharded = self._source.shard(num_shards=self._world_size, index=self._rank)
        if self._exclude_list:
            exclude_set = set(self._exclude_list)
            filtered = sharded.filter(lambda x: x["dataset_index"] not in exclude_set)
            self.dataset = filtered.shuffle(seed=seed)
        else:
            self.dataset = sharded.shuffle(seed=seed)
        self.effective_size = len(self.dataset) - (len(self.dataset) % self._batch_size)

    def reshuffle(self, epoch: int | None = None, **kwargs):
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch
        self.batches_processed = 0
        self._apply_exclude_and_shuffle(self.seed + epoch)
        self._current_iter = None

    def get_mock_batch(self) -> dict[str, Any]:
        return self.dataset[0]
