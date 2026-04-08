from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from open_instruct import data_types


@dataclass
class ProcessedResult:
    """All list fields have length generation_config.n (one per sample)."""

    result: data_types.GenerationResult
    queries: list[list[int]]
    ground_truths: list[list[int]]
    datasets: list[str]
    raw_queries: list[str]
    active_tools: list[list[str] | None]
    decoded_responses: list[str]
    reward_scores: list[float]
    reward_metrics: dict[str, Any]
    percent_solved: float


@dataclass
class ItemMetadata:
    insert_order: int
    sample_count: int
    priority: float


class Selector(Protocol):
    def select(self, keys: list[str], metadata: dict[str, ItemMetadata], n: int) -> list[str]: ...


class Uniform:
    def select(self, keys: list[str], metadata: dict[str, ItemMetadata], n: int) -> list[str]:
        n = min(n, len(keys))
        indices = np.random.choice(len(keys), size=n, replace=False)
        return [keys[i] for i in indices]


class Prioritized:
    def select(self, keys: list[str], metadata: dict[str, ItemMetadata], n: int) -> list[str]:
        n = min(n, len(keys))
        weights = np.array([metadata[k].priority for k in keys], dtype=np.float64)
        total = weights.sum()
        probs = np.ones(len(keys)) / len(keys) if total == 0 else weights / total
        indices = np.random.choice(len(keys), size=n, replace=False, p=probs)
        return [keys[i] for i in indices]


class Fifo:
    def select(self, keys: list[str], metadata: dict[str, ItemMetadata], n: int) -> list[str]:
        n = min(n, len(keys))
        sorted_keys = sorted(keys, key=lambda k: metadata[k].insert_order)
        return sorted_keys[:n]


class Lifo:
    def select(self, keys: list[str], metadata: dict[str, ItemMetadata], n: int) -> list[str]:
        n = min(n, len(keys))
        sorted_keys = sorted(keys, key=lambda k: metadata[k].insert_order, reverse=True)
        return sorted_keys[:n]


class RateLimiter(Protocol):
    def can_sample(self, table: Table) -> bool: ...


@dataclass
class MinSize:
    min_size: int

    def can_sample(self, table: Table) -> bool:
        return len(table._data) >= self.min_size


_SELECTORS: dict[str, type[Selector]] = {"uniform": Uniform, "prioritized": Prioritized, "fifo": Fifo, "lifo": Lifo}


@dataclass
class ReplayBufferConfig:
    capacity: int | None = None
    """Max items in replay buffer. None = global_batch_size (FIFO-equivalent default)."""
    sampler: Literal["uniform", "prioritized", "fifo", "lifo"] = "uniform"
    """Sampling strategy: 'uniform', 'prioritized', 'fifo', 'lifo'."""
    remover: Literal["uniform", "prioritized", "fifo", "lifo"] = "fifo"
    """Removal strategy when buffer is full: 'uniform', 'prioritized', 'fifo', 'lifo'."""
    max_times_sampled: int = 1
    """Evict items after being sampled this many times. 1 = each item used once (default)."""
    min_size: int | None = None
    """Min items before sampling is allowed. None = global_batch_size."""


def make_selector(name: str) -> Selector:
    cls = _SELECTORS.get(name)
    if cls is None:
        raise ValueError(f"Unknown selector: {name!r}. Options: {list(_SELECTORS)}")
    return cls()


class Table:
    def __init__(
        self,
        max_size: int,
        sampler: Selector,
        remover: Selector,
        max_times_sampled: int = 1,
        rate_limiter: RateLimiter | None = None,
    ):
        self._max_size = max_size
        self._sampler = sampler
        self._remover = remover
        self._max_times_sampled = max_times_sampled
        self._rate_limiter: RateLimiter = rate_limiter if rate_limiter is not None else MinSize(1)
        self._lock = threading.Lock()
        self._can_sample = threading.Condition(self._lock)
        self._data: dict[str, ProcessedResult] = {}
        self._metadata: dict[str, ItemMetadata] = {}
        self._insert_counter: int = 0
        self._shutdown: bool = False

    def insert(self, key: str, data: ProcessedResult, priority: float = 1.0) -> None:
        with self._can_sample:
            self._data[key] = data
            self._metadata[key] = ItemMetadata(insert_order=self._insert_counter, sample_count=0, priority=priority)
            self._insert_counter += 1
            self._evict_overflow()
            self._can_sample.notify_all()

    def sample(self, n: int) -> list[ProcessedResult] | None:
        with self._can_sample:
            while not self._rate_limiter.can_sample(self):
                if self._shutdown:
                    return None
                self._can_sample.wait()
            if self._shutdown:
                return None
            keys = list(self._data.keys())
            selected_keys = self._sampler.select(keys, self._metadata, n)
            results = [self._data[k] for k in selected_keys]
            for k in selected_keys:
                self._metadata[k].sample_count += 1
            self._evict_oversampled(selected_keys)
            return results

    def shutdown(self) -> None:
        with self._can_sample:
            self._shutdown = True
            self._can_sample.notify_all()

    def _evict_overflow(self) -> None:
        excess = len(self._data) - self._max_size
        if excess <= 0:
            return
        keys = list(self._data.keys())
        to_remove = self._remover.select(keys, self._metadata, excess)
        for k in to_remove:
            del self._data[k]
            del self._metadata[k]

    def _evict_oversampled(self, candidates: list[str]) -> None:
        for k in candidates:
            if self._metadata[k].sample_count >= self._max_times_sampled:
                del self._data[k]
                del self._metadata[k]

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown
