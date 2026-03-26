"""Replay buffer for GRPO training, inspired by Google DeepMind's Reverb."""

import collections
import random
from dataclasses import dataclass
from enum import Enum


class SamplerType(str, Enum):
    FIFO = "fifo"
    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"


class EvictionType(str, Enum):
    FIFO = "fifo"
    AFTER_N_SAMPLES = "after_n_samples"


@dataclass
class ReplayGroup:
    group_id: str
    queries: list[list[int]]
    responses: list[list[int]]
    masks: list[list[int]]
    logprobs: list[list[float]]
    scores: list[float]
    finish_reasons: list[str]
    insertion_step: int
    sample_count: int = 0
    priority: float = 1.0
    dataset_name: str = ""
    raw_query: str = ""
    active_tools: list[str] | None = None


class SumTree:
    """Binary sum-tree for O(log n) proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data_indices: dict[str, int] = {}
        self.index_to_key: dict[int, str] = {}
        self._next_idx = 0
        self._size = 0

    def _propagate(self, idx: int):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent //= 2

    def total(self) -> float:
        return self.tree[1] if self._size > 0 else 0.0

    def add(self, key: str, priority: float):
        if key in self.data_indices:
            self.update(key, priority)
            return
        if self._size >= self.capacity:
            raise RuntimeError("SumTree is full")
        idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self.capacity
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        self.data_indices[key] = idx
        self.index_to_key[idx] = key
        self._propagate(tree_idx)
        self._size += 1

    def update(self, key: str, priority: float):
        idx = self.data_indices[key]
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)

    def remove(self, key: str):
        if key not in self.data_indices:
            return
        idx = self.data_indices[key]
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = 0.0
        self._propagate(tree_idx)
        del self.data_indices[key]
        del self.index_to_key[idx]
        self._size -= 1

    def sample_one(self, rng: random.Random) -> str:
        s = rng.uniform(0, self.total())
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return self.index_to_key[data_idx]

    def __len__(self) -> int:
        return self._size


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        sampler_type: SamplerType | str = SamplerType.FIFO,
        eviction_type: EvictionType | str = EvictionType.FIFO,
        max_sample_count: int = 1,
        seed: int = 42,
    ):
        self.capacity = capacity
        self.sampler_type = SamplerType(sampler_type)
        self.eviction_type = EvictionType(eviction_type)
        self.max_sample_count = max_sample_count
        self.rng = random.Random(seed)
        self._seed = seed

        self._groups: collections.OrderedDict[str, ReplayGroup] = collections.OrderedDict()
        self._sum_tree: SumTree | None = None
        if self.sampler_type == SamplerType.PRIORITIZED:
            self._sum_tree = SumTree(capacity)

    def insert(self, groups: list[ReplayGroup]) -> None:
        for group in groups:
            if group.group_id in self._groups:
                self._remove(group.group_id)
            self._groups[group.group_id] = group
            if self._sum_tree is not None:
                self._sum_tree.add(group.group_id, group.priority)
        self._evict_overflow()

    def sample(self, k: int) -> list[ReplayGroup]:
        if k > len(self._groups):
            raise ValueError(f"Cannot sample {k} groups from buffer of size {len(self._groups)}")
        if k == 0:
            return []

        if self.sampler_type == SamplerType.FIFO:
            sampled = self._sample_fifo(k)
        elif self.sampler_type == SamplerType.UNIFORM:
            sampled = self._sample_uniform(k)
        elif self.sampler_type == SamplerType.PRIORITIZED:
            sampled = self._sample_prioritized(k)
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")

        for group in sampled:
            group.sample_count += 1

        self._evict_sampled(sampled)
        return sampled

    def _sample_fifo(self, k: int) -> list[ReplayGroup]:
        keys = list(self._groups.keys())[:k]
        return [self._groups[key] for key in keys]

    def _sample_uniform(self, k: int) -> list[ReplayGroup]:
        keys = self.rng.sample(list(self._groups.keys()), k)
        return [self._groups[key] for key in keys]

    def _sample_prioritized(self, k: int) -> list[ReplayGroup]:
        assert self._sum_tree is not None
        sampled_keys: set[str] = set()
        sampled: list[ReplayGroup] = []
        attempts = 0
        max_attempts = k * 20
        while len(sampled) < k and attempts < max_attempts:
            key = self._sum_tree.sample_one(self.rng)
            if key not in sampled_keys:
                sampled_keys.add(key)
                sampled.append(self._groups[key])
            attempts += 1
        if len(sampled) < k:
            remaining_keys = [key for key in self._groups if key not in sampled_keys]
            self.rng.shuffle(remaining_keys)
            for key in remaining_keys[: k - len(sampled)]:
                sampled.append(self._groups[key])
        return sampled

    def update_priorities(self, updates: dict[str, float]) -> None:
        for group_id, priority in updates.items():
            if group_id in self._groups:
                self._groups[group_id].priority = priority
                if self._sum_tree is not None:
                    self._sum_tree.update(group_id, priority)

    def _remove(self, group_id: str):
        if group_id in self._groups:
            del self._groups[group_id]
            if self._sum_tree is not None:
                self._sum_tree.remove(group_id)

    def _evict_overflow(self):
        while len(self._groups) > self.capacity:
            oldest_key = next(iter(self._groups))
            self._remove(oldest_key)

    def _evict_sampled(self, sampled: list[ReplayGroup]):
        if self.eviction_type == EvictionType.AFTER_N_SAMPLES:
            for g in sampled:
                if g.sample_count >= self.max_sample_count:
                    self._remove(g.group_id)

    def mean_sample_count(self) -> float:
        if not self._groups:
            return 0.0
        return sum(g.sample_count for g in self._groups.values()) / len(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def state_dict(self) -> dict:
        groups_data = []
        for group in self._groups.values():
            groups_data.append(
                {
                    "group_id": group.group_id,
                    "queries": group.queries,
                    "responses": group.responses,
                    "masks": group.masks,
                    "logprobs": group.logprobs,
                    "scores": group.scores,
                    "finish_reasons": group.finish_reasons,
                    "insertion_step": group.insertion_step,
                    "sample_count": group.sample_count,
                    "priority": group.priority,
                    "dataset_name": group.dataset_name,
                    "raw_query": group.raw_query,
                    "active_tools": group.active_tools,
                }
            )
        return {
            "capacity": self.capacity,
            "sampler_type": self.sampler_type.value,
            "eviction_type": self.eviction_type.value,
            "max_sample_count": self.max_sample_count,
            "seed": self._seed,
            "rng_state": self.rng.getstate(),
            "groups": groups_data,
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = state["capacity"]
        self.sampler_type = SamplerType(state["sampler_type"])
        self.eviction_type = EvictionType(state["eviction_type"])
        self.max_sample_count = state["max_sample_count"]
        self._seed = state["seed"]
        self.rng.setstate(state["rng_state"])

        self._groups = collections.OrderedDict()
        self._sum_tree = None
        if self.sampler_type == SamplerType.PRIORITIZED:
            self._sum_tree = SumTree(self.capacity)

        for gd in state["groups"]:
            group = ReplayGroup(
                group_id=gd["group_id"],
                queries=gd["queries"],
                responses=gd["responses"],
                masks=gd["masks"],
                logprobs=gd["logprobs"],
                scores=gd["scores"],
                finish_reasons=gd["finish_reasons"],
                insertion_step=gd["insertion_step"],
                sample_count=gd["sample_count"],
                priority=gd["priority"],
                dataset_name=gd["dataset_name"],
                raw_query=gd["raw_query"],
                active_tools=gd["active_tools"],
            )
            self._groups[group.group_id] = group
            if self._sum_tree is not None:
                self._sum_tree.add(group.group_id, group.priority)
