"""Reorder complete accepted groups without changing the provider's draw policy.

The native buffer owns filtering, capacity, waiting, retries and accounting. This
list selects a complete cohort before returning its first member. Incomplete
pools stay queued; no prompt is dropped to make a homogeneous batch.
"""

import random
from collections import defaultdict, deque


class PoolBatchQueue(list):
    def __init__(self, *, size, fraction, field, values, seed, version, stale):
        super().__init__()
        self.size, self.fraction, self.field = size, fraction, field
        self.values, self.seed = frozenset(values), seed
        self.version, self.stale = version, stale
        self.cohort = deque()
        self.batch_version = None
        self.batch_count = 0
        self.pool_batches = 0

    def pool(self, entry):
        labels = set()
        for sample in entry.group:
            label = sample.metadata or {}
            for part in self.field.split("."):
                label = label.get(part) if isinstance(label, dict) else None
            if not isinstance(label, str):
                raise ValueError(f"Pool metadata.{self.field} must be a string")
            labels.add(label)
        if len(labels) != 1 or not labels <= self.values:
            raise ValueError(f"Every prompt group must have one declared metadata.{self.field} label; got {labels}")
        return next(iter(labels))

    def append(self, entry):
        self.pool(entry)
        super().append(entry)

    def __bool__(self):
        if self.cohort:
            if self.version() != self.batch_version:
                raise RuntimeError("Policy version changed inside a reserved pool batch")
            return True
        # Let the native buffer reject stale groups through its ordinary path.
        if any(self.stale(entry) for entry in self):
            return True
        if len(self) < self.size:
            return False
        version = self.version()
        homogeneous = random.Random(f"{self.seed}:{version}").random() < self.fraction
        if homogeneous:
            pools = defaultdict(list)
            for entry in self:
                pools[self.pool(entry)].append(entry)
            # Oldest ready pool first avoids preferring a uniformly chosen small
            # pool, which would change the requested source distribution.
            ready = next((entries for entries in pools.values() if len(entries) >= self.size), None)
            if ready is None:
                return False
            selected = ready[: self.size]
        else:
            selected = self[: self.size]
        self.cohort.extend(selected)
        self.batch_version = version
        self.batch_count += 1
        self.pool_batches += homogeneous
        return True

    def pop(self, index=0):
        if index != 0:
            raise ValueError("PoolBatchQueue supports native FIFO pop(0) only")
        if not self.cohort:
            # __bool__ exposes stale entries before admitting a new cohort.
            for position, entry in enumerate(self):
                if self.stale(entry):
                    return super().pop(position)
            raise RuntimeError("Native buffer popped without a ready cohort")
        entry = self.cohort.popleft()
        for position, candidate in enumerate(self):
            if candidate is entry:
                return super().pop(position)
        raise RuntimeError("Reserved pool group disappeared from the completed queue")
