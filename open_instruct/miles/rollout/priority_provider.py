"""Static reward-history priorities with independent, cyclic group traversal."""

import hashlib
import math
import random

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


class PriorityPromptProvider:
    """Draw a group by probability, then traverse its rows without replacement.

    Small preferred groups cycle independently of large groups. This intentionally
    changes exposure; historical priority is a fixed hint, never live feedback.
    """

    requires_feedback = False

    def __init__(self, dataset, *, field, weights, seed, shuffle):
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Priority sampling requires a metadata field")
        if (
            not isinstance(weights, list)
            or len(weights) != 4
            or any(type(w) not in (int, float) or not math.isfinite(w) or not 0 <= w <= 1 for w in weights)
            or not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-9)
        ):
            raise ValueError("Priority sampling requires four finite probabilities summing to 1")
        if dataset is None or not len(dataset):
            raise ValueError("Priority sampling requires a nonempty dataset")
        self.samples = dataset.origin_samples
        self.field, self.weights, self.seed, self.shuffle = field, list(weights), seed, shuffle
        self.rows = [[] for _ in range(4)]
        digest = hashlib.sha256()
        for index, sample in enumerate(self.samples):
            group = (sample.metadata or {}).get(field)
            if type(group) is not int or not 1 <= group <= 4:
                raise ValueError(f"Priority row {index}: metadata.{field} must be an integer in 1..4; got {group!r}")
            self.rows[group - 1].append(index)
            digest.update(bytes([group]))
        self.counts = list(map(len, self.rows))
        for group, (count, weight) in enumerate(zip(self.counts, weights, strict=True), 1):
            if weight and not count:
                raise ValueError(
                    f"Priority group {group} has positive probability but no rows; set its weight to zero"
                )
        self.identity = {
            "kind": "priority/v1",
            "field": field,
            "weights": self.weights,
            "seed": seed,
            "shuffle": shuffle,
            "counts": self.counts,
            "assignments_sha256": digest.hexdigest(),
        }
        self.rng = random.Random(seed)
        self.epochs, self.offsets = [0] * 4, [0] * 4
        self.orders = [self._order(i, 0) for i in range(4)]
        logger.info("Priority sampling field=%s counts=%s probabilities=%s", field, self.counts, weights)

    def _order(self, group, epoch):
        rows = self.rows[group].copy()
        if self.shuffle:
            random.Random(f"{self.seed}:priority:{group + 1}:{epoch}").shuffle(rows)
        return rows

    def next_samples(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        result = []
        for _ in range(n):
            group = self.rng.choices(range(4), weights=self.weights, k=1)[0]
            if self.offsets[group] == self.counts[group]:
                self.epochs[group] += 1
                self.offsets[group] = 0
                self.orders[group] = self._order(group, self.epochs[group])
            result.append(self.samples[self.orders[group][self.offsets[group]]])
            self.offsets[group] += 1
        return result

    def feedback(self, records):
        """Historical priority remains fixed throughout the run."""

    def state_dict(self):
        return {
            **self.identity,
            "rng": self.rng.getstate(),
            "epochs": self.epochs.copy(),
            "offsets": self.offsets.copy(),
        }

    def load_state_dict(self, state):
        for key, expected in self.identity.items():
            if state.get(key) != expected:
                raise ValueError(f"Priority checkpoint configuration mismatch for {key}")
        epochs, offsets = state.get("epochs"), state.get("offsets")
        if not isinstance(epochs, list) or not isinstance(offsets, list) or len(epochs) != 4 or len(offsets) != 4:
            raise ValueError("Invalid priority checkpoint cursors")
        for epoch, offset, size in zip(epochs, offsets, self.counts, strict=True):
            if type(epoch) is not int or epoch < 0 or type(offset) is not int or not 0 <= offset <= size:
                raise ValueError("Invalid priority checkpoint cursor")
        rng = random.Random()
        rng.setstate(state["rng"])
        # Validate before mutating; rebuilding only changed epochs keeps ordinary
        # delivery rollback cheap even for million-row datasets.
        orders = [
            self._order(i, epoch) if epoch != self.epochs[i] else self.orders[i] for i, epoch in enumerate(epochs)
        ]
        self.orders, self.epochs, self.offsets, self.rng = orders, epochs.copy(), offsets.copy(), rng
