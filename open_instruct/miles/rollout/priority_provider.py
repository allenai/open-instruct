"""Exhaust historical priority groups in order, shuffling only within a group."""

import bisect
import hashlib
import itertools
import math
import random
from array import array

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def _metadata_value(metadata, path):
    value = metadata
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


class PriorityPromptProvider:
    """Visit each row once: all of group 1, then 2, then 3, then 4.

    A seeded permutation within each group fixes the complete traversal before
    training begins. No reward feedback or group probabilities affect selection.
    """

    requires_feedback = False

    def __init__(self, dataset, *, field, seed, shuffle, source_field=None, source_weights=None):
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Priority sampling requires a metadata field")
        if dataset is None or not len(dataset):
            raise ValueError("Priority sampling requires a nonempty dataset")
        if (source_field is None) != (source_weights is None):
            raise ValueError("Priority source weighting requires both source_field and source_weights")
        if source_weights is not None:
            if not isinstance(source_field, str) or not source_field.strip():
                raise ValueError("Priority source field must be a nonempty metadata key")
            if not isinstance(source_weights, dict) or not source_weights:
                raise ValueError("Priority source weights must be a nonempty mapping")
            for name, weight in source_weights.items():
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or type(weight) not in (int, float)
                    or not math.isfinite(weight)
                    or weight <= 0
                ):
                    raise ValueError("Priority source names must be nonempty and weights finite and positive")
            if not math.isfinite(sum(source_weights.values())):
                raise ValueError("Priority source weight total must be finite")
        self.samples = dataset.origin_samples
        rows = [{} for _ in range(4)]
        digest = hashlib.sha256()
        for index, sample in enumerate(self.samples):
            group = (sample.metadata or {}).get(field)
            if type(group) is not int or not 1 <= group <= 4:
                raise ValueError(f"Priority row {index}: metadata.{field} must be an integer in 1..4; got {group!r}")
            source = _metadata_value(sample.metadata, source_field) if source_field else ""
            if source_weights is not None and (not isinstance(source, str) or source not in source_weights):
                raise ValueError(
                    f"Priority row {index}: metadata.{source_field}={source!r} has no configured source weight"
                )
            rows[group - 1].setdefault(source, array("Q")).append(index)
            digest.update(bytes([group]))
            if source_field:
                encoded = source.encode()
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        self.counts = [sum(map(len, members.values())) for members in rows]
        self.order = array("Q")
        self.source_counts = [{name: len(indices) for name, indices in members.items()} for members in rows]
        for group, members in enumerate(rows, 1):
            for source, indices in members.items():
                if shuffle:
                    random.Random(
                        f"{seed}:priority:{group}:{source}" if source_field else f"{seed}:priority:{group}"
                    ).shuffle(indices)
            if source_weights is None:
                self.order.extend(members.get("", ()))
            else:
                self._merge_sources(members, source_weights, seed=f"{seed}:priority-sources:{group}")
        self.identity = {
            "kind": "priority/v2",
            "field": field,
            "source_field": source_field,
            "source_weights": dict(source_weights) if source_weights is not None else None,
            "seed": seed,
            "shuffle": shuffle,
            "counts": self.counts,
            "assignments_sha256": digest.hexdigest(),
        }
        self.offset = 0
        if source_weights is not None:
            logger.info(
                "Priority source field=%s weights=%s group_source_counts=%s",
                source_field,
                source_weights,
                self.source_counts,
            )
        logger.info(
            "Priority sampling field=%s group_counts=%s order=1,2,3,4 seed=%s shuffle=%s",
            field,
            self.counts,
            seed,
            shuffle,
        )

    def _merge_sources(self, members, weights, *, seed):
        # Build the weighted merge once, so training-time selection remains a
        # slice and checkpoints need only one offset. Each source keeps its own
        # permutation; only exhausted sources change the active probabilities.
        names = sorted(members)
        offsets = dict.fromkeys(names, 0)
        rng = random.Random(seed)
        cumulative = list(itertools.accumulate(weights[name] for name in names))
        while names:
            index = bisect.bisect_right(cumulative, rng.random() * cumulative[-1]) if len(names) > 1 else 0
            name = names[index]
            offset = offsets[name]
            self.order.append(members[name][offset])
            offsets[name] = offset + 1
            if offsets[name] == len(members[name]):
                names.pop(index)
                cumulative = list(itertools.accumulate(weights[name] for name in names))

    def available(self):
        return self.offset < len(self.order)

    def next_samples(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if n > len(self.order) - self.offset:
            raise StopIteration("Historical priority inventory exhausted")
        result = [self.samples[index] for index in self.order[self.offset : self.offset + n]]
        self.offset += n
        return result

    def feedback(self, records):
        """Historical priority remains fixed throughout the run."""

    def state_dict(self):
        return {**self.identity, "offset": self.offset}

    def load_state_dict(self, state):
        for key, expected in self.identity.items():
            if state.get(key) != expected:
                raise ValueError(f"Priority checkpoint configuration mismatch for {key}")
        offset = state.get("offset")
        if type(offset) is not int or not 0 <= offset <= len(self.order):
            raise ValueError("Invalid priority checkpoint cursor")
        self.offset = offset
