"""Prompt selection before MILES expands each prompt into a response group."""

from typing import Any, Protocol

from miles.utils.types import Sample


class PromptProvider(Protocol):
    """Select N fresh prompts; buffering and response-group IDs belong to the caller.

    Callers serialize selection, feedback, and checkpoints. Providers must return
    exactly N prompts (zero is a no-op), and include all selection state in their
    checkpoint. Returned prompts are templates: the adapter deep-copies them.
    Feedback uses inference records; delivery from the recorder is future work.
    """

    def next_samples(self, n: int) -> list[Sample]: ...
    def feedback(self, records: list[dict[str, Any]]) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class ShuffledPromptProvider:
    """The existing MILES in-memory dataset, seeded shuffle, and epoch cursor."""

    def __init__(self, dataset, *, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_id = 0
        self.sample_offset = 0
        # MILES' data-source constructor has already shuffled epoch zero.

    def next_samples(self, n: int) -> list[Sample]:
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if self.dataset is None:
            return [Sample() for _ in range(n)]
        if n and not len(self.dataset):
            raise ValueError("Cannot select prompts from an empty dataset")
        prompts = []
        while n:
            if self.sample_offset == len(self.dataset):
                self.epoch_id += 1
                if self.shuffle:
                    self.dataset.shuffle(self.epoch_id)
                self.sample_offset = 0
            take = min(n, len(self.dataset) - self.sample_offset)
            prompts.extend(self.dataset.samples[self.sample_offset : self.sample_offset + take])
            self.sample_offset += take
            n -= take
        return prompts

    def feedback(self, records: list[dict[str, Any]]) -> None:
        """The shuffle baseline ignores outcomes."""

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "shuffle",
            "dataset_size": len(self.dataset) if self.dataset is not None else None,
            "seed": getattr(self.dataset, "seed", None),
            "shuffle": self.shuffle,
            "epoch_id": self.epoch_id,
            "sample_offset": self.sample_offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        if any(state.get(k) != expected[k] for k in ("kind", "dataset_size", "seed", "shuffle")):
            raise ValueError("Prompt provider checkpoint does not match this shuffle configuration")
        epoch, offset = state.get("epoch_id"), state.get("sample_offset")
        size = len(self.dataset) if self.dataset is not None else 0
        if type(epoch) is not int or epoch < 0 or type(offset) is not int or not 0 <= offset <= size:
            raise ValueError("Invalid prompt provider cursor")
        if self.dataset is not None and self.shuffle:
            self.dataset.shuffle(epoch)
        self.epoch_id, self.sample_offset = epoch, offset
