"""Backend-independent policy versions and durable resume manifests."""

import dataclasses
import json
import os
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class PolicyClock:
    completed_steps: int = 0
    published_step: int = -1
    next_rollout_id: int = 0

    def validate_versions(self, versions: list[int], max_lag: int) -> int:
        if not versions:
            raise ValueError("Rollouts must carry behavior policy versions")
        if any(type(v) is not int or v < 0 or v > self.published_step for v in versions):
            raise ValueError("Missing, invalid, or future behavior policy version")
        lag = self.completed_steps - min(versions)
        if lag > max_lag:
            raise ValueError(f"Learner policy lag {lag} exceeds {max_lag}")
        return lag

    def optimizer_step(self, successful: bool) -> None:
        if not successful:
            raise RuntimeError("Optimizer step skipped; no new policy can be published")
        self.completed_steps += 1

    def published(self) -> None:
        self.published_step = self.completed_steps

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "PolicyClock":
        clock = cls(**state)
        if any(type(v) is not int for v in state.values()):
            raise ValueError("Policy clock fields must be integers")
        if (
            clock.completed_steps < 0
            or clock.next_rollout_id < 0
            or not -1 <= clock.published_step <= clock.completed_steps
        ):
            raise ValueError("Invalid policy clock checkpoint")
        return clock


def atomic_json(path: Path, value: Any) -> None:
    """Publish only after the complete JSON has reached the filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
