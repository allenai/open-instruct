"""DPPO loss settings, importable without torch (the run-file contract validates them on CPU)."""

import dataclasses
import math
import os

ENV_PREFIX = "OI_DPPO_"
# ``ppo``: MILES' clipped policy loss. ``dppo``: Open Instruct's DPPO trust-region mask
# (megatron_grpo.dppo_loss), which needs the rollout engine's behavior log-probs.
POLICY_LOSSES = ("ppo", "dppo")
LOSS_FUNCTION = "open_instruct.miles.megatron_grpo.dppo_loss.policy_loss"
DIVERGENCE_TYPES = ("tv", "kl")


@dataclasses.dataclass(frozen=True)
class Settings:
    divergence_type: str = "tv"
    threshold: float = 0.1

    def __post_init__(self):
        if self.divergence_type not in DIVERGENCE_TYPES:
            raise ValueError(f"DPPO divergence type must be one of {DIVERGENCE_TYPES}, got {self.divergence_type!r}")
        if not (math.isfinite(self.threshold) and self.threshold > 0):
            raise ValueError(f"DPPO threshold must be a finite positive number, got {self.threshold!r}")

    def environment(self):
        return {ENV_PREFIX + "DIVERGENCE_TYPE": self.divergence_type, ENV_PREFIX + "THRESHOLD": repr(self.threshold)}

    @classmethod
    def from_environment(cls, environment=None):
        environment = os.environ if environment is None else environment
        missing = [ENV_PREFIX + key for key in ("DIVERGENCE_TYPE", "THRESHOLD") if ENV_PREFIX + key not in environment]
        if missing:
            raise ValueError(f"DPPO loss selected without {', '.join(missing)}")
        return cls(
            divergence_type=environment[ENV_PREFIX + "DIVERGENCE_TYPE"],
            threshold=float(environment[ENV_PREFIX + "THRESHOLD"]),
        )
