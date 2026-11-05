"""Use the shared verifier adapter without selecting the Core training backend."""

import copy
from types import SimpleNamespace

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards


async def score(sample, registry, args=None):
    scoring = copy.copy(args) if args is not None else SimpleNamespace()
    # Shared verifier scoring follows the demo baseline, including zero reward
    # for length-truncated responses; OPD training uses its separate teacher loss.
    scoring.olmo_core = CoreConfig(reward_config=registry, reward_final_answer_only=False)
    return await rewards.registered_reward(scoring, sample)
