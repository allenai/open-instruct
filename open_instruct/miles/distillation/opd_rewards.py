"""Use the shared verifier adapter without selecting the Core training backend."""

import copy
from types import SimpleNamespace

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards


async def score(sample, registry, args=None):
    scoring = copy.copy(args) if args is not None else SimpleNamespace()
    # The historical OPD evaluation grades the whole answer, including truncated
    # responses. Keep that protocol independent of future GRPO reward defaults.
    scoring.olmo_core = CoreConfig(reward_config=registry, reward_zero_truncated=False, reward_final_answer_only=False)
    return await rewards.registered_reward(scoring, sample)
