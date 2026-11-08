"""Compatibility paths for overlong filtering; the implementation lives in ``exclusion``.

Existing configurations select
``open_instruct.miles.rewards.truncation.exclude_truncated`` and
``open_instruct.miles.rewards.truncation.filter_finished_reward_variance``. Both
now also leave ungraded responses out of the group baseline and the variance
check; see :mod:`open_instruct.miles.rewards.exclusion`.
"""

from open_instruct.miles.rewards.exclusion import exclude_truncated, filter_finished_reward_variance

__all__ = ["exclude_truncated", "filter_finished_reward_variance"]
