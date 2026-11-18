"""Reward bridge for native TITO agents that return an episode verdict."""


async def episode_reward(args, sample, **kwargs):
    """Use the agent's verdict; missing verdicts stay missing for group filtering.

    TITO v2 normally sets Sample.reward during collection. This fallback handles
    episodes that expose the verdict only in their returned metadata, without
    treating a missing sandbox/test result as a policy failure.
    """
    return (sample.metadata or {}).get("reward")
