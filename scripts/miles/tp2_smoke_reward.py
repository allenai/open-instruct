"""Synthetic sibling rewards for refresh mechanics; not a learning objective."""


async def score(args, sample, **kwargs):
    # Consecutive siblings get different rewards even when every response hits
    # the short smoke-test length limit, ensuring optimizer updates change weights.
    return float(sample.index % 2)
