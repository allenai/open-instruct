"""Synthetic sibling rewards for bounded machinery checks, not learning runs."""


async def score(args, sample, **kwargs):
    return float(sample.index % 2)
