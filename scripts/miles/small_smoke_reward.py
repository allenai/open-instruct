"""Synthetic sibling rewards for a Small optimizer/publication mechanics check.

This deliberately bypasses dataset grading; managed judge canaries run separately.
It is not a learning objective or a quality evaluation.
"""


async def score(args, sample, **kwargs):
    return float(sample.index % 2)
