"""Synthetic rewards for online-filter lifecycle qualification; not task accuracy."""


async def score(args, sample, **kwargs):
    """Cycle all-zero, all-one and mixed groups for a two-response smoke run."""
    # Evaluation uses ungrouped samples; its synthetic score is not task accuracy.
    group = sample.group_index
    kind = group % 3 if group is not None else 2
    sample.metadata = {**(sample.metadata or {}), "synthetic_filter_fixture": kind}
    return float(kind if kind < 2 else sample.index % 2)
