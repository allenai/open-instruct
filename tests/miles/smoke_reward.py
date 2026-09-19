async def score(args, sample, **kwargs):
    return float(sample.index % 2)


async def filtering_score(args, sample, **kwargs):
    """Synthetic filter qualification only: cycle all-zero, all-one and mixed groups."""
    # Evaluation uses ungrouped samples; its synthetic score is not task accuracy.
    group = sample.group_index
    kind = group % 3 if group is not None else 2
    sample.metadata = {**(sample.metadata or {}), "synthetic_filter_fixture": kind}
    return float(kind if kind < 2 else sample.index % 2)
