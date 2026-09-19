async def score(args, sample, **kwargs):
    return float(sample.index % 2)
