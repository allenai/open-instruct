"""Qualification checks must remain active under python -O."""


def require(condition, message):
    if not condition:
        raise ValueError(message)
