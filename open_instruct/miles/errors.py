"""Exceptions distinguishing actionable input errors from runtime failures."""


class InputError(ValueError):
    """Invalid user input that the CLI can explain without a Python traceback."""
