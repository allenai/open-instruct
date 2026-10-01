"""Identify configuration and input problems that a user can fix before training.
The shared InputError type lets planning, preparation and launch report these
failures consistently while preserving ordinary exceptions for runtime faults.
"""


class InputError(ValueError):
    """Invalid user input that the CLI can explain without a Python traceback."""
