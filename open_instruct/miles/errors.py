"""Exceptions distinguishing actionable input errors from runtime failures."""


class GenerationInterrupted(RuntimeError):
    """A producer must quiesce before resuming under a new policy."""


class InputError(ValueError):
    """Invalid user input that the CLI can explain without a Python traceback."""
