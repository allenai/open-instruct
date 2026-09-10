"""Terminal rollout lifecycle exceptions."""


class GenerationInterrupted(RuntimeError):
    """A producer must quiesce before resuming under a new policy."""
