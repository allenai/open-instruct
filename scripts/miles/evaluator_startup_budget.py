"""Launch the pinned evaluator with an explicit provider initialization budget.

This standalone application shim changes only the coordinator's readiness wait.
The evaluator's worker crash checks, providers, sampling and task execution stay
in the installed runtime. Set OLMO_EVAL_PROVIDER_INIT_TIMEOUT_SECONDS explicitly;
the surrounding job timeout remains the final wall-clock bound.
"""

import functools
import importlib
import inspect
import os
import sys


def configure(runner, timeout):
    if not 900 < timeout <= 3600:
        raise ValueError("Provider initialization budget must be in (900, 3600] seconds")
    original = runner.wait_for_init_times
    signature = inspect.signature(original)
    if signature.parameters["timeout"].default != 900:
        raise RuntimeError("Unexpected evaluator readiness contract; revalidate the pinned runtime")
    # Bind the installed implementation rather than replacing its liveness checks.
    runner.wait_for_init_times = functools.partial(original, timeout=timeout)


def main():
    timeout = int(os.environ["OLMO_EVAL_PROVIDER_INIT_TIMEOUT_SECONDS"])
    runner = importlib.import_module("olmo_eval.runners.asynq.runner")
    configure(runner, timeout)
    print(f"[provider-startup] coordinator initialization budget={timeout}s; worker crash checks retained", flush=True)
    return importlib.import_module("olmo_eval.cli").main()


if __name__ == "__main__":
    sys.exit(main())
